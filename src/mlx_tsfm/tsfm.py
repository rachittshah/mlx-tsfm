"""Model-agnostic core for time-series foundation model inference.

``TSFMModel`` owns the pipeline shared by every TSFM backend — instance-normalize the context,
patchify it, run the (backend-specific) transformer body via ``encode``, project with the point +
quantile heads, denormalize, and slice to the requested horizon. A concrete backend only implements
``encode(patches, mask) -> (B, d_model)``.

The registry (``register_model`` / ``load``) mirrors ``mlx_lm`` ergonomics: ``load("chronos-2")``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .heads import DEFAULT_QUANTILES, PointHead, QuantileHead
from .patching import InstanceNorm, Patchifier


@dataclass
class Forecast:
    """A forecast: a point path plus an optional quantile band, in the original data scale."""

    point: mx.array               # (B, H)
    quantiles: mx.array | None    # (B, H, Q) or None
    q_levels: list[float]         # the quantile levels of the last axis


@dataclass
class TSFMConfig:
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    patch_len: int = 32
    horizon_max: int = 64
    context_len: int = 512
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES


def _as_2d(x) -> mx.array:
    """Coerce a list / numpy / mlx series of shape (L,) or (B, L) into a float32 (B, L) array."""
    if isinstance(x, mx.array):
        arr = x.astype(mx.float32)
    else:
        arr = mx.array(np.asarray(x, dtype=np.float32))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"expected a (L,) or (B, L) series, got shape {tuple(arr.shape)}")
    return arr


class TSFMModel(nn.Module):
    """Base class for TSFM backends. Subclasses implement ``encode``."""

    def __init__(self, config: TSFMConfig):
        super().__init__()
        self.config = config
        self.norm = InstanceNorm()
        self.patcher = Patchifier(config.patch_len)
        self.point_head = PointHead(config.d_model, config.horizon_max)
        self.quantile_head = QuantileHead(
            config.d_model, config.horizon_max, len(config.quantile_levels)
        )

    # --- backend hook -----------------------------------------------------------------
    def encode(self, patches: mx.array, mask: mx.array) -> mx.array:
        """Map ``(B, num_patches, patch_len)`` patches to a ``(B, d_model)`` summary hidden state."""
        raise NotImplementedError

    # --- shared inference pipeline ----------------------------------------------------
    def forecast_batch(self, contexts, horizon: int, quantiles=None) -> Forecast:
        if horizon <= 0 or horizon > self.config.horizon_max:
            raise ValueError(
                f"horizon must be in 1..{self.config.horizon_max}, got {horizon}"
            )
        x = _as_2d(contexts)
        x_norm, stats = self.norm.normalize(x)
        patches, mask = self.patcher.patchify(x_norm)
        h = self.encode(patches, mask)                      # (B, d_model)

        point = self.point_head(h)[:, :horizon]             # (B, horizon), normalized
        point = self.norm.denormalize(point, stats)

        q_all = self.quantile_head(h)[:, :horizon, :]       # (B, horizon, Q), normalized
        mean, std = stats
        q_all = q_all * std[..., None] + mean[..., None]    # denormalize over (H, Q)

        levels = list(self.config.quantile_levels)
        if quantiles is not None:
            idx = [min(range(len(levels)), key=lambda i: abs(levels[i] - q)) for q in quantiles]
            q_out = q_all[:, :, idx]
            levels = list(quantiles)
        else:
            q_out = q_all
        return Forecast(point=point, quantiles=q_out, q_levels=levels)

    def forecast(self, context, horizon: int, quantiles=None) -> Forecast:
        """Forecast a single series (shape ``(L,)`` or ``(1, L)``)."""
        return self.forecast_batch(_as_2d(context), horizon, quantiles)


# --- registry -------------------------------------------------------------------------
_REGISTRY: dict[str, Callable[..., TSFMModel]] = {}


def register_model(name: str, builder: Callable[..., TSFMModel]) -> None:
    """Register a builder callable under ``name`` so ``load(name, **kw)`` returns a model."""
    _REGISTRY[name] = builder


def load(model_id: str, **kwargs) -> TSFMModel:
    """Build a registered TSFM backend by id, e.g. ``load("toy", horizon_max=64)``."""
    # Import built-in backends lazily so registration happens without a circular import.
    import mlx_tsfm.models  # noqa: F401

    if model_id not in _REGISTRY:
        raise KeyError(
            f"unknown model '{model_id}'. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[model_id](**kwargs)
