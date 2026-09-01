"""Forecasting output heads.

A time-series foundation model does not emit a distribution over a vocabulary; it emits numeric
forecasts. These heads project the final hidden state to either a point forecast or a set of
quantiles, replacing the ``nn.Linear -> softmax`` vocab head of an LLM (and the cross-entropy loss
is replaced downstream by a pinball/quantile loss).
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

# TimesFM 3 emits the 10th..90th deciles.
DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class PointHead(nn.Module):
    """Project a ``(B, d_model)`` hidden state to a ``(B, horizon)`` point forecast."""

    def __init__(self, d_model: int, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.proj = nn.Linear(d_model, horizon)

    def __call__(self, h: mx.array) -> mx.array:
        return self.proj(h)


class QuantileHead(nn.Module):
    """Project a hidden state to ``(B, horizon, n_quantiles)`` quantile forecasts.

    The raw projection is sorted ascending along the quantile axis so the output never exhibits
    quantile crossing (q10 <= q20 <= ... <= q90), which a naive linear head does not guarantee.
    """

    def __init__(self, d_model: int, horizon: int, n_quantiles: int = 9):
        super().__init__()
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.proj = nn.Linear(d_model, horizon * n_quantiles)

    def __call__(self, h: mx.array) -> mx.array:
        b = h.shape[0]
        raw = self.proj(h).reshape(b, self.horizon, self.n_quantiles)
        return mx.sort(raw, axis=-1)
