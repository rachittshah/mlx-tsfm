"""Patch front-end for time-series foundation models.

Two pieces an LLM stack does not provide:

- ``InstanceNorm`` — per-series (per-window) normalization. TSFMs are trained on normalized
  context and forecast in normalized space; the caller denormalizes the horizon with the *same*
  statistics. This replaces an LLM tokenizer's role of mapping raw input to model space.
- ``Patchifier`` — splits a 1-D series into fixed-length patches (TimesFM/PatchTST style), the
  "tokens" of a time-series transformer. Padding is added on the LEFT so the most recent step
  always lands at the end of the last patch.
"""
from __future__ import annotations

import mlx.core as mx

Stats = tuple[mx.array, mx.array]  # (mean, std), each shape (B, 1)


class InstanceNorm:
    """Per-series mean/variance normalization over the time axis of a ``(B, L)`` context."""

    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def normalize(self, x: mx.array) -> tuple[mx.array, Stats]:
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + self.eps
        return (x - mean) / std, (mean, std)

    def denormalize(self, y: mx.array, stats: Stats) -> mx.array:
        mean, std = stats
        return y * std + mean


class Patchifier:
    """Split ``(B, L)`` into ``(B, num_patches, patch_len)`` with a validity mask.

    ``num_patches = ceil(L / patch_len)``; ``(num_patches * patch_len - L)`` padding values are
    prepended (left) so the newest observation is the last real element. ``mask`` is 1.0 on real
    positions and 0.0 on padding.
    """

    def __init__(self, patch_len: int = 32):
        if patch_len <= 0:
            raise ValueError("patch_len must be positive")
        self.patch_len = patch_len

    def patchify(self, x: mx.array) -> tuple[mx.array, mx.array]:
        if x.ndim != 2:
            raise ValueError(f"expected (B, L), got shape {x.shape}")
        b, length = x.shape
        p = self.patch_len
        num_patches = (length + p - 1) // p
        pad = num_patches * p - length

        if pad:
            x_padded = mx.concatenate([mx.zeros((b, pad), dtype=x.dtype), x], axis=1)
            valid = mx.concatenate([mx.zeros((b, pad)), mx.ones((b, length))], axis=1)
        else:
            x_padded = x
            valid = mx.ones((b, length))

        patches = x_padded.reshape(b, num_patches, p)
        mask = valid.reshape(b, num_patches, p)
        return patches, mask
