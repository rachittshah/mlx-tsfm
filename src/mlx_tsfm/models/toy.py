"""`toy` — a small, randomly-initialized patched transformer.

Not a pretrained model: it exists so the full inference pipeline (patching, normalization,
transformer body, heads, denormalization, batching, quantization, `mx.compile`, benchmarking) can
run and be tested end-to-end on MLX without downloading a multi-hundred-MB checkpoint. Real
backends (TimesFM, Chronos-2) follow the same ``encode`` contract.
"""
from __future__ import annotations

import dataclasses

import mlx.core as mx
import mlx.nn as nn

from ..tsfm import TSFMConfig, TSFMModel, register_model


class _Block(nn.Module):
    """Pre-norm transformer block over the patch sequence."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm1(x)
        x = x + self.attn(h, h, h)
        x = x + self.mlp(self.norm2(x))
        return x


class ToyTSFM(TSFMModel):
    def __init__(self, config: TSFMConfig):
        super().__init__(config)
        d = config.d_model
        self.patch_embed = nn.Linear(config.patch_len, d)
        max_patches = max(1, (config.context_len + config.patch_len - 1) // config.patch_len)
        self.pos = nn.Embedding(max_patches, d)
        self.blocks = [_Block(d, config.n_heads) for _ in range(config.n_layers)]
        self.final_norm = nn.LayerNorm(d)

    def encode(self, patches: mx.array, mask: mx.array) -> mx.array:
        # zero out fully-padded patches so they don't contribute
        keep = (mask.sum(axis=-1, keepdims=True) > 0).astype(patches.dtype)  # (B, N, 1)
        h = self.patch_embed(patches) * keep                                 # (B, N, d_model)
        n = h.shape[1]
        h = h + self.pos(mx.arange(n))[None]                                 # add positions
        for blk in self.blocks:
            h = blk(h)
        h = self.final_norm(h)
        return h[:, -1, :]                                                   # last-patch summary


def _build_toy(**kw) -> ToyTSFM:
    fields = {f.name for f in dataclasses.fields(TSFMConfig)}
    cfg = {k: v for k, v in kw.items() if k in fields}
    return ToyTSFM(TSFMConfig(**cfg))


register_model("toy", _build_toy)
