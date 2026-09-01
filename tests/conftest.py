"""Shared test fixtures.

Registers a tiny, random-weight `_stub` backend so the model-agnostic infrastructure (forecast
pipeline, quantization, benchmarking, CLIs) can be tested fast and offline, without downloading a
real multi-hundred-MB checkpoint. Real backends (timesfm-3.0) are covered by the guarded tests in
test_timesfm.py.
"""
from __future__ import annotations

import dataclasses

import mlx.core as mx
import mlx.nn as nn

from mlx_tsfm.tsfm import TSFMConfig, TSFMModel, register_model


class _StubTSFM(TSFMModel):
    def __init__(self, config: TSFMConfig):
        super().__init__(config)
        self.embed = nn.Linear(config.patch_len, config.d_model, bias=False)
        self.blocks = [nn.Linear(config.d_model, config.d_model) for _ in range(2)]

    def encode(self, patches: mx.array, mask: mx.array) -> mx.array:
        h = self.embed(patches.mean(axis=1))  # (B, d_model)
        for blk in self.blocks:
            h = nn.relu(blk(h))
        return h


def _build_stub(**kw) -> _StubTSFM:
    fields = {f.name for f in dataclasses.fields(TSFMConfig)}
    cfg = {k: v for k, v in kw.items() if k in fields}
    return _StubTSFM(TSFMConfig(**cfg))


register_model("_stub", _build_stub)
