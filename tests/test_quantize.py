"""Tests for int8 mixed-precision quantization (body int8, forecasting heads kept full precision)."""
import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_tsfm import load
from mlx_tsfm.quantize import quantize_model


def _m():
    return load("_stub", d_model=64, n_heads=4, n_layers=2, patch_len=16,
                horizon_max=32, context_len=128)


def test_forecast_still_works_after_quantize():
    m = _m()
    quantize_model(m, bits=8, group_size=64)
    fc = m.forecast(np.arange(128.0, dtype=np.float32), horizon=16)
    mx.eval(fc.point)
    assert fc.point.shape == (1, 16)
    assert not bool(mx.isnan(fc.point).any().item())


def test_body_is_quantized_but_heads_are_skipped():
    m = _m()
    quantize_model(m, bits=8, group_size=64, skip_heads=True)
    keys = [k for k, _ in tree_flatten(m.parameters())]
    # quantized Linear layers gain a `scales` parameter
    assert any("blocks" in k and "scales" in k for k in keys), "transformer body should be quantized"
    assert not any("head" in k and "scales" in k for k in keys), "heads must stay full precision"
