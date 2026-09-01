"""End-to-end tests for the `toy` reference backend running on MLX."""
import mlx.core as mx
import numpy as np

from mlx_tsfm import load


def _toy(**kw):
    kw.setdefault("d_model", 32)
    kw.setdefault("n_heads", 4)
    kw.setdefault("patch_len", 16)
    kw.setdefault("horizon_max", 64)
    return load("toy", **kw)


def test_load_toy_runs_end_to_end():
    m = _toy()
    ctx = np.sin(np.linspace(0, 20, 200)).astype(np.float32)
    fc = m.forecast(ctx, horizon=24, quantiles=[0.1, 0.5, 0.9])
    mx.eval(fc.point, fc.quantiles)
    assert fc.point.shape == (1, 24)
    assert fc.quantiles.shape == (1, 24, 3)
    assert not bool(mx.isnan(fc.point).any().item())
    assert not bool(mx.isnan(fc.quantiles).any().item())


def test_batch_forecast_shape():
    m = _toy()
    contexts = np.stack([np.sin(np.linspace(0, 10, 128)),
                         np.cos(np.linspace(0, 10, 128)),
                         np.linspace(-1, 1, 128),
                         np.zeros(128)]).astype(np.float32)
    fc = m.forecast_batch(contexts, horizon=16)
    assert fc.point.shape == (4, 16)


def test_forecast_is_deterministic():
    m = _toy()
    ctx = np.arange(128.0, dtype=np.float32)
    a = m.forecast(ctx, horizon=8).point
    b = m.forecast(ctx, horizon=8).point
    assert mx.allclose(a, b).item()


def test_batch_equals_individual_forecasts():
    """Per-series normalization + per-batch attention must not leak across the batch."""
    m = _toy()
    contexts = np.stack([np.sin(np.linspace(0, 12, 96)),
                         np.linspace(0, 5, 96),
                         np.cos(np.linspace(3, 9, 96))]).astype(np.float32)
    batch = m.forecast_batch(contexts, horizon=12).point
    for i in range(contexts.shape[0]):
        single = m.forecast(contexts[i], horizon=12).point
        assert mx.allclose(batch[i], single[0], atol=1e-4).item()


def test_quantiles_are_ordered_after_denormalization():
    m = _toy()
    ctx = (np.random.RandomState(0).randn(120) * 10 + 50).astype(np.float32)
    fc = m.forecast(ctx, horizon=10)  # all 9 deciles
    diffs = fc.quantiles[:, :, 1:] - fc.quantiles[:, :, :-1]
    assert bool((diffs >= -1e-4).all().item())
