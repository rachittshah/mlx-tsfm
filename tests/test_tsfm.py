"""Tests for the model-agnostic core: Forecast, TSFMModel base pipeline, and the registry."""
import mlx.core as mx
import numpy as np
import pytest

from mlx_tsfm.tsfm import (
    Forecast,
    TSFMConfig,
    TSFMModel,
    load,
    register_model,
)


class _ConstEncode(TSFMModel):
    """Minimal concrete model: encode returns zeros, exercising the base pipeline only."""

    def encode(self, patches, mask):
        b = patches.shape[0]
        return mx.zeros((b, self.config.d_model))


def _model(**kw):
    cfg = {"d_model": 16, "patch_len": 32, "horizon_max": 64}
    cfg.update(kw)
    return _ConstEncode(TSFMConfig(**cfg))


class TestForecast:
    def test_holds_point_quantiles_and_levels(self):
        fc = Forecast(point=mx.zeros((1, 3)), quantiles=mx.zeros((1, 3, 9)), q_levels=[0.5])
        assert fc.point.shape == (1, 3)
        assert fc.quantiles.shape == (1, 3, 9)
        assert fc.q_levels == [0.5]


class TestBasePipeline:
    def test_forecast_single_series_shapes(self):
        m = _model()
        ctx = np.sin(np.linspace(0, 10, 200))  # 1-D
        fc = m.forecast(ctx, horizon=24)
        assert fc.point.shape == (1, 24)
        assert fc.quantiles.shape == (1, 24, 9)

    def test_forecast_accepts_python_list_and_mx_array(self):
        m = _model()
        assert m.forecast([1.0, 2.0, 3.0, 4.0], horizon=2).point.shape == (1, 2)
        assert m.forecast(mx.arange(50.0), horizon=5).point.shape == (1, 5)

    def test_forecast_batch_shapes(self):
        m = _model()
        contexts = np.stack([np.arange(64.0), np.arange(64.0) * 2])  # (2, 64)
        fc = m.forecast_batch(contexts, horizon=16)
        assert fc.point.shape == (2, 16)
        assert fc.quantiles.shape == (2, 16, 9)

    def test_horizon_beyond_max_raises(self):
        m = _model(horizon_max=32)
        with pytest.raises(ValueError):
            m.forecast(np.arange(64.0), horizon=33)

    def test_quantile_subset_selection(self):
        m = _model()
        fc = m.forecast(np.arange(100.0), horizon=8, quantiles=[0.1, 0.5, 0.9])
        assert fc.q_levels == [0.1, 0.5, 0.9]
        assert fc.quantiles.shape == (1, 8, 3)

    def test_outputs_are_finite(self):
        m = _model()
        fc = m.forecast(np.arange(128.0), horizon=32)
        mx.eval(fc.point, fc.quantiles)
        assert not bool(mx.isnan(fc.point).any().item())
        assert not bool(mx.isnan(fc.quantiles).any().item())


class TestRegistry:
    def test_register_and_load(self):
        register_model("dummy_test", lambda **kw: _model(**kw))
        m = load("dummy_test")
        assert isinstance(m, TSFMModel)

    def test_load_unknown_raises(self):
        with pytest.raises(KeyError):
            load("no_such_model_xyz")

    def test_load_passes_kwargs_to_builder(self):
        register_model("dummy_kw", lambda **kw: _model(**kw))
        m = load("dummy_kw", horizon_max=48)
        assert m.config.horizon_max == 48
