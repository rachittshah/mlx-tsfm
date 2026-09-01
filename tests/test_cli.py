"""Tests for the forecast CLI helpers and the convert scaffold."""
import numpy as np
import pytest

from mlx_tsfm.convert import convert
from mlx_tsfm.forecast import load_series, run_forecast


class TestForecastCLI:
    def test_run_forecast_returns_a_forecast(self):
        series = np.sin(np.linspace(0, 10, 128)).astype(np.float32)
        fc = run_forecast("_stub", series, horizon=8, quantiles=[0.5],
                          d_model=32, n_heads=4, patch_len=16)
        assert fc.point.shape == (1, 8)
        assert fc.quantiles.shape == (1, 8, 1)

    def test_load_series_roundtrips_npy(self, tmp_path):
        arr = np.arange(50.0, dtype=np.float32)
        p = tmp_path / "series.npy"
        np.save(p, arr)
        loaded = load_series(str(p))
        assert np.allclose(loaded, arr)


class TestConvert:
    def test_unsupported_model_raises_explicitly(self):
        # timesfm-3.0 conversion is implemented; other backends are not yet
        with pytest.raises(NotImplementedError) as e:
            convert("amazon/chronos-2", "out/chronos-2-mlx")
        assert "chronos-2" in str(e.value)

    def test_timesfm3_is_a_supported_target(self):
        from mlx_tsfm.convert import _SUPPORTED

        assert _SUPPORTED.get("google/timesfm-3.0-pytorch") == "timesfm3"
