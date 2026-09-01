"""Tests for the forecast CLI helpers and the convert scaffold."""
import numpy as np
import pytest

from mlx_tsfm.convert import convert
from mlx_tsfm.forecast import load_series, run_forecast


class TestForecastCLI:
    def test_run_forecast_returns_a_forecast(self):
        series = np.sin(np.linspace(0, 10, 128)).astype(np.float32)
        fc = run_forecast("toy", series, horizon=8, quantiles=[0.5],
                          d_model=32, n_heads=4, patch_len=16)
        assert fc.point.shape == (1, 8)
        assert fc.quantiles.shape == (1, 8, 1)

    def test_load_series_roundtrips_npy(self, tmp_path):
        arr = np.arange(50.0, dtype=np.float32)
        p = tmp_path / "series.npy"
        np.save(p, arr)
        loaded = load_series(str(p))
        assert np.allclose(loaded, arr)


class TestConvertScaffold:
    def test_convert_not_implemented_is_explicit(self):
        # honest scaffold: real weight conversion is not built yet, but the surface exists
        with pytest.raises(NotImplementedError) as e:
            convert("google/timesfm-3.0-pytorch", "out/timesfm-3.0-mlx")
        assert "timesfm-3.0" in str(e.value)
