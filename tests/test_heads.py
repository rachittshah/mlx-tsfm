"""Tests for the forecasting heads that replace an LLM's vocab-softmax output."""
import mlx.core as mx

from mlx_tsfm.heads import PointHead, QuantileHead


class TestPointHead:
    def test_maps_hidden_state_to_horizon(self):
        head = PointHead(d_model=16, horizon=24)
        h = mx.zeros((4, 16))  # (B, d_model)
        y = head(h)
        assert y.shape == (4, 24)

    def test_is_deterministic_for_fixed_weights(self):
        head = PointHead(d_model=8, horizon=5)
        h = mx.random.normal((2, 8))
        a, b = head(h), head(h)
        assert mx.allclose(a, b).item()


class TestQuantileHead:
    def test_output_shape_is_batch_horizon_quantiles(self):
        head = QuantileHead(d_model=16, horizon=24, n_quantiles=9)
        h = mx.zeros((4, 16))
        q = head(h)
        assert q.shape == (4, 24, 9)

    def test_quantiles_are_monotonically_non_decreasing(self):
        # a real forecaster must not let q10 exceed q90 ("quantile crossing")
        head = QuantileHead(d_model=32, horizon=10, n_quantiles=9)
        h = mx.random.normal((8, 32))
        q = head(h)
        diffs = q[:, :, 1:] - q[:, :, :-1]
        assert bool((diffs >= -1e-6).all().item())

    def test_default_nine_quantiles(self):
        head = QuantileHead(d_model=8, horizon=4)
        q = head(mx.zeros((1, 8)))
        assert q.shape == (1, 4, 9)
