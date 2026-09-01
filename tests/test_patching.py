"""Tests for the patch front-end: per-series instance normalization + patchifier."""
import mlx.core as mx
import pytest

from mlx_tsfm.patching import InstanceNorm, Patchifier


class TestInstanceNorm:
    def test_normalize_denormalize_roundtrip_is_identity(self):
        norm = InstanceNorm()
        x = mx.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
        x_norm, stats = norm.normalize(x)
        recovered = norm.denormalize(x_norm, stats)
        assert mx.allclose(recovered, x, atol=1e-4).item()

    def test_normalized_series_is_zero_mean_unit_std(self):
        norm = InstanceNorm()
        x = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0]])
        x_norm, _ = norm.normalize(x)
        means = x_norm.mean(axis=1)
        stds = x_norm.std(axis=1)
        assert mx.allclose(means, mx.zeros_like(means), atol=1e-4).item()
        assert mx.allclose(stds, mx.ones_like(stds), atol=1e-2).item()

    def test_constant_series_does_not_produce_nan(self):
        norm = InstanceNorm()
        x = mx.array([[5.0, 5.0, 5.0, 5.0]])  # zero variance
        x_norm, stats = norm.normalize(x)
        mx.eval(x_norm)
        assert not bool(mx.isnan(x_norm).any().item())
        # denormalize still recovers the constant
        assert mx.allclose(norm.denormalize(x_norm, stats), x, atol=1e-4).item()

    def test_denormalize_applies_stats_to_a_shorter_horizon(self):
        # stats come from context (len 5); denormalize a horizon of len 3
        norm = InstanceNorm()
        x = mx.array([[0.0, 10.0, 20.0, 30.0, 40.0]])
        _, stats = norm.normalize(x)
        y_norm = mx.zeros((1, 3))  # normalized zeros -> should map back to the mean
        y = norm.denormalize(y_norm, stats)
        assert y.shape == (1, 3)
        assert mx.allclose(y, mx.full((1, 3), 20.0), atol=1e-4).item()  # mean of x is 20


class TestPatchifier:
    def test_divisible_length_shapes_and_full_mask(self):
        p = Patchifier(patch_len=32)
        x = mx.arange(64.0).reshape(1, 64)
        patches, mask = p.patchify(x)
        assert patches.shape == (1, 2, 32)
        assert mask.shape == (1, 2, 32)
        assert bool((mask == 1.0).all().item())  # no padding

    def test_divisible_length_preserves_content_in_order(self):
        p = Patchifier(patch_len=32)
        x = mx.arange(64.0).reshape(1, 64)
        patches, _ = p.patchify(x)
        flat = patches.reshape(1, 64)
        assert mx.allclose(flat, x).item()

    def test_non_divisible_length_left_pads_and_masks(self):
        p = Patchifier(patch_len=32)
        x = mx.arange(50.0).reshape(1, 50)  # needs 2 patches, 14 pad on the left
        patches, mask = p.patchify(x)
        assert patches.shape == (1, 2, 32)
        # 14 padded positions at the front are masked out
        assert float(mask.sum().item()) == 50.0
        assert bool((mask[0, 0, :14] == 0.0).all().item())
        assert bool((mask[0, 0, 14:] == 1.0).all().item())

    def test_non_divisible_length_recovers_original_after_stripping_pad(self):
        p = Patchifier(patch_len=32)
        x = mx.arange(50.0).reshape(1, 50)
        patches, _ = p.patchify(x)
        flat = patches.reshape(1, 64)
        assert mx.allclose(flat[:, 14:], x).item()  # last 50 are the real series

    def test_batch_is_supported(self):
        p = Patchifier(patch_len=16)
        x = mx.arange(2 * 48.0).reshape(2, 48)
        patches, mask = p.patchify(x)
        assert patches.shape == (2, 3, 16)
        assert mask.shape == (2, 3, 16)
