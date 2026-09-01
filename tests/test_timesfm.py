"""Tests for the MLX-native TimesFM 3.0 backend.

The structural test is fast (no weights). The real-weight and parity tests are guarded: they skip
unless the 1.3GB checkpoint is already cached locally (and, for parity, unless MLX_TSFM_REF_DIR
points at a checkout of the reference `timesfm3` source), so a normal `pytest` run stays fast and
offline.
"""
from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

from mlx_tsfm.models.timesfm import TimesFM3

REPO = "google/timesfm-3.0-pytorch"


def _cached_weights() -> str | None:
    try:
        from huggingface_hub import try_to_load_from_cache

        p = try_to_load_from_cache(REPO, "model.safetensors")
        return p if isinstance(p, str) and os.path.exists(p) else None
    except Exception:
        return None


def test_param_tree_matches_checkpoint_names():
    """The MLX module's parameter tree must map 1:1 onto the checkpoint tensor names."""
    try:
        from huggingface_hub import get_safetensors_metadata
    except Exception:
        pytest.skip("huggingface_hub unavailable")
    try:
        md = get_safetensors_metadata(REPO)
    except Exception:
        pytest.skip("HF metadata unreachable (offline)")
    ckpt_names = set(md.weight_map)
    from mlx.utils import tree_flatten

    model = TimesFM3()
    mine = {k for k, _ in tree_flatten(model.parameters())}
    assert mine == ckpt_names


@pytest.mark.skipif(_cached_weights() is None, reason="TimesFM-3 weights not cached locally")
def test_real_weights_load_and_forecast():
    from mlx_tsfm.convert import load_timesfm3_weights

    model = TimesFM3()
    load_timesfm3_weights(model, weights_path=_cached_weights())
    ctx = np.sin(np.linspace(0, 40, 512)).astype(np.float32)
    fc = model.forecast(ctx, horizon=64, quantiles=[0.1, 0.5, 0.9])
    mx.eval(fc.point, fc.quantiles)
    assert fc.point.shape == (1, 64)
    assert fc.quantiles.shape == (1, 64, 3)
    assert not bool(mx.isnan(fc.point).any().item())


@pytest.mark.skipif(
    _cached_weights() is None or not os.environ.get("MLX_TSFM_REF_DIR"),
    reason="needs cached weights and MLX_TSFM_REF_DIR pointing at the reference timesfm3 source",
)
def test_numeric_parity_vs_pytorch_reference():
    """MLX decode() must match the reference torch decode() to < 1e-4 (float32 noise)."""
    import sys

    ref_dir = os.environ["MLX_TSFM_REF_DIR"]
    sys.path.insert(0, os.path.dirname(ref_dir.rstrip("/")))
    import importlib

    ref_mod = importlib.import_module(os.path.basename(ref_dir.rstrip("/")) + ".model")
    import torch
    from safetensors.torch import load_file as load_torch

    wp = _cached_weights()
    ref = ref_mod.TimesFM3Torch()
    ref.load_state_dict(load_torch(wp), strict=False)
    ref.eval()

    from mlx_tsfm.convert import load_timesfm3_weights

    m = TimesFM3()
    load_timesfm3_weights(m, weights_path=wp)

    ctx = np.sin(np.linspace(0, 40, 512)).astype(np.float32)
    # horizon 64 = single forecast patch; horizon 128 exercises multi-patch + cpm_revin_refine
    for horizon in (64, 128):
        with torch.no_grad():
            ref_logits = ref.decode(torch.tensor(ctx)[None, None, :], horizon=horizon).numpy()
        mine = np.array(m.decode(mx.array(ctx)[None, None, :], horizon))
        assert np.abs(ref_logits - mine).max() < 1e-4, f"parity failed at horizon {horizon}"
