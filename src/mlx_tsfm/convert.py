"""`mlx_tsfm.convert` — load TimesFM PyTorch checkpoints into MLX.

The MLX module names in `models/timesfm.py` mirror the checkpoint tensor names exactly, so loading
is a near-identity mapping: read the safetensors, wrap each tensor as an ``mx.array``, and
``update`` the model. TimesFM 3.0 weights are non-commercial (research use only).
"""
from __future__ import annotations

import argparse

import mlx.core as mx

_SUPPORTED = {
    "google/timesfm-3.0-pytorch": "timesfm3",
}


def load_timesfm3_weights(model, *, repo: str = "google/timesfm-3.0-pytorch", weights_path: str | None = None):
    """Load real TimesFM-3 weights from a safetensors checkpoint into an MLX ``TimesFM3`` model."""
    from safetensors.numpy import load_file
    from mlx.utils import tree_flatten, tree_unflatten

    if weights_path is None:
        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download(repo, "model.safetensors")

    sd = load_file(weights_path)  # {name: np.ndarray}, framework-agnostic
    ckpt = {k: mx.array(v) for k, v in sd.items()}

    # sanity: every checkpoint tensor must have a home in the model, and vice versa
    model_keys = {k for k, _ in tree_flatten(model.parameters())}
    ckpt_keys = set(ckpt)
    missing = model_keys - ckpt_keys
    extra = ckpt_keys - model_keys
    if missing or extra:
        raise ValueError(
            f"weight/name mismatch: {len(missing)} model params unmapped "
            f"(e.g. {sorted(missing)[:3]}), {len(extra)} checkpoint tensors unused "
            f"(e.g. {sorted(extra)[:3]})"
        )

    model.update(tree_unflatten(list(ckpt.items())))
    mx.eval(model.parameters())
    return model


def convert(hf_id: str, out: str, *, dtype: str = "bf16", quantize: str | None = None) -> str:
    """Convert an HF TimesFM checkpoint to an MLX safetensors directory at ``out``."""
    if _SUPPORTED.get(hf_id) != "timesfm3":
        raise NotImplementedError(
            f"Conversion for '{hf_id}' is not implemented yet. Supported: {sorted(_SUPPORTED)}."
        )
    import os

    from mlx.utils import tree_flatten

    from .models.timesfm import TimesFM3

    model = TimesFM3()
    load_timesfm3_weights(model)
    if dtype in ("bf16", "fp16"):
        cast = mx.bfloat16 if dtype == "bf16" else mx.float16
        model.update(tree_unflatten_cast(model, cast))
    if quantize:
        from .quantize import quantize_model

        quantize_model(model, bits=8 if quantize == "int8" else 4)

    os.makedirs(out, exist_ok=True)
    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(os.path.join(out, "model.safetensors"), flat)
    return out


def tree_unflatten_cast(model, dtype):
    from mlx.utils import tree_flatten, tree_unflatten

    return tree_unflatten([(k, v.astype(dtype)) for k, v in tree_flatten(model.parameters())])


def main() -> int:
    ap = argparse.ArgumentParser(prog="mlx_tsfm.convert", description="Convert a TimesFM checkpoint to MLX.")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="fp32", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("-q", "--quantize", choices=["int8", "int4"], default=None)
    args = ap.parse_args()
    try:
        out = convert(args.hf, args.out, dtype=args.dtype, quantize=args.quantize)
        print(f"wrote {out}/model.safetensors")
        return 0
    except NotImplementedError as e:
        print(f"[not yet implemented] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
