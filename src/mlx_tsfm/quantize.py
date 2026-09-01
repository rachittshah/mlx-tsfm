"""Int8 (and int4) mixed-precision quantization for TSFM backends.

Research finding baked in: quantize the transformer *body* to int8 (≈ free quality) but keep the
calibration-sensitive forecasting heads at full precision (int4 on the quantile head wrecks
coverage). ``mlx.nn.quantize`` handles the affine group-quant; this wrapper supplies the predicate
that (a) skips the heads, (b) skips the tiny patch-embed, and (c) only quantizes ``Linear`` layers
whose input dim is divisible by ``group_size``.
"""
from __future__ import annotations

import argparse
import json

import mlx.nn as nn

from .tsfm import TSFMModel


def quantize_model(
    model: TSFMModel,
    *,
    bits: int = 8,
    group_size: int = 64,
    skip_heads: bool = True,
) -> TSFMModel:
    """Quantize the body of ``model`` in place and return it."""

    def predicate(path: str, module: nn.Module):
        if not isinstance(module, nn.Linear):
            return False
        if skip_heads and "head" in path:
            return False
        # affine group-quant requires the input dimension to be a multiple of group_size
        return (module.weight.shape[1] % group_size) == 0

    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=predicate)
    return model


def main() -> int:
    ap = argparse.ArgumentParser(prog="mlx_tsfm.quantize", description="Quantize a TSFM backend.")
    ap.add_argument("--model", required=True, help="registered model id (e.g. 'toy', 'chronos-2')")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--no-skip-heads", action="store_true", help="also quantize the heads (not recommended)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from mlx.utils import tree_flatten

    from . import load

    model = load(args.model)
    quantize_model(model, bits=args.bits, group_size=args.group_size, skip_heads=not args.no_skip_heads)
    keys = [k for k, _ in tree_flatten(model.parameters())]
    quantized = sorted({k.rsplit(".", 1)[0] for k in keys if k.endswith("scales")})
    report = {"model": args.model, "bits": args.bits, "group_size": args.group_size,
              "quantized_layers": len(quantized)}
    print(json.dumps(report, indent=2) if args.json else
          f"quantized {len(quantized)} layers of '{args.model}' to int{args.bits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
