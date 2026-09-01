"""`mlx_tsfm.convert` — convert PyTorch/JAX TSFM checkpoints to MLX safetensors.

Scaffold. The per-backend weight mapping (patch-embed, alternating temporal/variate attention,
point + quantile heads) plus a per-layer numeric-parity gate (< 1e-3 vs a torch/MPS oracle) is the
next milestone (M0). The CLI surface exists so the workflow is discoverable, and each backend fails
loudly until its converter lands.
"""
from __future__ import annotations

import argparse

# HF ids -> converter status. All pending until the numeric-parity port lands.
_BACKENDS = {
    "amazon/chronos-2": "chronos2",
    "google/timesfm-3.0-pytorch": "timesfm3",
    "google/timesfm-2.0-500m-pytorch": "timesfm2",
}


def convert(hf_id: str, out: str, *, dtype: str = "bf16", quantize: str | None = None) -> str:
    """Convert an HF checkpoint to an MLX model directory at ``out`` (not yet implemented)."""
    raise NotImplementedError(
        f"Weight conversion for '{hf_id}' is not implemented yet. "
        f"Planned backends: {sorted(_BACKENDS)}. "
        "Milestone M0 adds the weight mapping + a <1e-3 per-layer parity gate vs the PyTorch oracle."
    )


def main() -> int:
    ap = argparse.ArgumentParser(prog="mlx_tsfm.convert", description="Convert a TSFM checkpoint to MLX.")
    ap.add_argument("--hf", required=True, help="Hugging Face model id")
    ap.add_argument("--out", required=True, help="output MLX model directory")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("-q", "--quantize", choices=["int8", "int4"], default=None)
    args = ap.parse_args()
    try:
        out = convert(args.hf, args.out, dtype=args.dtype, quantize=args.quantize)
        print(f"wrote {out}")
        return 0
    except NotImplementedError as e:
        print(f"[not yet implemented] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
