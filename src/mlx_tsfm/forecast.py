"""`mlx_tsfm.forecast` — forecast a series from a .npy file via a registered backend."""
from __future__ import annotations

import argparse

import numpy as np

from .tsfm import Forecast


def load_series(path: str) -> np.ndarray:
    """Load a 1-D (or (B, L)) series from a .npy file."""
    return np.load(path)


def run_forecast(model_id: str, series, horizon: int, quantiles=None, **load_kwargs) -> Forecast:
    from . import load

    return load(model_id, **load_kwargs).forecast(series, horizon, quantiles=quantiles)


def main() -> int:
    ap = argparse.ArgumentParser(prog="mlx_tsfm.forecast", description="Forecast a .npy series.")
    ap.add_argument("--model", default="toy")
    ap.add_argument("--input", required=True, help="path to a .npy series")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--quantiles", default=None, help="comma-separated, e.g. 0.1,0.5,0.9")
    ap.add_argument("--quantize", choices=["int8", "int4"], default=None)
    args = ap.parse_args()

    import mlx.core as mx

    from . import load

    q = [float(x) for x in args.quantiles.split(",")] if args.quantiles else None
    model = load(args.model)
    if args.quantize:
        from .quantize import quantize_model

        quantize_model(model, bits=8 if args.quantize == "int8" else 4)

    fc = model.forecast(load_series(args.input), args.horizon, quantiles=q)
    mx.eval(fc.point, fc.quantiles)
    print("point:", np.array(fc.point)[0].round(4).tolist())
    if q:
        print("quantiles", fc.q_levels, "shape", tuple(fc.quantiles.shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
