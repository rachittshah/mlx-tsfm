"""Latency / throughput benchmark for TSFM inference on Apple Silicon.

MLX is lazy: array ops build a graph and only run on ``mx.eval``. The timed region therefore forces
evaluation with ``mx.eval`` so we measure compute, not graph-building. Warmup iterations (which pay
the first-call/compile cost) are excluded, and we report p50/p99 latency plus series/sec throughput
(``batch / p50_seconds`` — the metric that shows batching's near-linear win).
"""
from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx

from .tsfm import TSFMModel


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def benchmark(
    model: TSFMModel,
    *,
    context_len: int,
    horizon: int,
    batch_sizes: list[int],
    n_warmup: int = 5,
    n_iters: int = 20,
    quantiles=None,
) -> dict[int, dict]:
    """Benchmark ``model.forecast_batch`` across ``batch_sizes``; return per-batch metrics."""
    results: dict[int, dict] = {}
    for b in batch_sizes:
        contexts = mx.random.normal((b, context_len))
        mx.eval(contexts)

        def run():
            fc = model.forecast_batch(contexts, horizon, quantiles=quantiles)
            mx.eval(fc.point, fc.quantiles)  # force compute (lazy graph)

        for _ in range(n_warmup):
            run()

        samples = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            run()
            samples.append((time.perf_counter() - t0) * 1000.0)  # ms

        samples.sort()
        p50 = _percentile(samples, 0.50)
        results[b] = {
            "batch": b,
            "p50_ms": p50,
            "p99_ms": _percentile(samples, 0.99),
            "throughput_per_s": b / (p50 / 1000.0) if p50 > 0 else 0.0,
        }
    return results


def _param_count(model: TSFMModel) -> int:
    from mlx.utils import tree_flatten

    return sum(int(v.size) for _, v in tree_flatten(model.parameters()))


def main() -> int:
    ap = argparse.ArgumentParser(prog="mlx_tsfm.bench", description="Benchmark TSFM inference.")
    ap.add_argument("--model", default="toy")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--batch", default="1,8,32", help="comma-separated batch sizes")
    ap.add_argument("--quantize", choices=["int8", "int4"], default=None)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--json", action="store_true")
    # toy-scaling passthrough (lets you run a prod-scale config with random weights)
    ap.add_argument("--d-model", type=int)
    ap.add_argument("--n-layers", type=int)
    ap.add_argument("--n-heads", type=int)
    ap.add_argument("--patch-len", type=int)
    args = ap.parse_args()

    from . import load

    cfg = {"horizon_max": max(args.horizon, 64), "context_len": args.context}
    for k, v in [("d_model", args.d_model), ("n_layers", args.n_layers),
                 ("n_heads", args.n_heads), ("patch_len", args.patch_len)]:
        if v is not None:
            cfg[k] = v
    model = load(args.model, **cfg)

    if args.quantize:
        from .quantize import quantize_model

        quantize_model(model, bits=8 if args.quantize == "int8" else 4)

    batch_sizes = [int(x) for x in args.batch.split(",")]
    res = benchmark(model, context_len=args.context, horizon=args.horizon,
                    batch_sizes=batch_sizes, n_warmup=args.warmup, n_iters=args.iters)

    meta = {"model": args.model, "params": _param_count(model), "context": args.context,
            "horizon": args.horizon, "quantize": args.quantize or "none"}
    if args.json:
        print(json.dumps({"meta": meta, "results": res}, indent=2))
    else:
        print(f"model={meta['model']} params={meta['params']/1e6:.1f}M "
              f"ctx={args.context} horizon={args.horizon} quant={meta['quantize']}")
        print(f"{'batch':>6} {'p50_ms':>10} {'p99_ms':>10} {'series/s':>12}")
        for b in batch_sizes:
            r = res[b]
            print(f"{b:>6} {r['p50_ms']:>10.2f} {r['p99_ms']:>10.2f} {r['throughput_per_s']:>12.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
