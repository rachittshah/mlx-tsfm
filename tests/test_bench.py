"""Tests for the benchmark harness (latency/throughput with lazy-eval barriers)."""
from mlx_tsfm import load
from mlx_tsfm.bench import benchmark


def test_benchmark_returns_metrics_per_batch_size():
    m = load("_stub", d_model=32, n_heads=4, n_layers=2, patch_len=16,
             horizon_max=64, context_len=256)
    res = benchmark(m, context_len=128, horizon=16, batch_sizes=[1, 4],
                    n_warmup=1, n_iters=3)
    assert set(res.keys()) == {1, 4}
    for b, r in res.items():
        assert r["p50_ms"] > 0.0
        assert r["p99_ms"] >= r["p50_ms"]
        assert r["throughput_per_s"] > 0.0
        assert r["batch"] == b


def test_throughput_uses_batch_size():
    # throughput_per_s == batch / p50_seconds, so a bigger batch at similar latency reports more
    m = load("_stub", d_model=32, n_heads=4, n_layers=1, patch_len=16,
             horizon_max=32, context_len=128)
    res = benchmark(m, context_len=64, horizon=8, batch_sizes=[8], n_warmup=1, n_iters=2)
    r = res[8]
    expected = 8 / (r["p50_ms"] / 1000.0)
    assert abs(r["throughput_per_s"] - expected) / expected < 0.01
