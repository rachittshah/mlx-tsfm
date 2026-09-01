# mlx-tsfm

**Time-Series Foundation Model inference on Apple Silicon**, built on top of
[`mlx`](https://github.com/ml-explore/mlx) and [`mlx-lm`](https://github.com/ml-explore/mlx-lm).

`mlx-tsfm` is a model-agnostic inference layer for time-series foundation models (TSFMs) —
TimesFM, Chronos-2, Moirai — on M-series Macs. It reuses `mlx-lm`'s quantization, LoRA/adapter,
and `mx.compile` machinery, and adds the parts a *forecaster* needs that an LLM stack does not:
a patch front-end, per-series instance normalization, and point + quantile regression heads
(instead of a token softmax).

> Status: **early** (`0.0.x`). The model-agnostic core + a runnable reference (`toy`) backend
> work today and are covered by tests. The real weight-loading backends (TimesFM, Chronos-2) are
> in progress — see [Roadmap](#roadmap).

## Why a separate package (not `mlx-lm`)

`mlx-lm` is for autoregressive **language** models: tokenizer in, vocab-softmax out, sampled decode
loop. A TSFM is **non-autoregressive numeric regression**: patches in, point + quantile bands out,
single forward pass. `mlx_lm.convert` assumes a token-embedding + softmax model and does not apply.
`mlx-tsfm` follows the community `mlx-<domain>` convention (`mlx-vlm`, `mlx-audio`) and depends on
`mlx-lm` for the pieces that *are* shared.

## Install

```bash
uv venv && uv pip install -e .          # inference (mlx, mlx-lm, numpy)
uv pip install -e '.[torch]'            # + weight conversion from PyTorch checkpoints
uv pip install -e '.[dev]'              # + pytest, pre-commit
```

Requires Apple Silicon (arm64 macOS) — MLX does not run on Intel Macs.

## Quickstart

```python
import numpy as np
from mlx_tsfm import load

model = load("toy", horizon_max=64)          # runnable reference backend, random weights
ctx = np.sin(np.linspace(0, 20, 256))         # a 1-D context series
fc = model.forecast(ctx, horizon=24, quantiles=[0.1, 0.5, 0.9])
print(fc.point.shape)       # (1, 24)         point forecast
print(fc.quantiles.shape)   # (1, 24, 3)      probabilistic band
```

Batch many series through the single non-autoregressive forward pass (the #1 optimization):

```python
contexts = np.stack([series1, series2, series3])   # (B, L)
fc = model.forecast_batch(contexts, horizon=24)    # fc.point: (B, 24)
```

## CLI

```bash
mlx_tsfm.forecast --model toy --input series.npy --horizon 24 --quantiles 0.1,0.5,0.9
mlx_tsfm.bench    --model toy --context 512 --horizon 64 --batch 1,8,32 --json
# (in progress) mlx_tsfm.convert  --hf amazon/chronos-2 --out models/chronos-2-mlx
# (in progress) mlx_tsfm.quantize --model models/chronos-2-mlx --bits 8 --group-size 64
```

## Model support & licensing

| Model | Type | Weights license | Commercial | Status |
|---|---|---|---|---|
| `toy` | reference patched transformer (random init) | — | — | **works** (for testing the pipeline) |
| Chronos-2 (`amazon/chronos-2`) | encoder, quantile | Apache-2.0 | ✅ | **recommended default** — in progress |
| TimesFM 3.0 (`google/timesfm-3.0-pytorch`) | decoder, patch, 9-quantile | non-commercial | ❌ research only | in progress |
| TimesFM 2.5 / 2.0 | decoder, patch | Apache-2.0 (verify 2.5) | ✅ | planned |
| Moirai (Salesforce) | encoder, any-variate | CC-BY-NC-4.0 | ❌ | planned |

**mlx-tsfm redistributes no model weights.** Backends convert from the user's own Hugging Face
download at runtime. TimesFM 3.0 weights are non-commercial and are supported for research only;
the recommended commercially-clean default is **Chronos-2 (Apache-2.0)**.

## Benchmarks (local)

Measured on an **Apple M4 Max (38 GB)** with a prod-scale `toy` config (`d_model=1024`, 24 layers,
16 heads, patch 32) — **302.9M params**, context 512, horizon 64, random weights (this measures the
*inference engineering*, not forecast accuracy):

| precision | batch=1 p50 | batch=8 | batch=32 | batch=32 throughput |
|---|---|---|---|---|
| fp32 | 7.95 ms | 12.39 ms | 38.06 ms | **841 series/s** |
| int8 (body) | **5.06 ms** | 12.02 ms | 37.80 ms | 847 series/s |

Two findings, exactly as predicted: **batching scales throughput near-linearly** (126 → 646 → 841
series/s), and **int8 helps most at batch 1** (7.95 → 5.06 ms, 1.6×; weight-memory-bound), converging
with fp32 at large batch (compute-bound). Reproduce:

```bash
mlx_tsfm.bench --model toy --d-model 1024 --n-layers 24 --n-heads 16 --patch-len 32 \
               --context 512 --horizon 64 --batch 1,8,32           # add --quantize int8
```

## Roadmap

- [x] Model-agnostic core: patching, instance norm, point/quantile heads, `TSFMModel` base + registry, `Forecast`, batching
- [x] `toy` reference backend — end-to-end forecast on MLX, tested (36 tests)
- [x] int8 mixed-precision quantization (body int8, heads full-precision) via `mlx.nn.quantize`
- [x] `bench` (latency/throughput, lazy-eval barriers) + `forecast` CLIs
- [ ] `convert` (PyTorch → MLX) with per-layer parity gate (<1e-3 vs torch oracle)
- [ ] Chronos-2 backend (commercial default), then TimesFM 3 (research)
- [ ] `mx.compile` at fixed shapes; LoRA adapters (milbench-rl format)

## License

MIT (code). Model weights retain their own licenses (see table above).
