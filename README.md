# mlx-tsfm

**Time-Series Foundation Model inference on Apple Silicon**, built on top of
[`mlx`](https://github.com/ml-explore/mlx) and [`mlx-lm`](https://github.com/ml-explore/mlx-lm).

`mlx-tsfm` is a model-agnostic inference layer for time-series foundation models (TSFMs) on M-series
Macs. It ships an **MLX-native port of [TimesFM 3.0](https://huggingface.co/google/timesfm-3.0-pytorch)**
— numerically verified against the PyTorch reference (max abs error **2.1e-6**, float32 noise) — plus
the parts a *forecaster* needs that an LLM stack does not: a patch front-end, per-series instance
normalization (RevIN), and point + quantile regression heads instead of a token softmax.

> Status: **early** (`0.0.x`). TimesFM 3.0 loads real weights and runs on MLX with verified parity.
> Chronos-2 and `mx.compile` are next — see [Roadmap](#roadmap).

## Why a separate package (not `mlx-lm`)

`mlx-lm` is for autoregressive **language** models: tokenizer in, vocab-softmax out, sampled decode
loop. A TSFM is **non-autoregressive numeric regression**: patches in, point + quantile bands out,
single forward pass. `mlx_lm.convert` assumes a token-embedding + softmax model and does not apply.
`mlx-tsfm` follows the community `mlx-<domain>` convention (`mlx-vlm`, `mlx-audio`) and reuses
`mlx-lm` for what *is* shared (quantization, LoRA/adapters, `mx.compile`).

## Install

```bash
uv venv && uv pip install -e .          # inference (mlx, mlx-lm, numpy)
uv pip install -e '.[torch]'            # + weight conversion / numeric-parity oracle (torch)
uv pip install -e '.[dev]'              # + pytest
```

Requires Apple Silicon (arm64 macOS) — MLX does not run on Intel Macs.

## Quickstart

```python
import numpy as np
from mlx_tsfm import load

model = load("timesfm-3.0")                    # downloads 1.3GB on first use (research license)
ctx = np.sin(np.linspace(0, 40, 512))          # a 1-D context series
fc = model.forecast(ctx, horizon=64, quantiles=[0.1, 0.5, 0.9])
print(fc.point.shape)       # (1, 64)          median forecast
print(fc.quantiles.shape)   # (1, 64, 3)       probabilistic band
```

Batch many series through the single non-autoregressive forward pass:

```python
contexts = np.stack([series1, series2, series3])   # (B, L)
fc = model.forecast_batch(contexts, horizon=64)    # fc.point: (B, 64)
```

## CLI

```bash
mlx_tsfm.forecast --model timesfm-3.0 --input series.npy --horizon 64 --quantiles 0.1,0.5,0.9
mlx_tsfm.bench    --model timesfm-3.0 --context 512 --horizon 64 --batch 1,8,32 [--quantize int8]
mlx_tsfm.convert  --hf google/timesfm-3.0-pytorch --out models/timesfm-3.0-mlx --dtype bf16
```

## Numeric parity

The MLX `decode()` matches the reference `timesfm3` PyTorch `decode()` to **~1e-6 max abs error**
(float32 noise) across horizons 24–512 — including the multi-patch path with iterative CPM RevIN
refinement — on multiple signals. All 445 checkpoint tensors map 1:1 onto the MLX parameter tree.
The parity test (`tests/test_timesfm.py`) runs against a checkout of the reference source when
`MLX_TSFM_REF_DIR` is set.

## Benchmarks (real TimesFM-3, Apple M4 Max)

330.7M params, context 512, horizon 64:

| precision | batch=1 p50 | batch=8 | batch=32 | throughput @32 |
|---|---|---|---|---|
| fp32 | 20.4 ms | 30.8 ms | 67.4 ms | **475 series/s** |
| int8 (body) | 19.7 ms | 29.3 ms | 68.6 ms | 467 series/s |

Batching scales throughput near-linearly (49 → 259 → 475 series/s) — the single highest-leverage
optimization, since TimesFM 3 decodes the whole horizon in one non-autoregressive pass. int8 gives a
small batch-1 win (weight-memory-bound); the short patch sequence keeps larger batches GEMM-bound.

## Model support & licensing

| Model | Type | Weights license | Commercial | Status |
|---|---|---|---|---|
| TimesFM 3.0 (`google/timesfm-3.0-pytorch`) | decoder, patch, 9-quantile | non-commercial | ❌ research only | **works — MLX-native, parity-verified** |
| Chronos-2 (`amazon/chronos-2`) | encoder, quantile | Apache-2.0 | ✅ | planned (commercial default) |
| TimesFM 2.5 / 2.0 | decoder, patch | Apache-2.0 (verify 2.5) | ✅ | planned |

**mlx-tsfm redistributes no model weights.** Backends convert from the user's own Hugging Face
download. TimesFM 3.0 weights are **non-commercial (research only)**; a commercially-clean default
(Chronos-2, Apache-2.0) is on the roadmap.

## Roadmap

- [x] Model-agnostic core: patching, instance norm, point/quantile heads, `TSFMModel` base + registry, `Forecast`, batching
- [x] **TimesFM 3.0 MLX-native backend — real weights, numeric parity verified (2.1e-6)**
- [x] `convert` (PyTorch → MLX) with a 1:1 tensor-name map; int8 mixed-precision quantization
- [x] `bench` (lazy-eval-barrier latency/throughput) + `forecast` CLIs
- [ ] Chronos-2 backend (commercial default) + TimesFM 2.5
- [ ] `mx.compile` at fixed shapes; multivariate covariates; LoRA adapters (milbench-rl format)

## License

MIT (code). Model weights retain their own licenses (see table above).
