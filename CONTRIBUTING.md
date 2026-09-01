# Contributing to mlx-tsfm

Thanks for your interest! `mlx-tsfm` is an MLX-native inference layer for time-series foundation
models on Apple Silicon.

## Setup

```bash
uv venv && uv pip install -e '.[dev]'   # add [torch] for the parity oracle / conversion
uv run pytest -q
```

Requires Apple Silicon (arm64 macOS) — MLX does not run on Intel.

## Ground rules

- **Correctness is verified against a reference.** New model backends must ship a numeric-parity
  test against the upstream (PyTorch/JAX) implementation. TimesFM 3.0 parity runs when
  `MLX_TSFM_REF_DIR` points at a checkout of the reference `timesfm3` source (see
  `tests/test_timesfm.py`). Aim for < 1e-4 max abs error across a range of horizons.
- **Test-first.** Every change lands with a test; the fast suite must stay green and offline (real
  weights are downloaded only in guarded tests).
- **No redistributed weights.** Backends convert from the user's own Hugging Face download. Respect
  each model's weight license — TimesFM 3.0 is non-commercial (research only).
- **Style.** `uvx pre-commit run --all` (black/ruff) before submitting.

## Adding a model backend

1. Add `src/mlx_tsfm/models/<name>.py` implementing the architecture (mirror the checkpoint tensor
   names so conversion is a near-identity map).
2. Register it in `src/mlx_tsfm/models/__init__.py`.
3. Add a converter path in `convert.py` and a parity test.
4. Prefer commercially-clean weights (Apache-2.0) for anything meant as a default.

## Attribution

Ports of third-party model code must preserve upstream copyright/notices (see `NOTICE`) and state
the changes made.
