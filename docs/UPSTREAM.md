# Pushing mlx-tsfm upstream into the MLX ecosystem

Status: standalone repo is public and MIT-licensed, with a **parity-verified** MLX-native TimesFM 3.0
backend (max abs error ~1e-6 vs the PyTorch reference across horizons 24–512). That verification is
the credibility centerpiece for every upstream ask below.

## The landscape (why the full package doesn't "merge into MLX")

"MLX" is the `ml-explore` org with several repos, each scoped differently:

| Repo | Scope | Fit for us |
|---|---|---|
| [`ml-explore/mlx`](https://github.com/ml-explore/mlx) | the array framework | only small, general **primitive** PRs — and we needed none (the port used existing ops) |
| [`ml-explore/mlx-lm`](https://github.com/ml-explore/mlx-lm) | **LLMs only** | out of scope (non-autoregressive regressor); we *depend* on it |
| [`ml-explore/mlx-examples`](https://github.com/ml-explore/mlx-examples) | self-contained example dirs | **a `timesfm/` example is the mergeable "official MLX" artifact** |
| community `mlx-<domain>` (`mlx-vlm`, `mlx-audio`) | standalone PyPI pkgs | **this is what `mlx-tsfm` is** — a peer, not a merge target |

## Priority plan

### 1. Contribute the MLX backend to `google-research/timesfm` (strongest home)
Our `models/timesfm.py` is a translation of their Apache-2.0 `timesfm3` code, so contributing it back
is the cleanest licensing story — and it lives *with the model*, like Amazon's experimental
[`mlx` branch of `chronos-forecasting`](https://github.com/amazon-science/chronos-forecasting/tree/mlx).
- **Action:** open an issue proposing an MLX inference backend (they ship flax + torch; MLX is a
  natural third). Lead with the parity numbers + `tests/test_timesfm.py`.
- **Shape:** `src/timesfm3/mlx/` (mirrors their `torch/`), reusing our modules; **Apache-2.0** to match
  their repo (our standalone stays MIT).
- **Effort:** medium. Gate: their maintainers' interest — confirm via the issue before a large PR.

### 2. A `timesfm/` example in `ml-explore/mlx-examples` (the official-MLX showcase)
No time-series example exists there yet — additive and on-theme (they host Whisper/FLUX/CLIP).
- **Constraint:** examples are **self-contained** (own `requirements.txt`, no dependency on a
  community package). So it's a minimal single-file demo whose README points to `mlx-tsfm`.
- **Licensing:** the example downloads the user's own weights (redistributes none), so a TimesFM-3
  research demo is fine; **lead with Chronos-2 (Apache-2.0)** once that backend exists for a broadly
  usable default.
- **Process:** fork → add dir → `uvx pre-commit run --all` → tests → add your name → PR.
- **AI-disclosure:** MLX permits AI-assisted code but requires you understand every line and
  **disclose AI use**; write the PR prose yourself.

### 3. Listings (fast, do first)
- Comment on [MLX Community Projects #654](https://github.com/ml-explore/mlx/discussions/654).
- Submit to [awesome-mlx](https://github.com/raullenchai/awesome-mlx) (issue form, no min stars) and
  the [mlx-lm awesome list](https://github.com/ml-explore/mlx-lm/discussions/1048).

### 4. Core `mlx` primitive PRs — only if needed
None required so far; the port ran on existing ops. Contribute a focused PR only if a real gap
appears (tests that fail on `main` + a benchmark).

## Sequencing

1. **Now:** repo public + listed (step 3). ✅ public
2. **Next:** build the **Chronos-2 (Apache-2.0)** backend → gives a commercial-clean default and a
   cleaner mlx-examples example.
3. Open the **google-research/timesfm** MLX-backend issue with the parity evidence (step 1).
4. Submit the **mlx-examples** example (step 2), leading with Chronos-2.

## Open decisions
- Standalone repo stays **MIT**; the upstreamed copy in google-research/timesfm would be **Apache-2.0**
  to match — confirm this is acceptable to you.
- Lead the public example with **Chronos-2** (recommended) vs TimesFM-3 (research-only weights).
