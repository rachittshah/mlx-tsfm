"""TimesFM 3.0 — MLX-native inference port of google/timesfm-3.0-pytorch.

A faithful translation of the reference `timesfm3` PyTorch implementation
(https://github.com/google-research/timesfm, Apache-2.0 code) to MLX. Module attribute names
mirror the checkpoint tensor names exactly, so weight conversion is a near-identity mapping
(see convert.py). The architecture (from the model's config.json):

  - input_patch_len=32, output_patch_len=64, 9 quantiles, model_dims=1280, 20 layers, 16 heads
  - "Stacked Mixing Transformer": per layer = sequence-attention -> variate-attention -> FFN,
    each a pre-RMSNorm sublayer whose output is post-RMSNorm'd and added to the residual
  - RoPE on the sequence axis, per-head QK RMSNorm, Pax-style PerDimScale on queries
  - iterative running-stats RevIN, optional linear detrending, patch stitching

Weights are non-commercial (research use only). This backend loads them for MLX inference.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from ..tsfm import Forecast

_RECIP_SOFTPLUS_0 = 1.442695041
_RMS_EPS = 1.1920929e-07  # torch.finfo(float32).eps — matches nn.RMSNorm(eps=None)
_DIV_TOL = 1e-6


# --------------------------------------------------------------------------- primitives
def _softplus(x: mx.array) -> mx.array:
    return mx.logaddexp(x, mx.zeros_like(x))


def _rms_norm(x: mx.array, weight: mx.array | None, eps: float = _RMS_EPS) -> mx.array:
    out = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return out * weight if weight is not None else out


def _roll_back1(x: mx.array, axis: int) -> mx.array:
    """torch.roll(x, shifts=-1, dims=axis) — move each slice one step toward index 0, wrap."""
    n = x.shape[axis]
    idx = mx.concatenate([mx.arange(1, n), mx.array([0])])
    return mx.take(x, idx, axis=axis)


def _rope(x: mx.array, position: mx.array, min_ts: float = 1.0, max_ts: float = 10000.0) -> mx.array:
    """Rotary embedding on (b, n, h, hd); position (b, n). Matches the reference half-split form."""
    hd = x.shape[-1]
    half = hd // 2
    fraction = 2.0 * mx.arange(half).astype(mx.float32) / hd
    timescale = min_ts * (max_ts / min_ts) ** fraction        # (half,)
    sinusoid = position[:, :, None, None].astype(mx.float32) / timescale.reshape(1, 1, 1, -1)
    sin, cos = mx.sin(sinusoid), mx.cos(sinusoid)
    first, second = mx.split(x, 2, axis=-1)
    return mx.concatenate([first * cos - second * sin, second * cos + first * sin], axis=-1)


def _revin(x: mx.array, mu: mx.array, sigma: mx.array, reverse: bool = False) -> mx.array:
    """Expand mu/sigma to x's rank and (de)normalize. mu/sigma have 1 or 2 fewer dims than x."""
    if mu.ndim == x.ndim - 1:
        mu, sigma = mu[..., None], sigma[..., None]
    elif mu.ndim == x.ndim - 2:
        mu, sigma = mu[..., None, None], sigma[..., None, None]
    if reverse:
        return x * sigma + mu
    safe_sigma = mx.where(sigma < _DIV_TOL, 1.0, sigma)
    return (x - mu) / safe_sigma


def _update_running_stats(n, mu, sigma, x, mask):
    """Welford-style merge of a patch (x, mask) into running (n, mu, sigma). Shapes (b,v) / (b,v,p)."""
    legit = ~mask
    inc_n = legit.astype(mx.float32).sum(axis=-1)
    safe_inc_n = mx.where(inc_n == 0, 1.0, inc_n)
    inc_sum = mx.where(legit, x, 0.0).sum(axis=-1)
    inc_mu = mx.where(inc_n == 0, 0.0, inc_sum / safe_inc_n)
    diff_sq = mx.where(legit, (x - inc_mu[..., None]) ** 2, 0.0)
    inc_var = mx.where(inc_n == 0, 0.0, diff_sq.sum(axis=-1) / safe_inc_n)
    inc_sigma = mx.sqrt(inc_var)
    new_n = n + inc_n
    safe_new_n = mx.where(new_n == 0, 1.0, new_n)
    new_mu = mx.where(new_n == 0, 0.0, (n * mu + inc_mu * inc_n) / safe_new_n)
    new_var = mx.where(
        new_n == 0, 0.0,
        (n * sigma * sigma + inc_n * inc_sigma * inc_sigma
         + n * (mu - new_mu) ** 2 + inc_n * (inc_mu - new_mu) ** 2) / safe_new_n,
    )
    return new_n, new_mu, mx.sqrt(new_var)


def _get_running_stats(values: mx.array, masks: mx.array):
    """Cumulative causal running (n, mean, std) per patch. values/masks: (b, v, n, p)."""
    b, v, n, _ = values.shape
    cur = (mx.zeros((b, v)), mx.zeros((b, v)), mx.zeros((b, v)))
    out_n, out_mu, out_sigma = [], [], []
    for i in range(n):
        cur = _update_running_stats(*cur, values[:, :, i, :], masks[:, :, i, :])
        out_n.append(cur[0]); out_mu.append(cur[1]); out_sigma.append(cur[2])
    return mx.stack(out_n, axis=2), mx.stack(out_mu, axis=2), mx.stack(out_sigma, axis=2)


def _cpm_revin_refine(raw_logits, revin_n, revin_mu, revin_sigma, patch_cpm_mask,
                      median_q_idx, rolls, patch_len, num_quantiles, value_clip):
    """Iterative RevIN refinement at CPM (horizon) patches — matches the reference decode()."""
    b, v, n, _ = raw_logits.shape
    median = raw_logits.reshape(b, v, n, rolls, patch_len, num_quantiles)[:, :, :, :, :, median_q_idx]
    carry = (mx.zeros((b, v)), mx.zeros((b, v)), mx.zeros((b, v)))
    anchor = mx.zeros((b, v, rolls, patch_len))
    block_offset = mx.zeros((b,), dtype=mx.int32)
    step_masks = mx.zeros((b, v, patch_len), dtype=mx.bool_)
    ref_mu, ref_sigma = [], []
    for i in range(n):
        is_cpm = patch_cpm_mask[:, i:i + 1]                                   # (b,1)
        onehot = (mx.arange(rolls)[None, :] == block_offset[:, None]).astype(mx.float32)
        predicted_step = (onehot[:, None, :, None] * anchor).sum(axis=2)      # (b,v,p)
        new_n, new_mu, new_sigma = _update_running_stats(*carry, predicted_step, step_masks)
        out_n = mx.where(is_cpm, new_n, revin_n[:, :, i])
        out_mu = mx.where(is_cpm, new_mu, revin_mu[:, :, i])
        out_sigma = mx.where(is_cpm, new_sigma, revin_sigma[:, :, i])
        new_block_offset = mx.where(is_cpm[:, 0], (block_offset + 1) % rolls, mx.zeros_like(block_offset))
        should_update = (new_block_offset == 0)[:, None, None, None]
        step_pred = mx.clip(_revin(median[:, :, i], out_mu, out_sigma, reverse=True), -value_clip, value_clip)
        anchor = mx.where(should_update, step_pred, anchor)
        carry = (out_n, out_mu, out_sigma)
        block_offset = new_block_offset
        ref_mu.append(out_mu); ref_sigma.append(out_sigma)
    return mx.stack(ref_mu, axis=2), mx.stack(ref_sigma, axis=2)


def _output_patch_via_roll(x: mx.array, rolls: int):
    """Build (b, v, n, p*rolls) future-covariate patches by rolling patch index; + wrap mask."""
    b, v, n, p = x.shape
    cur = x
    parts = []
    for _ in range(rolls):
        cur = _roll_back1(cur, axis=2)
        parts.append(cur)
    result = mx.concatenate(parts, axis=-1)  # (b, v, n, rolls*p)
    patch_idx = mx.arange(n)[:, None]
    point_idx = mx.arange(rolls * p)[None, :]
    source_patch = patch_idx + 1 + point_idx // p
    wrap = (source_patch >= n)[None, None, :, :]
    return result, wrap


def _stitch_patches(patch_preds: mx.array, patch_len: int) -> mx.array:
    """Linearly stitch overlapping patch predictions. patch_preds: (b, v, np, patch+overlap, q)."""
    b, v, num_patches, total_len, q = patch_preds.shape
    overlap = total_len - patch_len
    if num_patches == 1:
        return patch_preds[:, :, 0, :, :]
    w = mx.linspace(1.0, 0.0, overlap).reshape(1, 1, 1, overlap, 1)
    first = patch_preds[:, :, 0, :patch_len, :]
    prev, nxt = patch_preds[:, :, :-1, :, :], patch_preds[:, :, 1:, :, :]
    stitched = w * prev[:, :, :, patch_len:, :] + (1.0 - w) * nxt[:, :, :, :overlap, :]
    middles = nxt[:, :, :, overlap:patch_len, :]
    chunks = mx.concatenate([stitched, middles], axis=3).reshape(b, v, (num_patches - 1) * patch_len, q)
    tail = patch_preds[:, :, -1, patch_len:, :]
    return mx.concatenate([first, chunks, tail], axis=2)


# --------------------------------------------------------------------------- config
@dataclass
class TimesFMConfig:
    input_patch_len: int = 32
    output_patch_len: int = 64
    model_dims: int = 1280
    hidden_dims: int = 1280
    num_layers: int = 20
    num_heads: int = 16
    quantiles: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    use_variate_attention: bool = True
    use_stitching: bool = True
    use_linear_detrending: bool = True
    linear_detrending_threshold: float = 0.5
    value_clip: float = 1e20
    use_compile: bool = True   # mx.compile the forward pass (fuses kernels, cuts dispatch overhead)

    @property
    def head_dim(self) -> int:
        return self.model_dims // self.num_heads

    @property
    def num_quantiles(self) -> int:
        return len(self.quantiles)

    @property
    def rolls(self) -> int:
        return self.output_patch_len // self.input_patch_len


# --------------------------------------------------------------------------- modules
class _RMSNorm(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.weight = mx.ones((dims,))

    def __call__(self, x):
        return _rms_norm(x, self.weight)


class _PerDimScale(nn.Module):
    def __init__(self, num_dims: int):
        super().__init__()
        self.per_dim_scale = mx.zeros((num_dims,))
        self._num_dims = num_dims

    def __call__(self, x):
        return x * _RECIP_SOFTPLUS_0 / math.sqrt(self._num_dims) * _softplus(self.per_dim_scale)


class _MHA(nn.Module):
    def __init__(self, cfg: TimesFMConfig, use_rope: bool, causal: bool):
        super().__init__()
        d = cfg.model_dims
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.use_rope = use_rope
        self.causal = causal
        self.query_proj = nn.Linear(d, d, bias=False)
        self.key_proj = nn.Linear(d, d, bias=False)
        self.value_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.query_ln = _RMSNorm(self.head_dim)
        self.key_ln = _RMSNorm(self.head_dim)
        self.per_dim_scale = _PerDimScale(self.head_dim)

    def __call__(self, x: mx.array, patch_mask: mx.array) -> mx.array:
        B, N, _ = x.shape
        # Single-position attention (variate attention with one variate): softmax over a single
        # key is exactly 1, so the output equals the value projection. Skip Q/K/RoPE/norms/softmax.
        if N == 1:
            return self.out_proj(self.value_proj(x))
        h, hd = self.num_heads, self.head_dim
        q = self.query_proj(x).reshape(B, N, h, hd)
        k = self.key_proj(x).reshape(B, N, h, hd)
        v = self.value_proj(x).reshape(B, N, h, hd)
        if self.use_rope:
            pos = mx.broadcast_to(mx.arange(N)[None, :], (B, N))
            q = _rope(q, pos)
            k = _rope(k, pos)
        q = self.query_ln(q)
        k = self.key_ln(k)
        q = self.per_dim_scale(q)
        # (B, N, h, hd) -> (B, h, N, hd)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        # attention mask (True=attend): causal AND not a masked KV patch
        kv_valid = (~patch_mask)[:, None, None, :]                       # (B,1,1,N)
        if self.causal:
            qi = mx.arange(N)[None, None, :, None]
            ki = mx.arange(N)[None, None, None, :]
            attend = (qi >= ki) & kv_valid
        else:
            attend = mx.broadcast_to(kv_valid, (B, 1, N, N))
        bias = mx.where(attend, 0.0, -1e9)
        q = q * math.sqrt(hd)                                            # rescale_logits=False
        logits = (q @ k.transpose(0, 1, 3, 2)) + bias
        w = mx.softmax(logits, axis=-1)
        out = w @ v                                                      # (B,h,N,hd)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, h * hd)
        return self.out_proj(out)


class _MixingLayer(nn.Module):
    def __init__(self, cfg: TimesFMConfig):
        super().__init__()
        d = cfg.model_dims
        self.pre_seq_attn_ln = _RMSNorm(d)
        self.post_seq_attn_ln = _RMSNorm(d)
        self.seq_attn = _MHA(cfg, use_rope=True, causal=True)
        self.use_var = cfg.use_variate_attention
        if self.use_var:
            self.pre_var_attn_ln = _RMSNorm(d)
            self.post_var_attn_ln = _RMSNorm(d)
            self.var_attn = _MHA(cfg, use_rope=False, causal=False)
        self.pre_ff_ln = _RMSNorm(d)
        self.post_ff_ln = _RMSNorm(d)
        self.ff0 = nn.Linear(d, cfg.hidden_dims, bias=False)
        self.ff1 = nn.Linear(cfg.hidden_dims, d, bias=False)

    def __call__(self, x: mx.array, patch_mask: mx.array) -> mx.array:
        b, v, n, d = x.shape
        # sequence attention over n, batched across (b, v)
        sa_in = self.pre_seq_attn_ln(x).reshape(b * v, n, d)
        sa = self.seq_attn(sa_in, patch_mask.reshape(b * v, n)).reshape(b, v, n, d)
        h1 = self.post_seq_attn_ln(sa) + x
        # variate attention over v, batched across (b, n)
        if self.use_var:
            va_in = self.pre_var_attn_ln(h1).transpose(0, 2, 1, 3).reshape(b * n, v, d)
            va_mask = patch_mask.transpose(0, 2, 1).reshape(b * n, v)
            va = self.var_attn(va_in, va_mask).reshape(b, n, v, d).transpose(0, 2, 1, 3)
            h2 = self.post_var_attn_ln(va) + h1
        else:
            h2 = h1
        ff = self.ff1(nn.relu(self.ff0(self.pre_ff_ln(h2))))
        return self.post_ff_ln(ff) + h2


class TimesFM3(nn.Module):
    """MLX-native TimesFM 3.0. Use `TimesFM3.from_pretrained()` to load real weights."""

    def __init__(self, config: TimesFMConfig | None = None):
        super().__init__()
        cfg = config or TimesFMConfig()
        self.config = cfg
        self.pre_transformer_resblock = _ResidualBlock(2 * (cfg.input_patch_len + cfg.output_patch_len),
                                                       cfg.model_dims)
        self.transformer_stack = _Stack(cfg)
        self.output_head = nn.Linear(cfg.model_dims, cfg.output_patch_len * cfg.num_quantiles, bias=True)
        self._cdtype = mx.float32   # transformer compute dtype (bf16/fp16 optional; stats stay fp32)

    # ---- forward over patched inputs (matches reference forward, target-only path) ----
    def _forward_logits(self, values, masks, patch_is_target, patch_cpm_mask=None):
        cfg = self.config
        running_n, mu, sigma = _get_running_stats(values, masks)
        vals_norm = _revin(values, mu, sigma)
        vals_norm = mx.where(masks, 0.0, vals_norm)
        vals_fcov, wrap = _output_patch_via_roll(values, cfg.rolls)
        vals_fcov = _revin(vals_fcov, mu, sigma)
        masks_fcov_raw, _ = _output_patch_via_roll(masks.astype(mx.float32), cfg.rolls)
        masks_fcov = (masks_fcov_raw > 0.5) | patch_is_target[..., None] | wrap
        vals_fcov = mx.where(masks_fcov, 0.0, vals_fcov)
        vals_cat = mx.concatenate([vals_norm, vals_fcov], axis=-1)
        masks_cat = mx.concatenate([masks, masks_fcov], axis=-1)
        resblock_in = mx.concatenate([vals_cat, masks_cat.astype(mx.float32)], axis=-1)
        x = self.pre_transformer_resblock(resblock_in.astype(self._cdtype))
        patch_mask = masks_cat.astype(mx.float32).min(axis=3) > 0.5      # all-masked patch
        eff = mx.cumprod(patch_mask.astype(mx.int32), axis=2) > 0        # leading masked only
        x = self.transformer_stack(x, eff)
        raw = self.output_head(x).astype(mx.float32)                     # (b,v,n,64*9), normalized
        if patch_cpm_mask is not None:
            ref_mu, ref_sigma = _cpm_revin_refine(
                raw, running_n, mu, sigma, patch_cpm_mask,
                cfg.num_quantiles // 2, cfg.rolls, cfg.input_patch_len,
                cfg.num_quantiles, cfg.value_clip)
            cpm = patch_cpm_mask[:, None, :]                             # (b,1,n)
            mu = mx.where(cpm, ref_mu, mu)
            sigma = mx.where(cpm, ref_sigma, sigma)
        raw = _revin(raw, mu, sigma, reverse=True)
        raw = mx.clip(raw, -cfg.value_clip, cfg.value_clip)
        b, v, n = raw.shape[:3]
        return raw.reshape(b, v, n, cfg.output_patch_len, cfg.num_quantiles)

    def _forward_fn(self):
        """Return the forward, mx.compiled once (lazily, after weights are loaded).

        Compiling fuses the whole pass (including the unrolled running-stats / CPM-refine loops)
        into one graph, which removes the per-op and Python-loop dispatch overhead that dominates
        latency at this model size. MLX recompiles per unique input shape and caches.
        """
        if not self.config.use_compile:
            return self._forward_logits
        if getattr(self, "_compiled_forward", None) is None:
            self._compiled_forward = mx.compile(self._forward_logits)
        return self._compiled_forward

    def decode(self, target: mx.array, horizon: int) -> mx.array:
        """target: (b, 1, context) -> logits (b, 1, horizon, num_quantiles). Target-only path."""
        cfg = self.config
        b, num_target, context = target.shape
        p = cfg.input_patch_len
        pad = (p - (context % p)) % p
        mask = mx.zeros((b, context + pad), dtype=mx.bool_)
        if pad:
            target = mx.concatenate([mx.zeros((b, num_target, pad)), target], axis=-1)
            mask = mx.concatenate([mx.ones((b, pad), dtype=mx.bool_),
                                   mx.zeros((b, context), dtype=mx.bool_)], axis=-1)
            context = context + pad
        num_ctx_patches = context // p

        extract_len = min(2 * p, cfg.output_patch_len)
        overlap = extract_len - p
        num_forecast_patches = max(math.ceil((horizon - overlap) / p), 1)
        num_hor_patches = num_forecast_patches + cfg.rolls - 1
        padded_h = num_hor_patches * p

        ctx_masks = mx.broadcast_to(mask[:, None, :], (b, num_target, context))
        ctx_vals = target

        # linear detrending
        m_trend, c_trend, apply_detrend = self._detrend(ctx_vals, ctx_masks, context)
        if cfg.use_linear_detrending:
            t = mx.arange(-(context - 1), 1).astype(mx.float32)[None, None, :] / context
            detr = ctx_vals - (m_trend[..., None] * t + c_trend[..., None])
            ctx_vals = mx.where(apply_detrend[..., None], detr, ctx_vals)
        ctx_vals = mx.where(ctx_masks, 0.0, ctx_vals)

        hor_vals = mx.zeros((b, num_target, padded_h))
        hor_masks = mx.ones((b, num_target, padded_h), dtype=mx.bool_)
        all_vals = mx.concatenate([ctx_vals, hor_vals], axis=-1)
        all_masks = mx.concatenate([ctx_masks, hor_masks], axis=-1)

        n_tot = num_ctx_patches + num_hor_patches
        values_bvnp = all_vals.reshape(b, num_target, n_tot, p)
        masks_bvnp = all_masks.reshape(b, num_target, n_tot, p)
        patch_is_target = mx.ones((b, num_target, n_tot), dtype=mx.bool_)

        # horizon CPM mask: context patches False, horizon patches True
        horizon_cpm = mx.concatenate(
            [mx.zeros((b, num_ctx_patches), dtype=mx.bool_),
             mx.ones((b, num_hor_patches), dtype=mx.bool_)], axis=1)
        logits = self._forward_fn()(values_bvnp, masks_bvnp, patch_is_target, horizon_cpm)

        # stitch forecast patches into the horizon
        fidx = mx.arange(num_forecast_patches) + (num_ctx_patches - 1)
        patch_preds = mx.take(logits, fidx, axis=2)[:, :, :, :extract_len, :]
        horizon_logits = _stitch_patches(patch_preds, p)[:, :, :horizon, :]

        if cfg.use_linear_detrending:
            tf = mx.arange(1, horizon + 1).astype(mx.float32) / context
            trend = m_trend[:, :, None] * tf[None, None, :] + c_trend[:, :, None]
            trend = mx.where(apply_detrend[:, :, None], trend, 0.0)
            horizon_logits = horizon_logits + trend[:, :, :, None]
        return horizon_logits

    def _detrend(self, ctx_vals, ctx_masks, context):
        cfg = self.config
        b, v, _ = ctx_vals.shape
        if not cfg.use_linear_detrending:
            z = mx.zeros((b, v))
            return z, z, mx.zeros((b, v), dtype=mx.bool_)
        t = mx.arange(-(context - 1), 1).astype(mx.float32)[None, None, :] / context
        valid = (~ctx_masks).astype(mx.float32)
        n_v = valid.sum(axis=-1)
        sum_t = mx.where(~ctx_masks, t, 0.0).sum(axis=-1)
        sum_t2 = mx.where(~ctx_masks, t * t, 0.0).sum(axis=-1)
        sum_y = mx.where(~ctx_masks, ctx_vals, 0.0).sum(axis=-1)
        sum_ty = mx.where(~ctx_masks, t * ctx_vals, 0.0).sum(axis=-1)
        det = n_v * sum_t2 - sum_t ** 2
        safe = mx.where(det == 0.0, 1.0, det)
        m = mx.where(det == 0.0, 0.0, (n_v * sum_ty - sum_t * sum_y) / safe)
        c = mx.where(det == 0.0,
                     mx.where(n_v > 0, sum_y / mx.maximum(n_v, 1.0), 0.0),
                     (sum_y - m * sum_t) / mx.maximum(n_v, 1.0))
        detr = ctx_vals - (m[..., None] * t + c[..., None])
        mean_y = sum_y / mx.maximum(n_v, 1.0)
        sum_y2 = mx.where(~ctx_masks, ctx_vals ** 2, 0.0).sum(axis=-1)
        std_orig = mx.sqrt(mx.maximum(sum_y2 / mx.maximum(n_v, 1.0) - mean_y ** 2, 0.0))
        sum_yd = mx.where(~ctx_masks, detr, 0.0).sum(axis=-1)
        mean_yd = sum_yd / mx.maximum(n_v, 1.0)
        sum_yd2 = mx.where(~ctx_masks, detr ** 2, 0.0).sum(axis=-1)
        std_det = mx.sqrt(mx.maximum(sum_yd2 / mx.maximum(n_v, 1.0) - mean_yd ** 2, 0.0))
        apply = std_det < cfg.linear_detrending_threshold * std_orig
        return m, c, apply

    # ---- public forecast API (returns our Forecast type) ----
    def forecast_batch(self, contexts, horizon: int, quantiles=None) -> Forecast:
        x = _as_2d(contexts)                       # (b, L)
        logits = self.decode(x[:, None, :], horizon)   # (b, 1, horizon, q)
        q_all = logits[:, 0, :, :]                 # (b, horizon, q)
        levels = list(self.config.quantiles)
        median = q_all[:, :, self.config.num_quantiles // 2]
        if quantiles is not None:
            idx = [min(range(len(levels)), key=lambda i: abs(levels[i] - q)) for q in quantiles]
            q_all = q_all[:, :, idx]
            levels = list(quantiles)
        return Forecast(point=median, quantiles=q_all, q_levels=levels)

    def forecast(self, context, horizon: int, quantiles=None) -> Forecast:
        return self.forecast_batch(_as_2d(context), horizon, quantiles)

    @classmethod
    def from_pretrained(cls, repo="google/timesfm-3.0-pytorch", weights_path=None,
                        dtype="fp32", compile=True):
        """Load real weights. dtype 'bf16'/'fp16' runs the transformer in reduced precision
        (faster, small accuracy cost); 'fp32' (default) keeps the verified ~1e-6 parity."""
        from mlx.utils import tree_map

        from ..convert import load_timesfm3_weights

        model = cls(TimesFMConfig(use_compile=compile))
        load_timesfm3_weights(model, repo=repo, weights_path=weights_path)
        if dtype in ("bf16", "fp16"):
            ct = mx.bfloat16 if dtype == "bf16" else mx.float16
            for sub in (model.pre_transformer_resblock, model.transformer_stack, model.output_head):
                sub.update(tree_map(lambda p: p.astype(ct), sub.parameters()))
            model._cdtype = ct
        return model


class _ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.hidden_layer = nn.Linear(in_dim, out_dim, bias=False)
        self.output_layer = nn.Linear(out_dim, out_dim, bias=False)
        self.residual_layer = nn.Linear(in_dim, out_dim, bias=False)

    def __call__(self, x):
        return self.output_layer(nn.relu(self.hidden_layer(x))) + self.residual_layer(x)


class _Stack(nn.Module):
    def __init__(self, cfg: TimesFMConfig):
        super().__init__()
        self.layers = [_MixingLayer(cfg) for _ in range(cfg.num_layers)]

    def __call__(self, x, patch_mask):
        for layer in self.layers:
            x = layer(x, patch_mask)
        return x


def _as_2d(x) -> mx.array:
    import numpy as np
    arr = x.astype(mx.float32) if isinstance(x, mx.array) else mx.array(np.asarray(x, dtype=np.float32))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _build(**kw) -> TimesFM3:
    return TimesFM3.from_pretrained()


# registered lazily in models/__init__.py
