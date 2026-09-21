# pylint: skip-file
"""Native JAX (Flax NNX) port of Gemma 4 — text-only tower.

Faithful line-for-line port of the classes in
``transformers/models/gemma4/modeling_gemma4.py`` that make up the text
tower. Skips: audio (Gemma4Audio*), vision (Gemma4Vision*), multimodal
orchestrators (Gemma4Model, Gemma4ForConditionalGeneration,
Gemma4MultimodalEmbedder), and MoE (Gemma4TextExperts, Gemma4TextRouter) —
E4B is dense per-layer.

Critical divergences from the torch reference documented inline with
`# PORT:` comments. The biggest one is attention dispatch: HF uses
``ALL_ATTENTION_FUNCTIONS[impl]``; we call XLA SDPA directly via
``_attn_xla_sdpa``, or the Pallas `splash_attention` kernel (exp 35) via
``model.pallas_attention.splash_attention`` when ``JAX_ATTENTION_IMPL=splash``
is set. Selection is per-call (env var read at forward time), so the same
compiled module handles both paths across processes.
"""
from __future__ import annotations

import math
import os
from typing import Any, Optional

from flax import nnx
import jax
import jax.numpy as jnp
from transformers import Gemma4TextConfig


# -----------------------------------------------------------------------------
# Small helpers (stateless pure jax — not nnx.Modules)
# -----------------------------------------------------------------------------


def _rotate_half(x: jax.Array) -> jax.Array:
  """Rotate the last dim by half. Matches HF rotate_half."""
  d = x.shape[-1]
  x1 = x[..., : d // 2]
  x2 = x[..., d // 2 :]
  return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(
    x: jax.Array,  # (B, T, H, D)
    cos: jax.Array,  # (B, T, D)
    sin: jax.Array,  # (B, T, D)
    unsqueeze_dim: int = 2,
) -> jax.Array:
  """Apply RoPE matching HF apply_rotary_pos_emb with unsqueeze_dim=2."""
  cos = jnp.expand_dims(cos, unsqueeze_dim)
  sin = jnp.expand_dims(sin, unsqueeze_dim)
  return (x * cos) + (_rotate_half(x) * sin)


def _repeat_kv(hidden_states: jax.Array, n_rep: int) -> jax.Array:
  """GQA KV-head replication. Shape (B, K, T, D) -> (B, K*n_rep, T, D)."""
  if n_rep == 1:
    return hidden_states
  b, k, t, d = hidden_states.shape
  hidden_states = jnp.broadcast_to(
      hidden_states[:, :, None, :, :], (b, k, n_rep, t, d)
  )
  return hidden_states.reshape(b, k * n_rep, t, d)


def _gelu_pytorch_tanh(x: jax.Array) -> jax.Array:
  """Match torch.nn.functional.gelu(approximate='tanh').

  HF's 'gelu_pytorch_tanh' activation is exactly jax.nn.gelu with
  approximate=True.
  """
  return jax.nn.gelu(x, approximate=True)


# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------


class Gemma4RMSNorm(nnx.Module):
  """Port of Gemma4RMSNorm. HF init: weight=ones.

  Forward multiplies normed output by ``weight`` (NOT ``1 + weight`` — that was
  the Gemma 2/3 convention). Fp32 compute; cast back to input dtype at the end.

  If ``with_scale`` is False, the weight buffer is omitted and forward
  returns the normalized tensor unscaled (still upcast-then-downcast).

  Exp 52+: `weights_dtype` controls storage (fp32 master or bf16 legacy)
  and is the only dtype that actually matters here — norms are inherently
  upcast-to-fp32 in forward. ``dtype`` is accepted as a legacy alias for
  ``weights_dtype`` for backward compat.
  """

  def __init__(
      self,
      dim: int,
      eps: float = 1e-6,
      *,
      with_scale: bool = True,
      weights_dtype: jnp.dtype | None = None,
      compute_dtype: jnp.dtype | None = None,
      dtype: jnp.dtype = jnp.float32,  # legacy alias
      rngs: nnx.Rngs,
  ):
    if weights_dtype is None:
      weights_dtype = dtype
    if compute_dtype is None:
      compute_dtype = dtype
    self.eps = eps
    self.with_scale = with_scale
    self.dim = dim
    self.weights_dtype = weights_dtype
    self.compute_dtype = compute_dtype
    if with_scale:
      # Init to ones — matches HF nn.Parameter(torch.ones(dim)).
      self.weight = nnx.Param(jnp.ones((dim,), dtype=weights_dtype))

  def __call__(self, x: jax.Array) -> jax.Array:
    in_dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    mean_sq = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + self.eps
    # torch port uses `pow(mean_sq, -0.5)` (not rsqrt) for xla parity.
    normed = x_f32 * jnp.pow(mean_sq, jnp.float32(-0.5))
    if self.with_scale:
      normed = normed * self.weight.value.astype(jnp.float32)
    return normed.astype(in_dtype)


class Gemma4TextScaledWordEmbedding(nnx.Module):
  """Embedding table that multiplies output by ``sqrt(embedding_dim)``.

  Matches HF's Gemma4TextScaledWordEmbedding: the scale factor is baked as
  a scalar constant (the HF code registers it as a non-persistent buffer
  ``embed_scale`` then casts to weight dtype at forward time).

  Exp 52+: storage = ``weights_dtype`` (fp32 master or bf16 legacy).
  Output is cast to ``compute_dtype`` (bf16 under AMP) so downstream
  attention / MLP receives bf16 activations regardless of fp32 master
  weights. Gather on fp32 table is fine; the single downcast to bf16
  happens once per forward, matches the "fp32 master + bf16 compute"
  contract.
  """

  def __init__(
      self,
      num_embeddings: int,
      embedding_dim: int,
      embed_scale: float = 1.0,
      *,
      weights_dtype: jnp.dtype | None = None,
      compute_dtype: jnp.dtype | None = None,
      dtype: jnp.dtype = jnp.bfloat16,  # legacy alias
      rngs: nnx.Rngs,
  ):
    if weights_dtype is None:
      weights_dtype = dtype
    if compute_dtype is None:
      compute_dtype = dtype
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.embed_scale = float(embed_scale)
    self.weights_dtype = weights_dtype
    self.compute_dtype = compute_dtype
    # Init random-normal — real weights come from from_pretrained.
    key = rngs.params()
    self.weight = nnx.Param(
        jax.random.normal(
            key, (num_embeddings, embedding_dim), dtype=weights_dtype
        )
        * 0.02
    )

  def __call__(self, input_ids: jax.Array) -> jax.Array:
    # Gather in storage dtype, then downcast to compute dtype so
    # downstream ops see bf16 activations under fp32-master AMP.
    out = self.weight.value[input_ids]  # (B, T, D) in weights_dtype
    if out.dtype != self.compute_dtype:
      out = out.astype(self.compute_dtype)
    scale = jnp.array(self.embed_scale, dtype=out.dtype)
    return out * scale


class Linear(nnx.Module):
  """Minimal bias-free Linear matching torch.nn.Linear convention.

  HF's ``nn.Linear(in, out)`` stores weight as (out, in) and computes
  ``x @ weight.T``. We keep the **same storage shape** (out, in) so the
  HF safetensors keys map 1:1 without a transpose at load time.

  Exp 52+: mixed precision. ``weights_dtype`` is the storage dtype
  (fp32 master) and ``compute_dtype`` is the dtype used for the matmul
  (bf16 compute). When they differ the matmul downcasts W to
  compute_dtype at call time (a single HBM→HBM bf16 cast per forward;
  for fp32→bf16 matmuls XLA also folds the cast into the dot when it
  can). Activations coming in are expected to already be in
  compute_dtype; we cast defensively to guarantee homogeneous dot-dtype.
  Bias, if present, is stored in weights_dtype and cast to the output's
  dtype on add.
  """

  def __init__(
      self,
      in_features: int,
      out_features: int,
      *,
      bias: bool = False,
      weights_dtype: jnp.dtype | None = None,
      compute_dtype: jnp.dtype | None = None,
      dtype: jnp.dtype = jnp.bfloat16,  # legacy alias
      rngs: nnx.Rngs,
  ):
    if weights_dtype is None:
      weights_dtype = dtype
    if compute_dtype is None:
      compute_dtype = dtype
    self.in_features = in_features
    self.out_features = out_features
    self.weights_dtype = weights_dtype
    self.compute_dtype = compute_dtype
    key = rngs.params()
    # Store (out, in) — matches torch nn.Linear.weight shape.
    init_std = 1.0 / math.sqrt(in_features)
    self.weight = nnx.Param(
        jax.random.uniform(
            key,
            (out_features, in_features),
            minval=-init_std,
            maxval=init_std,
            dtype=weights_dtype,
        )
    )
    if bias:
      self.bias = nnx.Param(jnp.zeros((out_features,), dtype=weights_dtype))
    else:
      self.bias = None

  def __call__(self, x: jax.Array) -> jax.Array:
    # x: (..., in_features); weight: (out, in); result: (..., out).
    # Downcast weight to compute_dtype (no-op if storage already = compute).
    # Cast activation defensively so the dot sees a single dtype.
    w = self.weight.value
    if w.dtype != self.compute_dtype:
      w = w.astype(self.compute_dtype)
    if x.dtype != self.compute_dtype:
      x = x.astype(self.compute_dtype)
    out = x @ w.T
    if self.bias is not None:
      b = self.bias.value
      if b.dtype != out.dtype:
        b = b.astype(out.dtype)
      out = out + b
    return out


# -----------------------------------------------------------------------------
# RoPE
# -----------------------------------------------------------------------------


def _compute_default_inv_freq(rope_theta: float, head_dim: int) -> jax.Array:
  """HF compute_default_rope_parameters: inv_freq[i] = base^(-2i/head_dim),

  for i in [0, head_dim/2).
  """
  idxs = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
  return 1.0 / (rope_theta ** (idxs / float(head_dim)))


def _compute_proportional_inv_freq(
    rope_theta: float, head_dim: int, partial_rotary_factor: float
) -> jax.Array:
  """HF _compute_proportional_rope_parameters with partial_rotary_factor:

  the first ``2 * rope_angles`` dims get real inv_freq; the rest are zero
  (nope — no position). rope_angles = int(factor * head_dim // 2).
  """
  rope_angles = int(partial_rotary_factor * head_dim // 2)
  idxs = jnp.arange(0, 2 * rope_angles, 2, dtype=jnp.float32)
  inv_rotated = 1.0 / (rope_theta ** (idxs / float(head_dim)))
  nope_angles = head_dim // 2 - rope_angles
  if nope_angles > 0:
    inv_freq = jnp.concatenate(
        [inv_rotated, jnp.zeros((nope_angles,), dtype=jnp.float32)], axis=0
    )
  else:
    inv_freq = inv_rotated
  return inv_freq


class Gemma4TextRotaryEmbedding(nnx.Module):
  """Precomputes inv_freq for each layer type and builds (cos, sin) per forward.

  Two variants: 'sliding_attention' uses default RoPE with theta=10_000 and
  head_dim=256 (E4B). 'full_attention' uses proportional RoPE with
  partial_rotary_factor=0.25, theta=1_000_000, head_dim=512 (global_head_dim).
  """

  def __init__(self, config: Gemma4TextConfig):
    self.config = config
    self.layer_types = sorted(set(config.layer_types))
    inv_freqs: dict[str, jax.Array] = {}
    attn_scaling: dict[str, float] = {}
    for layer_type in self.layer_types:
      rope_params = config.rope_parameters[layer_type]
      rope_type = rope_params["rope_type"]
      theta = float(rope_params["rope_theta"])
      if layer_type == "full_attention":
        head_dim = config.global_head_dim or config.head_dim
      else:
        head_dim = config.head_dim
      if rope_type == "proportional":
        factor = float(rope_params.get("partial_rotary_factor", 1.0))
        inv_freq = _compute_proportional_inv_freq(theta, head_dim, factor)
        inv_freq = inv_freq / float(rope_params.get("factor", 1.0))
      else:  # 'default'
        inv_freq = _compute_default_inv_freq(theta, head_dim)
      inv_freqs[layer_type] = inv_freq  # non-trainable constant
      attn_scaling[layer_type] = 1.0
    # Wrap dict-of-arrays with nnx.data so NNX treats it as pytree data.
    self._inv_freqs = nnx.data(inv_freqs)
    self._attn_scaling = attn_scaling

  def __call__(
      self, position_ids: jax.Array, layer_type: str, dtype: jnp.dtype
  ):
    """position_ids: (B, T). Returns (cos, sin) each shape (B, T, head_dim)."""
    inv_freq = self._inv_freqs[layer_type]  # (D/2,)
    # freqs = inv_freq[None, None, :] * position_ids[:, :, None]
    pos = position_ids.astype(jnp.float32)
    freqs = pos[:, :, None] * inv_freq[None, None, :]
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # (B, T, D)
    scaling = self._attn_scaling[layer_type]
    cos = jnp.cos(emb) * scaling
    sin = jnp.sin(emb) * scaling
    return cos.astype(dtype), sin.astype(dtype)


# -----------------------------------------------------------------------------
# MLP and Attention
# -----------------------------------------------------------------------------


class Gemma4TextMLP(nnx.Module):
  """Gate + up + down projections with gelu_pytorch_tanh."""

  def __init__(
      self,
      config: Gemma4TextConfig,
      layer_idx: int,
      *,
      weights_dtype: jnp.dtype | None = None,
      compute_dtype: jnp.dtype | None = None,
      dtype: jnp.dtype = jnp.bfloat16,  # legacy alias
      rngs: nnx.Rngs,
  ):
    if weights_dtype is None:
      weights_dtype = dtype
    if compute_dtype is None:
      compute_dtype = dtype
    self.config = config
    # PORT: E4B has use_double_wide_mlp=False (confirmed). We fix
    # intermediate_size = config.intermediate_size for simplicity and skip
    # the shared-layer double-wide branch. If use_double_wide_mlp ever
    # matters, gate this on (config.use_double_wide_mlp and is_kv_shared).
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    lin_kwargs = dict(
        weights_dtype=weights_dtype, compute_dtype=compute_dtype, rngs=rngs
    )
    self.gate_proj = Linear(
        self.hidden_size, self.intermediate_size, **lin_kwargs
    )
    self.up_proj = Linear(
        self.hidden_size, self.intermediate_size, **lin_kwargs
    )
    self.down_proj = Linear(
        self.intermediate_size, self.hidden_size, **lin_kwargs
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    return self.down_proj(
        _gelu_pytorch_tanh(self.gate_proj(x)) * self.up_proj(x)
    )


def _attn_xla_sdpa(
    q: jax.Array,  # (B, Hq, T, D)
    k: jax.Array,  # (B, Hkv, T, D)
    v: jax.Array,  # (B, Hkv, T, D)
    mask: Optional[jax.Array],  # (B, 1, T, T) additive; None for plain causal
    *,
    num_key_value_groups: int,
    scaling: float,
    is_causal: bool,
) -> jax.Array:
  """Baseline SDPA via jnp.matmul — simple and portable.

  We repeat KV to match Q heads (GQA) rather than rely on a backend that
  supports it.

  Returns attn_output with shape (B, T, Hq, D) — transposed to the layout
  the caller expects (so it can reshape to (B, T, Hq*D)).
  """
  k_rep = _repeat_kv(k, num_key_value_groups)
  v_rep = _repeat_kv(v, num_key_value_groups)

  # (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
  attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k_rep) * scaling
  # Compose causal mask w/ optional additive mask. The simple path: build
  # a causal bool mask and add the additive mask if supplied.
  if is_causal:
    T_q = q.shape[-2]
    T_k = k_rep.shape[-2]
    causal = jnp.greater_equal(
        jnp.arange(T_q)[:, None], jnp.arange(T_k)[None, :]
    )  # (T_q, T_k)
    neg_inf = jnp.array(
        jnp.finfo(attn_weights.dtype).min, dtype=attn_weights.dtype
    )
    attn_weights = jnp.where(causal[None, None, :, :], attn_weights, neg_inf)
  if mask is not None:
    attn_weights = attn_weights + mask

  # upcast softmax to fp32, matching HF eager_attention_forward
  attn_weights = jax.nn.softmax(
      attn_weights.astype(jnp.float32), axis=-1
  ).astype(q.dtype)
  out = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v_rep)  # (B, H, T, D)
  # Transpose to (B, T, H, D)
  return jnp.transpose(out, (0, 2, 1, 3))


class Gemma4TextAttention(nnx.Module):
  """GQA attention with per-layer head_dim (E4B: 256 sliding, 512 full),

  q_norm / k_norm RMSNorms per-head, partial RoPE on full-attention layers
  (handled upstream by the rope's inv_freq zeros), and optional KV sharing
  for layers at index >= first_kv_shared_layer_idx.

  PORT: attention_k_eq_v=False on E4B (confirmed). Shared layers do NOT
  own k_proj/v_proj (HF drops them via _keys_to_ignore_on_load_unexpected)
  — they reuse KV from a source layer.
  """

  def __init__(
      self,
      config: Gemma4TextConfig,
      layer_idx: int,
      *,
      weights_dtype: jnp.dtype | None = None,
      compute_dtype: jnp.dtype | None = None,
      dtype: jnp.dtype = jnp.bfloat16,  # legacy alias
      rngs: nnx.Rngs,
  ):
    if weights_dtype is None:
      weights_dtype = dtype
    if compute_dtype is None:
      compute_dtype = dtype
    self.config = config
    self.layer_idx = layer_idx
    self.layer_type = config.layer_types[layer_idx]
    self.is_sliding = self.layer_type == "sliding_attention"
    self.sliding_window = config.sliding_window if self.is_sliding else None
    self.weights_dtype = weights_dtype
    self.compute_dtype = compute_dtype

    # Head dim by layer_type.
    if not self.is_sliding and config.global_head_dim:
      self.head_dim = config.global_head_dim
    else:
      self.head_dim = config.head_dim

    num_key_value_heads = config.num_key_value_heads
    self.num_kv_heads = num_key_value_heads
    self.num_heads = config.num_attention_heads
    self.num_key_value_groups = self.num_heads // num_key_value_heads
    # PORT: HF Gemma 4 sets scaling=1.0 (Q/K RMSNorm'd per-head, so
    # the usual 1/sqrt(head_dim) scaling is NOT applied). Confirmed
    # against modeling_gemma4.py line 1154.
    self.scaling = 1.0

    # KV sharing.
    first_shared = config.num_hidden_layers - getattr(
        config, "num_kv_shared_layers", 0
    )
    self.is_kv_shared_layer = layer_idx >= first_shared > 0
    prev_layers = config.layer_types[:first_shared]
    if self.is_kv_shared_layer:
      # find last non-shared layer of same type
      self.kv_shared_layer_index = (
          len(prev_layers)
          - 1
          - prev_layers[::-1].index(config.layer_types[layer_idx])
      )
      self.store_full_length_kv = False
    else:
      self.kv_shared_layer_index = None
      self.store_full_length_kv = layer_idx == len(
          prev_layers
      ) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])

    lin_kwargs = dict(
        weights_dtype=weights_dtype, compute_dtype=compute_dtype, rngs=rngs
    )
    norm_kwargs = dict(
        weights_dtype=weights_dtype, compute_dtype=compute_dtype, rngs=rngs
    )

    # Projections.
    hidden = config.hidden_size
    self.q_proj = Linear(
        hidden,
        self.num_heads * self.head_dim,
        bias=config.attention_bias,
        **lin_kwargs,
    )
    # q_norm is per-head (head_dim).
    self.q_norm = Gemma4RMSNorm(
        self.head_dim, eps=config.rms_norm_eps, **norm_kwargs
    )

    if not self.is_kv_shared_layer:
      self.k_norm = Gemma4RMSNorm(
          self.head_dim, eps=config.rms_norm_eps, **norm_kwargs
      )
      # v_norm has with_scale=False (no parameter).
      self.v_norm = Gemma4RMSNorm(
          self.head_dim,
          eps=config.rms_norm_eps,
          with_scale=False,
          **norm_kwargs,
      )
      self.k_proj = Linear(
          hidden,
          num_key_value_heads * self.head_dim,
          bias=config.attention_bias,
          **lin_kwargs,
      )
      self.v_proj = Linear(
          hidden,
          num_key_value_heads * self.head_dim,
          bias=config.attention_bias,
          **lin_kwargs,
      )
    else:
      self.k_norm = None
      self.v_norm = None
      self.k_proj = None
      self.v_proj = None

    self.o_proj = Linear(
        self.num_heads * self.head_dim,
        hidden,
        bias=config.attention_bias,
        **lin_kwargs,
    )

  def __call__(
      self,
      hidden_states: jax.Array,  # (B, T, hidden)
      position_embeddings: tuple[
          jax.Array, jax.Array
      ],  # (cos, sin) each (B, T, head_dim)
      attention_mask: Optional[jax.Array],
      shared_kv_states: dict[int, tuple[jax.Array, jax.Array]],
  ):
    B, T, _ = hidden_states.shape
    cos, sin = position_embeddings

    # Q path.
    q = self.q_proj(hidden_states)  # (B, T, Hq*D)
    q = q.reshape(B, T, self.num_heads, self.head_dim)
    q = self.q_norm(q)
    q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2)
    q = jnp.transpose(q, (0, 2, 1, 3))  # (B, Hq, T, D)

    # KV path: either from projections (own layer) or from shared_kv_states.
    if self.is_kv_shared_layer:
      k, v = shared_kv_states[self.kv_shared_layer_index]
    else:
      k = self.k_proj(hidden_states).reshape(
          B, T, self.num_kv_heads, self.head_dim
      )
      v = self.v_proj(hidden_states).reshape(
          B, T, self.num_kv_heads, self.head_dim
      )
      k = self.k_norm(k)
      k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2)
      k = jnp.transpose(k, (0, 2, 1, 3))  # (B, Hkv, T, D)
      v = self.v_norm(v)
      v = jnp.transpose(v, (0, 2, 1, 3))  # (B, Hkv, T, D)
      # Note: caller side fills shared_kv_states[layer_idx] after this
      # returns if self.store_full_length_kv.

    # Attention. Two paths:
    #   - "xla"    (default): jnp.einsum/softmax SDPA; handles arbitrary
    #              attention_mask; is_causal=True adds the triangular mask.
    #   - "splash" (exp 35)  : jax.experimental.pallas.ops.tpu.splash_attention
    #              via shard_map. No pre-kernel scaling (scaling=1.0 here
    #              matches splash's no-1/sqrt-d convention). GQA native —
    #              no _repeat_kv. Causal + sliding-window mask wired into
    #              the kernel's MaskInfo builder (LocalMask / CausalMask).
    #              Selected by env var JAX_ATTENTION_IMPL=splash.
    if os.environ.get("JAX_ATTENTION_IMPL", "xla").lower() == "splash":
      # Defer import to avoid a hard dependency for the xla path.
      from .pallas_attention import splash_attention
      # scaling is 1.0 for Gemma4 (q_norm/k_norm pre-normalize); splash
      # does not apply 1/sqrt(d) internally either, so this matches.
      attn_out = splash_attention(
          q,
          k,
          v,
          sliding_window=self.sliding_window,
      )  # (B, T, Hq, D)
    else:
      attn_out = _attn_xla_sdpa(
          q,
          k,
          v,
          attention_mask,
          num_key_value_groups=self.num_key_value_groups,
          scaling=self.scaling,
          is_causal=True,
      )  # (B, T, Hq, D)

    # o_proj over concatenated heads.
    attn_out = attn_out.reshape(B, T, self.num_heads * self.head_dim)
    attn_out = self.o_proj(attn_out)
    return attn_out, (k, v)


# -----------------------------------------------------------------------------
# Decoder Layer
# -----------------------------------------------------------------------------


class Gemma4TextDecoderLayer(nnx.Module):
  """Pre/post-norm around self-attn + MLP; optional per-layer input (PLE)

  residual.

  NB: ``layer_scalar`` is a 1-element buffer (torch ``register_buffer``) set
  to 1.0 at init. HF stores it as ``layers.{i}.layer_scalar``; we treat it
  as a learnable param to keep weight loading uniform (safetensors don't
  distinguish buffer vs param). The scalar is multiplied in at the end of
  forward.
  """

  def __init__(
      self,
      config: Gemma4TextConfig,
      layer_idx: int,
      *,
      weights_dtype: jnp.dtype | None = None,
      compute_dtype: jnp.dtype | None = None,
      dtype: jnp.dtype = jnp.bfloat16,  # legacy alias
      rngs: nnx.Rngs,
  ):
    if weights_dtype is None:
      weights_dtype = dtype
    if compute_dtype is None:
      compute_dtype = dtype
    self.config = config
    self.layer_idx = layer_idx
    self.hidden_size = config.hidden_size
    self.weights_dtype = weights_dtype
    self.compute_dtype = compute_dtype

    child_kwargs = dict(
        weights_dtype=weights_dtype, compute_dtype=compute_dtype, rngs=rngs
    )
    self.self_attn = Gemma4TextAttention(config, layer_idx, **child_kwargs)
    self.mlp = Gemma4TextMLP(config, layer_idx, **child_kwargs)

    eps = config.rms_norm_eps
    self.input_layernorm = Gemma4RMSNorm(
        self.hidden_size, eps=eps, **child_kwargs
    )
    self.post_attention_layernorm = Gemma4RMSNorm(
        self.hidden_size, eps=eps, **child_kwargs
    )
    self.pre_feedforward_layernorm = Gemma4RMSNorm(
        self.hidden_size, eps=eps, **child_kwargs
    )
    self.post_feedforward_layernorm = Gemma4RMSNorm(
        self.hidden_size, eps=eps, **child_kwargs
    )

    # layer_scalar: (1,) buffer, init 1.0. Stored in weights_dtype so it
    # tracks the fp32-master / bf16-legacy split consistently.
    self.layer_scalar = nnx.Param(jnp.ones((1,), dtype=weights_dtype))

    # Per-Layer Embeddings (PLE) residual — E4B has hidden_size_per_layer_input=256.
    self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
    if self.hidden_size_per_layer_input:
      self.per_layer_input_gate = Linear(
          self.hidden_size,
          self.hidden_size_per_layer_input,
          **child_kwargs,
      )
      self.per_layer_projection = Linear(
          self.hidden_size_per_layer_input,
          self.hidden_size,
          **child_kwargs,
      )
      self.post_per_layer_input_norm = Gemma4RMSNorm(
          self.hidden_size,
          eps=eps,
          **child_kwargs,
      )

    # MoE — skipped (E4B: enable_moe_block=False).
    assert not config.enable_moe_block, "MoE not ported (E4B is dense)."

  def __call__(
      self,
      hidden_states: jax.Array,
      per_layer_input: Optional[jax.Array],
      shared_kv_states: dict[int, tuple[jax.Array, jax.Array]],
      position_embeddings: tuple[jax.Array, jax.Array],
      attention_mask: Optional[jax.Array],
  ) -> jax.Array:
    residual = hidden_states
    x = self.input_layernorm(hidden_states)
    attn_out, kv = self.self_attn(
        x,
        position_embeddings,
        attention_mask,
        shared_kv_states,
    )
    if self.self_attn.store_full_length_kv:
      shared_kv_states[self.self_attn.layer_idx] = kv
    x = self.post_attention_layernorm(attn_out)
    hidden_states = residual + x

    # MLP block.
    residual = hidden_states
    x = self.pre_feedforward_layernorm(hidden_states)
    x = self.mlp(x)
    x = self.post_feedforward_layernorm(x)
    hidden_states = residual + x

    # PLE residual.
    if self.hidden_size_per_layer_input:
      residual = hidden_states
      x = self.per_layer_input_gate(hidden_states)
      x = _gelu_pytorch_tanh(x)
      if per_layer_input is not None:
        x = x * per_layer_input
      x = self.per_layer_projection(x)
      x = self.post_per_layer_input_norm(x)
      hidden_states = residual + x

    # Cast layer_scalar to the hidden-states dtype so mixed-precision
    # (fp32 master + bf16 compute) doesn't silently upcast the residual.
    hidden_states = hidden_states * self.layer_scalar.value.astype(
        hidden_states.dtype
    )
    return hidden_states


# -----------------------------------------------------------------------------
# Text Model
# -----------------------------------------------------------------------------


class Gemma4TextModel(nnx.Module):
  """Full text tower: embed_tokens + PLE preprocessing + N decoder layers +

  final norm. Handles the hybrid full/sliding layer pattern by dispatching
  per-layer position_embeddings and attention_mask.

  For training, we build a simple causal mask (plus an additive
  sliding-window mask for sliding layers). Splash attention would take
  over this job in a follow-up — see `_attn_xla_sdpa`.
  """

  def __init__(
      self,
      config: Gemma4TextConfig,
      *,
      weights_dtype: jnp.dtype | None = None,
      compute_dtype: jnp.dtype | None = None,
      dtype: jnp.dtype = jnp.bfloat16,  # legacy alias
      rngs: nnx.Rngs,
  ):
    if weights_dtype is None:
      weights_dtype = dtype
    if compute_dtype is None:
      compute_dtype = dtype
    self.config = config
    self.dtype = dtype  # legacy field kept for any external readers
    self.weights_dtype = weights_dtype
    self.compute_dtype = compute_dtype
    self.vocab_size = config.vocab_size
    self.num_hidden_layers = config.num_hidden_layers

    child_kwargs = dict(
        weights_dtype=weights_dtype, compute_dtype=compute_dtype, rngs=rngs
    )
    self.embed_tokens = Gemma4TextScaledWordEmbedding(
        config.vocab_size,
        config.hidden_size,
        embed_scale=config.hidden_size**0.5,
        **child_kwargs,
    )
    # One decoder layer per index. NNX requires wrapping list-of-modules
    # with nnx.data(...) so it participates in the pytree.
    self.layers = nnx.data([
        Gemma4TextDecoderLayer(config, i, **child_kwargs)
        for i in range(config.num_hidden_layers)
    ])
    self.norm = Gemma4RMSNorm(
        config.hidden_size, eps=config.rms_norm_eps, **child_kwargs
    )
    self.rotary_emb = Gemma4TextRotaryEmbedding(config)

    # PLE components. E4B has hidden_size_per_layer_input=256, enabling PLE.
    self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
    if self.hidden_size_per_layer_input:
      packed_dim = config.num_hidden_layers * config.hidden_size_per_layer_input
      self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
          config.vocab_size_per_layer_input,
          packed_dim,
          embed_scale=config.hidden_size_per_layer_input**0.5,
          **child_kwargs,
      )
      self.per_layer_input_scale = 2.0**-0.5
      self.per_layer_model_projection = Linear(
          config.hidden_size,
          packed_dim,
          **child_kwargs,
      )
      self.per_layer_model_projection_scale = config.hidden_size**-0.5
      self.per_layer_projection_norm = Gemma4RMSNorm(
          config.hidden_size_per_layer_input,
          eps=config.rms_norm_eps,
          **child_kwargs,
      )

  def _build_masks(
      self, seq_len: int, position_ids: jax.Array, dtype: jnp.dtype
  ) -> dict[str, Optional[jax.Array]]:
    """Build one additive mask per unique layer_type.

    Our baseline attention kernel already handles the causal part internally;
    for sliding_attention we add a -inf band outside the window. For
    full_attention we can return None (plain causal).
    """
    masks: dict[str, Optional[jax.Array]] = {}
    for layer_type in set(self.config.layer_types):
      if layer_type == "full_attention":
        masks[layer_type] = None
        continue
      if layer_type == "sliding_attention":
        window = self.config.sliding_window
        # mask[i, j] = 0 if (i - j) in [0, window); else -inf (the
        # causal part is applied in _attn_xla_sdpa, so this really
        # only needs to mask out positions *older* than the window).
        i = jnp.arange(seq_len)[:, None]
        j = jnp.arange(seq_len)[None, :]
        in_window = (i - j) < window
        neg_inf = jnp.array(jnp.finfo(dtype).min, dtype=dtype)
        mask = jnp.where(in_window, jnp.array(0.0, dtype=dtype), neg_inf)
        masks[layer_type] = mask[None, None, :, :]  # (1, 1, T, T)
        continue
      raise NotImplementedError(f"Unknown layer_type={layer_type}")
    return masks

  def _per_layer_inputs(self, input_ids: jax.Array) -> jax.Array:
    """Token-identity component of PLE — lookup + reshape to (B, T, L, D_ple)."""
    packed = self.embed_tokens_per_layer(input_ids)  # (B, T, L*D_ple)
    B, T = input_ids.shape
    return packed.reshape(
        B,
        T,
        self.num_hidden_layers,
        self.hidden_size_per_layer_input,
    )

  def _project_per_layer(
      self, inputs_embeds: jax.Array, per_layer_inputs_tokens: jax.Array
  ) -> jax.Array:
    """Context + identity PLE combine, matching HF project_per_layer_inputs."""
    # (B, T, L*D_ple)
    proj = (
        self.per_layer_model_projection(inputs_embeds)
        * self.per_layer_model_projection_scale
    )
    B, T, _ = inputs_embeds.shape
    proj = proj.reshape(
        B,
        T,
        self.num_hidden_layers,
        self.hidden_size_per_layer_input,
    )
    proj = self.per_layer_projection_norm(proj)
    # Combine.
    combined = (proj + per_layer_inputs_tokens) * self.per_layer_input_scale
    return combined

  def __call__(
      self,
      input_ids: jax.Array,  # (B, T) int32
      position_ids: Optional[jax.Array] = None,
  ) -> jax.Array:
    B, T = input_ids.shape
    hidden_states = self.embed_tokens(input_ids)  # (B, T, D)

    if self.hidden_size_per_layer_input:
      ple_tokens = self._per_layer_inputs(input_ids)
      per_layer_inputs = self._project_per_layer(hidden_states, ple_tokens)
    else:
      per_layer_inputs = None

    if position_ids is None:
      position_ids = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32), (B, T))

    # Precompute rope per layer_type.
    position_embeddings = {
        lt: self.rotary_emb(position_ids, lt, hidden_states.dtype)
        for lt in set(self.config.layer_types)
    }
    masks = self._build_masks(T, position_ids, hidden_states.dtype)

    # Exp 49/50: env-gated scan-over-layers dispatch. Default off => the
    # exp-36 Python for-loop below runs as baseline. Set
    # JAX_SCAN_LAYERS=1 to take the two-group scan path (compile-time
    # win + steady-state TPS near exp 36; see exp 50 writeup).
    if os.environ.get("JAX_SCAN_LAYERS") == "1":
      from .scan_layers import collect_stacked_weights, scan_layers as _scan_layers_fn

      attn_impl = os.environ.get("JAX_ATTENTION_IMPL", "xla").lower()
      first_shared = self.config.num_hidden_layers - getattr(
          self.config, "num_kv_shared_layers", 0
      )
      stacked = collect_stacked_weights(
          list(self.layers), list(self.config.layer_types), first_shared
      )
      hidden_states = _scan_layers_fn(
          stacked,
          hidden_states,
          per_layer_inputs,
          position_embeddings,
          masks,
          self.config,
          attn_impl,
      )
      hidden_states = self.norm(hidden_states)
      return hidden_states

    shared_kv_states: dict[int, tuple[jax.Array, jax.Array]] = {}
    for i, layer in enumerate(self.layers):
      layer_type = self.config.layer_types[i]
      per_layer_in = (
          per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
      )
      hidden_states = layer(
          hidden_states,
          per_layer_in,
          shared_kv_states,
          position_embeddings[layer_type],
          masks[layer_type],
      )
    hidden_states = self.norm(hidden_states)
    return hidden_states


class Gemma4ForCausalLM(nnx.Module):
  """Wraps ``Gemma4TextModel`` with a tied LM head + final logit softcapping.

  Weight tying: ``lm_head.weight`` IS ``embed_tokens.weight`` (same object).
  We implement this by NOT instantiating a separate ``lm_head`` param and
  instead doing a matmul against ``self.model.embed_tokens.weight`` at
  forward time. This avoids duplicate storage and keeps sharding coherent.
  """

  def __init__(
      self,
      config: Gemma4TextConfig,
      *,
      weights_dtype: jnp.dtype | None = None,
      compute_dtype: jnp.dtype | None = None,
      dtype: jnp.dtype = jnp.bfloat16,  # legacy alias
      rngs: nnx.Rngs,
  ):
    if weights_dtype is None:
      weights_dtype = dtype
    if compute_dtype is None:
      compute_dtype = dtype
    self.config = config
    self.dtype = dtype  # legacy
    self.weights_dtype = weights_dtype
    self.compute_dtype = compute_dtype
    self.model = Gemma4TextModel(
        config,
        weights_dtype=weights_dtype,
        compute_dtype=compute_dtype,
        rngs=rngs,
    )
    # No separate lm_head param when tie_word_embeddings=True (E4B).
    self._tied = bool(config.tie_word_embeddings)
    if not self._tied:
      self.lm_head = Linear(
          config.hidden_size,
          config.vocab_size,
          bias=False,
          weights_dtype=weights_dtype,
          compute_dtype=compute_dtype,
          rngs=rngs,
      )

  def __call__(
      self,
      input_ids: jax.Array,
      position_ids: Optional[jax.Array] = None,
      *,
      return_hidden: bool = False,
  ) -> jax.Array:
    hidden = self.model(input_ids, position_ids)  # (B, T, D)
    if return_hidden:
      # Exp 47 (levanter fused CE) consumes raw hidden + lm_head weight
      # and applies softcap inline inside the Pallas kernel, so we
      # skip the [B, T, V] logits materialization entirely here.
      return hidden
    if self._tied:
      # lm_head(x) = x @ embed_weight.T  — weight is (vocab, hidden).
      # Under fp32 master + bf16 compute: downcast weight to hidden's
      # dtype so the dot runs in compute_dtype (matches Linear path).
      weight = self.model.embed_tokens.weight.value
      if weight.dtype != hidden.dtype:
        weight = weight.astype(hidden.dtype)
      logits = hidden @ weight.T
    else:
      logits = self.lm_head(hidden)
    # Final logit softcapping.
    sc = self.config.final_logit_softcapping
    if sc is not None and sc > 0:
      sc_f = jnp.float32(sc)
      logits = sc_f * jnp.tanh(logits.astype(jnp.float32) / sc_f)
      logits = logits.astype(hidden.dtype)
    return logits

  def lm_head_weight(self) -> jax.Array:
    """Return the ``[V, H]`` lm_head / embedding weight used by the fused CE path.

    Weight is tied to ``embed_tokens`` when ``tie_word_embeddings=True``
    (Gemma 4 E4B). Callers in the fused-CE path transpose to ``[H, V]``.
    """
    if self._tied:
      return self.model.embed_tokens.weight.value
    return self.lm_head.weight.value


__all__ = [
    "Gemma4RMSNorm",
    "Gemma4TextScaledWordEmbedding",
    "Gemma4TextRotaryEmbedding",
    "Gemma4TextMLP",
    "Gemma4TextAttention",
    "Gemma4TextDecoderLayer",
    "Gemma4TextModel",
    "Gemma4ForCausalLM",
    "apply_rotary_pos_emb",
]
