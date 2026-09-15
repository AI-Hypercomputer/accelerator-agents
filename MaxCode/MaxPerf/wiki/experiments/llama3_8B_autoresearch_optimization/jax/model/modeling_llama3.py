# pylint: skip-file
"""Native JAX (Flax NNX) port of Llama 3 8B.

Faithful port of the classes in ``transformers/models/llama/modeling_llama.py``
that make up Llama 3 8B's text tower. No multimodal / MoE / per-head norms —
the architecture is intentionally simpler than Gemma 4. Built to mirror the
sibling Gemma 4 NNX trainer's structure
(`../../gemma4_autoresearch_optimization/jax/model/modeling_gemma4.py`)
so all ops/trainer code looks familiar.

Critical Llama 3 8B facts (from `transformers.LlamaConfig` defaults +
`meta-llama/Meta-Llama-3-8B`):

  vocab_size                   128_256
  hidden_size                    4_096
  intermediate_size            14_336
  num_hidden_layers                 32
  num_attention_heads               32
  num_key_value_heads                8        # GQA 4:1
  head_dim                          128
  rope_theta                500_000.0
  rms_norm_eps                    1e-5
  hidden_act                    "silu"
  tie_word_embeddings              False     # Llama 3 has its own lm_head

Param naming matches HF dot-for-dot so the weight loader can map keys
directly:

  model.embed_tokens.weight                              (V, D)
  model.layers.{i}.self_attn.{q,k,v,o}_proj.weight       (out, in)
  model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight  (D,)
  model.layers.{i}.mlp.{gate,up,down}_proj.weight        (out, in)
  model.norm.weight                                      (D,)
  lm_head.weight                                         (V, D)

Attention dispatch is env-gated by ``JAX_ATTENTION_IMPL``:

  - "splash" (default for the trainer once the splash mesh is registered):
    routes through the `splash_attention` shard_map wrapper in
    `../splash_attn.py`. Tokamax knobs (`use_base2_exp`, `fuse_reciprocal`,
    `max_logit_const`) are taken from env vars set by the trainer.
  - "xla":   plain `jnp.einsum` SDPA with explicit `_repeat_kv` for GQA
    (used as the SDPA fallback / numerical reference).
"""

from __future__ import annotations

import math
import os
from typing import Optional

from flax import nnx
import jax
import jax.numpy as jnp
from transformers import LlamaConfig

# -----------------------------------------------------------------------------
# Stateless helpers (pure jax — not nnx.Modules)
# -----------------------------------------------------------------------------


def _rotate_half(x: jax.Array) -> jax.Array:
  """Rotate the last dim by half. Matches HF rotate_half."""
  d = x.shape[-1]
  x1 = x[..., : d // 2]
  x2 = x[..., d // 2 :]
  return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(
    q: jax.Array,
    k: jax.Array,
    cos: jax.Array,  # (B, T, D)
    sin: jax.Array,
    unsqueeze_dim: int = 1,
) -> tuple[jax.Array, jax.Array]:
  """Apply RoPE to (q, k).

  HF Llama uses ``unsqueeze_dim=1`` because q/k have layout (B, H, T, D) — i.e.
  the inserted axis is the heads axis.
  """
  cos = jnp.expand_dims(cos, unsqueeze_dim)
  sin = jnp.expand_dims(sin, unsqueeze_dim)
  return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (
      _rotate_half(k) * sin
  )


def _repeat_kv(hidden_states: jax.Array, n_rep: int) -> jax.Array:
  """GQA KV-head replication. (B, K, T, D) -> (B, K*n_rep, T, D)."""
  if n_rep == 1:
    return hidden_states
  b, k, t, d = hidden_states.shape
  hidden_states = jnp.broadcast_to(
      hidden_states[:, :, None, :, :], (b, k, n_rep, t, d)
  )
  return hidden_states.reshape(b, k * n_rep, t, d)


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------


class LlamaRMSNorm(nnx.Module):
  """RMSNorm matching HF: weight init=ones, fp32 compute, downcast at end.

  HF: ``self.weight * x * rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)``

  Mixed-precision: storage = ``weights_dtype`` (fp32 for AMP master), the
  actual normalize-and-multiply runs in fp32 and the output is cast back
  to the activation's input dtype.
  """

  def __init__(
      self,
      dim: int,
      eps: float = 1e-5,
      *,
      weights_dtype: jnp.dtype = jnp.bfloat16,
      compute_dtype: jnp.dtype = jnp.bfloat16,
      rngs: nnx.Rngs,
  ):
    del compute_dtype  # norms always run in fp32 internally
    del rngs  # not random
    self.eps = eps
    self.dim = dim
    self.weights_dtype = weights_dtype
    # Init to ones (matches HF nn.Parameter(torch.ones(dim))).
    self.weight = nnx.Param(jnp.ones((dim,), dtype=weights_dtype))

  def __call__(self, x: jax.Array) -> jax.Array:
    # MaxText-style RMSNorm: emit the full norm in one fused expression
    # so XLA can pattern-match it into a single fused region (vs our
    # earlier 5-op sequence that fragmented into 3-4 distinct fusions).
    in_dtype = x.dtype
    x32 = x.astype(jnp.float32)
    rsqrt_var = jax.lax.rsqrt(
        jnp.mean(x32 * x32, axis=-1, keepdims=True) + jnp.float32(self.eps)
    )
    return (x32 * rsqrt_var * self.weight.value.astype(jnp.float32)).astype(
        in_dtype
    )


class Linear(nnx.Module):
  """Bias-free Linear matching torch.nn.Linear convention.

  Storage shape is (out, in) so HF safetensors keys map 1:1 with no
  transpose at load time. Forward computes ``x @ weight.T``.

  Mixed-precision: weight is stored at ``weights_dtype`` (fp32 master)
  and downcast to ``compute_dtype`` (bf16) inside ``__call__`` so the
  matmul runs in compute_dtype. XLA folds the cast into the dot when it
  can. Activations coming in are expected to already be in compute_dtype;
  cast defensively so the dot sees a single dtype.
  """

  def __init__(
      self,
      in_features: int,
      out_features: int,
      *,
      bias: bool = False,
      weights_dtype: jnp.dtype = jnp.bfloat16,
      compute_dtype: jnp.dtype = jnp.bfloat16,
      rngs: nnx.Rngs,
  ):
    self.in_features = in_features
    self.out_features = out_features
    self.weights_dtype = weights_dtype
    self.compute_dtype = compute_dtype
    key = rngs.params()
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
    # MaxText-style: unconditional cast (XLA folds same-dtype away) + a
    # single dot_general with explicit contraction axes (avoids the .T
    # transpose op our older `x @ w.T` formulation emitted). This
    # collapses the per-call HLO from ~3 ops (cast, transpose, dot) to
    # 1 op (dot with cast prologue), which is the per-layer-fusion-count
    # difference observed vs MaxText (exp 79 vs maxtext baseline profile).
    w = jnp.asarray(self.weight.value, self.compute_dtype)
    x = jnp.asarray(x, self.compute_dtype)
    out = jax.lax.dot_general(
        x,
        w,
        dimension_numbers=(((x.ndim - 1,), (1,)), ((), ())),
        precision=jax.lax.Precision.DEFAULT,
    )
    if self.bias is not None:
      out = out + jnp.asarray(self.bias.value, out.dtype)
    return out


class LlamaEmbedding(nnx.Module):
  """Token embedding lookup.

  Storage in `weights_dtype`; output cast to `compute_dtype` so downstream
  attention/MLP see bf16 activations under fp32-master AMP. No scaling factor
  (Llama, unlike Gemma, does NOT multiply by sqrt(hidden_size) at the embedding
  boundary).
  """

  def __init__(
      self,
      num_embeddings: int,
      embedding_dim: int,
      *,
      weights_dtype: jnp.dtype = jnp.bfloat16,
      compute_dtype: jnp.dtype = jnp.bfloat16,
      rngs: nnx.Rngs,
  ):
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.weights_dtype = weights_dtype
    self.compute_dtype = compute_dtype
    key = rngs.params()
    # Random-normal init — real weights come from from_pretrained.
    self.weight = nnx.Param(
        jax.random.normal(
            key, (num_embeddings, embedding_dim), dtype=weights_dtype
        )
        * 0.02
    )

  def __call__(self, input_ids: jax.Array) -> jax.Array:
    out = self.weight.value[input_ids]  # (B, T, D) in weights_dtype
    if out.dtype != self.compute_dtype:
      out = out.astype(self.compute_dtype)
    return out


# -----------------------------------------------------------------------------
# RoPE
# -----------------------------------------------------------------------------


def _compute_default_inv_freq(rope_theta: float, head_dim: int) -> jax.Array:
  """HF compute_default_rope_parameters: inv_freq[i] = base^(-2i/head_dim)."""
  idxs = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
  return 1.0 / (rope_theta ** (idxs / float(head_dim)))


class LlamaRotaryEmbedding(nnx.Module):
  """Precomputes inv_freq once and builds (cos, sin) per forward pass.

  Llama 3 8B uses the "default" RoPE type with ``rope_theta=500_000`` and
  ``head_dim=128``. No partial-rotary, no scaling — much simpler than
  Gemma 4's per-layer-type machinery.
  """

  def __init__(self, config: LlamaConfig):
    self.config = config
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    # Llama config carries rope_theta either at the top level (older
    # configs) or inside `rope_parameters` dict (newer transformers).
    if hasattr(config, "rope_parameters") and config.rope_parameters:
      rope_theta = float(
          config.rope_parameters.get(
              "rope_theta", getattr(config, "rope_theta", 500_000.0)
          )
      )
    else:
      rope_theta = float(getattr(config, "rope_theta", 500_000.0))
    inv_freq = _compute_default_inv_freq(rope_theta, head_dim)
    # Wrap with nnx.data so NNX treats it as constant pytree data
    # (non-trainable; not a Param).
    self._inv_freq = nnx.data(inv_freq)
    self.attention_scaling = 1.0

  def __call__(
      self, position_ids: jax.Array, dtype: jnp.dtype
  ) -> tuple[jax.Array, jax.Array]:
    """position_ids: (B, T). Returns (cos, sin) each (B, T, head_dim)."""
    pos = position_ids.astype(jnp.float32)
    # (B, T, D/2) = (B, T, 1) * (1, 1, D/2)
    freqs = pos[:, :, None] * self._inv_freq[None, None, :]
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # (B, T, D)
    cos = jnp.cos(emb) * self.attention_scaling
    sin = jnp.sin(emb) * self.attention_scaling
    return cos.astype(dtype), sin.astype(dtype)


# -----------------------------------------------------------------------------
# MLP (SwiGLU: gate + up + down)
# -----------------------------------------------------------------------------


class LlamaMLP(nnx.Module):
  """down_proj(silu(gate_proj(x)) * up_proj(x))."""

  def __init__(
      self,
      config: LlamaConfig,
      *,
      weights_dtype: jnp.dtype = jnp.bfloat16,
      compute_dtype: jnp.dtype = jnp.bfloat16,
      rngs: nnx.Rngs,
  ):
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    bias = bool(getattr(config, "mlp_bias", False))
    lin = dict(
        weights_dtype=weights_dtype, compute_dtype=compute_dtype, rngs=rngs
    )
    self.gate_proj = Linear(
        self.hidden_size, self.intermediate_size, bias=bias, **lin
    )
    self.up_proj = Linear(
        self.hidden_size, self.intermediate_size, bias=bias, **lin
    )
    self.down_proj = Linear(
        self.intermediate_size, self.hidden_size, bias=bias, **lin
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


# -----------------------------------------------------------------------------
# Attention
# -----------------------------------------------------------------------------


def _attn_xla_sdpa(
    q: jax.Array,  # (B, Hq, T, D)
    k: jax.Array,  # (B, Hkv, T, D)
    v: jax.Array,  # (B, Hkv, T, D)
    *,
    num_key_value_groups: int,
    scaling: float,
    is_causal: bool = True,
    attention_mask: Optional[jax.Array] = None,
) -> jax.Array:
  """Plain SDPA via einsum.

  We repeat KV up to Q heads (GQA-> MHA) and apply causal masking inline.
  Returns (B, T, Hq, D) — transposed back to the layout the caller expects so it
  can reshape to (B, T, Hq*D).

  Used as the SDPA fallback / numerical reference.
  """
  k_rep = _repeat_kv(k, num_key_value_groups)
  v_rep = _repeat_kv(v, num_key_value_groups)

  attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k_rep) * scaling
  if is_causal:
    T_q = q.shape[-2]
    T_k = k_rep.shape[-2]
    causal = jnp.greater_equal(
        jnp.arange(T_q)[:, None], jnp.arange(T_k)[None, :]
    )
    neg_inf = jnp.array(
        jnp.finfo(attn_weights.dtype).min, dtype=attn_weights.dtype
    )
    attn_weights = jnp.where(causal[None, None, :, :], attn_weights, neg_inf)
  if attention_mask is not None:
    attn_weights = attn_weights + attention_mask

  # upcast softmax to fp32, matching HF eager_attention_forward
  attn_weights = jax.nn.softmax(
      attn_weights.astype(jnp.float32), axis=-1
  ).astype(q.dtype)
  out = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v_rep)  # (B, H, T, D)
  return jnp.transpose(out, (0, 2, 1, 3))  # (B, T, Hq, D)


# Module-level mesh slot the trainer fills in before forward (mirrors the
# pattern used in Gemma's pallas_attention.py). Splash needs a concrete Mesh
# because Mosaic custom calls cannot be auto-partitioned.
_SPLASH_MESH = None


def set_splash_mesh(mesh) -> None:
  """Register the device mesh that splash_attention's shard_map wraps around.

  Trainer must call this once before the first forward.
  """
  global _SPLASH_MESH
  _SPLASH_MESH = mesh


def _attn_splash(
    q: jax.Array,  # (B, Hq, T, D)
    k: jax.Array,  # (B, Hkv, T, D)
    v: jax.Array,  # (B, Hkv, T, D)
    *,
    num_key_value_groups: int,
) -> jax.Array:
  """Dispatch into the tokamax/splash kernel (sibling `splash_attn.py`).

  Splash is GQA-native (broadcasts kv heads internally), so we DON'T call
  `_repeat_kv` here. Returns (B, T, Hq, D) to match _attn_xla_sdpa's
  contract.
  """
  # Defer the import so the xla path doesn't hard-depend on the kernel.
  # `splash_attn.py` lives in the trainer's top-level dir (the parent of
  # this `model/` package). The trainer prepends that dir to sys.path
  # before any model import, so the absolute import works.
  try:
    import splash_attn  # type: ignore
  except ImportError as e:
    raise RuntimeError(
        "JAX_ATTENTION_IMPL=splash but `splash_attn.py` is not "
        "importable. Trainer should add the trainer dir to sys.path "
        "or call `set_splash_mesh(...)` from a context where the "
        "module is reachable."
    ) from e

  if _SPLASH_MESH is None:
    raise RuntimeError(
        "splash kernel needs a registered mesh. Call "
        "`model.modeling_llama3.set_splash_mesh(mesh)` once at startup."
    )

  # The kernel expects 4D tensors with kv heads. Llama 3 8B has Hq=32,
  # Hkv=8 — splash internally handles the broadcast when q has more heads
  # than k/v. Sharding pattern: replicate seq + head_dim, shard fsdp/tp on
  # the (B, H) axes. This matches the torchax sibling.
  from jax.sharding import PartitionSpec as P

  q_sharding = P("fsdp", "tp", None, None)
  return splash_attn.tpu_splash_attention(
      _SPLASH_MESH,
      q_sharding,
      True,
      q,
      k,
      v,
      None,
  )


class LlamaAttention(nnx.Module):
  """GQA attention. q_proj/k_proj/v_proj/o_proj only — no per-head norms."""

  def __init__(
      self,
      config: LlamaConfig,
      layer_idx: int,
      *,
      weights_dtype: jnp.dtype = jnp.bfloat16,
      compute_dtype: jnp.dtype = jnp.bfloat16,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.layer_idx = layer_idx
    self.weights_dtype = weights_dtype
    self.compute_dtype = compute_dtype

    self.head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    self.num_heads = config.num_attention_heads
    self.num_kv_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_kv_heads
    self.scaling = self.head_dim**-0.5

    bias = bool(getattr(config, "attention_bias", False))
    lin = dict(
        weights_dtype=weights_dtype, compute_dtype=compute_dtype, rngs=rngs
    )
    hidden = config.hidden_size
    self.q_proj = Linear(
        hidden, self.num_heads * self.head_dim, bias=bias, **lin
    )
    self.k_proj = Linear(
        hidden, self.num_kv_heads * self.head_dim, bias=bias, **lin
    )
    self.v_proj = Linear(
        hidden, self.num_kv_heads * self.head_dim, bias=bias, **lin
    )
    self.o_proj = Linear(
        self.num_heads * self.head_dim, hidden, bias=bias, **lin
    )

  def __call__(
      self,
      hidden_states: jax.Array,  # (B, T, hidden)
      position_embeddings: tuple[jax.Array, jax.Array],
      attention_mask: Optional[jax.Array] = None,
  ) -> jax.Array:
    B, T, _ = hidden_states.shape
    cos, sin = position_embeddings

    # (B, T, Hq*D) -> (B, T, Hq, D) -> (B, Hq, T, D)
    q = self.q_proj(hidden_states).reshape(B, T, self.num_heads, self.head_dim)
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = self.k_proj(hidden_states).reshape(
        B, T, self.num_kv_heads, self.head_dim
    )
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = self.v_proj(hidden_states).reshape(
        B, T, self.num_kv_heads, self.head_dim
    )
    v = jnp.transpose(v, (0, 2, 1, 3))

    # RoPE on Q and K. HF uses unsqueeze_dim=1 for the (B, H, T, D) layout.
    q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

    impl = os.environ.get("JAX_ATTENTION_IMPL", "splash").lower()
    if impl == "splash":
      attn_out = _attn_splash(
          q,
          k,
          v,
          num_key_value_groups=self.num_key_value_groups,
      )  # (B, T, Hq, D)
    else:
      attn_out = _attn_xla_sdpa(
          q,
          k,
          v,
          num_key_value_groups=self.num_key_value_groups,
          scaling=self.scaling,
          is_causal=True,
          attention_mask=attention_mask,
      )  # (B, T, Hq, D)

    attn_out = attn_out.reshape(B, T, self.num_heads * self.head_dim)
    return self.o_proj(attn_out)


# -----------------------------------------------------------------------------
# Decoder layer + Model + ForCausalLM
# -----------------------------------------------------------------------------


class LlamaDecoderLayer(nnx.Module):
  """input_layernorm -> self_attn -> +res -> post_attn_layernorm -> mlp -> +res."""

  def __init__(
      self,
      config: LlamaConfig,
      layer_idx: int,
      *,
      weights_dtype: jnp.dtype = jnp.bfloat16,
      compute_dtype: jnp.dtype = jnp.bfloat16,
      rngs: nnx.Rngs,
  ):
    self.layer_idx = layer_idx
    self.hidden_size = config.hidden_size
    kwargs = dict(
        weights_dtype=weights_dtype, compute_dtype=compute_dtype, rngs=rngs
    )
    self.self_attn = LlamaAttention(config, layer_idx, **kwargs)
    self.mlp = LlamaMLP(config, **kwargs)
    eps = config.rms_norm_eps
    self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=eps, **kwargs)
    self.post_attention_layernorm = LlamaRMSNorm(
        config.hidden_size,
        eps=eps,
        **kwargs,
    )

  def __call__(
      self,
      hidden_states: jax.Array,
      position_embeddings: tuple[jax.Array, jax.Array],
      attention_mask: Optional[jax.Array] = None,
  ) -> jax.Array:
    residual = hidden_states
    x = self.input_layernorm(hidden_states)
    x = self.self_attn(x, position_embeddings, attention_mask)
    hidden_states = residual + x

    residual = hidden_states
    x = self.post_attention_layernorm(hidden_states)
    x = self.mlp(x)
    return residual + x


class LlamaModel(nnx.Module):
  """embed_tokens + N decoder layers + final norm.

  RoPE is precomputed once at __init__ as a constant (single layer-type, simpler
  than Gemma 4).
  """

  def __init__(
      self,
      config: LlamaConfig,
      *,
      weights_dtype: jnp.dtype = jnp.bfloat16,
      compute_dtype: jnp.dtype = jnp.bfloat16,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.weights_dtype = weights_dtype
    self.compute_dtype = compute_dtype
    self.vocab_size = config.vocab_size
    self.num_hidden_layers = config.num_hidden_layers

    kwargs = dict(
        weights_dtype=weights_dtype, compute_dtype=compute_dtype, rngs=rngs
    )
    self.embed_tokens = LlamaEmbedding(
        config.vocab_size,
        config.hidden_size,
        **kwargs,
    )
    # nnx.data list participates in pytree; needed for sharding traversal.
    self.layers = nnx.data([
        LlamaDecoderLayer(config, i, **kwargs)
        for i in range(config.num_hidden_layers)
    ])
    self.norm = LlamaRMSNorm(
        config.hidden_size,
        eps=config.rms_norm_eps,
        **kwargs,
    )
    self.rotary_emb = LlamaRotaryEmbedding(config)

  def __call__(
      self,
      input_ids: jax.Array,  # (B, T) int32
      position_ids: Optional[jax.Array] = None,
  ) -> jax.Array:
    B, T = input_ids.shape
    hidden_states = self.embed_tokens(input_ids)

    if position_ids is None:
      position_ids = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32), (B, T))

    cos, sin = self.rotary_emb(position_ids, hidden_states.dtype)
    # We don't build an explicit additive attention_mask: the splash
    # kernel and XLA SDPA path both apply causal masking internally.
    for layer in self.layers:
      hidden_states = layer(hidden_states, (cos, sin), attention_mask=None)
    return self.norm(hidden_states)


class LlamaForCausalLM(nnx.Module):
  """Llama 3 8B head: model + (untied) lm_head.

  ``skip_lm_head=True`` makes forward return the pre-projection hidden states
  for fused-CE paths.
  """

  def __init__(
      self,
      config: LlamaConfig,
      *,
      weights_dtype: jnp.dtype = jnp.bfloat16,
      compute_dtype: jnp.dtype = jnp.bfloat16,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.weights_dtype = weights_dtype
    self.compute_dtype = compute_dtype
    kwargs = dict(
        weights_dtype=weights_dtype, compute_dtype=compute_dtype, rngs=rngs
    )
    self.model = LlamaModel(config, **kwargs)
    # Llama 3 has its OWN lm_head (no weight tying — config defaults
    # `tie_word_embeddings=False`).
    self.lm_head = Linear(
        config.hidden_size,
        config.vocab_size,
        bias=False,
        **kwargs,
    )
    self.skip_lm_head = False

  def __call__(
      self,
      input_ids: jax.Array,
      position_ids: Optional[jax.Array] = None,
      *,
      return_hidden: bool = False,
  ) -> jax.Array:
    hidden = self.model(input_ids, position_ids)
    if return_hidden or self.skip_lm_head:
      return hidden
    return self.lm_head(hidden)

  def lm_head_weight(self) -> jax.Array:
    """Return the [V, H] lm_head weight (used by fused-CE callers)."""
    return self.lm_head.weight.value


# -----------------------------------------------------------------------------
# Scan-over-layers
# -----------------------------------------------------------------------------
# Mirrors `../torchax/model/scan.py`: stack the 32 LlamaDecoderLayers' Params
# along a leading dim and run them via `jax.lax.scan` so XLA's compile-time
# HBM analysis sees one body's worth of buffers instead of the 32-unrolled
# sum. Confirmed by torchax exp 60/78: this is the only fitting remat at
# bs=3 seq=8192 (dots_saveable OOMs by 19-42 GiB).
#
# Implementation note: we expose the stacked weights as a parallel set of
# sub-modules (same hierarchy as a single LlamaDecoderLayer, but with each
# leaf carrying a ``(num_layers, ...)`` array). This lets the existing
# sharding traversal and weight loader walk by attribute path. At forward
# time we rebuild a "template" decoder layer's GraphDef from a freshly
# constructed module and feed it to `nnx.merge` inside the scan body.


class _ScannedLlamaAttention(nnx.Module):
  """Stacked LlamaAttention.

  Same submodule names as LlamaAttention but each leaf has a leading layer-stack
  dim.
  """

  def __init__(
      self,
      config: LlamaConfig,
      *,
      weights_dtype: jnp.dtype,
      compute_dtype: jnp.dtype,
      rngs: nnx.Rngs,
  ):
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    n = config.num_hidden_layers
    bias = bool(getattr(config, "attention_bias", False))
    hidden = config.hidden_size
    Hq, Hkv = config.num_attention_heads, config.num_key_value_heads

    def _stacked(out_dim, in_dim):
      return _StackedLinear(
          n,
          out_dim,
          in_dim,
          bias=bias,
          weights_dtype=weights_dtype,
          compute_dtype=compute_dtype,
          rngs=rngs,
      )

    self.q_proj = _stacked(Hq * head_dim, hidden)
    self.k_proj = _stacked(Hkv * head_dim, hidden)
    self.v_proj = _stacked(Hkv * head_dim, hidden)
    self.o_proj = _stacked(hidden, Hq * head_dim)


class _ScannedLlamaMLP(nnx.Module):

  def __init__(
      self,
      config: LlamaConfig,
      *,
      weights_dtype: jnp.dtype,
      compute_dtype: jnp.dtype,
      rngs: nnx.Rngs,
  ):
    n = config.num_hidden_layers
    bias = bool(getattr(config, "mlp_bias", False))
    hidden = config.hidden_size
    ffn = config.intermediate_size

    def _stacked(out_dim, in_dim):
      return _StackedLinear(
          n,
          out_dim,
          in_dim,
          bias=bias,
          weights_dtype=weights_dtype,
          compute_dtype=compute_dtype,
          rngs=rngs,
      )

    self.gate_proj = _stacked(ffn, hidden)
    self.up_proj = _stacked(ffn, hidden)
    self.down_proj = _stacked(hidden, ffn)


class _ScannedLlamaDecoderLayer(nnx.Module):
  """Stacked decoder layer. Holds N copies of the per-layer params."""

  def __init__(
      self,
      config: LlamaConfig,
      *,
      weights_dtype: jnp.dtype,
      compute_dtype: jnp.dtype,
      rngs: nnx.Rngs,
  ):
    n = config.num_hidden_layers
    self.self_attn = _ScannedLlamaAttention(
        config,
        weights_dtype=weights_dtype,
        compute_dtype=compute_dtype,
        rngs=rngs,
    )
    self.mlp = _ScannedLlamaMLP(
        config,
        weights_dtype=weights_dtype,
        compute_dtype=compute_dtype,
        rngs=rngs,
    )
    # RMSNorm weights stacked along leading dim.
    self.input_layernorm = _StackedRMSNorm(
        n,
        config.hidden_size,
        eps=config.rms_norm_eps,
        weights_dtype=weights_dtype,
    )
    self.post_attention_layernorm = _StackedRMSNorm(
        n,
        config.hidden_size,
        eps=config.rms_norm_eps,
        weights_dtype=weights_dtype,
    )


class _StackedLinear(nnx.Module):
  """Linear weight stacked along a leading layer dim."""

  def __init__(
      self,
      n_layers: int,
      out_features: int,
      in_features: int,
      *,
      bias: bool,
      weights_dtype: jnp.dtype,
      compute_dtype: jnp.dtype,
      rngs: nnx.Rngs,
  ):
    self.n_layers = n_layers
    self.in_features = in_features
    self.out_features = out_features
    self.weights_dtype = weights_dtype
    self.compute_dtype = compute_dtype
    key = rngs.params()
    init_std = 1.0 / math.sqrt(in_features)
    self.weight = nnx.Param(
        jax.random.uniform(
            key,
            (n_layers, out_features, in_features),
            minval=-init_std,
            maxval=init_std,
            dtype=weights_dtype,
        )
    )
    if bias:
      self.bias = nnx.Param(
          jnp.zeros((n_layers, out_features), dtype=weights_dtype)
      )
    else:
      self.bias = None


class _StackedRMSNorm(nnx.Module):
  """RMSNorm weight stacked along a leading layer dim."""

  def __init__(
      self,
      n_layers: int,
      dim: int,
      eps: float = 1e-5,
      *,
      weights_dtype: jnp.dtype,
  ):
    self.n_layers = n_layers
    self.dim = dim
    self.eps = eps
    self.weights_dtype = weights_dtype
    self.weight = nnx.Param(jnp.ones((n_layers, dim), dtype=weights_dtype))


def _decoder_call(
    layer_params: dict,
    hidden: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    config: LlamaConfig,
    compute_dtype: jnp.dtype,
) -> jax.Array:
  """Functional decoder-layer forward.

  ``layer_params`` is a dict of per-layer arrays (already indexed out of the
  stacked state) named like HF: ``{'input_layernorm.weight': ...,
  'self_attn.q_proj.weight': ...}``. Mirrors the body of
  LlamaDecoderLayer.__call__ exactly.
  """
  B, T, _ = hidden.shape
  head_dim = getattr(
      config, "head_dim", config.hidden_size // config.num_attention_heads
  )
  Hq = config.num_attention_heads
  Hkv = config.num_key_value_heads
  n_groups = Hq // Hkv
  scaling = head_dim**-0.5
  eps = config.rms_norm_eps

  def _rmsnorm(x, w):
    in_dtype = x.dtype
    x32 = x.astype(jnp.float32)
    rsqrt_var = jax.lax.rsqrt(
        jnp.mean(x32 * x32, axis=-1, keepdims=True) + jnp.float32(eps)
    )
    return (x32 * rsqrt_var * w.astype(jnp.float32)).astype(in_dtype)

  def _matmul(x, w):
    # Unconditional cast (XLA folds same-dtype away) + dot_general with
    # explicit contraction. Avoids the .T transpose op our older
    # `x @ w.T` formulation emitted.
    x = jnp.asarray(x, compute_dtype)
    w = jnp.asarray(w, compute_dtype)
    return jax.lax.dot_general(
        x,
        w,
        dimension_numbers=(((x.ndim - 1,), (1,)), ((), ())),
        precision=jax.lax.Precision.DEFAULT,
    )

  # MaxText-style named-checkpoint markers so a `save_only_these_names`
  # remat policy can target the QKV/O/MLP projection outputs without
  # having to reach inside the dot_general unnamed.
  _name = jax.ad_checkpoint.checkpoint_name

  residual = hidden
  x = _rmsnorm(hidden, layer_params["input_layernorm.weight"])
  # Attention.
  q = _name(_matmul(x, layer_params["self_attn.q_proj.weight"]), "query_proj")
  q = q.reshape(B, T, Hq, head_dim)
  q = jnp.transpose(q, (0, 2, 1, 3))
  k = _name(_matmul(x, layer_params["self_attn.k_proj.weight"]), "key_proj")
  k = k.reshape(B, T, Hkv, head_dim)
  k = jnp.transpose(k, (0, 2, 1, 3))
  v = _name(_matmul(x, layer_params["self_attn.v_proj.weight"]), "value_proj")
  v = v.reshape(B, T, Hkv, head_dim)
  v = jnp.transpose(v, (0, 2, 1, 3))
  q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

  impl = os.environ.get("JAX_ATTENTION_IMPL", "splash").lower()
  if impl == "splash":
    attn_out = _attn_splash(q, k, v, num_key_value_groups=n_groups)
  else:
    attn_out = _attn_xla_sdpa(
        q,
        k,
        v,
        num_key_value_groups=n_groups,
        scaling=scaling,
        is_causal=True,
        attention_mask=None,
    )
  attn_out = attn_out.reshape(B, T, Hq * head_dim)
  attn_out = _name(
      _matmul(attn_out, layer_params["self_attn.o_proj.weight"]), "out_proj"
  )
  hidden = residual + attn_out

  residual = hidden
  x = _rmsnorm(hidden, layer_params["post_attention_layernorm.weight"])
  # SwiGLU MLP.
  g = _name(_matmul(x, layer_params["mlp.gate_proj.weight"]), "mlpwi_0")
  u = _name(_matmul(x, layer_params["mlp.up_proj.weight"]), "mlpwi_1")
  h = _name(
      _matmul(jax.nn.silu(g) * u, layer_params["mlp.down_proj.weight"]),
      "mlpwo",
  )
  return residual + h


# Path strings under `_ScannedLlamaDecoderLayer` whose stacked param feeds
# `_decoder_call`. Order: leading dim is the layer index.
_SCANNED_PARAM_PATHS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def _gather_scanned_state(scanned_layer: _ScannedLlamaDecoderLayer) -> dict:
  """Walk `scanned_layer` to a dict {path: stacked_array} the scan body

  can indexes layer-by-layer.
  """
  out = {}
  for path in _SCANNED_PARAM_PATHS:
    node: object = scanned_layer
    for part in path.split("."):
      node = getattr(node, part)
    # node is the final nnx.Param after walking attribute chain.
    out[path] = node.value
  return out


class LlamaForCausalLMScan(nnx.Module):
  """Drop-in replacement for ``LlamaForCausalLM`` with scan-over-layers.

  Param-tree layout (each leaf carries a (num_layers, *original) leading
  dim):

    model.embed_tokens.weight                                       (V, H)
    model.scanned_layers.input_layernorm.weight                     (N, H)
    model.scanned_layers.post_attention_layernorm.weight            (N, H)
    model.scanned_layers.self_attn.{q,k,v,o}_proj.weight            (N, ...)
    model.scanned_layers.mlp.{gate,up,down}_proj.weight             (N, ...)
    model.norm.weight                                               (H,)
    lm_head.weight                                                  (V, H)

  Sharding plan in `model/sharding.py` (`SCAN_SHARDING_PLAN`) prepends a
  leading ``None`` to every per-layer PartitionSpec so the stacked layer
  dim stays unsharded.
  """

  def __init__(
      self,
      config: LlamaConfig,
      *,
      weights_dtype: jnp.dtype = jnp.bfloat16,
      compute_dtype: jnp.dtype = jnp.bfloat16,
      scan_remat_policy=None,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.weights_dtype = weights_dtype
    self.compute_dtype = compute_dtype
    self.scan_remat_policy = scan_remat_policy
    kwargs = dict(
        weights_dtype=weights_dtype, compute_dtype=compute_dtype, rngs=rngs
    )

    # The non-scanned outer module is a "thin" LlamaModel with only
    # embed_tokens, norm, and rotary_emb; the layer stack is held by
    # `scanned_layers` instead of `layers`.
    self.model = _LlamaModelScanShell(config, **kwargs)
    self.lm_head = Linear(
        config.hidden_size,
        config.vocab_size,
        bias=False,
        **kwargs,
    )
    self.skip_lm_head = False

  def __call__(
      self,
      input_ids: jax.Array,
      position_ids: Optional[jax.Array] = None,
      *,
      return_hidden: bool = False,
  ) -> jax.Array:
    B, T = input_ids.shape
    hidden = self.model.embed_tokens(input_ids)
    if position_ids is None:
      position_ids = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32), (B, T))
    cos, sin = self.model.rotary_emb(position_ids, hidden.dtype)

    stacked = _gather_scanned_state(self.model.scanned_layers)
    cfg = self.config
    cdt = self.compute_dtype

    # Optional pre-cast of all stacked weights to compute_dtype once
    # (outside the scan loop) so the scan body reads bf16 from HBM
    # directly. With AMP master fp32 weights this lets one fp32->bf16
    # cast happen once per train_step instead of being implicit at
    # each of the 32 scan iterations. Risk: doubles weight HBM peak
    # while both copies are alive; gated by env-var to A/B test.
    if os.environ.get("JAX_PRECAST_BF16_WEIGHTS", "0") == "1":
      stacked = {k: v.astype(cdt) for k, v in stacked.items()}

    def body(carry_hidden, per_layer):
      new_hidden = _decoder_call(
          per_layer,
          carry_hidden,
          cos,
          sin,
          config=cfg,
          compute_dtype=cdt,
      )
      return new_hidden, None

    if self.scan_remat_policy is not None:
      body = jax.checkpoint(body, policy=self.scan_remat_policy)

    unroll = int(os.environ.get("JAX_SCAN_UNROLL", "1"))
    hidden, _ = jax.lax.scan(body, hidden, stacked, unroll=unroll)
    hidden = self.model.norm(hidden)
    if return_hidden or self.skip_lm_head:
      return hidden
    return self.lm_head(hidden)

  def lm_head_weight(self) -> jax.Array:
    return self.lm_head.weight.value


class _LlamaModelScanShell(nnx.Module):
  """Outer model module for the scan path.

  Holds only embed_tokens, norm, rotary_emb, and `scanned_layers` (the stacked
  decoder layer).

  Param paths under this module match HF's ``model.<...>`` prefix, with
  the stacked layer params reachable via
  ``scanned_layers.{input_layernorm,post_attention_layernorm,self_attn,mlp}.<...>``.
  """

  def __init__(
      self,
      config: LlamaConfig,
      *,
      weights_dtype: jnp.dtype,
      compute_dtype: jnp.dtype,
      rngs: nnx.Rngs,
  ):
    self.config = config
    kwargs = dict(
        weights_dtype=weights_dtype, compute_dtype=compute_dtype, rngs=rngs
    )
    self.embed_tokens = LlamaEmbedding(
        config.vocab_size,
        config.hidden_size,
        **kwargs,
    )
    self.scanned_layers = _ScannedLlamaDecoderLayer(
        config,
        weights_dtype=weights_dtype,
        compute_dtype=compute_dtype,
        rngs=rngs,
    )
    self.norm = LlamaRMSNorm(
        config.hidden_size,
        eps=config.rms_norm_eps,
        **kwargs,
    )
    self.rotary_emb = LlamaRotaryEmbedding(config)


__all__ = [
    "LlamaRMSNorm",
    "LlamaEmbedding",
    "LlamaRotaryEmbedding",
    "LlamaMLP",
    "LlamaAttention",
    "LlamaDecoderLayer",
    "LlamaModel",
    "LlamaForCausalLM",
    "LlamaForCausalLMScan",
    "apply_rotary_pos_emb",
    "set_splash_mesh",
]
