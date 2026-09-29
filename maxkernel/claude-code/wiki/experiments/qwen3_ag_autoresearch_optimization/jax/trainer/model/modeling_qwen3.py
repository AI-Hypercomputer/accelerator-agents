"""Native JAX (Flax NNX) port of Qwen3 (dense, e.g. Qwen3-8B).

Faithful port of `transformers/models/qwen3/modeling_qwen3.py`'s text tower.
Qwen3 ≈ Llama3 (GQA + SwiGLU + RMSNorm + RoPE, untied lm_head) with ONE
structural addition — **QK-norm**: a per-head RMSNorm over `head_dim` applied to
Q and K *after* the q/k projection+reshape and *before* RoPE. Qwen3 also drops
the QKV bias Qwen2 carried.

Param naming matches HF dot-for-dot so the weight loader maps keys directly:

  model.embed_tokens.weight                                   (V, D)
  model.layers.{i}.self_attn.{q,k,v,o}_proj.weight            (out, in)
  model.layers.{i}.self_attn.{q,k}_norm.weight                (head_dim,)   <-
  Qwen3
  model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight  (D,)
  model.layers.{i}.mlp.{gate,up,down}_proj.weight             (out, in)
  model.norm.weight                                           (D,)
  lm_head.weight                                              (V, D)

This is the MINIMAL trainer-less model: attention is plain XLA SDPA
(`jnp.einsum`),
so it runs on CPU and is numerically comparable to HF eager attention. Splash
attention + scan-over-layers are later optimizations (they attach when the jax
trainer is built), kept out of here so the baseline stays simple and
CPU-runnable.
"""

from __future__ import annotations

import math
from typing import Optional

from flax import nnx
import jax
import jax.numpy as jnp
from transformers import Qwen3Config

# -----------------------------------------------------------------------------
# Stateless helpers (pure jax)
# -----------------------------------------------------------------------------


def _rotate_half(x: jax.Array) -> jax.Array:
  d = x.shape[-1]
  return jnp.concatenate((-x[..., d // 2 :], x[..., : d // 2]), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
  """HF apply_rotary_pos_emb. q/k are (B, H, T, Dh); cos/sin are (B, T, Dh)."""
  cos = jnp.expand_dims(cos, unsqueeze_dim)
  sin = jnp.expand_dims(sin, unsqueeze_dim)
  return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (
      _rotate_half(k) * sin
  )


def _repeat_kv(x: jax.Array, n_rep: int) -> jax.Array:
  if n_rep == 1:
    return x
  b, k, t, d = x.shape
  return jnp.broadcast_to(x[:, :, None, :, :], (b, k, n_rep, t, d)).reshape(
      b, k * n_rep, t, d
  )


def _attn_xla_sdpa(
    q, k, v, *, num_key_value_groups, scaling, attention_mask=None
):
  """Plain causal SDPA via einsum (GQA via _repeat_kv).

  fp32 softmax (matches HF eager_attention_forward). Returns (B, T, Hq, Dh).
  """
  k_rep = _repeat_kv(k, num_key_value_groups)
  v_rep = _repeat_kv(v, num_key_value_groups)
  aw = jnp.einsum("bhqd,bhkd->bhqk", q, k_rep) * scaling
  Tq, Tk = q.shape[-2], k_rep.shape[-2]
  causal = jnp.arange(Tq)[:, None] >= jnp.arange(Tk)[None, :]
  neg_inf = jnp.array(jnp.finfo(aw.dtype).min, dtype=aw.dtype)
  aw = jnp.where(causal[None, None, :, :], aw, neg_inf)
  if attention_mask is not None:
    aw = aw + attention_mask
  aw = jax.nn.softmax(aw.astype(jnp.float32), axis=-1).astype(q.dtype)
  out = jnp.einsum("bhqk,bhkd->bhqd", aw, v_rep)  # (B, H, T, Dh)
  return jnp.transpose(out, (0, 2, 1, 3))  # (B, T, Hq, Dh)


# -----------------------------------------------------------------------------
# Splash Attention Integration
# -----------------------------------------------------------------------------

import os

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
        "`model.modeling_qwen3.set_splash_mesh(mesh)` once at startup."
    )

  # The kernel expects 4D tensors with kv heads. Qwen3 has Hq>Hkv.
  # Splash internally handles the broadcast when q has more heads
  # than k/v. Sharding pattern: replicate seq + head_dim, shard fsdp/tp on
  # the (B, H) axes. Layout is (B, H, T, D).
  from jax.sharding import PartitionSpec as P

  q_sharding = P("fsdp", "tp", None, None)
  out = splash_attn.tpu_splash_attention(
      _SPLASH_MESH,
      q_sharding,
      True,
      q,
      k,
      v,
      None,
  )
  return jnp.transpose(out, (0, 2, 1, 3))


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------


class Qwen3RMSNorm(nnx.Module):
  """RMSNorm: fp32 normalize, weight*..., downcast to input dtype.

  weight init = ones (matches HF). Used both for the (D,) layer norms and the
  (head_dim,) QK-norms — same op, different `dim`.
  """

  def __init__(
      self,
      dim,
      eps=1e-6,
      *,
      weights_dtype=jnp.bfloat16,
      compute_dtype=jnp.bfloat16,
      rngs: nnx.Rngs = None,
  ):
    del compute_dtype, rngs
    self.eps = eps
    self.dim = dim
    self.weight = nnx.Param(jnp.ones((dim,), dtype=weights_dtype))

  def __call__(self, x):
    in_dtype = x.dtype
    x32 = x.astype(jnp.float32)
    rsqrt = jax.lax.rsqrt(
        jnp.mean(x32 * x32, axis=-1, keepdims=True) + jnp.float32(self.eps)
    )
    return (x32 * rsqrt * self.weight.value.astype(jnp.float32)).astype(
        in_dtype
    )


class Linear(nnx.Module):
  """Bias-free Linear, storage (out, in) so HF keys map 1:1. y = x @ w.T."""

  def __init__(
      self,
      in_features,
      out_features,
      *,
      bias=False,
      weights_dtype=jnp.bfloat16,
      compute_dtype=jnp.bfloat16,
      rngs: nnx.Rngs,
  ):
    self.in_features = in_features
    self.out_features = out_features
    self.compute_dtype = compute_dtype
    key = rngs.params()
    s = 1.0 / math.sqrt(in_features)
    self.weight = nnx.Param(
        jax.random.uniform(
            key,
            (out_features, in_features),
            minval=-s,
            maxval=s,
            dtype=weights_dtype,
        )
    )
    self.bias = (
        nnx.Param(jnp.zeros((out_features,), dtype=weights_dtype))
        if bias
        else None
    )

  def __call__(self, x):
    w = jnp.asarray(self.weight.value, self.compute_dtype)
    x = jnp.asarray(x, self.compute_dtype)
    out = jax.lax.dot_general(
        x,
        w,
        (((x.ndim - 1,), (1,)), ((), ())),
        precision=jax.lax.Precision.DEFAULT,
    )
    if self.bias is not None:
      out = out + jnp.asarray(self.bias.value, out.dtype)
    return out


class Qwen3Embedding(nnx.Module):

  def __init__(
      self,
      num_embeddings,
      embedding_dim,
      *,
      weights_dtype=jnp.bfloat16,
      compute_dtype=jnp.bfloat16,
      rngs: nnx.Rngs,
  ):
    self.compute_dtype = compute_dtype
    key = rngs.params()
    self.weight = nnx.Param(
        jax.random.normal(
            key, (num_embeddings, embedding_dim), dtype=weights_dtype
        )
        * 0.02
    )

  def __call__(self, input_ids):
    out = self.weight.value[input_ids]
    return (
        out.astype(self.compute_dtype)
        if out.dtype != self.compute_dtype
        else out
    )


def _compute_default_inv_freq(rope_theta, head_dim):
  idxs = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
  return 1.0 / (rope_theta ** (idxs / float(head_dim)))


class Qwen3RotaryEmbedding(nnx.Module):

  def __init__(self, config: Qwen3Config):
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    rope_theta = float(getattr(config, "rope_theta", 1_000_000.0))
    self._inv_freq = nnx.data(_compute_default_inv_freq(rope_theta, head_dim))
    self.attention_scaling = 1.0

  def __call__(self, position_ids, dtype):
    pos = position_ids.astype(jnp.float32)
    freqs = pos[:, :, None] * self._inv_freq[None, None, :]
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return (jnp.cos(emb) * self.attention_scaling).astype(dtype), (
        jnp.sin(emb) * self.attention_scaling
    ).astype(dtype)


class Qwen3MLP(nnx.Module):

  def __init__(
      self,
      config,
      *,
      weights_dtype=jnp.bfloat16,
      compute_dtype=jnp.bfloat16,
      rngs,
  ):
    bias = bool(getattr(config, "mlp_bias", False))
    lin = dict(
        weights_dtype=weights_dtype, compute_dtype=compute_dtype, rngs=rngs
    )
    self.gate_proj = Linear(
        config.hidden_size, config.intermediate_size, bias=bias, **lin
    )
    self.up_proj = Linear(
        config.hidden_size, config.intermediate_size, bias=bias, **lin
    )
    self.down_proj = Linear(
        config.intermediate_size, config.hidden_size, bias=bias, **lin
    )

  def __call__(self, x):
    return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Attention(nnx.Module):
  """GQA attention WITH Qwen3 QK-norm (RMSNorm over head_dim on Q and K,

  applied after the proj+reshape and before RoPE).
  """

  def __init__(
      self,
      config,
      layer_idx,
      *,
      weights_dtype=jnp.bfloat16,
      compute_dtype=jnp.bfloat16,
      rngs,
  ):
    self.head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    self.num_heads = config.num_attention_heads
    self.num_kv_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_kv_heads
    self.scaling = self.head_dim**-0.5
    hidden = config.hidden_size
    bias = bool(getattr(config, "attention_bias", False))
    lin = dict(
        weights_dtype=weights_dtype, compute_dtype=compute_dtype, rngs=rngs
    )
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
    # Qwen3 QK-norm — RMSNorm over head_dim.
    eps = config.rms_norm_eps
    self.q_norm = Qwen3RMSNorm(
        self.head_dim, eps=eps, weights_dtype=weights_dtype
    )
    self.k_norm = Qwen3RMSNorm(
        self.head_dim, eps=eps, weights_dtype=weights_dtype
    )

  def __call__(self, hidden_states, position_embeddings, attention_mask=None):
    B, T, _ = hidden_states.shape
    cos, sin = position_embeddings
    # proj -> (B, T, H, Dh) -> QK-norm over Dh -> transpose -> (B, H, T, Dh)
    q = self.q_proj(hidden_states).reshape(B, T, self.num_heads, self.head_dim)
    k = self.k_proj(hidden_states).reshape(
        B, T, self.num_kv_heads, self.head_dim
    )
    v = self.v_proj(hidden_states).reshape(
        B, T, self.num_kv_heads, self.head_dim
    )
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    from .fused_rope import fused_qknorm_rope

    q = fused_qknorm_rope(
        q, self.q_norm.weight.value, cos, sin, eps=self.q_norm.eps
    )
    k = fused_qknorm_rope(
        k, self.k_norm.weight.value, cos, sin, eps=self.k_norm.eps
    )

    impl = os.environ.get("JAX_ATTENTION_IMPL", "xla").lower()
    if impl == "splash":
      attn_out = _attn_splash(
          q,
          k,
          v,
          num_key_value_groups=self.num_key_value_groups,
      )
    else:
      attn_out = _attn_xla_sdpa(
          q,
          k,
          v,
          num_key_value_groups=self.num_key_value_groups,
          scaling=self.scaling,
          attention_mask=attention_mask,
      )
    attn_out = attn_out.reshape(B, T, self.num_heads * self.head_dim)
    return self.o_proj(attn_out)


class Qwen3DecoderLayer(nnx.Module):

  def __init__(
      self,
      config,
      layer_idx,
      *,
      weights_dtype=jnp.bfloat16,
      compute_dtype=jnp.bfloat16,
      rngs,
  ):
    kw = dict(
        weights_dtype=weights_dtype, compute_dtype=compute_dtype, rngs=rngs
    )
    self.self_attn = Qwen3Attention(config, layer_idx, **kw)
    self.mlp = Qwen3MLP(config, **kw)
    eps = config.rms_norm_eps
    self.input_layernorm = Qwen3RMSNorm(
        config.hidden_size, eps=eps, weights_dtype=weights_dtype
    )
    self.post_attention_layernorm = Qwen3RMSNorm(
        config.hidden_size, eps=eps, weights_dtype=weights_dtype
    )

  def __call__(self, hidden_states, position_embeddings, attention_mask=None):
    if not hasattr(self, "_ckpt_call"):
      import jax

      self._ckpt_call = jax.checkpoint(self._call_impl)
    return self._ckpt_call(hidden_states, position_embeddings, attention_mask)

  def _call_impl(self, hidden_states, position_embeddings, attention_mask=None):
    residual = hidden_states
    x = self.self_attn(
        self.input_layernorm(hidden_states), position_embeddings, attention_mask
    )
    hidden_states = residual + x
    residual = hidden_states
    x = self.mlp(self.post_attention_layernorm(hidden_states))
    return residual + x


class Qwen3Model(nnx.Module):

  def __init__(
      self,
      config,
      *,
      weights_dtype=jnp.bfloat16,
      compute_dtype=jnp.bfloat16,
      rngs,
  ):
    self.config = config
    kw = dict(
        weights_dtype=weights_dtype, compute_dtype=compute_dtype, rngs=rngs
    )
    self.embed_tokens = Qwen3Embedding(
        config.vocab_size, config.hidden_size, **kw
    )
    self.layers = nnx.data([
        Qwen3DecoderLayer(config, i, **kw)
        for i in range(config.num_hidden_layers)
    ])
    self.norm = Qwen3RMSNorm(
        config.hidden_size, eps=config.rms_norm_eps, weights_dtype=weights_dtype
    )
    self.rotary_emb = Qwen3RotaryEmbedding(config)

  def __call__(self, input_ids, position_ids=None):
    B, T = input_ids.shape
    hidden_states = self.embed_tokens(input_ids)
    if position_ids is None:
      position_ids = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32), (B, T))
    cos, sin = self.rotary_emb(position_ids, hidden_states.dtype)
    for layer in self.layers:
      hidden_states = layer(hidden_states, (cos, sin), attention_mask=None)
    return self.norm(hidden_states)


class Qwen3ForCausalLM(nnx.Module):
  """Qwen3 head: model + (untied) lm_head.

  `skip_lm_head=True` returns the pre-projection hidden states (for fused-CE
  paths later).
  """

  def __init__(
      self,
      config,
      *,
      weights_dtype=jnp.bfloat16,
      compute_dtype=jnp.bfloat16,
      rngs,
  ):
    self.config = config
    kw = dict(
        weights_dtype=weights_dtype, compute_dtype=compute_dtype, rngs=rngs
    )
    self.model = Qwen3Model(config, **kw)
    self.lm_head = Linear(
        config.hidden_size, config.vocab_size, bias=False, **kw
    )
    self.skip_lm_head = False

  def __call__(self, input_ids, position_ids=None, *, return_hidden=False):
    hidden = self.model(input_ids, position_ids)
    if return_hidden or self.skip_lm_head:
      return hidden
    return self.lm_head(hidden)

  def lm_head_weight(self):
    return self.lm_head.weight.value


__all__ = [
    "Qwen3RMSNorm",
    "Qwen3Embedding",
    "Qwen3RotaryEmbedding",
    "Qwen3MLP",
    "Qwen3Attention",
    "Qwen3DecoderLayer",
    "Qwen3Model",
    "Qwen3ForCausalLM",
    "apply_rotary_pos_emb",
    "set_splash_mesh",
]
