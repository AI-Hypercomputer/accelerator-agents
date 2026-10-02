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
import os
from typing import Optional

from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
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


# Splash attention (TPU Pallas, GQA-native). The trainer registers the mesh via
# set_splash_mesh(mesh) before forward; selected by JAX_ATTENTION_IMPL=splash.
_SPLASH_MESH = None


def set_splash_mesh(mesh) -> None:
  global _SPLASH_MESH
  _SPLASH_MESH = mesh


# Activation sharding constraints, mirroring MaxText's nn.with_logical_constraint
# (activation_batch/length/embed): pin every layer-boundary activation to the FSDP
# data-parallel layout (batch sharded on fsdp, length/embed/mlp replicated) so the
# SPMD partitioner can't insert relayout copies or reshard the residual stream.
_SHARD_ACTS = False


def set_shard_acts(enabled: bool) -> None:
  global _SHARD_ACTS
  _SHARD_ACTS = bool(enabled)


def _sac(x):
  if _SHARD_ACTS and _SPLASH_MESH is not None:
    from jax.sharding import NamedSharding, PartitionSpec

    spec = PartitionSpec("fsdp", *((None,) * (x.ndim - 1)))
    return jax.lax.with_sharding_constraint(
        x, NamedSharding(_SPLASH_MESH, spec)
    )
  return x


def _attn_splash(q, k, v):
  """Dispatch into the splash kernel (sibling splash_attn.py).

  GQA-native — do NOT _repeat_kv. q/k/v are (B, H, T, Dh); returns (B, T, Hq,
  Dh).
  """
  import splash_attn  # trainer dir is on sys.path

  if _SPLASH_MESH is None:
    raise RuntimeError(
        "splash needs a mesh — call set_splash_mesh(mesh) at startup."
    )
  from jax.sharding import PartitionSpec as P

  q_sharding = P("fsdp", "tp", None, None)
  return splash_attn.tpu_splash_attention(
      _SPLASH_MESH, q_sharding, True, q, k, v, None
  )


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
    # Downcast the normalized activation to compute dtype BEFORE the weight
    # scale (matches MaxText layers/normalizations.py rms_norm), so the (B,T,D)
    # weight-multiply runs in bf16 not fp32 — halves bytes on the norm's
    # dominant elementwise and on its recompute under nothing_saveable. In fp32
    # (the equivalence test path) this is identical to the fp32-throughout form.
    y = (x32 * rsqrt).astype(in_dtype)
    return y * self.weight.value.astype(in_dtype)


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
    cn = checkpoint_name
    g = cn(self.gate_proj(x), "mlpwi")
    u = cn(self.up_proj(x), "mlpwi")
    return self.down_proj(_sac(jax.nn.silu(g) * u))


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
    cn = checkpoint_name
    # proj -> (B, T, H, Dh) -> QK-norm over Dh -> transpose -> (B, H, T, Dh)
    q = self.q_norm(
        cn(self.q_proj(hidden_states), "query_proj").reshape(
            B, T, self.num_heads, self.head_dim
        )
    )
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = self.k_norm(
        cn(self.k_proj(hidden_states), "key_proj").reshape(
            B, T, self.num_kv_heads, self.head_dim
        )
    )
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = cn(self.v_proj(hidden_states), "value_proj").reshape(
        B, T, self.num_kv_heads, self.head_dim
    )
    v = jnp.transpose(v, (0, 2, 1, 3))
    q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    if os.environ.get("JAX_ATTENTION_IMPL", "xla").lower() == "splash":
      attn_out = _attn_splash(q, k, v)  # GQA-native; no N² scores
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
    return cn(self.o_proj(attn_out), "out_proj")


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
    hidden_states = _sac(hidden_states)
    residual = hidden_states
    x = _sac(
        self.self_attn(
            _sac(self.input_layernorm(hidden_states)),
            position_embeddings,
            attention_mask,
        )
    )
    hidden_states = residual + x
    residual = hidden_states
    x = self.mlp(_sac(self.post_attention_layernorm(hidden_states)))
    return residual + x


class Qwen3Model(nnx.Module):

  def __init__(
      self,
      config,
      *,
      weights_dtype=jnp.bfloat16,
      compute_dtype=jnp.bfloat16,
      rngs,
      use_remat: bool = False,
      offload_remat: bool = False,
      use_scan: bool = False,
  ):
    self.config = config
    self.use_remat = use_remat  # per-layer jax.checkpoint
    self.offload_remat = (
        offload_remat  # offload named activations to host DRAM vs recompute
    )
    self.use_scan = (
        use_scan  # lax.scan over the stacked layer (1 compiled body)
    )
    self.num_layers = int(config.num_hidden_layers)
    kw = dict(
        weights_dtype=weights_dtype, compute_dtype=compute_dtype, rngs=rngs
    )
    self.embed_tokens = Qwen3Embedding(
        config.vocab_size, config.hidden_size, **kw
    )

    # Stacked decoder layers: ONE Qwen3DecoderLayer whose params carry a leading
    # [num_layers] axis (vmap-init with split rngs → N independent inits). This is
    # the scan-native layout — nnx.split yields stacked state directly, so lax.scan
    # iterates it with NO runtime weight copy. HF layer i ↔ stacked[i].
    @nnx.split_rngs(splits=self.num_layers)
    @nnx.vmap(in_axes=(0,), out_axes=0)
    def _make_stacked(r):
      return Qwen3DecoderLayer(
          config,
          0,
          weights_dtype=weights_dtype,
          compute_dtype=compute_dtype,
          rngs=r,
      )

    self.layers = _make_stacked(rngs)
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
    # remat policy: nothing_saveable (recompute) or offload named activations to host.
    # Offload set mirrors MaxText's qwen3-8b reference recipe: the decoder layer input
    # (residual checkpoint boundary) + attention projections; mlpwi stays recomputed.
    _policy = (
        (
            jax.checkpoint_policies.save_and_offload_only_these_names(
                names_which_can_be_saved=(),
                names_which_can_be_offloaded=(
                    "decoder_layer_input",
                    "query_proj",
                    "key_proj",
                    "value_proj",
                    "out_proj",
                ),
                offload_src="device",
                offload_dst="pinned_host",
            )
            if self.offload_remat
            else jax.checkpoint_policies.nothing_saveable
        )
        if self.use_remat
        else None
    )
    gdef, state = nnx.split(
        self.layers
    )  # state leaves carry the leading [N] layer axis
    if self.use_scan:
      # One compiled layer body, looped over the [N] axis (MaxText scan_layers).
      def _body(h, layer_state):
        h = checkpoint_name(h, "decoder_layer_input")
        return (
            nnx.merge(gdef, layer_state)(h, (cos, sin), attention_mask=None),
            None,
        )

      if self.use_remat:
        _body = jax.checkpoint(_body, policy=_policy)
      hidden_states, _ = jax.lax.scan(_body, hidden_states, state)
    else:
      # Unrolled over stacked-state slices (numerically identical to scan).
      for i in range(self.num_layers):
        layer_state = jax.tree.map(lambda x, i=i: x[i], state)

        def _call(h, _ls=layer_state):
          h = checkpoint_name(h, "decoder_layer_input")
          return nnx.merge(gdef, _ls)(h, (cos, sin), attention_mask=None)

        hidden_states = (
            jax.checkpoint(_call, policy=_policy)(hidden_states)
            if self.use_remat
            else _call(hidden_states)
        )
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
      use_remat: bool = False,
      offload_remat: bool = False,
      use_scan: bool = False,
  ):
    self.config = config
    kw = dict(
        weights_dtype=weights_dtype, compute_dtype=compute_dtype, rngs=rngs
    )
    self.model = Qwen3Model(
        config,
        use_remat=use_remat,
        offload_remat=offload_remat,
        use_scan=use_scan,
        **kw,
    )
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
