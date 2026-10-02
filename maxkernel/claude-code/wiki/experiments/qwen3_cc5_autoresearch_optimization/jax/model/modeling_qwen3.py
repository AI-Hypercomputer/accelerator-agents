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


_ATTN_NAMES = ("attn_q", "attn_k", "attn_v", "attn_out")


def _remat_policy(remat: str):
  """Map JAX_REMAT_POLICY to a jax.checkpoint policy.

  "save_attn"    — keep q/k/v/attn_out in HBM (splash never re-entered).
  "offload_attn" — same names, but offloaded to pinned host memory instead
                   of HBM (the MaxText recipe mechanism: activations cost
                   no HBM; the HOST_OFFLOAD scheduler flags overlap the
                   transfers). FFN/norms rematerialized as in save_attn.
  "selective"    — dots_with_no_batch_dims_saveable (NaNs with splash, v012).
  "full" / other — checkpoint everything.
  """
  if remat == "save_attn":
    return jax.checkpoint_policies.save_only_these_names(*_ATTN_NAMES)
  if remat == "save_qkv_ctx":
    # v044: keep q/k/v in HBM, offload the splash kernel residuals
    # ("context" = out+lse, ~4.6 GiB stacked at bs2) to pinned host --
    # the backward then consumes the saved residuals instead of
    # re-running the splash forward. Requires SPLASH_RESIDUAL_CKPT_NAME
    # =context so the kernel emits the named residuals.
    return jax.checkpoint_policies.save_and_offload_only_these_names(
        names_which_can_be_saved=["attn_q", "attn_k", "attn_v"],
        names_which_can_be_offloaded=["context"],
        offload_src="device",
        offload_dst="pinned_host",
    )
  if remat == "save_qkv_ctx_hbm":
    # Variant: keep the context residuals in HBM (may OOM at bs2).
    return jax.checkpoint_policies.save_only_these_names(
        "attn_q", "attn_k", "attn_v", "context"
    )
  if remat == "save_qkv":
    # v041 ablation: drop attn_out from the save list (it is ~6.75 GiB
    # scan-stacked at bs3/seq8192 and the backward re-runs splash fwd
    # anyway). Saves q/k/v only; attn_out + FFN/norms rematerialized.
    return jax.checkpoint_policies.save_only_these_names(
        "attn_q", "attn_k", "attn_v"
    )
  if remat == "offload_attn":
    return jax.checkpoint_policies.save_and_offload_only_these_names(
        names_which_can_be_saved=[],
        names_which_can_be_offloaded=list(_ATTN_NAMES),
        offload_src="device",
        offload_dst="pinned_host",
    )
  if remat == "selective":
    return jax.checkpoint_policies.dots_with_no_batch_dims_saveable
  return None


# Splash needs a concrete Mesh because Mosaic custom calls cannot be
# auto-partitioned; the trainer registers it once at startup via
# `set_splash_mesh` before the first forward.
_SPLASH_MESH = None


def set_splash_mesh(mesh) -> None:
  global _SPLASH_MESH
  _SPLASH_MESH = mesh


def _attn_splash(q, k, v):
  """Splash attention via the shard_map wrapper in `../splash_attn.py`.

  q: (B, Hq, T, D) — must be pre-scaled by the caller (the upstream splash
  kernel applies no 1/sqrt(head_dim) factor). k/v: (B, Hkv, T, D); splash is
  GQA-native so no `_repeat_kv`. The kernel returns (B, Hq, T, D); transpose
  to (B, T, Hq, D) to match `_attn_xla_sdpa`'s return contract.
  """
  import splash_attn

  if _SPLASH_MESH is None:
    raise RuntimeError(
        "splash kernel needs a registered mesh. Call "
        "`model.modeling_qwen3.set_splash_mesh(mesh)` once at startup."
    )
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
  )  # (B, Hq, T, D)
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
      io_layout_ok=True,
      rngs: nnx.Rngs,
  ):
    self.in_features = in_features
    self.out_features = out_features
    self.compute_dtype = compute_dtype
    key = rngs.params()
    s = 1.0 / math.sqrt(in_features)
    # v050 (env JAX_WEIGHT_LAYOUT=io): store the kernel (in, out) and
    # contract x@W without a transpose (MaxText DenseGeneral layout).
    # The SAME random draw is transposed at init, so values (and the
    # seed-0 loss series) are identical to the (out, in) layout.
    # lm_head is excluded (io_layout_ok=False): the tokamax CE custom
    # VJP consumes it directly with an (out,in)-shape contract.
    self._io_layout = (
        io_layout_ok and os.environ.get("JAX_WEIGHT_LAYOUT", "oi") == "io"
    )
    w0 = jax.random.uniform(
        key,
        (out_features, in_features),
        minval=-s,
        maxval=s,
        dtype=weights_dtype,
    )
    self.weight = nnx.Param(w0.T if self._io_layout else w0)
    self.bias = (
        nnx.Param(jnp.zeros((out_features,), dtype=weights_dtype))
        if bias
        else None
    )

  def __call__(self, x):
    w = jnp.asarray(self.weight.value, self.compute_dtype)
    x = jnp.asarray(x, self.compute_dtype)
    contract = (
        (((x.ndim - 1,), (0,)), ((), ()))
        if self._io_layout
        else (((x.ndim - 1,), (1,)), ((), ()))
    )
    out = jax.lax.dot_general(
        x, w, contract, precision=jax.lax.Precision.DEFAULT
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
    q = self.q_norm(
        self.q_proj(hidden_states).reshape(B, T, self.num_heads, self.head_dim)
    )
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = self.k_norm(
        self.k_proj(hidden_states).reshape(
            B, T, self.num_kv_heads, self.head_dim
        )
    )
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = self.v_proj(hidden_states).reshape(
        B, T, self.num_kv_heads, self.head_dim
    )
    v = jnp.transpose(v, (0, 2, 1, 3))
    q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    # Names referenced by the "save_attn" remat policy — q/k/v and the
    # kernel output are saved so the splash custom_vjp is never
    # re-entered during rematerialization (v012's NaN suspect).
    from jax.ad_checkpoint import checkpoint_name

    q = checkpoint_name(q, "attn_q")
    k = checkpoint_name(k, "attn_k")
    v = checkpoint_name(v, "attn_v")
    impl = os.environ.get("JAX_ATTENTION_IMPL", "xla").lower()
    if impl == "splash":
      attn_out = _attn_splash(q * self.scaling, k, v)
    else:
      attn_out = _attn_xla_sdpa(
          q,
          k,
          v,
          num_key_value_groups=self.num_key_value_groups,
          scaling=self.scaling,
          attention_mask=attention_mask,
      )
    attn_out = checkpoint_name(attn_out, "attn_out")
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
    # Env-gated per-layer activation checkpointing. "save_attn" saves
    # q/k/v + the attention-kernel output (splash custom_vjp never
    # re-entered — v012's per-layer dots-saveable policy produced NaN +
    # 27% overhead) and rematerializes norms + FFN intermediates (the
    # activation hog). "selective" / "full" kept for comparison.
    remat = os.environ.get("JAX_REMAT_POLICY", "").lower()
    # JAX_REMAT_SKIP_EVERY=N exempts layers with index % N == 0 from
    # checkpointing (their activations stay resident) — spends HBM
    # headroom to cut the recompute tax proportionally. 0 = remat all.
    skip_every = int(os.environ.get("JAX_REMAT_SKIP_EVERY", "0"))
    if remat:
      policy = _remat_policy(remat)
      for i, layer in enumerate(self.layers):
        if skip_every and i % skip_every == 0:
          hidden_states = layer(hidden_states, (cos, sin), attention_mask=None)
        else:
          hidden_states = jax.checkpoint(
              lambda hs, c, s, _l=layer: _l(hs, (c, s), attention_mask=None),
              policy=policy,
          )(hidden_states, cos, sin)
    else:
      for layer in self.layers:
        hidden_states = layer(hidden_states, (cos, sin), attention_mask=None)
    return self.norm(hidden_states)


class Qwen3ModelScan(nnx.Module):
  """Qwen3Model variant that stores the 36 decoder layers as ONE stacked

  module (every param carries a leading [n_layers] axis) and applies them
  with `jax.lax.scan` — O(1) compile instead of O(n_layers), one HLO body.

  Construction builds the per-layer modules with the SAME rng stream and
  order as `Qwen3Model`, then stacks their states once — so at a fixed seed
  the stacked weights equal the loop model's weights exactly (the CPU smoke
  asserts logits parity). Sharding: stacked leaves get `(None,) + <per-layer
  spec>` via the fallback in `sharding.build_plan`.

  Remat composes by checkpointing the scan body (uniform across layers —
  `JAX_REMAT_SKIP_EVERY` does not apply under scan).
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
    self.embed_tokens = Qwen3Embedding(
        config.vocab_size, config.hidden_size, **kw
    )
    layers = [
        Qwen3DecoderLayer(config, i, **kw)
        for i in range(config.num_hidden_layers)
    ]
    graphdef, _ = nnx.split(layers[0])
    states = [nnx.split(l)[1] for l in layers]
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    del layers, states
    self.layers = nnx.merge(graphdef, stacked)
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
    graphdef, state = nnx.split(self.layers)

    def body(hidden, layer_state):
      layer = nnx.merge(graphdef, layer_state)
      return layer(hidden, (cos, sin), attention_mask=None), None

    remat = os.environ.get("JAX_REMAT_POLICY", "").lower()
    if remat:
      body = jax.checkpoint(body, policy=_remat_policy(remat))
    # JAX_SCAN_UNROLL=N replicates the body N x per loop iteration. With
    # FSDP-sharded stacked weights this gives the latency-hiding scheduler
    # a window to overlap layer k+1's weight all-gather under layer k's
    # compute INSIDE one iteration -- the manual route to the AG overlap
    # that the libtpu pipeliner flags failed to produce (v030-v032).
    # Numerics-identical (pure loop restructuring). Remat granularity
    # becomes N layers per checkpoint when combined with JAX_REMAT_POLICY.
    unroll = int(os.environ.get("JAX_SCAN_UNROLL", "1"))
    hidden_states, _ = jax.lax.scan(body, hidden_states, state, unroll=unroll)
    return self.norm(hidden_states)


class Qwen3ForCausalLM(nnx.Module):
  """Qwen3 head: model + (untied) lm_head.

  `skip_lm_head=True` returns the pre-projection hidden states (for fused-CE
  paths later). Env `JAX_USE_SCAN=1` (read at construction) selects the
  scan-over-layers model body.
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
    model_cls = (
        Qwen3ModelScan
        if os.environ.get("JAX_USE_SCAN", "0") == "1"
        else Qwen3Model
    )
    self.model = model_cls(config, **kw)
    self.lm_head = Linear(
        config.hidden_size,
        config.vocab_size,
        bias=False,
        io_layout_ok=False,
        **kw,
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
