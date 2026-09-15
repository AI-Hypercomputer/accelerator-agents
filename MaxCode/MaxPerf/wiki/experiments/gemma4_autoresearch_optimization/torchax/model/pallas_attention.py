# pylint: skip-file
"""Pallas `splash_attention` wiring for Gemma 4 under torchax.

Registers a HuggingFace `ALL_ATTENTION_FUNCTIONS` entry named `"splash_pallas"`
that routes Gemma4's attention call through
`jax.experimental.pallas.ops.tpu.splash_attention` via
`torchax.interop.call_jax`. Replaces the default XLA SDPA path (which
materializes the full `[B, n_heads, S, S]` score tensor) with a tiled
TPU-native Pallas kernel.

References:
  - `wiki/experiments/gemma4_autoresearch_optimization/program.md` — Pallas
    kernel landscape; what's allowed to change.
  - `wiki/codebases/torchax.md` — `interop.call_jax`, `jax_view`, `torch_view`.
  - `wiki/codebases/tokamax.md` +
  `wiki/sources/2026-tokamax-splash-attention.md`
    — splash kernel surface and GQA native support.
  - `wiki/concepts/splash-attention.md`,
  `wiki/concepts/attention-block-sizes.md`.
  - Upstream example:
  `raw/code/torchax/examples/train_llama_torchtitan/splash_attn.py`.
  - Splash source: `jax.experimental.pallas.ops.tpu.splash_attention`
    (`splash_attention_kernel.py`, `splash_attention_mask.py`).

Gemma 4 architecture assumptions (program.md):
  - `num_attention_heads = 8`, `num_key_value_heads = 2` (GQA 4:1).
  - `head_dim = 256`.
  - Per-layer dispatch: the `sliding_window` kwarg is `None` on full-attention
    layers and `512` on sliding-attention layers. RoPE, q_norm, k_norm,
    partial-rotary, KV sharing are all handled inside `Gemma4TextAttention`
    BEFORE this attention interface is called — we only see post-rotated,
    post-normed Q/K/V.

Kernel call shape (from `splash_attention_kernel._splash_attention_forward`):
  Per-example (after `vmap` over batch):
    q : [num_q_heads, seq_len, head_dim]
    k : [num_kv_heads, seq_len, head_dim]
    v : [num_kv_heads, seq_len, head_dim]
  HF-interface input layout is [B, n_heads, S, head_dim] for Q and
  [B, n_kv_heads, S, head_dim] for K/V, so a plain `jax.vmap` over axis 0
  matches.

Mask-builder convention (verified against
`splash_attention_mask.make_local_attention_mask`):
    LocalMask(shape, window_size=(left, right), offset=...)
    -> attends to  q_idx - left <= kv_idx <= q_idx + right.
  For Gemma4 causal + sliding-window-512 training we want "attend to the
  past W tokens plus current", which is  window_size=(W, 0).
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch

# Imports that pull in JAX / Pallas are deferred inside functions so this
# module can be imported in environments without a TPU present.


# ---------------------------------------------------------------------------
# Module-level mesh handle
# ---------------------------------------------------------------------------
#
# The splash kernel is a Mosaic / Pallas custom-call and JAX refuses to
# auto-partition it across a sharded jit context (`NotImplementedError: Mosaic
# kernels cannot be automatically partitioned. Please wrap the call in a
# shard_map.`). We therefore wrap the kernel body in `jax.shard_map` with
# explicit in/out PartitionSpecs. shard_map needs a concrete `Mesh` (not an
# `AbstractMesh`), so `train.py` passes the mesh through `register_splash_attention`
# and we stash it here.
_MESH: "Optional[Any]" = None


# ---------------------------------------------------------------------------
# JAX-side splash kernel factory (cached by (seq_len, n_q_heads, sliding_window))
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _build_splash_kernel(
    seq_len: int, num_q_heads: int, sliding_window: int | None
):
  """Build a splash-attention kernel specialized for a (S, H_q, W) combo.

  Cached because `make_splash_mha_single_device` precomputes a `MaskInfo`
  structure from the mask — we don't want to rebuild it every forward.
  """
  from jax.experimental.pallas.ops.tpu.splash_attention import (
      splash_attention_kernel as sa_kernel,
      splash_attention_mask as sa_mask,
  )

  # Block sizes: heuristic default from tokamax's TPU backend is 128 across
  # the board. For Gemma4 at seq_len=1024 we bump fwd blocks to 512 and use
  # per-seq-len caps (see `pallas_mosaic_tpu.py` for the autotune search
  # space). Under-defaults are safe; over-defaults just reduce occupancy.
  # `seq_len >= 1024 => block >= 1024` is a tokamax pruning rule but the
  # upstream kernel also accepts 512 — we pick 512 to leave room for future
  # autotune over {128, 256, 512, 1024}.
  block_q = min(1024, seq_len)
  block_kv = min(1024, seq_len)
  block_kv_compute = min(1024, seq_len)
  # Exp 17 — enable the fused backward kernel. `use_fused_bwd_kernel=True`
  # requires the dQ-path block sizes (`block_q_dq`, `block_kv_dq`) to be
  # OMITTED (splash raises `ValueError: Block sizes for dq kernel are
  # not needed with a fused kernel` if they're set, which broke exp 16
  # and triggered our XLA fallback for every layer).
  # Exp 24 — switch Q layout to SEQ_MINOR. tokamax's pallas_mosaic_tpu
  # exposes q_layout/k_layout/v_layout as autotune knobs; HEAD_DIM_MINOR
  # is the default but SEQ_MINOR can improve HBM locality when the batch
  # dimension is being streamed.
  block_sizes = sa_kernel.BlockSizes(
      block_q=block_q,
      block_kv=block_kv,
      block_kv_compute=block_kv_compute,
      # Backward-pass tiles for dKV. The fused path still uses these.
      block_q_dkv=min(1024, seq_len),
      block_kv_dkv=min(1024, seq_len),
      block_kv_dkv_compute=min(1024, seq_len),
      # block_q_dq / block_kv_dq intentionally omitted — see note above.
      use_fused_bwd_kernel=True,
      q_layout=sa_kernel.QKVLayout.SEQ_MINOR,
      k_layout=sa_kernel.QKVLayout.SEQ_MINOR,
      v_layout=sa_kernel.QKVLayout.SEQ_MINOR,
  )

  # Per-head mask. MHA mode requires exactly `num_q_heads` masks (splash
  # reshapes to (kv_heads, q_heads_per_kv_head, ...) internally for GQA —
  # see `splash_attention_kernel.py` lines 422-428).
  if sliding_window is not None and sliding_window > 0:
    # Causal sliding-window: attend to the last `sliding_window` tokens
    # including self, which is window_size=(W, 0) in LocalMask's
    # (left, right) convention.
    head_mask = sa_mask.LocalMask(
        shape=(seq_len, seq_len),
        window_size=(int(sliding_window), 0),
        offset=0,
    )
  else:
    head_mask = sa_mask.CausalMask(shape=(seq_len, seq_len))

  multi_head_mask = sa_mask.MultiHeadMask(masks=(head_mask,) * num_q_heads)

  # `make_splash_mha_single_device` returns a `SplashAttentionKernel`
  # callable that expects per-example q,k,v (rank-3). We `vmap` over the
  # batch axis below.
  kernel = sa_kernel.make_splash_mha_single_device(
      mask=multi_head_mask,
      block_sizes=block_sizes,
      attn_logits_soft_cap=None,
  )
  return kernel


def _jax_splash_fwd(
    q, k, v, seq_len: int, num_q_heads: int, sliding_window: int | None
):
  """JAX-side forward. q,k,v already in JAX-land with leading batch axis.

  Wrapped in `jax.shard_map` so the Mosaic custom-call sees per-chip local
  tensors (batch B/fsdp) rather than a globally-sharded array — JAX refuses
  to auto-partition Mosaic kernels. All four tensors (Q, K, V, out) are
  batch-sharded on the `'fsdp'` axis; heads / seq / head_dim are replicated.
  """
  import jax
  from jax.sharding import PartitionSpec as P

  kernel = _build_splash_kernel(seq_len, num_q_heads, sliding_window)

  # Per-chip body: vmap over the (per-chip) batch axis. segment_ids=None
  # because training packs a single document per row; the causal/local mask
  # carries all the structure.
  def _inner(qj, kj, vj):
    return jax.vmap(kernel, in_axes=(0, 0, 0))(qj, kj, vj)

  if _MESH is None:
    # No mesh registered — caller didn't go through register_splash_attention(mesh).
    # Fall back to plain vmap (single-device runs, unit tests).
    return _inner(q, k, v)

  in_specs = (
      P("fsdp", None, None, None),  # Q: [B, H, S, D]
      P("fsdp", None, None, None),  # K: [B, Hkv, S, D]
      P("fsdp", None, None, None),  # V: [B, Hkv, S, D]
  )
  out_specs = P("fsdp", None, None, None)  # [B, H, S, D]

  sharded_inner = jax.shard_map(
      _inner,
      mesh=_MESH,
      in_specs=in_specs,
      out_specs=out_specs,
      check_vma=False,
  )
  return sharded_inner(q, k, v)


def _xla_fallback_fwd(q, k, v, scaling: float, sliding_window: int | None):
  """Fallback path — plain XLA dot-product attention.

  Used only if the splash kernel raises (e.g., unsupported `head_dim`). Kept
  simple: full materialization of attention logits, GQA handled by broadcasting.
  """
  import jax
  import jax.numpy as jnp

  # q: [B, Hq, S, D], k,v: [B, Hkv, S, D]
  b, hq, s, d = q.shape
  hkv = k.shape[1]
  if hq != hkv:
    # Broadcast K/V from Hkv -> Hq by repeat
    rep = hq // hkv
    k = jnp.repeat(k, rep, axis=1)
    v = jnp.repeat(v, rep, axis=1)
  logits = jnp.einsum("bhsd,bhtd->bhst", q, k) * scaling
  # Causal + optional sliding-window mask.
  q_idx = jnp.arange(s)[:, None]
  kv_idx = jnp.arange(s)[None, :]
  mask = kv_idx <= q_idx
  if sliding_window is not None and sliding_window > 0:
    mask = mask & (kv_idx >= q_idx - int(sliding_window))
  neg = jnp.finfo(logits.dtype).min
  logits = jnp.where(mask[None, None, :, :], logits, neg)
  probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(
      logits.dtype
  )
  out = jnp.einsum("bhst,bhtd->bhsd", probs, v)
  return out


# ---------------------------------------------------------------------------
# HuggingFace attention-interface function
# ---------------------------------------------------------------------------


def splash_attention_fn(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
  """Drop-in replacement for SDPA in Gemma 4.

  Inputs (torch tensors in torchax-land):
      query_states: [B, num_q_heads=8, S, head_dim=256]
      key_states:   [B, num_kv_heads=2, S, head_dim=256]
      value_states: [B, num_kv_heads=2, S, head_dim=256]

  Returns:
      attn_output: [B, S, num_q_heads, head_dim] (the transpose the reshape
                   on line 1255 of modeling_gemma4.py expects).
      attn_weights: None (training doesn't need them).
  """
  from torchax import interop

  if dropout != 0.0:
    raise NotImplementedError(
        f"splash_attention_fn does not support dropout > 0 (got {dropout})."
    )

  # HF's `attention_mask` is redundant for our training setup: Gemma4 layer
  # dispatch already tells us the sliding_window via the `sliding_window`
  # kwarg, and we pack docs without padding. If a padded-batch caller ever
  # shows up, this assumption needs `SegmentIds` wiring.
  _ = attention_mask  # intentionally ignored; see docstring.

  if scaling is None:
    scaling = float(module.head_dim) ** -0.5

  # Apply scaling on Q before the kernel — the splash kernel multiplies
  # q @ k^T directly without an internal 1/sqrt(d) factor, so we fold the
  # scale into Q. This also makes bf16 numerics match the XLA-SDPA path.
  q = query_states * scaling
  k = key_states
  v = value_states

  b, num_q_heads, seq_len, head_dim = q.shape

  def _jax_fn(qj, kj, vj):
    # Primary: splash pallas. Fallback: plain XLA if the shape is
    # unsupported by the kernel (e.g., head_dim=256 is fine on v5+/v6e but
    # we fall through gracefully just in case).
    try:
      return _jax_splash_fwd(qj, kj, vj, seq_len, num_q_heads, sliding_window)
    except Exception as e:  # pragma: no cover - defensive
      print(f"[splash_pallas] fallback to XLA: {type(e).__name__}: {e}")
      return _xla_fallback_fwd(qj, kj, vj, 1.0, sliding_window)

  # call_jax: torch_view <- jax_fn(jax_view(...))
  # Returns shape [B, num_q_heads, S, head_dim].
  out = interop.call_jax(_jax_fn, q, k, v)

  # HF's `attn_output.reshape(*input_shape, -1)` at line 1255 expects
  # [B, S, num_q_heads, head_dim]. Transpose 1<->2 to land there. This
  # matches the behavior of eager_attention_forward (which does
  # `.transpose(1, 2).contiguous()` at the end).
  out = out.transpose(1, 2).contiguous()
  return out, None


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

ATTN_IMPL_NAME = "splash_pallas"


def register_splash_attention(mesh: "Any") -> str:
  """Register `splash_attention_fn` under ALL_ATTENTION_FUNCTIONS.

  Args:
      mesh: the active ``jax.sharding.Mesh`` (must carry an ``'fsdp'`` axis).
        Stashed in a module global so the per-layer attention call can wrap the
        Pallas kernel in ``jax.shard_map``. Required — Mosaic custom-calls
        cannot be auto-partitioned, so we must shard_map them with a concrete
        mesh.

  Returns:
      The implementation key to set on ``config._attn_implementation``.
  """
  global _MESH
  _MESH = mesh
  from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

  ALL_ATTENTION_FUNCTIONS[ATTN_IMPL_NAME] = splash_attention_fn
  return ATTN_IMPL_NAME
