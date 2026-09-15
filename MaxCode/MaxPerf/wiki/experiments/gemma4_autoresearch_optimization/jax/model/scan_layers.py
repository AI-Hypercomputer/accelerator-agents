# pylint: skip-file
"""Scan-over-layers for Gemma 4 E4B native-JAX port (exp 49 + exp 50 tuning).

Replaces the 42-iteration Python for-loop in ``Gemma4TextModel.__call__``
with **two pairs** of nested ``jax.lax.scan``s — one pair for the 24
non-KV-shared layers, and a separate pair for the 18 KV-shared layers.

Why two separate scan groups (exp 50 fix D).
-----------------------------------------------------------------------
Gemma 4 E4B's 42 layers are structured as ``[S S S S S F] × 7`` where
S = sliding_attention (head_dim=256) and F = full_attention
(head_dim=512). The last 18 layers (indices 24-41) are
``is_kv_shared_layer=True`` — they do NOT have k_proj / v_proj / k_norm
parameters and borrow K/V from the last non-shared layer of matching
type (layer 22 for sliding, layer 23 for full).

Exp 49 used a single outer scan over all 7 super-blocks with "zero-stub"
k_proj / v_proj weights for the 18 shared layers so that the stacked
weight pytrees were homogeneous. That wasted ~35 ms/step on matmuls
whose output was always discarded via ``jnp.where(is_kv_shared, ...)``.

Exp 50 splits the scan into two groups:

  non_shared  = layers 0..23   = [S S S S S F] × 4  (24 layers)
  shared      = layers 24..41  = [S S S S S F] × 3  (18 layers)

The non-shared group computes real K/V via k_proj/v_proj and stores the
two store-layer tensors (layer 22 + layer 23 K/V) into its final carry.
Those four tensors seed the shared group's initial carry; the shared
group's body has no k_proj/v_proj matmuls at all — it just reads K/V
from the carry and runs attention + MLP + PLE. This eliminates the
"wasted matmul" regression and matches exp 36's compute exactly.

Why keep the two-level (super-block × inner) nesting.
-----------------------------------------------------------------------
Sliding and full layers have different ``head_dim`` (256 vs 512), so
their weight shapes differ and cannot be stacked together. The existing
two-level structure (outer super-block × inner sliding) already handles
that cleanly. We keep it, but narrow each outer scan to 4 or 3
super-blocks instead of 7, and drop the KV-sharing branching entirely.

Remat strategy (exp 50 fix A → B).
-----------------------------------------------------------------------
Exp 49 wrapped every scan body inner iteration in a bare
``jax.checkpoint`` to avoid a 35 GiB scan-activation OOM. That forces
uniform per-layer full-activation remat, which is much coarser than
what exp 36's XLA loop does under the outer
``jax.checkpoint(forward_loss, policy=checkpoint_dots_with_no_batch_dims)``
wrap in train.py.

Fix A (remove inner checkpoint entirely) was tried first and OOM'd at
~52 GiB — scan materializes the full activation stack across all
``n_blocks × 5`` inner iterations (each `bf16[4,5,3,1024,10240]` ~=
1.17 GiB) for the backward pass. Fix B (apply the exp-36 selective
``checkpoint_dots_with_no_batch_dims`` policy directly to the scan
body) fits and matches exp 36's remat choices on a per-layer basis.

Env gate: set ``JAX_SCAN_LAYERS=1`` to select this path.
Default off — exp 36 (Python for-loop) remains the baseline.
"""

from __future__ import annotations

import os
from typing import Optional

import jax
import jax.numpy as jnp

from .modeling_gemma4 import (
    _attn_xla_sdpa,
    _gelu_pytorch_tanh,
    apply_rotary_pos_emb,
)

# ---------------------------------------------------------------------------
# Env gate
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
  return os.environ.get("JAX_SCAN_LAYERS") == "1"


# ---------------------------------------------------------------------------
# Gemma 4 E4B structural constants (computed once from config)
# ---------------------------------------------------------------------------


def _partition_layers(
    layer_types: list[str], first_shared: int
) -> tuple[int, int, int, int]:
  """Validate the E4B pattern and return

  (n_blocks_non_shared, n_blocks_shared, block_len, sliding_per_block).

  Asserts that ``first_shared`` falls on a super-block boundary (i.e.
  layers 0..first_shared-1 form an integer number of super-blocks and
  layers first_shared..end form another integer number). For Gemma 4
  E4B this is 24 → 4 non-shared blocks + 3 shared blocks of 6 layers.
  """
  assert len(layer_types) == 42, f"expected 42 layers, got {len(layer_types)}"
  block_len = 6
  sliding_per_block = 5
  assert first_shared % block_len == 0, (
      f"first_shared ({first_shared}) must align to block_len ({block_len}); "
      "current Gemma 4 E4B has first_shared=24"
  )
  n_blocks_total = len(layer_types) // block_len
  n_blocks_non_shared = first_shared // block_len
  n_blocks_shared = n_blocks_total - n_blocks_non_shared
  # Validate the pattern: every super-block is [S,S,S,S,S,F].
  for b in range(n_blocks_total):
    for i in range(sliding_per_block):
      idx = b * block_len + i
      assert (
          layer_types[idx] == "sliding_attention"
      ), f"layer {idx} expected sliding, got {layer_types[idx]}"
    full_idx = b * block_len + sliding_per_block
    assert (
        layer_types[full_idx] == "full_attention"
    ), f"layer {full_idx} expected full, got {layer_types[full_idx]}"
  return n_blocks_non_shared, n_blocks_shared, block_len, sliding_per_block


# ---------------------------------------------------------------------------
# Weight extraction — lift NNX Param values into plain arrays
# ---------------------------------------------------------------------------


def _layer_weights_non_shared(layer) -> dict:
  """Extract all trainable arrays from a *non-KV-shared*

  Gemma4TextDecoderLayer into a flat dict-of-arrays.
  """
  out = {}
  out["input_layernorm_w"] = layer.input_layernorm.weight[...]
  out["post_attention_layernorm_w"] = layer.post_attention_layernorm.weight[...]
  out["pre_feedforward_layernorm_w"] = layer.pre_feedforward_layernorm.weight[
      ...
  ]
  out["post_feedforward_layernorm_w"] = layer.post_feedforward_layernorm.weight[
      ...
  ]
  out["layer_scalar"] = layer.layer_scalar[...]
  out["q_proj_w"] = layer.self_attn.q_proj.weight[...]
  out["q_norm_w"] = layer.self_attn.q_norm.weight[...]
  out["o_proj_w"] = layer.self_attn.o_proj.weight[...]
  assert layer.self_attn.k_proj is not None, "non-shared layer missing k_proj"
  out["k_proj_w"] = layer.self_attn.k_proj.weight[...]
  out["v_proj_w"] = layer.self_attn.v_proj.weight[...]
  out["k_norm_w"] = layer.self_attn.k_norm.weight[...]
  out["mlp_gate_w"] = layer.mlp.gate_proj.weight[...]
  out["mlp_up_w"] = layer.mlp.up_proj.weight[...]
  out["mlp_down_w"] = layer.mlp.down_proj.weight[...]
  out["ple_gate_w"] = layer.per_layer_input_gate.weight[...]
  out["ple_proj_w"] = layer.per_layer_projection.weight[...]
  out["ple_post_norm_w"] = layer.post_per_layer_input_norm.weight[...]
  return out


def _layer_weights_shared(layer) -> dict:
  """Extract trainable arrays from a *KV-shared* Gemma4TextDecoderLayer.

  Omits k_proj / v_proj / k_norm — those are None on shared layers and the
  shared-group body reads K/V from the scan carry instead.
  """
  out = {}
  out["input_layernorm_w"] = layer.input_layernorm.weight[...]
  out["post_attention_layernorm_w"] = layer.post_attention_layernorm.weight[...]
  out["pre_feedforward_layernorm_w"] = layer.pre_feedforward_layernorm.weight[
      ...
  ]
  out["post_feedforward_layernorm_w"] = layer.post_feedforward_layernorm.weight[
      ...
  ]
  out["layer_scalar"] = layer.layer_scalar[...]
  out["q_proj_w"] = layer.self_attn.q_proj.weight[...]
  out["q_norm_w"] = layer.self_attn.q_norm.weight[...]
  out["o_proj_w"] = layer.self_attn.o_proj.weight[...]
  out["mlp_gate_w"] = layer.mlp.gate_proj.weight[...]
  out["mlp_up_w"] = layer.mlp.up_proj.weight[...]
  out["mlp_down_w"] = layer.mlp.down_proj.weight[...]
  out["ple_gate_w"] = layer.per_layer_input_gate.weight[...]
  out["ple_proj_w"] = layer.per_layer_projection.weight[...]
  out["ple_post_norm_w"] = layer.post_per_layer_input_norm.weight[...]
  return out


def _stack_list_of_dicts(dicts: list[dict]) -> dict:
  """Given a list of same-keyed dicts of arrays, stack each leaf along a

  new leading axis.
  """
  keys = dicts[0].keys()
  return {k: jnp.stack([d[k] for d in dicts], axis=0) for k in keys}


def collect_stacked_weights(
    layers: list, layer_types: list[str], first_shared: int
) -> dict:
  """Stack per-group weights for the two scan groups.

  Returns::

      {
          "non_shared": {
              "sliding": <leading [n_blocks_non_shared, 5, ...] pytree>,
              "full":    <leading [n_blocks_non_shared, ...]    pytree>,
          },
          "shared": {
              "sliding": <leading [n_blocks_shared, 5, ...] pytree — no
              k/v/k_norm>,
              "full":    <leading [n_blocks_shared, ...]    pytree — no
              k/v/k_norm>,
          },
      }
  """
  n_ns, n_sh, block_len, sliding_per_block = _partition_layers(
      layer_types, first_shared
  )

  def _group(blocks_range, extractor):
    sliding_per_blocks = []  # list of blocks, each a list of 5 dicts
    full_per_blocks = []  # list of blocks, each a single dict
    for b in blocks_range:
      sliding_per_blocks.append([
          extractor(layers[b * block_len + i]) for i in range(sliding_per_block)
      ])
      full_per_blocks.append(
          extractor(layers[b * block_len + sliding_per_block])
      )
    # Stack inner (5) then outer (n_blocks).
    per_block_sliding = [_stack_list_of_dicts(x) for x in sliding_per_blocks]
    sliding_stacked = _stack_list_of_dicts(per_block_sliding)
    full_stacked = _stack_list_of_dicts(full_per_blocks)
    return {"sliding": sliding_stacked, "full": full_stacked}

  non_shared = _group(range(0, n_ns), _layer_weights_non_shared)
  shared = _group(range(n_ns, n_ns + n_sh), _layer_weights_shared)
  return {"non_shared": non_shared, "shared": shared}


# ---------------------------------------------------------------------------
# Functional layer bodies (pure JAX, no NNX)
# ---------------------------------------------------------------------------


def _rms_norm(x, weight, eps: float, with_scale: bool = True):
  """Pure-JAX RMSNorm matching Gemma4RMSNorm."""
  in_dtype = x.dtype
  x_f32 = x.astype(jnp.float32)
  mean_sq = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + eps
  normed = x_f32 * jnp.pow(mean_sq, jnp.float32(-0.5))
  if with_scale:
    normed = normed * weight.astype(jnp.float32)
  return normed.astype(in_dtype)


def _matmul_amp(x: jax.Array, w: jax.Array) -> jax.Array:
  """Helper: under fp32-master + bf16-compute AMP, downcast w to x's dtype

  so the dot runs in compute_dtype (bf16) and the output keeps that dtype.
  Matches modeling_gemma4.py's Linear forward contract.
  """
  if w.dtype != x.dtype:
    w = w.astype(x.dtype)
  return x @ w.T


def _attention_non_shared(
    hidden_states: jax.Array,  # (B, T, hidden)
    position_embeddings: tuple[jax.Array, jax.Array],
    attention_mask: Optional[jax.Array],
    *,
    weights: dict,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_key_value_groups: int,
    sliding_window: Optional[int],
    rms_eps: float,
    attn_impl: str,
):
  """Attention for a non-KV-shared layer.

  Computes Q, K, V and returns (attn_out, k, v) so the caller can carry K/V to
  later layers.
  """
  B, T, _ = hidden_states.shape
  cos, sin = position_embeddings

  q = _matmul_amp(hidden_states, weights["q_proj_w"])
  q = q.reshape(B, T, num_heads, head_dim)
  q = _rms_norm(q, weights["q_norm_w"], rms_eps, with_scale=True)
  q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2)
  q = jnp.transpose(q, (0, 2, 1, 3))

  k = _matmul_amp(hidden_states, weights["k_proj_w"])
  v = _matmul_amp(hidden_states, weights["v_proj_w"])
  k = k.reshape(B, T, num_kv_heads, head_dim)
  v = v.reshape(B, T, num_kv_heads, head_dim)
  k = _rms_norm(k, weights["k_norm_w"], rms_eps, with_scale=True)
  k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2)
  k = jnp.transpose(k, (0, 2, 1, 3))
  v = _rms_norm(v, None, rms_eps, with_scale=False)
  v = jnp.transpose(v, (0, 2, 1, 3))

  if attn_impl == "splash":
    from .pallas_attention import splash_attention

    attn_out = splash_attention(q, k, v, sliding_window=sliding_window)
  else:
    attn_out = _attn_xla_sdpa(
        q,
        k,
        v,
        attention_mask,
        num_key_value_groups=num_key_value_groups,
        scaling=1.0,
        is_causal=True,
    )

  attn_out = attn_out.reshape(B, T, num_heads * head_dim)
  attn_out = _matmul_amp(attn_out, weights["o_proj_w"])
  return attn_out, k, v


def _attention_shared(
    hidden_states: jax.Array,
    position_embeddings: tuple[jax.Array, jax.Array],
    attention_mask: Optional[jax.Array],
    *,
    weights: dict,
    borrowed_k: jax.Array,
    borrowed_v: jax.Array,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_key_value_groups: int,
    sliding_window: Optional[int],
    rms_eps: float,
    attn_impl: str,
):
  """Attention for a KV-shared layer.

  Computes only Q; K and V come from the scan carry.
  """
  B, T, _ = hidden_states.shape
  cos, sin = position_embeddings

  q = _matmul_amp(hidden_states, weights["q_proj_w"])
  q = q.reshape(B, T, num_heads, head_dim)
  q = _rms_norm(q, weights["q_norm_w"], rms_eps, with_scale=True)
  q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2)
  q = jnp.transpose(q, (0, 2, 1, 3))

  if attn_impl == "splash":
    from .pallas_attention import splash_attention

    attn_out = splash_attention(
        q, borrowed_k, borrowed_v, sliding_window=sliding_window
    )
  else:
    attn_out = _attn_xla_sdpa(
        q,
        borrowed_k,
        borrowed_v,
        attention_mask,
        num_key_value_groups=num_key_value_groups,
        scaling=1.0,
        is_causal=True,
    )

  attn_out = attn_out.reshape(B, T, num_heads * head_dim)
  attn_out = _matmul_amp(attn_out, weights["o_proj_w"])
  return attn_out


def _mlp_and_ple(
    hidden_states: jax.Array,
    residual_in: jax.Array,
    per_layer_input: Optional[jax.Array],
    *,
    weights: dict,
    rms_eps: float,
):
  """Shared MLP + PLE + final layer_scalar tail used by both layer bodies.

  ``hidden_states`` is the post-attention residual sum.
  """
  residual = hidden_states
  x = _rms_norm(hidden_states, weights["pre_feedforward_layernorm_w"], rms_eps)
  gate = _matmul_amp(x, weights["mlp_gate_w"])
  up = _matmul_amp(x, weights["mlp_up_w"])
  x = _gelu_pytorch_tanh(gate) * up
  x = _matmul_amp(x, weights["mlp_down_w"])
  x = _rms_norm(x, weights["post_feedforward_layernorm_w"], rms_eps)
  hidden_states = residual + x

  residual = hidden_states
  x = _matmul_amp(hidden_states, weights["ple_gate_w"])
  x = _gelu_pytorch_tanh(x)
  if per_layer_input is not None:
    x = x * per_layer_input
  x = _matmul_amp(x, weights["ple_proj_w"])
  x = _rms_norm(x, weights["ple_post_norm_w"], rms_eps)
  hidden_states = residual + x

  # Cast layer_scalar to the hidden-states dtype so fp32-master doesn't
  # silently upcast the residual back to fp32 (matches Gemma4TextDecoderLayer).
  hidden_states = hidden_states * weights["layer_scalar"].astype(
      hidden_states.dtype
  )
  return hidden_states


def _non_shared_layer_body(
    hidden_states: jax.Array,
    per_layer_input: Optional[jax.Array],
    position_embeddings: tuple[jax.Array, jax.Array],
    attention_mask: Optional[jax.Array],
    *,
    weights: dict,
    cfg: dict,
    attn_impl: str,
):
  """One full non-shared decoder layer.

  Returns (new_hidden, k_local, v_local).
  """
  residual = hidden_states
  x = _rms_norm(hidden_states, weights["input_layernorm_w"], cfg["rms_eps"])
  attn_out, k_local, v_local = _attention_non_shared(
      x,
      position_embeddings,
      attention_mask,
      weights=weights,
      num_heads=cfg["num_heads"],
      num_kv_heads=cfg["num_kv_heads"],
      head_dim=cfg["head_dim"],
      num_key_value_groups=cfg["num_key_value_groups"],
      sliding_window=cfg["sliding_window"],
      rms_eps=cfg["rms_eps"],
      attn_impl=attn_impl,
  )
  x = _rms_norm(attn_out, weights["post_attention_layernorm_w"], cfg["rms_eps"])
  hidden_states = residual + x
  hidden_states = _mlp_and_ple(
      hidden_states,
      residual,
      per_layer_input,
      weights=weights,
      rms_eps=cfg["rms_eps"],
  )
  return hidden_states, k_local, v_local


def _shared_layer_body(
    hidden_states: jax.Array,
    per_layer_input: Optional[jax.Array],
    position_embeddings: tuple[jax.Array, jax.Array],
    attention_mask: Optional[jax.Array],
    *,
    weights: dict,
    cfg: dict,
    borrowed_k: jax.Array,
    borrowed_v: jax.Array,
    attn_impl: str,
):
  """One full shared decoder layer.

  Returns new_hidden only (K/V are read from the scan carry, not produced here).
  """
  residual = hidden_states
  x = _rms_norm(hidden_states, weights["input_layernorm_w"], cfg["rms_eps"])
  attn_out = _attention_shared(
      x,
      position_embeddings,
      attention_mask,
      weights=weights,
      borrowed_k=borrowed_k,
      borrowed_v=borrowed_v,
      num_heads=cfg["num_heads"],
      num_kv_heads=cfg["num_kv_heads"],
      head_dim=cfg["head_dim"],
      num_key_value_groups=cfg["num_key_value_groups"],
      sliding_window=cfg["sliding_window"],
      rms_eps=cfg["rms_eps"],
      attn_impl=attn_impl,
  )
  x = _rms_norm(attn_out, weights["post_attention_layernorm_w"], cfg["rms_eps"])
  hidden_states = residual + x
  hidden_states = _mlp_and_ple(
      hidden_states,
      residual,
      per_layer_input,
      weights=weights,
      rms_eps=cfg["rms_eps"],
  )
  return hidden_states


# ---------------------------------------------------------------------------
# Scan dispatch
# ---------------------------------------------------------------------------


def scan_layers(
    stacked_weights: dict,  # {"non_shared": {...}, "shared": {...}}
    hidden_states: jax.Array,
    per_layer_inputs: Optional[jax.Array],
    position_embeddings: dict,  # {layer_type: (cos, sin)}
    masks: dict,  # {layer_type: mask_or_None}
    config,
    attn_impl: str,
) -> jax.Array:
  """Apply all 42 decoder layers via two pairs of nested scans.

  Group 1 (non-shared, layers 0..23): 4 super-blocks × (inner 5 sliding
  + inline 1 full), produces the sliding-store K/V (layer 22) and
  full-store K/V (layer 23) in its final carry.

  Group 2 (shared, layers 24..41): 3 super-blocks × (inner 5 sliding +
  inline 1 full), seeded with the stored K/V from group 1; no
  k_proj/v_proj compute.
  """
  first_shared = config.num_hidden_layers - config.num_kv_shared_layers
  n_ns, n_sh, block_len, sliding_per_block = _partition_layers(
      config.layer_types, first_shared
  )
  num_heads = config.num_attention_heads
  num_kv_heads = config.num_key_value_heads
  num_key_value_groups = num_heads // num_kv_heads
  sliding_hd = config.head_dim
  full_hd = config.global_head_dim or config.head_dim
  sliding_window = config.sliding_window
  rms_eps = config.rms_norm_eps

  B, T, _ = hidden_states.shape

  # Per-group static config dicts.
  sliding_cfg = {
      "num_heads": num_heads,
      "num_kv_heads": num_kv_heads,
      "head_dim": sliding_hd,
      "num_key_value_groups": num_key_value_groups,
      "sliding_window": sliding_window,
      "rms_eps": rms_eps,
  }
  full_cfg = {
      "num_heads": num_heads,
      "num_kv_heads": num_kv_heads,
      "head_dim": full_hd,
      "num_key_value_groups": num_key_value_groups,
      "sliding_window": None,
      "rms_eps": rms_eps,
  }
  sliding_pos = position_embeddings["sliding_attention"]
  full_pos = position_embeddings["full_attention"]
  sliding_mask = masks["sliding_attention"]
  full_mask = masks["full_attention"]

  # Selective-remat policy that matches what exp 36's outer
  # `jax.checkpoint(forward_loss, policy=...)` achieves per-layer in the
  # Python for-loop. Applied per-iteration inside scan so backward does
  # not materialize the full (n_blocks × 5) activation stack. Omitting
  # this results in ~52 GiB scan-temp OOM (measured 2026-04-24).
  from jax import checkpoint_policies as _ckpt_policies
  import functools as _functools

  _REMAT_POLICY = _ckpt_policies.checkpoint_dots_with_no_batch_dims
  _ckpt_with_policy = _functools.partial(jax.checkpoint, policy=_REMAT_POLICY)

  # Per-layer-input slicing. per_layer_inputs is (B, T, 42, D_ple).
  # Split into non-shared (layers 0..first_shared-1) and shared
  # (layers first_shared..), then reshape each into
  # (B, T, n_blocks, block_len, D_ple) → split inner vs full.
  if per_layer_inputs is not None:
    D_ple = per_layer_inputs.shape[-1]
    ple_ns = per_layer_inputs[:, :, :first_shared, :]  # (B,T,24,D)
    ple_sh = per_layer_inputs[:, :, first_shared:, :]  # (B,T,18,D)
    ple_ns = ple_ns.reshape(B, T, n_ns, block_len, D_ple)
    ple_sh = ple_sh.reshape(B, T, n_sh, block_len, D_ple)
    ple_ns_sliding = ple_ns[:, :, :, :sliding_per_block, :]  # (B,T,n_ns,5,D)
    ple_ns_full = ple_ns[:, :, :, sliding_per_block, :]  # (B,T,n_ns,D)
    ple_sh_sliding = ple_sh[:, :, :, :sliding_per_block, :]  # (B,T,n_sh,5,D)
    ple_sh_full = ple_sh[:, :, :, sliding_per_block, :]  # (B,T,n_sh,D)
  else:
    ple_ns_sliding = ple_ns_full = ple_sh_sliding = ple_sh_full = None

  # -----------------------------------------------------------------------
  # Group 1: non-shared scan over 4 super-blocks.
  # Carry: (hidden, k_sliding_store, v_sliding_store, k_full_store, v_full_store).
  # The last sliding layer per super-block overwrites k/v_sliding_store
  # (layer 4 inner, every block) — after 4 blocks the sliding store holds
  # layer 22's K/V (store = last non-shared sliding). Similarly full store
  # holds layer 23's K/V after block 3. Inside the loop, blocks 0-2 also
  # overwrite, but since only the final carry leaves the scan, we don't
  # care about intermediate store values.
  # -----------------------------------------------------------------------

  @_ckpt_with_policy
  def _ns_sliding_step(hidden, ple_i, w):
    return _non_shared_layer_body(
        hidden,
        ple_i,
        sliding_pos,
        sliding_mask,
        weights=w,
        cfg=sliding_cfg,
        attn_impl=attn_impl,
    )

  @_ckpt_with_policy
  def _ns_full_step(hidden, ple_i, w):
    return _non_shared_layer_body(
        hidden,
        ple_i,
        full_pos,
        full_mask,
        weights=w,
        cfg=full_cfg,
        attn_impl=attn_impl,
    )

  def _inner_sliding_ns_body(carry, scan_in):
    hidden, ks, vs = carry
    w = scan_in["weights"]
    ple_i = scan_in["ple"]
    new_hidden, k_local, v_local = _ns_sliding_step(hidden, ple_i, w)
    # Thread the latest sliding K/V through the carry — after all 5
    # iterations the carry holds the last layer's K/V, which for the
    # final super-block is the sliding store (layer 22). We don't emit
    # scan outputs (saves ~5x K/V memory vs collecting and slicing).
    return (new_hidden, k_local, v_local), None

  def _outer_ns_body(carry, scan_in):
    hidden, ks, vs, kf, vf = carry
    inner_w = scan_in["inner_sliding_weights"]  # (5, ...)
    full_w = scan_in["full_weights"]  # (...)
    inner_ple = scan_in["inner_ple"]  # (5, B, T, D_ple) or None
    full_ple = scan_in["full_ple"]  # (B, T, D_ple) or None

    inner_in = {"weights": inner_w, "ple": inner_ple}
    (hidden, ks_new, vs_new), _ = jax.lax.scan(
        _inner_sliding_ns_body,
        (hidden, ks, vs),
        inner_in,
    )

    # One full-attention layer per super-block (inline).
    hidden, kf_new, vf_new = _ns_full_step(hidden, full_ple, full_w)
    return (hidden, ks_new, vs_new, kf_new, vf_new), None

  # Initial carry for non-shared group.
  # Use zero placeholders for store tensors — they get overwritten every block.
  ks_init = jnp.zeros(
      (B, num_kv_heads, T, sliding_hd), dtype=hidden_states.dtype
  )
  vs_init = jnp.zeros_like(ks_init)
  kf_init = jnp.zeros((B, num_kv_heads, T, full_hd), dtype=hidden_states.dtype)
  vf_init = jnp.zeros_like(kf_init)

  outer_ns_in = {
      "inner_sliding_weights": stacked_weights["non_shared"][
          "sliding"
      ],  # (n_ns, 5, ...)
      "full_weights": stacked_weights["non_shared"]["full"],  # (n_ns, ...)
      "inner_ple": (
          # reshape (B,T,n_ns,5,D) → (n_ns, 5, B, T, D)
          jnp.transpose(ple_ns_sliding, (2, 3, 0, 1, 4))
          if ple_ns_sliding is not None
          else None
      ),
      "full_ple": (
          jnp.transpose(ple_ns_full, (2, 0, 1, 3))
          if ple_ns_full is not None
          else None
      ),
  }
  init_ns = (hidden_states, ks_init, vs_init, kf_init, vf_init)
  (hidden_states, ks_store, vs_store, kf_store, vf_store), _ = jax.lax.scan(
      _outer_ns_body,
      init_ns,
      outer_ns_in,
  )

  # -----------------------------------------------------------------------
  # Group 2: shared scan over 3 super-blocks (no k_proj/v_proj).
  # Carry is just hidden. K/V come from the per-group constants
  # (ks_store, vs_store) / (kf_store, vf_store) that were stored after
  # group 1 finished.
  # -----------------------------------------------------------------------

  @_ckpt_with_policy
  def _sh_sliding_step(hidden, ple_i, w, bk, bv):
    return _shared_layer_body(
        hidden,
        ple_i,
        sliding_pos,
        sliding_mask,
        weights=w,
        cfg=sliding_cfg,
        borrowed_k=bk,
        borrowed_v=bv,
        attn_impl=attn_impl,
    )

  @_ckpt_with_policy
  def _sh_full_step(hidden, ple_i, w, bk, bv):
    return _shared_layer_body(
        hidden,
        ple_i,
        full_pos,
        full_mask,
        weights=w,
        cfg=full_cfg,
        borrowed_k=bk,
        borrowed_v=bv,
        attn_impl=attn_impl,
    )

  def _inner_sliding_sh_body(carry, scan_in):
    hidden = carry
    w = scan_in["weights"]
    ple_i = scan_in["ple"]
    new_hidden = _sh_sliding_step(hidden, ple_i, w, ks_store, vs_store)
    return new_hidden, None

  def _outer_sh_body(carry, scan_in):
    hidden = carry
    inner_w = scan_in["inner_sliding_weights"]
    full_w = scan_in["full_weights"]
    inner_ple = scan_in["inner_ple"]
    full_ple = scan_in["full_ple"]

    inner_in = {"weights": inner_w, "ple": inner_ple}
    hidden, _ = jax.lax.scan(_inner_sliding_sh_body, hidden, inner_in)

    hidden = _sh_full_step(hidden, full_ple, full_w, kf_store, vf_store)
    return hidden, None

  outer_sh_in = {
      "inner_sliding_weights": stacked_weights["shared"][
          "sliding"
      ],  # (n_sh, 5, ...)
      "full_weights": stacked_weights["shared"]["full"],  # (n_sh, ...)
      "inner_ple": (
          jnp.transpose(ple_sh_sliding, (2, 3, 0, 1, 4))
          if ple_sh_sliding is not None
          else None
      ),
      "full_ple": (
          jnp.transpose(ple_sh_full, (2, 0, 1, 3))
          if ple_sh_full is not None
          else None
      ),
  }
  hidden_states, _ = jax.lax.scan(_outer_sh_body, hidden_states, outer_sh_in)
  return hidden_states


__all__ = ["is_enabled", "collect_stacked_weights", "scan_layers"]
