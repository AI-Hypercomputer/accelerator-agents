---
title: 'Module: tpu_inference/layers/common/fused_moe_gmm.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/common/fused_moe_gmm.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.common.fused_moe_gmm`/
symbols:
  fused_moe_func: fused_moe_func().
  moe_gmm_local: moe_gmm_local().
  ragged_gather_reduce: ragged_gather_reduce.
  ragged_gather: ragged_gather.
  logger: logger.
  _apply_all_gather_fp8: _apply_all_gather_fp8().
  tensor_parallel_gmm: tensor_parallel_gmm().
  expert_parallel_gmm: expert_parallel_gmm().
  fused_moe_func._process_tokens_locally: fused_moe_func()._process_tokens_locally().
  gmm_wrapper: gmm_wrapper().
  _override_token_indices_for_random_routing: _override_token_indices_for_random_routing().
  all_gather_topk_indices_and_weights: all_gather_topk_indices_and_weights().
  _permute_tokens_for_chunked_rs: _permute_tokens_for_chunked_rs().
  apply_scoring_fn: apply_scoring_fn().
  valid_rows_mask: valid_rows_mask().
  TARGET_SLOT_CHUNK_SIZE: TARGET_SLOT_CHUNK_SIZE.
---
# Module: [`tpu_inference/layers/common/fused_moe_gmm.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/fused_moe_gmm.py)

## Functions
- `_apply_all_gather_fp8(hidden_states: jax.Array, mesh: Mesh, dtype: jnp.dtype)` — [`L536`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/fused_moe_gmm.py#L536)
- `_override_token_indices_for_random_routing(topk_indices: jax.Array, global_num_experts: int)` — [`L64`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/fused_moe_gmm.py#L64)
- `_permute_tokens_for_chunked_rs(x: jax.Array, dp_size: int, num_chunks: int)` — [`L159`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/fused_moe_gmm.py#L159) — Permutes tokens to correct for chunked reduce-scatter.
- `_process_tokens_locally(hidden_states_local, topk_indices_local)` — [`L670`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/fused_moe_gmm.py#L670)
- `all_gather_topk_indices_and_weights(topk_indices: jax.Array, topk_weights: jax.Array, dtype: jnp.dtype, mesh: Mesh)` — [`L83`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/fused_moe_gmm.py#L83)
- `apply_scoring_fn(scoring_fn: str, x: jax.Array)` — [`L110`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/fused_moe_gmm.py#L110)
- `expert_parallel_gmm(x: jax.Array, w1: jax.Array, w1_scale: jax.Array | None, w1_bias: jax.Array | None, w2: jax.Array, w2_scale: jax.Array | None, w2_bias: jax.Array | None, group_sizes: jax.Array, topk_argsort_revert_indices: jax.Array, topk_weights: jax.Array, *, activation: str, topk: int, mesh: Mesh, enable_rs_kernel: bool = False, onehot_moe_permute_threshold: int = 0, moe_chunk_size: int = 0, scatter_results: bool = False)` — [`L453`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/fused_moe_gmm.py#L453)
- `fused_moe_func(hidden_states: jax.Array, w1: jax.Array, w2: jax.Array, w1_scale: jax.Array | None, w2_scale: jax.Array | None, w1_bias: jax.Array | None, w2_bias: jax.Array | None, gating_output: jax.Array, topk: int, renormalize: bool, mesh: Mesh, use_ep: bool, activation: str, scoring_fn: str, all_gather_fp8: bool = False, enable_rs_kernel: bool = False, onehot_moe_permute_threshold: int = 0, scatter_results: bool = False, hash_based_topk_indices: jax.Array | None = None, expert_score_correction_bias: jax.Array | None = None, moe_chunk_size: int = 0)` — [`L572`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/fused_moe_gmm.py#L572) — Route tokens in hidden_states into each experts based on routing.
- `gmm_wrapper(lhs, rhs, rhs_scale, rhs_bias, group_sizes, group_offset, fuse_act=None, preferred_element_type=None)` — [`L123`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/fused_moe_gmm.py#L123)
- `moe_gmm_local(x: jax.Array, w1: jax.Array, w1_scale: jax.Array | None, w1_bias: jax.Array | None, w2: jax.Array, w2_scale: jax.Array | None, w2_bias: jax.Array | None, group_sizes: jax.Array, group_offset: jax.Array, topk_argsort_revert_indices: jax.Array, topk_weights: jax.Array, *, activation: str, topk: int, parallelism: Literal["tp", "ep"], enable_rs_kernel: bool = False, onehot_moe_permute_threshold: int = 0, scatter_results: bool = False, moe_chunk_size: int = 0)` — [`L182`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/fused_moe_gmm.py#L182) — Main MoE logic on a local shard can run in TP or EP mode.
- `tensor_parallel_gmm(x: jax.Array, w1: jax.Array, w1_scale: jax.Array | None, w1_bias: jax.Array | None, w2: jax.Array, w2_scale: jax.Array | None, w2_bias: jax.Array | None, group_sizes: jax.Array, topk_argsort_revert_indices: jax.Array, topk_weights: jax.Array, *, activation: str, topk: int, mesh: Mesh, enable_rs_kernel: bool = False, onehot_moe_permute_threshold: int = 0, scatter_results: bool = False, moe_chunk_size: int = 0)` — [`L369`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/fused_moe_gmm.py#L369)
- `valid_rows_mask(batch_size: int, group_sizes: jax.Array, group_start: jax.Array, group_end: jax.Array)` — [`L145`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/fused_moe_gmm.py#L145) — Mask indicating rows processed by current shard.

## Module values
- `TARGET_SLOT_CHUNK_SIZE` — [`L61`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/fused_moe_gmm.py#L61)
- `logger` — [`L41`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/fused_moe_gmm.py#L41)
- `ragged_gather` — [`L49`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/fused_moe_gmm.py#L49)
- `ragged_gather_reduce` — [`L53`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/fused_moe_gmm.py#L53)

