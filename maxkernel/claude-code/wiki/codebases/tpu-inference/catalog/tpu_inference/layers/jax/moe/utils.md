---
title: 'Module: tpu_inference/layers/jax/moe/utils.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/moe/utils.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.moe.utils`/
symbols:
  select_moe_backend: select_moe_backend().
  get_all_to_all_params_fn: get_all_to_all_params_fn().
  get_all_to_all_params_fn.transform_array: get_all_to_all_params_fn().transform_array().
  gmm_fn: gmm_fn().
  TransformStrategy: TransformStrategy#
  logger: logger.
  modeling_flax_utils: modeling_flax_utils.
  local_permute_fn: local_permute_fn().
  sort_activations_fn: sort_activations_fn().
  global_permute_fn: global_permute_fn().
  unpermute_fn: unpermute_fn().
  get_expert_parallelism: get_expert_parallelism().
  TransformStrategy.INPUT_OFFSET: TransformStrategy#INPUT_OFFSET.
  TransformStrategy.SEND_SIZE: TransformStrategy#SEND_SIZE.
  TransformStrategy.OUTPUT_OFFSET: TransformStrategy#OUTPUT_OFFSET.
  TransformStrategy.RECV_SIZE: TransformStrategy#RECV_SIZE.
  set_xla_metadata: set_xla_metadata.
---
# Module: [`tpu_inference/layers/jax/moe/utils.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/utils.py)

## Classes
### `TransformStrategy`  ·  implements/extends Enum
- def: [`tpu_inference/layers/jax/moe/utils.py:37`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/utils.py#L37)
- signature: `class TransformStrategy(enum.Enum):`
- members:
  - `INPUT_OFFSET` — [`L38`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/utils.py#L38)
  - `OUTPUT_OFFSET` — [`L40`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/utils.py#L40)
  - `RECV_SIZE` — [`L41`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/utils.py#L41)
  - `SEND_SIZE` — [`L39`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/utils.py#L39)
- used by: [`get_all_to_all_params_fn`](utils.md#get_all_to_all_params_fn), [`transform_array`](utils.md#get_all_to_all_params_fn.transform_array)

## Functions
- `get_all_to_all_params_fn(all_shards_group_sizes, shard_id, num_expert_parallelism, is_batch_sharded=True)` — [`L147`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/utils.py#L147) — Stateless parameter generation for ragged_all_to_all.
- `get_expert_parallelism(expert_axis_name: str, mesh: Mesh)` — [`L239`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/utils.py#L239) — Returns the expert parallelism number from the mesh.
- `global_permute_fn(inputs_TD: jax.Array, selected_experts_TX: jax.Array, num_experts_per_tok: int, num_local_experts: int)` — [`L50`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/utils.py#L50) — Stateless global permute: Sorts tokens by assigned expert.
- `gmm_fn(inputs, kernel, group_sizes, tile_size, moe_backend, dtype, quantized_dtype)` — [`L205`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/utils.py#L205) — Stateless Grouped Matrix Multiply.
- `local_permute_fn(inputs, global_group_sizes, local_expert_size, shard_index, is_offset, global_sorted_experts=None)` — [`L98`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/utils.py#L98) — Stateless local permutation logic.
- `select_moe_backend(use_ep: bool)` — [`L259`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/utils.py#L259) — Selects the MoE backend for the JAX path.
- `sort_activations_fn(inputs: jax.Array, sort_indices: jax.Array)` — [`L44`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/utils.py#L44) — Stateless sort of activations.
- `transform_array(input_array, shard_id, strategy, is_batch_sharded)` — [`L153`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/utils.py#L153)
- `unpermute_fn(processed_tokens: jax.Array, sort_indices: jax.Array, router_weights_TX: jax.Array, num_experts_per_tok: int, output_dtype)` — [`L78`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/utils.py#L78) — Stateless global unpermute logic.

## Module values
- `logger` — [`L31`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/utils.py#L31)
- `modeling_flax_utils` — [`L32`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/utils.py#L32)
- `set_xla_metadata` — [`L33`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/utils.py#L33)

