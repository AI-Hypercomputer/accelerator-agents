---
title: 'Module: tpu_inference/kernels/megablox/gmm.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/megablox/gmm.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.megablox.gmm`/
symbols:
  gmm: gmm().
  gmm.kernel._accum: gmm().kernel()._accum().
  _get_store_mask: _get_store_mask().
  make_group_metadata: make_group_metadata().
  _zero_uninitialized_memory: _zero_uninitialized_memory().
  gmm.kernel: gmm().kernel().
  partial: partial.
  GroupMetadata: GroupMetadata#
  GroupMetadata.group_offsets: GroupMetadata#group_offsets.
  _validate_args: _validate_args().
  _calculate_irregular_num_tiles: _calculate_irregular_num_tiles().
  GroupMetadata.group_ids: GroupMetadata#group_ids.
  gmm.kernel.mask_k_rem: gmm().kernel().mask_k_rem().
  gmm.kernel._accum._wrapper: gmm().kernel()._accum()._wrapper().
  _calculate_num_tiles: _calculate_num_tiles().
  GroupMetadata.m_tile_ids: GroupMetadata#m_tile_ids.
  LutFn: LutFn.
  gmm.lhs_transform_indices: gmm().lhs_transform_indices().
  gmm.rhs_transform_indices: gmm().rhs_transform_indices().
  gmm.rhs_scale_transform_indices: gmm().rhs_scale_transform_indices().
  gmm.rhs_bias_transform_indices: gmm().rhs_bias_transform_indices().
  gmm.out_transform_indices: gmm().out_transform_indices().
  gmm.kernel._zero_acc: gmm().kernel()._zero_acc().
  gmm.kernel._zero_acc._init_out: gmm().kernel()._zero_acc()._init_out().
---
# Module: [`tpu_inference/kernels/megablox/gmm.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py)

## Classes
### `GroupMetadata`
- def: [`tpu_inference/kernels/megablox/gmm.py:108`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L108)
- signature: `class GroupMetadata:`
- members:
  - `group_ids` — [`L110`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L110)
  - `group_offsets` — [`L109`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L109)
  - `m_tile_ids` — [`L111`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L111)
- used by: [`gmm`](gmm.md#gmm), [`_get_store_mask`](gmm.md#_get_store_mask), [`_zero_uninitialized_memory`](gmm.md#_zero_uninitialized_memory), [`make_group_metadata`](gmm.md#make_group_metadata)

## Functions
- `_accum(is_last_k_tile)` — [`L477`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L477)
- `_calculate_irregular_num_tiles(x: int, tx: int)` — [`L99`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L99)
- `_calculate_num_tiles(x: int, tx: int)` — [`L91`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L91)
- `_get_store_mask(*, grid_id: jnp.ndarray, group_metadata: GroupMetadata, tm: int, tn: int)` — [`L289`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L289) — Mask for rows that belong to the current group in the current tile.
- `_init_out()` — [`L465`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L465)
- `_validate_args(*, lhs: jnp.ndarray, rhs: jnp.ndarray, group_sizes: jnp.ndarray, rhs_scale: jnp.ndarray | None = None, rhs_bias: jnp.ndarray | None = None)` — [`L34`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L34) — Validates the arguments for the gmm function.
- `_wrapper(x)` — [`L483`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L483)
- `_zero_acc()` — [`L453`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L453)
- `_zero_uninitialized_memory(out: jnp.ndarray, *, start_group: jnp.ndarray, num_nonzero_groups: int, group_metadata: GroupMetadata)` — [`L305`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L305) — Zero out uninitialized memory from output.
- `gmm(lhs: jnp.ndarray, rhs: jnp.ndarray, group_sizes: jnp.ndarray, preferred_element_type: jnp.dtype = jnp.float32, rhs_scale: jnp.ndarray | None = None, rhs_bias: jnp.ndarray | None = None, tiling: tuple[int, int, int] | LutFn | None = None, group_offset: jnp.ndarray | None = None, existing_out: jnp.ndarray | None = None, interpret: bool = False)` — [`L328`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L328) — Compute lhs[sizes[i-1]:sizes[i], :] @ rhs for each group 'i'.
- `kernel(group_metadata, group_offset, lhs, rhs, rhs_scale, rhs_bias, existing_out, out, acc_scratch)` — [`L435`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L435)
- `lhs_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L530`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L530)
- `make_group_metadata(*, group_sizes: jnp.ndarray, m: int, tm: int, start_group: jnp.ndarray, num_nonzero_groups: int, visit_empty_groups: bool = True)` — [`L114`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L114) — Create the metadata needed for grouped matmul computation.
- `mask_k_rem(x, *, dim)` — [`L468`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L468)
- `out_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L559`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L559)
- `rhs_bias_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L553`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L553)
- `rhs_scale_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L546`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L546)
- `rhs_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L536`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L536)

## Module values
- `LutFn` — [`L320`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L320)
- `partial` — [`L31`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L31)

