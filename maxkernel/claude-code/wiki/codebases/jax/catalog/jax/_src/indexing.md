---
title: 'Module: jax/_src/indexing.py'
type: catalog
provenance: extracted
module: jax/_src/indexing.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.indexing`/
symbols:
  ds: ds.
  Slice: Slice#
  Slice.start: Slice#start.
  Slice.size: Slice#size.
  Slice.stride: Slice#stride.
  dslice: dslice().
  Slice.is_dynamic_start: Slice#is_dynamic_start().
  Slice.tree_flatten: Slice#tree_flatten().
  Slice.is_dynamic_size: Slice#is_dynamic_size().
  Slice.from_slice: Slice#from_slice().
  Slice.__post_init__: Slice#__post_init__().
  Slice.tree_unflatten: Slice#tree_unflatten().
---
# Module: [`jax/_src/indexing.py`](../../../../../../raw/code/jax/jax/_src/indexing.py)

## Classes
### `Slice`
- def: [`jax/_src/indexing.py:26`](../../../../../../raw/code/jax/jax/_src/indexing.py#L26)
- doc: A slice with a start index and a size.
- signature: `class Slice:`
- members:
  - `from_slice(cls, slc: slice, size: int)` — [`L68`](../../../../../../raw/code/jax/jax/_src/indexing.py#L68)
  - `is_dynamic_size(self)` — [`L46`](../../../../../../raw/code/jax/jax/_src/indexing.py#L46)
  - `is_dynamic_start(self)` — [`L42`](../../../../../../raw/code/jax/jax/_src/indexing.py#L42)
  - `tree_flatten(self)` — [`L49`](../../../../../../raw/code/jax/jax/_src/indexing.py#L49)
  - `tree_unflatten(cls, aux_data, children)` — [`L61`](../../../../../../raw/code/jax/jax/_src/indexing.py#L61)
  - `size` — [`L34`](../../../../../../raw/code/jax/jax/_src/indexing.py#L34)
  - `start` — [`L33`](../../../../../../raw/code/jax/jax/_src/indexing.py#L33)
  - `stride` — [`L35`](../../../../../../raw/code/jax/jax/_src/indexing.py#L35)
- protocol/private: `__post_init__`[`L37`](../../../../../../raw/code/jax/jax/_src/indexing.py#L37)
- uses (calls/refs, reference-scoped): [`Array`](basearray.md#Array), [`is_dim`](core.md#is_dim), [`register_pytree_node_class`](tree_util.md#register_pytree_node_class), [`canonicalize_slice`](core.md#canonicalize_slice)
- used by: [`_index_to_gather`](numpy/indexing.md#_index_to_gather), [`to_block_mapping`](pallas/core.md#BlockSpec.to_block_mapping), [`to_dynamic_slice`](numpy/indexing.md#NDIndexer.to_dynamic_slice), [`_compute_offsets_from_indices`](pallas/triton/lowering.md#_compute_offsets_from_indices), [`_load_lowering_rule`](pallas/mosaic/lowering.md#_load_lowering_rule), [`_swap_discharge_rule`](pallas/primitives.md#_swap_discharge_rule), [`_get_pull_rule`](pallas/fuser/block_spec.md#_get_pull_rule), [`_get_eval_rule`](pallas/fuser/block_spec.md#_get_eval_rule), [`_load_discharge_rule`](pallas/primitives.md#_load_discharge_rule), [`from_index`](numpy/indexing.md#IndexType.from_index), [`_apply_mask_and_soft_cap`](../experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.md#_apply_mask_and_soft_cap), [`_swap_eval_rule`](pallas/fuser/block_spec.md#_swap_eval_rule), [`from_indices_shape`](state/indexing.md#NDIndexer.from_indices_shape), [`_atomic_store_discharge_rule`](pallas/mosaic_gpu/primitives.md#_atomic_store_discharge_rule), [`_convert_to_gather_arrays`](state/discharge.md#_convert_to_gather_arrays), [`_make_block_slice`](pallas/mosaic/pipeline.md#_make_block_slice), [`_reinterpret_int4_as_uint8`](pallas/triton/lowering.md#_reinterpret_int4_as_uint8), [`_store_slice_to_kernel_input`](pallas/mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call._execute_grid_for_core._body._store_slice_to_kernel_input), [`body`](../experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.md#flash_attention_kernel.body), [`fetch_and_add`](pallas/mosaic/sc_primitives.md#fetch_and_add), [`_ndindexer_indices`](pallas/mosaic_gpu/lowering.md#_ndindexer_indices), [`_atomic_rmw_discharge_rule`](pallas/triton/primitives.md#_atomic_rmw_discharge_rule), [`commute_ndindexer`](pallas/mosaic_gpu/core.md#UntilingTransform.commute_ndindexer), [`_batch_indexer`](state/primitives.md#_batch_indexer), [`__post_init__`](state/indexing.md#NDIndexer.__post_init__), [`commute_ndindexer`](pallas/mosaic_gpu/core.md#ExpandLeadingBatchDimensionsTransform.commute_ndindexer), [`_is_contiguous_int4`](pallas/triton/lowering.md#_is_contiguous_int4), [`body`](../experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.md#_flash_attention_dkv_kernel.body), [`_get_barrier_base_index`](pallas/mosaic_gpu/primitives.md#_get_barrier_base_index), [`run`](../experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.md#_flash_attention_dq_kernel.run), [`_offset_indexer`](pallas/fuser/block_spec.md#_offset_indexer), [`compute_gmem_slice`](pallas/mosaic_gpu/pipeline.md#BufferedRef.compute_gmem_slice), [`_pp_slice`](state/indexing.md#_pp_slice), [`merge_indexers`](pallas/mosaic_gpu/lowering.md#merge_indexers), [`pretty_print`](state/indexing.md#NDIndexer.pretty_print), [`_extract_indirect_offsets`](pallas/mosaic/sc_lowering.md#_extract_indirect_offsets), [`_is_trivial_indexer`](state/discharge.md#_is_trivial_indexer), [`_maybe_convert_to_dynamic_slice`](state/discharge.md#_maybe_convert_to_dynamic_slice), [`commute_transpose_indexer`](pallas/mosaic_gpu/core.md#commute_transpose_indexer), [`unpack_ndindexer`](state/indexing.md#unpack_ndindexer)  (+15 more)

## Functions
- `dslice(start: int | Array | None, size: int | Array | None = None, stride: int | None = None)` — [`L75`](../../../../../../raw/code/jax/jax/_src/indexing.py#L75) — Constructs a ``Slice`` from a start index and a size.

## Module values
- `ds` — [`L118`](../../../../../../raw/code/jax/jax/_src/indexing.py#L118)

