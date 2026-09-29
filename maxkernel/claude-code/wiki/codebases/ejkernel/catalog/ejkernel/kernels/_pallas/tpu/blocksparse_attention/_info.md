---
title: 'Module: ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.blocksparse_attention._info`/
symbols:
  _process_mask: _process_mask().
  MaskInfo: MaskInfo#
  _process_dynamic_mask: _process_dynamic_mask().
  MaskInfo.partial_mask_blocks: MaskInfo#partial_mask_blocks.
  MaskInfo.data_next: MaskInfo#data_next.
  MaskInfo.q_sequence: MaskInfo#q_sequence.
  MaskInfo.mask_next: MaskInfo#mask_next.
  MaskInfo.block_mask: MaskInfo#block_mask.
  _get_mask_info_for_shard: _get_mask_info_for_shard().
  _shrink_mask_info: _shrink_mask_info().
  _shrink_mask_info_dkv: _shrink_mask_info_dkv().
  _check_mask: _check_mask().
  _HashableNDArray.array: _HashableNDArray#array.
  _HashableNDArray.__eq__: _HashableNDArray#__eq__().
  process_mask: process_mask.
  process_mask_dkv: process_mask_dkv.
  process_dynamic_mask: process_dynamic_mask.
  process_dynamic_mask_dkv: process_dynamic_mask_dkv.
  _downcast_to_small_type: _downcast_to_small_type().
  _HashableNDArray: _HashableNDArray#
  _HashableNDArray.__init__: _HashableNDArray#__init__().
  _HashableNDArray.__hash__: _HashableNDArray#__hash__().
  MaskInfo.is_dynamic_mask: MaskInfo#is_dynamic_mask.
  _process_dynamic_mask._downcast: _process_dynamic_mask()._downcast().
  _process_mask.set_block_mask: _process_mask().set_block_mask().
  _slice_mask_info: _slice_mask_info().
  _process_mask.assign_unique_ids: _process_mask().assign_unique_ids().
  _shrink_mask_info.select_cols: _shrink_mask_info().select_cols().
  _shrink_mask_info_dkv.select_rows: _shrink_mask_info_dkv().select_rows().
---
# Module: [`ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py)

## Classes
### `MaskInfo`  ·  implements/extends NamedTuple
- def: [`ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py:79`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L79) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info.md)
- doc: Contains runtime masking information for the Splash attention kernel.
- signature: `class MaskInfo(NamedTuple):`
- members:
  - `block_mask` — [`L124`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L124) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info.md)
  - `data_next` — [`L122`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L122) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info.md)
  - `is_dynamic_mask` — [`L127`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L127) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info.md)
  - `mask_next` — [`L123`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L123) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info.md)
  - `partial_mask_blocks` — [`L125`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L125) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info.md)
  - `q_sequence` — [`L126`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L126) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info.md)
- used by: [`_splash_attention_bwd_dkv`](_kernel.md#_splash_attention_bwd_dkv), [`_splash_attention_bwd_dq`](_kernel.md#_splash_attention_bwd_dq), [`_process_mask`](_info.md#_process_mask), [`_splash_attention_bwd`](_kernel.md#_splash_attention_bwd), [`manual_sharding_spec`](_kernel.md#SplashAttentionKernel.manual_sharding_spec), [`body`](../ring_attention/_pallas_impl_bwd.md#_ring_attention_forward.body), [`_process_dynamic_mask`](_info.md#_process_dynamic_mask), [`_splash_attention`](_kernel.md#_splash_attention), [`MaskInfo`](../ring_attention/_pallas_impl_bwd.md#MaskInfo), [`_splash_attention_custom`](_kernel.md#_splash_attention_custom), [`_splash_attention_fwd`](_kernel.md#_splash_attention_fwd), [`_splash_attention_forward`](_kernel.md#_splash_attention_forward), [`_collapse_partial_mask_blocks`](_kernel.md#_splash_attention._collapse_partial_mask_blocks), [`SplashResidualsType`](_kernel.md#SplashResidualsType), [`tree_unflatten`](_kernel.md#SplashAttentionKernel.tree_unflatten), [`__init__`](_kernel.md#SplashAttentionKernel.__init__)

### `_HashableNDArray`
- def: [`ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py:198`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L198) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info.md)
- doc: Helper to make a numpy array hashable: can be added associative containers.
- signature: `class _HashableNDArray:`
- members:
  - `array` — [`L205`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L205) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info.md)
- protocol/private: `__eq__`[`L213`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L213), `__hash__`[`L210`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L210), `__init__`[`L207`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L207)
- used by: [`_process_mask`](_info.md#_process_mask)

## Functions
- `_check_mask(mask: mask_lib.Mask)` — [`L166`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L166) — Check that the given mask is valid.
- `_downcast(array: jax.Array, max_value: int)` — [`L493`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L493)
- `_downcast_to_small_type(array: np.ndarray)` — [`L130`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L130) — Downcast numpy array. — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info.md)
- `_get_mask_info_for_shard(output_shape: tuple[int, int, int], has_mask_next: bool, mask: mask_lib.MultiHeadMask | jax.Array, block_shape: tuple[int, int], coords_to_partial_mask_block_index: dict[tuple[int, int, int], int], masks_per_head_shard: int, head_start: int, num_heads: int, q_seq_start: int, q_seq_shard_size: int, blocked_q_seq_start: int, is_dkv: bool)` — [`L219`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L219) — Process a slice of the mask to compute data_next and mask_next.
- `_process_dynamic_mask(mask: jax.Array, block_shape: tuple[int, int], is_dkv: bool, *, downcast_smem_data: bool = True, head_shards: int = 1, q_seq_shards: int = 1, shrink_grid: bool = True)` — [`L358`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L358) — Similar to `_process_mask` but the mask must be a dynamic array. — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info.md)
- `_process_mask(mask: mask_lib.MultiHeadMask, block_shape: tuple[int, int], is_dkv: bool, *, downcast_smem_data: bool = True, head_shards: int = 1, q_seq_shards: int = 1, shrink_grid: bool = True)` — [`L526`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L526) — Transform a dense mask into a sparse representation. — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info.md)
- `_shrink_mask_info(*, block_mask: np.ndarray, data_next: np.ndarray, mask_next: np.ndarray, head_shards: int)` — [`L858`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L858) — Shrink forward/dQ mask info by removing empty KV columns. — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info.md)
- `_shrink_mask_info_dkv(*, block_mask: np.ndarray, data_next: np.ndarray, mask_next: np.ndarray, head_shards: int)` — [`L936`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L936) — Shrink dKV mask info by removing empty Q rows. — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info.md)
- `_slice_mask_info(*, block_mask: np.ndarray, data_next: np.ndarray, mask_next: np.ndarray, head_shards: int, slice_function: Callable[[np.ndarray], np.ndarray])` — [`L1013`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L1013) — Apply a slicing function to mask info arrays across all head shards.
- `assign_unique_ids(objects)` — [`L593`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L593)
- `select_cols(array)` — [`L913`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L913)
- `select_rows(array)` — [`L990`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L990)
- `set_block_mask(mask_id: int, q_index: int, kv_index: int, value: int)` — [`L628`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L628)

## Module values
- `process_dynamic_mask` — [`L1066`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L1066) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info.md)
- `process_dynamic_mask_dkv` — [`L1068`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L1068) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info.md)
- `process_mask` — [`L1062`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L1062) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info.md)
- `process_mask_dkv` — [`L1064`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info.py#L1064) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info.md)

