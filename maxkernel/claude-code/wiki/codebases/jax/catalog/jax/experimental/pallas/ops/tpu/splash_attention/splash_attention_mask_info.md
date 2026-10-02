---
title: 'Module: jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask_info`/
symbols:
  _process_dynamic_mask: _process_dynamic_mask().
  _process_mask: _process_mask().
  MaskInfo: MaskInfo#
  MaskInfo.partial_mask_blocks: MaskInfo#partial_mask_blocks.
  MaskInfo.data_next: MaskInfo#data_next.
  MaskInfo.q_sequence: MaskInfo#q_sequence.
  _process_dynamic_mask._downcast: _process_dynamic_mask()._downcast().
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
  MaskInfo.is_dynamic_mask: MaskInfo#is_dynamic_mask.
  _downcast_to_small_type: _downcast_to_small_type().
  _HashableNDArray: _HashableNDArray#
  _HashableNDArray.__init__: _HashableNDArray#__init__().
  _HashableNDArray.__hash__: _HashableNDArray#__hash__().
  _process_mask.set_block_mask: _process_mask().set_block_mask().
  _slice_mask_info: _slice_mask_info().
  _process_mask.assign_unique_ids: _process_mask().assign_unique_ids().
  _shrink_mask_info.select_cols: _shrink_mask_info().select_cols().
  _shrink_mask_info_dkv.select_rows: _shrink_mask_info_dkv().select_rows().
---
# Module: [`jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py)

## Classes
### `MaskInfo`  ·  implements/extends NamedTuple
- def: [`jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py:32`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L32)
- doc: Contains runtime masking information for the Splash attention kernel.
- signature: `class MaskInfo(NamedTuple):`
- members:
  - `block_mask` — [`L77`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L77)
  - `data_next` — [`L75`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L75)
  - `is_dynamic_mask` — [`L80`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L80)
  - `mask_next` — [`L76`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L76)
  - `partial_mask_blocks` — [`L78`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L78)
  - `q_sequence` — [`L79`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L79)
- uses (calls/refs, reference-scoped): [`Array`](../../../../../_src/basearray.md#Array)
- used by: [`_splash_attention_bwd_dkv`](splash_attention_kernel.md#_splash_attention_bwd_dkv), [`_splash_attention_bwd_dq`](splash_attention_kernel.md#_splash_attention_bwd_dq), [`_splash_attention_bwd`](splash_attention_kernel.md#_splash_attention_bwd), [`_process_dynamic_mask`](splash_attention_mask_info.md#_process_dynamic_mask), [`_process_mask`](splash_attention_mask_info.md#_process_mask), [`manual_sharding_spec`](splash_attention_kernel.md#SplashAttentionKernel.manual_sharding_spec), [`_splash_attention`](splash_attention_kernel.md#_splash_attention), [`_splash_attention_custom`](splash_attention_kernel.md#_splash_attention_custom), [`__call__`](splash_attention_kernel.md#SplashAttentionKernel.__call__), [`_splash_attention_fwd`](splash_attention_kernel.md#_splash_attention_fwd), [`_splash_attention_forward`](splash_attention_kernel.md#_splash_attention_forward), [`SplashResidualsType`](splash_attention_kernel.md#SplashResidualsType), [`_collapse_partial_mask_blocks`](splash_attention_kernel.md#_splash_attention._collapse_partial_mask_blocks), [`tree_unflatten`](splash_attention_kernel.md#SplashAttentionKernel.tree_unflatten), [`__init__`](splash_attention_kernel.md#SplashAttentionKernel.__init__)

### `_HashableNDArray`
- def: [`jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py:152`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L152)
- doc: Helper to make a numpy array hashable: can be added associative containers.
- signature: `class _HashableNDArray:`
- members:
  - `array` — [`L158`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L158)
- protocol/private: `__eq__`[`L166`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L166), `__hash__`[`L163`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L163), `__init__`[`L160`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L160)
- used by: [`_process_mask`](splash_attention_mask_info.md#_process_mask)

## Functions
- `_check_mask(mask: mask_lib.Mask)` — [`L119`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L119) — Check that the given mask is valid.
- `_downcast(array: jax.Array, max_value: int)` — [`L477`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L477)
- `_downcast_to_small_type(array: np.ndarray)` — [`L83`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L83) — Downcast numpy array.
- `_get_mask_info_for_shard(output_shape: tuple[int, int, int], has_mask_next: bool, mask: mask_lib.MultiHeadMask | jax.Array, block_shape: tuple[int, int], coords_to_partial_mask_block_index: dict[tuple[int, int, int], int], masks_per_head_shard: int, head_start: int, num_heads: int, q_seq_start: int, q_seq_shard_size: int, blocked_q_seq_start: int, is_dkv: bool)` — [`L172`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L172) — Process a slice of the mask to compute data_next and mask_next.
- `_process_dynamic_mask(mask: jax.Array, block_shape: tuple[int, int], is_dkv: bool, *, downcast_smem_data: bool = True, head_shards: int = 1, q_seq_shards: int = 1, shrink_grid: bool = True)` — [`L316`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L316) — Similar to `_process_mask` but the mask must be a dynamic array.
- `_process_mask(mask: mask_lib.MultiHeadMask, block_shape: tuple[int, int], is_dkv: bool, *, downcast_smem_data: bool = True, head_shards: int = 1, q_seq_shards: int = 1, shrink_grid: bool = True)` — [`L518`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L518) — Transform a dense mask into a sparse representation.
- `_shrink_mask_info(*, block_mask: np.ndarray, data_next: np.ndarray, mask_next: np.ndarray | None, head_shards: int)` — [`L965`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L965)
- `_shrink_mask_info_dkv(*, block_mask: np.ndarray, data_next: np.ndarray, mask_next: np.ndarray | None, head_shards: int)` — [`L1031`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L1031)
- `_slice_mask_info(*, block_mask: np.ndarray, data_next: np.ndarray, mask_next: np.ndarray | None, head_shards: int, slice_function: Callable[[np.ndarray], np.ndarray])` — [`L1096`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L1096)
- `assign_unique_ids(objects)` — [`L592`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L592)
- `select_cols(array)` — [`L1008`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L1008)
- `select_rows(array)` — [`L1073`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L1073)
- `set_block_mask(mask_id: int, q_index: int, kv_index: int, value: int)` — [`L654`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L654)

## Module values
- `process_dynamic_mask` — [`L1132`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L1132)
- `process_dynamic_mask_dkv` — [`L1133`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L1133)
- `process_mask` — [`L1129`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L1129)
- `process_mask_dkv` — [`L1130`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py#L1130)

