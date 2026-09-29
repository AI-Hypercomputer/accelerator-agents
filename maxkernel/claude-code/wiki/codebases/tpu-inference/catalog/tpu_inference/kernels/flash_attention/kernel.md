---
title: 'Module: tpu_inference/kernels/flash_attention/kernel.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/flash_attention/kernel.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.flash_attention.kernel`/
symbols:
  _flash_attention_impl: _flash_attention_impl().
  flash_attention: flash_attention().
  encoder_only_flash_attention: encoder_only_flash_attention().
  BlockSizes.get_default: BlockSizes#get_default().
  encoder_only_flash_attention.build_segment_ids: encoder_only_flash_attention().build_segment_ids().
  mha_reference_no_custom_vjp: mha_reference_no_custom_vjp().
  MIN_BLOCK_SIZE: MIN_BLOCK_SIZE.
  _fwd_cost_estimate: _fwd_cost_estimate().
  mha_reference: mha_reference().
  BlockSizes.__post_init__: BlockSizes#__post_init__().
  _flash_attention_kernel_single_batch.run._body: _flash_attention_kernel_single_batch().run()._body().
  NUM_LANES: NUM_LANES.
  SegmentIds: SegmentIds#
  BlockSizes: BlockSizes#
  _flash_attention_kernel: _flash_attention_kernel().
  _flash_attention_kernel_single_batch_single_step: _flash_attention_kernel_single_batch_single_step().
  _mha_reference: _mha_reference().
  _flash_attention_impl.kv_index_map: _flash_attention_impl().kv_index_map().
  BlockSizes.block_q: BlockSizes#block_q.
  BlockSizes.block_k: BlockSizes#block_k.
  below_or_on_diag: below_or_on_diag().
  _verify_block: _verify_block().
  _flash_attention: _flash_attention().
  _flash_attention_kernel_single_batch: _flash_attention_kernel_single_batch().
  _flash_attention_impl.ab_index_map: _flash_attention_impl().ab_index_map().
  _flash_attention_impl.kv_segment_ids_index_map: _flash_attention_impl().kv_segment_ids_index_map().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  BlockSizes.block_k_major: BlockSizes#block_k_major.
  BlockSizes.block_b: BlockSizes#block_b.
  encoder_only_flash_attention.pad_token: encoder_only_flash_attention().pad_token().
  NUM_SUBLANES: NUM_SUBLANES.
  SegmentIds.q: SegmentIds#q.
  SegmentIds.kv: SegmentIds#kv.
  encoder_only_flash_attention.build_segment_ids.build_padded_segment: encoder_only_flash_attention().build_segment_ids().build_padded_segment().
  TRANS_B_DIM_NUMBERS: TRANS_B_DIM_NUMBERS.
  _bytes: _bytes().
  _flash_attention_impl.lm_index_map: _flash_attention_impl().lm_index_map().
  BlockSizes.verify_major_minor: BlockSizes#verify_major_minor().
  _flash_attention_impl.q_index_map: _flash_attention_impl().q_index_map().
  _flash_attention_impl.o_index_map: _flash_attention_impl().o_index_map().
  _flash_attention_impl.q_segment_ids_index_map: _flash_attention_impl().q_segment_ids_index_map().
  _flash_attention_kernel_single_batch.start_new_sequence: _flash_attention_kernel_single_batch().start_new_sequence().
  _flash_attention_kernel_single_batch.run: _flash_attention_kernel_single_batch().run().
  _flash_attention_kernel_single_batch.store_output: _flash_attention_kernel_single_batch().store_output().
---
# Module: [`tpu_inference/kernels/flash_attention/kernel.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py)

## Classes
### `BlockSizes`
- def: [`tpu_inference/kernels/flash_attention/kernel.py:41`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L41)
- doc: Tile sizes parameterizing FlashAttention kernels.
- signature: `class BlockSizes:`
- members:
  - `get_default(cls, batch_size, num_heads, q_seq_len, kv_len, d_model)` — [`L66`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L66)
  - `verify_major_minor(prefix, suffix, major, minor)` — [`L54`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L54)
  - `block_b` — [`L50`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L50)
  - `block_k` — [`L49`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L49)
  - `block_k_major` — [`L48`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L48)
  - `block_q` — [`L47`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L47)
- protocol/private: `__post_init__`[`L52`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L52)
- used by: [`flash_attention`](kernel.md#flash_attention), [`encoder_only_flash_attention`](kernel.md#encoder_only_flash_attention)

### `SegmentIds`  ·  implements/extends NamedTuple
- def: [`tpu_inference/kernels/flash_attention/kernel.py:23`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L23)
- doc: SegmentIds for Q and KV sequences.
- signature: `class SegmentIds(NamedTuple):`
- members:
  - `kv` — [`L37`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L37)
  - `q` — [`L36`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L36)
- used by: [`build_segment_ids`](kernel.md#encoder_only_flash_attention.build_segment_ids), [`mha_reference_no_custom_vjp`](kernel.md#mha_reference_no_custom_vjp), [`_fwd_cost_estimate`](kernel.md#_fwd_cost_estimate), [`mha_reference`](kernel.md#mha_reference), [`_mha_reference`](kernel.md#_mha_reference)

## Functions
- `_body(start_k)` — [`L359`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L359)
- `_bytes(x: jax.Array | jax.ShapeDtypeStruct)` — [`L533`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L533)
- `_flash_attention(q, k, v, ab, segment_ids, save_residuals, causal, sm_scale, block_sizes, vmem_limit_bytes, debug)` — [`L260`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L260)
- `_flash_attention_impl(q, k, v, ab, segment_ids, save_residuals, causal, sm_scale, block_b, block_q, block_k_major, block_k, vmem_limit_bytes, debug)` — [`L567`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L567)
- `_flash_attention_kernel(q_tile_ref, *args, **kwargs)` — [`L301`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L301)
- `_flash_attention_kernel_single_batch(batch_idx: tuple[int, ...], q_tile_ref, k_tile_ref, v_tile_ref, ab_tile_ref, q_segment_ids_tile_ref, kv_segment_ids_tile_ref, o_tile_ref, l_ref, m_ref, m_scratch_ref, l_scratch_ref, acc_scratch_ref, *, causal, sm_scale, block_k, kv_seq_len, mask_value)` — [`L312`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L312)
- `_flash_attention_kernel_single_batch_single_step(batch_idx: tuple[int, ...], q_tile_ref, k_tile_ref, v_tile_ref, ab_tile_ref, q_segment_ids_tile_ref, kv_segment_ids_tile_ref, o_tile_ref, l_ref: Any | None = None, m_ref: Any | None = None, *, causal, sm_scale, block_k, kv_seq_len, mask_value)` — [`L458`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L458)
- `_fwd_cost_estimate(q: jax.Array, k: jax.Array, v: jax.Array, ab: jax.Array | None, segment_ids: SegmentIds | None, *, causal: bool, sm_scale: jax.Array | None, kernel_inputs_specs, kernel_outputs_specs)` — [`L537`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L537)
- `_mha_reference(q, k, v, ab, segment_ids: SegmentIds | None, causal: bool, mask_value: float, sm_scale: float, save_residuals: bool)` — [`L851`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L851)
- `_verify_block(block_name, dim_name, block, dim, should_divide=True)` — [`L875`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L875)
- `ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L619`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L619)
- `below_or_on_diag(r, r_blk_size, c, c_blk_size)` — [`L295`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L295)
- `build_padded_segment(size: int)` — [`L145`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L145)
- `build_segment_ids()` — [`L132`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L132)
- `encoder_only_flash_attention(q, k, v, seq_lens, *, causal: bool = False, sm_scale: float | None = None, sliding_window: int | None = None, block_sizes: BlockSizes | None = None, vmem_limit_bytes: int | None = None, debug: bool = False)` — [`L85`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L85)
- `flash_attention(q, k, v, ab=None, segment_ids=None, *, causal: bool = False, sm_scale: float = 1, block_sizes: BlockSizes | None = None, vmem_limit_bytes: int | None = None, debug: bool = False)` — [`L197`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L197)
- `kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L605`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L605)
- `kv_segment_ids_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L695`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L695)
- `lm_index_map(batch_index, head_index, q_seq_index, _)` — [`L641`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L641)
- `mha_reference(q, k, v, ab, segment_ids: SegmentIds | None = None, causal: bool = False, mask_value: float = DEFAULT_MASK_VALUE, sm_scale=1)` — [`L828`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L828)
- `mha_reference_no_custom_vjp(q, k, v, ab: jax.Array | None = None, segment_ids: SegmentIds | None = None, *, causal: bool = False, mask_value: float = DEFAULT_MASK_VALUE, sm_scale: float = 1, save_residuals: bool = False)` — [`L780`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L780)
- `o_index_map(batch_index, head_index, q_seq_index, _)` — [`L638`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L638)
- `pad_token(t: jax.Array, size)` — [`L112`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L112)
- `q_index_map(batch_index, head_index, q_seq_index, _)` — [`L602`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L602)
- `q_segment_ids_index_map(batch_index, head_index, q_seq_index, _)` — [`L691`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L691)
- `run()` — [`L356`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L356)
- `start_new_sequence()` — [`L340`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L340)
- `store_output()` — [`L447`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L447)

## Module values
- `DEFAULT_MASK_VALUE` — [`L18`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L18)
- `MIN_BLOCK_SIZE` — [`L291`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L291)
- `NUM_LANES` — [`L19`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L19)
- `NUM_SUBLANES` — [`L20`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L20)
- `TRANS_B_DIM_NUMBERS` — [`L292`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L292)

