---
title: 'Module: ringattention/ringattention_pallas_tpu.py'
type: catalog
provenance: extracted
module: ringattention/ringattention_pallas_tpu.py
status: fresh
symbol_base: scip-python python ringattention 0.0.0 `ringattention.ringattention_pallas_tpu`/
symbols:
  _flash_attention_impl: _flash_attention_impl().
  _flash_attention_bwd_dkv: _flash_attention_bwd_dkv().
  _ring_flash_attention_fwd_tpu: _ring_flash_attention_fwd_tpu().
  PatchBlockSpec: PatchBlockSpec#
  _ring_flash_attention_bwd_tpu: _ring_flash_attention_bwd_tpu().
  _flash_attention_bwd: _flash_attention_bwd().
  _flash_attention_bwd_dq: _flash_attention_bwd_dq().
  BlockSizes.get_default: BlockSizes#get_default().
  MIN_BLOCK_SIZE: MIN_BLOCK_SIZE.
  BlockSizes.__post_init__: BlockSizes#__post_init__().
  BlockSizes.has_backward_blocks: BlockSizes#has_backward_blocks().
  _ring_flash_attention_fwd_tpu.scan_kv_block: _ring_flash_attention_fwd_tpu().scan_kv_block().
  _ring_flash_attention_bwd_tpu.scan_kv_block: _ring_flash_attention_bwd_tpu().scan_kv_block().
  NUM_LANES: NUM_LANES.
  _verify_block: _verify_block().
  _flash_attention_dkv_kernel.q_body.k_body: _flash_attention_dkv_kernel().q_body().k_body().
  _flash_attention_dq_kernel.body: _flash_attention_dq_kernel().body().
  _flash_attention_kernel_single_batch.run.body: _flash_attention_kernel_single_batch().run().body().
  BlockSizes.block_q_major_dkv: BlockSizes#block_q_major_dkv.
  BlockSizes.block_k_major_dkv: BlockSizes#block_k_major_dkv.
  BlockSizes.block_k_dkv: BlockSizes#block_k_dkv.
  BlockSizes.block_q_dkv: BlockSizes#block_q_dkv.
  BlockSizes.block_k_major_dq: BlockSizes#block_k_major_dq.
  BlockSizes.block_k_dq: BlockSizes#block_k_dq.
  NUM_SUBLANES: NUM_SUBLANES.
  BlockSizes.block_q_dq: BlockSizes#block_q_dq.
  TRANS_B_DIM_NUMBERS: TRANS_B_DIM_NUMBERS.
  ring_flash_attention_tpu: ring_flash_attention_tpu().
  _flash_attention_impl.kv_index_map: _flash_attention_impl().kv_index_map().
  _flash_attention_bwd_dkv.qo_index_map: _flash_attention_bwd_dkv().qo_index_map().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  BlockSizes.block_k_major: BlockSizes#block_k_major.
  BlockSizes.block_k: BlockSizes#block_k.
  BlockSizes.verify_major_minor: BlockSizes#verify_major_minor().
  _flash_attention_impl.lm_index_map: _flash_attention_impl().lm_index_map().
  BlockSizes: BlockSizes#
  _flash_attention: _flash_attention().
  _flash_attention_fwd: _flash_attention_fwd().
  _flash_attention_kernel: _flash_attention_kernel().
  _flash_attention_kernel_single_batch: _flash_attention_kernel_single_batch().
  _flash_attention_impl.ab_index_map: _flash_attention_impl().ab_index_map().
  _flash_attention_impl.kv_segment_ids_index_map: _flash_attention_impl().kv_segment_ids_index_map().
  _flash_attention_dkv_kernel: _flash_attention_dkv_kernel().
  _flash_attention_dkv_kernel.q_body: _flash_attention_dkv_kernel().q_body().
  _flash_attention_bwd_dkv.q_segment_ids_index_map: _flash_attention_bwd_dkv().q_segment_ids_index_map().
  _flash_attention_dq_kernel: _flash_attention_dq_kernel().
  _flash_attention_bwd_dq.kv_index_map: _flash_attention_bwd_dq().kv_index_map().
  _flash_attention_bwd_dq.kv_segment_ids_index_map: _flash_attention_bwd_dq().kv_segment_ids_index_map().
  BlockSizes.block_q: BlockSizes#block_q.
  BlockSizes.block_b: BlockSizes#block_b.
  _flash_attention_bwd_dq.qo_index_map: _flash_attention_bwd_dq().qo_index_map().
  _flash_attention_dkv_kernel.run: _flash_attention_dkv_kernel().run().
  _flash_attention_dq_kernel.run: _flash_attention_dq_kernel().run().
  SegmentIds.q: SegmentIds#q.
  SegmentIds.kv: SegmentIds#kv.
  _flash_attention_impl.q_index_map: _flash_attention_impl().q_index_map().
  SegmentIds: SegmentIds#
  _flash_attention_impl.o_index_map: _flash_attention_impl().o_index_map().
  _flash_attention_impl.q_segment_ids_index_map: _flash_attention_impl().q_segment_ids_index_map().
  _flash_attention_bwd_dkv.kv_index_map: _flash_attention_bwd_dkv().kv_index_map().
  _flash_attention_bwd_dkv.lm_index_map: _flash_attention_bwd_dkv().lm_index_map().
  _flash_attention_bwd_dkv.ab_index_map: _flash_attention_bwd_dkv().ab_index_map().
  _flash_attention_bwd_dkv.kv_segment_ids_index_map: _flash_attention_bwd_dkv().kv_segment_ids_index_map().
  _flash_attention_bwd_dkv.dkv_index_map: _flash_attention_bwd_dkv().dkv_index_map().
  _flash_attention_bwd_dq.lm_index_map: _flash_attention_bwd_dq().lm_index_map().
  _flash_attention_bwd_dq.ab_index_map: _flash_attention_bwd_dq().ab_index_map().
  _flash_attention_bwd_dq.q_segment_ids_index_map: _flash_attention_bwd_dq().q_segment_ids_index_map().
  PatchBlockSpec.__init__: PatchBlockSpec#__init__().
  _flash_attention_kernel_single_batch.start_new_sequence: _flash_attention_kernel_single_batch().start_new_sequence().
  _flash_attention_kernel_single_batch.run: _flash_attention_kernel_single_batch().run().
  _flash_attention_kernel_single_batch.store_output: _flash_attention_kernel_single_batch().store_output().
  _flash_attention_dkv_kernel.start_new_sequence: _flash_attention_dkv_kernel().start_new_sequence().
  _flash_attention_dkv_kernel.end_of_q_sequence: _flash_attention_dkv_kernel().end_of_q_sequence().
  _flash_attention_dq_kernel.start_new_sequence: _flash_attention_dq_kernel().start_new_sequence().
  _flash_attention_dq_kernel.zero_out_ds: _flash_attention_dq_kernel().zero_out_ds().
  _flash_attention_dq_kernel.end_of_kv_sequence: _flash_attention_dq_kernel().end_of_kv_sequence().
---
# Module: [`ringattention/ringattention_pallas_tpu.py`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py)

## Classes
### `BlockSizes`
- def: [`ringattention/ringattention_pallas_tpu.py:219`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L219) — documented in [ringattention-ringattention_pallas_tpu](../../concepts/ringattention-ringattention_pallas_tpu.md)
- members:
  - `get_default(cls, batch_size, num_heads, q_seq_len, kv_len, d_model)` — [`L273`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L273)
  - `has_backward_blocks(self)` — [`L260`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L260)
  - `verify_major_minor(prefix, suffix, major, minor)` — [`L235`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L235)
  - `block_b` — [`L223`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L223) — documented in [ringattention-ringattention_pallas_tpu](../../concepts/ringattention-ringattention_pallas_tpu.md)
  - `block_k` — [`L222`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L222) — documented in [ringattention-ringattention_pallas_tpu](../../concepts/ringattention-ringattention_pallas_tpu.md)
  - `block_k_dkv` — [`L227`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L227)
  - `block_k_dq` — [`L231`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L231)
  - `block_k_major` — [`L221`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L221) — documented in [ringattention-ringattention_pallas_tpu](../../concepts/ringattention-ringattention_pallas_tpu.md)
  - `block_k_major_dkv` — [`L226`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L226)
  - `block_k_major_dq` — [`L230`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L230)
  - `block_q` — [`L220`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L220) — documented in [ringattention-ringattention_pallas_tpu](../../concepts/ringattention-ringattention_pallas_tpu.md)
  - `block_q_dkv` — [`L228`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L228)
  - `block_q_dq` — [`L232`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L232)
  - `block_q_major_dkv` — [`L225`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L225)
- protocol/private: `__post_init__`[`L234`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L234)
- used by: [`_ring_flash_attention_fwd_tpu`](ringattention_pallas_tpu.md#_ring_flash_attention_fwd_tpu), [`_ring_flash_attention_bwd_tpu`](ringattention_pallas_tpu.md#_ring_flash_attention_bwd_tpu), [`_flash_attention_bwd`](ringattention_pallas_tpu.md#_flash_attention_bwd)

### `PatchBlockSpec`  ·  implements/extends BlockSpec
- def: [`ringattention/ringattention_pallas_tpu.py:18`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L18) — documented in [ringattention-ringattention_pallas_tpu](../../concepts/ringattention-ringattention_pallas_tpu.md)
- signature: `class PatchBlockSpec(pl.BlockSpec):`
- protocol/private: `__init__`[`L19`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L19)
- used by: [`_flash_attention_bwd_dkv`](ringattention_pallas_tpu.md#_flash_attention_bwd_dkv), [`_flash_attention_impl`](ringattention_pallas_tpu.md#_flash_attention_impl), [`_flash_attention_bwd_dq`](ringattention_pallas_tpu.md#_flash_attention_bwd_dq)

### `SegmentIds`
- def: [`ringattention/ringattention_pallas_tpu.py:201`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L201) — documented in [ringattention-ringattention_pallas_tpu](../../concepts/ringattention-ringattention_pallas_tpu.md)
- members:
  - `kv` — [`L215`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L215) — documented in [ringattention-ringattention_pallas_tpu](../../concepts/ringattention-ringattention_pallas_tpu.md)
  - `q` — [`L214`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L214) — documented in [ringattention-ringattention_pallas_tpu](../../concepts/ringattention-ringattention_pallas_tpu.md)
- used by: [`scan_kv_block`](ringattention_pallas_tpu.md#_ring_flash_attention_bwd_tpu.scan_kv_block), [`scan_kv_block`](ringattention_pallas_tpu.md#_ring_flash_attention_fwd_tpu.scan_kv_block)

## Functions
- `_flash_attention(q, k, v, carry, q_chunk_idx_start, k_chunk_idx_start, ab, segment_ids, save_residuals, causal_block_size, sm_scale, block_sizes, debug)` — [`L290`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L290)
- `_flash_attention_bwd(save_residuals: bool, causal_block_size: Optional[int], sm_scale: float, block_sizes: BlockSizes, debug: bool, q_chunk_idx_start, k_chunk_idx_start, residuals, do)` — [`L360`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L360) — VJP rule for FlashAttention. — documented in [ringattention-ringattention_pallas_tpu](../../concepts/ringattention-ringattention_pallas_tpu.md)
- `_flash_attention_bwd_dkv(q_chunk_idx_start, k_chunk_idx_start, q, k, v, ab, segment_ids, l, m, do, di, *, block_q_major: int | None, block_q: int | None, block_k_major: int | None, block_k: int | None, sm_scale: float, causal_block_size: Optional[int] = None, mask_value: float = DEFAULT_MASK_VALUE, debug: bool = False)` — [`L1029`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1029) — documented in [ringattention-ringattention_pallas_tpu](../../concepts/ringattention-ringattention_pallas_tpu.md)
- `_flash_attention_bwd_dq(q_chunk_idx_start, k_chunk_idx_start, q, k, v, ab, segment_ids, l, m, do, di, *, block_q_major: int | None, block_k_major: int | None, block_k: int | None, sm_scale: float, causal_block_size: Optional[int], mask_value: float, debug: bool)` — [`L1424`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1424) — documented in [ringattention-ringattention_pallas_tpu](../../concepts/ringattention-ringattention_pallas_tpu.md)
- `_flash_attention_dkv_kernel(q_chunk_idx_start_ref, k_chunk_idx_start_ref, q_tile_ref, k_tile_ref, v_tile_ref, ab_tile_ref, q_segment_ids_tile_ref, kv_segment_ids_tile_ref, l_tile_ref, m_tile_ref, do_tile_ref, di_tile_ref, dk_tile_ref, dv_tile_ref, dk_scratch_ref, dv_scratch_ref, *, sm_scale: float, causal_block_size: Optional[int], mask_value: float, q_seq_len: int, block_q: int, block_k: int)` — [`L854`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L854)
- `_flash_attention_dq_kernel(q_chunk_idx_start_ref, k_chunk_idx_start_ref, q_tile_ref, k_tile_ref, v_tile_ref, ab_tile_ref, q_segment_ids_tile_ref, kv_segment_ids_tile_ref, l_tile_ref, m_tile_ref, do_tile_ref, di_tile_ref, dq_tile_ref, dq_scratch_ref, ds_tile_ref, *, sm_scale: float, causal_block_size: Optional[int], mask_value: float, kv_seq_len: int, block_k: int)` — [`L1265`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1265)
- `_flash_attention_fwd(q, k, v, carry, q_chunk_idx_start, k_chunk_idx_start, ab, segment_ids, save_residuals, causal_block_size, sm_scale, block_sizes, debug)` — [`L325`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L325) — documented in [ringattention-ringattention_pallas_tpu](../../concepts/ringattention-ringattention_pallas_tpu.md)
- `_flash_attention_impl(q, k, v, carry, q_chunk_idx_start, k_chunk_idx_start, ab, segment_ids, save_residuals, causal_block_size, sm_scale, block_b, block_q, block_k_major, block_k, debug)` — [`L616`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L616) — documented in [ringattention-ringattention_pallas_tpu](../../concepts/ringattention-ringattention_pallas_tpu.md)
- `_flash_attention_kernel(q_idx_chunk_start, k_idx_chunk_start, q_tile_ref, *args, **kwargs)` — [`L434`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L434)
- `_flash_attention_kernel_single_batch(batch_idx: tuple[int, ...], q_chunk_idx_start_ref, k_chunk_idx_start_ref, q_tile_ref, k_tile_ref, v_tile_ref, acc_tile_ref, l_tile_ref, m_tile_ref, ab_tile_ref, q_segment_ids_tile_ref, kv_segment_ids_tile_ref, o_tile_ref, m_scratch_ref, l_scratch_ref, acc_scratch_ref, l_ref: Any | None = None, m_ref: Any | None = None, *, causal_block_size, sm_scale, block_k, kv_seq_len, mask_value, block_q)` — [`L455`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L455) — documented in [ringattention-ringattention_pallas_tpu](../../concepts/ringattention-ringattention_pallas_tpu.md)
- `_ring_flash_attention_bwd_tpu(axis_name, float32_logits, blockwise_kwargs, res, g)` — [`L107`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L107)
- `_ring_flash_attention_fwd_tpu(q, k, v, attn_bias, segment_ids, cache_idx, axis_name, float32_logits, blockwise_kwargs)` — [`L23`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L23) — documented in [ringattention-ringattention_pallas_tpu](../../concepts/ringattention-ringattention_pallas_tpu.md)
- `_verify_block(block_name, dim_name, block, dim, should_divide=True)` — [`L1645`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1645) — documented in [ringattention-ringattention_pallas_tpu](../../concepts/ringattention-ringattention_pallas_tpu.md)
- `ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref)` — [`L685`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L685) — documented in [ringattention-ringattention_jax](../../concepts/ringattention-ringattention_jax.md)
- `ab_index_map(batch_index, head_index, kv_seq_index, q_seq_index, q_idx_ref, k_idx_ref)` — [`L1124`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1124)
- `ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref)` — [`L1512`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1512)
- `body(i, _)` — [`L514`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L514)
- `body(i, _)` — [`L1301`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1301)
- `dkv_index_map(batch_index, head_index, kv_seq_index, _, q_idx_ref, k_idx_ref)` — [`L1212`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1212)
- `end_of_kv_sequence()` — [`L1419`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1419)
- `end_of_q_sequence()` — [`L1024`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1024)
- `k_body(i, _)` — [`L896`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L896)
- `kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref)` — [`L664`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L664) — documented in [ringattention-ringattention_jax](../../concepts/ringattention-ringattention_jax.md)
- `kv_index_map(batch_index, head_index, kv_seq_index, _, q_idx_ref, k_idx_ref)` — [`L1104`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1104)
- `kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref)` — [`L1474`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1474)
- `kv_segment_ids_index_map(batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref)` — [`L768`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L768)
- `kv_segment_ids_index_map(batch_index, head_index, kv_seq_index, _, q_idx_ref, k_idx_ref)` — [`L1162`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1162)
- `kv_segment_ids_index_map(batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref)` — [`L1536`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1536)
- `lm_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref)` — [`L705`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L705)
- `lm_index_map(batch_index, head_index, _, q_seq_index, q_idx_ref, k_idx_ref)` — [`L1112`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1112)
- `lm_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref)` — [`L1500`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1500)
- `o_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref)` — [`L702`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L702)
- `q_body(j, _)` — [`L893`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L893)
- `q_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref)` — [`L661`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L661)
- `q_segment_ids_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref)` — [`L762`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L762)
- `q_segment_ids_index_map(batch_index, head_index, kv_seq_index, q_seq_index, q_idx_ref, k_idx_ref)` — [`L1142`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1142)
- `q_segment_ids_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref)` — [`L1530`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1530)
- `qo_index_map(batch_index, head_index, kv_seq_index, q_seq_index, q_idx_ref, k_idx_ref)` — [`L1077`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1077)
- `qo_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref)` — [`L1468`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1468)
- `ring_flash_attention_tpu(q, k, v, attn_bias, segment_ids, cache_idx, axis_name, float32_logits, blockwise_kwargs)` — [`L190`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L190) — documented in [ringattention-ringattention_pallas_tpu](../../concepts/ringattention-ringattention_pallas_tpu.md)
- `run()` — [`L510`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L510)
- `run()` — [`L1020`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1020)
- `run()` — [`L1410`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1410)
- `scan_kv_block(carry, idx)` — [`L64`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L64) — documented in [ringattention-ringattention_pallas_tpu](../../concepts/ringattention-ringattention_pallas_tpu.md)
- `scan_kv_block(carry, idx)` — [`L144`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L144)
- `start_new_sequence()` — [`L489`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L489)
- `start_new_sequence()` — [`L889`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L889)
- `start_new_sequence()` — [`L1298`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1298)
- `store_output()` — [`L608`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L608)
- `zero_out_ds()` — [`L1414`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1414)

## Module values
- `DEFAULT_MASK_VALUE` — [`L197`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L197)
- `MIN_BLOCK_SIZE` — [`L430`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L430) — documented in [ringattention-ringattention_pallas_tpu](../../concepts/ringattention-ringattention_pallas_tpu.md)
- `NUM_LANES` — [`L198`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L198) — documented in [ringattention-ringattention_pallas_tpu](../../concepts/ringattention-ringattention_pallas_tpu.md)
- `NUM_SUBLANES` — [`L199`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L199)
- `TRANS_B_DIM_NUMBERS` — [`L431`](../../../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L431)

