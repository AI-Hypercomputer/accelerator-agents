---
title: 'Module: jax/experimental/pallas/ops/tpu/flash_attention.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/tpu/flash_attention.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.tpu.flash_attention`/
symbols:
  _flash_attention_bwd_dkv: _flash_attention_bwd_dkv().
  _flash_attention_bwd_dq: _flash_attention_bwd_dq().
  _flash_attention_impl: _flash_attention_impl().
  _flash_attention_kernel_single_batch.run._body: _flash_attention_kernel_single_batch().run()._body().
  _flash_attention_kernel_single_batch_single_step: _flash_attention_kernel_single_batch_single_step().
  _flash_attention_dkv_kernel.q_body.k_body: _flash_attention_dkv_kernel().q_body().k_body().
  _flash_attention_dq_kernel.body: _flash_attention_dq_kernel().body().
  _flash_attention_bwd: _flash_attention_bwd().
  mha_reference_bwd: mha_reference_bwd().
  mha_reference_no_custom_vjp: mha_reference_no_custom_vjp().
  BlockSizes.get_default: BlockSizes#get_default().
  MIN_BLOCK_SIZE: MIN_BLOCK_SIZE.
  _fwd_cost_estimate: _fwd_cost_estimate().
  BlockSizes.__post_init__: BlockSizes#__post_init__().
  BlockSizes.has_backward_blocks: BlockSizes#has_backward_blocks().
  mha_reference: mha_reference().
  NUM_LANES: NUM_LANES.
  _flash_attention_kernel_single_batch.start_new_sequence: _flash_attention_kernel_single_batch().start_new_sequence().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  _bytes: _bytes().
  _verify_block: _verify_block().
  _mha_reference: _mha_reference().
  flash_attention: flash_attention().
  below_or_on_diag: below_or_on_diag().
  _flash_attention_dq_kernel: _flash_attention_dq_kernel().
  _flash_attention: _flash_attention().
  _flash_attention_dkv_kernel.run: _flash_attention_dkv_kernel().run().
  _flash_attention_dq_kernel.run: _flash_attention_dq_kernel().run().
  _flash_attention_impl.kv_index_map: _flash_attention_impl().kv_index_map().
  _flash_attention_bwd_dkv.qo_index_map: _flash_attention_bwd_dkv().qo_index_map().
  NUM_SUBLANES: NUM_SUBLANES.
  SegmentIds: SegmentIds#
  TRANS_B_DIM_NUMBERS: TRANS_B_DIM_NUMBERS.
  _flash_attention_kernel: _flash_attention_kernel().
  _flash_attention_kernel_single_batch: _flash_attention_kernel_single_batch().
  _flash_attention_impl.ab_index_map: _flash_attention_impl().ab_index_map().
  _flash_attention_impl.kv_segment_ids_index_map: _flash_attention_impl().kv_segment_ids_index_map().
  _flash_attention_dkv_kernel: _flash_attention_dkv_kernel().
  _flash_attention_dkv_kernel.q_body: _flash_attention_dkv_kernel().q_body().
  _flash_attention_bwd_dkv.q_segment_ids_index_map: _flash_attention_bwd_dkv().q_segment_ids_index_map().
  _flash_attention_bwd_dq.kv_index_map: _flash_attention_bwd_dq().kv_index_map().
  _flash_attention_bwd_dq.kv_segment_ids_index_map: _flash_attention_bwd_dq().kv_segment_ids_index_map().
  _mha_reference_fwd: _mha_reference_fwd().
  BlockSizes.block_q_major_dkv: BlockSizes#block_q_major_dkv.
  BlockSizes.block_k_major_dkv: BlockSizes#block_k_major_dkv.
  BlockSizes.block_k_dkv: BlockSizes#block_k_dkv.
  BlockSizes.block_q_dkv: BlockSizes#block_q_dkv.
  BlockSizes.block_k_major_dq: BlockSizes#block_k_major_dq.
  BlockSizes.block_k_dq: BlockSizes#block_k_dq.
  _flash_attention_dkv_kernel.start_new_sequence: _flash_attention_dkv_kernel().start_new_sequence().
  _flash_attention_dq_kernel.start_new_sequence: _flash_attention_dq_kernel().start_new_sequence().
  _flash_attention_dq_kernel.zero_out_ds: _flash_attention_dq_kernel().zero_out_ds().
  _flash_attention_dq_kernel.end_of_kv_sequence: _flash_attention_dq_kernel().end_of_kv_sequence().
  SegmentIds.q: SegmentIds#q.
  SegmentIds.kv: SegmentIds#kv.
  BlockSizes: BlockSizes#
  BlockSizes.verify_major_minor: BlockSizes#verify_major_minor().
  _flash_attention_fwd: _flash_attention_fwd().
  _mha_reference_bwd: _mha_reference_bwd().
  BlockSizes.block_q_dq: BlockSizes#block_q_dq.
  _flash_attention_bwd_dq.qo_index_map: _flash_attention_bwd_dq().qo_index_map().
  _flash_attention_kernel_single_batch.run: _flash_attention_kernel_single_batch().run().
  _flash_attention_kernel_single_batch.store_output: _flash_attention_kernel_single_batch().store_output().
  _flash_attention_dkv_kernel.end_of_q_sequence: _flash_attention_dkv_kernel().end_of_q_sequence().
  BlockSizes.block_k_major: BlockSizes#block_k_major.
  BlockSizes.block_k: BlockSizes#block_k.
  _flash_attention_impl.lm_index_map: _flash_attention_impl().lm_index_map().
  BlockSizes.block_q: BlockSizes#block_q.
  BlockSizes.block_b: BlockSizes#block_b.
  _flash_attention_impl.q_index_map: _flash_attention_impl().q_index_map().
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
---
# Module: [`jax/experimental/pallas/ops/tpu/flash_attention.py`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py)

## Classes
### `BlockSizes`
- def: [`jax/experimental/pallas/ops/tpu/flash_attention.py:52`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L52)
- doc: Tile sizes parameterizing FlashAttention kernels.
- signature: `class BlockSizes:`
- members:
  - `get_default(cls, batch_size, num_heads, q_seq_len, kv_len, d_model)` — [`L113`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L113)
  - `has_backward_blocks(self)` — [`L100`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L100)
  - `verify_major_minor(prefix, suffix, major, minor)` — [`L73`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L73)
  - `block_b` — [`L61`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L61)
  - `block_k` — [`L60`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L60)
  - `block_k_dkv` — [`L65`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L65)
  - `block_k_dq` — [`L69`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L69)
  - `block_k_major` — [`L59`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L59)
  - `block_k_major_dkv` — [`L64`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L64)
  - `block_k_major_dq` — [`L68`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L68)
  - `block_q` — [`L58`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L58)
  - `block_q_dkv` — [`L66`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L66)
  - `block_q_dq` — [`L70`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L70)
  - `block_q_major_dkv` — [`L63`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L63)
- protocol/private: `__post_init__`[`L72`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L72)
- used by: [`_flash_attention_bwd`](flash_attention.md#_flash_attention_bwd), [`flash_attention`](flash_attention.md#flash_attention)

### `SegmentIds`  ·  implements/extends NamedTuple
- def: [`jax/experimental/pallas/ops/tpu/flash_attention.py:34`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L34)
- doc: SegmentIds for Q and KV sequences.
- signature: `class SegmentIds(NamedTuple):`
- members:
  - `kv` — [`L48`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L48)
  - `q` — [`L47`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L47)
- uses (calls/refs, reference-scoped): [`Array`](../../../../_src/basearray.md#Array)
- used by: [`mha_reference_bwd`](flash_attention.md#mha_reference_bwd), [`mha_reference_no_custom_vjp`](flash_attention.md#mha_reference_no_custom_vjp), [`_fwd_cost_estimate`](flash_attention.md#_fwd_cost_estimate), [`mha_reference`](flash_attention.md#mha_reference), [`_mha_reference`](flash_attention.md#_mha_reference), [`_mha_reference_fwd`](flash_attention.md#_mha_reference_fwd)

## Functions
- `_body(start_k)` — [`L387`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L387)
- `_bytes(x: jax.Array | jax.ShapeDtypeStruct)` — [`L560`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L560)
- `_flash_attention(q, k, v, ab, segment_ids, save_residuals, causal, sm_scale, block_sizes, debug)` — [`L205`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L205)
- `_flash_attention_bwd(save_residuals: bool, causal: bool, sm_scale: float, block_sizes: BlockSizes, debug: bool, residuals, do)` — [`L254`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L254) — VJP rule for FlashAttention.
- `_flash_attention_bwd_dkv(q, k, v, ab, segment_ids, l, m, do, di, *, block_q_major: int, block_q: int, block_k_major: int, block_k: int, sm_scale: float, causal: bool = False, mask_value: float = DEFAULT_MASK_VALUE, debug: bool = False)` — [`L941`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L941)
- `_flash_attention_bwd_dq(q, k, v, ab, segment_ids, l, m, do, di, *, block_q_major: int, block_k_major: int, block_k: int, sm_scale: float, causal: bool, mask_value: float, debug: bool)` — [`L1287`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1287)
- `_flash_attention_dkv_kernel(q_tile_ref, k_tile_ref, v_tile_ref, ab_tile_ref, q_segment_ids_tile_ref, kv_segment_ids_tile_ref, l_tile_ref, m_tile_ref, do_tile_ref, di_tile_ref, dk_tile_ref, dv_tile_ref, dk_scratch_ref, dv_scratch_ref, *, sm_scale: float, causal: bool, mask_value: float, q_seq_len: int, block_q: int, block_k: int)` — [`L796`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L796)
- `_flash_attention_dq_kernel(q_tile_ref, k_tile_ref, v_tile_ref, ab_tile_ref, q_segment_ids_tile_ref, kv_segment_ids_tile_ref, l_tile_ref, m_tile_ref, do_tile_ref, di_tile_ref, dq_tile_ref, ds_tile_ref, dq_scratch_ref, *, sm_scale: float, causal: bool, mask_value: float, kv_seq_len: int, block_k: int)` — [`L1146`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1146)
- `_flash_attention_fwd(q, k, v, ab, segment_ids, save_residuals, causal, sm_scale, block_sizes, debug)` — [`L234`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L234)
- `_flash_attention_impl(q, k, v, ab, segment_ids, save_residuals, causal, sm_scale, block_b, block_q, block_k_major, block_k, debug)` — [`L589`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L589)
- `_flash_attention_kernel(q_tile_ref, *args, **kwargs)` — [`L331`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L331)
- `_flash_attention_kernel_single_batch(batch_idx: tuple[int, ...], q_tile_ref, k_tile_ref, v_tile_ref, ab_tile_ref, q_segment_ids_tile_ref, kv_segment_ids_tile_ref, o_tile_ref, l_ref, m_ref, m_scratch_ref, l_scratch_ref, acc_scratch_ref, *, causal, sm_scale, block_k, kv_seq_len, mask_value)` — [`L342`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L342)
- `_flash_attention_kernel_single_batch_single_step(batch_idx: tuple[int, ...], q_tile_ref, k_tile_ref, v_tile_ref, ab_tile_ref, q_segment_ids_tile_ref, kv_segment_ids_tile_ref, o_tile_ref, l_ref: Any | None = None, m_ref: Any | None = None, *, causal, sm_scale, block_k, kv_seq_len, mask_value)` — [`L484`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L484)
- `_fwd_cost_estimate(q: jax.Array, k: jax.Array, v: jax.Array, ab: jax.Array | None, segment_ids: SegmentIds | None, *, causal: bool, sm_scale: jax.Array | None, kernel_inputs_specs, kernel_outputs_specs)` — [`L564`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L564)
- `_mha_reference(q, k, v, ab, segment_ids: SegmentIds | None, causal: bool, mask_value: float, sm_scale: float, save_residuals: bool)` — [`L1554`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1554)
- `_mha_reference_bwd(causal: bool, mask_value: float, sm_scale: float, save_residuals: bool, residuals, do)` — [`L1677`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1677)
- `_mha_reference_fwd(q, k, v, ab, segment_ids: SegmentIds | None, causal: bool, mask_value: float, sm_scale: float, save_residuals: bool)` — [`L1578`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1578)
- `_verify_block(block_name, dim_name, block, dim, should_divide=True)` — [`L1707`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1707)
- `ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L635`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L635)
- `ab_index_map(batch_index, head_index, kv_seq_index, q_seq_index)` — [`L1024`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1024)
- `ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L1363`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1363)
- `below_or_on_diag(r, r_blk_size, c, c_blk_size)` — [`L325`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L325)
- `body(i, _)` — [`L1177`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1177)
- `dkv_index_map(batch_index, head_index, kv_seq_index, _)` — [`L1100`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1100)
- `end_of_kv_sequence()` — [`L1282`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1282)
- `end_of_q_sequence()` — [`L936`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L936)
- `flash_attention(q, k, v, ab=None, segment_ids=None, *, causal: bool = False, sm_scale: float = 1, block_sizes: BlockSizes | None = None, debug: bool = False)` — [`L140`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L140)
- `k_body(i, _)` — [`L832`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L832)
- `kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L622`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L622)
- `kv_index_map(batch_index, head_index, kv_seq_index, _)` — [`L1004`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1004)
- `kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L1331`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1331)
- `kv_segment_ids_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L711`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L711)
- `kv_segment_ids_index_map(batch_index, head_index, kv_seq_index, _)` — [`L1053`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1053)
- `kv_segment_ids_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L1380`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1380)
- `lm_index_map(batch_index, head_index, q_seq_index, _)` — [`L659`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L659)
- `lm_index_map(batch_index, head_index, _, q_seq_index)` — [`L1012`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1012)
- `lm_index_map(batch_index, head_index, q_seq_index, _)` — [`L1351`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1351)
- `mha_reference(q, k, v, ab, segment_ids: SegmentIds | None = None, causal: bool = False, mask_value: float = DEFAULT_MASK_VALUE, sm_scale=1)` — [`L1530`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1530)
- `mha_reference_bwd(q, k, v, ab, segment_ids: SegmentIds | None, o, l, m, do, causal: bool = False, mask_value: float = DEFAULT_MASK_VALUE, sm_scale: float = 1)` — [`L1615`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1615)
- `mha_reference_no_custom_vjp(q, k, v, ab: jax.Array | None = None, segment_ids: SegmentIds | None = None, *, causal: bool = False, mask_value: float = DEFAULT_MASK_VALUE, sm_scale: float = 1, save_residuals: bool = False)` — [`L1482`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1482)
- `o_index_map(batch_index, head_index, q_seq_index, _)` — [`L656`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L656)
- `q_body(j, _)` — [`L830`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L830)
- `q_index_map(batch_index, head_index, q_seq_index, _)` — [`L619`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L619)
- `q_segment_ids_index_map(batch_index, head_index, q_seq_index, _)` — [`L707`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L707)
- `q_segment_ids_index_map(batch_index, head_index, kv_seq_index, q_seq_index)` — [`L1037`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1037)
- `q_segment_ids_index_map(batch_index, head_index, q_seq_index, _)` — [`L1376`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1376)
- `qo_index_map(batch_index, head_index, kv_seq_index, q_seq_index)` — [`L983`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L983)
- `qo_index_map(batch_index, head_index, q_seq_index, _)` — [`L1325`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1325)
- `run()` — [`L385`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L385)
- `run()` — [`L932`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L932)
- `run()` — [`L1273`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1273)
- `start_new_sequence()` — [`L369`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L369)
- `start_new_sequence()` — [`L826`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L826)
- `start_new_sequence()` — [`L1174`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1174)
- `store_output()` — [`L476`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L476)
- `zero_out_ds()` — [`L1277`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L1277)

## Module values
- `DEFAULT_MASK_VALUE` — [`L29`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L29)
- `MIN_BLOCK_SIZE` — [`L321`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L321)
- `NUM_LANES` — [`L30`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L30)
- `NUM_SUBLANES` — [`L31`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L31)
- `TRANS_B_DIM_NUMBERS` — [`L322`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py#L322)

