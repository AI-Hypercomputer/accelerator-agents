---
title: 'Module: ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.flash_attention._pallas_impl_bwd`/_flash_attention_
symbols:
  _flash_attention_bwd_dkv: bwd_dkv().
  _flash_attention_bwd: bwd().
  _flash_attention_bwd_dq: bwd_dq().
  _flash_attention_dkv_kernel.q_body.k_body: dkv_kernel().q_body().k_body().
  _flash_attention_dq_kernel.body: dq_kernel().body().
  _flash_attention_bwd_dkv.qo_index_map: bwd_dkv().qo_index_map().
  _flash_attention_dkv_kernel: dkv_kernel().
  _flash_attention_dkv_kernel.q_body: dkv_kernel().q_body().
  _flash_attention_bwd_dkv.q_segment_ids_index_map: bwd_dkv().q_segment_ids_index_map().
  _flash_attention_dq_kernel: dq_kernel().
  _flash_attention_bwd_dq.kv_index_map: bwd_dq().kv_index_map().
  _flash_attention_bwd_dq.kv_segment_ids_index_map: bwd_dq().kv_segment_ids_index_map().
  _flash_attention_bwd_dq.qo_index_map: bwd_dq().qo_index_map().
  _flash_attention_dkv_kernel.run: dkv_kernel().run().
  _flash_attention_dq_kernel.run: dq_kernel().run().
  _flash_attention_bwd_dkv.kv_index_map: bwd_dkv().kv_index_map().
  _flash_attention_bwd_dkv.lm_index_map: bwd_dkv().lm_index_map().
  _flash_attention_bwd_dkv.ab_index_map: bwd_dkv().ab_index_map().
  _flash_attention_bwd_dkv.kv_segment_ids_index_map: bwd_dkv().kv_segment_ids_index_map().
  _flash_attention_bwd_dkv.dkv_index_map: bwd_dkv().dkv_index_map().
  _flash_attention_bwd_dq.lm_index_map: bwd_dq().lm_index_map().
  _flash_attention_bwd_dq.ab_index_map: bwd_dq().ab_index_map().
  _flash_attention_bwd_dq.q_segment_ids_index_map: bwd_dq().q_segment_ids_index_map().
  _flash_attention_dkv_kernel.start_new_sequence: dkv_kernel().start_new_sequence().
  _flash_attention_dkv_kernel.end_of_q_sequence: dkv_kernel().end_of_q_sequence().
  _flash_attention_dq_kernel.start_new_sequence: dq_kernel().start_new_sequence().
  _flash_attention_dq_kernel.zero_out_ds: dq_kernel().zero_out_ds().
  _flash_attention_dq_kernel.end_of_kv_sequence: dq_kernel().end_of_kv_sequence().
---
# Module: [`ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py)

## Functions
- `_flash_attention_bwd(save_residuals: bool, causal: bool, softmax_scale: float, block_sizes: BlockSizes, sliding_window, logits_soft_cap, residuals, do)` — [`L828`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L828) — VJP backward rule for Flash Attention.
- `_flash_attention_bwd_dkv(q, k, v, ab, segment_ids, l, m, do, di, *, block_q_major: int | None, block_q: int | None, block_k_major: int | None, block_k: int | None, softmax_scale: float, sliding_window, logits_soft_cap, causal: bool = False, mask_value: float = DEFAULT_MASK_VALUE)` — [`L241`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L241) — Compute gradients with respect to keys and values.
- `_flash_attention_bwd_dq(q, k, v, ab, segment_ids, l, m, do, di, *, block_q_major: int | None, block_k_major: int | None, block_k: int | None, softmax_scale: float, sliding_window, logits_soft_cap, causal: bool, mask_value: float)` — [`L624`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L624) — Compute gradients with respect to queries and attention bias.
- `_flash_attention_dkv_kernel(q_tile_ref, k_tile_ref, v_tile_ref, ab_tile_ref, q_segment_ids_tile_ref, kv_segment_ids_tile_ref, l_tile_ref, m_tile_ref, do_tile_ref, di_tile_ref, dk_tile_ref, dv_tile_ref, dk_scratch_ref, dv_scratch_ref, *, softmax_scale: float, sliding_window, logits_soft_cap, causal: bool, mask_value: float, q_seq_len: int, block_q: int, block_k: int)` — [`L74`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L74) — Pallas kernel for computing dK and dV gradients.
- `_flash_attention_dq_kernel(q_tile_ref, k_tile_ref, v_tile_ref, ab_tile_ref, q_segment_ids_tile_ref, kv_segment_ids_tile_ref, l_tile_ref, m_tile_ref, do_tile_ref, di_tile_ref, dq_tile_ref, ds_tile_ref, dq_scratch_ref, *, softmax_scale: float, sliding_window, logits_soft_cap, causal: bool, mask_value: float, kv_seq_len: int, block_k: int)` — [`L458`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L458) — Pallas kernel for computing dQ gradients.
- `ab_index_map(batch_index, head_index, kv_seq_index, q_seq_index)` — [`L348`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L348)
- `ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L724`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L724)
- `body(i, _)` — [`L522`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L522)
- `dkv_index_map(batch_index, head_index, kv_seq_index, _)` — [`L411`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L411)
- `end_of_kv_sequence()` — [`L619`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L619)
- `end_of_q_sequence()` — [`L236`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L236)
- `k_body(i, _)` — [`L147`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L147)
- `kv_index_map(batch_index, head_index, kv_seq_index, _)` — [`L328`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L328)
- `kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L696`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L696)
- `kv_segment_ids_index_map(batch_index, head_index, kv_seq_index, _)` — [`L369`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L369)
- `kv_segment_ids_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L737`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L737)
- `lm_index_map(batch_index, head_index, _, q_seq_index)` — [`L336`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L336)
- `lm_index_map(batch_index, head_index, q_seq_index, _)` — [`L712`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L712)
- `q_body(j, _)` — [`L144`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L144)
- `q_segment_ids_index_map(batch_index, head_index, kv_seq_index, q_seq_index)` — [`L357`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L357)
- `q_segment_ids_index_map(batch_index, head_index, q_seq_index, _)` — [`L733`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L733)
- `qo_index_map(batch_index, head_index, kv_seq_index, q_seq_index)` — [`L310`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L310)
- `qo_index_map(batch_index, head_index, q_seq_index, _)` — [`L690`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L690)
- `run()` — [`L232`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L232)
- `run()` — [`L610`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L610)
- `start_new_sequence()` — [`L140`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L140)
- `start_new_sequence()` — [`L519`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L519)
- `zero_out_ds()` — [`L614`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L614)

