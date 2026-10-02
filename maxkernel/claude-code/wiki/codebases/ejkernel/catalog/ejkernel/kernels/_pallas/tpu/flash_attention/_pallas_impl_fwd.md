---
title: 'Module: ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.flash_attention._pallas_impl_fwd`/_flash_attention_
symbols:
  _flash_attention_impl: impl().
  _flash_attention_kernel_single_batch.run._body: kernel_single_batch().run()._body().
  _flash_attention_kernel: kernel().
  _flash_attention_kernel_single_batch_single_step: kernel_single_batch_single_step().
  _flash_attention_fwd: fwd().
  _flash_attention_impl.kv_index_map: impl().kv_index_map().
  _flash_attention_kernel_single_batch: kernel_single_batch().
  _flash_attention_impl.ab_index_map: impl().ab_index_map().
  _flash_attention_impl.kv_segment_ids_index_map: impl().kv_segment_ids_index_map().
  _flash_attention_kernel_single_batch.run._body.l_broadcast: kernel_single_batch().run()._body().l_broadcast().
  _flash_attention_impl.lm_index_map: impl().lm_index_map().
  _flash_attention_impl.q_index_map: impl().q_index_map().
  _flash_attention_impl.o_index_map: impl().o_index_map().
  _flash_attention_impl.q_segment_ids_index_map: impl().q_segment_ids_index_map().
  _flash_attention_kernel_single_batch.start_new_sequence: kernel_single_batch().start_new_sequence().
  _flash_attention_kernel_single_batch.run: kernel_single_batch().run().
  _flash_attention_kernel_single_batch.store_output: kernel_single_batch().store_output().
---
# Module: [`ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_fwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_fwd.py)

## Functions
- `_body(start_k)` — [`L243`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_fwd.py#L243)
- `_flash_attention_fwd(q, k, v, ab, segment_ids, save_residuals, causal, softmax_scale, block_sizes, sliding_window, logits_soft_cap)` — [`L83`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_fwd.py#L83) — Forward pass entry point for Flash Attention with residual saving.
- `_flash_attention_impl(q, k, v, ab, segment_ids, save_residuals, causal, softmax_scale, sliding_window, logits_soft_cap, block_b, block_q, block_k_major, block_k)` — [`L444`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_fwd.py#L444) — Core Flash Attention implementation using Pallas TPU kernels.
- `_flash_attention_kernel(q_tile_ref, *args, **kwargs)` — [`L140`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_fwd.py#L140) — Pallas kernel dispatcher for Flash Attention forward pass.
- `_flash_attention_kernel_single_batch(batch_idx: tuple[int, ...], q_tile_ref, k_tile_ref, v_tile_ref, ab_tile_ref, q_segment_ids_tile_ref, kv_segment_ids_tile_ref, o_tile_ref, l_ref, m_ref, m_scratch_ref, l_scratch_ref, acc_scratch_ref, *, causal, softmax_scale, sliding_window, logits_soft_cap, block_k, kv_seq_len, mask_value)` — [`L167`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_fwd.py#L167) — Multi-step Flash Attention kernel for a single batch element.
- `_flash_attention_kernel_single_batch_single_step(batch_idx: tuple[int, ...], q_tile_ref, k_tile_ref, v_tile_ref, ab_tile_ref, q_segment_ids_tile_ref, kv_segment_ids_tile_ref, o_tile_ref, l_ref: Any | None = None, m_ref: Any | None = None, *, causal, softmax_scale, sliding_window, logits_soft_cap, block_k, kv_seq_len, mask_value)` — [`L336`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_fwd.py#L336) — Single-step Flash Attention kernel for short KV sequences.
- `ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L523`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_fwd.py#L523)
- `kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L512`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_fwd.py#L512)
- `kv_segment_ids_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L590`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_fwd.py#L590)
- `l_broadcast(l)` — [`L307`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_fwd.py#L307)
- `lm_index_map(batch_index, head_index, q_seq_index, _)` — [`L542`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_fwd.py#L542)
- `o_index_map(batch_index, head_index, q_seq_index, _)` — [`L539`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_fwd.py#L539)
- `q_index_map(batch_index, head_index, q_seq_index, _)` — [`L509`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_fwd.py#L509)
- `q_segment_ids_index_map(batch_index, head_index, q_seq_index, _)` — [`L586`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_fwd.py#L586)
- `run()` — [`L241`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_fwd.py#L241)
- `start_new_sequence()` — [`L229`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_fwd.py#L229)
- `store_output()` — [`L328`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_fwd.py#L328)

