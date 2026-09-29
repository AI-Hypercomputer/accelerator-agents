---
title: 'Module: ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.flash_mla._pallas_impl_fwd`/
symbols:
  _flash_mla_pallas_call: _flash_mla_pallas_call().
  _flash_mla_kernel_single_batch._run: _flash_mla_kernel_single_batch()._run().
  flash_mla_impl: flash_mla_impl().
  _flash_mla_kernel: _flash_mla_kernel().
  _flash_mla_kernel_single_batch: _flash_mla_kernel_single_batch().
  _flash_mla_pallas_call.kv_index_map: _flash_mla_pallas_call().kv_index_map().
  _flash_mla_pallas_call.bk_index_map: _flash_mla_pallas_call().bk_index_map().
  _flash_mla_pallas_call.bias_index_map: _flash_mla_pallas_call().bias_index_map().
  _flash_mla_fwd: _flash_mla_fwd().
  _flash_mla_bwd: _flash_mla_bwd().
  ROPE_NONE: ROPE_NONE.
  ROPE_FUSED: ROPE_FUSED.
  _flash_mla_kernel_single_batch._run.l_broadcast: _flash_mla_kernel_single_batch()._run().l_broadcast().
  ROPE_DECOUPLED: ROPE_DECOUPLED.
  _flash_mla_pallas_call.w_index_map: _flash_mla_pallas_call().w_index_map().
  _flash_mla_pallas_call.q_index_map: _flash_mla_pallas_call().q_index_map().
  _flash_mla_pallas_call.bq_index_map: _flash_mla_pallas_call().bq_index_map().
  _flash_mla_pallas_call.o_index_map: _flash_mla_pallas_call().o_index_map().
  _flash_mla_pallas_call.lm_out_index_map: _flash_mla_pallas_call().lm_out_index_map().
  _flash_mla_kernel_single_batch._init: _flash_mla_kernel_single_batch()._init().
  _flash_mla_kernel_single_batch._store: _flash_mla_kernel_single_batch()._store().
---
# Module: [`ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py)

## Functions
- `_flash_mla_bwd(rope_mode, causal, softmax_scale, sliding_window, logits_soft_cap, block_b, block_q, block_k, residuals, do)` — [`L600`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py#L600) — Backward rule for custom_vjp: compute gradients via Pallas kernels.
- `_flash_mla_fwd(q, kv_latent, w_kc, w_vc, b_q, b_k, bias, rope_mode, causal, softmax_scale, sliding_window, logits_soft_cap, block_b, block_q, block_k)` — [`L560`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py#L560) — Forward rule for custom_vjp: run Pallas kernel and save residuals.
- `_flash_mla_kernel(q_tile_ref, *args, save_residuals=False, **kwargs)` — [`L64`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py#L64) — Pallas kernel entry-point: iterates over ``block_b`` batch elements.
- `_flash_mla_kernel_single_batch(bi, q_tile_ref, kv_tile_ref, w_kc_tile_ref, w_vc_tile_ref, bq_tile_ref, bk_tile_ref, bias_tile_ref, o_tile_ref, l_out_tile_ref, m_out_tile_ref, m_scratch_ref, l_scratch_ref, acc_scratch_ref, *, save_residuals, rope_mode, d_nope, causal, softmax_scale, sliding_window, logits_soft_cap, block_k, kv_seq_len, mask_value)` — [`L76`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py#L76) — MLA attention kernel for a single batch element within a tile.
- `_flash_mla_pallas_call(q, kv_latent, w_kc, w_vc, b_q, b_k, bias, *, rope_mode, causal, softmax_scale, sliding_window, logits_soft_cap, block_b, block_q, block_k, save_residuals=False)` — [`L298`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py#L298) — Set up grid, BlockSpecs, scratch memory and launch the Pallas kernel.
- `_init()` — [`L156`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py#L156)
- `_run()` — [`L167`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py#L167)
- `_store()` — [`L291`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py#L291)
- `bias_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L396`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py#L396)
- `bk_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L384`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py#L384)
- `bq_index_map(batch_index, head_index, q_seq_index, _)` — [`L380`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py#L380)
- `flash_mla_impl(q, kv_latent, w_kc, w_vc, b_q, b_k, bias, rope_mode, causal, softmax_scale, sliding_window, logits_soft_cap, block_b, block_q, block_k)` — [`L494`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py#L494) — Flash MLA forward with custom VJP for gradient computation.
- `kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L364`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py#L364)
- `l_broadcast(val)` — [`L267`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py#L267)
- `lm_out_index_map(batch_index, head_index, q_seq_index, _)` — [`L427`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py#L427)
- `o_index_map(batch_index, head_index, q_seq_index, _)` — [`L414`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py#L414)
- `q_index_map(batch_index, head_index, q_seq_index, _)` — [`L361`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py#L361)
- `w_index_map(batch_index, head_index, q_seq_index, kv_seq_index)` — [`L375`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py#L375)

## Module values
- `ROPE_DECOUPLED` — [`L61`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py#L61)
- `ROPE_FUSED` — [`L60`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py#L60)
- `ROPE_NONE` — [`L59`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py#L59)

