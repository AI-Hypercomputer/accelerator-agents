---
title: 'Module: ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_bwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_bwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.gated_delta_rule._pallas_impl_bwd`/_
symbols:
  _bwd_one_chunk._s: bwd_one_chunk()._s().
  _chunk_gdr_bwd: chunk_gdr_bwd().
  _single_step_gdr_bwd: single_step_gdr_bwd().
  _bwd_one_chunk: bwd_one_chunk().
  _gdr_bwd_grad_kernel: gdr_bwd_grad_kernel().
  _run_bwd_grad_step: run_bwd_grad_step().
  _cast_grad: cast_grad().
  _chunk_gdr_bwd._safe_cast: chunk_gdr_bwd()._safe_cast().
  _run_single_step_backward: run_single_step_backward().
  _chunk_gdr_bwd.grad_step: chunk_gdr_bwd().grad_step().
  _gdr_single_step_bwd_dma_kernel: gdr_single_step_bwd_dma_kernel().
  _gdr_single_step_bwd_kernel: gdr_single_step_bwd_kernel().
---
# Module: [`ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_bwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_bwd.py)

## Functions
- `_bwd_one_chunk(q, k, v, beta, decay, d_out, state_pre, d_state_next, C)` — [`L37`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_bwd.py#L37) — Compute gradients for a single GDR chunk via full re-materialisation.
- `_cast_grad(x, dtype)` — [`L354`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_bwd.py#L354) — Cast a gradient array to ``dtype``, passing ``None`` through unchanged.
- `_chunk_gdr_bwd(chunk_size: int, use_qk_l2norm: bool, res: tuple, g: tuple[Float[Array, ...], Float[Array, ...]])` — [`L370`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_bwd.py#L370) — Compute gradients for the chunked GDR forward pass via reverse scan.
- `_gdr_bwd_grad_kernel(state_pre_ref, q_ref, k_ref, v_ref, beta_ref, decay_ref, d_out_ref, d_state_next_ref, d_state_ref, d_q_ref, d_k_ref, d_v_ref, d_beta_ref, d_decay_ref)` — [`L169`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_bwd.py#L169) — Pallas kernel for one reverse-scan step, processing ``num_fused`` chunks.
- `_gdr_single_step_bwd_dma_kernel(q_ref, k_ref, v_ref, beta_ref, decay_ref, state_prev_ref, d_out_ref, d_state_next_ref, d_q_ref, d_k_ref, d_v_ref, d_beta_ref, d_decay_ref, d_state_ref, state_tile_ref, d_state_next_tile_ref, dma_sem_ref)` — [`L612`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_bwd.py#L612) — DMA-backed Pallas kernel for single-step GDR backward.
- `_gdr_single_step_bwd_kernel(q_ref, k_ref, v_ref, beta_ref, decay_ref, state_prev_ref, d_out_ref, d_state_next_ref, d_q_ref, d_k_ref, d_v_ref, d_beta_ref, d_decay_ref, d_state_ref)` — [`L532`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_bwd.py#L532) — Pallas kernel for single-step GDR backward pass.
- `_run_bwd_grad_step(state_pre, q_i, k_i, v_i, beta_i, decay_i, d_out_i, d_state_next)` — [`L271`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_bwd.py#L271) — Launch the backward gradient Pallas kernel for one fused chunk group.
- `_run_single_step_backward(query, key, value, beta, decay, recurrent_state, d_out, d_state_next)` — [`L721`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_bwd.py#L721) — Launch the single-step backward Pallas kernel.
- `_s(x)` — [`L81`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_bwd.py#L81) — Sanitize an intermediate tensor by replacing NaN/Inf with zero.
- `_safe_cast(x, dtype)` — [`L502`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_bwd.py#L502) — Sanitize a gradient and cast it to the input dtype.
- `_single_step_gdr_bwd(use_qk_l2norm: bool, res: tuple, g: tuple[Float[Array, ...], Float[Array, ...]])` — [`L793`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_bwd.py#L793) — Custom-VJP backward for the single-step GDR path.
- `grad_step(d_state_next, inputs)` — [`L442`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_bwd.py#L442) — Reverse-scan step computing one chunk group's gradients.

