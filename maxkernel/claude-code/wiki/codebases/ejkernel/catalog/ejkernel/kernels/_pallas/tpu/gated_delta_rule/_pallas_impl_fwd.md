---
title: 'Module: ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.gated_delta_rule._pallas_impl_fwd`/_
symbols:
  _dot: dot().
  _chunk_blockspec: chunk_blockspec().
  _run_phase1.bs3: run_phase1().bs3().
  _chunk_gdr_fwd_core: chunk_gdr_fwd_core().
  _run_phase1_indexed_grouped.bs_v: run_phase1_indexed_grouped().bs_v().
  _phase1_kernel_train_cumsum: phase1_kernel_train_cumsum().
  _run_phase1: run_phase1().
  _run_phase1_indexed_grouped: run_phase1_indexed_grouped().
  _chunk_gdr_grouped_fwd_core: chunk_gdr_grouped_fwd_core().
  _chunk_gdr_grouped_fwd_rule: chunk_gdr_grouped_fwd_rule().
  _exact_strict_lower_inv_block: exact_strict_lower_inv_block().
  _neumann_inv: neumann_inv().
  _process_one_chunk: process_one_chunk().
  _phase1_kernel_fwd_cumsum: phase1_kernel_fwd_cumsum().
  _single_step_gdr_fwd_impl: single_step_gdr_fwd_impl().
  _chunk_gdr_grouped_bwd_rule: chunk_gdr_grouped_bwd_rule().
  _chunk_gdr_fwd_rule: chunk_gdr_fwd_rule().
  _run_single_step_forward: run_single_step_forward().
  _single_step_gdr_fwd: single_step_gdr_fwd().
  _phase2_scan_body._s: phase2_scan_body()._s().
  _chunk_gdr_grouped_fwd_pallas_chunk: chunk_gdr_grouped_fwd_pallas_chunk().
  _chunk_gdr_grouped_fwd: chunk_gdr_grouped_fwd().
  _chunk_gdr_fwd_pallas_chunk: chunk_gdr_fwd_pallas_chunk().
  _chunk_gdr_fwd: chunk_gdr_fwd().
  _N_FUSE: N_FUSE.
  _exact_strict_lower_inv: exact_strict_lower_inv().
  _phase2_scan_body: phase2_scan_body().
  _chunk_gdr_grouped_fwd_impl: chunk_gdr_grouped_fwd_impl().
  _chunk_gdr_fwd_impl: chunk_gdr_fwd_impl().
  _chunk_gdr_bwd_rule: chunk_gdr_bwd_rule().
  _single_step_gdr_fwd_rule: single_step_gdr_fwd_rule().
  _single_step_gdr_bwd_rule: single_step_gdr_bwd_rule().
  _phase1_kernel_train_cumsum._s: phase1_kernel_train_cumsum()._s().
  _run_phase1_indexed_grouped.bs_qk: run_phase1_indexed_grouped().bs_qk().
  _phase2_scan_body_infer: phase2_scan_body_infer().
  _repeat_grouped_heads: repeat_grouped_heads().
  _sum_grouped_head_grads: sum_grouped_head_grads().
  _cast_custom_vjp_primal_output: cast_custom_vjp_primal_output().
  _P: P.
  _exact_strict_lower_inv_rows: exact_strict_lower_inv_rows().
  _gdr_single_step_fwd_dma_kernel: gdr_single_step_fwd_dma_kernel().
  _gdr_single_step_fwd_kernel: gdr_single_step_fwd_kernel().
---
# Module: [`ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py)

## Functions
- `_cast_custom_vjp_primal_output(output, input_dtype, use_input_dtype_phase1_outputs, use_input_dtype_state)` — [`L732`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L732) — Match the custom-VJP fwd-rule output dtype to the decorated primal.
- `_chunk_blockspec(shape: tuple[int, ...])` — [`L48`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L48) — Create a Pallas BlockSpec indexed by ``(batch, head)`` with remaining axes at 0.
- `_chunk_gdr_bwd_rule(chunk_size, use_qk_l2norm, use_input_dtype_phase1_outputs, use_input_dtype_state, res, g)` — [`L1098`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L1098) — Custom-VJP backward rule: delegates to ``_chunk_gdr_bwd`` in ``_pallas_impl_bwd``.
- `_chunk_gdr_fwd(query: Float[Array, "batch num_heads seq_len head_dim"], key: Float[Array, "batch num_heads seq_len head_dim"], value: Float[Array, "batch num_heads seq_len d_state"], beta: Float[Array, "batch num_heads seq_len"], decay: Float[Array, "batch num_heads seq_len"] | None, chunk_size: int = 64, initial_state: Float[Array, "batch num_heads head_dim d_state"] | None = None, use_qk_l2norm: bool = True, use_input_dtype_phase1_outputs: bool = False, use_input_dtype_state: bool = False)` — [`L1108`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L1108) — Multi-token chunked GDR forward path.
- `_chunk_gdr_fwd_core(query, key, value, beta, decay, chunk_size, initial_state, use_qk_l2norm, *, save_residual: bool, inference: bool = False, use_input_dtype_phase1_outputs: bool = False, use_input_dtype_state: bool = False)` — [`L563`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L563) — Two-phase chunked GDR forward pass (shared by training and inference).
- `_chunk_gdr_fwd_impl(query, key, value, beta, decay, chunk_size, initial_state, use_qk_l2norm, use_input_dtype_phase1_outputs, use_input_dtype_state)` — [`L996`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L996) — Inference-only wrapper: calls ``_chunk_gdr_fwd_core`` without saving residuals.
- `_chunk_gdr_fwd_pallas_chunk(query: Float[Array, "batch num_heads seq_len head_dim"], key: Float[Array, "batch num_heads seq_len head_dim"], value: Float[Array, "batch num_heads seq_len d_state"], beta: Float[Array, "batch num_heads seq_len"], decay: Float[Array, "batch num_heads seq_len"] | None, chunk_size: int = 64, initial_state: Float[Array, "batch num_heads head_dim d_state"] | None = None, use_qk_l2norm: bool = True, use_input_dtype_phase1_outputs: bool = False, use_input_dtype_state: bool = False)` — [`L1027`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L1027) — Chunked forward pass for GDR on TPU via 2-phase Pallas kernel.
- `_chunk_gdr_fwd_rule(query, key, value, beta, decay, chunk_size, initial_state, use_qk_l2norm, use_input_dtype_phase1_outputs, use_input_dtype_state)` — [`L1057`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L1057) — Custom-VJP forward rule: training mode — runs Phase 1+2 and saves residuals.
- `_chunk_gdr_grouped_bwd_rule(chunk_size, use_qk_l2norm, use_input_dtype_phase1_outputs, use_input_dtype_state, res, g)` — [`L935`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L935) — Custom-VJP backward rule: repeated-head backward, then reduce Q/K grads.
- `_chunk_gdr_grouped_fwd(query: Float[Array, "batch num_key_heads seq_len head_dim"], key: Float[Array, "batch num_key_heads seq_len head_dim"], value: Float[Array, "batch num_value_heads seq_len d_state"], beta: Float[Array, "batch num_value_heads seq_len"], decay: Float[Array, "batch num_value_heads seq_len"] | None, chunk_size: int = 64, initial_state: Float[Array, "batch num_value_heads head_dim d_state"] | None = None, use_qk_l2norm: bool = True, use_input_dtype_phase1_outputs: bool = False, use_input_dtype_state: bool = False)` — [`L966`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L966) — Multi-token grouped-head chunked GDR forward path.
- `_chunk_gdr_grouped_fwd_core(query, key, value, beta, decay, chunk_size, initial_state, use_qk_l2norm, *, use_input_dtype_phase1_outputs: bool = False, use_input_dtype_state: bool = False)` — [`L739`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L739) — Forward-only grouped-head chunked GDR.
- `_chunk_gdr_grouped_fwd_impl(query, key, value, beta, decay, chunk_size, initial_state, use_qk_l2norm, use_input_dtype_phase1_outputs, use_input_dtype_state)` — [`L832`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L832) — Inference-only grouped wrapper: grouped Phase 1 + existing Phase 2.
- `_chunk_gdr_grouped_fwd_pallas_chunk(query: Float[Array, "batch num_key_heads seq_len head_dim"], key: Float[Array, "batch num_key_heads seq_len head_dim"], value: Float[Array, "batch num_value_heads seq_len d_state"], beta: Float[Array, "batch num_value_heads seq_len"], decay: Float[Array, "batch num_value_heads seq_len"] | None, chunk_size: int = 64, initial_state: Float[Array, "batch num_value_heads head_dim d_state"] | None = None, use_qk_l2norm: bool = True, use_input_dtype_phase1_outputs: bool = False, use_input_dtype_state: bool = False)` — [`L860`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L860) — Grouped-head chunked forward pass for GDR on TPU.
- `_chunk_gdr_grouped_fwd_rule(query, key, value, beta, decay, chunk_size, initial_state, use_qk_l2norm, use_input_dtype_phase1_outputs, use_input_dtype_state)` — [`L890`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L890) — Custom-VJP forward rule using the existing repeated-head residual path.
- `_dot(a, b)` — [`L43`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L43) — 2-D matrix multiply with the module-level ``_P`` precision setting.
- `_exact_strict_lower_inv(A, C, strict_lower=None, lower_mask=None)` — [`L93`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L93) — Compute ``(I - A)^{-1}`` for strict-lower ``A`` exactly.
- `_exact_strict_lower_inv_block(A, C)` — [`L76`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L76) — Recursively invert ``I - A`` for strict-lower ``A`` using block matmuls.
- `_exact_strict_lower_inv_rows(A, C, strict_lower=None, lower_mask=None)` — [`L53`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L53) — Compute ``(I - A)^{-1}`` for small strict-lower matrices row by row.
- `_gdr_single_step_fwd_dma_kernel(q_ref, k_ref, v_ref, beta_ref, decay_ref, state_ref, out_ref, final_state_ref, state_tile_ref, dma_sem_ref)` — [`L1213`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L1213) — DMA-backed Pallas kernel for one GDR decode step (seq_len == 1).
- `_gdr_single_step_fwd_kernel(q_ref, k_ref, v_ref, beta_ref, decay_ref, state_ref, out_ref, final_state_ref)` — [`L1163`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L1163) — Pallas kernel for one GDR decode step (seq_len == 1), non-DMA variant.
- `_neumann_inv(A, C, strict_lower=None, lower_mask=None)` — [`L123`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L123) — Compatibility wrapper for the exact strict-lower inverse.
- `_phase1_kernel_fwd_cumsum(q_ref, k_ref, v_ref, beta_ref, g_cumsum_ref, value_local_ref, k_cumdecay_ref, attn_qk_ref, q_scaled_ref, k_scaled_ref, g_end_exp_ref)` — [`L183`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L183) — Phase 1 forward kernel that consumes precomputed cumulative decay.
- `_phase1_kernel_train_cumsum(q_ref, k_ref, v_ref, beta_ref, g_cumsum_ref, value_local_ref, k_cumdecay_ref, attn_qk_ref, q_scaled_ref, k_scaled_ref, g_end_exp_ref, attn_inv_ref)` — [`L240`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L240) — Training Phase 1 kernel that consumes precomputed cumulative decay.
- `_phase2_scan_body(state, inputs)` — [`L497`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L497) — Phase 2 scan body (training mode): 4 matmuls + element-wise ops.
- `_phase2_scan_body_infer(state, inputs)` — [`L532`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L532) — Phase 2 scan body (inference mode): faster variant without nan-guards.
- `_process_one_chunk(q, k, v, beta, decay, state, C)` — [`L133`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L133) — Run the GDR recurrence on a single chunk of length C.
- `_repeat_grouped_heads(x, repeats: int)` — [`L720`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L720) — Repeat grouped key/query heads to match value-head layout.
- `_run_phase1(query_c, key_c, value_c, beta_c, decay_c, *, inference=False, use_input_dtype_outputs: bool = False)` — [`L299`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L299) — Launch the Phase 1 Pallas kernel over ALL chunks simultaneously.
- `_run_phase1_indexed_grouped(query_c, key_c, value_c, beta_c, decay_c, *, expand_ratio: int, use_input_dtype_outputs: bool = False)` — [`L429`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L429) — Launch Phase 1 with value-head parallelism and grouped Q/K indexing.
- `_run_single_step_forward(query, key, value, beta, decay, recurrent_state)` — [`L1282`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L1282) — Launch the single-step GDR Pallas kernel and return (output, new_state).
- `_s(x)` — [`L286`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L286) — Replace NaN/+Inf/-Inf entries in ``x`` with 0.0 before it is written out.
- `_s(x)` — [`L517`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L517) — Replace NaN/+Inf/-Inf entries in ``x`` with 0.0 for stable scan gradients.
- `_single_step_gdr_bwd_rule(use_qk_l2norm, res, g)` — [`L1399`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L1399) — Custom-VJP backward rule for single-step GDR: delegates to Pallas bwd kernel.
- `_single_step_gdr_fwd(query, key, value, beta, decay, recurrent_state, use_qk_l2norm=True)` — [`L1370`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L1370) — Single-step GDR forward with custom VJP for gradient support.
- `_single_step_gdr_fwd_impl(query, key, value, beta, decay, recurrent_state, use_qk_l2norm)` — [`L1328`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L1328) — Shared forward computation for the single-step GDR path (forward + backward rule).
- `_single_step_gdr_fwd_rule(query, key, value, beta, decay, recurrent_state, use_qk_l2norm)` — [`L1391`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L1391) — Custom-VJP forward rule for single-step GDR: runs and saves residuals.
- `_sum_grouped_head_grads(x, num_key_heads: int, expand_ratio: int)` — [`L725`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L725) — Collapse repeated-head Q/K gradients back to grouped-head layout.
- `bs3(shape)` — [`L338`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L338) — Build a per-chunk BlockSpec of block shape ``(1, 1, 1, *shape)``.
- `bs_qk(shape)` — [`L452`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L452)
- `bs_v(shape)` — [`L458`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L458)

## Module values
- `_N_FUSE` — [`L40`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L40)
- `_P` — [`L39`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L39)

