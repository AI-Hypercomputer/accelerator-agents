---
title: 'Module: ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.ragged_gated_delta_rule_v2._gdn_recurrent_scan_v2`/
symbols:
  inner_kernel.process_prefill.process_transition_prefill: inner_kernel().process_prefill().process_transition_prefill().
  recurrent_scan: recurrent_scan().
  inner_kernel.l2_normalize: inner_kernel().l2_normalize().
  inner_kernel.process_prefill.process_regular_prefill: inner_kernel().process_prefill().process_regular_prefill().
  inner_kernel.process_prefill.process_prefill_dispatch: inner_kernel().process_prefill().process_prefill_dispatch().
  fused_kernel: fused_kernel().
  inner_kernel.decode_wrapper.process_decode.do_work: inner_kernel().decode_wrapper().process_decode().do_work().
  inner_kernel.process_prefill.process_regular_prefill.init_state: inner_kernel().process_prefill().process_regular_prefill().init_state().
  invert_triangular_matrix: invert_triangular_matrix().
  inner_kernel: inner_kernel().
  create_block_specs: create_block_specs().
  fused_kernel._run_with_scratch: fused_kernel()._run_with_scratch().
  inner_kernel.decode_wrapper: inner_kernel().decode_wrapper().
  inner_kernel.process_prefill: inner_kernel().process_prefill().
  get_qkv_index_map_v2: get_qkv_index_map_v2().
  invert_triangular_matrix.local_forward_sub: invert_triangular_matrix().local_forward_sub().
  inner_kernel.decode_wrapper.get_target_idx: inner_kernel().decode_wrapper().get_target_idx().
  inner_kernel.decode_wrapper.process_decode: inner_kernel().decode_wrapper().process_decode().
  inner_kernel.process_prefill.process_regular_prefill.init_state.load_from_hbm: inner_kernel().process_prefill().process_regular_prefill().init_state().load_from_hbm().
  inner_kernel.process_prefill.process_regular_prefill.init_state.zero_state: inner_kernel().process_prefill().process_regular_prefill().init_state().zero_state().
  inner_kernel.process_prefill.process_transition_prefill.do_write: inner_kernel().process_prefill().process_transition_prefill().do_write().
  inner_kernel.process_prefill.process_transition_prefill.load_t_state: inner_kernel().process_prefill().process_transition_prefill().load_t_state().
  inner_kernel.do_stitch: inner_kernel().do_stitch().
  inner_kernel.process_prefill.process_regular_prefill.store_state: inner_kernel().process_prefill().process_regular_prefill().store_state().
  inner_kernel.process_prefill.process_transition_prefill.load_first_state: inner_kernel().process_prefill().process_transition_prefill().load_first_state().
  inner_kernel.process_prefill.process_transition_prefill.do_final_write: inner_kernel().process_prefill().process_transition_prefill().do_final_write().
---
# Module: [`ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py)

## Functions
- `_run_with_scratch(scratch_ref, decode_state_scratch_ref, state_commit_scratch_ref, decode_output_scratch_ref, decode_read_sems, decode_write_sem, prefill_sem)` — [`L1151`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L1151) — Build and run the pipelined inner kernel with allocated scratch.
- `create_block_specs(schedule_table, chunk_size, BT, d, n_v, d_v, alignment=16, sink_offset=0)` — [`L966`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L966) — Build the input/output ``pl.BlockSpec`` lists for the GDN scan kernel.
- `decode_wrapper()` — [`L247`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L247) — Process the decode batch for the current grid step.
- `do_final_write()` — [`L841`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L841) — DMA-store the trailing prefill request's state after the loop.
- `do_stitch()` — [`L885`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L885) — Merge overlapping decode and prefill outputs at the decode boundary.
- `do_work()` — [`L282`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L282) — Run the single-token GDR update for valid lane ``b``.
- `do_write(current_r=current_r, c_slot=c_slot)` — [`L745`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L745) — DMA-store the finished request's state back to HBM.
- `fused_kernel(mixed_qkv_ref, aliased_recurrent_state_ref, state_indices_ref, has_initial_state_ref, a_raw_ref, b_raw_ref, a_log_ref, dt_bias_ref, schedule_table_ref, decode_tokens_ref, total_blocks_ref, recurrent_state_ref, output_ref, *, C: int, BT: int, n_kq: int, n_v: int, d_k: int, d_v: int, use_qk_norm_in_gdn: bool, sublanesize: int, prefill_only: bool)` — [`L1071`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L1071) — Outer Pallas program that drives the fused GDN recurrent scan.
- `get_qkv_index_map_v2(step, schedule_table, valid_col, offset_col, count_col, alignment=16, block_size=64, sink_offset=0)` — [`L922`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L922) — Compute the ``(offset, 0)`` BlockSpec index for a QKV block.
- `get_target_idx(b)` — [`L256`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L256) — Resolve the state-pool slot for decode lane ``b``.
- `init_state()` — [`L448`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L448) — Initialize the prefill recurrent state for the first chunk.
- `inner_kernel(prefill_qkv_ref, decode_qkv_ref, prefill_a_raw_ref, decode_a_raw_ref, prefill_b_raw_ref, decode_b_raw_ref, a_log_ref, dt_bias_ref, prefill_output_ref, decode_output_ref, schedule_table, state_indices, has_initial_state, *, recurrent_state_in, recurrent_state_out, C: int, BT: int, n_kq: int, n_v: int, d_k: int, d_v: int, use_qk_norm_in_gdn: bool, sublanesize: int, prefill_only: bool, prefill_scratch, decode_state_scratch, state_commit_scratch, decode_output_scratch, decode_read_semaphores, decode_write_semaphore, prefill_semaphore, decode_tokens)` — [`L118`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L118) — Per-grid-step body that runs one decode batch or one prefill block.
- `invert_triangular_matrix(A, block_size=16)` — [`L56`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L56) — Inverts a unit lower triangular matrix A block-wise.
- `l2_normalize(x, eps=0.000001)` — [`L229`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L229) — L2-normalize ``x`` along its last axis.
- `load_first_state()` — [`L702`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L702) — DMA-load the initial state for the block's first request.
- `load_from_hbm()` — [`L460`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L460) — DMA-load this request's recurrent state from HBM into scratch.
- `load_t_state(t_req=t_req, t_slot=t_slot)` — [`L775`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L775) — DMA-load the incoming request's initial state into its slot.
- `local_forward_sub(A_mat, b_mat)` — [`L69`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L69) — Solve ``A_mat @ x = b_mat`` for a unit lower triangular diagonal block.
- `process_decode(b, _)` — [`L269`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L269) — ``fori_loop`` body running one decode token's recurrent update.
- `process_prefill()` — [`L421`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L421) — Process the prefill work for the current grid step.
- `process_prefill_dispatch()` — [`L865`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L865) — Dispatch to transition or regular prefill based on the step flag.
- `process_regular_prefill()` — [`L432`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L432) — Chunkwise WY-form GDR reduction for one chunk-aligned prefill block.
- `process_transition_prefill()` — [`L638`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L638) — Token-by-token GDR scan for a sublane block straddling boundaries.
- `recurrent_scan(mixed_qkv: jax.Array, b: jax.Array, a: jax.Array, recurrent_state: jax.Array, A_log: jax.Array, dt_bias: jax.Array, query_start_loc: jax.Array, state_indices: jax.Array, distribution: jax.Array, *, n_kq: int, n_v: int, d_k: int, d_v: int, chunk_size: int = 128, BT: int = 128, use_qk_norm_in_gdn: bool = True, has_initial_state: jax.Array | None = None, prefill_only: bool = False)` — [`L1248`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L1248) — Fused recurrent scan kernel for GDN on TPU v7+.
- `store_state()` — [`L608`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L608) — DMA-store the final prefill state to HBM on the last chunk.
- `zero_state()` — [`L477`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_gdn_recurrent_scan_v2.py#L477) — Zero-initialize the active prefill scratch state (cold start).

