---
title: 'Module: tpu_inference/kernels/gdn/v2/recurrent_scan_v2.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/gdn/v2/recurrent_scan_v2.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.gdn.v2.recurrent_scan_v2`/
symbols:
  inner_kernel: inner_kernel().
  create_block_specs: create_block_specs().
  fused_kernel: fused_kernel().
  recurrent_scan: recurrent_scan().
  fused_kernel._run_with_scratch: fused_kernel()._run_with_scratch().
  invert_triangular_matrix: invert_triangular_matrix().
  inner_kernel.decode_wrapper: inner_kernel().decode_wrapper().
  inner_kernel.process_prefill: inner_kernel().process_prefill().
  get_qkv_index_map_v2: get_qkv_index_map_v2().
  invert_triangular_matrix.local_forward_sub: invert_triangular_matrix().local_forward_sub().
  inner_kernel.do_stitch: inner_kernel().do_stitch().
---
# Module: [`tpu_inference/kernels/gdn/v2/recurrent_scan_v2.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_v2.py)

## Functions
- `_run_with_scratch(scratch_ref, decode_state_scratch_ref, decode_load_scratch_ref, decode_store_scratch_ref, decode_output_scratch_ref, decode_read_sems, decode_write_sem, prefill_sem)` — [`L455`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_v2.py#L455)
- `create_block_specs(schedule_table, chunk_size, BT, d, n_v, d_v, alignment=16, sink_offset=0)` — [`L331`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_v2.py#L331) — Creates block specs for recurrent scan kernel.
- `decode_wrapper()` — [`L261`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_v2.py#L261)
- `do_stitch()` — [`L283`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_v2.py#L283)
- `fused_kernel(mixed_qkv_ref, aliased_recurrent_state_ref, state_indices_ref, has_initial_state_ref, a_raw_ref, b_raw_ref, a_log_ref, dt_bias_ref, schedule_table_ref, decode_tokens_ref, total_blocks_ref, recurrent_state_ref, output_ref, *, C: int, BT: int, n_kq: int, n_v: int, d_k: int, d_v: int, use_qk_norm_in_gdn: bool, sublanesize: int)` — [`L412`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_v2.py#L412) — Fused kernel for recurrent scan.
- `get_qkv_index_map_v2(step, schedule_table, valid_col, offset_col, alignment=16, block_size=64, sink_offset=0)` — [`L312`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_v2.py#L312)
- `inner_kernel(prefill_qkv_ref, decode_qkv_ref, prefill_a_raw_ref, decode_a_raw_ref, prefill_b_raw_ref, decode_b_raw_ref, a_log_ref, dt_bias_ref, prefill_output_ref, decode_output_ref, schedule_table, state_indices, has_initial_state, *, recurrent_state_in, recurrent_state_out, C: int, BT: int, n_kq: int, n_v: int, d_k: int, d_v: int, use_qk_norm_in_gdn: bool, sublanesize: int, prefill_scratch, decode_state_scratch, state_commit_scratch, decode_load_scratch, decode_store_scratch, decode_output_scratch, decode_read_semaphores, decode_write_semaphore, prefill_semaphore, decode_tokens)` — [`L80`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_v2.py#L80) — Inner kernel for recurrent scan processing both prefill and decode.
- `invert_triangular_matrix(A, block_size=16)` — [`L30`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_v2.py#L30) — Inverts a unit lower triangular matrix A block-wise.
- `local_forward_sub(A_mat, b_mat)` — [`L43`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_v2.py#L43)
- `process_prefill()` — [`L268`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_v2.py#L268)
- `recurrent_scan(mixed_qkv: jax.Array, b: jax.Array, a: jax.Array, recurrent_state: jax.Array, A_log: jax.Array, dt_bias: jax.Array, query_start_loc: jax.Array, state_indices: jax.Array, distribution: jax.Array, *, n_kq: int, n_v: int, d_k: int, d_v: int, chunk_size: int = 128, BT: int = 128, use_qk_norm_in_gdn: bool = True, has_initial_state: jax.Array | None = None, vmem_limit_bytes: int | None = None, race_detect_enable: bool = False)` — [`L555`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_v2.py#L555) — Fused recurrent scan kernel for GDN on TPU v7.

