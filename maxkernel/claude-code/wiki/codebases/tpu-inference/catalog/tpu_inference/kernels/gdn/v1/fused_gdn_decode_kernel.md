---
title: 'Module: tpu_inference/kernels/gdn/v1/fused_gdn_decode_kernel.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/gdn/v1/fused_gdn_decode_kernel.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.gdn.v1.fused_gdn_decode_kernel`/
symbols:
  fused_decoding_gdn: fused_decoding_gdn().
  _decode_kernel_main: _decode_kernel_main().
  _decode_kernel_main.token_map: _decode_kernel_main().token_map().
  ragged_gated_delta_rule_decode_only: ragged_gated_delta_rule_decode_only().
  get_default_block_sizes: get_default_block_sizes().
  _decode_kernel_main._inner_kernel: _decode_kernel_main()._inner_kernel().
  _decode_kernel_main._first_load: _decode_kernel_main()._first_load().
  _decode_kernel_main._inner_kernel._prefetch: _decode_kernel_main()._inner_kernel()._prefetch().
  _decode_kernel_main._inner_kernel._process_token: _decode_kernel_main()._inner_kernel()._process_token().
  _decode_kernel_main._inner_kernel._wait_prev_store: _decode_kernel_main()._inner_kernel()._wait_prev_store().
  _decode_kernel_main._inner_kernel._start_store: _decode_kernel_main()._inner_kernel()._start_store().
  _decode_kernel_main._drain_other: _decode_kernel_main()._drain_other().
---
# Module: [`tpu_inference/kernels/gdn/v1/fused_gdn_decode_kernel.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_decode_kernel.py)

## Functions
- `_decode_kernel_main(q_hbm, k_hbm, v_hbm, g_hbm, b_hbm, state_indices_ref, a_log_hbm, dt_bias_hbm, distribution_ref, _state_init_ref, o_hbm, state_hbm, h_bufs, h_load_sems, h_store_sems, *, H_qk: int, H_v: int, K: int, V: int, scale: float, use_qk_l2norm: bool, use_gate_in_kernel: bool, lower_bound: float | None, bt: int, apply_silu: bool)` — [`L95`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_decode_kernel.py#L95)
- `_drain_other()` — [`L377`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_decode_kernel.py#L377)
- `_first_load()` — [`L154`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_decode_kernel.py#L154)
- `_inner_kernel(q_ref, k_ref, v_ref, g_ref, b_ref, a_log_ref, dt_bias_ref, o_ref, h_bufs_s, state_indices_s, h_load_sems_s, h_store_sems_s)` — [`L163`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_decode_kernel.py#L163)
- `_prefetch()` — [`L202`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_decode_kernel.py#L202)
- `_process_token()` — [`L254`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_decode_kernel.py#L254)
- `_start_store()` — [`L326`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_decode_kernel.py#L326)
- `_wait_prev_store()` — [`L314`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_decode_kernel.py#L314)
- `fused_decoding_gdn(q: jax.Array, k: jax.Array, v: jax.Array, g: jax.Array, initial_state: jax.Array, state_indices: jax.Array, distribution: jax.Array, b: jax.Array | None, *, scale: float, use_qk_l2norm_in_kernel: bool = False, use_gate_in_kernel: bool = False, A_log: jax.Array | None = None, dt_bias: jax.Array | None = None, lower_bound: float | None = None, apply_silu: bool = False)` — [`L399`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_decode_kernel.py#L399) — Fused recurrent GDN single-step decode.
- `get_default_block_sizes(T: int, H_qk: int, H_v: int, K: int, V: int, dtype, state_dtype, use_gate_in_kernel: bool, has_dt_bias: bool, vmem_bytes_limit: int)` — [`L34`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_decode_kernel.py#L34) — Choose bt to balance pipelining and VMEM utilization to minimize latency
- `ragged_gated_delta_rule_decode_only(mixed_qkv, b, a, recurrent_state, A_log, dt_bias, query_start_loc, state_indices, distribution, has_initial_state=None, *, n_kq, n_v, d_k, d_v, apply_silu=False)` — [`L541`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_decode_kernel.py#L541) — Adapter for decode-only branch matching ragged_gated_delta_rule interface.
- `token_map(i)` — [`L129`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_decode_kernel.py#L129)

