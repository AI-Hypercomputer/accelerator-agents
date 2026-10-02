---
title: 'Module: tpu_inference/kernels/gdn/v1/fused_gdn_kernel_wrapper.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/gdn/v1/fused_gdn_kernel_wrapper.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.gdn.v1.fused_gdn_kernel_wrapper`/
symbols:
  _dispatch_with_distribution: _dispatch_with_distribution().
  fused_gdn: fused_gdn().
  ragged_gated_delta_rule: ragged_gated_delta_rule().
---
# Module: [`tpu_inference/kernels/gdn/v1/fused_gdn_kernel_wrapper.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_kernel_wrapper.py)

## Functions
- `_dispatch_with_distribution(q, k, v, cu_seqlens, g, initial_state, state_indices, b, has_initial_state, *, scale, use_qk_l2norm, use_gate_in_kernel, A_log, dt_bias, lower_bound, distribution)` — [`L30`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_kernel_wrapper.py#L30) — Dispatch to decode and recurrent kernels following the RPA pattern.
- `fused_gdn(q: jax.Array, k: jax.Array, v: jax.Array, cu_seqlens: jax.Array, g: jax.Array, initial_state: jax.Array, state_indices: jax.Array, distribution: jax.Array, b: jax.Array | None = None, has_initial_state: jax.Array | None = None, scale: float | None = None, use_qk_l2norm_in_kernel: bool = False, use_gate_in_kernel: bool = False, A_log: jax.Array | None = None, dt_bias: jax.Array | None = None, lower_bound: float | None = None)` — [`L113`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_kernel_wrapper.py#L113) — Fused recurrent GDN forward pass.
- `ragged_gated_delta_rule(mixed_qkv, b, a, recurrent_state, A_log, dt_bias, query_start_loc, state_indices, distribution, has_initial_state=None, *, n_kq, n_v, d_k, d_v)` — [`L236`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_kernel_wrapper.py#L236) — Adapter matching the ragged_gated_delta_rule_{ref,chunked} interface.

