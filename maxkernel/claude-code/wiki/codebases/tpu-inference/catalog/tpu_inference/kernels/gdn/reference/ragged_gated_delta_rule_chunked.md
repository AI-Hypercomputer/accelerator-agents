---
title: 'Module: tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_chunked.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_chunked.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.gdn.reference.ragged_gated_delta_rule_chunked`/
symbols:
  l2norm: l2norm().
  pack_inputs_single_stream: pack_inputs_single_stream().
  ragged_gated_delta_rule_mixed_prefill: ragged_gated_delta_rule_mixed_prefill().
  recurrent_gated_delta_rule_step: recurrent_gated_delta_rule_step().
  ragged_gated_delta_rule_decode_only: ragged_gated_delta_rule_decode_only().
---
# Module: [`tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_chunked.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_chunked.py)

## Functions
- `l2norm(x: jnp.ndarray, dim: int=-1, eps: float=1e-06)` — [`L29`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_chunked.py#L29) — Normalizes x along the specified dimension using L2 norm.
- `pack_inputs_single_stream(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, g: jnp.ndarray, beta: jnp.ndarray, query_start_loc: jnp.ndarray, distribution: jnp.ndarray, chunk_size: int, compute_dtype: jnp.dtype=jnp.bfloat16)` — [`L50`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_chunked.py#L50) — Pads each sequence to multiple of chunk_size and concatenates.
- `ragged_gated_delta_rule_decode_only(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, b_reshaped: jnp.ndarray, a_reshaped: jnp.ndarray, recurrent_state: jnp.ndarray, A_log: jnp.ndarray, dt_bias: jnp.ndarray, query_start_loc: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, use_qk_norm_in_gdn: bool)` — [`L545`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_chunked.py#L545) — Applies gated delta rule for decode-only case (sequence lengths = 1).
- `ragged_gated_delta_rule_mixed_prefill(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, b_reshaped: jnp.ndarray, a_reshaped: jnp.ndarray, A_log: jnp.ndarray, dt_bias: jnp.ndarray, query_start_loc: jnp.ndarray, recurrent_state: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, has_initial_state: jnp.ndarray | None=None, chunk_size: int=64, use_qk_norm_in_gdn: bool=False, compute_dtype: jnp.dtype=jnp.bfloat16, precision: jax.lax.Precision=jax.lax.Precision.HIGHEST, preferred_element_type: jnp.dtype=jnp.float32, triangle_solver_impl: triangle_solver.TriangleSolverImpl=triangle_solver.TriangleSolverImpl.GAUSSIAN)` — [`L201`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_chunked.py#L201) — Applies chunked gated delta rule for mixed prefill case.
- `recurrent_gated_delta_rule_step(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, g: jnp.ndarray, beta: jnp.ndarray, state: jnp.ndarray | None=None)` — [`L488`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_chunked.py#L488) — Single-step recurrent update for decode.

