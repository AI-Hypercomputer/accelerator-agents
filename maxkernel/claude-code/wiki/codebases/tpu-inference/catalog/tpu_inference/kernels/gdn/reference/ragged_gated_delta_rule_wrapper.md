---
title: 'Module: tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_wrapper.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_wrapper.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.gdn.reference.ragged_gated_delta_rule_wrapper`/
symbols:
  RaggedGatedDeltaRuleConfig: RaggedGatedDeltaRuleConfig#
  RaggedGatedDeltaRuleConfig.prefill_impl: RaggedGatedDeltaRuleConfig#prefill_impl.
  RaggedGatedDeltaRuleConfig.decode_impl: RaggedGatedDeltaRuleConfig#decode_impl.
  RaggedGatedDeltaRuleConfig.use_qk_norm_in_gdn: RaggedGatedDeltaRuleConfig#use_qk_norm_in_gdn.
  RaggedGatedDeltaRuleImpl: RaggedGatedDeltaRuleImpl#
  RaggedGatedDeltaRuleImpl.REF: RaggedGatedDeltaRuleImpl#REF.
  RaggedGatedDeltaRuleImpl.CHUNKED_JAX_PD: RaggedGatedDeltaRuleImpl#CHUNKED_JAX_PD.
  RaggedGatedDeltaRuleImpl.CHUNKED_KERNEL_PD: RaggedGatedDeltaRuleImpl#CHUNKED_KERNEL_PD.
  RaggedGatedDeltaRuleImpl.CHUNKED_KERNEL_P_RECURRENT_KERNEL_D: RaggedGatedDeltaRuleImpl#CHUNKED_KERNEL_P_RECURRENT_KERNEL_D.
  RaggedGatedDeltaRuleImpl.prefill_impl: RaggedGatedDeltaRuleImpl#prefill_impl().
  RaggedGatedDeltaRuleImpl.decode_impl: RaggedGatedDeltaRuleImpl#decode_impl().
  RaggedGatedDeltaRuleImpl.to_config: RaggedGatedDeltaRuleImpl#to_config().
  ragged_gated_delta_rule_wrapper: ragged_gated_delta_rule_wrapper().
---
# Module: [`tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_wrapper.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_wrapper.py)

## Classes
### `RaggedGatedDeltaRuleConfig`
- def: [`tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_wrapper.py:31`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_wrapper.py#L31)
- signature: `class RaggedGatedDeltaRuleConfig:`
- members:
  - `decode_impl` — [`L33`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_wrapper.py#L33)
  - `prefill_impl` — [`L32`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_wrapper.py#L32)
  - `use_qk_norm_in_gdn` — [`L34`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_wrapper.py#L34)

### `RaggedGatedDeltaRuleImpl`
- def: [`tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_wrapper.py:37`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_wrapper.py#L37)
- doc: Implementation options for the ragged gated delta rule.
- signature: `class RaggedGatedDeltaRuleImpl(enum.Enum):`
- members:
  - `decode_impl(self)` — [`L55`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_wrapper.py#L55)
  - `prefill_impl(self)` — [`L45`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_wrapper.py#L45)
  - `to_config(self)` — [`L64`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_wrapper.py#L64)
  - `CHUNKED_JAX_PD` — [`L40`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_wrapper.py#L40)
  - `CHUNKED_KERNEL_PD` — [`L41`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_wrapper.py#L41)
  - `CHUNKED_KERNEL_P_RECURRENT_KERNEL_D` — [`L42`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_wrapper.py#L42)
  - `REF` — [`L39`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_wrapper.py#L39)

## Functions
- `ragged_gated_delta_rule_wrapper(config: RaggedGatedDeltaRuleConfig, mixed_qkv: jnp.ndarray, b: jnp.ndarray, a: jnp.ndarray, recurrent_state: jnp.ndarray, A_log: jnp.ndarray, dt_bias: jnp.ndarray, query_start_loc: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, has_initial_state: jnp.ndarray, *, chunk_size: int, triangle_solver_impl: triangle_solver.TriangleSolverImpl=triangle_solver.TriangleSolverImpl.GAUSSIAN, n_kq: int, n_v: int, d_k: int, d_v: int)` — [`L85`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_wrapper.py#L85) — Applies the gated delta rule over ragged seq lengths using various implementations.

