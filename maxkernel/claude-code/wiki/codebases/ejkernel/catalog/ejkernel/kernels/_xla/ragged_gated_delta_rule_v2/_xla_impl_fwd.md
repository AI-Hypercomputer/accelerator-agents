---
title: 'Module: ejkernel/kernels/_xla/ragged_gated_delta_rule_v2/_xla_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_xla/ragged_gated_delta_rule_v2/_xla_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._xla.ragged_gated_delta_rule_v2._xla_impl_fwd`/
symbols:
  ragged_gated_delta_rule_mixed_prefill: ragged_gated_delta_rule_mixed_prefill().
  ragged_gated_delta_rule: ragged_gated_delta_rule().
  l2norm: l2norm().
  ragged_gated_delta_rule_decode_only: ragged_gated_delta_rule_decode_only().
  ragged_gated_delta_rule.decode_only_branch: ragged_gated_delta_rule().decode_only_branch().
  ragged_gated_delta_rule.mixed_prefill_branch: ragged_gated_delta_rule().mixed_prefill_branch().
  ragged_gated_delta_rule_mixed_prefill.to_chunk: ragged_gated_delta_rule_mixed_prefill().to_chunk().
  TriangleSolverImpl.__call__: TriangleSolverImpl#__call__().
  ragged_gated_delta_rule_v2: ragged_gated_delta_rule_v2().
  TriangleSolverImpl: TriangleSolverImpl#
  ragged_gated_delta_rule_mixed_prefill.to_chunk_scalar: ragged_gated_delta_rule_mixed_prefill().to_chunk_scalar().
  triangular_inverse_jax: triangular_inverse_jax().
  TriangleSolverImpl.JAX: TriangleSolverImpl#JAX.
  pack_inputs_single_stream: pack_inputs_single_stream().
  ragged_gated_delta_rule_mixed_prefill.scan_body: ragged_gated_delta_rule_mixed_prefill().scan_body().
  _reorder_concatenated_tensor_for_sharding: _reorder_concatenated_tensor_for_sharding().
  newton_schulz_inverse_ref: newton_schulz_inverse_ref().
  recurrent_gated_delta_rule_step: recurrent_gated_delta_rule_step().
---
# Module: [`ejkernel/kernels/_xla/ragged_gated_delta_rule_v2/_xla_impl_fwd.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule_v2/_xla_impl_fwd.py)

## Classes
### `TriangleSolverImpl`  ·  implements/extends StrEnum
- def: [`ejkernel/kernels/_xla/ragged_gated_delta_rule_v2/_xla_impl_fwd.py:165`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule_v2/_xla_impl_fwd.py#L165)
- doc: Selector for the unit lower-triangular inverse implementation.
- signature: `class TriangleSolverImpl(enum.StrEnum):`
- members:
  - `__call__(self, A)` — [`L180`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule_v2/_xla_impl_fwd.py#L180) — Invoke the selected unit lower-triangular inverse implementation.
  - `JAX` — [`L178`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule_v2/_xla_impl_fwd.py#L178)
- uses (calls/refs, reference-scoped): [`triangular_inverse_jax`](_xla_impl_fwd.md#triangular_inverse_jax)
- used by: [`ragged_gated_delta_rule_mixed_prefill`](_xla_impl_fwd.md#ragged_gated_delta_rule_mixed_prefill)

## Functions
- `_reorder_concatenated_tensor_for_sharding(concatenated_tensor: jax.Array, split_sizes: tuple[int, ...], n_shards: int, dim: int)` — [`L52`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule_v2/_xla_impl_fwd.py#L52) — Reorder a fused feature axis so per-shard slices are interleaved.
- `decode_only_branch(_)` — [`L951`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule_v2/_xla_impl_fwd.py#L951) — Run the decode-only fast path under :func:`lax.cond`.
- `l2norm(x: jnp.ndarray, dim: int = -1, eps: float = 0.000001)` — [`L194`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule_v2/_xla_impl_fwd.py#L194) — Normalize ``x`` to unit L2 norm along ``dim``.
- `mixed_prefill_branch(_)` — [`L981`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule_v2/_xla_impl_fwd.py#L981) — Run the chunked mixed-prefill path under :func:`lax.cond`.
- `newton_schulz_inverse_ref(A, n=None)` — [`L100`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule_v2/_xla_impl_fwd.py#L100) — Reference Newton-Schulz inverse for unit lower-triangular matrices.
- `pack_inputs_single_stream(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, g: jnp.ndarray, beta: jnp.ndarray, query_start_loc: jnp.ndarray, distribution: jnp.ndarray, chunk_size: int, compute_dtype: jnp.dtype = jnp.bfloat16)` — [`L217`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule_v2/_xla_impl_fwd.py#L217) — Pads each sequence to multiple of chunk_size and concatenates.
- `ragged_gated_delta_rule(mixed_qkv: jnp.ndarray, b: jnp.ndarray, a: jnp.ndarray, recurrent_state: jnp.ndarray, A_log: jnp.ndarray, dt_bias: jnp.ndarray, query_start_loc: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, has_initial_state: jnp.ndarray | None = None, *, n_kq: int, n_v: int, d_k: int, d_v: int, chunk_size: int = 64, use_qk_norm_in_gdn: bool = True, apply_silu_in_gdr: bool = False, use_recurrent_scan_prefill: bool = False, mask_initial_state: bool = False)` — [`L856`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule_v2/_xla_impl_fwd.py#L856) — Applies the gated delta rule over ragged seq lengths
- `ragged_gated_delta_rule_decode_only(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, b_reshaped: jnp.ndarray, a_reshaped: jnp.ndarray, recurrent_state: jnp.ndarray, A_log: jnp.ndarray, dt_bias: jnp.ndarray, query_start_loc: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, use_qk_norm_in_gdn: bool)` — [`L747`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule_v2/_xla_impl_fwd.py#L747) — Applies gated delta rule for decode-only case (sequence lengths = 1).
- `ragged_gated_delta_rule_mixed_prefill(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, b_reshaped: jnp.ndarray, a_reshaped: jnp.ndarray, A_log: jnp.ndarray, dt_bias: jnp.ndarray, query_start_loc: jnp.ndarray, recurrent_state: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, has_initial_state: jnp.ndarray, chunk_size: int = 64, use_qk_norm_in_gdn: bool = False, mask_initial_state: bool = False, compute_dtype: jnp.dtype = jnp.bfloat16, precision: jax.lax.Precision = jax.lax.Precision.HIGHEST, preferred_element_type: jnp.dtype = jnp.float32, triangle_solver_impl: TriangleSolverImpl | None = None)` — [`L363`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule_v2/_xla_impl_fwd.py#L363) — Applies chunked gated delta rule for mixed prefill case.
- `ragged_gated_delta_rule_v2(mixed_qkv: jnp.ndarray, b: jnp.ndarray, a: jnp.ndarray, recurrent_state: jnp.ndarray, A_log: jnp.ndarray, dt_bias: jnp.ndarray, query_start_loc: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, has_initial_state: jnp.ndarray | None = None, *, n_kq: int, n_v: int, d_k: int, d_v: int, chunk_size: int = 64, use_qk_norm_in_gdn: bool = True, apply_silu_in_gdr: bool = False, use_recurrent_scan_prefill: bool = False, mask_initial_state: bool = False, runtime_dtype: jnp.dtype | None = None)` — [`L1018`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule_v2/_xla_impl_fwd.py#L1018) — Run the unsharded packed-inference GDN v2 XLA kernel.
- `recurrent_gated_delta_rule_step(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, g: jnp.ndarray, beta: jnp.ndarray, state: jnp.ndarray | None = None)` — [`L692`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule_v2/_xla_impl_fwd.py#L692) — Single-step recurrent update for gated-delta-rule decode.
- `scan_body(h, args)` — [`L605`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule_v2/_xla_impl_fwd.py#L605) — Inter-chunk recurrence body for the chunked GDR prefill scan.
- `to_chunk(x)` — [`L484`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule_v2/_xla_impl_fwd.py#L484) — Reshape ``[total, H, D]`` into ``[num_chunks, H, chunk_size, D]``.
- `to_chunk_scalar(x)` — [`L495`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule_v2/_xla_impl_fwd.py#L495) — Reshape ``[total, H]`` into ``[num_chunks, H, chunk_size]``.
- `triangular_inverse_jax(A)` — [`L142`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ragged_gated_delta_rule_v2/_xla_impl_fwd.py#L142) — Backend-agnostic unit lower-triangular inverse via ``jax.scipy``.

