---
title: 'Module: ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.ragged_gated_delta_rule_v2._pallas_impl_fwd`/
symbols:
  ragged_gated_delta_rule_mixed_prefill: ragged_gated_delta_rule_mixed_prefill().
  TriangleSolverImpl.__call__: TriangleSolverImpl#__call__().
  ragged_gated_delta_rule_decode_only: ragged_gated_delta_rule_decode_only().
  _select_pallas_gdn_btok: _select_pallas_gdn_btok().
  set_gdn_kernel_tile_policy: set_gdn_kernel_tile_policy().
  ragged_gated_delta_rule: ragged_gated_delta_rule().
  ragged_gated_delta_rule.mixed_prefill_branch: ragged_gated_delta_rule().mixed_prefill_branch().
  TriangleSolverImpl: TriangleSolverImpl#
  ragged_gated_delta_rule_v2: ragged_gated_delta_rule_v2().
  _PALLAS_GDN_TILE_POLICY: _PALLAS_GDN_TILE_POLICY.
  decompose_triangular_matrix_inverse_pallas: decompose_triangular_matrix_inverse_pallas().
  l2norm: l2norm().
  newton_schulz_inverse_pallas_kernel: newton_schulz_inverse_pallas_kernel().
  newton_schulz_inverse_pallas: newton_schulz_inverse_pallas().
  decompose_triangular_matrix_inverse_pallas_kernel: decompose_triangular_matrix_inverse_pallas_kernel().
  _pallas_gdn_decode_call: _pallas_gdn_decode_call().
  ragged_gated_delta_rule.decode_only_branch: ragged_gated_delta_rule().decode_only_branch().
  ragged_gated_delta_rule.mixed_prefill_branch.recurrent_prefill_only: ragged_gated_delta_rule().mixed_prefill_branch().recurrent_prefill_only().
  ragged_gated_delta_rule.mixed_prefill_branch.jax_mixed_prefill: ragged_gated_delta_rule().mixed_prefill_branch().jax_mixed_prefill().
  TriangleSolverImpl.GAUSSIAN: TriangleSolverImpl#GAUSSIAN.
  ragged_gated_delta_rule_mixed_prefill.to_chunk: ragged_gated_delta_rule_mixed_prefill().to_chunk().
  TriangleSolverImpl.JAX: TriangleSolverImpl#JAX.
  ragged_gated_delta_rule_mixed_prefill.to_chunk_scalar: ragged_gated_delta_rule_mixed_prefill().to_chunk_scalar().
  newton_schulz_inverse_ref: newton_schulz_inverse_ref().
  local_forward_substitution: local_forward_substitution().
  triangular_inverse_jax: triangular_inverse_jax().
  TriangleSolverImpl.NEWTON_SCHULZ: TriangleSolverImpl#NEWTON_SCHULZ.
  pack_inputs_single_stream: pack_inputs_single_stream().
  ragged_gated_delta_rule_mixed_prefill.scan_body: ragged_gated_delta_rule_mixed_prefill().scan_body().
  _pallas_gdn_decode_kernel: _pallas_gdn_decode_kernel().
  _PALLAS_GDN_BTOK_CANDIDATES: _PALLAS_GDN_BTOK_CANDIDATES.
  _reorder_concatenated_tensor_for_sharding: _reorder_concatenated_tensor_for_sharding().
  recurrent_gated_delta_rule_step: recurrent_gated_delta_rule_step().
---
# Module: [`ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py)

## Classes
### `TriangleSolverImpl`  ·  implements/extends StrEnum
- def: [`ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py:361`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L361)
- doc: Selector for the unit lower-triangular inverse implementation.
- signature: `class TriangleSolverImpl(enum.StrEnum):`
- members:
  - `__call__(self, A)` — [`L377`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L377) — Invoke the selected inverse implementation.
  - `GAUSSIAN` — [`L373`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L373)
  - `JAX` — [`L375`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L375)
  - `NEWTON_SCHULZ` — [`L374`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L374)
- uses (calls/refs, reference-scoped): [`decompose_triangular_matrix_inverse_pallas`](_pallas_impl_fwd.md#decompose_triangular_matrix_inverse_pallas), [`newton_schulz_inverse_pallas`](_pallas_impl_fwd.md#newton_schulz_inverse_pallas), [`triangular_inverse_jax`](_pallas_impl_fwd.md#triangular_inverse_jax)
- used by: [`ragged_gated_delta_rule_mixed_prefill`](_pallas_impl_fwd.md#ragged_gated_delta_rule_mixed_prefill)

## Functions
- `_pallas_gdn_decode_call(q, k, v, beta, exp_g, state, valid, *, b_tok: int)` — [`L1051`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L1051) — Run the fused Pallas decode kernel and return ``(outputs, new_state)``.
- `_pallas_gdn_decode_kernel(q_ref, k_ref, v_ref, beta_ref, exp_g_ref, state_ref, valid_ref, out_ref, new_state_ref)` — [`L899`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L899) — Process ``B_TOK`` tokens (unrolled) with one slot's state each.
- `_reorder_concatenated_tensor_for_sharding(concatenated_tensor: jax.Array, split_sizes: tuple[int, ...], n_shards: int, dim: int)` — [`L63`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L63) — Arrange fused ``[A|B|C]`` features into per-shard interleaved order.
- `_select_pallas_gdn_btok(num_tokens: int, n_v: int, d_k: int, d_v: int, dtype, kernel_tile_policy: KernelTilePolicy | str | None = None)` — [`L1000`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L1000) — Choose a TPU tile that keeps Mosaic VMEM windows under control.
- `decode_only_branch(_)` — [`L1439`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L1439) — Run the decode-only fast path under :func:`lax.cond`.
- `decompose_triangular_matrix_inverse_pallas(A, *, n_block_size=64, block_size=16)` — [`L280`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L280) — Pallas TPU kernel that inverts a stack of unit lower-triangular matrices.
- `decompose_triangular_matrix_inverse_pallas_kernel(A_ref, x_ref, *, block_size=16)` — [`L243`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L243) — Pallas kernel body for blockwise unit lower-triangular inversion.
- `jax_mixed_prefill(_)` — [`L1522`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L1522) — Run the chunked GDR prefill for the mixed decode+prefill case.
- `l2norm(x: jnp.ndarray, dim: int = -1, eps: float = 0.000001)` — [`L400`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L400) — Normalize ``x`` to unit L2 norm along ``dim``.
- `local_forward_substitution(A, b)` — [`L205`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L205) — Solve :math:`A X = b` row-by-row for unit lower-triangular ``A``.
- `mixed_prefill_branch(_)` — [`L1471`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L1471) — Run the chunked mixed-prefill path under :func:`lax.cond`.
- `newton_schulz_inverse_pallas(A, *, block_size=64)` — [`L171`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L171) — Newton-Schulz iteration for unit lower triangular matrices on Pallas.
- `newton_schulz_inverse_pallas_kernel(A_ref, x_ref)` — [`L156`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L156) — Pallas kernel body that wraps :func:`newton_schulz_inverse_ref`.
- `newton_schulz_inverse_ref(A, n=None)` — [`L114`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L114) — Reference Newton-Schulz inverse for unit lower-triangular matrices.
- `pack_inputs_single_stream(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, g: jnp.ndarray, beta: jnp.ndarray, query_start_loc: jnp.ndarray, distribution: jnp.ndarray, chunk_size: int, compute_dtype: jnp.dtype = jnp.bfloat16)` — [`L423`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L423) — Pads each sequence to multiple of chunk_size and concatenates.
- `ragged_gated_delta_rule(mixed_qkv: jnp.ndarray, b: jnp.ndarray, a: jnp.ndarray, recurrent_state: jnp.ndarray, A_log: jnp.ndarray, dt_bias: jnp.ndarray, query_start_loc: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, has_initial_state: jnp.ndarray | None = None, *, n_kq: int, n_v: int, d_k: int, d_v: int, chunk_size: int = 64, use_qk_norm_in_gdn: bool = True, apply_silu_in_gdr: bool = False, use_recurrent_scan_prefill: bool = False, mask_initial_state: bool = False, kernel_tile_policy: KernelTilePolicy | str = "auto", use_fused_gdn_decode: bool = False)` — [`L1338`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L1338) — Applies the gated delta rule over ragged seq lengths
- `ragged_gated_delta_rule_decode_only(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, b_reshaped: jnp.ndarray, a_reshaped: jnp.ndarray, recurrent_state: jnp.ndarray, A_log: jnp.ndarray, dt_bias: jnp.ndarray, query_start_loc: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, use_qk_norm_in_gdn: bool, kernel_tile_policy: KernelTilePolicy | str = "auto", use_fused_gdn_decode: bool = False)` — [`L1167`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L1167) — Applies gated delta rule for decode-only case (sequence lengths = 1).
- `ragged_gated_delta_rule_mixed_prefill(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, b_reshaped: jnp.ndarray, a_reshaped: jnp.ndarray, A_log: jnp.ndarray, dt_bias: jnp.ndarray, query_start_loc: jnp.ndarray, recurrent_state: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, has_initial_state: jnp.ndarray, chunk_size: int = 64, use_qk_norm_in_gdn: bool = False, mask_initial_state: bool = False, compute_dtype: jnp.dtype = jnp.bfloat16, precision: jax.lax.Precision = jax.lax.Precision.HIGHEST, preferred_element_type: jnp.dtype = jnp.float32, triangle_solver_impl: TriangleSolverImpl | None = None)` — [`L569`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L569) — Applies chunked gated delta rule for mixed prefill case.
- `ragged_gated_delta_rule_v2(mixed_qkv: jnp.ndarray, b: jnp.ndarray, a: jnp.ndarray, recurrent_state: jnp.ndarray, A_log: jnp.ndarray, dt_bias: jnp.ndarray, query_start_loc: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, has_initial_state: jnp.ndarray | None = None, *, n_kq: int, n_v: int, d_k: int, d_v: int, chunk_size: int = 64, use_qk_norm_in_gdn: bool = True, apply_silu_in_gdr: bool = False, use_recurrent_scan_prefill: bool = False, mask_initial_state: bool = False, kernel_tile_policy: KernelTilePolicy | str = "auto", use_fused_gdn_decode: bool = False, runtime_dtype: jnp.dtype | None = None)` — [`L1579`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L1579) — Run the unsharded packed-inference GDN v2 kernel with dtype coercion.
- `recurrent_gated_delta_rule_step(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, g: jnp.ndarray, beta: jnp.ndarray, state: jnp.ndarray | None = None)` — [`L1111`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L1111) — Single-step recurrent update for gated-delta-rule decode.
- `recurrent_prefill_only(_)` — [`L1487`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L1487) — Run the recurrent-scan kernel for the pure-prefill case.
- `scan_body(h, args)` — [`L812`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L812) — Inter-chunk recurrence body for the chunked GDR prefill scan.
- `set_gdn_kernel_tile_policy(policy: str)` — [`L981`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L981) — Set the TPU Pallas GDN decode tile policy for future traces.
- `to_chunk(x)` — [`L691`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L691) — Reshape ``[total, H, D]`` into ``[num_chunks, H, chunk_size, D]``.
- `to_chunk_scalar(x)` — [`L702`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L702) — Reshape ``[total, H]`` into ``[num_chunks, H, chunk_size]``.
- `triangular_inverse_jax(A)` — [`L335`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L335) — Backend-agnostic unit lower-triangular inverse via ``jax.scipy``.

## Module values
- `_PALLAS_GDN_BTOK_CANDIDATES` — [`L978`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L978)
- `_PALLAS_GDN_TILE_POLICY` — [`L977`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L977)

