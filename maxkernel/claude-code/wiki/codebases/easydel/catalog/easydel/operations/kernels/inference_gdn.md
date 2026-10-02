---
title: 'Module: easydel/operations/kernels/inference_gdn.py'
type: catalog
provenance: extracted
module: easydel/operations/kernels/inference_gdn.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.operations.kernels.inference_gdn`/
symbols:
  RaggedGatedDeltaRule.get_requirements: RaggedGatedDeltaRule#get_requirements().
  ragged_gated_delta_rule_mixed_prefill: ragged_gated_delta_rule_mixed_prefill().
  RaggedGatedDeltaRule.forward_native: RaggedGatedDeltaRule#forward_native().
  TriangleSolverImpl.__call__: TriangleSolverImpl#__call__().
  RaggedGatedDeltaRule: RaggedGatedDeltaRule#
  ragged_gated_delta_rule_decode_only: ragged_gated_delta_rule_decode_only().
  RaggedGatedDeltaRule._mapped: RaggedGatedDeltaRule#_mapped().
  TriangleSolverImpl: TriangleSolverImpl#
  _pallas_gdn_decode_call: _pallas_gdn_decode_call().
  ragged_gated_delta_rule: ragged_gated_delta_rule().
  decompose_triangular_matrix_inverse_pallas: decompose_triangular_matrix_inverse_pallas().
  l2norm: l2norm().
  newton_schulz_inverse_pallas_kernel: newton_schulz_inverse_pallas_kernel().
  newton_schulz_inverse_pallas: newton_schulz_inverse_pallas().
  decompose_triangular_matrix_inverse_pallas_kernel: decompose_triangular_matrix_inverse_pallas_kernel().
  ragged_gated_delta_rule.decode_only_branch: ragged_gated_delta_rule().decode_only_branch().
  ragged_gated_delta_rule.mixed_prefill_branch: ragged_gated_delta_rule().mixed_prefill_branch().
  TriangleSolverImpl.GAUSSIAN: TriangleSolverImpl#GAUSSIAN.
  ragged_gated_delta_rule_mixed_prefill.to_chunk: ragged_gated_delta_rule_mixed_prefill().to_chunk().
  _PALLAS_GDN_BTOK: _PALLAS_GDN_BTOK.
  RaggedGatedDeltaRule.forward_tpu: RaggedGatedDeltaRule#forward_tpu().
  RaggedGatedDeltaRule.forward_gpu: RaggedGatedDeltaRule#forward_gpu().
  RaggedGatedDeltaRule.forward_cpu: RaggedGatedDeltaRule#forward_cpu().
  RaggedGatedDeltaRule.forward_cuda: RaggedGatedDeltaRule#forward_cuda().
  RaggedGatedDeltaRule.forward_rocm: RaggedGatedDeltaRule#forward_rocm().
  TriangleSolverImpl.JAX: TriangleSolverImpl#JAX.
  ragged_gated_delta_rule_mixed_prefill.to_chunk_scalar: ragged_gated_delta_rule_mixed_prefill().to_chunk_scalar().
  newton_schulz_inverse_ref: newton_schulz_inverse_ref().
  local_forward_substitution: local_forward_substitution().
  triangular_inverse_jax: triangular_inverse_jax().
  TriangleSolverImpl.NEWTON_SCHULZ: TriangleSolverImpl#NEWTON_SCHULZ.
  pack_inputs_single_stream: pack_inputs_single_stream().
  ragged_gated_delta_rule_mixed_prefill.scan_body: ragged_gated_delta_rule_mixed_prefill().scan_body().
  _pallas_gdn_decode_kernel: _pallas_gdn_decode_kernel().
  recurrent_gated_delta_rule_step: recurrent_gated_delta_rule_step().
  RaggedGatedDeltaRule.get_impl_name: RaggedGatedDeltaRule#get_impl_name().
---
# Module: [`easydel/operations/kernels/inference_gdn.py`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py)

## Classes
### `RaggedGatedDeltaRule`  ·  implements/extends OperationImpl
- def: [`easydel/operations/kernels/inference_gdn.py:1008`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L1008)
- doc: Ragged Gated Delta Rule operation wrapping the inlined vLLM kernel.
- signature: `class RaggedGatedDeltaRule(OperationImpl):`
- members:
  - `forward_cpu(self, *args, **kwargs)` — [`L1222`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L1222)
  - `forward_cuda(self, *args, **kwargs)` — [`L1225`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L1225)
  - `forward_gpu(self, *args, **kwargs)` — [`L1219`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L1219)
  - `forward_native(self, mixed_qkv: jnp.ndarray, b: jnp.ndarray, a: jnp.ndarray, recurrent_state: jnp.ndarray, A_log: jnp.ndarray, dt_bias: jnp.ndarray, query_start_loc: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, *, n_kq: int, n_v: int, d_k: int, d_v: int, chunk_size: int = 64, use_qk_norm_in_gdn: bool = True)` — [`L1049`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L1049) — Forward pass for ragged gated delta rule.
  - `forward_rocm(self, *args, **kwargs)` — [`L1228`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L1228)
  - `forward_tpu(self, *args, **kwargs)` — [`L1216`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L1216)
  - `get_impl_name(cls)` — [`L1022`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L1022)
  - `get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED)` — [`L1026`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L1026) — Returns requirements for RaggedGatedDeltaRule.
- protocol/private: `_mapped`[`L1149`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L1149)
- uses (calls/refs, reference-scoped): [`MetadataField`](../requirements/types.md#MetadataField), [`CacheType`](../requirements/types.md#CacheType), [`metadata`](../_base_operation.md#BaseOperation.metadata), [`ExecutionMode`](../requirements/types.md#ExecutionMode), [`OperationRequirements`](../requirements/requirements.md#OperationRequirements), [`OperationImpl`](../_operation_impl.md#OperationImpl), [`get_shardings`](../_operation_meta.md#OperationMetadata.get_shardings), [`build`](../requirements/builder.md#RequirementsBuilder.build), [`mesh`](../_operation_meta.md#OperationMetadata.mesh), [`RequirementsBuilder`](../requirements/builder.md#RequirementsBuilder), [`register`](../_base_operation.md#OperationRegistry.register), [`MIXED`](../requirements/types.md#ExecutionMode.MIXED), [`require_metadata`](../requirements/builder.md#RequirementsBuilder.require_metadata), [`support_cache`](../requirements/builder.md#RequirementsBuilder.support_cache), [`query`](../_operation_meta.md#AttnShardingRules.query), [`optional_metadata`](../requirements/builder.md#RequirementsBuilder.optional_metadata), [`get_mode`](../_operation_impl.md#OperationImpl.get_mode), [`HYBRID`](../requirements/types.md#CacheType.HYBRID), [`runtime_dtype`](../_operation_meta.md#OperationMetadata.runtime_dtype), [`RECURRENT`](../requirements/types.md#CacheType.RECURRENT), [`POSITIONS`](../requirements/types.md#MetadataField.POSITIONS), [`SEQ_LENS`](../requirements/types.md#MetadataField.SEQ_LENS), [`LOGITS_INDICES`](../requirements/types.md#MetadataField.LOGITS_INDICES), [`STATE_INDICES`](../requirements/types.md#MetadataField.STATE_INDICES), [`HAS_INITIAL_STATE`](../requirements/types.md#MetadataField.HAS_INITIAL_STATE), [`ragged_gated_delta_rule`](inference_gdn.md#ragged_gated_delta_rule)
- used by: [`OperationImpl`](../_operation_impl.md#OperationImpl), [`get_requirements`](../_base_operation.md#BaseOperation.get_requirements), [`forward_native`](../_base_operation.md#BaseOperation.forward_native), [`get_impl_name`](../_base_operation.md#BaseOperation.get_impl_name), [`forward_gpu`](../_base_operation.md#BaseOperation.forward_gpu), [`forward_tpu`](../_base_operation.md#BaseOperation.forward_tpu), [`forward_cpu`](../_base_operation.md#BaseOperation.forward_cpu), [`forward_cuda`](../_base_operation.md#BaseOperation.forward_cuda), [`forward_rocm`](../_base_operation.md#BaseOperation.forward_rocm), [`_apply_qwen3_next_packed_updates`](../../modules/qwen3_next/modeling_qwen3_next.md#_apply_qwen3_next_packed_updates), [`_apply_qwen3_next_packed_updates_ragged`](../../modules/qwen3_next/modeling_qwen3_next.md#_apply_qwen3_next_packed_updates_ragged), [`ragged_gdr_op`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextLinearAttention.ragged_gdr_op)

### `TriangleSolverImpl`  ·  implements/extends StrEnum
- def: [`easydel/operations/kernels/inference_gdn.py:205`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L205)
- signature: `class TriangleSolverImpl(enum.StrEnum):`
- members:
  - `GAUSSIAN` — [`L206`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L206)
  - `JAX` — [`L208`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L208)
  - `NEWTON_SCHULZ` — [`L207`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L207)
- protocol/private: `__call__`[`L210`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L210)
- uses (calls/refs, reference-scoped): [`decompose_triangular_matrix_inverse_pallas`](inference_gdn.md#decompose_triangular_matrix_inverse_pallas), [`newton_schulz_inverse_pallas`](inference_gdn.md#newton_schulz_inverse_pallas), [`triangular_inverse_jax`](inference_gdn.md#triangular_inverse_jax)
- used by: [`ragged_gated_delta_rule_mixed_prefill`](inference_gdn.md#ragged_gated_delta_rule_mixed_prefill)

## Functions
- `_pallas_gdn_decode_call(q, k, v, beta, exp_g, state, valid)` — [`L694`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L694) — Run the fused Pallas kernel. Returns (new_state_pool, outputs_3d).
- `_pallas_gdn_decode_kernel(q_ref, k_ref, v_ref, beta_ref, exp_g_ref, state_ref, valid_ref, out_ref, new_state_ref)` — [`L638`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L638) — Per-program: process `B_TOK` tokens (unrolled) with one slot's state each.
- `decode_only_branch(_)` — [`L968`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L968)
- `decompose_triangular_matrix_inverse_pallas(A, *, n_block_size=64, block_size=16)` — [`L150`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L150) — Inverts unit lower triangular matrices using a block-wise approach in Pallas.
- `decompose_triangular_matrix_inverse_pallas_kernel(A_ref, x_ref, *, block_size=16)` — [`L128`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L128)
- `l2norm(x: jnp.ndarray, dim: int = -1, eps: float = 0.000001)` — [`L222`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L222) — Normalizes x along the specified dimension using L2 norm.
- `local_forward_substitution(A, b)` — [`L101`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L101) — Solves A X = B for unit lower triangular matrix A using forward substitution.
- `mixed_prefill_branch(_)` — [`L985`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L985)
- `newton_schulz_inverse_pallas(A, *, block_size=64)` — [`L81`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L81) — Newton-Schulz iteration for unit lower triangular matrices on Pallas.
- `newton_schulz_inverse_pallas_kernel(A_ref, x_ref)` — [`L77`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L77)
- `newton_schulz_inverse_ref(A, n=None)` — [`L38`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L38) — Inverse of unit lower triangular matrix using Newton-Schulz iteration.
- `pack_inputs_single_stream(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, g: jnp.ndarray, beta: jnp.ndarray, query_start_loc: jnp.ndarray, distribution: jnp.ndarray, chunk_size: int, compute_dtype: jnp.dtype = jnp.bfloat16)` — [`L237`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L237) — Pads each sequence to multiple of chunk_size and concatenates.
- `ragged_gated_delta_rule(mixed_qkv: jnp.ndarray, b: jnp.ndarray, a: jnp.ndarray, recurrent_state: jnp.ndarray, A_log: jnp.ndarray, dt_bias: jnp.ndarray, query_start_loc: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, *, n_kq: int, n_v: int, d_k: int, d_v: int, chunk_size: int = 64, use_qk_norm_in_gdn: bool = True)` — [`L901`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L901) — Applies the gated delta rule over ragged seq lengths
- `ragged_gated_delta_rule_decode_only(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, b_reshaped: jnp.ndarray, a_reshaped: jnp.ndarray, recurrent_state: jnp.ndarray, A_log: jnp.ndarray, dt_bias: jnp.ndarray, query_start_loc: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, use_qk_norm_in_gdn: bool)` — [`L772`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L772) — Applies gated delta rule for decode-only case (sequence lengths = 1).
- `ragged_gated_delta_rule_mixed_prefill(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, b_reshaped: jnp.ndarray, a_reshaped: jnp.ndarray, A_log: jnp.ndarray, dt_bias: jnp.ndarray, query_start_loc: jnp.ndarray, recurrent_state: jnp.ndarray, state_indices: jnp.ndarray, distribution: jnp.ndarray, chunk_size: int = 64, use_qk_norm_in_gdn: bool = False, compute_dtype: jnp.dtype = jnp.bfloat16, precision: jax.lax.Precision = jax.lax.Precision.HIGHEST, preferred_element_type: jnp.dtype = jnp.float32, triangle_solver_impl: TriangleSolverImpl | None = None)` — [`L376`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L376) — Applies chunked gated delta rule for mixed prefill case.
- `recurrent_gated_delta_rule_step(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, g: jnp.ndarray, beta: jnp.ndarray, state: jnp.ndarray | None = None)` — [`L735`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L735) — Single-step recurrent update for decode.
- `scan_body(h, args)` — [`L570`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L570)
- `to_chunk(x)` — [`L475`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L475)
- `to_chunk_scalar(x)` — [`L478`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L478)
- `triangular_inverse_jax(A)` — [`L191`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L191) — Pure-JAX inverse of unit lower-triangular matrices (any backend).

## Module values
- `_PALLAS_GDN_BTOK` — [`L691`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/inference_gdn.py#L691)

