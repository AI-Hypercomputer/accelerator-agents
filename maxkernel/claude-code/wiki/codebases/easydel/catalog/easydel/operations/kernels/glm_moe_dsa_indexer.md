---
title: 'Module: easydel/operations/kernels/glm_moe_dsa_indexer.py'
type: catalog
provenance: extracted
module: easydel/operations/kernels/glm_moe_dsa_indexer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.operations.kernels.glm_moe_dsa_indexer`/
symbols:
  GlmMoeDsaIndexerOp.get_requirements: GlmMoeDsaIndexerOp#get_requirements().
  GlmMoeDsaIndexerOutput: GlmMoeDsaIndexerOutput#
  GlmMoeDsaIndexerOp.forward_native: GlmMoeDsaIndexerOp#forward_native().
  GlmMoeDsaIndexerOutput.cached_keys: GlmMoeDsaIndexerOutput#cached_keys.
  GlmMoeDsaIndexerOp: GlmMoeDsaIndexerOp#
  GlmMoeDsaIndexerOp.forward_tpu: GlmMoeDsaIndexerOp#forward_tpu().
  GlmMoeDsaIndexerOp.forward_gpu: GlmMoeDsaIndexerOp#forward_gpu().
  GlmMoeDsaIndexerOp.forward_cpu: GlmMoeDsaIndexerOp#forward_cpu().
  GlmMoeDsaIndexerOp.forward_cuda: GlmMoeDsaIndexerOp#forward_cuda().
  GlmMoeDsaIndexerOp.forward_rocm: GlmMoeDsaIndexerOp#forward_rocm().
  GlmMoeDsaIndexerOp.__call__: GlmMoeDsaIndexerOp#__call__().
  GlmMoeDsaIndexerOp._apply_rope_interleaved: GlmMoeDsaIndexerOp#_apply_rope_interleaved().
  GlmMoeDsaIndexerOp._apply_rope_split: GlmMoeDsaIndexerOp#_apply_rope_split().
  GlmMoeDsaIndexerOutput.topk_indices: GlmMoeDsaIndexerOutput#topk_indices.
  GlmMoeDsaIndexerOp.get_impl_name: GlmMoeDsaIndexerOp#get_impl_name().
  __all__: __all__.
---
# Module: [`easydel/operations/kernels/glm_moe_dsa_indexer.py`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/glm_moe_dsa_indexer.py)

## Classes
### `GlmMoeDsaIndexerOp`  ·  implements/extends OperationImpl
- def: [`easydel/operations/kernels/glm_moe_dsa_indexer.py:37`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/glm_moe_dsa_indexer.py#L37)
- doc: Computes DSA top-k token indices with an optional indexer-local key cache.
- signature: `class GlmMoeDsaIndexerOp(OperationImpl):`
- members:
  - `__call__(self, query_states: Float[Array, "batch seq n_heads head_dim"], key_states: Float[Array, "batch seq head_dim"], head_weights: Float[Array, "batch seq n_heads"], position_ids: Int[Array, "batch seq"], qk_rope_head_dim: int, index_topk: int, softmax_scale: float, frequencies: Float[Array, "max_seq rope_dim_x2"] | None = None, attention_mask: Bool[Array, "batch seq kv"] | Float[Array, "batch seq kv"] | None = None, cached_keys: Float[Array, "batch cached_seq head_dim"] | None = None, use_cache: bool = False, reset_cache: bool = False, indexer_rope_interleave: bool = False, **kwargs)` — [`L236`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/glm_moe_dsa_indexer.py#L236) — Execute the DSA indexer by dispatching to the appropriate backend.
  - `forward_cpu(self, *args, **kwargs)` — [`L224`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/glm_moe_dsa_indexer.py#L224) — CPU forward pass. Delegates to ``forward_native``.
  - `forward_cuda(self, *args, **kwargs)` — [`L228`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/glm_moe_dsa_indexer.py#L228) — CUDA forward pass. Delegates to ``forward_native``.
  - `forward_gpu(self, *args, **kwargs)` — [`L220`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/glm_moe_dsa_indexer.py#L220) — GPU forward pass. Delegates to ``forward_native``.
  - `forward_native(self, query_states: Float[Array, "batch seq n_heads head_dim"], key_states: Float[Array, "batch seq head_dim"], head_weights: Float[Array, "batch seq n_heads"], position_ids: Int[Array, "batch seq"], qk_rope_head_dim: int, index_topk: int, softmax_scale: float, frequencies: Float[Array, "max_seq rope_dim_x2"] | None = None, attention_mask: Bool[Array, "batch seq kv"] | Float[Array, "batch seq kv"] | None = None, cached_keys: Float[Array, "batch cached_seq head_dim"] | None = None, use_cache: bool = False, reset_cache: bool = False, indexer_rope_interleave: bool = False, **ignore)` — [`L83`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/glm_moe_dsa_indexer.py#L83) — Compute DSA top-k token indices for dynamic sparse attention.
  - `forward_rocm(self, *args, **kwargs)` — [`L232`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/glm_moe_dsa_indexer.py#L232) — ROCm forward pass. Delegates to ``forward_native``.
  - `forward_tpu(self, *args, **kwargs)` — [`L216`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/glm_moe_dsa_indexer.py#L216) — TPU forward pass. Delegates to ``forward_native``.
  - `get_impl_name(cls)` — [`L41`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/glm_moe_dsa_indexer.py#L41)
  - `get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED)` — [`L45`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/glm_moe_dsa_indexer.py#L45)
- protocol/private: `_apply_rope_interleaved`[`L59`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/glm_moe_dsa_indexer.py#L59), `_apply_rope_split`[`L71`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/glm_moe_dsa_indexer.py#L71)
- uses (calls/refs, reference-scoped): [`MetadataField`](../requirements/types.md#MetadataField), [`CacheType`](../requirements/types.md#CacheType), [`ExecutionMode`](../requirements/types.md#ExecutionMode), [`OperationRequirements`](../requirements/requirements.md#OperationRequirements), [`OperationImpl`](../_operation_impl.md#OperationImpl), [`build`](../requirements/builder.md#RequirementsBuilder.build), [`__call__`](../_base_operation.md#BaseOperation.__call__), [`RequirementsBuilder`](../requirements/builder.md#RequirementsBuilder), [`MIXED`](../requirements/types.md#ExecutionMode.MIXED), [`require_metadata`](../requirements/builder.md#RequirementsBuilder.require_metadata), [`support_cache`](../requirements/builder.md#RequirementsBuilder.support_cache), [`TRANSFORMER`](../requirements/types.md#CacheType.TRANSFORMER), [`HYBRID`](../requirements/types.md#CacheType.HYBRID), [`GlmMoeDsaIndexerOutput`](glm_moe_dsa_indexer.md#GlmMoeDsaIndexerOutput), [`POSITIONS`](../requirements/types.md#MetadataField.POSITIONS), [`requires_cache`](../requirements/builder.md#RequirementsBuilder.requires_cache)
- used by: [`OperationImpl`](../_operation_impl.md#OperationImpl), [`get_requirements`](../_base_operation.md#BaseOperation.get_requirements), [`forward_native`](../_base_operation.md#BaseOperation.forward_native), [`get_impl_name`](../_base_operation.md#BaseOperation.get_impl_name), [`forward_gpu`](../_base_operation.md#BaseOperation.forward_gpu), [`forward_tpu`](../_base_operation.md#BaseOperation.forward_tpu), [`forward_cpu`](../_base_operation.md#BaseOperation.forward_cpu), [`forward_cuda`](../_base_operation.md#BaseOperation.forward_cuda), [`forward_rocm`](../_base_operation.md#BaseOperation.forward_rocm), [`indexer_op`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaIndexer.indexer_op)

### `GlmMoeDsaIndexerOutput`  ·  implements/extends OperationOutput
- def: [`easydel/operations/kernels/glm_moe_dsa_indexer.py:29`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/glm_moe_dsa_indexer.py#L29)
- doc: Output container for the GLM-MoE-DSA indexer operation.
- signature: `class GlmMoeDsaIndexerOutput(OperationOutput):`
- members:
  - `cached_keys` — [`L33`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/glm_moe_dsa_indexer.py#L33)
  - `topk_indices` — [`L32`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/glm_moe_dsa_indexer.py#L32)
- uses (calls/refs, reference-scoped): [`register`](../_base_operation.md#OperationRegistry.register), [`OperationOutput`](../_operation_impl.md#OperationOutput)
- used by: [`forward_mla`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaAttention.forward_mla), [`__call__`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaIndexer.__call__), [`forward_native`](glm_moe_dsa_indexer.md#GlmMoeDsaIndexerOp.forward_native), [`OperationOutput`](../_operation_impl.md#OperationOutput), [`__call__`](glm_moe_dsa_indexer.md#GlmMoeDsaIndexerOp.__call__), [`forward_cpu`](glm_moe_dsa_indexer.md#GlmMoeDsaIndexerOp.forward_cpu), [`forward_cuda`](glm_moe_dsa_indexer.md#GlmMoeDsaIndexerOp.forward_cuda), [`forward_gpu`](glm_moe_dsa_indexer.md#GlmMoeDsaIndexerOp.forward_gpu), [`forward_rocm`](glm_moe_dsa_indexer.md#GlmMoeDsaIndexerOp.forward_rocm), [`forward_tpu`](glm_moe_dsa_indexer.md#GlmMoeDsaIndexerOp.forward_tpu)

## Module values
- `__all__` — [`L292`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/glm_moe_dsa_indexer.py#L292)

