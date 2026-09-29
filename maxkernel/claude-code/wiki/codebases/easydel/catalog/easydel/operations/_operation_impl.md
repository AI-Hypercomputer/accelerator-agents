---
title: 'Module: easydel/operations/_operation_impl.py'
type: catalog
provenance: extracted
module: easydel/operations/_operation_impl.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.operations._operation_impl`/
symbols:
  OperationImpl: OperationImpl#
  OperationImpl.create_stable_sharding: OperationImpl#create_stable_sharding().
  OperationImpl.get_instance_requirements: OperationImpl#get_instance_requirements().
  OperationImpl.get_mode: OperationImpl#get_mode().
  OperationOutput: OperationOutput#
  OperationImpl.__init__: OperationImpl#__init__().
  OperationImpl.get_impl_metadata: OperationImpl#get_impl_metadata().
  __all__: __all__.
  RUNTIME_MODE_TYPES: RUNTIME_MODE_TYPES.
  logger: logger.
  NOT_GIVEN: NOT_GIVEN.
  BATCH: BATCH.
  QUERY_LENGTH: QUERY_LENGTH.
  KV_LENGTH: KV_LENGTH.
  HEAD: HEAD.
  KV_HEAD: KV_HEAD.
  HEAD_DIM: HEAD_DIM.
  KV_HEAD_DIM: KV_HEAD_DIM.
  BIAS_HEAD_SEQ: BIAS_HEAD_SEQ.
  BIAS_KV_SEQ: BIAS_KV_SEQ.
  OperationImpl._split_attention_mask: OperationImpl#_split_attention_mask().
  OperationImpl._combine_query_kv_masks: OperationImpl#_combine_query_kv_masks().
  OperationImpl._create_causal_mask: OperationImpl#_create_causal_mask().
  OperationImpl.repeat_kv_heads: OperationImpl#repeat_kv_heads().
  OperationImpl._handle_kvhead: OperationImpl#_handle_kvhead().
---
# Module: [`easydel/operations/_operation_impl.py`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py)

## Classes
### `OperationImpl`  ·  implements/extends BaseOperation
- def: [`easydel/operations/_operation_impl.py:96`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L96)
- doc: Abstract Base Class for specific attention implementations.
- signature: `class OperationImpl(BaseOperation):`
- members:
  - `__init__(self, metadata: OperationMetadata)` — [`L109`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L109) — Initializes the attention implementation with its metadata.
  - `_combine_query_kv_masks(q_mask: Bool[Array, "... q_seq"], kv_mask: Bool[Array, "... kv_seq"])` — [`L208`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L208) — Combines separate query and key-value masks into a standard attention mask.
  - `_create_causal_mask(qseq: int)` — [`L231`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L231) — Creates a causal attention mask (lower triangular).
  - `_handle_kvhead(self, array: Float[Array, "batch heads q_seq kv_seq"] | None, num_q_heads: int, num_kv_heads: int)` — [`L269`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L269) — Processes an attention bias or similar array based on head configuration (GQA/MQA).
  - `_split_attention_mask(attn_mask: Bool[Array, "... seq_len seq_len"])` — [`L183`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L183) — Splits a combined attention mask into separate query and key-value masks.
  - `create_stable_sharding(self, state_ps: Ps | None = None, preserved_indices: list[int] | None = None, clone_ps: Ps | None = None, dep: Ps | bool | None = True, tensor: Float[Array, ...] | None = None)` — [`L320`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L320) — Helper to create a PartitionSpec, potentially preserving only certain axes.
  - `get_impl_metadata(self)` — [`L119`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L119) — Returns the metadata associated with this attention implementation instance.
  - `get_instance_requirements(self, mode: ExecutionMode = ExecutionMode.MIXED)` — [`L133`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L133) — Returns the operation requirements, applying instance-level overrides.
  - `get_mode(self, query: Float[Array, "batch ... num_heads head_dim"], BTHD: bool = True)` — [`L169`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L169) — Determines the runtime mode (normal or generation) based on query shape.
  - `repeat_kv_heads(k: Float[Array, "batch seq_len num_kv_heads head_dim"], v: Float[Array, "batch seq_len num_kv_heads head_dim"], num_reps: int)` — [`L245`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L245) — Repeats Key and Value heads for Grouped Query Operation (GQA) or Multi-Query Operation (MQA).
- uses (calls/refs, reference-scoped): [`metadata`](_base_operation.md#BaseOperation.metadata), [`ExecutionMode`](requirements/types.md#ExecutionMode), [`OperationRequirements`](requirements/requirements.md#OperationRequirements), [`get_requirements`](_base_operation.md#BaseOperation.get_requirements), [`OperationMetadata`](_operation_meta.md#OperationMetadata), [`mesh`](_operation_meta.md#OperationMetadata.mesh), [`MIXED`](requirements/types.md#ExecutionMode.MIXED), [`BaseOperation`](_base_operation.md#BaseOperation), [`with_requires_cache`](requirements/requirements.md#OperationRequirements.with_requires_cache), [`VanillaAttn`](kernels/vanilla_attention.md#VanillaAttn), [`ScaledDotProductAttn`](kernels/scaled_dot_product_attention.md#ScaledDotProductAttn), [`GatedDeltaRuleOp`](kernels/gated_delta_rule.md#GatedDeltaRuleOp), [`RaggedGatedDeltaRule`](kernels/inference_gdn.md#RaggedGatedDeltaRule), [`_RaggedPageAttn`](kernels/ragged_page_attention.md#_RaggedPageAttn), [`KernelDeltaAttnOp`](kernels/kda.md#KernelDeltaAttnOp), [`RingAttn`](kernels/ring_attention.md#RingAttn), [`SSM1Op`](kernels/ssm1.md#SSM1Op), [`SSM2Op`](kernels/ssm2.md#SSM2Op), [`AutoRegressiveDecodeAttn`](kernels/decode_attention.md#AutoRegressiveDecodeAttn), [`PagedFlashAttn`](kernels/paged_flash_attention.md#PagedFlashAttn), [`UnifiedAttn`](kernels/unified_attention.md#UnifiedAttn), [`BlockSparseAttn`](kernels/blocksparse_attention.md#BlockSparseAttn), [`FlashAttn`](kernels/flash_attention.md#FlashAttn), [`GlmMoeDsaIndexerOp`](kernels/glm_moe_dsa_indexer.md#GlmMoeDsaIndexerOp), [`MultiLatentRaggedPageAttn`](kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttn), [`RaggedCausalConv1D`](kernels/inference_conv1d.md#RaggedCausalConv1D), [`MultiLatentRaggedPageAttnV2`](kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttnV2), [`requires_cache`](_operation_meta.md#OperationMetadata.requires_cache), [`RUNTIME_MODE_TYPES`](_operation_impl.md#RUNTIME_MODE_TYPES)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`forward_native`](kernels/paged_flash_attention.md#PagedFlashAttn.forward_native), [`forward_native`](kernels/blocksparse_attention.md#BlockSparseAttn.forward_native), [`forward_native`](kernels/decode_attention.md#AutoRegressiveDecodeAttn.forward_native), [`forward_native`](kernels/ring_attention.md#RingAttn.forward_native), [`forward_native`](kernels/flash_attention.md#FlashAttn.forward_native), [`forward_native`](kernels/gated_delta_rule.md#GatedDeltaRuleOp.forward_native), [`forward_native`](kernels/kda.md#KernelDeltaAttnOp.forward_native), [`forward_native`](kernels/scaled_dot_product_attention.md#ScaledDotProductAttn.forward_native), [`forward_native`](kernels/vanilla_attention.md#VanillaAttn.forward_native), [`forward_ragged`](kernels/gated_delta_rule.md#GatedDeltaRuleOp.forward_ragged), [`BaseOperation`](_base_operation.md#BaseOperation), [`grouped_gdr_decode_shard_map_pallas`](kernels/gated_delta_rule.md#GatedDeltaRuleOp.grouped_gdr_decode_shard_map_pallas), [`VanillaAttn`](kernels/vanilla_attention.md#VanillaAttn), [`forward_native`](kernels/inference_gdn.md#RaggedGatedDeltaRule.forward_native), [`ScaledDotProductAttn`](kernels/scaled_dot_product_attention.md#ScaledDotProductAttn), [`GatedDeltaRuleOp`](kernels/gated_delta_rule.md#GatedDeltaRuleOp), [`RaggedGatedDeltaRule`](kernels/inference_gdn.md#RaggedGatedDeltaRule), [`_RaggedPageAttn`](kernels/ragged_page_attention.md#_RaggedPageAttn), [`KernelDeltaAttnOp`](kernels/kda.md#KernelDeltaAttnOp), [`RingAttn`](kernels/ring_attention.md#RingAttn), [`SSM1Op`](kernels/ssm1.md#SSM1Op), [`SSM2Op`](kernels/ssm2.md#SSM2Op), [`AutoRegressiveDecodeAttn`](kernels/decode_attention.md#AutoRegressiveDecodeAttn), [`PagedFlashAttn`](kernels/paged_flash_attention.md#PagedFlashAttn), [`UnifiedAttn`](kernels/unified_attention.md#UnifiedAttn), [`BlockSparseAttn`](kernels/blocksparse_attention.md#BlockSparseAttn), [`FlashAttn`](kernels/flash_attention.md#FlashAttn), [`GlmMoeDsaIndexerOp`](kernels/glm_moe_dsa_indexer.md#GlmMoeDsaIndexerOp), [`MultiLatentRaggedPageAttn`](kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttn), [`RaggedCausalConv1D`](kernels/inference_conv1d.md#RaggedCausalConv1D), [`MultiLatentRaggedPageAttnV2`](kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttnV2), [`get_impl_metadata`](_base_operation.md#BaseOperation.get_impl_metadata)

### `OperationOutput`
- def: [`easydel/operations/_operation_impl.py:90`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L90)
- doc: This dataclass encapsulates the results computation
- signature: `class OperationOutput:`
- uses (calls/refs, reference-scoped): [`AttentionOutput`](_attention_outputs.md#AttentionOutput), [`GlmMoeDsaIndexerOutput`](kernels/glm_moe_dsa_indexer.md#GlmMoeDsaIndexerOutput)
- used by: [`AttentionOutput`](_attention_outputs.md#AttentionOutput), [`GlmMoeDsaIndexerOutput`](kernels/glm_moe_dsa_indexer.md#GlmMoeDsaIndexerOutput)

## Module values
- `BATCH` — [`L78`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L78)
- `BIAS_HEAD_SEQ` — [`L85`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L85)
- `BIAS_KV_SEQ` — [`L86`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L86)
- `HEAD` — [`L81`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L81)
- `HEAD_DIM` — [`L83`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L83)
- `KV_HEAD` — [`L82`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L82)
- `KV_HEAD_DIM` — [`L84`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L84)
- `KV_LENGTH` — [`L80`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L80)
- `NOT_GIVEN` — [`L76`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L76)
- `QUERY_LENGTH` — [`L79`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L79)
- `RUNTIME_MODE_TYPES` — [`L77`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L77)
- `__all__` — [`L69`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L69)
- `logger` — [`L73`](../../../../../../raw/code/EasyDeL/easydel/operations/_operation_impl.py#L73)

