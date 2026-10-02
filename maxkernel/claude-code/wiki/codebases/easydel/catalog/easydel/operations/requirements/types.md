---
title: 'Module: easydel/operations/requirements/types.py'
type: catalog
provenance: extracted
module: easydel/operations/requirements/types.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.operations.requirements.types`/
symbols:
  MetadataField: MetadataField#
  CacheType: CacheType#
  ExecutionMode: ExecutionMode#
  ExecutionMode.MIXED: ExecutionMode#MIXED.
  CacheType.any: CacheType#any().
  MetadataField.basic: MetadataField#basic().
  CacheType.TRANSFORMER: CacheType#TRANSFORMER.
  CacheType.RAGGED_PAGES: CacheType#RAGGED_PAGES.
  CacheType.HYBRID: CacheType#HYBRID.
  CacheType.RECURRENT: CacheType#RECURRENT.
  MetadataField.POSITIONS: MetadataField#POSITIONS.
  MetadataField.SEQ_LENS: MetadataField#SEQ_LENS.
  MetadataField.ragged: MetadataField#ragged().
  MetadataField.NONE: MetadataField#NONE.
  MetadataField.LOGITS_INDICES: MetadataField#LOGITS_INDICES.
  MetadataField.paged_v2: MetadataField#paged_v2().
  MetadataField.paged_v3: MetadataField#paged_v3().
  MetadataField.recurrent: MetadataField#recurrent().
  CacheType.attention: CacheType#attention().
  ExecutionMode.PREFILL: ExecutionMode#PREFILL.
  MetadataField.CONTEXT_LENS: MetadataField#CONTEXT_LENS.
  MetadataField.PAGES_TABLES: MetadataField#PAGES_TABLES.
  MetadataField.STATE_INDICES: MetadataField#STATE_INDICES.
  ExecutionMode.DECODE: ExecutionMode#DECODE.
  MetadataField.QUERY_START_LOC: MetadataField#QUERY_START_LOC.
  MetadataField.HAS_INITIAL_STATE: MetadataField#HAS_INITIAL_STATE.
  MetadataField.REQUEST_DISTRIBUTION: MetadataField#REQUEST_DISTRIBUTION.
  CacheType.is_compatible_with: CacheType#is_compatible_with().
  MetadataField.SLOT_MAPPING: MetadataField#SLOT_MAPPING.
  CacheType.NONE: CacheType#NONE.
  __all__: __all__.
---
# Module: [`easydel/operations/requirements/types.py`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py)

## Classes
### `CacheType`  ·  implements/extends Flag
- def: [`easydel/operations/requirements/types.py:113`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L113) — documented in [easydel-operations-requirements-types](../../../../concepts/easydel-operations-requirements-types.md)
- doc: Cache types that operations can support.
- signature: `class CacheType(Flag):`
- members:
  - `any(cls)` — [`L135`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L135) — Any cache type - operation is cache-agnostic. — documented in [easydel-operations-requirements-types](../../../../concepts/easydel-operations-requirements-types.md)
  - `attention(cls)` — [`L140`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L140) — Cache types suitable for attention operations.
  - `is_compatible_with(self, other: CacheType)` — [`L144`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L144) — Check if this cache type is compatible with another.
  - `HYBRID` — [`L131`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L131)
  - `NONE` — [`L120`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L120)
  - `RAGGED_PAGES` — [`L125`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L125)
  - `RECURRENT` — [`L128`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L128)
  - `TRANSFORMER` — [`L122`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L122)
- used by: [`_get_operations_cache_info_from_config`](../../infra/mixins/operation_cache.md#OperationCacheMixin._get_operations_cache_info_from_config), [`get_operations_cache_info_dynamic`](../../infra/mixins/operation_cache.md#OperationCacheMixin.get_operations_cache_info_dynamic), [`get_requirements`](../_base_operation.md#BaseOperation.get_requirements), [`_build_cache_info_from_layers`](../../infra/mixins/operation_cache.md#OperationCacheMixin._build_cache_info_from_layers), [`get_requirements`](../kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttn.get_requirements), [`get_requirements`](../kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttnV2.get_requirements), [`get_requirements`](../kernels/ragged_page_attention.md#RaggedPageAttnV2.get_requirements), [`get_requirements`](../kernels/ragged_page_attention.md#RaggedPageAttnV3.get_requirements), [`get_requirements`](../kernels/unified_attention.md#UnifiedAttn.get_requirements), [`get_requirements`](../kernels/gated_delta_rule.md#GatedDeltaRuleOp.get_requirements), [`get_requirements`](../kernels/kda.md#KernelDeltaAttnOp.get_requirements), [`get_requirements`](../kernels/ssm1.md#SSM1Op.get_requirements), [`get_requirements`](../kernels/ssm2.md#SSM2Op.get_requirements), [`get_requirements`](../kernels/inference_gdn.md#RaggedGatedDeltaRule.get_requirements), [`create`](requirements.md#OperationRequirements.create), [`supported`](requirements.md#CacheRequirements.supported), [`get_requirements`](../kernels/inference_conv1d.md#RaggedCausalConv1D.get_requirements), [`get_requirements`](../kernels/glm_moe_dsa_indexer.md#GlmMoeDsaIndexerOp.get_requirements), [`get_requirements`](../kernels/paged_flash_attention.md#PagedFlashAttn.get_requirements), [`get_requirements`](../kernels/decode_attention.md#AutoRegressiveDecodeAttn.get_requirements), [`validate_cache_compatibility`](validation.md#validate_cache_compatibility), [`get_requirements`](../kernels/blocksparse_attention.md#BlockSparseAttn.get_requirements), [`get_requirements`](../kernels/flash_attention.md#FlashAttn.get_requirements), [`get_requirements`](../kernels/ring_attention.md#RingAttn.get_requirements), [`get_requirements`](../kernels/scaled_dot_product_attention.md#ScaledDotProductAttn.get_requirements), [`get_requirements`](../kernels/vanilla_attention.md#VanillaAttn.get_requirements), [`default`](requirements.md#OperationRequirements.default), [`support_cache`](builder.md#RequirementsBuilder.support_cache), [`is_recurrent_layer`](../../infra/mixins/operation_cache.md#LayerOperationInfo.is_recurrent_layer), [`preferred`](requirements.md#CacheRequirements.preferred), [`get_common_cache_types`](validation.md#RequirementsValidator.get_common_cache_types), [`supported_cache_types`](../../infra/mixins/operation_cache.md#LayerOperationInfo.supported_cache_types), [`is_attention_layer`](../../infra/mixins/operation_cache.md#LayerOperationInfo.is_attention_layer), [`__init__`](validation.md#RequirementsValidator.__init__), [`prefer_cache`](builder.md#ModeSpecificBuilder.prefer_cache), [`support_cache`](builder.md#ModeSpecificBuilder.support_cache), [`is_compatible_with`](requirements.md#CacheRequirements.is_compatible_with), [`prefer_cache`](builder.md#RequirementsBuilder.prefer_cache), [`_supported_cache`](builder.md#RequirementsBuilder._supported_cache), [`add_cache_support`](builder.md#RequirementsBuilder.add_cache_support)  (+2 more)

### `ExecutionMode`  ·  implements/extends Enum
- def: [`easydel/operations/requirements/types.py:26`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L26) — documented in [easydel-operations-requirements-types](../../../../concepts/easydel-operations-requirements-types.md)
- doc: Execution mode for inference operations.
- signature: `class ExecutionMode(Enum):`
- members:
  - `DECODE` — [`L34`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L34)
  - `MIXED` — [`L35`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L35) — documented in [easydel-operations-requirements-types](../../../../concepts/easydel-operations-requirements-types.md)
  - `PREFILL` — [`L33`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L33)
- used by: [`_get_operations_cache_info_from_config`](../../infra/mixins/operation_cache.md#OperationCacheMixin._get_operations_cache_info_from_config), [`get_operations_cache_info_dynamic`](../../infra/mixins/operation_cache.md#OperationCacheMixin.get_operations_cache_info_dynamic), [`get_requirements`](../_base_operation.md#BaseOperation.get_requirements), [`get_requirements`](../kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttn.get_requirements), [`get_requirements`](../kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttnV2.get_requirements), [`get_requirements`](../kernels/ragged_page_attention.md#RaggedPageAttnV2.get_requirements), [`get_requirements`](../kernels/ragged_page_attention.md#RaggedPageAttnV3.get_requirements), [`get_requirements`](../kernels/unified_attention.md#UnifiedAttn.get_requirements), [`get_requirements`](../kernels/gated_delta_rule.md#GatedDeltaRuleOp.get_requirements), [`get_requirements`](../kernels/kda.md#KernelDeltaAttnOp.get_requirements), [`get_requirements`](../kernels/ssm1.md#SSM1Op.get_requirements), [`get_requirements`](../kernels/ssm2.md#SSM2Op.get_requirements), [`get_requirements`](../kernels/inference_gdn.md#RaggedGatedDeltaRule.get_requirements), [`get_requirements`](../kernels/inference_conv1d.md#RaggedCausalConv1D.get_requirements), [`get_requirements`](../kernels/glm_moe_dsa_indexer.md#GlmMoeDsaIndexerOp.get_requirements), [`get_requirements`](../kernels/paged_flash_attention.md#PagedFlashAttn.get_requirements), [`get_requirements`](../kernels/decode_attention.md#AutoRegressiveDecodeAttn.get_requirements), [`get_requirements`](../kernels/blocksparse_attention.md#BlockSparseAttn.get_requirements), [`get_requirements`](../kernels/flash_attention.md#FlashAttn.get_requirements), [`get_requirements`](../kernels/ring_attention.md#RingAttn.get_requirements), [`get_requirements`](../kernels/scaled_dot_product_attention.md#ScaledDotProductAttn.get_requirements), [`get_requirements`](../kernels/vanilla_attention.md#VanillaAttn.get_requirements), [`get_combined_requirements`](../executor.md#OperationExecutor.get_combined_requirements), [`get_operations_cache_info`](../../infra/mixins/operation_cache.md#OperationCacheMixin.get_operations_cache_info), [`get`](requirements.md#ModeSpecificRequirements.get), [`get_operation`](../executor.md#OperationExecutor.get_operation), [`_get_operation_requirements`](../../infra/mixins/operation_cache.md#OperationCacheMixin._get_operation_requirements), [`get_instance_requirements`](../_operation_impl.md#OperationImpl.get_instance_requirements), [`get_requirements`](../executor.md#OperationExecutor.get_requirements), [`__init__`](validation.md#RequirementsValidator.__init__), [`get`](builder.md#ModeSpecificBuilder.get), [`get_operation_name`](../executor.md#OperationExecutor.get_operation_name)

### `MetadataField`  ·  implements/extends Flag
- def: [`easydel/operations/requirements/types.py:38`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L38) — documented in [easydel-operations-requirements-types](../../../../concepts/easydel-operations-requirements-types.md)
- doc: Metadata fields that operations can require.
- signature: `class MetadataField(Flag):`
- members:
  - `basic(cls)` — [`L88`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L88) — Basic metadata for simple attention operations. — documented in [easydel-operations-requirements-types](../../../../concepts/easydel-operations-requirements-types.md)
  - `paged_v2(cls)` — [`L98`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L98) — Metadata for RPA v2 paged attention. — documented in [easydel-operations-requirements-types](../../../../concepts/easydel-operations-requirements-types.md)
  - `paged_v3(cls)` — [`L103`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L103) — Metadata for RPA v3 paged attention.
  - `ragged(cls)` — [`L93`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L93) — Metadata for ragged batch format. — documented in [easydel-operations-requirements-types](../../../../concepts/easydel-operations-requirements-types.md)
  - `recurrent(cls)` — [`L108`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L108) — Metadata for recurrent/state space models.
  - `CONTEXT_LENS` — [`L57`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L57)
  - `HAS_INITIAL_STATE` — [`L77`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L77)
  - `LOGITS_INDICES` — [`L84`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L84)
  - `NONE` — [`L51`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L51)
  - `PAGES_TABLES` — [`L67`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L67)
  - `POSITIONS` — [`L60`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L60)
  - `QUERY_START_LOC` — [`L64`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L64)
  - `REQUEST_DISTRIBUTION` — [`L73`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L73)
  - `SEQ_LENS` — [`L54`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L54)
  - `SLOT_MAPPING` — [`L70`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L70)
  - `STATE_INDICES` — [`L80`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L80)
- used by: [`_get_operations_cache_info_from_config`](../../infra/mixins/operation_cache.md#OperationCacheMixin._get_operations_cache_info_from_config), [`get_requirements`](../_base_operation.md#BaseOperation.get_requirements), [`_build_cache_info_from_layers`](../../infra/mixins/operation_cache.md#OperationCacheMixin._build_cache_info_from_layers), [`get_requirements`](../kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttn.get_requirements), [`get_requirements`](../kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttnV2.get_requirements), [`get_requirements`](../kernels/ragged_page_attention.md#RaggedPageAttnV2.get_requirements), [`get_requirements`](../kernels/ragged_page_attention.md#RaggedPageAttnV3.get_requirements), [`get_requirements`](../kernels/unified_attention.md#UnifiedAttn.get_requirements), [`get_requirements`](../kernels/gated_delta_rule.md#GatedDeltaRuleOp.get_requirements), [`get_requirements`](../kernels/kda.md#KernelDeltaAttnOp.get_requirements), [`get_requirements`](../kernels/ssm1.md#SSM1Op.get_requirements), [`get_requirements`](../kernels/ssm2.md#SSM2Op.get_requirements), [`get_requirements`](../kernels/inference_gdn.md#RaggedGatedDeltaRule.get_requirements), [`create`](requirements.md#OperationRequirements.create), [`get_requirements`](../kernels/inference_conv1d.md#RaggedCausalConv1D.get_requirements), [`get_requirements`](../kernels/glm_moe_dsa_indexer.md#GlmMoeDsaIndexerOp.get_requirements), [`get_requirements`](../kernels/paged_flash_attention.md#PagedFlashAttn.get_requirements), [`validate_metadata_availability`](validation.md#validate_metadata_availability), [`get_requirements`](../kernels/decode_attention.md#AutoRegressiveDecodeAttn.get_requirements), [`get_requirements`](../kernels/blocksparse_attention.md#BlockSparseAttn.get_requirements), [`get_requirements`](../kernels/flash_attention.md#FlashAttn.get_requirements), [`get_requirements`](../kernels/ring_attention.md#RingAttn.get_requirements), [`get_requirements`](../kernels/scaled_dot_product_attention.md#ScaledDotProductAttn.get_requirements), [`get_requirements`](../kernels/vanilla_attention.md#VanillaAttn.get_requirements), [`default`](requirements.md#OperationRequirements.default), [`require_metadata`](builder.md#RequirementsBuilder.require_metadata), [`required`](requirements.md#MetadataRequirements.required), [`optional_metadata`](builder.md#RequirementsBuilder.optional_metadata), [`optional`](requirements.md#MetadataRequirements.optional), [`get_combined_metadata_requirements`](validation.md#RequirementsValidator.get_combined_metadata_requirements), [`__init__`](validation.md#RequirementsValidator.__init__), [`optional_metadata`](builder.md#ModeSpecificBuilder.optional_metadata), [`require_metadata`](builder.md#ModeSpecificBuilder.require_metadata), [`get_metadata_field_names`](validation.md#get_metadata_field_names), [`all_fields`](requirements.md#MetadataRequirements.all_fields), [`_optional_metadata`](builder.md#RequirementsBuilder._optional_metadata), [`_required_metadata`](builder.md#RequirementsBuilder._required_metadata), [`combined_metadata`](../../infra/mixins/operation_cache.md#OperationsCacheInfo.combined_metadata), [`is_satisfied_by`](requirements.md#MetadataRequirements.is_satisfied_by), [`missing_fields`](requirements.md#MetadataRequirements.missing_fields)

## Module values
- `__all__` — [`L19`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/types.py#L19)

