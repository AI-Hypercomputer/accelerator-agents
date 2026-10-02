---
title: 'Module: easydel/operations/requirements/requirements.py'
type: catalog
provenance: extracted
module: easydel/operations/requirements/requirements.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.operations.requirements.requirements`/
symbols:
  OperationRequirements: OperationRequirements#
  OperationRequirements.cache: OperationRequirements#cache.
  OperationRequirements.create: OperationRequirements#create().
  CacheRequirements.supported: CacheRequirements#supported.
  OperationRequirements.default: OperationRequirements#default().
  OperationRequirements.with_requires_cache: OperationRequirements#with_requires_cache().
  MetadataRequirements.required: MetadataRequirements#required.
  ModeSpecificRequirements.get: ModeSpecificRequirements#get().
  CacheRequirements.cache_view_class: CacheRequirements#cache_view_class.
  OperationRequirements.metadata: OperationRequirements#metadata.
  MetadataRequirements.optional: MetadataRequirements#optional.
  CacheRequirements.preferred: CacheRequirements#preferred.
  MetadataRequirements: MetadataRequirements#
  CacheRequirements.requires_cache: CacheRequirements#requires_cache.
  CacheRequirements.__or__: CacheRequirements#__or__().
  OperationRequirements.name: OperationRequirements#name.
  CacheRequirements: CacheRequirements#
  OperationRequirements.__or__: OperationRequirements#__or__().
  CacheRequirements.is_compatible_with: CacheRequirements#is_compatible_with().
  ModeSpecificRequirements.prefill: ModeSpecificRequirements#prefill.
  ModeSpecificRequirements.decode: ModeSpecificRequirements#decode.
  MetadataRequirements.__or__: MetadataRequirements#__or__().
  MetadataRequirements.__and__: MetadataRequirements#__and__().
  MetadataRequirements.all_fields: MetadataRequirements#all_fields().
  MetadataRequirements.is_satisfied_by: MetadataRequirements#is_satisfied_by().
  MetadataRequirements.missing_fields: MetadataRequirements#missing_fields().
  ModeSpecificRequirements: ModeSpecificRequirements#
  ModeSpecificRequirements.mixed: ModeSpecificRequirements#mixed.
  ModeSpecificRequirements.uniform: ModeSpecificRequirements#uniform().
  __all__: __all__.
---
# Module: [`easydel/operations/requirements/requirements.py`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py)

## Classes
### `CacheRequirements`
- def: [`easydel/operations/requirements/requirements.py:81`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L81)
- doc: Requirements for cache types.
- signature: `class CacheRequirements:`
- members:
  - `__or__(self, other: CacheRequirements)` — [`L98`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L98) — Union of two cache requirements (intersection of supported types).
  - `is_compatible_with(self, cache_type: CacheType)` — [`L123`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L123) — Check if a cache type is compatible with these requirements.
  - `cache_view_class` — [`L96`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L96)
  - `preferred` — [`L94`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L94)
  - `requires_cache` — [`L95`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L95)
  - `supported` — [`L93`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L93)
- uses (calls/refs, reference-scoped): [`CacheType`](types.md#CacheType), [`any`](types.md#CacheType.any), [`is_compatible_with`](types.md#CacheType.is_compatible_with)
- used by: [`_get_operations_cache_info_from_config`](../../infra/mixins/operation_cache.md#OperationCacheMixin._get_operations_cache_info_from_config), [`get_operations_cache_info_dynamic`](../../infra/mixins/operation_cache.md#OperationCacheMixin.get_operations_cache_info_dynamic), [`build`](builder.md#RequirementsBuilder.build), [`cache`](requirements.md#OperationRequirements.cache), [`create`](requirements.md#OperationRequirements.create), [`get_operations_cache_view`](../../infra/mixins/operation_cache.md#OperationCacheMixin.get_operations_cache_view), [`validate_cache_compatibility`](validation.md#validate_cache_compatibility), [`get_required_cache_class`](../../infra/mixins/operation_cache.md#OperationCacheMixin.get_required_cache_class), [`get_operations_cache_info_by_slot`](../../infra/mixins/operation_cache.md#OperationCacheMixin.get_operations_cache_info_by_slot), [`with_requires_cache`](requirements.md#OperationRequirements.with_requires_cache), [`get_common_cache_types`](validation.md#RequirementsValidator.get_common_cache_types), [`get_unique_cache_view_classes`](../../infra/mixins/operation_cache.md#OperationCacheMixin.get_unique_cache_view_classes), [`get_operations_cache_view_by_slot`](../../infra/mixins/operation_cache.md#OperationCacheMixin.get_operations_cache_view_by_slot), [`requires_cache`](../executor.md#OperationExecutor.requires_cache)

### `MetadataRequirements`
- def: [`easydel/operations/requirements/requirements.py:27`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L27)
- doc: Requirements for metadata fields.
- signature: `class MetadataRequirements:`
- members:
  - `__and__(self, other: MetadataRequirements)` — [`L45`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L45) — Intersection of two metadata requirements.
  - `__or__(self, other: MetadataRequirements)` — [`L38`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L38) — Union of two metadata requirements.
  - `all_fields(self)` — [`L53`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L53) — All fields (required + optional).
  - `is_satisfied_by(self, available: MetadataField)` — [`L57`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L57) — Check if all required fields are available.
  - `missing_fields(self, available: MetadataField)` — [`L68`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L68) — Get required fields that are missing from available.
  - `optional` — [`L36`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L36)
  - `required` — [`L35`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L35)
- uses (calls/refs, reference-scoped): [`MetadataField`](types.md#MetadataField), [`NONE`](types.md#MetadataField.NONE)
- used by: [`_get_operations_cache_info_from_config`](../../infra/mixins/operation_cache.md#OperationCacheMixin._get_operations_cache_info_from_config), [`_build_cache_info_from_layers`](../../infra/mixins/operation_cache.md#OperationCacheMixin._build_cache_info_from_layers), [`build`](builder.md#RequirementsBuilder.build), [`create`](requirements.md#OperationRequirements.create), [`validate_metadata_availability`](validation.md#validate_metadata_availability), [`metadata`](requirements.md#OperationRequirements.metadata), [`get_combined_metadata_requirements`](validation.md#RequirementsValidator.get_combined_metadata_requirements)

### `ModeSpecificRequirements`
- def: [`easydel/operations/requirements/requirements.py:242`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L242)
- doc: Requirements that vary by execution mode.
- signature: `class ModeSpecificRequirements:`
- members:
  - `get(self, mode: ExecutionMode)` — [`L258`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L258) — Get requirements for a specific execution mode.
  - `uniform(cls, requirements: OperationRequirements)` — [`L278`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L278) — Create mode-specific requirements with same requirements for all modes.
  - `decode` — [`L255`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L255)
  - `mixed` — [`L256`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L256)
  - `prefill` — [`L254`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L254)
- uses (calls/refs, reference-scoped): [`ExecutionMode`](types.md#ExecutionMode), [`OperationRequirements`](requirements.md#OperationRequirements), [`default`](requirements.md#OperationRequirements.default), [`PREFILL`](types.md#ExecutionMode.PREFILL), [`DECODE`](types.md#ExecutionMode.DECODE)
- used by: [`build`](builder.md#ModeSpecificBuilder.build), [`get`](builder.md#ModeSpecificBuilder.get)

### `OperationRequirements`
- def: [`easydel/operations/requirements/requirements.py:136`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L136)
- doc: Complete requirements for an operation.
- signature: `class OperationRequirements:`
- members:
  - `__or__(self, other: OperationRequirements)` — [`L151`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L151) — Union of two operation requirements.
  - `create(cls, name: str = "", required_metadata: MetadataField = MetadataField.NONE, optional_metadata: MetadataField = MetadataField.NONE, supported_cache: CacheType | None = None, preferred_cache: CacheType | None = None, requires_cache: bool = True, cache_view_class: type | None = None)` — [`L160`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L160) — Convenience factory method for creating requirements.
  - `default(cls, name: str = "")` — [`L201`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L201) — Create default requirements (basic metadata, any cache).
  - `with_requires_cache(self, requires_cache: bool)` — [`L216`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L216) — Create a copy with a modified requires_cache value.
  - `cache` — [`L148`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L148)
  - `metadata` — [`L147`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L147)
  - `name` — [`L149`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L149)
- uses (calls/refs, reference-scoped): [`MetadataField`](types.md#MetadataField), [`CacheType`](types.md#CacheType), [`supported`](requirements.md#CacheRequirements.supported), [`any`](types.md#CacheType.any), [`basic`](types.md#MetadataField.basic), [`required`](requirements.md#MetadataRequirements.required), [`cache_view_class`](requirements.md#CacheRequirements.cache_view_class), [`optional`](requirements.md#MetadataRequirements.optional), [`preferred`](requirements.md#CacheRequirements.preferred), [`MetadataRequirements`](requirements.md#MetadataRequirements), [`requires_cache`](requirements.md#CacheRequirements.requires_cache), [`CacheRequirements`](requirements.md#CacheRequirements), [`NONE`](types.md#MetadataField.NONE)
- used by: [`_get_operations_cache_info_from_config`](../../infra/mixins/operation_cache.md#OperationCacheMixin._get_operations_cache_info_from_config), [`get_operations_cache_info_dynamic`](../../infra/mixins/operation_cache.md#OperationCacheMixin.get_operations_cache_info_dynamic), [`get_requirements`](../_base_operation.md#BaseOperation.get_requirements), [`_build_cache_info_from_layers`](../../infra/mixins/operation_cache.md#OperationCacheMixin._build_cache_info_from_layers), [`build`](builder.md#RequirementsBuilder.build), [`get_requirements`](../kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttn.get_requirements), [`get_requirements`](../kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttnV2.get_requirements), [`get_requirements`](../kernels/ragged_page_attention.md#RaggedPageAttnV2.get_requirements), [`get_requirements`](../kernels/ragged_page_attention.md#RaggedPageAttnV3.get_requirements), [`get_requirements`](../kernels/unified_attention.md#UnifiedAttn.get_requirements), [`get_requirements`](../kernels/gated_delta_rule.md#GatedDeltaRuleOp.get_requirements), [`get_requirements`](../kernels/kda.md#KernelDeltaAttnOp.get_requirements), [`get_requirements`](../kernels/ssm1.md#SSM1Op.get_requirements), [`get_requirements`](../kernels/ssm2.md#SSM2Op.get_requirements), [`get_requirements`](../kernels/inference_gdn.md#RaggedGatedDeltaRule.get_requirements), [`get_requirements`](../kernels/inference_conv1d.md#RaggedCausalConv1D.get_requirements), [`get_operations_cache_view`](../../infra/mixins/operation_cache.md#OperationCacheMixin.get_operations_cache_view), [`get_requirements`](../kernels/glm_moe_dsa_indexer.md#GlmMoeDsaIndexerOp.get_requirements), [`get_requirements`](../kernels/paged_flash_attention.md#PagedFlashAttn.get_requirements), [`validate_metadata_availability`](validation.md#validate_metadata_availability), [`get_requirements`](../kernels/decode_attention.md#AutoRegressiveDecodeAttn.get_requirements), [`validate_cache_compatibility`](validation.md#validate_cache_compatibility), [`validate`](validation.md#RequirementsValidator.validate), [`get_required_cache_class`](../../infra/mixins/operation_cache.md#OperationCacheMixin.get_required_cache_class), [`get_requirements`](../kernels/blocksparse_attention.md#BlockSparseAttn.get_requirements), [`get_requirements`](../kernels/flash_attention.md#FlashAttn.get_requirements), [`get_requirements`](../kernels/ring_attention.md#RingAttn.get_requirements), [`get_requirements`](../kernels/scaled_dot_product_attention.md#ScaledDotProductAttn.get_requirements), [`get_requirements`](../kernels/vanilla_attention.md#VanillaAttn.get_requirements), [`get_operations_cache_info_by_slot`](../../infra/mixins/operation_cache.md#OperationCacheMixin.get_operations_cache_info_by_slot), [`get_combined_requirements`](../executor.md#OperationExecutor.get_combined_requirements), [`get`](requirements.md#ModeSpecificRequirements.get), [`_get_operation_requirements`](../../infra/mixins/operation_cache.md#OperationCacheMixin._get_operation_requirements), [`get_instance_requirements`](../_operation_impl.md#OperationImpl.get_instance_requirements), [`validate_all`](validation.md#RequirementsValidator.validate_all), [`get_requirements`](../executor.md#OperationExecutor.get_requirements), [`requirements`](../../infra/mixins/operation_cache.md#LayerOperationInfo.requirements), [`get_combined_metadata_requirements`](validation.md#RequirementsValidator.get_combined_metadata_requirements), [`get_common_cache_types`](validation.md#RequirementsValidator.get_common_cache_types), [`get_unique_cache_view_classes`](../../infra/mixins/operation_cache.md#OperationCacheMixin.get_unique_cache_view_classes)  (+8 more)

## Module values
- `__all__` — [`L23`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/requirements.py#L23)

