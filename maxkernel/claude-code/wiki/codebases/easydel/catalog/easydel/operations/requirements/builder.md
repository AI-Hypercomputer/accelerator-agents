---
title: 'Module: easydel/operations/requirements/builder.py'
type: catalog
provenance: extracted
module: easydel/operations/requirements/builder.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.operations.requirements.builder`/
symbols:
  RequirementsBuilder.build: RequirementsBuilder#build().
  RequirementsBuilder: RequirementsBuilder#
  RequirementsBuilder.require_metadata: RequirementsBuilder#require_metadata().
  RequirementsBuilder.support_cache: RequirementsBuilder#support_cache().
  ModeSpecificBuilder.build: ModeSpecificBuilder#build().
  RequirementsBuilder.optional_metadata: RequirementsBuilder#optional_metadata().
  RequirementsBuilder.use_cache_view: RequirementsBuilder#use_cache_view().
  ModeSpecificBuilder._current_builder: ModeSpecificBuilder#_current_builder.
  ModeSpecificBuilder.for_mixed: ModeSpecificBuilder#for_mixed().
  ModeSpecificBuilder: ModeSpecificBuilder#
  ModeSpecificBuilder.require_metadata: ModeSpecificBuilder#require_metadata().
  ModeSpecificBuilder.optional_metadata: ModeSpecificBuilder#optional_metadata().
  ModeSpecificBuilder.support_cache: ModeSpecificBuilder#support_cache().
  ModeSpecificBuilder.prefer_cache: ModeSpecificBuilder#prefer_cache().
  ModeSpecificBuilder.get: ModeSpecificBuilder#get().
  RequirementsBuilder.prefer_cache: RequirementsBuilder#prefer_cache().
  RequirementsBuilder._supported_cache: RequirementsBuilder#_supported_cache.
  RequirementsBuilder.add_cache_support: RequirementsBuilder#add_cache_support().
  ModeSpecificBuilder._mixed_builder: ModeSpecificBuilder#_mixed_builder.
  ModeSpecificBuilder.for_prefill: ModeSpecificBuilder#for_prefill().
  ModeSpecificBuilder.for_decode: ModeSpecificBuilder#for_decode().
  ModeSpecificBuilder.requires_cache: ModeSpecificBuilder#requires_cache().
  ModeSpecificBuilder.no_cache_required: ModeSpecificBuilder#no_cache_required().
  ModeSpecificBuilder.use_cache_view: ModeSpecificBuilder#use_cache_view().
  RequirementsBuilder._required_metadata: RequirementsBuilder#_required_metadata.
  RequirementsBuilder._optional_metadata: RequirementsBuilder#_optional_metadata.
  RequirementsBuilder.requires_cache: RequirementsBuilder#requires_cache().
  RequirementsBuilder.no_cache_required: RequirementsBuilder#no_cache_required().
  ModeSpecificBuilder._prefill_builder: ModeSpecificBuilder#_prefill_builder.
  RequirementsBuilder._preferred_cache: RequirementsBuilder#_preferred_cache.
  ModeSpecificBuilder._decode_builder: ModeSpecificBuilder#_decode_builder.
  RequirementsBuilder._requires_cache: RequirementsBuilder#_requires_cache.
  RequirementsBuilder._cache_view_class: RequirementsBuilder#_cache_view_class.
  RequirementsBuilder._name: RequirementsBuilder#_name.
  ModeSpecificBuilder._name: ModeSpecificBuilder#_name.
  __all__: __all__.
  RequirementsBuilder.__init__: RequirementsBuilder#__init__().
  ModeSpecificBuilder.__init__: ModeSpecificBuilder#__init__().
---
# Module: [`easydel/operations/requirements/builder.py`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py)

## Classes
### `ModeSpecificBuilder`
- def: [`easydel/operations/requirements/builder.py:179`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L179)
- doc: Builder for mode-specific requirements.
- signature: `class ModeSpecificBuilder:`
- members:
  - `__init__(self, name: str = "")` — [`L193`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L193) — Initialize the builder.
  - `build(self)` — [`L317`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L317) — Build the ModeSpecificRequirements instance.
  - `for_decode(self)` — [`L214`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L214) — Switch to configuring decode requirements.
  - `for_mixed(self)` — [`L223`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L223) — Switch to configuring mixed mode requirements.
  - `for_prefill(self)` — [`L205`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L205) — Switch to configuring prefill requirements.
  - `get(self, mode: ExecutionMode)` — [`L329`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L329) — Build and get requirements for a specific mode.
  - `no_cache_required(self)` — [`L296`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L296) — Mark current mode as not requiring cache.
  - `optional_metadata(self, fields: MetadataField)` — [`L248`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L248) — Add optional metadata fields for current mode.
  - `prefer_cache(self, cache_type: CacheType)` — [`L272`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L272) — Set preferred cache type for current mode.
  - `require_metadata(self, fields: MetadataField)` — [`L236`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L236) — Add required metadata fields for current mode.
  - `requires_cache(self, value: bool = True)` — [`L284`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L284) — Set whether current mode requires cache.
  - `support_cache(self, cache_types: CacheType)` — [`L260`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L260) — Set supported cache types for current mode.
  - `use_cache_view(self, cache_view_class: type)` — [`L305`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L305) — Set the cache view class for current mode.
- protocol/private: `_current_builder`[`L203`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L203), `_decode_builder`[`L201`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L201), `_mixed_builder`[`L202`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L202), `_name`[`L199`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L199), `_prefill_builder`[`L200`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L200)
- uses (calls/refs, reference-scoped): [`MetadataField`](types.md#MetadataField), [`CacheType`](types.md#CacheType), [`ExecutionMode`](types.md#ExecutionMode), [`OperationRequirements`](requirements.md#OperationRequirements), [`build`](builder.md#RequirementsBuilder.build), [`RequirementsBuilder`](builder.md#RequirementsBuilder), [`require_metadata`](builder.md#RequirementsBuilder.require_metadata), [`support_cache`](builder.md#RequirementsBuilder.support_cache), [`get`](requirements.md#ModeSpecificRequirements.get), [`optional_metadata`](builder.md#RequirementsBuilder.optional_metadata), [`use_cache_view`](builder.md#RequirementsBuilder.use_cache_view), [`prefer_cache`](builder.md#RequirementsBuilder.prefer_cache), [`decode`](requirements.md#ModeSpecificRequirements.decode), [`prefill`](requirements.md#ModeSpecificRequirements.prefill), [`requires_cache`](builder.md#RequirementsBuilder.requires_cache), [`ModeSpecificRequirements`](requirements.md#ModeSpecificRequirements), [`no_cache_required`](builder.md#RequirementsBuilder.no_cache_required), [`mixed`](requirements.md#ModeSpecificRequirements.mixed)

### `RequirementsBuilder`
- def: [`easydel/operations/requirements/builder.py:30`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L30)
- doc: Fluent builder for creating OperationRequirements.
- signature: `class RequirementsBuilder:`
- members:
  - `__init__(self, name: str = "")` — [`L43`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L43) — Initialize the builder.
  - `add_cache_support(self, cache_types: CacheType)` — [`L96`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L96) — Add additional supported cache types.
  - `build(self)` — [`L158`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L158) — Build the OperationRequirements instance.
  - `no_cache_required(self)` — [`L132`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L132) — Mark this operation as not requiring cache.
  - `optional_metadata(self, fields: MetadataField)` — [`L69`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L69) — Add optional metadata fields.
  - `prefer_cache(self, cache_type: CacheType)` — [`L108`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L108) — Set preferred cache type.
  - `require_metadata(self, fields: MetadataField)` — [`L57`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L57) — Add required metadata fields.
  - `requires_cache(self, value: bool = True)` — [`L120`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L120) — Set whether this operation requires cache.
  - `support_cache(self, cache_types: CacheType)` — [`L81`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L81) — Set supported cache types.
  - `use_cache_view(self, cache_view_class: type)` — [`L143`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L143) — Set the cache view class this operation requires.
- protocol/private: `_cache_view_class`[`L55`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L55), `_name`[`L49`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L49), `_optional_metadata`[`L51`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L51), `_preferred_cache`[`L53`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L53), `_required_metadata`[`L50`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L50), `_requires_cache`[`L54`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L54), `_supported_cache`[`L52`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L52)
- uses (calls/refs, reference-scoped): [`MetadataField`](types.md#MetadataField), [`CacheType`](types.md#CacheType), [`OperationRequirements`](requirements.md#OperationRequirements), [`cache`](requirements.md#OperationRequirements.cache), [`supported`](requirements.md#CacheRequirements.supported), [`any`](types.md#CacheType.any), [`required`](requirements.md#MetadataRequirements.required), [`cache_view_class`](requirements.md#CacheRequirements.cache_view_class), [`metadata`](requirements.md#OperationRequirements.metadata), [`optional`](requirements.md#MetadataRequirements.optional), [`preferred`](requirements.md#CacheRequirements.preferred), [`MetadataRequirements`](requirements.md#MetadataRequirements), [`requires_cache`](requirements.md#CacheRequirements.requires_cache), [`name`](requirements.md#OperationRequirements.name), [`CacheRequirements`](requirements.md#CacheRequirements), [`NONE`](types.md#MetadataField.NONE)
- used by: [`get_requirements`](../kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttn.get_requirements), [`get_requirements`](../kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttnV2.get_requirements), [`get_requirements`](../kernels/ragged_page_attention.md#RaggedPageAttnV2.get_requirements), [`get_requirements`](../kernels/ragged_page_attention.md#RaggedPageAttnV3.get_requirements), [`get_requirements`](../kernels/unified_attention.md#UnifiedAttn.get_requirements), [`get_requirements`](../kernels/gated_delta_rule.md#GatedDeltaRuleOp.get_requirements), [`get_requirements`](../kernels/kda.md#KernelDeltaAttnOp.get_requirements), [`get_requirements`](../kernels/ssm1.md#SSM1Op.get_requirements), [`get_requirements`](../kernels/ssm2.md#SSM2Op.get_requirements), [`get_requirements`](../kernels/inference_gdn.md#RaggedGatedDeltaRule.get_requirements), [`get_requirements`](../kernels/inference_conv1d.md#RaggedCausalConv1D.get_requirements), [`get_requirements`](../kernels/glm_moe_dsa_indexer.md#GlmMoeDsaIndexerOp.get_requirements), [`get_requirements`](../kernels/paged_flash_attention.md#PagedFlashAttn.get_requirements), [`build`](builder.md#ModeSpecificBuilder.build), [`_current_builder`](builder.md#ModeSpecificBuilder._current_builder), [`for_mixed`](builder.md#ModeSpecificBuilder.for_mixed), [`optional_metadata`](builder.md#ModeSpecificBuilder.optional_metadata), [`prefer_cache`](builder.md#ModeSpecificBuilder.prefer_cache), [`require_metadata`](builder.md#ModeSpecificBuilder.require_metadata), [`support_cache`](builder.md#ModeSpecificBuilder.support_cache), [`_mixed_builder`](builder.md#ModeSpecificBuilder._mixed_builder), [`no_cache_required`](builder.md#ModeSpecificBuilder.no_cache_required), [`requires_cache`](builder.md#ModeSpecificBuilder.requires_cache), [`use_cache_view`](builder.md#ModeSpecificBuilder.use_cache_view), [`_prefill_builder`](builder.md#ModeSpecificBuilder._prefill_builder), [`_decode_builder`](builder.md#ModeSpecificBuilder._decode_builder)

## Module values
- `__all__` — [`L27`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/builder.py#L27)

