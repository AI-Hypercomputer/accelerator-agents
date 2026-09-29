---
title: 'Module: easydel/operations/_base_operation.py'
type: catalog
provenance: extracted
module: easydel/operations/_base_operation.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.operations._base_operation`/
symbols:
  BaseOperation.metadata: BaseOperation#metadata.
  BaseOperation.get_requirements: BaseOperation#get_requirements().
  BaseOperation.get_impl_name: BaseOperation#get_impl_name().
  BaseOperation.forward_native: BaseOperation#forward_native().
  BaseOperation.forward_gpu: BaseOperation#forward_gpu().
  BaseOperation.forward_tpu: BaseOperation#forward_tpu().
  BaseOperation.__call__: BaseOperation#__call__().
  BaseOperation.forward_cpu: BaseOperation#forward_cpu().
  BaseOperation.forward_rocm: BaseOperation#forward_rocm().
  BaseOperation.forward_cuda: BaseOperation#forward_cuda().
  OperationRegistry.register: OperationRegistry#register().
  BaseOperation: BaseOperation#
  OperationRegistry.create: OperationRegistry#create().
  OperationRegistry._registry: OperationRegistry#_registry.
  OperationRegistry.get: OperationRegistry#get().
  BaseOperation.EasyDeLBackends: BaseOperation#EasyDeLBackends().
  logger: logger.
  BaseOperation.get_impl_metadata: BaseOperation#get_impl_metadata().
  _I: _I.
  BaseOperation.forward_tt: BaseOperation#forward_tt().
  OperationRegistry: OperationRegistry#
  OperationRegistry.list_implementations: OperationRegistry#list_implementations().
  __all__: __all__.
  BaseOperation.current_backend: BaseOperation#current_backend().
---
# Module: [`easydel/operations/_base_operation.py`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py)

## Classes
### `BaseOperation`  ·  implements/extends ABC
- def: [`easydel/operations/_base_operation.py:42`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L42)
- doc: Abstract Base Class for defining operations with potential backend-specific implementations.
- signature: `class BaseOperation(ABC):`
- members:
  - `EasyDeLBackends(self)` — [`L291`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L291)
  - `__call__(self, *args, **kwargs)` — [`L253`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L253) — Executes the appropriate forward method based on the detected JAX backend.
  - `current_backend(self)` — [`L130`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L130) — Returns the current JAX default backend as a lowercase string literal.
  - `forward_cpu(self, *args, **kwargs)` — [`L188`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L188) — CPU-specific implementation of the operation.
  - `forward_cuda(self, *args, **kwargs)` — [`L237`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L237) — CUDA (NVIDIA GPU)-specific implementation of the operation.
  - `forward_gpu(self, *args, **kwargs)` — [`L204`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L204) — Generic GPU-specific implementation of the operation.
  - `forward_native(self, *args, **kwargs)` — [`L140`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L140) — The core, backend-agnostic implementation of the operation.
  - `forward_rocm(self, *args, **kwargs)` — [`L221`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L221) — ROCm (AMD GPU)-specific implementation of the operation.
  - `forward_tpu(self, *args, **kwargs)` — [`L156`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L156) — TPU-specific implementation of the operation.
  - `forward_tt(self, *args, **kwargs)` — [`L172`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L172) — TT-specific implementation of the operation.
  - `get_impl_metadata(self)` — [`L76`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L76) — Returns the `OperationMetadata` associated with this implementation instance.
  - `get_impl_name(cls)` — [`L64`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L64) — Returns the unique name(s) identifying this attention implementation.
  - `get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED)` — [`L85`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L85) — Returns the operation requirements for metadata and cache types.
  - `metadata` — [`L60`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L60)
- uses (calls/refs, reference-scoped): [`MetadataField`](requirements/types.md#MetadataField), [`CacheType`](requirements/types.md#CacheType), [`ExecutionMode`](requirements/types.md#ExecutionMode), [`OperationRequirements`](requirements/requirements.md#OperationRequirements), [`OperationImpl`](_operation_impl.md#OperationImpl), [`forward_native`](kernels/paged_flash_attention.md#PagedFlashAttn.forward_native), [`forward_native`](kernels/unified_attention.md#UnifiedAttn.forward_native), [`OperationMetadata`](_operation_meta.md#OperationMetadata), [`forward_native`](kernels/blocksparse_attention.md#BlockSparseAttn.forward_native), [`get_requirements`](kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttn.get_requirements), [`get_requirements`](kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttnV2.get_requirements), [`get_requirements`](kernels/ragged_page_attention.md#RaggedPageAttnV2.get_requirements), [`get_requirements`](kernels/ragged_page_attention.md#RaggedPageAttnV3.get_requirements), [`get_requirements`](kernels/unified_attention.md#UnifiedAttn.get_requirements), [`get_requirements`](kernels/gated_delta_rule.md#GatedDeltaRuleOp.get_requirements), [`get_requirements`](kernels/kda.md#KernelDeltaAttnOp.get_requirements), [`get_requirements`](kernels/ssm1.md#SSM1Op.get_requirements), [`get_requirements`](kernels/ssm2.md#SSM2Op.get_requirements), [`forward_native`](kernels/decode_attention.md#AutoRegressiveDecodeAttn.forward_native), [`forward_native`](kernels/ring_attention.md#RingAttn.forward_native), [`get_requirements`](kernels/inference_gdn.md#RaggedGatedDeltaRule.get_requirements), [`create`](requirements/requirements.md#OperationRequirements.create), [`forward_native`](kernels/flash_attention.md#FlashAttn.forward_native), [`forward_native`](kernels/gated_delta_rule.md#GatedDeltaRuleOp.forward_native), [`check_bool_flag`](../utils/helpers.md#check_bool_flag), [`forward_native`](kernels/kda.md#KernelDeltaAttnOp.forward_native), [`get_requirements`](kernels/inference_conv1d.md#RaggedCausalConv1D.get_requirements), [`forward_native`](kernels/scaled_dot_product_attention.md#ScaledDotProductAttn.forward_native), [`forward_native`](kernels/vanilla_attention.md#VanillaAttn.forward_native), [`get_requirements`](kernels/glm_moe_dsa_indexer.md#GlmMoeDsaIndexerOp.get_requirements), [`get_requirements`](kernels/paged_flash_attention.md#PagedFlashAttn.get_requirements), [`EasyDeLBackends`](../infra/etils.md#EasyDeLBackends), [`MIXED`](requirements/types.md#ExecutionMode.MIXED), [`get_requirements`](kernels/decode_attention.md#AutoRegressiveDecodeAttn.get_requirements), [`get_requirements`](kernels/blocksparse_attention.md#BlockSparseAttn.get_requirements), [`get_requirements`](kernels/flash_attention.md#FlashAttn.get_requirements), [`get_requirements`](kernels/ring_attention.md#RingAttn.get_requirements), [`get_requirements`](kernels/scaled_dot_product_attention.md#ScaledDotProductAttn.get_requirements), [`get_requirements`](kernels/vanilla_attention.md#VanillaAttn.get_requirements), [`any`](requirements/types.md#CacheType.any)  (+115 more)
- used by: [`get_operations_cache_info_dynamic`](../infra/mixins/operation_cache.md#OperationCacheMixin.get_operations_cache_info_dynamic), [`OperationImpl`](_operation_impl.md#OperationImpl), [`forward_native`](kernels/paged_flash_attention.md#PagedFlashAttn.forward_native), [`_forward_v2_turboquant`](kernels/ragged_page_attention.md#_RaggedPageAttn._forward_v2_turboquant), [`_forward_v3_turboquant`](kernels/ragged_page_attention.md#_RaggedPageAttn._forward_v3_turboquant), [`forward_core`](kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttn.forward_core), [`forward_core`](kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttnV2.forward_core), [`_forward_v3_standard`](kernels/ragged_page_attention.md#_RaggedPageAttn._forward_v3_standard), [`forward_native`](kernels/unified_attention.md#UnifiedAttn.forward_native), [`forward_native`](kernels/blocksparse_attention.md#BlockSparseAttn.forward_native), [`create_stable_sharding`](_operation_impl.md#OperationImpl.create_stable_sharding), [`forward_native`](kernels/decode_attention.md#AutoRegressiveDecodeAttn.forward_native), [`forward_native`](kernels/ring_attention.md#RingAttn.forward_native), [`forward_v2`](kernels/ragged_page_attention.md#_RaggedPageAttn.forward_v2), [`forward_native`](kernels/flash_attention.md#FlashAttn.forward_native), [`forward_native`](kernels/gated_delta_rule.md#GatedDeltaRuleOp.forward_native), [`forward_native`](kernels/kda.md#KernelDeltaAttnOp.forward_native), [`forward_native`](kernels/scaled_dot_product_attention.md#ScaledDotProductAttn.forward_native), [`forward_native`](kernels/vanilla_attention.md#VanillaAttn.forward_native), [`register`](_base_operation.md#OperationRegistry.register), [`forward_ragged`](kernels/gated_delta_rule.md#GatedDeltaRuleOp.forward_ragged), [`get_combined_requirements`](executor.md#OperationExecutor.get_combined_requirements), [`grouped_gdr_decode_shard_map_pallas`](kernels/gated_delta_rule.md#GatedDeltaRuleOp.grouped_gdr_decode_shard_map_pallas), [`forward_native`](kernels/inference_gdn.md#RaggedGatedDeltaRule.forward_native), [`get_operation`](executor.md#OperationExecutor.get_operation), [`_get_operation_requirements`](../infra/mixins/operation_cache.md#OperationCacheMixin._get_operation_requirements), [`get_instance_requirements`](_operation_impl.md#OperationImpl.get_instance_requirements), [`_fallback_attention`](kernels/flash_attention.md#FlashAttn._fallback_attention), [`get_requirements`](executor.md#OperationExecutor.get_requirements), [`decode_operation`](executor.md#OperationExecutor.decode_operation), [`__call__`](kernels/ragged_page_attention.md#_RaggedPageAttn.__call__), [`create`](_base_operation.md#OperationRegistry.create), [`_mapped`](kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttn._mapped), [`_mapped`](kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttnV2._mapped), [`_mapped`](kernels/ragged_page_attention.md#_RaggedPageAttn._mapped), [`_mapped`](kernels/unified_attention.md#UnifiedAttn._mapped), [`grouped_gdr_decode`](kernels/gated_delta_rule.md#GatedDeltaRuleOp.grouped_gdr_decode), [`_run_vanilla_fallback`](kernels/blocksparse_attention.md#BlockSparseAttn._run_vanilla_fallback), [`prefill_impl`](executor.md#OperationExecutor.prefill_impl), [`prefill_operation`](executor.md#OperationExecutor.prefill_operation)  (+24 more)

### `OperationRegistry`
- def: [`easydel/operations/_base_operation.py:300`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L300)
- doc: Registry for discovering and managing different `OperationImpl` classes.
- signature: `class OperationRegistry:`
- members:
  - `create(cls, impl_name: str, metadata: OperationMetadata, requires_cache: bool | None = None)` — [`L377`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L377) — Creates an instance of an attention implementation by name.
  - `get(cls, impl_name: str)` — [`L353`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L353) — Retrieves an attention implementation class by its registered name.
  - `list_implementations(cls)` — [`L413`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L413) — Returns a list of names of all registered attention implementations.
  - `register(cls, impl_cls: type[_I])` — [`L311`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L311) — Class method decorator to register an `OperationImpl` subclass.
- protocol/private: `_registry`[`L308`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L308)
- uses (calls/refs, reference-scoped): [`OperationMetadata`](_operation_meta.md#OperationMetadata), [`get_impl_name`](_base_operation.md#BaseOperation.get_impl_name), [`BaseOperation`](_base_operation.md#BaseOperation), [`logger`](_base_operation.md#logger), [`_I`](_base_operation.md#_I), [`requires_cache`](_operation_meta.md#OperationMetadata.requires_cache)
- used by: [`VanillaAttn`](kernels/vanilla_attention.md#VanillaAttn), [`ScaledDotProductAttn`](kernels/scaled_dot_product_attention.md#ScaledDotProductAttn), [`_get_operation_requirements`](../infra/mixins/operation_cache.md#OperationCacheMixin._get_operation_requirements), [`impl_decode`](../layers/attention/_flexible.md#FlexibleAttentionModule.impl_decode), [`RaggedGatedDeltaRule`](kernels/inference_gdn.md#RaggedGatedDeltaRule), [`impl`](../layers/attention/_flexible.md#FlexibleAttentionModule.impl), [`KernelDeltaAttnOp`](kernels/kda.md#KernelDeltaAttnOp), [`RingAttn`](kernels/ring_attention.md#RingAttn), [`AutoRegressiveDecodeAttn`](kernels/decode_attention.md#AutoRegressiveDecodeAttn), [`PagedFlashAttn`](kernels/paged_flash_attention.md#PagedFlashAttn), [`RaggedPageAttnV2`](kernels/ragged_page_attention.md#RaggedPageAttnV2), [`RaggedPageAttnV3`](kernels/ragged_page_attention.md#RaggedPageAttnV3), [`UnifiedAttn`](kernels/unified_attention.md#UnifiedAttn), [`ssm_state`](kernels/ssm1.md#SSM1Output.ssm_state), [`ssm_state`](kernels/ssm2.md#SSM2Output.ssm_state), [`cached_keys`](kernels/glm_moe_dsa_indexer.md#GlmMoeDsaIndexerOutput.cached_keys), [`MultiLatentRaggedPageAttn`](kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttn), [`RaggedCausalConv1D`](kernels/inference_conv1d.md#RaggedCausalConv1D), [`recurrent_state`](kernels/gated_delta_rule.md#GatedDeltaRuleOutput.recurrent_state), [`MultiLatentRaggedPageAttnV2`](kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttnV2), [`logger`](kernels/blocksparse_attention.md#logger), [`logger`](kernels/flash_attention.md#logger), [`__all__`](_operation_impl.md#__all__)

## Module values
- `_I` — [`L297`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L297)
- `__all__` — [`L37`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L37)
- `logger` — [`L39`](../../../../../../raw/code/EasyDeL/easydel/operations/_base_operation.py#L39)

