---
title: 'Module: tpu_inference/models/vllm/vllm_model_wrapper_context.py'
type: catalog
provenance: extracted
module: tpu_inference/models/vllm/vllm_model_wrapper_context.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.vllm.vllm_model_wrapper_context`/
symbols:
  set_vllm_model_wrapper_context: set_vllm_model_wrapper_context().
  get_vllm_model_wrapper_context: get_vllm_model_wrapper_context().
  VllmModelWrapperContext.kv_caches: VllmModelWrapperContext#kv_caches.
  _vllm_model_wrapper_context._vllm_model_wrapper_context: _vllm_model_wrapper_context._vllm_model_wrapper_context.
  VllmModelWrapperContext.layer_name_to_kvcache_index: VllmModelWrapperContext#layer_name_to_kvcache_index.
  VllmModelWrapperContext.mesh: VllmModelWrapperContext#mesh.
  VllmModelWrapperContext: VllmModelWrapperContext#
  VllmModelWrapperContext.vllm_config: VllmModelWrapperContext#vllm_config.
  VllmModelWrapperContext.expert_indices_list: VllmModelWrapperContext#expert_indices_list.
---
# Module: [`tpu_inference/models/vllm/vllm_model_wrapper_context.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper_context.py)

## Classes
### `VllmModelWrapperContext`
- def: [`tpu_inference/models/vllm/vllm_model_wrapper_context.py:25`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper_context.py#L25)
- signature: `class VllmModelWrapperContext:`
- members:
  - `expert_indices_list` — [`L30`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper_context.py#L30)
  - `kv_caches` — [`L26`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper_context.py#L26)
  - `layer_name_to_kvcache_index` — [`L28`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper_context.py#L28)
  - `mesh` — [`L27`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper_context.py#L27)
  - `vllm_config` — [`L29`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper_context.py#L29)
- used by: [`vllm_moe_apply`](../../layers/vllm/interface/moe.md#vllm_moe_apply), [`step_fun_impl`](vllm_model_wrapper.md#VllmModelWrapper.step_fun_impl), [`draft_step_fun_impl`](vllm_model_wrapper.md#VllmModelWrapper.draft_step_fun_impl), [`forward`](../../layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_compressor.md#VllmDeepseekCompressor.forward), [`set_vllm_model_wrapper_context`](vllm_model_wrapper_context.md#set_vllm_model_wrapper_context), [`get_vllm_model_wrapper_context`](vllm_model_wrapper_context.md#get_vllm_model_wrapper_context), [`forward`](../../layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_indexer.md#VllmDeepseekV4Indexer.forward), [`forward_mqa`](../../layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.md#VllmDeepseekV4MLAAttention.forward_mqa), [`_vllm_model_wrapper_context`](vllm_model_wrapper_context.md#_vllm_model_wrapper_context._vllm_model_wrapper_context)

## Functions
- `get_vllm_model_wrapper_context()` — [`L36`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper_context.py#L36)
- `set_vllm_model_wrapper_context(*, kv_caches: List[jax.Array], mesh: Mesh, layer_name_to_kvcache_index: Dict[str, int] = None, vllm_config: Optional[VllmConfig] = None)` — [`L45`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper_context.py#L45)

## Module values
- `_vllm_model_wrapper_context` — [`L33`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper_context.py#L33)

