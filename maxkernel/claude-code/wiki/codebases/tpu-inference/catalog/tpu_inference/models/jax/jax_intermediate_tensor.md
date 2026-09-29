---
title: 'Module: tpu_inference/models/jax/jax_intermediate_tensor.py'
type: catalog
provenance: extracted
module: tpu_inference/models/jax/jax_intermediate_tensor.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.jax.jax_intermediate_tensor`/JaxIntermediateTensors#
symbols:
  JaxIntermediateTensors: ''
  JaxIntermediateTensors.tensors: tensors.
  JaxIntermediateTensors.tree_flatten: tree_flatten().
  JaxIntermediateTensors.to_torch: to_torch().
  JaxIntermediateTensors.kv_connector_output: kv_connector_output.
  JaxIntermediateTensors.__getitem__: __getitem__().
  JaxIntermediateTensors.__setitem__: __setitem__().
  JaxIntermediateTensors.keys: keys().
  JaxIntermediateTensors.items: items().
  JaxIntermediateTensors.__len__: __len__().
  JaxIntermediateTensors.block_until_ready: block_until_ready().
  JaxIntermediateTensors.from_torch: from_torch().
  JaxIntermediateTensors.tree_unflatten: tree_unflatten().
---
# Module: [`tpu_inference/models/jax/jax_intermediate_tensor.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/jax_intermediate_tensor.py)

## Classes
### `JaxIntermediateTensors`
- def: [`tpu_inference/models/jax/jax_intermediate_tensor.py:32`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/jax_intermediate_tensor.py#L32) — documented in [root](../../../../concepts/root.md)
- doc: For all pipeline stages except the last, we need to return the
- signature: `class JaxIntermediateTensors:`
- members:
  - `block_until_ready(self)` — [`L88`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/jax_intermediate_tensor.py#L88)
  - `from_torch(cls, torch_obj: IntermediateTensors)` — [`L61`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/jax_intermediate_tensor.py#L61)
  - `items(self)` — [`L82`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/jax_intermediate_tensor.py#L82)
  - `keys(self)` — [`L79`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/jax_intermediate_tensor.py#L79)
  - `to_torch(self)` — [`L66`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/jax_intermediate_tensor.py#L66)
  - `tree_flatten(self)` — [`L51`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/jax_intermediate_tensor.py#L51)
  - `tree_unflatten(cls, aux_data, children)` — [`L57`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/jax_intermediate_tensor.py#L57)
  - `kv_connector_output` — [`L49`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/jax_intermediate_tensor.py#L49)
  - `tensors` — [`L48`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/jax_intermediate_tensor.py#L48)
- protocol/private: `__getitem__`[`L70`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/jax_intermediate_tensor.py#L70), `__len__`[`L85`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/jax_intermediate_tensor.py#L85), `__setitem__`[`L76`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/jax_intermediate_tensor.py#L76)
- used by: `_execute_model`, [`_precompile_backbone_with_inputs_embeds`](../../runner/compilation_manager.md#CompilationManager._precompile_backbone_with_inputs_embeds), [`_precompile_backbone_text_only`](../../runner/compilation_manager.md#CompilationManager._precompile_backbone_text_only), [`step_fun_impl`](../vllm/vllm_model_wrapper.md#VllmModelWrapper.step_fun_impl), [`__call__`](llama4.md#Llama4ForCausalLM.__call__), [`__call__`](llama_guard_4.md#LlamaGuard4ForCausalLM.__call__), [`__call__`](llama3.md#LlamaModel.__call__), `execute_model`, [`__call__`](gemma4_mm.md#Gemma4ForConditionalGeneration.__call__), [`__call__`](deepseek_v3.md#DeepseekV3ForCausalLM.__call__), [`__call__`](gemma4.md#Gemma4ForCausalLM.__call__), [`__call__`](qwen2_5_vl.md#Qwen2_5_VLForConditionalGeneration.__call__), [`__call__`](qwen2.md#Qwen2ForCausalLM.__call__), [`__call__`](qwen3_moe.md#Qwen3MoeForCausalLM.__call__), [`__call__`](qwen3.md#Qwen3ForCausalLM.__call__), [`__call__`](llama3.md#LlamaForCausalLM.__call__)

