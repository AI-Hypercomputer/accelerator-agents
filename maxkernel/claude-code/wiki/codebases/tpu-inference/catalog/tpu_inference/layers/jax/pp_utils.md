---
title: 'Module: tpu_inference/layers/jax/pp_utils.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/pp_utils.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.pp_utils`/
symbols:
  PPMissingLayer: PPMissingLayer#
  make_layers: make_layers().
  get_start_end_layer: get_start_end_layer().
  LayerFn: LayerFn#
  PPMissingLayer.__init__: PPMissingLayer#__init__().
  PPMissingLayer.__call__: PPMissingLayer#__call__().
  PPMissingLayer.load_weights: PPMissingLayer#load_weights().
  LayerFn.__call__: LayerFn#__call__().
---
# Module: [`tpu_inference/layers/jax/pp_utils.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/pp_utils.py)

## Classes
### `LayerFn`  ·  implements/extends Protocol
- def: [`tpu_inference/layers/jax/pp_utils.py:42`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/pp_utils.py#L42)
- signature: `class LayerFn(Protocol):`
- protocol/private: `__call__`[`L44`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/pp_utils.py#L44)
- used by: [`make_layers`](pp_utils.md#make_layers)

### `PPMissingLayer`  ·  implements/extends JaxModule
- def: [`tpu_inference/layers/jax/pp_utils.py:24`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/pp_utils.py#L24)
- doc: A placeholder layer for missing layers in a pipeline parallel model.
- signature: `class PPMissingLayer(JaxModule):`
- members:
  - `__call__(self, *args, **kwargs)` — [`L32`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/pp_utils.py#L32) — Return the first arg from args or the first value from kwargs.
  - `load_weights(self, weights: Iterable, *args, **kwargs)` — [`L36`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/pp_utils.py#L36) — No-op for loading weights.
- protocol/private: `__init__`[`L29`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/pp_utils.py#L29)
- uses (calls/refs, reference-scoped): [`JaxModule`](../__init__.md#JaxModule)
- used by: [`end_layer`](../../models/jax/llama4.md#Llama4ForCausalLM.end_layer), [`JaxModule`](../__init__.md#JaxModule), [`end_layer`](../../models/jax/llama_guard_4.md#LlamaGuard4ForCausalLM.end_layer), [`load_weights`](../../models/jax/llama4.md#Llama4WeightLoader.load_weights), [`pp_missing_layers`](../../models/jax/llama_guard_4.md#LlamaGuard4WeightLoader.pp_missing_layers), [`lm_head`](../../models/jax/llama4.md#Llama4ForCausalLM.lm_head), [`lm_head`](../../models/jax/llama_guard_4.md#LlamaGuard4ForCausalLM.lm_head), [`embedder`](../../models/jax/llama_guard_4.md#LlamaGuard4ForCausalLM.embedder), [`embedder`](../../models/jax/llama4.md#Llama4ForCausalLM.embedder), [`make_layers`](pp_utils.md#make_layers), [`embed_tokens`](../../models/jax/gemma4.md#Gemma4Model.embed_tokens), [`embed_tokens`](../../models/jax/deepseek_v3.md#DeepSeekV3.embed_tokens), [`visual`](../../models/jax/qwen2_5_vl.md#Qwen2_5_VLForConditionalGeneration.visual), [`lm_head`](../../models/jax/llama3.md#LlamaModel.lm_head), [`__call__`](../../models/jax/qwen3_moe.md#Qwen3MoeModel.__call__), [`lm_head`](../../models/jax/deepseek_v3.md#DeepseekV3ForCausalLM.lm_head), [`pp_missing_layers`](../../models/jax/qwen2_5_vl.md#Qwen2_5_VLForConditionalGeneration.pp_missing_layers), [`compute_logits`](../../models/jax/qwen3.md#Qwen3ForCausalLM.compute_logits), [`embed_tokens`](../../models/jax/qwen2.md#Qwen2Model.embed_tokens), [`norm`](../../models/jax/deepseek_v3.md#DeepSeekV3.norm), [`embed_tokens`](../../models/jax/qwen3_moe.md#Qwen3MoeModel.embed_tokens), [`embed`](../../models/jax/llama3.md#LlamaModel.embed), [`embed_tokens`](../../models/jax/qwen3.md#Qwen3Model.embed_tokens), [`lm_head`](../../models/jax/gemma4_mm.md#Gemma4ForConditionalGeneration.lm_head), [`lm_head`](../../models/jax/qwen2_5_vl.md#Qwen2_5_VLForConditionalGeneration.lm_head), [`norm`](../../models/jax/gemma4.md#Gemma4Model.norm), [`norm`](../../models/jax/llama3.md#LlamaModel.norm), [`norm`](../../models/jax/qwen2.md#Qwen2Model.norm), [`norm`](../../models/jax/qwen3_moe.md#Qwen3MoeModel.norm), [`lm_head`](../../models/jax/qwen3.md#Qwen3ForCausalLM.lm_head), [`norm`](../../models/jax/qwen3.md#Qwen3Model.norm), [`lm_head`](../../models/jax/gemma4.md#Gemma4ForCausalLM.lm_head), [`lm_head`](../../models/jax/qwen3_moe.md#Qwen3MoeForCausalLM.lm_head), [`pp_missing_layers`](../../models/jax/llama3.md#LlamaForCausalLM.pp_missing_layers)

## Functions
- `get_start_end_layer(num_hidden_layers: int, rank: int, world_size: int)` — [`L48`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/pp_utils.py#L48)
- `make_layers(num_hidden_layers: int, layer_fn: LayerFn)` — [`L53`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/pp_utils.py#L53)

