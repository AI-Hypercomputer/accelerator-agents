---
title: 'Module: tpu_inference/layers/jax/embed.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/embed.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.embed`/JaxEmbed#
symbols:
  JaxEmbed: ''
  JaxEmbed.quant_method: quant_method.
  JaxEmbed.weight: weight.
  JaxEmbed.decode: decode().
  JaxEmbed.__call__: __call__().
  JaxEmbed.__init__: __init__().
  JaxEmbed.__getattr__: __getattr__().
---
# Module: [`tpu_inference/layers/jax/embed.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/embed.py)

## Classes
### `JaxEmbed`  ·  implements/extends Embed, JaxModule
- def: [`tpu_inference/layers/jax/embed.py:25`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/embed.py#L25)
- doc: Embedding layer for JAX.
- signature: `class JaxEmbed(nnx.Embed, JaxModule):`
- members:
  - `decode(self, x: jax.Array)` — [`L64`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/embed.py#L64)
  - `quant_method` — [`L46`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/embed.py#L46)
  - `weight` — [`L41`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/embed.py#L41)
- protocol/private: `__call__`[`L59`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/embed.py#L59), `__getattr__`[`L54`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/embed.py#L54), `__init__`[`L28`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/embed.py#L28)
- uses (calls/refs, reference-scoped): [`JaxModule`](../__init__.md#JaxModule), [`QuantizeMethodBase`](quantization/__init__.md#QuantizeMethodBase), [`QuantizationConfig`](quantization/configs.md#QuantizationConfig), [`apply_jax`](quantization/__init__.md#QuantizeMethodBase.apply_jax), [`create_weights_jax`](quantization/__init__.md#QuantizeMethodBase.create_weights_jax), [`get_quant_method`](quantization/configs.md#QuantizationConfig.get_quant_method), `embedding`
- used by: [`JaxModule`](../__init__.md#JaxModule), [`embed_tokens`](../../models/jax/gemma4.md#Gemma4Model.embed_tokens), [`embed_tokens`](../../models/jax/deepseek_v3.md#DeepSeekV3.embed_tokens), [`embed_tokens_per_layer`](../../models/jax/gemma4.md#Gemma4Model.embed_tokens_per_layer), [`compute_logits`](../../models/jax/gemma4_mtp.md#Gemma4MTPForCausalLM.compute_logits), [`compute_logits`](../../models/jax/qwen3.md#Qwen3ForCausalLM.compute_logits), [`embed_tokens`](../../models/jax/qwen2.md#Qwen2Model.embed_tokens), [`embed_tokens`](../../models/jax/qwen3_moe.md#Qwen3MoeModel.embed_tokens), [`embed_tokens`](../../models/jax/gemma4_mtp.md#Gemma4MultiTokenPredictor.embed_tokens), [`embed_tokens`](../../models/jax/qwen3.md#Qwen3Model.embed_tokens), [`compute_logits`](../../models/jax/gemma4_mm.md#Gemma4ForConditionalGeneration.compute_logits), [`compute_logits`](../../models/jax/gemma4.md#Gemma4ForCausalLM.compute_logits), [`compute_logits`](../../models/jax/qwen3_moe.md#Qwen3MoeForCausalLM.compute_logits), [`compute_logits`](../../models/jax/qwen2_5_vl.md#Qwen2_5_VLForConditionalGeneration.compute_logits), [`compute_logits`](../../models/jax/qwen2.md#Qwen2ForCausalLM.compute_logits)

