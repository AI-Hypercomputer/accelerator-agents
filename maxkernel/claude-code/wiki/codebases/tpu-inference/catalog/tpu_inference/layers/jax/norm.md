---
title: 'Module: tpu_inference/layers/jax/norm.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/norm.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.norm`/Jax
symbols:
  JaxRmsNorm: RmsNorm#
  JaxRmsNorm.quant_method: RmsNorm#quant_method.
  JaxRmsNorm.weight: RmsNorm#weight.
  JaxRmsNorm.__call__: RmsNorm#__call__().
  JaxLayerNorm.weight: LayerNorm#weight.
  JaxRmsNorm.__init__: RmsNorm#__init__().
  JaxRmsNorm.__getattr__: RmsNorm#__getattr__().
  JaxLayerNorm: LayerNorm#
  JaxLayerNorm.__getattr__: LayerNorm#__getattr__().
  JaxLayerNorm.__init__: LayerNorm#__init__().
---
# Module: [`tpu_inference/layers/jax/norm.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/norm.py)

## Classes
### `JaxLayerNorm`  ·  implements/extends JaxModule, LayerNorm
- def: [`tpu_inference/layers/jax/norm.py:68`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/norm.py#L68)
- doc: LayerNorm layer that inherits from JaxModule and maps scale to weight for compatibility.
- signature: `class JaxLayerNorm(nnx.LayerNorm, JaxModule):`
- members:
  - `weight` — [`L79`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/norm.py#L79)
- protocol/private: `__getattr__`[`L82`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/norm.py#L82), `__init__`[`L71`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/norm.py#L71)
- uses (calls/refs, reference-scoped): [`JaxModule`](../__init__.md#JaxModule), `scale`
- used by: [`JaxModule`](../__init__.md#JaxModule)

### `JaxRmsNorm`  ·  implements/extends JaxModule, RMSNorm
- def: [`tpu_inference/layers/jax/norm.py:25`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/norm.py#L25)
- doc: RmsNorm layer for JAX.
- signature: `class JaxRmsNorm(nnx.RMSNorm, JaxModule):`
- members:
  - `quant_method` — [`L47`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/norm.py#L47)
  - `weight` — [`L42`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/norm.py#L42)
- protocol/private: `__call__`[`L60`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/norm.py#L60), `__getattr__`[`L55`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/norm.py#L55), `__init__`[`L28`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/norm.py#L28)
- uses (calls/refs, reference-scoped): [`JaxModule`](../__init__.md#JaxModule), [`QuantizeMethodBase`](quantization/__init__.md#QuantizeMethodBase), [`QuantizationConfig`](quantization/configs.md#QuantizationConfig), [`apply_jax`](quantization/__init__.md#QuantizeMethodBase.apply_jax), [`create_weights_jax`](quantization/__init__.md#QuantizeMethodBase.create_weights_jax), [`get_quant_method`](quantization/configs.md#QuantizationConfig.get_quant_method), `scale`
- used by: [`JaxModule`](../__init__.md#JaxModule), [`get_decoder_layer`](../../models/jax/deepseek_v3.md#DeepSeekV3.get_decoder_layer), [`kv_a_layernorm`](../../models/jax/deepseek_v3.md#DeepseekV3BaseAttention.kv_a_layernorm), [`q_a_layernorm`](../../models/jax/deepseek_v3.md#DeepseekV3BaseAttention.q_a_layernorm), [`norm`](../../models/jax/deepseek_v3.md#DeepSeekV3.norm), [`post_per_layer_input_norm`](../../models/jax/gemma4.md#Gemma4DecoderLayer.post_per_layer_input_norm), [`__init__`](../../models/jax/deepseek_v3.md#DeepseekV3DecoderLayer.__init__), [`norm`](../../models/jax/gemma4.md#Gemma4Model.norm), [`norm`](../../models/jax/qwen2.md#Qwen2Model.norm), [`norm`](../../models/jax/qwen3_moe.md#Qwen3MoeModel.norm), [`k_norm`](../../models/jax/gemma4.md#Gemma4Attention.k_norm), [`k_norm`](../../models/jax/qwen3.md#Qwen3Attention.k_norm), [`norm`](../../models/jax/qwen3.md#Qwen3Model.norm), [`q_norm`](../../models/jax/gemma4_mtp.md#Gemma4MTPAttention.q_norm), [`q_norm`](../../models/jax/gemma4.md#Gemma4Attention.q_norm), [`q_norm`](../../models/jax/qwen3.md#Qwen3Attention.q_norm), [`pre_feedforward_layernorm`](../../models/jax/gemma4.md#Gemma4DecoderLayer.pre_feedforward_layernorm), [`k_norm`](../../models/jax/gemma4_mm.md#Gemma4VisionFlashAttention.k_norm), [`norm`](../../models/jax/gemma4_mtp.md#Gemma4MultiTokenPredictor.norm), [`post_feedforward_layernorm`](../../models/jax/gemma4.md#Gemma4DecoderLayer.post_feedforward_layernorm), [`q_norm`](../../models/jax/gemma4_mm.md#Gemma4VisionFlashAttention.q_norm), [`v_norm`](../../models/jax/gemma4.md#Gemma4Attention.v_norm), [`per_layer_projection_norm`](../../models/jax/gemma4.md#Gemma4Model.per_layer_projection_norm), [`pre_feedforward_layernorm_2`](../../models/jax/gemma4.md#Gemma4DecoderLayer.pre_feedforward_layernorm_2), [`input_layernorm`](../../models/jax/gemma4_mm.md#Gemma4VisionEncoderLayer.input_layernorm), [`input_layernorm`](../../models/jax/gemma4_mtp.md#Gemma4MTPDecoderLayer.input_layernorm), [`input_layernorm`](../../models/jax/gemma4.md#Gemma4DecoderLayer.input_layernorm), [`input_layernorm`](../../models/jax/qwen2.md#Qwen2DecoderLayer.input_layernorm), [`input_layernorm`](../../models/jax/qwen3_moe.md#Qwen3MoeDecoderLayer.input_layernorm), [`norm`](../../models/jax/gemma4.md#Gemma4Router.norm), [`post_attention_layernorm`](../../models/jax/gemma4_mm.md#Gemma4VisionEncoderLayer.post_attention_layernorm), [`post_attention_layernorm`](../../models/jax/gemma4_mtp.md#Gemma4MTPDecoderLayer.post_attention_layernorm), [`post_attention_layernorm`](../../models/jax/gemma4.md#Gemma4DecoderLayer.post_attention_layernorm), [`post_attention_layernorm`](../../models/jax/qwen2.md#Qwen2DecoderLayer.post_attention_layernorm), [`post_attention_layernorm`](../../models/jax/qwen3_moe.md#Qwen3MoeDecoderLayer.post_attention_layernorm), [`post_feedforward_layernorm`](../../models/jax/gemma4_mm.md#Gemma4VisionEncoderLayer.post_feedforward_layernorm), [`post_feedforward_layernorm`](../../models/jax/gemma4_mtp.md#Gemma4MTPDecoderLayer.post_feedforward_layernorm), [`pre_feedforward_layernorm`](../../models/jax/gemma4_mm.md#Gemma4VisionEncoderLayer.pre_feedforward_layernorm), [`pre_feedforward_layernorm`](../../models/jax/gemma4_mtp.md#Gemma4MTPDecoderLayer.pre_feedforward_layernorm), [`v_norm`](../../models/jax/gemma4_mm.md#Gemma4VisionFlashAttention.v_norm)  (+5 more)

