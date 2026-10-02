---
title: 'Module: tpu_inference/layers/jax/linear.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/linear.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.linear`/
symbols:
  JaxEinsum: JaxEinsum#
  JaxEinsum.weight: JaxEinsum#weight.
  JaxLinear: JaxLinear#
  JaxLmHead: JaxLmHead#
  JaxEinsum.quant_method: JaxEinsum#quant_method.
  JaxQKVParallelLinear.__call__: JaxQKVParallelLinear#__call__().
  JaxEinsum.__call__: JaxEinsum#__call__().
  JaxQKVParallelLinear.v_size: JaxQKVParallelLinear#v_size.
  JaxLmHead.weight: JaxLmHead#weight.
  JaxMergedColumnParallelLinear: JaxMergedColumnParallelLinear#
  JaxLinear.__init__: JaxLinear#__init__().
  JaxEinsum.prefix: JaxEinsum#prefix.
  JaxMergedColumnParallelLinear.output_sizes: JaxMergedColumnParallelLinear#output_sizes.
  JaxEinsum.__init__: JaxEinsum#__init__().
  JaxLmHead.__call__: JaxLmHead#__call__().
  JaxQKVParallelLinear: JaxQKVParallelLinear#
  JaxMergedColumnParallelLinear.__init__: JaxMergedColumnParallelLinear#__init__().
  JaxQKVParallelLinear.head_dim: JaxQKVParallelLinear#head_dim.
  logger: logger.
  JaxQKVParallelLinear.__init__: JaxQKVParallelLinear#__init__().
  JaxQKVParallelLinear.num_kv_heads: JaxQKVParallelLinear#num_kv_heads.
  JaxQKVParallelLinear.q_size: JaxQKVParallelLinear#q_size.
  JaxQKVParallelLinear.k_size: JaxQKVParallelLinear#k_size.
  JaxEinsum.kernel_init: JaxEinsum#kernel_init.
  JaxQKVParallelLinear.num_heads: JaxQKVParallelLinear#num_heads.
  JaxLmHead.__init__: JaxLmHead#__init__().
  JaxLmHead.prefix: JaxLmHead#prefix.
  JaxLmHead.quant_method: JaxLmHead#quant_method.
---
# Module: [`tpu_inference/layers/jax/linear.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py)

## Classes
### `JaxEinsum`  ·  implements/extends Einsum, JaxModule
- def: [`tpu_inference/layers/jax/linear.py:29`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L29)
- doc: Einsum layer for JAX.
- signature: `class JaxEinsum(nnx.Einsum, JaxModule):`
- members:
  - `kernel_init` — [`L67`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L67)
  - `prefix` — [`L75`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L75)
  - `quant_method` — [`L78`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L78)
  - `weight` — [`L71`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L71)
- protocol/private: `__call__`[`L87`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L87), `__init__`[`L40`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L40)
- uses (calls/refs, reference-scoped): [`JaxModule`](../__init__.md#JaxModule), [`QuantizeMethodBase`](quantization/__init__.md#QuantizeMethodBase), [`QuantizationConfig`](quantization/configs.md#QuantizationConfig), [`apply_jax`](quantization/__init__.md#QuantizeMethodBase.apply_jax), [`JaxLinear`](linear.md#JaxLinear), [`create_weights_jax`](quantization/__init__.md#QuantizeMethodBase.create_weights_jax), [`get_quant_method`](quantization/configs.md#QuantizationConfig.get_quant_method), `bias`, `einsum_str`, [`MLAEinsum`](../../models/jax/deepseek_v3.md#MLAEinsum), `kernel`
- used by: [`JaxModule`](../__init__.md#JaxModule), [`process_weights_after_loading`](quantization/fp8.md#Fp8BlockwiseLinearMethod.process_weights_after_loading), [`create_weights_jax`](quantization/fp8.md#Fp8BlockwiseLinearMethod.create_weights_jax), [`get_quant_method`](quantization/unquantized.md#UnquantizedConfig.get_quant_method), [`get_quant_method`](quantization/fp8.md#Fp8Config.get_quant_method), [`create_weights_jax`](quantization/fp8.md#Fp8TensorwiseLinearMethod.create_weights_jax), [`kv_b_proj`](../../models/jax/deepseek_v3.md#DeepseekV3BaseAttention.kv_b_proj), [`load_weights`](../../models/jax/deepseek_v3.md#MLAEinsum.load_weights), [`kv_b_proj`](../../models/jax/deepseek_v3.md#DeepseekV3Attention.kv_b_proj), [`__call__`](../../models/jax/deepseek_v3.md#DeepSeekV3Router.__call__), [`apply_jax`](quantization/unquantized.md#UnquantizedLinearMethod.apply_jax), [`kv_a_proj_with_mqa`](../../models/jax/deepseek_v3.md#DeepseekV3BaseAttention.kv_a_proj_with_mqa), [`q_b_proj`](../../models/jax/deepseek_v3.md#DeepseekV3BaseAttention.q_b_proj), [`moe_backend`](../../models/jax/deepseek_v3.md#DeepSeekV3Router.moe_backend), [`o_proj`](../../models/jax/deepseek_v3.md#DeepseekV3BaseAttention.o_proj), [`create_weights_jax`](quantization/unquantized.md#UnquantizedMergedLinearMethod.create_weights_jax), [`q_a_proj`](../../models/jax/deepseek_v3.md#DeepseekV3BaseAttention.q_a_proj), [`JaxLinear`](linear.md#JaxLinear), [`k_proj`](../../models/jax/gemma4_mm.md#Gemma4VisionFlashAttention.k_proj), [`o_proj`](../../models/jax/gemma4_mm.md#Gemma4VisionFlashAttention.o_proj), [`q_proj`](../../models/jax/gemma4_mm.md#Gemma4VisionFlashAttention.q_proj), [`v_proj`](../../models/jax/gemma4_mm.md#Gemma4VisionFlashAttention.v_proj), [`__call__`](linear.md#JaxQKVParallelLinear.__call__), [`k_proj`](../../models/jax/gemma4.md#Gemma4Attention.k_proj), [`q_proj`](../../models/jax/gemma4.md#Gemma4Attention.q_proj), [`down_proj`](../../models/jax/gemma4_mm.md#Gemma4VisionMLP.down_proj), [`gate_proj`](../../models/jax/gemma4_mm.md#Gemma4VisionMLP.gate_proj), [`k_proj`](../../models/jax/qwen2.md#Qwen2Attention.k_proj), [`k_proj`](../../models/jax/qwen3.md#Qwen3Attention.k_proj), [`o_proj`](../../models/jax/gemma4_mtp.md#Gemma4MTPAttention.o_proj), [`o_proj`](../../models/jax/gemma4.md#Gemma4Attention.o_proj), [`o_proj`](../../models/jax/qwen2.md#Qwen2Attention.o_proj), [`o_proj`](../../models/jax/qwen3.md#Qwen3Attention.o_proj), [`q_proj`](../../models/jax/gemma4_mtp.md#Gemma4MTPAttention.q_proj), [`q_proj`](../../models/jax/qwen2.md#Qwen2Attention.q_proj), [`up_proj`](../../models/jax/gemma4_mm.md#Gemma4VisionMLP.up_proj), [`v_proj`](../../models/jax/qwen2.md#Qwen2Attention.v_proj), [`v_proj`](../../models/jax/qwen3.md#Qwen3Attention.v_proj), [`__init__`](quantization/fp8.md#Fp8BlockwiseLinearMethod.__init__), [`down_proj`](../../models/jax/deepseek_v3.md#DeepseekV3MLP.down_proj)  (+14 more)

### `JaxLinear`  ·  implements/extends JaxEinsum
- def: [`tpu_inference/layers/jax/linear.py:97`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L97)
- doc: Linear layer for JAX.
- signature: `class JaxLinear(JaxEinsum):`
- protocol/private: `__init__`[`L109`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L109)
- uses (calls/refs, reference-scoped): [`JaxEinsum`](linear.md#JaxEinsum), [`QuantizationConfig`](quantization/configs.md#QuantizationConfig), [`JaxMergedColumnParallelLinear`](linear.md#JaxMergedColumnParallelLinear), [`__init__`](linear.md#JaxEinsum.__init__)
- used by: [`JaxEinsum`](linear.md#JaxEinsum), [`JaxMergedColumnParallelLinear`](linear.md#JaxMergedColumnParallelLinear), [`post_projection`](../../models/jax/gemma4_mtp.md#Gemma4MultiTokenPredictor.post_projection), [`pre_projection`](../../models/jax/gemma4_mtp.md#Gemma4MultiTokenPredictor.pre_projection), [`gate`](../../models/jax/qwen3_moe.md#Qwen3MoeSparseMoeBlock.gate), [`down_proj`](../../models/jax/gemma4.md#Gemma4MLP.down_proj), [`down_proj`](../../models/jax/qwen2.md#Qwen2MLP.down_proj), [`gate_proj`](../../models/jax/qwen2.md#Qwen2MLP.gate_proj), [`output_sizes`](linear.md#JaxMergedColumnParallelLinear.output_sizes), [`proj`](../../models/jax/gemma4.md#Gemma4Router.proj), [`up_proj`](../../models/jax/qwen2.md#Qwen2MLP.up_proj), [`centroids`](../../models/jax/gemma4_mtp.md#Gemma4MTPMaskedEmbedder.centroids)

### `JaxLmHead`  ·  implements/extends Einsum, JaxModule
- def: [`tpu_inference/layers/jax/linear.py:128`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L128)
- doc: Output projection (vocab head).
- signature: `class JaxLmHead(nnx.Einsum, JaxModule):`
- members:
  - `prefix` — [`L168`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L168)
  - `quant_method` — [`L169`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L169)
  - `weight` — [`L164`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L164)
- protocol/private: `__call__`[`L171`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L171), `__init__`[`L142`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L142)
- uses (calls/refs, reference-scoped): [`JaxModule`](../__init__.md#JaxModule), `einsum_str`, `kernel`
- used by: [`JaxModule`](../__init__.md#JaxModule), [`lm_head`](../../models/jax/deepseek_v3.md#DeepseekV3ForCausalLM.lm_head), [`_get_full_lm_head_weight`](../../models/jax/gemma4_mtp.md#Gemma4MTPForCausalLM._get_full_lm_head_weight), [`lm_head`](../../models/jax/gemma4_mm.md#Gemma4ForConditionalGeneration.lm_head), [`lm_head`](../../models/jax/qwen2_5_vl.md#Qwen2_5_VLForConditionalGeneration.lm_head), [`lm_head`](../../models/jax/qwen3.md#Qwen3ForCausalLM.lm_head), [`final_logit_softcapping`](../../models/jax/gemma4_mm.md#Gemma4ForConditionalGeneration.final_logit_softcapping), [`lm_head`](../../models/jax/gemma4.md#Gemma4ForCausalLM.lm_head), [`lm_head`](../../models/jax/qwen3_moe.md#Qwen3MoeForCausalLM.lm_head), [`lm_head`](../../models/jax/gemma4_mtp.md#Gemma4MTPForCausalLM.lm_head), [`lm_head`](../../models/jax/qwen2.md#Qwen2ForCausalLM.lm_head)

### `JaxMergedColumnParallelLinear`  ·  implements/extends JaxLinear
- def: [`tpu_inference/layers/jax/linear.py:175`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L175)
- doc: Merged version of JaxLinear. This is used to fuse multiple
- signature: `class JaxMergedColumnParallelLinear(JaxLinear):`
- members:
  - `output_sizes` — [`L204`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L204)
- protocol/private: `__init__`[`L192`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L192)
- uses (calls/refs, reference-scoped): [`QuantizationConfig`](quantization/configs.md#QuantizationConfig), [`JaxLinear`](linear.md#JaxLinear), [`UnquantizedConfig`](quantization/unquantized.md#UnquantizedConfig), [`__init__`](linear.md#JaxLinear.__init__), [`JaxQKVParallelLinear`](linear.md#JaxQKVParallelLinear)
- used by: [`get_quant_method`](quantization/unquantized.md#UnquantizedConfig.get_quant_method), [`JaxLinear`](linear.md#JaxLinear), [`v_size`](linear.md#JaxQKVParallelLinear.v_size), [`gate_up_proj`](../../models/jax/gemma4.md#Gemma4MLP.gate_up_proj), [`JaxQKVParallelLinear`](linear.md#JaxQKVParallelLinear)

### `JaxQKVParallelLinear`  ·  implements/extends JaxMergedColumnParallelLinear
- def: [`tpu_inference/layers/jax/linear.py:223`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L223)
- doc: Fused QKV Parallel Linear layer for JAX-native models.
- signature: `class JaxQKVParallelLinear(JaxMergedColumnParallelLinear):`
- members:
  - `head_dim` — [`L243`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L243)
  - `k_size` — [`L246`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L246)
  - `num_heads` — [`L241`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L241)
  - `num_kv_heads` — [`L242`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L242)
  - `q_size` — [`L245`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L245)
  - `v_size` — [`L247`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L247)
- protocol/private: `__call__`[`L262`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L262), `__init__`[`L230`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L230)
- uses (calls/refs, reference-scoped): [`QuantizationConfig`](quantization/configs.md#QuantizationConfig), [`__call__`](linear.md#JaxEinsum.__call__), [`JaxMergedColumnParallelLinear`](linear.md#JaxMergedColumnParallelLinear), [`__init__`](linear.md#JaxMergedColumnParallelLinear.__init__)
- used by: [`v_proj`](../../models/jax/gemma4.md#Gemma4Attention.v_proj), [`JaxMergedColumnParallelLinear`](linear.md#JaxMergedColumnParallelLinear)

## Module values
- `logger` — [`L26`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/linear.py#L26)

