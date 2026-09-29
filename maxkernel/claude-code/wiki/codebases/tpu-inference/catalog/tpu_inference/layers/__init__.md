---
title: 'Module: tpu_inference/layers/__init__.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/__init__.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax`/JaxModule
symbols:
  JaxModule: '#'
  JaxModuleList: List#
  JaxModule.named_children: '#named_children().'
  JaxModule.named_parameters: '#named_parameters().'
  JaxModule.named_modules: '#named_modules().'
  JaxModule.children: '#children().'
  JaxModuleList.children: List#children().
  JaxModuleList.named_children: List#named_children().
  JaxModuleList.named_modules: List#named_modules().
  JaxModuleList.__init__: List#__init__().
  JaxModule._get_name: '#_get_name().'
  JaxModuleList._get_name: List#_get_name().
  JaxModuleList.named_parameters: List#named_parameters().
---
# Module: [`tpu_inference/layers/__init__.py`](../../../../../../raw/code/tpu-inference/tpu_inference/layers/__init__.py)

## Classes
### `JaxModule`  ·  implements/extends Module
- def: [`tpu_inference/layers/__init__.py:20`](../../../../../../raw/code/tpu-inference/tpu_inference/layers/__init__.py#L20)
- doc: Base module for JAX layers, extending flax.nnx.Module.
- signature: `class JaxModule(nnx.Module):`
- members:
  - `children(self)` — [`L73`](../../../../../../raw/code/tpu-inference/tpu_inference/layers/__init__.py#L73) — Yields immediate child modules.
  - `named_children(self)` — [`L52`](../../../../../../raw/code/tpu-inference/tpu_inference/layers/__init__.py#L52) — Returns an iterator over immediate children modules.
  - `named_modules(self, memo: set | None = None, prefix: str = "", remove_duplicate: bool = True)` — [`L82`](../../../../../../raw/code/tpu-inference/tpu_inference/layers/__init__.py#L82) — Yields (name, module) for self and every descendant module.
  - `named_parameters(self, prefix: str = "", recurse=True)` — [`L27`](../../../../../../raw/code/tpu-inference/tpu_inference/layers/__init__.py#L27) — Yields the named parameters of the module.
- protocol/private: `_get_name`[`L24`](../../../../../../raw/code/tpu-inference/tpu_inference/layers/__init__.py#L24)
- uses (calls/refs, reference-scoped): [`JaxEinsum`](jax/linear.md#JaxEinsum), [`JaxRmsNorm`](jax/norm.md#JaxRmsNorm), [`PPMissingLayer`](jax/pp_utils.md#PPMissingLayer), [`JaxMoE`](jax/moe/moe.md#JaxMoE), [`JaxModuleList`](__init__.md#JaxModuleList), [`JaxLmHead`](jax/linear.md#JaxLmHead), [`JaxEmbed`](jax/embed.md#JaxEmbed), [`JaxRoutedExperts`](jax/moe/moe.md#JaxRoutedExperts), [`Qwen2Model`](../models/jax/qwen2.md#Qwen2Model), [`DeepseekV3BaseAttention`](../models/jax/deepseek_v3.md#DeepseekV3BaseAttention), [`Gemma4ForCausalLM`](../models/jax/gemma4.md#Gemma4ForCausalLM), [`Qwen2DecoderLayer`](../models/jax/qwen2.md#Qwen2DecoderLayer), [`DeepseekV3ForCausalLM`](../models/jax/deepseek_v3.md#DeepseekV3ForCausalLM), [`Gemma4ForConditionalGeneration`](../models/jax/gemma4_mm.md#Gemma4ForConditionalGeneration), [`Gemma4MTPForCausalLM`](../models/jax/gemma4_mtp.md#Gemma4MTPForCausalLM), [`Qwen2ForCausalLM`](../models/jax/qwen2.md#Qwen2ForCausalLM), [`Qwen3ForCausalLM`](../models/jax/qwen3.md#Qwen3ForCausalLM), [`Qwen3MoeForCausalLM`](../models/jax/qwen3_moe.md#Qwen3MoeForCausalLM), [`DeepseekV3MLP`](../models/jax/deepseek_v3.md#DeepseekV3MLP), [`Gemma4MLP`](../models/jax/gemma4.md#Gemma4MLP), [`Gemma4Model`](../models/jax/gemma4.md#Gemma4Model), [`Qwen2MLP`](../models/jax/qwen2.md#Qwen2MLP), [`Qwen3Attention`](../models/jax/qwen3.md#Qwen3Attention), [`DeepSeekV3`](../models/jax/deepseek_v3.md#DeepSeekV3), [`DeepseekV2Moe`](../models/jax/deepseek_v3.md#DeepseekV2Moe), [`DeepseekV3DecoderLayer`](../models/jax/deepseek_v3.md#DeepseekV3DecoderLayer), [`Gemma4Attention`](../models/jax/gemma4.md#Gemma4Attention), [`Gemma4DecoderLayer`](../models/jax/gemma4.md#Gemma4DecoderLayer), [`Gemma4MTPAttention`](../models/jax/gemma4_mtp.md#Gemma4MTPAttention), [`Gemma4MTPDecoderLayer`](../models/jax/gemma4_mtp.md#Gemma4MTPDecoderLayer), [`Gemma4MTPMaskedEmbedder`](../models/jax/gemma4_mtp.md#Gemma4MTPMaskedEmbedder), [`Gemma4MultiTokenPredictor`](../models/jax/gemma4_mtp.md#Gemma4MultiTokenPredictor), [`Gemma4MultimodalEmbedder`](../models/jax/gemma4_mm.md#Gemma4MultimodalEmbedder), [`Gemma4Router`](../models/jax/gemma4.md#Gemma4Router), [`Gemma4VisionEncoderLayer`](../models/jax/gemma4_mm.md#Gemma4VisionEncoderLayer), [`Gemma4VisionFlashAttention`](../models/jax/gemma4_mm.md#Gemma4VisionFlashAttention), [`Gemma4VisionMLP`](../models/jax/gemma4_mm.md#Gemma4VisionMLP), [`Gemma4VisionModel`](../models/jax/gemma4_mm.md#Gemma4VisionModel), [`Gemma4VisionPatchEmbedder`](../models/jax/gemma4_mm.md#Gemma4VisionPatchEmbedder), [`Gemma4VisionPooler`](../models/jax/gemma4_mm.md#Gemma4VisionPooler)  (+7 more)
- used by: [`moe_apply`](common/moe.md#moe_apply), [`JaxEinsum`](jax/linear.md#JaxEinsum), [`JaxRmsNorm`](jax/norm.md#JaxRmsNorm), [`apply_jax`](jax/quantization/fp8.md#Fp8FusedMoEMethod.apply_jax), [`PPMissingLayer`](jax/pp_utils.md#PPMissingLayer), [`create_weights_jax`](jax/quantization/fp8.md#Fp8BlockwiseLinearMethod.create_weights_jax), [`get_quant_method`](jax/quantization/unquantized.md#UnquantizedConfig.get_quant_method), [`get_quant_method`](jax/quantization/fp8.md#Fp8Config.get_quant_method), [`apply_jax`](jax/quantization/fp8.md#Fp8BlockwiseLinearMethod.apply_jax), [`JaxMoE`](jax/moe/moe.md#JaxMoE), [`load_weights`](../models/jax/deepseek_v3.md#MLAEinsum.load_weights), [`apply_jax`](jax/quantization/fp8.md#Fp8TensorwiseLinearMethod.apply_jax), [`apply_jax`](jax/quantization/unquantized.md#UnquantizedLinearMethod.apply_jax), [`_get_nnx_model`](../models/common/model_loader.md#_get_nnx_model), [`_load_module`](../models/jax/utils/weight_utils.md#JaxAutoWeightsLoader._load_module), [`apply_jax`](jax/quantization/__init__.md#QuantizeMethodBase.apply_jax), [`load_weights`](../models/jax/deepseek_v3.md#DeepseekV3ForCausalLM.load_weights), [`JaxLmHead`](jax/linear.md#JaxLmHead), [`_process_weights_after_loading`](../models/jax/utils/weight_utils.md#JaxDummyModelLoader._process_weights_after_loading), [`JaxEmbed`](jax/embed.md#JaxEmbed), [`create_weights_jax`](jax/quantization/__init__.md#QuantizeMethodBase.create_weights_jax), [`JaxRoutedExperts`](jax/moe/moe.md#JaxRoutedExperts), [`get_quant_method`](jax/quantization/configs.md#QuantizationConfig.get_quant_method), [`process_weights_after_loading`](jax/quantization/__init__.md#QuantizeMethodBase.process_weights_after_loading), [`load_weights`](../models/jax/utils/weight_utils.md#JaxDummyModelLoader.load_weights), [`pooler_weights`](../models/jax/utils/weight_utils.md#JaxAutoWeightsLoader.pooler_weights), [`Qwen2Model`](../models/jax/qwen2.md#Qwen2Model), [`DeepseekV3BaseAttention`](../models/jax/deepseek_v3.md#DeepseekV3BaseAttention), [`Gemma4ForCausalLM`](../models/jax/gemma4.md#Gemma4ForCausalLM), [`_map_weights`](../models/jax/utils/weight_utils.md#JaxAutoWeightsLoader._map_weights), [`Qwen2DecoderLayer`](../models/jax/qwen2.md#Qwen2DecoderLayer), [`children`](__init__.md#JaxModuleList.children), [`DeepseekV3ForCausalLM`](../models/jax/deepseek_v3.md#DeepseekV3ForCausalLM), [`Gemma4ForConditionalGeneration`](../models/jax/gemma4_mm.md#Gemma4ForConditionalGeneration), [`Gemma4MTPForCausalLM`](../models/jax/gemma4_mtp.md#Gemma4MTPForCausalLM), [`Qwen2ForCausalLM`](../models/jax/qwen2.md#Qwen2ForCausalLM), [`Qwen3ForCausalLM`](../models/jax/qwen3.md#Qwen3ForCausalLM), [`Qwen3MoeForCausalLM`](../models/jax/qwen3_moe.md#Qwen3MoeForCausalLM), [`named_children`](__init__.md#JaxModuleList.named_children), [`DeepseekV3MLP`](../models/jax/deepseek_v3.md#DeepseekV3MLP)  (+32 more)

### `JaxModuleList`
- def: [`tpu_inference/layers/__init__.py:111`](../../../../../../raw/code/tpu-inference/tpu_inference/layers/__init__.py#L111)
- members:
  - `__init__(self, modules: Iterable[JaxModule])` — [`L114`](../../../../../../raw/code/tpu-inference/tpu_inference/layers/__init__.py#L114) — Initializes the JaxModuleList.
  - `children(self)` — [`L150`](../../../../../../raw/code/tpu-inference/tpu_inference/layers/__init__.py#L150) — Yields the contained modules (see ``torch.nn.ModuleList.children``).
  - `named_children(self)` — [`L137`](../../../../../../raw/code/tpu-inference/tpu_inference/layers/__init__.py#L137) — Returns an iterator over the modules in the list with their indices as names.
  - `named_modules(self, memo: set | None = None, prefix: str = "", remove_duplicate: bool = True)` — [`L155`](../../../../../../raw/code/tpu-inference/tpu_inference/layers/__init__.py#L155) — Yields (name, module) for self, every module in the list, and their descendants.
  - `named_parameters(self, prefix: str = "", recurse=True)` — [`L127`](../../../../../../raw/code/tpu-inference/tpu_inference/layers/__init__.py#L127) — Yields the named parameters of all modules in the list.
- protocol/private: `_get_name`[`L124`](../../../../../../raw/code/tpu-inference/tpu_inference/layers/__init__.py#L124)
- uses (calls/refs, reference-scoped): [`JaxModule`](__init__.md#JaxModule)
- used by: [`_process_weights_after_loading`](../models/jax/utils/weight_utils.md#JaxDummyModelLoader._process_weights_after_loading), [`named_children`](__init__.md#JaxModule.named_children), [`_process_weights_after_loading_jax`](../models/common/pathways_dummy_loader.md#_process_weights_after_loading_jax), [`named_modules`](__init__.md#JaxModule.named_modules), [`children`](__init__.md#JaxModule.children)

