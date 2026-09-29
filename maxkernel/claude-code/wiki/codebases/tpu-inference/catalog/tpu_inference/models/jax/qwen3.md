---
title: 'Module: tpu_inference/models/jax/qwen3.py'
type: catalog
provenance: extracted
module: tpu_inference/models/jax/qwen3.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.jax.qwen3`/
symbols:
  Qwen3Attention.__call__: Qwen3Attention#__call__().
  Qwen3ForCausalLM.compute_logits: Qwen3ForCausalLM#compute_logits().
  Qwen3Attention.head_dim: Qwen3Attention#head_dim.
  Qwen3Attention.mesh: Qwen3Attention#mesh.
  Qwen3Attention.k_proj: Qwen3Attention#k_proj.
  Qwen3Attention.v_proj: Qwen3Attention#v_proj.
  Qwen3Attention.o_proj: Qwen3Attention#o_proj.
  Qwen3Model.is_last_rank: Qwen3Model#is_last_rank.
  Qwen3Model.embed_tokens: Qwen3Model#embed_tokens.
  Qwen3ForCausalLM.model: Qwen3ForCausalLM#model.
  Qwen3ForCausalLM.lm_head: Qwen3ForCausalLM#lm_head.
  Qwen3Attention.q_norm: Qwen3Attention#q_norm.
  Qwen3Attention.k_norm: Qwen3Attention#k_norm.
  Qwen3Model.norm: Qwen3Model#norm.
  init_fn: init_fn.
  Qwen3ForCausalLM.__call__: Qwen3ForCausalLM#__call__().
  Qwen3Attention.head_dim_original: Qwen3Attention#head_dim_original.
  Qwen3Attention.hidden_size: Qwen3Attention#hidden_size.
  Qwen3Attention.num_heads: Qwen3Attention#num_heads.
  Qwen3DecoderLayer.__init__: Qwen3DecoderLayer#__init__().
  Qwen3Model.layers: Qwen3Model#layers.
  Qwen3ForCausalLM: Qwen3ForCausalLM#
  Qwen3Attention.num_kv_heads: Qwen3Attention#num_kv_heads.
  Qwen3Attention.q_proj: Qwen3Attention#q_proj.
  Qwen3Model.is_first_rank: Qwen3Model#is_first_rank.
  Qwen3Attention: Qwen3Attention#
  Qwen3Attention.kv_cache_quantized_dtype: Qwen3Attention#kv_cache_quantized_dtype.
  Qwen3DecoderLayer.input_layernorm: Qwen3DecoderLayer#input_layernorm.
  Qwen3DecoderLayer.post_attention_layernorm: Qwen3DecoderLayer#post_attention_layernorm.
  Qwen3Attention.rope_theta: Qwen3Attention#rope_theta.
  Qwen3Attention.rope_scaling: Qwen3Attention#rope_scaling.
  Qwen3Attention.rms_norm_eps: Qwen3Attention#rms_norm_eps.
  logger: logger.
  Qwen3DecoderLayer: Qwen3DecoderLayer#
  Qwen3Model: Qwen3Model#
  Qwen3Attention.__init__: Qwen3Attention#__init__().
  Qwen3DecoderLayer.self_attn: Qwen3DecoderLayer#self_attn.
  Qwen3DecoderLayer.mlp: Qwen3DecoderLayer#mlp.
  Qwen3Attention._k_scale: Qwen3Attention#_k_scale.
  Qwen3Attention._v_scale: Qwen3Attention#_v_scale.
  Qwen3Attention._q_scale: Qwen3Attention#_q_scale.
  Qwen3Model.__init__: Qwen3Model#__init__().
  Qwen3Model.start_layer: Qwen3Model#start_layer.
  Qwen3Model.end_layer: Qwen3Model#end_layer.
  Qwen3ForCausalLM.__init__: Qwen3ForCausalLM#__init__().
  Qwen3ForCausalLM.vllm_config: Qwen3ForCausalLM#vllm_config.
  Qwen3ForCausalLM.mesh: Qwen3ForCausalLM#mesh.
---
# Module: [`tpu_inference/models/jax/qwen3.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py)

## Classes
### `Qwen3Attention`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/qwen3.py:52`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L52)
- signature: `class Qwen3Attention(JaxModule):`
- members:
  - `head_dim` — [`L72`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L72)
  - `head_dim_original` — [`L70`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L70)
  - `hidden_size` — [`L63`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L63)
  - `k_norm` — [`L125`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L125)
  - `k_proj` — [`L115`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L115)
  - `kv_cache_quantized_dtype` — [`L159`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L159)
  - `mesh` — [`L80`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L80)
  - `num_heads` — [`L64`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L64)
  - `num_kv_heads` — [`L65`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L65)
  - `o_proj` — [`L145`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L145)
  - `q_norm` — [`L105`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L105)
  - `q_proj` — [`L95`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L95)
  - `rms_norm_eps` — [`L68`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L68)
  - `rope_scaling` — [`L67`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L67)
  - `rope_theta` — [`L66`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L66)
  - `v_proj` — [`L135`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L135)
- protocol/private: `__call__`[`L164`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L164), `__init__`[`L54`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L54), `_k_scale`[`L157`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L157), `_q_scale`[`L156`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L156), `_v_scale`[`L158`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L158)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`JaxEinsum`](../../layers/jax/linear.md#JaxEinsum), [`VllmQuantConfig`](../../layers/vllm/quantization/configs.md#VllmQuantConfig), [`JaxRmsNorm`](../../layers/jax/norm.md#JaxRmsNorm), [`input_positions`](../../layers/common/attention_metadata.md#AttentionMetadata.input_positions), [`apply_rope`](../../layers/jax/rope_interface.md#apply_rope), [`attention`](../../layers/common/attention_interface.md#attention), [`quantize_kv`](../../layers/common/__init__.md#quantize_kv), [`get_rope_scaling`](../../layers/jax/rope_interface.md#get_rope_scaling), [`init_fn`](qwen3.md#init_fn), [`get_rope_theta`](../../layers/jax/rope_interface.md#get_rope_theta), [`logger`](qwen3.md#logger), `hidden_size`, `rms_norm_eps`, [`LAYOUT_Q_PROJ_AS_NDH`](../../envs.md#LAYOUT_Q_PROJ_AS_NDH), `num_attention_heads`, `num_key_value_heads`
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`self_attn`](qwen3_moe.md#Qwen3MoeDecoderLayer.self_attn), [`self_attn`](qwen3.md#Qwen3DecoderLayer.self_attn)

### `Qwen3DecoderLayer`  ·  implements/extends Qwen2DecoderLayer
- def: [`tpu_inference/models/jax/qwen3.py:212`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L212)
- signature: `class Qwen3DecoderLayer(Qwen2DecoderLayer):`
- members:
  - `input_layernorm` — [`L225`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L225)
  - `mlp` — [`L252`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L252)
  - `post_attention_layernorm` — [`L242`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L242)
  - `self_attn` — [`L235`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L235)
- protocol/private: `__init__`[`L214`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L214)
- uses (calls/refs, reference-scoped): [`VllmQuantConfig`](../../layers/vllm/quantization/configs.md#VllmQuantConfig), [`JaxRmsNorm`](../../layers/jax/norm.md#JaxRmsNorm), [`init_fn`](qwen3.md#init_fn), [`Qwen2DecoderLayer`](qwen2.md#Qwen2DecoderLayer), [`Qwen2MLP`](qwen2.md#Qwen2MLP), [`Qwen3Attention`](qwen3.md#Qwen3Attention), `hidden_size`, `rms_norm_eps`
- used by: [`Qwen2DecoderLayer`](qwen2.md#Qwen2DecoderLayer), [`layers`](qwen3.md#Qwen3Model.layers)

### `Qwen3ForCausalLM`  ·  implements/extends JaxModule, LoadableWithIterator
- def: [`tpu_inference/models/jax/qwen3.py:323`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L323)
- signature: `class Qwen3ForCausalLM(JaxModule, LoadableWithIterator):`
- members:
  - `compute_logits(self, hidden_states: jax.Array)` — [`L386`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L386)
  - `lm_head` — [`L345`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L345)
  - `mesh` — [`L329`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L329)
  - `model` — [`L331`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L331)
  - `vllm_config` — [`L327`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L327)
- protocol/private: `__call__`[`L358`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L358), `__init__`[`L325`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L325)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`JaxIntermediateTensors`](jax_intermediate_tensor.md#JaxIntermediateTensors), [`PPMissingLayer`](../../layers/jax/pp_utils.md#PPMissingLayer), [`LoadableWithIterator`](utils/weight_utils.md#LoadableWithIterator), [`JaxLmHead`](../../layers/jax/linear.md#JaxLmHead), [`tensors`](jax_intermediate_tensor.md#JaxIntermediateTensors.tensors), [`JaxEmbed`](../../layers/jax/embed.md#JaxEmbed), [`embed_tokens`](qwen3.md#Qwen3Model.embed_tokens), [`is_last_rank`](qwen3.md#Qwen3Model.is_last_rank), [`decode`](../../layers/jax/embed.md#JaxEmbed.decode), [`Qwen3Model`](qwen3.md#Qwen3Model)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`_get_model_architecture`](../common/model_loader.md#_get_model_architecture), [`LoadableWithIterator`](utils/weight_utils.md#LoadableWithIterator)

### `Qwen3Model`  ·  implements/extends Qwen2Model
- def: [`tpu_inference/models/jax/qwen3.py:261`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L261)
- signature: `class Qwen3Model(Qwen2Model):`
- members:
  - `embed_tokens` — [`L283`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L283)
  - `end_layer` — [`L296`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L296)
  - `is_first_rank` — [`L275`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L275)
  - `is_last_rank` — [`L276`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L276)
  - `layers` — [`L296`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L296)
  - `norm` — [`L309`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L309)
  - `start_layer` — [`L296`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L296)
- protocol/private: `__init__`[`L263`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L263)
- uses (calls/refs, reference-scoped): [`JaxRmsNorm`](../../layers/jax/norm.md#JaxRmsNorm), [`PPMissingLayer`](../../layers/jax/pp_utils.md#PPMissingLayer), [`get_pp_group`](../../distributed/jax_parallel_state.md#get_pp_group), [`make_layers`](../../layers/jax/pp_utils.md#make_layers), [`JaxEmbed`](../../layers/jax/embed.md#JaxEmbed), [`is_first_rank`](../../distributed/jax_parallel_state.md#GroupCoordinator.is_first_rank), [`is_last_rank`](../../distributed/jax_parallel_state.md#GroupCoordinator.is_last_rank), [`Qwen2Model`](qwen2.md#Qwen2Model), [`init_fn`](qwen3.md#init_fn), [`Qwen3DecoderLayer`](qwen3.md#Qwen3DecoderLayer)
- used by: [`compute_logits`](qwen3.md#Qwen3ForCausalLM.compute_logits), [`model`](qwen3.md#Qwen3ForCausalLM.model), [`Qwen2Model`](qwen2.md#Qwen2Model)

## Module values
- `init_fn` — [`L49`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L49)
- `logger` — [`L47`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L47)

