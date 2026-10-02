---
title: 'Module: tpu_inference/models/jax/qwen2.py'
type: catalog
provenance: extracted
module: tpu_inference/models/jax/qwen2.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.jax.qwen2`/
symbols:
  Qwen2Attention.__call__: Qwen2Attention#__call__().
  Qwen2Model.embed_tokens: Qwen2Model#embed_tokens.
  Qwen2Model.__call__: Qwen2Model#__call__().
  Qwen2Attention.head_dim: Qwen2Attention#head_dim.
  init_fn: init_fn.
  Qwen2Attention.q_proj: Qwen2Attention#q_proj.
  Qwen2Attention.k_proj: Qwen2Attention#k_proj.
  Qwen2Attention.v_proj: Qwen2Attention#v_proj.
  Qwen2Attention.o_proj: Qwen2Attention#o_proj.
  Qwen2DecoderLayer.__call__: Qwen2DecoderLayer#__call__().
  Qwen2Model.is_last_rank: Qwen2Model#is_last_rank.
  Qwen2Model.norm: Qwen2Model#norm.
  Qwen2MLP.__init__: Qwen2MLP#__init__().
  Qwen2MLP.__call__: Qwen2MLP#__call__().
  Qwen2Model: Qwen2Model#
  Qwen2ForCausalLM.__call__: Qwen2ForCausalLM#__call__().
  Qwen2ForCausalLM.compute_logits: Qwen2ForCausalLM#compute_logits().
  Qwen2Attention.head_dim_original: Qwen2Attention#head_dim_original.
  Qwen2Attention.num_heads: Qwen2Attention#num_heads.
  Qwen2Attention.num_kv_heads: Qwen2Attention#num_kv_heads.
  Qwen2Model.layers: Qwen2Model#layers.
  Qwen2DecoderLayer: Qwen2DecoderLayer#
  Qwen2Attention.hidden_size: Qwen2Attention#hidden_size.
  Qwen2DecoderLayer.__init__: Qwen2DecoderLayer#__init__().
  Qwen2ForCausalLM: Qwen2ForCausalLM#
  Qwen2MLP.gate_proj: Qwen2MLP#gate_proj.
  Qwen2MLP.up_proj: Qwen2MLP#up_proj.
  Qwen2MLP.down_proj: Qwen2MLP#down_proj.
  Qwen2MLP.act_fn: Qwen2MLP#act_fn.
  Qwen2DecoderLayer.input_layernorm: Qwen2DecoderLayer#input_layernorm.
  Qwen2DecoderLayer.post_attention_layernorm: Qwen2DecoderLayer#post_attention_layernorm.
  Qwen2Model.is_first_rank: Qwen2Model#is_first_rank.
  Qwen2MLP: Qwen2MLP#
  Qwen2Attention.kv_cache_quantized_dtype: Qwen2Attention#kv_cache_quantized_dtype.
  Qwen2Attention.rope_theta: Qwen2Attention#rope_theta.
  Qwen2Attention.rope_scaling: Qwen2Attention#rope_scaling.
  Qwen2ForCausalLM.model: Qwen2ForCausalLM#model.
  modeling_flax_utils: modeling_flax_utils.
  Qwen2Attention: Qwen2Attention#
  Qwen2DecoderLayer.self_attn: Qwen2DecoderLayer#self_attn.
  Qwen2DecoderLayer.mlp: Qwen2DecoderLayer#mlp.
  Qwen2ForCausalLM.lm_head: Qwen2ForCausalLM#lm_head.
  logger: logger.
  Qwen2Attention.__init__: Qwen2Attention#__init__().
  Qwen2ForCausalLM.WeightLoader: Qwen2ForCausalLM#WeightLoader.
  Qwen2Attention.mesh: Qwen2Attention#mesh.
  Qwen2Attention._k_scale: Qwen2Attention#_k_scale.
  Qwen2Attention._v_scale: Qwen2Attention#_v_scale.
  Qwen2Model.start_layer: Qwen2Model#start_layer.
  Qwen2Model.end_layer: Qwen2Model#end_layer.
  Qwen2Attention._q_scale: Qwen2Attention#_q_scale.
  Qwen2Model.__init__: Qwen2Model#__init__().
  Qwen2ForCausalLM.__init__: Qwen2ForCausalLM#__init__().
  Qwen2ForCausalLM.vllm_config: Qwen2ForCausalLM#vllm_config.
  Qwen2ForCausalLM.mesh: Qwen2ForCausalLM#mesh.
---
# Module: [`tpu_inference/models/jax/qwen2.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py)

## Classes
### `Qwen2Attention`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/qwen2.py:105`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L105)
- signature: `class Qwen2Attention(JaxModule):`
- members:
  - `head_dim` — [`L124`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L124)
  - `head_dim_original` — [`L122`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L122)
  - `hidden_size` — [`L116`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L116)
  - `k_proj` — [`L145`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L145)
  - `kv_cache_quantized_dtype` — [`L180`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L180)
  - `mesh` — [`L132`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L132)
  - `num_heads` — [`L117`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L117)
  - `num_kv_heads` — [`L118`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L118)
  - `o_proj` — [`L167`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L167)
  - `q_proj` — [`L134`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L134)
  - `rope_scaling` — [`L120`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L120)
  - `rope_theta` — [`L119`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L119)
  - `v_proj` — [`L156`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L156)
- protocol/private: `__call__`[`L185`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L185), `__init__`[`L107`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L107), `_k_scale`[`L178`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L178), `_q_scale`[`L177`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L177), `_v_scale`[`L179`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L179)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`JaxEinsum`](../../layers/jax/linear.md#JaxEinsum), [`VllmQuantConfig`](../../layers/vllm/quantization/configs.md#VllmQuantConfig), [`input_positions`](../../layers/common/attention_metadata.md#AttentionMetadata.input_positions), [`apply_rope`](../../layers/jax/rope_interface.md#apply_rope), [`attention`](../../layers/common/attention_interface.md#attention), [`quantize_kv`](../../layers/common/__init__.md#quantize_kv), [`init_fn`](qwen2.md#init_fn), [`get_rope_scaling`](../../layers/jax/rope_interface.md#get_rope_scaling), [`get_rope_theta`](../../layers/jax/rope_interface.md#get_rope_theta), `hidden_size`, `num_attention_heads`, `num_key_value_heads`
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`self_attn`](qwen2.md#Qwen2DecoderLayer.self_attn)

### `Qwen2DecoderLayer`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/qwen2.py:230`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L230)
- signature: `class Qwen2DecoderLayer(JaxModule):`
- members:
  - `input_layernorm` — [`L243`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L243)
  - `mlp` — [`L268`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L268)
  - `post_attention_layernorm` — [`L259`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L259)
  - `self_attn` — [`L252`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L252)
- protocol/private: `__call__`[`L276`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L276), `__init__`[`L232`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L232)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`VllmQuantConfig`](../../layers/vllm/quantization/configs.md#VllmQuantConfig), [`JaxRmsNorm`](../../layers/jax/norm.md#JaxRmsNorm), [`init_fn`](qwen2.md#init_fn), [`Qwen2MLP`](qwen2.md#Qwen2MLP), [`Qwen2Attention`](qwen2.md#Qwen2Attention), [`Qwen3DecoderLayer`](qwen3.md#Qwen3DecoderLayer), `hidden_size`, `rms_norm_eps`
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`layers`](qwen2.md#Qwen2Model.layers), [`Qwen3DecoderLayer`](qwen3.md#Qwen3DecoderLayer)

### `Qwen2ForCausalLM`  ·  implements/extends JaxModule, LoadableWithIterator
- def: [`tpu_inference/models/jax/qwen2.py:381`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L381)
- signature: `class Qwen2ForCausalLM(JaxModule, LoadableWithIterator):`
- members:
  - `compute_logits(self, hidden_states: jax.Array)` — [`L438`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L438)
  - `WeightLoader` — [`L382`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L382)
  - `lm_head` — [`L402`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L402)
  - `mesh` — [`L388`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L388)
  - `model` — [`L390`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L390)
  - `vllm_config` — [`L386`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L386)
- protocol/private: `__call__`[`L410`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L410), `__init__`[`L384`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L384)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`JaxIntermediateTensors`](jax_intermediate_tensor.md#JaxIntermediateTensors), [`LoadableWithIterator`](utils/weight_utils.md#LoadableWithIterator), [`JaxLmHead`](../../layers/jax/linear.md#JaxLmHead), [`tensors`](jax_intermediate_tensor.md#JaxIntermediateTensors.tensors), [`embed_tokens`](qwen2.md#Qwen2Model.embed_tokens), [`StandardWeightLoader`](utils/weight_utils.md#StandardWeightLoader), [`Qwen2Model`](qwen2.md#Qwen2Model), [`decode`](../../layers/jax/embed.md#JaxEmbed.decode)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`_get_model_architecture`](../common/model_loader.md#_get_model_architecture), [`LoadableWithIterator`](utils/weight_utils.md#LoadableWithIterator)

### `Qwen2MLP`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/qwen2.py:53`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L53)
- signature: `class Qwen2MLP(JaxModule):`
- members:
  - `act_fn` — [`L95`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L95)
  - `down_proj` — [`L85`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L85)
  - `gate_proj` — [`L65`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L65)
  - `up_proj` — [`L75`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L75)
- protocol/private: `__call__`[`L97`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L97), `__init__`[`L55`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L55)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`VllmQuantConfig`](../../layers/vllm/quantization/configs.md#VllmQuantConfig), [`JaxLinear`](../../layers/jax/linear.md#JaxLinear), [`init_fn`](qwen2.md#init_fn), [`ACT2FN`](../../layers/jax/layers.md#FlaxUtils.ACT2FN), [`modeling_flax_utils`](qwen2.md#modeling_flax_utils), `hidden_size`, `hidden_act`, `intermediate_size`
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`mlp`](qwen2.md#Qwen2DecoderLayer.mlp), [`mlp`](qwen3.md#Qwen3DecoderLayer.mlp)

### `Qwen2Model`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/qwen2.py:297`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L297)
- signature: `class Qwen2Model(JaxModule):`
- members:
  - `embed_tokens` — [`L318`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L318)
  - `end_layer` — [`L330`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L330)
  - `is_first_rank` — [`L313`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L313)
  - `is_last_rank` — [`L314`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L314)
  - `layers` — [`L330`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L330)
  - `norm` — [`L344`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L344)
  - `start_layer` — [`L330`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L330)
- protocol/private: `__call__`[`L356`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L356), `__init__`[`L299`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L299)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`JaxRmsNorm`](../../layers/jax/norm.md#JaxRmsNorm), [`PPMissingLayer`](../../layers/jax/pp_utils.md#PPMissingLayer), [`get_pp_group`](../../distributed/jax_parallel_state.md#get_pp_group), [`make_layers`](../../layers/jax/pp_utils.md#make_layers), [`JaxEmbed`](../../layers/jax/embed.md#JaxEmbed), [`is_first_rank`](../../distributed/jax_parallel_state.md#GroupCoordinator.is_first_rank), [`init_fn`](qwen2.md#init_fn), [`is_last_rank`](../../distributed/jax_parallel_state.md#GroupCoordinator.is_last_rank), [`Qwen2DecoderLayer`](qwen2.md#Qwen2DecoderLayer), [`Qwen3Model`](qwen3.md#Qwen3Model)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`embed_input_ids`](qwen2_5_vl.md#Qwen2_5_VLForConditionalGeneration.embed_input_ids), [`model`](qwen2_5_vl.md#Qwen2_5_VLForConditionalGeneration.model), [`compute_logits`](qwen2_5_vl.md#Qwen2_5_VLForConditionalGeneration.compute_logits), [`compute_logits`](qwen2.md#Qwen2ForCausalLM.compute_logits), [`model`](qwen2.md#Qwen2ForCausalLM.model), [`Qwen3Model`](qwen3.md#Qwen3Model)

## Module values
- `init_fn` — [`L49`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L49)
- `logger` — [`L47`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L47)
- `modeling_flax_utils` — [`L50`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2.py#L50)

