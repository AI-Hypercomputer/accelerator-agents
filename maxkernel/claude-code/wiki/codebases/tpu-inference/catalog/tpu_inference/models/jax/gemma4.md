---
title: 'Module: tpu_inference/models/jax/gemma4.py'
type: catalog
provenance: extracted
module: tpu_inference/models/jax/gemma4.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.jax.gemma4`/
symbols:
  Gemma4Attention.__call__: Gemma4Attention#__call__().
  Gemma4DecoderLayer.__call__: Gemma4DecoderLayer#__call__().
  init_fn: init_fn.
  Gemma4Attention.v_proj: Gemma4Attention#v_proj.
  Gemma4Model.compute_per_layer_inputs: Gemma4Model#compute_per_layer_inputs().
  Gemma4Model.embed_tokens: Gemma4Model#embed_tokens.
  Gemma4Model.__call__: Gemma4Model#__call__().
  Gemma4Model.embed_tokens_per_layer: Gemma4Model#embed_tokens_per_layer.
  Gemma4Model.per_layer_projection_scale: Gemma4Model#per_layer_projection_scale.
  Gemma4Attention.q_proj: Gemma4Attention#q_proj.
  Gemma4Attention.k_proj: Gemma4Attention#k_proj.
  Gemma4DecoderLayer.post_per_layer_input_norm: Gemma4DecoderLayer#post_per_layer_input_norm.
  Gemma4Attention.rope_proportion: Gemma4Attention#rope_proportion.
  Gemma4Attention.o_proj: Gemma4Attention#o_proj.
  Gemma4ForCausalLM.compute_logits: Gemma4ForCausalLM#compute_logits().
  Gemma4Model.norm: Gemma4Model#norm.
  Gemma4Attention.head_dim: Gemma4Attention#head_dim.
  Gemma4Attention.head_dim_original: Gemma4Attention#head_dim_original.
  Gemma4Attention.q_norm: Gemma4Attention#q_norm.
  Gemma4Attention.k_norm: Gemma4Attention#k_norm.
  Gemma4Model.layers: Gemma4Model#layers.
  Gemma4Router.__call__: Gemma4Router#__call__().
  Gemma4MoE.load_weights: Gemma4MoE#load_weights().
  Gemma4ForCausalLM.__call__: Gemma4ForCausalLM#__call__().
  Gemma4DecoderLayer.pre_feedforward_layernorm: Gemma4DecoderLayer#pre_feedforward_layernorm.
  Gemma4ForCausalLM.lm_head: Gemma4ForCausalLM#lm_head.
  Gemma4Model.is_last_rank: Gemma4Model#is_last_rank.
  Gemma4ForCausalLM: Gemma4ForCausalLM#
  Gemma4Attention.num_kv_heads: Gemma4Attention#num_kv_heads.
  Gemma4Attention.mesh: Gemma4Attention#mesh.
  Gemma4Attention.v_norm: Gemma4Attention#v_norm.
  Gemma4DecoderLayer.post_feedforward_layernorm: Gemma4DecoderLayer#post_feedforward_layernorm.
  Gemma4Attention.kv_cache_quantized_dtype: Gemma4Attention#kv_cache_quantized_dtype.
  Gemma4DecoderLayer.per_layer_input_gate: Gemma4DecoderLayer#per_layer_input_gate.
  Gemma4ForCausalLM.final_logit_softcapping: Gemma4ForCausalLM#final_logit_softcapping.
  Gemma4MLP.__call__: Gemma4MLP#__call__().
  Gemma4Attention.is_sliding: Gemma4Attention#is_sliding.
  Gemma4DecoderLayer.moe: Gemma4DecoderLayer#moe.
  Gemma4ForCausalLM.load_weights: Gemma4ForCausalLM#load_weights().
  Gemma4DecoderLayer.per_layer_projection: Gemma4DecoderLayer#per_layer_projection.
  Gemma4DecoderLayer.pre_feedforward_layernorm_2: Gemma4DecoderLayer#pre_feedforward_layernorm_2.
  Gemma4Model.is_first_rank: Gemma4Model#is_first_rank.
  Gemma4Model.num_hidden_layers: Gemma4Model#num_hidden_layers.
  Gemma4Model.per_layer_model_projection: Gemma4Model#per_layer_model_projection.
  Gemma4Model.per_layer_projection_norm: Gemma4Model#per_layer_projection_norm.
  Gemma4Attention.layer_type: Gemma4Attention#layer_type.
  Gemma4Attention.rope_scaling: Gemma4Attention#rope_scaling.
  Gemma4ForCausalLM.model: Gemma4ForCausalLM#model.
  Gemma4MLP.gate_up_proj: Gemma4MLP#gate_up_proj.
  Gemma4MLP.down_proj: Gemma4MLP#down_proj.
  Gemma4Router.norm: Gemma4Router#norm.
  Gemma4Router.scale: Gemma4Router#scale.
  Gemma4Router.proj: Gemma4Router#proj.
  Gemma4Attention.sliding_window: Gemma4Attention#sliding_window.
  Gemma4Attention.kv_sharing_target_layer_name: Gemma4Attention#kv_sharing_target_layer_name.
  Gemma4DecoderLayer.input_layernorm: Gemma4DecoderLayer#input_layernorm.
  Gemma4DecoderLayer.post_attention_layernorm: Gemma4DecoderLayer#post_attention_layernorm.
  Gemma4MLP: Gemma4MLP#
  Gemma4Model: Gemma4Model#
  Gemma4MoE.__init__: Gemma4MoE#__init__().
  Gemma4Attention.hidden_size: Gemma4Attention#hidden_size.
  Gemma4DecoderLayer.layer_type: Gemma4DecoderLayer#layer_type.
  Gemma4DecoderLayer.mlp: Gemma4DecoderLayer#mlp.
  Gemma4DecoderLayer.enable_moe_block: Gemma4DecoderLayer#enable_moe_block.
  Gemma4DecoderLayer.router: Gemma4DecoderLayer#router.
  Gemma4DecoderLayer.post_feedforward_layernorm_1: Gemma4DecoderLayer#post_feedforward_layernorm_1.
  Gemma4DecoderLayer.post_feedforward_layernorm_2: Gemma4DecoderLayer#post_feedforward_layernorm_2.
  Gemma4Router.hidden_size: Gemma4Router#hidden_size.
  Gemma4Attention.num_heads: Gemma4Attention#num_heads.
  Gemma4Attention.rope_theta: Gemma4Attention#rope_theta.
  Gemma4Model.hidden_size_per_layer_input: Gemma4Model#hidden_size_per_layer_input.
  logger: logger.
  Gemma4Router: Gemma4Router#
  Gemma4Router.root_size: Gemma4Router#root_size.
  Gemma4MoE: Gemma4MoE#
  Gemma4Attention: Gemma4Attention#
  Gemma4DecoderLayer: Gemma4DecoderLayer#
  Gemma4DecoderLayer.self_attn: Gemma4DecoderLayer#self_attn.
  Gemma4DecoderLayer.experts: Gemma4DecoderLayer#experts.
  Gemma4Attention.rms_norm_eps: Gemma4Attention#rms_norm_eps.
  Gemma4Attention.qkv_proj: Gemma4Attention#qkv_proj.
  Gemma4Attention.is_kv_shared_layer: Gemma4Attention#is_kv_shared_layer.
  Gemma4Model.start_layer: Gemma4Model#start_layer.
  Gemma4MLP.__init__: Gemma4MLP#__init__().
  Gemma4Router.per_expert_scale: Gemma4Router#per_expert_scale.
  Gemma4Attention.__init__: Gemma4Attention#__init__().
  Gemma4DecoderLayer.__init__: Gemma4DecoderLayer#__init__().
  Gemma4DecoderLayer.is_sliding: Gemma4DecoderLayer#is_sliding.
  Gemma4ForCausalLM.WeightLoader: Gemma4ForCausalLM#WeightLoader.
  Gemma4DecoderLayer.hidden_size_per_layer_input: Gemma4DecoderLayer#hidden_size_per_layer_input.
  Gemma4Model.embedding_scale: Gemma4Model#embedding_scale.
  Gemma4Model.vocab_size_per_layer_input: Gemma4Model#vocab_size_per_layer_input.
  Gemma4Model.embed_scale_per_layer: Gemma4Model#embed_scale_per_layer.
  Gemma4Model.per_layer_input_scale: Gemma4Model#per_layer_input_scale.
  Gemma4Model.end_layer: Gemma4Model#end_layer.
  Gemma4ForCausalLM.packed_modules_mapping: Gemma4ForCausalLM#packed_modules_mapping.
  Gemma4MLP.act_fn: Gemma4MLP#act_fn.
  Gemma4Attention.scaling: Gemma4Attention#scaling.
  Gemma4Attention._k_scale: Gemma4Attention#_k_scale.
  Gemma4Attention._v_scale: Gemma4Attention#_v_scale.
  Gemma4DecoderLayer.layer_scalar: Gemma4DecoderLayer#layer_scalar.
  Gemma4Router.__init__: Gemma4Router#__init__().
  Gemma4Attention._q_scale: Gemma4Attention#_q_scale.
  Gemma4Model.__init__: Gemma4Model#__init__().
  Gemma4ForCausalLM.__init__: Gemma4ForCausalLM#__init__().
  Gemma4ForCausalLM.vllm_config: Gemma4ForCausalLM#vllm_config.
  Gemma4ForCausalLM.mesh: Gemma4ForCausalLM#mesh.
---
# Module: [`tpu_inference/models/jax/gemma4.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py)

## Classes
### `Gemma4Attention`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/gemma4.py:238`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L238)
- signature: `class Gemma4Attention(JaxModule):`
- members:
  - `head_dim` — [`L302`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L302)
  - `head_dim_original` — [`L289`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L289)
  - `hidden_size` — [`L249`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L249)
  - `is_kv_shared_layer` — [`L427`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L427)
  - `is_sliding` — [`L262`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L262)
  - `k_norm` — [`L380`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L380)
  - `k_proj` — [`L340`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L340)
  - `kv_cache_quantized_dtype` — [`L416`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L416)
  - `kv_sharing_target_layer_name` — [`L428`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L428)
  - `layer_type` — [`L257`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L257)
  - `mesh` — [`L304`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L304)
  - `num_heads` — [`L250`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L250)
  - `num_kv_heads` — [`L298`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L298)
  - `o_proj` — [`L400`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L400)
  - `q_norm` — [`L370`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L370)
  - `q_proj` — [`L326`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L326)
  - `qkv_proj` — [`L325`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L325)
  - `rms_norm_eps` — [`L251`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L251)
  - `rope_proportion` — [`L277`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L277)
  - `rope_scaling` — [`L276`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L276)
  - `rope_theta` — [`L270`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L270)
  - `scaling` — [`L254`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L254)
  - `sliding_window` — [`L263`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L263)
  - `v_norm` — [`L390`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L390)
  - `v_proj` — [`L353`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L353)
- protocol/private: `__call__`[`L436`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L436), `__init__`[`L240`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L240), `_k_scale`[`L414`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L414), `_q_scale`[`L413`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L413), `_v_scale`[`L415`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L415)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../../layers/common/sharding.md#ShardingAxisName), [`JaxModule`](../../layers/__init__.md#JaxModule), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`JaxEinsum`](../../layers/jax/linear.md#JaxEinsum), [`VllmQuantConfig`](../../layers/vllm/quantization/configs.md#VllmQuantConfig), [`JaxRmsNorm`](../../layers/jax/norm.md#JaxRmsNorm), [`input_positions`](../../layers/common/attention_metadata.md#AttentionMetadata.input_positions), [`apply_rope`](../../layers/jax/rope_interface.md#apply_rope), [`attention`](../../layers/common/attention_interface.md#attention), [`init_fn`](gemma4.md#init_fn), [`quantize_kv`](../../layers/common/__init__.md#quantize_kv), [`compute_kv_share_map`](../common/kv_share.md#compute_kv_share_map), [`JaxQKVParallelLinear`](../../layers/jax/linear.md#JaxQKVParallelLinear), [`logger`](gemma4.md#logger), [`normalize_rope_scaling`](../../layers/jax/rope_interface.md#normalize_rope_scaling)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`self_attn`](gemma4.md#Gemma4DecoderLayer.self_attn)

### `Gemma4DecoderLayer`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/gemma4.py:517`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L517)
- signature: `class Gemma4DecoderLayer(JaxModule):`
- members:
  - `enable_moe_block` — [`L652`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L652)
  - `experts` — [`L661`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L661)
  - `hidden_size_per_layer_input` — [`L542`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L542)
  - `input_layernorm` — [`L547`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L547)
  - `is_sliding` — [`L538`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L538)
  - `layer_scalar` — [`L545`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L545)
  - `layer_type` — [`L533`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L533)
  - `mlp` — [`L593`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L593)
  - `moe` — [`L693`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L693)
  - `per_layer_input_gate` — [`L617`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L617)
  - `per_layer_projection` — [`L627`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L627)
  - `post_attention_layernorm` — [`L564`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L564)
  - `post_feedforward_layernorm` — [`L601`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L601)
  - `post_feedforward_layernorm_1` — [`L667`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L667)
  - `post_feedforward_layernorm_2` — [`L675`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L675)
  - `post_per_layer_input_norm` — [`L637`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L637)
  - `pre_feedforward_layernorm` — [`L574`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L574)
  - `pre_feedforward_layernorm_2` — [`L683`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L683)
  - `router` — [`L654`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L654)
  - `self_attn` — [`L556`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L556)
- protocol/private: `__call__`[`L698`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L698), `__init__`[`L519`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L519)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`JaxEinsum`](../../layers/jax/linear.md#JaxEinsum), [`VllmQuantConfig`](../../layers/vllm/quantization/configs.md#VllmQuantConfig), [`JaxRmsNorm`](../../layers/jax/norm.md#JaxRmsNorm), [`init_fn`](gemma4.md#init_fn), [`Gemma4MLP`](gemma4.md#Gemma4MLP), [`compute_kv_share_map`](../common/kv_share.md#compute_kv_share_map), [`Gemma4Attention`](gemma4.md#Gemma4Attention), [`Gemma4MoE`](gemma4.md#Gemma4MoE), [`Gemma4Router`](gemma4.md#Gemma4Router)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`layers`](gemma4.md#Gemma4Model.layers)

### `Gemma4ForCausalLM`  ·  implements/extends JaxModule, LoadableWithIterator
- def: [`tpu_inference/models/jax/gemma4.py:1022`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L1022)
- signature: `class Gemma4ForCausalLM(JaxModule, LoadableWithIterator):`
- members:
  - `compute_logits(self, hidden_states: jax.Array)` — [`L1120`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L1120)
  - `load_weights(self, weights: Iterable[Tuple[str, Any]])` — [`L1069`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L1069)
  - `WeightLoader` — [`L1034`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L1034)
  - `final_logit_softcapping` — [`L1051`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L1051)
  - `lm_head` — [`L1059`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L1059)
  - `mesh` — [`L1040`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L1040)
  - `model` — [`L1042`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L1042)
  - `packed_modules_mapping` — [`L1023`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L1023)
  - `vllm_config` — [`L1038`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L1038)
- protocol/private: `__call__`[`L1083`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L1083), `__init__`[`L1036`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L1036)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`JaxIntermediateTensors`](jax_intermediate_tensor.md#JaxIntermediateTensors), [`PPMissingLayer`](../../layers/jax/pp_utils.md#PPMissingLayer), [`LoadableWithIterator`](utils/weight_utils.md#LoadableWithIterator), [`embed_tokens`](gemma4.md#Gemma4Model.embed_tokens), [`JaxLmHead`](../../layers/jax/linear.md#JaxLmHead), [`load_weights`](utils/weight_utils.md#LoadableWithIterator.load_weights), [`tensors`](jax_intermediate_tensor.md#JaxIntermediateTensors.tensors), [`StandardWeightLoader`](utils/weight_utils.md#StandardWeightLoader), [`layers`](gemma4.md#Gemma4Model.layers), [`decode`](../../layers/jax/embed.md#JaxEmbed.decode), [`is_last_rank`](gemma4.md#Gemma4Model.is_last_rank), [`Gemma4Model`](gemma4.md#Gemma4Model)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`LoadableWithIterator`](utils/weight_utils.md#LoadableWithIterator), [`load_weights`](utils/weight_utils.md#LoadableWithIterator.load_weights), [`packed_modules_mapping`](gemma4_mm.md#Gemma4ForConditionalGeneration.packed_modules_mapping), [`packed_modules_mapping`](gemma4_mtp.md#Gemma4MTPForCausalLM.packed_modules_mapping)

### `Gemma4MLP`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/gemma4.py:57`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L57)
- signature: `class Gemma4MLP(JaxModule):`
- members:
  - `act_fn` — [`L94`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L94)
  - `down_proj` — [`L84`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L84)
  - `gate_up_proj` — [`L74`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L74)
- protocol/private: `__call__`[`L96`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L96), `__init__`[`L59`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L59)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`VllmQuantConfig`](../../layers/vllm/quantization/configs.md#VllmQuantConfig), [`init_fn`](gemma4.md#init_fn), [`JaxLinear`](../../layers/jax/linear.md#JaxLinear), [`JaxMergedColumnParallelLinear`](../../layers/jax/linear.md#JaxMergedColumnParallelLinear)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`mlp`](gemma4.md#Gemma4DecoderLayer.mlp), [`mlp`](gemma4_mtp.md#Gemma4MTPDecoderLayer.mlp)

### `Gemma4MoE`  ·  implements/extends JaxRoutedExperts
- def: [`tpu_inference/models/jax/gemma4.py:163`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L163)
- doc: Mixture of Experts for Gemma4 using FusedMoE.
- signature: `class Gemma4MoE(JaxRoutedExperts):`
- members:
  - `load_weights(self, weights: Iterable)` — [`L197`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L197) — Load weights for Gemma4 MoE layer.
- protocol/private: `__init__`[`L172`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L172)
- uses (calls/refs, reference-scoped): [`load_nnx_param_from_reshaped_torch`](utils/weight_utils.md#load_nnx_param_from_reshaped_torch), [`JaxRoutedExperts`](../../layers/jax/moe/moe.md#JaxRoutedExperts), [`kernel_gating_EDF`](../../layers/jax/moe/moe.md#JaxRoutedExperts.kernel_gating_EDF), [`kernel_up_proj_EDF`](../../layers/jax/moe/moe.md#JaxRoutedExperts.kernel_up_proj_EDF), [`kernel_down_proj_EFD`](../../layers/jax/moe/moe.md#JaxRoutedExperts.kernel_down_proj_EFD), [`__init__`](../../layers/jax/moe/moe.md#JaxRoutedExperts.__init__)
- used by: [`JaxRoutedExperts`](../../layers/jax/moe/moe.md#JaxRoutedExperts), [`load_weights`](../../layers/jax/moe/moe.md#JaxRoutedExperts.load_weights), [`experts`](gemma4.md#Gemma4DecoderLayer.experts)

### `Gemma4Model`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/gemma4.py:760`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L760)
- signature: `class Gemma4Model(JaxModule):`
- members:
  - `compute_per_layer_inputs(self, input_ids: Optional[jax.Array], inputs_embeds: jax.Array, is_multimodal: Optional[jax.Array] = None)` — [`L894`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L894) — Compute per_layer_inputs of shape [T, L, P].
  - `embed_scale_per_layer` — [`L857`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L857)
  - `embed_tokens` — [`L791`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L791)
  - `embed_tokens_per_layer` — [`L812`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L812)
  - `embedding_scale` — [`L779`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L779)
  - `end_layer` — [`L868`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L868)
  - `hidden_size_per_layer_input` — [`L783`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L783)
  - `is_first_rank` — [`L775`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L775)
  - `is_last_rank` — [`L776`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L776)
  - `layers` — [`L868`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L868)
  - `norm` — [`L882`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L882)
  - `num_hidden_layers` — [`L787`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L787)
  - `per_layer_input_scale` — [`L858`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L858)
  - `per_layer_model_projection` — [`L835`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L835)
  - `per_layer_projection_norm` — [`L846`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L846)
  - `per_layer_projection_scale` — [`L859`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L859)
  - `start_layer` — [`L868`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L868)
  - `vocab_size_per_layer_input` — [`L785`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L785)
- protocol/private: `__call__`[`L966`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L966), `__init__`[`L762`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L762)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`JaxEinsum`](../../layers/jax/linear.md#JaxEinsum), [`JaxRmsNorm`](../../layers/jax/norm.md#JaxRmsNorm), [`PPMissingLayer`](../../layers/jax/pp_utils.md#PPMissingLayer), [`get_pp_group`](../../distributed/jax_parallel_state.md#get_pp_group), [`make_layers`](../../layers/jax/pp_utils.md#make_layers), [`init_fn`](gemma4.md#init_fn), [`load_nnx_param_from_reshaped_torch`](utils/weight_utils.md#load_nnx_param_from_reshaped_torch), [`JaxEmbed`](../../layers/jax/embed.md#JaxEmbed), [`is_first_rank`](../../distributed/jax_parallel_state.md#GroupCoordinator.is_first_rank), [`is_last_rank`](../../distributed/jax_parallel_state.md#GroupCoordinator.is_last_rank), [`weight`](../../layers/jax/embed.md#JaxEmbed.weight), [`Gemma4DecoderLayer`](gemma4.md#Gemma4DecoderLayer)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`filter_weights`](gemma4_mm.md#Gemma4ForConditionalGeneration.filter_weights), [`embed_input_ids`](gemma4_mm.md#Gemma4ForConditionalGeneration.embed_input_ids), [`compute_logits`](gemma4_mm.md#Gemma4ForConditionalGeneration.compute_logits), [`compute_logits`](gemma4.md#Gemma4ForCausalLM.compute_logits), [`final_logit_softcapping`](gemma4_mm.md#Gemma4ForConditionalGeneration.final_logit_softcapping), [`model`](gemma4_mm.md#Gemma4ForConditionalGeneration.model), [`final_logit_softcapping`](gemma4.md#Gemma4ForCausalLM.final_logit_softcapping), [`load_weights`](gemma4.md#Gemma4ForCausalLM.load_weights), [`model`](gemma4.md#Gemma4ForCausalLM.model)

### `Gemma4Router`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/gemma4.py:105`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L105)
- doc: Router for Gemma4 MoE that preprocesses input before projection.
- signature: `class Gemma4Router(JaxModule):`
- members:
  - `__call__(self, x: jax.Array)` — [`L154`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L154) — Returns raw router logits [T, E].
  - `hidden_size` — [`L125`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L125)
  - `norm` — [`L128`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L128)
  - `per_expert_scale` — [`L149`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L149)
  - `proj` — [`L141`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L141)
  - `root_size` — [`L139`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L139)
  - `scale` — [`L135`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L135)
- protocol/private: `__init__`[`L116`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L116)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`JaxRmsNorm`](../../layers/jax/norm.md#JaxRmsNorm), [`init_fn`](gemma4.md#init_fn), [`JaxLinear`](../../layers/jax/linear.md#JaxLinear)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`router`](gemma4.md#Gemma4DecoderLayer.router)

## Module values
- `init_fn` — [`L53`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L53)
- `logger` — [`L51`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4.py#L51)

