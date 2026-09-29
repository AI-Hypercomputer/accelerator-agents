---
title: 'Module: tpu_inference/models/jax/llama4.py'
type: catalog
provenance: extracted
module: tpu_inference/models/jax/llama4.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.jax.llama4`/
symbols:
  Llama4ForCausalLM.end_layer: Llama4ForCausalLM#end_layer.
  Llama4WeightLoader.load_weights: Llama4WeightLoader#load_weights().
  JAXLlama4VisionEncoderLayer.self_attn: JAXLlama4VisionEncoderLayer#self_attn.
  Llama4ForCausalLM.lm_head: Llama4ForCausalLM#lm_head.
  Llama4ForCausalLM.embedder: Llama4ForCausalLM#embedder.
  Llama4ForCausalLM.final_norm: Llama4ForCausalLM#final_norm.
  Llama4ForCausalLM.__call__: Llama4ForCausalLM#__call__().
  Llama4WeightLoader._map_llama4_gate_up_proj: Llama4WeightLoader#_map_llama4_gate_up_proj().
  JAXLlama4VisionModel.__call__: JAXLlama4VisionModel#__call__().
  Llama4ForCausalLM.load_weights: Llama4ForCausalLM#load_weights().
  Llama4ForCausalLM._print_model_architecture: Llama4ForCausalLM#_print_model_architecture().
  JAXLlama4VisionEncoderLayer.__call__: JAXLlama4VisionEncoderLayer#__call__().
  logger: logger.
  Llama4ForCausalLM.is_last_rank: Llama4ForCausalLM#is_last_rank.
  Llama4ForCausalLM.rng: Llama4ForCausalLM#rng.
  Llama4ForCausalLM.mesh: Llama4ForCausalLM#mesh.
  Llama4ForCausalLM.hidden_size: Llama4ForCausalLM#hidden_size.
  Llama4WeightLoader._transpose_map: Llama4WeightLoader#_transpose_map.
  Llama4ForCausalLM.rope_scaling: Llama4ForCausalLM#rope_scaling.
  Llama4WeightLoader._loaded_to_standardized_keys: Llama4WeightLoader#_loaded_to_standardized_keys.
  JAXLlama4VisionModel.positional_embedding_vlm: JAXLlama4VisionModel#positional_embedding_vlm.
  JAXUnfoldConvolution.__call__: JAXUnfoldConvolution#__call__().
  JAXLlama4VisionMLP2.__call__: JAXLlama4VisionMLP2#__call__().
  JAXLlama4VisionPixelShuffleMLP.__call__: JAXLlama4VisionPixelShuffleMLP#__call__().
  Llama4WeightLoader.map_loaded_to_standardized_name: Llama4WeightLoader#map_loaded_to_standardized_name().
  Llama4ForCausalLM.is_first_rank: Llama4ForCausalLM#is_first_rank.
  JAXLlama4VisionMLP2.dropout: JAXLlama4VisionMLP2#dropout.
  JAXLlama4VisionModel.num_patches: JAXLlama4VisionModel#num_patches.
  JAXLlama4VisionModel.class_embedding: JAXLlama4VisionModel#class_embedding.
  JAXLlama4VisionModel.layernorm_pre: JAXLlama4VisionModel#layernorm_pre.
  JAXLlama4VisionModel.layernorm_post: JAXLlama4VisionModel#layernorm_post.
  Llama4WeightLoader.expert_prefix: Llama4WeightLoader#expert_prefix.
  Llama4WeightLoader.pp_missing_layers: Llama4WeightLoader#pp_missing_layers.
  Llama4ForCausalLM.compute_logits: Llama4ForCausalLM#compute_logits().
  JAXLlama4VisionMLP.__call__: JAXLlama4VisionMLP#__call__().
  Llama4ForCausalLM.enable_return_routed_experts: Llama4ForCausalLM#enable_return_routed_experts.
  Llama4ForCausalLM.layers: Llama4ForCausalLM#layers.
  JAXLlama4VisionMLP2.dropout_rate: JAXLlama4VisionMLP2#dropout_rate.
  Llama4WeightLoader.expert_weights_buffer: Llama4WeightLoader#expert_weights_buffer.
  JAXLlama4VisionModel.hidden_size: JAXLlama4VisionModel#hidden_size.
  Llama4WeightLoader: Llama4WeightLoader#
  Llama4ForCausalLM.WeightLoader: Llama4ForCausalLM#WeightLoader.
  Llama4ForCausalLM.is_verbose: Llama4ForCausalLM#is_verbose.
  JAXLlama4VisionEncoderLayer.mlp: JAXLlama4VisionEncoderLayer#mlp.
  JAXLlama4VisionEncoder.layers: JAXLlama4VisionEncoder#layers.
  JAXLlama4VisionPixelShuffleMLP.pixel_shuffle_mlp: JAXLlama4VisionPixelShuffleMLP#pixel_shuffle_mlp.
  JAXLlama4VisionModel.patch_embedding: JAXLlama4VisionModel#patch_embedding.
  JAXLlama4VisionModel.model: JAXLlama4VisionModel#model.
  JAXLlama4VisionModel.vision_adapter: JAXLlama4VisionModel#vision_adapter.
  Llama4WeightLoader.is_verbose: Llama4WeightLoader#is_verbose.
  Llama4WeightLoader.quantization_config: Llama4WeightLoader#quantization_config.
  Llama4WeightLoader._get_layer_num: Llama4WeightLoader#_get_layer_num().
  Llama4ForCausalLM.vllm_config: Llama4ForCausalLM#vllm_config.
  Llama4ForCausalLM.hidden_act: Llama4ForCausalLM#hidden_act.
  Llama4ForCausalLM.rms_norm_eps: Llama4ForCausalLM#rms_norm_eps.
  Llama4ForCausalLM.start_layer: Llama4ForCausalLM#start_layer.
  Llama4WeightLoader.__init__: Llama4WeightLoader#__init__().
  JAXLlama4VisionEncoder.__call__: JAXLlama4VisionEncoder#__call__().
  JAXLlama4VisionModel.__init__: JAXLlama4VisionModel#__init__().
  JAXLlama4MultiModalProjector.__call__: JAXLlama4MultiModalProjector#__call__().
  Llama4ForCausalLM: Llama4ForCausalLM#
  Llama4ForCausalLM.vocab_size: Llama4ForCausalLM#vocab_size.
  Llama4ForCausalLM.num_layers: Llama4ForCausalLM#num_layers.
  Llama4ForCausalLM.intermediate_size_moe: Llama4ForCausalLM#intermediate_size_moe.
  Llama4ForCausalLM.num_local_experts: Llama4ForCausalLM#num_local_experts.
  Llama4ForCausalLM.no_rope_layer_interval: Llama4ForCausalLM#no_rope_layer_interval.
  Llama4ForCausalLM.interleave_moe_layer_step: Llama4ForCausalLM#interleave_moe_layer_step.
  Llama4ForCausalLM.num_attention_heads: Llama4ForCausalLM#num_attention_heads.
  Llama4ForCausalLM.num_key_value_heads: Llama4ForCausalLM#num_key_value_heads.
  Llama4ForCausalLM.head_dim: Llama4ForCausalLM#head_dim.
  JAXUnfoldConvolution.num_channels: JAXUnfoldConvolution#num_channels.
  JAXLlama4VisionModel: JAXLlama4VisionModel#
  JAXLlama4VisionModel.scale: JAXLlama4VisionModel#scale.
  JAXLlama4VisionModel.norm_eps: JAXLlama4VisionModel#norm_eps.
  JAXLlama4MultiModalProjector: JAXLlama4MultiModalProjector#
  Llama4WeightLoader.interleave_moe_layer_step: Llama4WeightLoader#interleave_moe_layer_step.
  Llama4WeightLoader._weight_shape_map: Llama4WeightLoader#_weight_shape_map.
  Llama4WeightLoader._get_expert_num: Llama4WeightLoader#_get_expert_num().
  Llama4ForCausalLM.intermediate_size_mlp: Llama4ForCausalLM#intermediate_size_mlp.
  Llama4ForCausalLM.num_shared_experts: Llama4ForCausalLM#num_shared_experts.
  Llama4ForCausalLM.use_qk_norm: Llama4ForCausalLM#use_qk_norm.
  JAXUnfoldConvolution: JAXUnfoldConvolution#
  JAXUnfoldConvolution.kernel_size: JAXUnfoldConvolution#kernel_size.
  JAXUnfoldConvolution.linear: JAXUnfoldConvolution#linear.
  JAXLlama4VisionMLP: JAXLlama4VisionMLP#
  JAXLlama4VisionMLP.fc1: JAXLlama4VisionMLP#fc1.
  JAXLlama4VisionMLP.fc2: JAXLlama4VisionMLP#fc2.
  JAXLlama4VisionEncoderLayer: JAXLlama4VisionEncoderLayer#
  JAXLlama4VisionEncoderLayer.input_layernorm: JAXLlama4VisionEncoderLayer#input_layernorm.
  JAXLlama4VisionEncoderLayer.post_attention_layernorm: JAXLlama4VisionEncoderLayer#post_attention_layernorm.
  JAXLlama4VisionEncoder: JAXLlama4VisionEncoder#
  jax_pixel_shuffle: jax_pixel_shuffle().
  JAXLlama4VisionMLP2: JAXLlama4VisionMLP2#
  JAXLlama4VisionMLP2.fc1: JAXLlama4VisionMLP2#fc1.
  JAXLlama4VisionMLP2.fc2: JAXLlama4VisionMLP2#fc2.
  JAXLlama4VisionPixelShuffleMLP: JAXLlama4VisionPixelShuffleMLP#
  JAXLlama4VisionPixelShuffleMLP.pixel_shuffle_ratio: JAXLlama4VisionPixelShuffleMLP#pixel_shuffle_ratio.
  JAXLlama4VisionModel.image_size: JAXLlama4VisionModel#image_size.
  JAXLlama4VisionModel.patch_size: JAXLlama4VisionModel#patch_size.
  JAXLlama4VisionModel.num_channels: JAXLlama4VisionModel#num_channels.
  JAXLlama4VisionModel.vision_rope: JAXLlama4VisionModel#vision_rope.
  JAXLlama4MultiModalProjector.linear: JAXLlama4MultiModalProjector#linear.
  Llama4ForCausalLM.__init__: Llama4ForCausalLM#__init__().
  JAXUnfoldConvolution.__init__: JAXUnfoldConvolution#__init__().
  JAXLlama4VisionMLP.__init__: JAXLlama4VisionMLP#__init__().
  JAXLlama4VisionEncoderLayer.__init__: JAXLlama4VisionEncoderLayer#__init__().
  JAXLlama4VisionEncoderLayer.hidden_size: JAXLlama4VisionEncoderLayer#hidden_size.
  JAXLlama4VisionEncoder.__init__: JAXLlama4VisionEncoder#__init__().
  JAXLlama4VisionMLP2.__init__: JAXLlama4VisionMLP2#__init__().
  JAXLlama4VisionPixelShuffleMLP.__init__: JAXLlama4VisionPixelShuffleMLP#__init__().
  JAXLlama4MultiModalProjector.__init__: JAXLlama4MultiModalProjector#__init__().
---
# Module: [`tpu_inference/models/jax/llama4.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py)

## Classes
### `JAXLlama4MultiModalProjector`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/llama4.py:1212`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1212)
- signature: `class JAXLlama4MultiModalProjector(nnx.Module):`
- members:
  - `linear` — [`L1220`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1220)
- protocol/private: `__call__`[`L1231`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1231), `__init__`[`L1214`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1214)
- used by: [`multi_modal_projector`](llama_guard_4.md#LlamaGuard4ForCausalLM.multi_modal_projector)

### `JAXLlama4VisionEncoder`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/llama4.py:950`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L950)
- signature: `class JAXLlama4VisionEncoder(nnx.Module):`
- members:
  - `layers` — [`L960`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L960)
- protocol/private: `__call__`[`L968`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L968), `__init__`[`L952`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L952)
- uses (calls/refs, reference-scoped): [`JAXLlama4VisionEncoderLayer`](llama4.md#JAXLlama4VisionEncoderLayer)
- used by: [`model`](llama4.md#JAXLlama4VisionModel.model)

### `JAXLlama4VisionEncoderLayer`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/llama4.py:848`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L848)
- signature: `class JAXLlama4VisionEncoderLayer(nnx.Module):`
- members:
  - `hidden_size` — [`L857`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L857)
  - `input_layernorm` — [`L892`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L892)
  - `mlp` — [`L887`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L887)
  - `post_attention_layernorm` — [`L900`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L900)
  - `self_attn` — [`L865`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L865)
- protocol/private: `__call__`[`L909`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L909), `__init__`[`L850`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L850)
- uses (calls/refs, reference-scoped): [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`input_positions`](../../layers/common/attention_metadata.md#AttentionMetadata.input_positions), [`dtype`](../../layers/jax/attention/llama4_attention.md#Llama4VisionAttention.dtype), [`head_dim`](../../layers/jax/attention/llama4_attention.md#Llama4VisionAttention.head_dim), [`rope_input_ordering`](../../layers/jax/attention/llama4_attention.md#Llama4VisionAttention.rope_input_ordering), [`Llama4VisionAttention`](../../layers/jax/attention/llama4_attention.md#Llama4VisionAttention), [`activation_attention_out_td`](../../layers/jax/attention/llama4_attention.md#Llama4VisionAttention.activation_attention_out_td), [`activation_attention_td`](../../layers/jax/attention/llama4_attention.md#Llama4VisionAttention.activation_attention_td), [`activation_q_td`](../../layers/jax/attention/llama4_attention.md#Llama4VisionAttention.activation_q_td), [`hidden_size`](../../layers/jax/attention/llama4_attention.md#Llama4VisionAttention.hidden_size), [`is_causal`](../../layers/jax/attention/llama4_attention.md#Llama4VisionAttention.is_causal), [`mesh`](../../layers/jax/attention/llama4_attention.md#Llama4VisionAttention.mesh), [`num_attention_heads`](../../layers/jax/attention/llama4_attention.md#Llama4VisionAttention.num_attention_heads), [`num_key_value_heads`](../../layers/jax/attention/llama4_attention.md#Llama4VisionAttention.num_key_value_heads), [`rope_scaling`](../../layers/jax/attention/llama4_attention.md#Llama4VisionAttention.rope_scaling), [`rope_theta`](../../layers/jax/attention/llama4_attention.md#Llama4VisionAttention.rope_theta), [`JAXLlama4VisionMLP`](llama4.md#JAXLlama4VisionMLP), [`rngs`](../../layers/jax/attention/llama4_attention.md#Llama4VisionAttention.rngs), [`temperature_tuning`](../../layers/jax/attention/llama4_attention.md#Llama4VisionAttention.temperature_tuning), [`temperature_tuning_floor_scale`](../../layers/jax/attention/llama4_attention.md#Llama4VisionAttention.temperature_tuning_floor_scale), [`temperature_tuning_scale`](../../layers/jax/attention/llama4_attention.md#Llama4VisionAttention.temperature_tuning_scale), [`use_qk_norm`](../../layers/jax/attention/llama4_attention.md#Llama4VisionAttention.use_qk_norm)
- used by: [`layers`](llama4.md#JAXLlama4VisionEncoder.layers)

### `JAXLlama4VisionMLP`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/llama4.py:809`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L809)
- signature: `class JAXLlama4VisionMLP(nnx.Module):`
- members:
  - `fc1` — [`L817`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L817)
  - `fc2` — [`L829`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L829)
- protocol/private: `__call__`[`L841`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L841), `__init__`[`L811`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L811)
- used by: [`mlp`](llama4.md#JAXLlama4VisionEncoderLayer.mlp)

### `JAXLlama4VisionMLP2`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/llama4.py:1020`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1020)
- signature: `class JAXLlama4VisionMLP2(nnx.Module):`
- members:
  - `dropout` — [`L1051`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1051)
  - `dropout_rate` — [`L1048`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1048)
  - `fc1` — [`L1027`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1027)
  - `fc2` — [`L1037`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1037)
- protocol/private: `__call__`[`L1055`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1055), `__init__`[`L1022`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1022)
- used by: [`pixel_shuffle_mlp`](llama4.md#JAXLlama4VisionPixelShuffleMLP.pixel_shuffle_mlp)

### `JAXLlama4VisionModel`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/llama4.py:1098`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1098)
- signature: `class JAXLlama4VisionModel(nnx.Module):`
- members:
  - `class_embedding` — [`L1126`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1126)
  - `hidden_size` — [`L1111`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1111)
  - `image_size` — [`L1109`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1109)
  - `layernorm_post` — [`L1143`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1143)
  - `layernorm_pre` — [`L1135`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1135)
  - `model` — [`L1152`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1152)
  - `norm_eps` — [`L1112`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1112)
  - `num_channels` — [`L1113`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1113)
  - `num_patches` — [`L1116`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1116)
  - `patch_embedding` — [`L1118`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1118)
  - `patch_size` — [`L1110`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1110)
  - `positional_embedding_vlm` — [`L1130`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1130)
  - `scale` — [`L1108`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1108)
  - `vision_adapter` — [`L1160`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1160)
  - `vision_rope` — [`L1158`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1158)
- protocol/private: `__call__`[`L1163`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1163), `__init__`[`L1100`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1100)
- uses (calls/refs, reference-scoped): [`Llama4VisionRotaryEmbedding`](../../layers/jax/rope.md#Llama4VisionRotaryEmbedding), [`JAXLlama4VisionEncoder`](llama4.md#JAXLlama4VisionEncoder), [`JAXLlama4VisionPixelShuffleMLP`](llama4.md#JAXLlama4VisionPixelShuffleMLP), [`JAXUnfoldConvolution`](llama4.md#JAXUnfoldConvolution)
- used by: [`vision_model`](llama_guard_4.md#LlamaGuard4ForCausalLM.vision_model)

### `JAXLlama4VisionPixelShuffleMLP`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/llama4.py:1072`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1072)
- signature: `class JAXLlama4VisionPixelShuffleMLP(nnx.Module):`
- members:
  - `pixel_shuffle_mlp` — [`L1081`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1081)
  - `pixel_shuffle_ratio` — [`L1080`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1080)
- protocol/private: `__call__`[`L1085`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1085), `__init__`[`L1074`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L1074)
- uses (calls/refs, reference-scoped): [`JAXLlama4VisionMLP2`](llama4.md#JAXLlama4VisionMLP2), [`jax_pixel_shuffle`](llama4.md#jax_pixel_shuffle)
- used by: [`vision_adapter`](llama4.md#JAXLlama4VisionModel.vision_adapter)

### `JAXUnfoldConvolution`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/llama4.py:748`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L748)
- doc: A module that performs an "unfold" convolution operation,
- signature: `class JAXUnfoldConvolution(nnx.Module):`
- members:
  - `kernel_size` — [`L768`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L768)
  - `linear` — [`L772`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L772)
  - `num_channels` — [`L769`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L769)
- protocol/private: `__call__`[`L783`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L783), `__init__`[`L762`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L762)
- used by: [`patch_embedding`](llama4.md#JAXLlama4VisionModel.patch_embedding)

### `Llama4ForCausalLM`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/llama4.py:417`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L417)
- signature: `class Llama4ForCausalLM(nnx.Module):`
- members:
  - `compute_logits(self, hidden_states: jax.Array)` — [`L739`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L739)
  - `load_weights(self, rng: jax.Array, cache_dir: Optional[str] = None)` — [`L685`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L685)
  - `WeightLoader` — [`L418`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L418)
  - `embedder` — [`L493`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L493)
  - `enable_return_routed_experts` — [`L435`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L435)
  - `end_layer` — [`L504`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L504)
  - `final_norm` — [`L644`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L644)
  - `head_dim` — [`L471`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L471)
  - `hidden_act` — [`L458`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L458)
  - `hidden_size` — [`L443`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L443)
  - `interleave_moe_layer_step` — [`L463`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L463)
  - `intermediate_size_mlp` — [`L451`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L451)
  - `intermediate_size_moe` — [`L449`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L449)
  - `is_first_rank` — [`L488`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L488)
  - `is_last_rank` — [`L489`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L489)
  - `is_verbose` — [`L433`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L433)
  - `layers` — [`L642`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L642)
  - `lm_head` — [`L653`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L653)
  - `mesh` — [`L432`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L432)
  - `no_rope_layer_interval` — [`L459`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L459)
  - `num_attention_heads` — [`L467`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L467)
  - `num_key_value_heads` — [`L469`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L469)
  - `num_layers` — [`L447`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L447)
  - `num_local_experts` — [`L456`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L456)
  - `num_shared_experts` — [`L473`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L473)
  - `rms_norm_eps` — [`L475`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L475)
  - `rng` — [`L431`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L431)
  - `rope_scaling` — [`L477`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L477)
  - `start_layer` — [`L504`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L504)
  - `use_qk_norm` — [`L486`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L486)
  - `vllm_config` — [`L427`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L427)
  - `vocab_size` — [`L442`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L442)
- protocol/private: `__call__`[`L698`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L698), `__init__`[`L420`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L420), `_print_model_architecture`[`L669`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L669)
- uses (calls/refs, reference-scoped): [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`load_weights`](llama4.md#Llama4WeightLoader.load_weights), [`JaxIntermediateTensors`](jax_intermediate_tensor.md#JaxIntermediateTensors), [`PPMissingLayer`](../../layers/jax/pp_utils.md#PPMissingLayer), [`get_pp_group`](../../distributed/jax_parallel_state.md#get_pp_group), [`JaxMoE`](../../layers/jax/moe/moe.md#JaxMoE), [`encode`](../../layers/jax/layers.md#Embedder.encode), [`is_first_rank`](../../distributed/jax_parallel_state.md#GroupCoordinator.is_first_rank), [`logger`](llama4.md#logger), [`is_last_rank`](../../distributed/jax_parallel_state.md#GroupCoordinator.is_last_rank), [`RMSNorm`](../../layers/jax/layers.md#RMSNorm), [`dtype`](../../layers/jax/layers.md#RMSNorm.dtype), [`activation_ffw_td`](../../layers/jax/layers.md#RMSNorm.activation_ffw_td), [`dtype`](../../layers/jax/layers.md#Embedder.dtype), [`dims`](../../layers/jax/layers.md#RMSNorm.dims), [`epsilon`](../../layers/jax/layers.md#RMSNorm.epsilon), [`random_init`](../../layers/jax/layers.md#RMSNorm.random_init), [`Embedder`](../../layers/jax/layers.md#Embedder), [`keyvalue_skh`](../../layers/jax/attention/attention.md#Attention.keyvalue_skh), [`DenseFFW`](../../layers/jax/layers.md#DenseFFW), [`dtype`](../../layers/jax/attention/attention.md#Attention.dtype), [`dtype`](../../layers/jax/moe/moe.md#Router.dtype), [`edf_sharding`](../../layers/jax/moe/moe.md#JaxMoE.edf_sharding), [`hidden_size`](../../layers/jax/layers.md#Embedder.hidden_size), [`rngs`](../../layers/jax/layers.md#RMSNorm.rngs), [`LMhead`](../../layers/jax/layers.md#LMhead), [`quant_config`](../../layers/jax/moe/moe.md#JaxMoE.quant_config), [`random_init`](../../layers/jax/layers.md#Embedder.random_init), [`vocab_size`](../../layers/jax/layers.md#Embedder.vocab_size), [`query_tnh`](../../layers/jax/attention/attention.md#Attention.query_tnh), [`activation_ffw_td`](../../layers/jax/moe/moe.md#JaxMoE.activation_ffw_td), [`dtype`](../../layers/jax/layers.md#DenseFFW.dtype), [`dtype`](../../layers/jax/moe/moe.md#JaxMoE.dtype), [`efd_sharding`](../../layers/jax/moe/moe.md#JaxMoE.efd_sharding), [`mesh`](../../layers/jax/attention/attention.md#Attention.mesh), [`Llama4Attention`](../../layers/jax/attention/llama4_attention.md#Llama4Attention), [`Router`](../../layers/jax/moe/moe.md#Router), [`activation_q_td`](../../layers/jax/attention/attention.md#Attention.activation_q_td), [`KVCacheType`](../../layers/jax/constants.md#KVCacheType.KVCacheType), [`mesh`](../../layers/jax/moe/moe.md#JaxMoE.mesh)  (+69 more)
- used by: [`_get_model_architecture`](../common/model_loader.md#_get_model_architecture)

### `Llama4WeightLoader`  ·  implements/extends BaseWeightLoader
- def: [`tpu_inference/models/jax/llama4.py:52`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L52)
- signature: `class Llama4WeightLoader(BaseWeightLoader):`
- members:
  - `_get_expert_num(self, loaded_key: str)` — [`L253`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L253) — Extracts the expect number from a HuggingFace weight key string.
  - `_get_layer_num(self, loaded_key: str)` — [`L243`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L243) — Extracts the layer number from a HuggingFace weight key string.
  - `_map_llama4_gate_up_proj(self, model_for_loading: nnx.Module, model_params: nnx.State, loaded_name: str, loaded_weight: jax.Array)` — [`L194`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L194) — HF's gate_up_proj is a fused tensor of gate and up projections. It needs to be split.
  - `load_weights(self, model_for_loading: nnx.Module)` — [`L263`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L263)
  - `map_loaded_to_standardized_name(self, loaded_key: str)` — [`L174`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L174)
  - `expert_prefix` — [`L68`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L68)
  - `expert_weights_buffer` — [`L67`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L67)
  - `interleave_moe_layer_step` — [`L61`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L61)
  - `is_verbose` — [`L59`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L59)
  - `pp_missing_layers` — [`L172`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L172)
  - `quantization_config` — [`L65`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L65)
- protocol/private: `__init__`[`L54`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L54), `_loaded_to_standardized_keys`[`L131`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L131), `_transpose_map`[`L76`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L76), `_weight_shape_map`[`L94`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L94)
- uses (calls/refs, reference-scoped): [`PPMissingLayer`](../../layers/jax/pp_utils.md#PPMissingLayer), [`cpu_mesh_context`](../../layers/common/utils.md#cpu_mesh_context), [`BaseWeightLoader`](utils/weight_utils.md#BaseWeightLoader), [`get_param`](utils/weight_utils.md#get_param), [`logger`](llama4.md#logger), [`print_param_info`](utils/weight_utils.md#print_param_info), [`convert_torch_to_jax_with_view`](utils/weight_utils.md#convert_torch_to_jax_with_view), [`_is_pp_missing_layer`](utils/weight_utils.md#_is_pp_missing_layer), [`get_weights_iterator`](utils/weight_utils.md#BaseWeightLoader.get_weights_iterator), [`shard_put`](../../layers/jax/misc.md#shard_put), [`transpose_params`](utils/weight_utils.md#transpose_params), [`__init__`](utils/weight_utils.md#BaseWeightLoader.__init__), [`reshape_params`](utils/weight_utils.md#reshape_params)
- used by: [`load_weights`](llama4.md#Llama4ForCausalLM.load_weights), [`BaseWeightLoader`](utils/weight_utils.md#BaseWeightLoader), [`WeightLoader`](llama4.md#Llama4ForCausalLM.WeightLoader)

## Functions
- `jax_pixel_shuffle(input_tensor: jax.Array, shuffle_ratio: float)` — [`L976`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L976) — Rearranges elements in a tensor of shape [B, L, C] according to a shuffle ratio.

## Module values
- `logger` — [`L49`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama4.py#L49)

