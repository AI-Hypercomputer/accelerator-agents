---
title: 'Module: easydel/modules/llama4/llama4_configuration.py'
type: catalog
provenance: extracted
module: easydel/modules/llama4/llama4_configuration.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.modules.llama4.llama4_configuration`/
symbols:
  Llama4TextConfig: Llama4TextConfig#
  Llama4VisionConfig: Llama4VisionConfig#
  Llama4Config.text_config: Llama4Config#text_config.
  Llama4Config.vision_config: Llama4Config#vision_config.
  Llama4Config: Llama4Config#
  Llama4VisionConfig.hidden_size: Llama4VisionConfig#hidden_size.
  Llama4VisionConfig.get_partition_rules: Llama4VisionConfig#get_partition_rules.
  Llama4TextConfig.head_dim: Llama4TextConfig#head_dim.
  Llama4TextConfig.get_partition_rules: Llama4TextConfig#get_partition_rules.
  Llama4Config.sub_configs: Llama4Config#sub_configs.
  Llama4Config.__init__: Llama4Config#__init__().
  _patch_hf_llama4_pooler_output: _patch_hf_llama4_pooler_output().
  _get_partition_rules: _get_partition_rules().
  Llama4VisionConfig.patch_size: Llama4VisionConfig#patch_size.
  Llama4TextConfig.hidden_size: Llama4TextConfig#hidden_size.
  Llama4VisionConfig.rope_theta: Llama4VisionConfig#rope_theta.
  Llama4TextConfig.__init__: Llama4TextConfig#__init__().
  Llama4TextConfig.router_jitter_noise: Llama4TextConfig#router_jitter_noise.
  Llama4TextConfig.attention_chunk_size: Llama4TextConfig#attention_chunk_size.
  Llama4Config.get_partition_rules: Llama4Config#get_partition_rules.
  logger: logger.
  _patch_hf_llama4_pooler_output._patched_get_image_features: _patch_hf_llama4_pooler_output()._patched_get_image_features().
  Llama4VisionConfig.num_channels: Llama4VisionConfig#num_channels.
  Llama4VisionConfig.num_attention_heads: Llama4VisionConfig#num_attention_heads.
  Llama4TextConfig.rms_norm_eps: Llama4TextConfig#rms_norm_eps.
  Llama4VisionConfig.num_hidden_layers: Llama4VisionConfig#num_hidden_layers.
  Llama4VisionConfig.image_size: Llama4VisionConfig#image_size.
  Llama4VisionConfig.initializer_range: Llama4VisionConfig#initializer_range.
  Llama4VisionConfig.attention_dropout: Llama4VisionConfig#attention_dropout.
  Llama4TextConfig.attn_temperature_tuning: Llama4TextConfig#attn_temperature_tuning.
  Llama4TextConfig.attn_scale: Llama4TextConfig#attn_scale.
  Llama4TextConfig.floor_scale: Llama4TextConfig#floor_scale.
  Llama4TextConfig.intermediate_size_mlp: Llama4TextConfig#intermediate_size_mlp.
  Llama4TextConfig.num_hidden_layers: Llama4TextConfig#num_hidden_layers.
  Llama4TextConfig.num_attention_heads: Llama4TextConfig#num_attention_heads.
  Llama4TextConfig.use_qk_norm: Llama4TextConfig#use_qk_norm.
  Llama4TextConfig.no_rope_layers: Llama4TextConfig#no_rope_layers.
  Llama4TextConfig.moe_layers: Llama4TextConfig#moe_layers.
  Llama4VisionConfig.model_type: Llama4VisionConfig#model_type.
  Llama4VisionConfig.base_config_key: Llama4VisionConfig#base_config_key.
  Llama4VisionConfig.__init__: Llama4VisionConfig#__init__().
  Llama4VisionConfig.hidden_act: Llama4VisionConfig#hidden_act.
  Llama4VisionConfig.intermediate_size: Llama4VisionConfig#intermediate_size.
  Llama4VisionConfig.vision_output_dim: Llama4VisionConfig#vision_output_dim.
  Llama4VisionConfig.norm_eps: Llama4VisionConfig#norm_eps.
  Llama4VisionConfig.pixel_shuffle_ratio: Llama4VisionConfig#pixel_shuffle_ratio.
  Llama4VisionConfig.projector_input_dim: Llama4VisionConfig#projector_input_dim.
  Llama4VisionConfig.projector_output_dim: Llama4VisionConfig#projector_output_dim.
  Llama4VisionConfig.multi_modal_projector_bias: Llama4VisionConfig#multi_modal_projector_bias.
  Llama4VisionConfig.projector_dropout: Llama4VisionConfig#projector_dropout.
  Llama4VisionConfig.vision_feature_layer: Llama4VisionConfig#vision_feature_layer.
  Llama4VisionConfig.vision_feature_select_strategy: Llama4VisionConfig#vision_feature_select_strategy.
  Llama4TextConfig.model_type: Llama4TextConfig#model_type.
  Llama4TextConfig.vocab_size: Llama4TextConfig#vocab_size.
  Llama4TextConfig.max_position_embeddings: Llama4TextConfig#max_position_embeddings.
  Llama4TextConfig.intermediate_size: Llama4TextConfig#intermediate_size.
  Llama4TextConfig.rope_scaling: Llama4TextConfig#rope_scaling.
  Llama4TextConfig.attention_bias: Llama4TextConfig#attention_bias.
  Llama4TextConfig.num_key_value_heads: Llama4TextConfig#num_key_value_heads.
  Llama4TextConfig.hidden_act: Llama4TextConfig#hidden_act.
  Llama4TextConfig.initializer_range: Llama4TextConfig#initializer_range.
  Llama4TextConfig.use_cache: Llama4TextConfig#use_cache.
  Llama4TextConfig.rope_theta: Llama4TextConfig#rope_theta.
  Llama4TextConfig.attention_dropout: Llama4TextConfig#attention_dropout.
  Llama4TextConfig.num_experts_per_tok: Llama4TextConfig#num_experts_per_tok.
  Llama4TextConfig.num_local_experts: Llama4TextConfig#num_local_experts.
  Llama4TextConfig.output_router_logits: Llama4TextConfig#output_router_logits.
  Llama4TextConfig.router_aux_loss_coef: Llama4TextConfig#router_aux_loss_coef.
  Llama4TextConfig.interleave_moe_layer_step: Llama4TextConfig#interleave_moe_layer_step.
  Llama4TextConfig.layer_types: Llama4TextConfig#layer_types.
  Llama4Config.model_type: Llama4Config#model_type.
  Llama4Config.attribute_map: Llama4Config#attribute_map.
  Llama4Config.boi_token_index: Llama4Config#boi_token_index.
  Llama4Config.eoi_token_index: Llama4Config#eoi_token_index.
  Llama4Config.image_token_index: Llama4Config#image_token_index.
  __all__: __all__.
---
# Module: [`easydel/modules/llama4/llama4_configuration.py`](../../../../../../../raw/code/EasyDeL/easydel/modules/llama4/llama4_configuration.py)

> **Collapsed catalog** (76 symbols) — anchors above resolve for citations; detailed member listing omitted (`coverage_collapse`). See the source link above, or the curated codebase page, for depth.
