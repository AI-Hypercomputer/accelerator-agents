---
title: 'Module: axlearn/common/attention.py'
type: catalog
provenance: extracted
module: axlearn/common/attention.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.attention`/
symbols:
  ForwardMode: ForwardMode#
  MultiheadAttention._forward_for_mode: MultiheadAttention#_forward_for_mode().
  MultiheadAttention._mask_tpl: MultiheadAttention#_mask_tpl.
  MultiheadAttention: MultiheadAttention#
  _TransformerRepeat._forward_for_mode: _TransformerRepeat#_forward_for_mode().
  BaseTransformerLayer: BaseTransformerLayer#
  TransformerLayer: TransformerLayer#
  QKVLinear.__init__: QKVLinear#__init__().
  BaseQKVLinear: BaseQKVLinear#
  StackedTransformerLayer: StackedTransformerLayer#
  TransformerFeedForwardLayer.__init__: TransformerFeedForwardLayer#__init__().
  TransformerAttentionLayer._forward_for_mode: TransformerAttentionLayer#_forward_for_mode().
  TransformerLayer._forward_for_mode: TransformerLayer#_forward_for_mode().
  MultiheadAttention._compute_attention: MultiheadAttention#_compute_attention().
  RoFormerQKVLinear.__init__: RoFormerQKVLinear#__init__().
  BaseTransformerLayer.Output: BaseTransformerLayer#Output#
  build_remat_spec: build_remat_spec().
  FusedGroupedQKVLinear.__init__: FusedGroupedQKVLinear#__init__().
  MultiheadAttentionXL.__init__: MultiheadAttentionXL#__init__().
  StackedTransformerLayer.initialize_parameters_recursively: StackedTransformerLayer#initialize_parameters_recursively().
  FusedQKVLinear.__init__: FusedQKVLinear#__init__().
  FusedQKVLinear.create_parameter_specs_recursively: FusedQKVLinear#create_parameter_specs_recursively().
  _TransformerPipeline.forward: _TransformerPipeline#forward().
  YaRNSinusoidalPositionalEmbedding.compute_rope_params: YaRNSinusoidalPositionalEmbedding#compute_rope_params().
  MultiheadAttention._create_layer_parameter_specs: MultiheadAttention#_create_layer_parameter_specs().
  MultiheadAttention.init_states: MultiheadAttention#init_states().
  TransformerAttentionLayer.__init__: TransformerAttentionLayer#__init__().
  PipelinedTransformerLayer.__init__: PipelinedTransformerLayer#__init__().
  StackedTransformerLayer._forward_for_mode: StackedTransformerLayer#_forward_for_mode().
  RoFormerQKVLinear: RoFormerQKVLinear#
  ForwardMode.PREFILL: ForwardMode#PREFILL.
  QLinear.__init__: QLinear#__init__().
  SigmoidAttention._compute_attention: SigmoidAttention#_compute_attention().
  TransformerFeedForwardLayer.forward: TransformerFeedForwardLayer#forward().
  QKVLinear: QKVLinear#
  PerDimScale: PerDimScale#
  MultiheadAttention.extend_step: MultiheadAttention#extend_step().
  PipelinedTransformerLayer.initialize_parameters_recursively: PipelinedTransformerLayer#initialize_parameters_recursively().
  QLinear.forward: QLinear#forward().
  GroupedQueryAttention: GroupedQueryAttention#
  FusedQKVLinear: FusedQKVLinear#
  TransformerAttentionLayer: TransformerAttentionLayer#
  set_attention_partition_specs: set_attention_partition_specs().
  set_double_shard_weights_config: set_double_shard_weights_config().
  ForwardMode.EXTEND_STEP: ForwardMode#EXTEND_STEP.
  BaseTransformerLayer.Config.input_dim: BaseTransformerLayer#Config#input_dim.
  TransformerLayer.__init__: TransformerLayer#__init__().
  ParallelTransformerLayer.__init__: ParallelTransformerLayer#__init__().
  BottleNeckAdapterTransformerLayer.__init__: BottleNeckAdapterTransformerLayer#__init__().
  RepeatedTransformerLayer.__init__: RepeatedTransformerLayer#__init__().
  PerDimScale.default_config: PerDimScale#default_config().
  MultiheadAttention.per_head_dim: MultiheadAttention#per_head_dim().
  ScaleQuery._scale_factor: ScaleQuery#_scale_factor.
  BaseStackedTransformerLayer: BaseStackedTransformerLayer#
  StackedTransformerLayer._layers: StackedTransformerLayer#_layers.
  QLinear: QLinear#
  BaseTransformerLayer.forward: BaseTransformerLayer#forward().
  BaseTransformerLayer.extend_step: BaseTransformerLayer#extend_step().
  FusedGroupedQKVLinear.forward: FusedGroupedQKVLinear#forward().
  TransformerAttentionLayer.extend_step: TransformerAttentionLayer#extend_step().
  StackedTransformerLayer.extend_step: StackedTransformerLayer#extend_step().
  RepeatedTransformerLayer: RepeatedTransformerLayer#
  ForwardMode.FORWARD: ForwardMode#FORWARD.
  MultiheadAttentionXL.default_config: MultiheadAttentionXL#default_config().
  BottleNeckAdapterTransformerLayer._forward_for_mode: BottleNeckAdapterTransformerLayer#_forward_for_mode().
  ScaleKey._scale_factor: ScaleKey#_scale_factor.
  MultiheadAttention.Config: MultiheadAttention#Config#
  MultiheadAttention.Config.input_linear: MultiheadAttention#Config#input_linear.
  BaseTransformerLayer.init_states: BaseTransformerLayer#init_states().
  SinusoidalPositionalEmbedding.forward: SinusoidalPositionalEmbedding#forward().
  BaseQKVLinear.forward: BaseQKVLinear#forward().
  FusedQKVLinear.initialize_parameters_recursively: FusedQKVLinear#initialize_parameters_recursively().
  FusedQKVLinear.forward: FusedQKVLinear#forward().
  MultiheadAttentionXL._create_layer_parameter_specs: MultiheadAttentionXL#_create_layer_parameter_specs().
  MultiheadAttentionXL._compute_logits: MultiheadAttentionXL#_compute_logits().
  TransformerFeedForwardLayer._linear1_activation: TransformerFeedForwardLayer#_linear1_activation().
  TransformerLayer.extend_step: TransformerLayer#extend_step().
  BottleNeckAdapterTransformerLayer.extend_step: BottleNeckAdapterTransformerLayer#extend_step().
  BaseStackedTransformerLayer.Config.layer: BaseStackedTransformerLayer#Config#layer.
  _TransformerRepeat.extend_step: _TransformerRepeat#extend_step().
  PerDimScale.Config: PerDimScale#Config#
  MultiheadAttention.forward: MultiheadAttention#forward().
  MultiheadAttention.Config.kv_cache: MultiheadAttention#Config#kv_cache.
  NormPosition: NormPosition#
  TransformerAttentionLayer.attention_thunk: TransformerAttentionLayer#attention_thunk().
  GroupedQKVLinear: GroupedQKVLinear#
  FusedGroupedQKVLinear: FusedGroupedQKVLinear#
  TransformerFeedForwardLayer: TransformerFeedForwardLayer#
  BaseTransformerLayer.Config: BaseTransformerLayer#Config#
  BaseMultiheadLinear._create_layer_parameter_specs: BaseMultiheadLinear#_create_layer_parameter_specs().
  BaseQKVLinear.Config: BaseQKVLinear#Config#
  ScaleQuery: ScaleQuery#
  MultiheadAttentionXL: MultiheadAttentionXL#
  TransformerAttentionLayer._create_layer_parameter_specs: TransformerAttentionLayer#_create_layer_parameter_specs().
  TransformerFeedForwardLayer._create_layer_parameter_specs: TransformerFeedForwardLayer#_create_layer_parameter_specs().
  TransformerLayer.Config.self_attention: TransformerLayer#Config#self_attention.
  TransformerLayer.forward: TransformerLayer#forward().
  ParallelTransformerLayer.forward: ParallelTransformerLayer#forward().
  MultiheadAttention.__init__: MultiheadAttention#__init__().
  TransformerAttentionLayer.Config.norm: TransformerAttentionLayer#Config#norm.
  TransformerFeedForwardLayer.Config.norm: TransformerFeedForwardLayer#Config#norm.
  BaseStackedTransformerLayer.Config.num_layers: BaseStackedTransformerLayer#Config#num_layers.
  TransformerLayer.Config: TransformerLayer#Config#
  BaseQKVLinear.Config.query_dim: BaseQKVLinear#Config#query_dim.
  BaseQKVLinear.Config.num_heads: BaseQKVLinear#Config#num_heads.
  BaseQKVLinear.Config.per_head_dim: BaseQKVLinear#Config#per_head_dim.
  BaseQKVLinear.Output: BaseQKVLinear#Output#
  RoFormerSinusoidalPositionalEmbedding: RoFormerSinusoidalPositionalEmbedding#
  StackedTransformerLayer._aggregate_layer_outputs: StackedTransformerLayer#_aggregate_layer_outputs().
  BaseTransformerLayer.Output.data: BaseTransformerLayer#Output#data.
  RoFormerSinusoidalPositionalEmbedding.forward: RoFormerSinusoidalPositionalEmbedding#forward().
  BaseScaleQK: BaseScaleQK#
  TransformerLayer.Config.feed_forward: TransformerLayer#Config#feed_forward.
  BaseStackedTransformerLayer.Config: BaseStackedTransformerLayer#Config#
  CausalAttentionLogitBiasLayer.forward: CausalAttentionLogitBiasLayer#forward().
  LearnedPositionalEmbedding._compute_fan_axes: LearnedPositionalEmbedding#_compute_fan_axes().
  BaseMultiheadLinear: BaseMultiheadLinear#
  QKVLinear.forward: QKVLinear#forward().
  GroupedQKVLinear.__init__: GroupedQKVLinear#__init__().
  RoFormerQKVLinear.forward: RoFormerQKVLinear#forward().
  MultiheadAttentionXL._scale_qk: MultiheadAttentionXL#_scale_qk().
  TransformerAttentionLayer.forward: TransformerAttentionLayer#forward().
  BottleNeckAdapterTransformerLayer.forward: BottleNeckAdapterTransformerLayer#forward().
  StackedTransformerLayer.forward: StackedTransformerLayer#forward().
  _TransformerRepeat.forward: _TransformerRepeat#forward().
  MultiheadInputLinear: MultiheadInputLinear#
  apply_attention_logit_biases: apply_attention_logit_biases().
  ALiBiAttentionLogitBiasLayer.forward: ALiBiAttentionLogitBiasLayer#forward().
  MultiheadAttention.Config.num_heads: MultiheadAttention#Config#num_heads.
  BaseScaleQK.Config: BaseScaleQK#Config#
  RoFormerQKVLinear.Config: RoFormerQKVLinear#Config#
  TransformerFeedForwardLayer.Config: TransformerFeedForwardLayer#Config#
  BaseMultiheadLinear.default_config: BaseMultiheadLinear#default_config().
  BaseQKVLinear.Config.key_dim: BaseQKVLinear#Config#key_dim.
  BaseQKVLinear.Config.value_dim: BaseQKVLinear#Config#value_dim.
  BaseQKVLinear.num_kv_heads: BaseQKVLinear#num_kv_heads().
  constant_scale_fn: constant_scale_fn().
  MultiheadAttention.Config.query_dim: MultiheadAttention#Config#query_dim.
  TransformerFeedForwardLayer.Config.input_dim: TransformerFeedForwardLayer#Config#input_dim.
  BottleNeckAdapterTransformerLayer.Config.layer: BottleNeckAdapterTransformerLayer#Config#layer.
  StackedTransformerLayer.Config.layer: StackedTransformerLayer#Config#layer.
  BaseMultiheadLinear._compute_fan_axes: BaseMultiheadLinear#_compute_fan_axes().
  MultiheadInputLinear.Config: MultiheadInputLinear#Config#
  softmax_with_biases: softmax_with_biases().
  TransformerAttentionLayer.Config: TransformerAttentionLayer#Config#
  TransformerAttentionLayer.Config.attention: TransformerAttentionLayer#Config#attention.
  scaled_hidden_dim: scaled_hidden_dim().
  _TransformerRepeat.init_states: _TransformerRepeat#init_states().
  RepeatedTransformerLayer.extend_step: RepeatedTransformerLayer#extend_step().
  PipelinedTransformerLayer: PipelinedTransformerLayer#
  PipelinedTransformerLayer.Config: PipelinedTransformerLayer#Config#
  SymmetricALiBiAttentionLogitBiasLayer.forward: SymmetricALiBiAttentionLogitBiasLayer#forward().
  LearnedPositionalEmbedding.default_config: LearnedPositionalEmbedding#default_config().
  ScaleFn: ScaleFn.
  RoFormerQKVLinear.Config.input_linear: RoFormerQKVLinear#Config#input_linear.
  MultiheadAttention.Config.query_scale: MultiheadAttention#Config#query_scale.
  MultiheadAttention.Config.key_scale: MultiheadAttention#Config#key_scale.
  TransformerFeedForwardLayer._get_activation: TransformerFeedForwardLayer#_get_activation().
  set_feed_forward_partition_specs: set_feed_forward_partition_specs().
  StackedTransformerLayer._update_data: StackedTransformerLayer#_update_data.
  StackedTransformerLayer.init_states: StackedTransformerLayer#init_states().
  _TransformerRepeat.layer_fn: _TransformerRepeat#layer_fn().
  PipelinedTransformerLayer.Config.num_stages: PipelinedTransformerLayer#Config#num_stages.
  PipelinedTransformerLayer.Config.stage: PipelinedTransformerLayer#Config#stage.
  BaseTransformerLayer.Output.self_attention_kv_state: BaseTransformerLayer#Output#self_attention_kv_state.
  BaseMultiheadLinear.Config: BaseMultiheadLinear#Config#
  MultiheadAttention.Config.dropout: MultiheadAttention#Config#dropout.
  TransformerFeedForwardLayer.Config.linear1: TransformerFeedForwardLayer#Config#linear1.
  AttentionLogitBiasLayer: AttentionLogitBiasLayer#
  BaseMultiheadLinear.Config.model_dim: BaseMultiheadLinear#Config#model_dim.
  BaseMultiheadLinear.Config.num_heads: BaseMultiheadLinear#Config#num_heads.
  BaseMultiheadLinear.Config.per_head_dim: BaseMultiheadLinear#Config#per_head_dim.
  YaRNSinusoidalPositionalEmbedding.forward: YaRNSinusoidalPositionalEmbedding#forward().
  apply_rotary_position_embeddings: apply_rotary_position_embeddings().
  ScaleQuery.default_scale_factor_config: ScaleQuery#default_scale_factor_config().
  ScaleKey.default_scale_factor_config: ScaleKey#default_scale_factor_config().
  MultiheadAttentionXL.Config.scale_position: MultiheadAttentionXL#Config#scale_position.
  TransformerAttentionLayer.Config.target_dim: TransformerAttentionLayer#Config#target_dim.
  update_data_with_skip_connection.update_data: update_data_with_skip_connection().update_data().
  StackedTransformerLayer._update_layer_kwargs: StackedTransformerLayer#_update_layer_kwargs().
  RepeatedTransformerLayer.Config.repeat: RepeatedTransformerLayer#Config#repeat.
  QKVLinear.Config: QKVLinear#Config#
  MultiheadAttention.Config.output_linear: MultiheadAttention#Config#output_linear.
  MultiheadAttention._compute_logits: MultiheadAttention#_compute_logits().
  TransformerFeedForwardLayer.Config.hidden_dim: TransformerFeedForwardLayer#Config#hidden_dim.
  TransformerFeedForwardLayer.Config.linear2: TransformerFeedForwardLayer#Config#linear2.
  TransformerFeedForwardLayer.Config.dropout: TransformerFeedForwardLayer#Config#dropout.
  LearnedPositionalEmbedding._create_layer_parameter_specs: LearnedPositionalEmbedding#_create_layer_parameter_specs().
  BaseMultiheadLinear.forward: BaseMultiheadLinear#forward().
  MultiheadInputLinear._bias_spec: MultiheadInputLinear#_bias_spec().
  MultiheadInputLinear._compute_fan_axes: MultiheadInputLinear#_compute_fan_axes().
  MultiheadOutputLinear._bias_spec: MultiheadOutputLinear#_bias_spec().
  MultiheadOutputLinear._compute_fan_axes: MultiheadOutputLinear#_compute_fan_axes().
  PerDimScale._create_layer_parameter_specs: PerDimScale#_create_layer_parameter_specs().
  ScaleQuery.__init__: ScaleQuery#__init__().
  ScaleQuery.forward: ScaleQuery#forward().
  ScaleKey.__init__: ScaleKey#__init__().
  MultiheadAttention.default_query_scale_config: MultiheadAttention#default_query_scale_config().
  MultiheadAttention.default_key_scale_config: MultiheadAttention#default_key_scale_config().
  GroupedQueryAttention._compute_logits: GroupedQueryAttention#_compute_logits().
  GroupedQueryAttention._compute_context: GroupedQueryAttention#_compute_context().
  MultiheadRelativePositionLinear._bias_spec: MultiheadRelativePositionLinear#_bias_spec().
  MultiheadRelativePositionLinear._compute_fan_axes: MultiheadRelativePositionLinear#_compute_fan_axes().
  MultiheadAttentionXL.ScalePosition: MultiheadAttentionXL#ScalePosition#
  MultiheadAttentionXL.forward: MultiheadAttentionXL#forward().
  MultiheadAttentionXL.extend_step: MultiheadAttentionXL#extend_step().
  StackedTransformerLayer.__init__: StackedTransformerLayer#__init__().
  RepeatedTransformerLayer.init_states: RepeatedTransformerLayer#init_states().
  alibi_get_slopes: alibi_get_slopes().
  SinusoidalPositionalEmbedding.Config: SinusoidalPositionalEmbedding#Config#
  MultiheadOutputLinear.Config: MultiheadOutputLinear#Config#
  GroupedQKVLinear.Config: GroupedQKVLinear#Config#
  QLinear.Config: QLinear#Config#
  FusedQKVLinear.Config: FusedQKVLinear#Config#
  FusedGroupedQKVLinear.Config: FusedGroupedQKVLinear#Config#
  RoFormerSinusoidalPositionalEmbedding.Config: RoFormerSinusoidalPositionalEmbedding#Config#
  TransformerAttentionLayer.init_states: TransformerAttentionLayer#init_states().
  TransformerLayer.Config.cross_attention: TransformerLayer#Config#cross_attention.
  StackedTransformerLayer.Config: StackedTransformerLayer#Config#
  RepeatedTransformerLayer.Config: RepeatedTransformerLayer#Config#
  PipelinedTransformerLayer.Config.pipeline: PipelinedTransformerLayer#Config#pipeline.
  FullAttentionLogitBiasLayer.forward: FullAttentionLogitBiasLayer#forward().
  MultiheadAttention.Config.mask: MultiheadAttention#Config#mask.
  MultiheadAttention.Output: MultiheadAttention#Output#
  MultiheadAttentionXL.Config: MultiheadAttentionXL#Config#
  CausalAttentionLogitBiasLayer: CausalAttentionLogitBiasLayer#
  BaseTransformerLayer.Output.self_attention_probs: BaseTransformerLayer#Output#self_attention_probs.
  BaseMultiheadLinear._einsum_expr: BaseMultiheadLinear#_einsum_expr().
  MultiheadOutputLinear: MultiheadOutputLinear#
  sigmoid_with_biases: sigmoid_with_biases().
  QKVLinear.Config.layer: QKVLinear#Config#layer.
  QLinear.Config.layer: QLinear#Config#layer.
  FusedQKVLinear.Config.layer: FusedQKVLinear#Config#layer.
  FusedGroupedQKVLinear.Config.layer: FusedGroupedQKVLinear#Config#layer.
  YaRNSinusoidalPositionalEmbedding.Config: YaRNSinusoidalPositionalEmbedding#Config#
  ScaleQuery.Config: ScaleQuery#Config#
  ScaleQuery.apply_scale_factor: ScaleQuery#apply_scale_factor().
  ScaleKey.Config: ScaleKey#Config#
  MultiheadAttentionXL.Config.relative_pos_emb: MultiheadAttentionXL#Config#relative_pos_emb.
  MultiheadAttentionXL.Config.relative_pos_linear: MultiheadAttentionXL#Config#relative_pos_linear.
  TransformerAttentionLayer.Config.dropout: TransformerAttentionLayer#Config#dropout.
  TransformerAttentionLayer.Config.stochastic_depth: TransformerAttentionLayer#Config#stochastic_depth.
  TransformerFeedForwardLayer.Config.stochastic_depth: TransformerFeedForwardLayer#Config#stochastic_depth.
  ParallelTransformerLayer.Config: ParallelTransformerLayer#Config#
  BottleNeckAdapterTransformerLayer.Config: BottleNeckAdapterTransformerLayer#Config#
  BottleNeckAdapterTransformerLayer.Config.adapter: BottleNeckAdapterTransformerLayer#Config#adapter.
  _TransformerRepeat.Config: _TransformerRepeat#Config#
  _TransformerPipeline.layer_fn: _TransformerPipeline#layer_fn().
  _save_and_offload_only_these_names_regex: _save_and_offload_only_these_names_regex().
  FusedGroupedQKVLinear.Config.num_kv_heads: FusedGroupedQKVLinear#Config#num_kv_heads.
  BaseScaleQK.Config.per_head_dim: BaseScaleQK#Config#per_head_dim.
  MultiheadAttention.Config.key_dim: MultiheadAttention#Config#key_dim.
  MultiheadAttention.Config.value_dim: MultiheadAttention#Config#value_dim.
  MultiheadAttention._scale_qk: MultiheadAttention#_scale_qk().
  MultiheadAttention._compute_context: MultiheadAttention#_compute_context().
  xl_attention_logits: xl_attention_logits().
  NormPosition.IN_NORM: NormPosition#IN_NORM.
  TransformerFeedForwardLayer.Config.structure: TransformerFeedForwardLayer#Config#structure.
  compute_padding_biases: compute_padding_biases().
  FullAttentionLogitBiasLayer: FullAttentionLogitBiasLayer#
  BaseTransformerLayer.Output.cross_attention_probs: BaseTransformerLayer#Output#cross_attention_probs.
  LearnedPositionalEmbedding.Config: LearnedPositionalEmbedding#Config#
  BaseQKVLinear.is_kv_sharing: BaseQKVLinear#is_kv_sharing().
  PerDimScale.forward: PerDimScale#forward().
  BaseScaleQK.forward: BaseScaleQK#forward().
  ScaleKey: ScaleKey#
  ScaleKey.forward: ScaleKey#forward().
  RoFormerQKVLinear.Config.rope_pos_emb_layer: RoFormerQKVLinear#Config#rope_pos_emb_layer.
  RoFormerQKVLinear.is_kv_sharing: RoFormerQKVLinear#is_kv_sharing().
  MultiheadAttention.Config.q_partition_spec: MultiheadAttention#Config#q_partition_spec.
  SigmoidAttention: SigmoidAttention#
  SigmoidAttention.Config: SigmoidAttention#Config#
  TransformerFeedForwardLayer._linear2: TransformerFeedForwardLayer#_linear2().
  TransformerLayer.init_states: TransformerLayer#init_states().
  ParallelTransformerLayer.Config.norm: ParallelTransformerLayer#Config#norm.
  ParallelTransformerLayer.Config.self_attention: ParallelTransformerLayer#Config#self_attention.
  ParallelTransformerLayer.Config.feed_forward: ParallelTransformerLayer#Config#feed_forward.
  BottleNeckAdapterTransformerLayer.init_states: BottleNeckAdapterTransformerLayer#init_states().
  UpdateDataFn.__call__: UpdateDataFn#__call__().
  RepeatedTransformerLayer.initialize_parameters_recursively: RepeatedTransformerLayer#initialize_parameters_recursively().
  RepeatedTransformerLayer.forward: RepeatedTransformerLayer#forward().
  PipelinedTransformerLayer.forward: PipelinedTransformerLayer#forward().
  RematRegexSavePatterns.FLASH_ATTENTION: RematRegexSavePatterns#FLASH_ATTENTION.
  AttentionLogitBiasLayer.forward: AttentionLogitBiasLayer#forward().
  ALiBiAttentionLogitBiasLayer.Config: ALiBiAttentionLogitBiasLayer#Config#
  SymmetricALiBiAttentionLogitBiasLayer.Config: SymmetricALiBiAttentionLogitBiasLayer#Config#
  LearnedPositionalEmbedding.embeddings: LearnedPositionalEmbedding#embeddings().
  GroupedQKVLinear.Config.num_kv_heads: GroupedQKVLinear#Config#num_kv_heads.
  YaRNSinusoidalPositionalEmbedding.Config.original_max_seq_length: YaRNSinusoidalPositionalEmbedding#Config#original_max_seq_length.
  ScaleQuery.Config.scale_factor: ScaleQuery#Config#scale_factor.
  ScaleQuery.Config.per_dim_scale: ScaleQuery#Config#per_dim_scale.
  ScaleKey.Config.scale_factor: ScaleKey#Config#scale_factor.
  MultiheadAttention.output_dim: MultiheadAttention#output_dim().
  MultiheadAttention._cap_logits: MultiheadAttention#_cap_logits().
  TransformerAttentionLayer.Config.source_dim: TransformerAttentionLayer#Config#source_dim.
  update_data_with_skip_connection: update_data_with_skip_connection().
  PipelinedTransformerLayer.Config.num_microbatches: PipelinedTransformerLayer#Config#num_microbatches.
  BaseQKVLinear.Output.query: BaseQKVLinear#Output#query.
  BaseQKVLinear.Output.key: BaseQKVLinear#Output#key.
  BaseQKVLinear.Output.value: BaseQKVLinear#Output#value.
  pow_scale_fn: pow_scale_fn().
  MultiheadAttention.Config.k_partition_spec: MultiheadAttention#Config#k_partition_spec.
  MultiheadRelativePositionLinear: MultiheadRelativePositionLinear#
  MultiheadAttentionXL.ScalePosition.QUERY: MultiheadAttentionXL#ScalePosition#QUERY.
  SinusoidalPositionalEmbedding.Config.dim: SinusoidalPositionalEmbedding#Config#dim.
  FusedQKVLinear.transform_factorization_spec: FusedQKVLinear#transform_factorization_spec().
  RoFormerSinusoidalPositionalEmbedding.Config.dim: RoFormerSinusoidalPositionalEmbedding#Config#dim.
  ScaleQuery.apply_norm: ScaleQuery#apply_norm().
  ScaleQuery.apply_per_dim_scale: ScaleQuery#apply_per_dim_scale().
  MultiheadAttention.hidden_dim: MultiheadAttention#hidden_dim().
  TransformerFeedForwardLayer.Config.activation: TransformerFeedForwardLayer#Config#activation.
  StackedTransformerLayer.Config.data_merger: StackedTransformerLayer#Config#data_merger.
  RematRegexSavePatterns.FEED_FORWARD: RematRegexSavePatterns#FEED_FORWARD.
  LearnedPositionalEmbedding: LearnedPositionalEmbedding#
  SinusoidalPositionalEmbedding: SinusoidalPositionalEmbedding#
  RoFormerSinusoidalPositionalEmbedding.default_query_positions: RoFormerSinusoidalPositionalEmbedding#default_query_positions().
  YaRNSinusoidalPositionalEmbedding: YaRNSinusoidalPositionalEmbedding#
  MultiheadAttention.Config.v_partition_spec: MultiheadAttention#Config#v_partition_spec.
  compute_gqa_logits: compute_gqa_logits().
  compute_gqa_context: compute_gqa_context().
  rel_pos_to_abs_pos: rel_pos_to_abs_pos().
  LearnedPositionalEmbedding.Config.dim: LearnedPositionalEmbedding#Config#dim.
  LearnedPositionalEmbedding.Config.shape: LearnedPositionalEmbedding#Config#shape.
  LearnedPositionalEmbedding.forward: LearnedPositionalEmbedding#forward().
  PerDimScale.Config.dim: PerDimScale#Config#dim.
  RoFormerQKVLinear.Config.rotary_value: RoFormerQKVLinear#Config#rotary_value.
  RoFormerQKVLinear.Config.query_scale: RoFormerQKVLinear#Config#query_scale.
  RoFormerQKVLinear.Config.key_scale: RoFormerQKVLinear#Config#key_scale.
  MultiheadAttention.Config.causal: MultiheadAttention#Config#causal.
  SigmoidAttention.Config.seq_len: SigmoidAttention#Config#seq_len.
  MultiheadAttentionXL.ScalePosition.LOGIT: MultiheadAttentionXL#ScalePosition#LOGIT.
  RematRegexSavePatterns: RematRegexSavePatterns#
  RematRegexSavePatterns.MOE_GATING: RematRegexSavePatterns#MOE_GATING.
  ALiBiAttentionLogitBiasLayer.Config.num_heads: ALiBiAttentionLogitBiasLayer#Config#num_heads.
  SymmetricALiBiAttentionLogitBiasLayer.Config.num_heads: SymmetricALiBiAttentionLogitBiasLayer#Config#num_heads.
  ScaleQuery.Config.norm: ScaleQuery#Config#norm.
  ScaleKey.Config.norm: ScaleKey#Config#norm.
  MultiheadAttention.Config.o_partition_spec: MultiheadAttention#Config#o_partition_spec.
  MultiheadAttention.Output.data: MultiheadAttention#Output#data.
  MultiheadAttention.Output.probs: MultiheadAttention#Output#probs.
  MultiheadAttention.Output.kv_state: MultiheadAttention#Output#kv_state.
  ParallelTransformerLayer: ParallelTransformerLayer#
  BottleNeckAdapterTransformerLayer: BottleNeckAdapterTransformerLayer#
  _TransformerRepeat: _TransformerRepeat#
  _TransformerPipeline: _TransformerPipeline#
  ALiBiAttentionLogitBiasLayer: ALiBiAttentionLogitBiasLayer#
  NormPosition.RES_NORM: NormPosition#RES_NORM.
  NormPosition.OUT_NORM: NormPosition#OUT_NORM.
  TransformerAttentionLayer.Config.structure: TransformerAttentionLayer#Config#structure.
  TransformerAttentionLayer.Output: TransformerAttentionLayer#Output#
  sinusoidal_positional_embeddings: sinusoidal_positional_embeddings().
  FusedQKVLinear.init: FusedQKVLinear#init().
  _rotary_sinusoidal_positional_embeddings: _rotary_sinusoidal_positional_embeddings().
  YaRNSinusoidalPositionalEmbedding.build_rotary_sinusoidal_positional_embeddings: YaRNSinusoidalPositionalEmbedding#build_rotary_sinusoidal_positional_embeddings().
  TransformerAttentionLayer.Output.data: TransformerAttentionLayer#Output#data.
  TransformerAttentionLayer.Output.probs: TransformerAttentionLayer#Output#probs.
  TransformerAttentionLayer.Output.kv_state: TransformerAttentionLayer#Output#kv_state.
  RematRegexSavePatterns.FLASH_CONTEXT: RematRegexSavePatterns#FLASH_CONTEXT.
  SymmetricALiBiAttentionLogitBiasLayer: SymmetricALiBiAttentionLogitBiasLayer#
  apply_rotary_position_embeddings.rotate_half: apply_rotary_position_embeddings().rotate_half().
  MultiheadAttention.Config.scale_kv_before_cache_update: MultiheadAttention#Config#scale_kv_before_cache_update.
  _next_power_of_two: _next_power_of_two().
  GroupedQKVLinear.num_kv_heads: GroupedQKVLinear#num_kv_heads().
  QLinear.is_kv_sharing: QLinear#is_kv_sharing().
  FusedGroupedQKVLinear.num_kv_heads: FusedGroupedQKVLinear#num_kv_heads().
  _find_correction_dim: _find_correction_dim().
  YaRNSinusoidalPositionalEmbedding.Config.scaling_factor: YaRNSinusoidalPositionalEmbedding#Config#scaling_factor.
  BaseStackedTransformerLayer.Config.peak_stochastic_depth_rate: BaseStackedTransformerLayer#Config#peak_stochastic_depth_rate.
  UpdateDataFn: UpdateDataFn#
  _TransformerRepeat.Config.carry: _TransformerRepeat#Config#carry.
  RematRegexSavePatterns.QKV_PROJ: RematRegexSavePatterns#QKV_PROJ.
  RematRegexSavePatterns.LINEAR1_X: RematRegexSavePatterns#LINEAR1_X.
  alibi_get_slopes.get_slopes_power_of_2: alibi_get_slopes().get_slopes_power_of_2().
  SinusoidalPositionalEmbedding.Config.min_timescale: SinusoidalPositionalEmbedding#Config#min_timescale.
  SinusoidalPositionalEmbedding.Config.max_timescale: SinusoidalPositionalEmbedding#Config#max_timescale.
  QKVLinear.Config.query_partition_spec: QKVLinear#Config#query_partition_spec.
  QKVLinear.Config.key_partition_spec: QKVLinear#Config#key_partition_spec.
  QKVLinear.Config.value_partition_spec: QKVLinear#Config#value_partition_spec.
  RoFormerSinusoidalPositionalEmbedding.Config.theta: RoFormerSinusoidalPositionalEmbedding#Config#theta.
  _linear_ramp_mask: _linear_ramp_mask().
  YaRNSinusoidalPositionalEmbedding.Config.beta_slow: YaRNSinusoidalPositionalEmbedding#Config#beta_slow.
  YaRNSinusoidalPositionalEmbedding.Config.beta_fast: YaRNSinusoidalPositionalEmbedding#Config#beta_fast.
  YaRNSinusoidalPositionalEmbedding.Config.extrapolation_factor: YaRNSinusoidalPositionalEmbedding#Config#extrapolation_factor.
  YaRNSinusoidalPositionalEmbedding.Config.attn_factor: YaRNSinusoidalPositionalEmbedding#Config#attn_factor.
  constant_scale_fn.constant_function: constant_scale_fn().constant_function().
  RoFormerQKVLinear.num_kv_heads: RoFormerQKVLinear#num_kv_heads().
  MultiheadAttention.Config.hidden_dim: MultiheadAttention#Config#hidden_dim.
  MultiheadAttention.Config.atten_logit_cap: MultiheadAttention#Config#atten_logit_cap.
  MultiheadAttentionXL.Config.pos_emb_dim: MultiheadAttentionXL#Config#pos_emb_dim.
  scaled_hidden_dim.scale_fn: scaled_hidden_dim().scale_fn().
  TransformerFeedForwardLayer.Config.add_value_rms_norm_summary: TransformerFeedForwardLayer#Config#add_value_rms_norm_summary.
  BottleNeckAdapterTransformerLayer.Config.bottleneck_ratio: BottleNeckAdapterTransformerLayer#Config#bottleneck_ratio.
  RematRegexSavePatterns.O_PROJ: RematRegexSavePatterns#O_PROJ.
  RematRegexSavePatterns.LINEAR2_X: RematRegexSavePatterns#LINEAR2_X.
  RematRegexSavePatterns.GATE_WEIGHTS: RematRegexSavePatterns#GATE_WEIGHTS.
  RematRegexSavePatterns.GATE_ASSIGNMENT: RematRegexSavePatterns#GATE_ASSIGNMENT.
  RematRegexSavePatterns.NATIVE_ATTENTION: RematRegexSavePatterns#NATIVE_ATTENTION.
  BaseMultiheadLinear.Config.bias: BaseMultiheadLinear#Config#bias.
  MultiheadInputLinear._einsum_expr: MultiheadInputLinear#_einsum_expr().
  MultiheadOutputLinear._einsum_expr: MultiheadOutputLinear#_einsum_expr().
  FusedGroupedQKVLinear.Config.output_partition_spec: FusedGroupedQKVLinear#Config#output_partition_spec.
  RoFormerQKVLinear.Config.partial_rope_factor: RoFormerQKVLinear#Config#partial_rope_factor.
  MultiheadAttention.Config.output_dim: MultiheadAttention#Config#output_dim.
  MultiheadAttention.Config.logit_sink: MultiheadAttention#Config#logit_sink.
  GroupedQueryAttention.num_kv_heads: GroupedQueryAttention#num_kv_heads().
  SigmoidAttention.Config.subtract_seq_len_bias: SigmoidAttention#Config#subtract_seq_len_bias.
  MultiheadRelativePositionLinear._einsum_expr: MultiheadRelativePositionLinear#_einsum_expr().
  TransformerAttentionLayer.Config.residual_gate_init: TransformerAttentionLayer#Config#residual_gate_init.
  TransformerFeedForwardLayer.Config.residual_weight: TransformerFeedForwardLayer#Config#residual_weight.
  TransformerFeedForwardLayer.Config.add_dead_neuron_summary: TransformerFeedForwardLayer#Config#add_dead_neuron_summary.
  TransformerFeedForwardLayer.Config.residual_gate_init: TransformerFeedForwardLayer#Config#residual_gate_init.
  RematRegexSavePatterns.CONTEXT: RematRegexSavePatterns#CONTEXT.
  RematRegexSavePatterns.INPUT: RematRegexSavePatterns#INPUT.
---
# Module: [`axlearn/common/attention.py`](../../../../../../raw/code/axlearn/axlearn/common/attention.py)

## Classes
### `ALiBiAttentionLogitBiasLayer`  ·  implements/extends CausalAttentionLogitBiasLayer
- def: [`axlearn/common/attention.py:4683`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4683)
- doc: attention logit bias layer in ALiBi.
- signature: `class ALiBiAttentionLogitBiasLayer(CausalAttentionLogitBiasLayer):`
- members:
  - `forward(self, *, segment_ids: Tensor, positions: Tensor)` — [`L4695`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4695) — Produces an attention logit biases of shape [batch_size, num_heads, seq_len, seq_len].
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`forward`](attention.md#CausalAttentionLogitBiasLayer.forward), [`apply_attention_logit_biases`](attention.md#apply_attention_logit_biases), [`alibi_get_slopes`](attention.md#alibi_get_slopes), [`CausalAttentionLogitBiasLayer`](attention.md#CausalAttentionLogitBiasLayer)
- used by: `test_alibi_attention_mask`, `test_packing`, [`forward`](attention.md#CausalAttentionLogitBiasLayer.forward), [`CausalAttentionLogitBiasLayer`](attention.md#CausalAttentionLogitBiasLayer)

### `AttentionLogitBiasLayer`  ·  implements/extends BaseLayer
- def: [`axlearn/common/attention.py:4590`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4590)
- doc: Base attention logit bias layer.
- signature: `class AttentionLogitBiasLayer(BaseLayer):`
- members:
  - `forward(self, *, segment_ids: Tensor, positions: Tensor)` — [`L4596`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4596) — Produces attention logit biases.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`BaseLayer`](base_layer.md#BaseLayer), [`forward`](attention.md#CausalAttentionLogitBiasLayer.forward), [`forward`](attention.md#FullAttentionLogitBiasLayer.forward), [`CausalAttentionLogitBiasLayer`](attention.md#CausalAttentionLogitBiasLayer), [`FullAttentionLogitBiasLayer`](attention.md#FullAttentionLogitBiasLayer)
- used by: [`BaseLayer`](base_layer.md#BaseLayer), [`CausalAttentionLogitBiasLayer`](attention.md#CausalAttentionLogitBiasLayer), [`FullAttentionLogitBiasLayer`](attention.md#FullAttentionLogitBiasLayer), [`attention_mask`](decoder.md#Decoder.Config.attention_mask)

### `BaseMultiheadLinear`  ·  implements/extends DenseGeneralBaseLayer
- def: [`axlearn/common/attention.py:459`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L459)
- doc: The linear layer used for multi-head attention.
- signature: `class BaseMultiheadLinear(DenseGeneralBaseLayer):`
- members:
  - `default_config(cls)` — [`L475`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L475)
  - `forward(self, inputs: Tensor)` — [`L498`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L498)
- protocol/private: `_compute_fan_axes`[`L505`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L505), `_create_layer_parameter_specs`[`L481`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L481), `_einsum_expr`[`L495`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L495)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`default_config`](config.md#Configurable.default_config), [`config`](config.md#Configurable.config), [`ParameterSpec`](base_layer.md#ParameterSpec), [`shape`](utils.md#TensorSpec.shape), [`mesh_axes`](utils.md#TensorSpec.mesh_axes), [`FanAxes`](param_init.md#FanAxes), [`parameters`](base_layer.md#BaseLayer.parameters), [`FactorizationSpec`](base_layer.md#FactorizationSpec), [`factorization`](base_layer.md#ParameterSpec.factorization), [`DenseGeneralBaseLayer`](quantized_dot_general/layers.md#DenseGeneralBaseLayer), [`axes`](base_layer.md#FactorizationSpec.axes), [`MultiheadInputLinear`](attention.md#MultiheadInputLinear), [`Config`](attention.md#BaseMultiheadLinear.Config), [`_compute_fan_axes`](attention.md#MultiheadInputLinear._compute_fan_axes), [`_compute_fan_axes`](attention.md#MultiheadOutputLinear._compute_fan_axes), [`_compute_fan_axes`](attention.md#MultiheadRelativePositionLinear._compute_fan_axes), [`MultiheadOutputLinear`](attention.md#MultiheadOutputLinear), [`einsum_maybe_quantized`](quantized_dot_general/layers.md#DenseGeneralBaseLayer.einsum_maybe_quantized), [`MultiheadRelativePositionLinear`](attention.md#MultiheadRelativePositionLinear), [`_einsum_expr`](attention.md#MultiheadInputLinear._einsum_expr), [`_einsum_expr`](attention.md#MultiheadOutputLinear._einsum_expr), [`_einsum_expr`](attention.md#MultiheadRelativePositionLinear._einsum_expr)
- used by: [`default_config`](config.md#Configurable.default_config), [`_create_layer_parameter_specs`](base_layer.md#BaseLayer._create_layer_parameter_specs), [`_compute_fan_axes`](base_layer.md#BaseLayer._compute_fan_axes), [`DenseGeneralBaseLayer`](quantized_dot_general/layers.md#DenseGeneralBaseLayer), [`MultiheadInputLinear`](attention.md#MultiheadInputLinear), [`Config`](attention.md#MultiheadInputLinear.Config), `test_unique_config_classes`, [`output_linear`](attention.md#MultiheadAttention.Config.output_linear), [`Config`](attention.md#MultiheadOutputLinear.Config), [`MultiheadOutputLinear`](attention.md#MultiheadOutputLinear), [`layer`](attention.md#FusedGroupedQKVLinear.Config.layer), [`layer`](attention.md#FusedQKVLinear.Config.layer), [`layer`](attention.md#QKVLinear.Config.layer), [`layer`](attention.md#QLinear.Config.layer), [`relative_pos_linear`](attention.md#MultiheadAttentionXL.Config.relative_pos_linear), [`MultiheadRelativePositionLinear`](attention.md#MultiheadRelativePositionLinear)

### `BaseQKVLinear`  ·  implements/extends BaseLayer
- def: [`axlearn/common/attention.py:661`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L661)
- doc: A layer that encapsulates mapping input queries, keys, and values to
- signature: `class BaseQKVLinear(BaseLayer):`
- members:
  - `forward(self, query: Tensor, *, key: Optional[Tensor] = None, value: Optional[Tensor] = None, kv_state: Optional[KVState] = None, query_positions: Optional[Tensor] = None)` — [`L703`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L703) — Computes per-head query, key, and value for the input query, key, value.
  - `is_kv_sharing(cls, cfg: Config)` — [`L690`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L690) — Check if the given config uses KV sharing (i.e., relies on external KVState).
  - `num_kv_heads(self)` — [`L700`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L700)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`BaseLayer`](base_layer.md#BaseLayer), [`KVState`](kv_cache/base_kv_cache.md#KVState), [`RoFormerQKVLinear`](attention.md#RoFormerQKVLinear), [`QKVLinear`](attention.md#QKVLinear), [`forward`](attention.md#QLinear.forward), [`FusedQKVLinear`](attention.md#FusedQKVLinear), [`QLinear`](attention.md#QLinear), [`forward`](attention.md#FusedGroupedQKVLinear.forward), [`forward`](attention.md#FusedQKVLinear.forward), [`FusedGroupedQKVLinear`](attention.md#FusedGroupedQKVLinear), [`Config`](attention.md#BaseQKVLinear.Config), [`Output`](attention.md#BaseQKVLinear.Output), [`forward`](attention.md#QKVLinear.forward), [`forward`](attention.md#RoFormerQKVLinear.forward), [`is_kv_sharing`](attention.md#RoFormerQKVLinear.is_kv_sharing), [`is_kv_sharing`](attention.md#QLinear.is_kv_sharing), [`num_kv_heads`](attention.md#FusedGroupedQKVLinear.num_kv_heads), [`num_kv_heads`](attention.md#GroupedQKVLinear.num_kv_heads), [`num_kv_heads`](attention.md#RoFormerQKVLinear.num_kv_heads)
- used by: [`BaseLayer`](base_layer.md#BaseLayer), `test_gqa_forward`, [`__init__`](attention.md#QKVLinear.__init__), `test_extend_step`, `test_gqa_extend_step`, `test_qkv_equality`, `test_roformer_qkv_linear`, [`_forward_for_mode`](rattention/rattention.md#ResidualLinearAttention._forward_for_mode), `test_gqa_prefill_states`, `test_prefill_states`, [`RoFormerQKVLinear`](attention.md#RoFormerQKVLinear), [`QKVLinear`](attention.md#QKVLinear), [`forward`](attention.md#QLinear.forward), [`FusedQKVLinear`](attention.md#FusedQKVLinear), [`QLinear`](attention.md#QLinear), `test_num_kv_heads`, [`input_linear`](attention.md#MultiheadAttention.Config.input_linear), [`extend_step`](rattention/rattention.md#ResidualLinearAttention.extend_step), [`forward`](attention.md#FusedQKVLinear.forward), [`FusedGroupedQKVLinear`](attention.md#FusedGroupedQKVLinear), [`forward`](attention.md#QKVLinear.forward), [`forward`](attention.md#RoFormerQKVLinear.forward), [`forward`](rattention/rattention.md#ResidualLinearAttention.forward), [`Config`](attention.md#RoFormerQKVLinear.Config), `forward`, [`input_linear`](attention.md#RoFormerQKVLinear.Config.input_linear), [`Config`](attention.md#QKVLinear.Config), [`Config`](attention.md#FusedGroupedQKVLinear.Config), [`Config`](attention.md#FusedQKVLinear.Config), [`Config`](attention.md#QLinear.Config)

### `BaseScaleQK`  ·  implements/extends BaseLayer
- def: [`axlearn/common/attention.py:1439`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1439)
- doc: Defines the common interface for scaling projected attention queries or keys.
- signature: `class BaseScaleQK(BaseLayer):`
- members:
  - `forward(self, proj: Tensor, *, positions: Optional[Tensor])` — [`L1452`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1452) — Scales the projected queries or keys.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`BaseLayer`](base_layer.md#BaseLayer), [`ScaleQuery`](attention.md#ScaleQuery), [`forward`](attention.md#ScaleQuery.forward), [`ScaleKey`](attention.md#ScaleKey), [`forward`](attention.md#ScaleKey.forward)
- used by: [`BaseLayer`](base_layer.md#BaseLayer), [`ScaleQuery`](attention.md#ScaleQuery), [`key_scale`](attention.md#MultiheadAttention.Config.key_scale), [`query_scale`](attention.md#MultiheadAttention.Config.query_scale), [`Config`](attention.md#ScaleKey.Config), [`Config`](attention.md#ScaleQuery.Config), [`ScaleKey`](attention.md#ScaleKey), [`key_scale`](attention.md#RoFormerQKVLinear.Config.key_scale), [`query_scale`](attention.md#RoFormerQKVLinear.Config.query_scale)

### `BaseStackedTransformerLayer`  ·  implements/extends BaseTransformerLayer
- def: [`axlearn/common/attention.py:3907`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3907)
- doc: The common interface of all stacked transformer layer classes.
- signature: `class BaseStackedTransformerLayer(BaseTransformerLayer):`
- uses (calls/refs, reference-scoped): [`BaseTransformerLayer`](attention.md#BaseTransformerLayer), [`StackedTransformerLayer`](attention.md#StackedTransformerLayer), [`RepeatedTransformerLayer`](attention.md#RepeatedTransformerLayer), [`PipelinedTransformerLayer`](attention.md#PipelinedTransformerLayer)
- used by: [`BaseTransformerLayer`](attention.md#BaseTransformerLayer), [`StackedTransformerLayer`](attention.md#StackedTransformerLayer), `test_transformer_extend_step`, [`build_remat_spec`](attention.md#build_remat_spec), `_convert_from_stacked_params`, [`RepeatedTransformerLayer`](attention.md#RepeatedTransformerLayer), `stack`, [`Config`](attention.md#PipelinedTransformerLayer.Config), [`PipelinedTransformerLayer`](attention.md#PipelinedTransformerLayer), [`transformer`](decoder.md#Decoder.Config.transformer), [`Config`](attention.md#RepeatedTransformerLayer.Config), [`Config`](attention.md#StackedTransformerLayer.Config)

### `BaseTransformerLayer`  ·  implements/extends BaseLayer
- def: [`axlearn/common/attention.py:176`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L176)
- doc: An abstract class to define the common interface of all *TransformerLayer classes, including:
- signature: `class BaseTransformerLayer(BaseLayer):`
- members:
  - `extend_step(self, cached_states: NestedTensor, data: Tensor, *, is_prefill: bool = False, self_attention_kv_state: Optional[KVState] = None, self_attention_logit_biases: Optional[Tensor] = None, cross_attention_data: Optional[Tensor] = None, cross_attention_logit_biases: Optional[Tensor] = None, target_segment_ids: Optional[Tensor] = None, page_pool: Optional[Nested[Tensor]] = None)` — [`L275`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L275) — Computes incremental outputs.
  - `forward(self, data: Tensor, *, self_attention_kv_state: Optional[KVState] = None, self_attention_logit_biases: Optional[Tensor] = None, cross_attention_data: Optional[Tensor] = None, cross_attention_logit_biases: Optional[Tensor] = None, target_segment_ids: Optional[Tensor] = None, target_positions: Optional[Tensor] = None, return_aux: Optional[set[str]] = None)` — [`L223`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L223) — Computes transformer layer outputs given full-sequence inputs.
  - `init_states(self, *, batch_size: int, max_len: int, dtype: jnp.dtype)` — [`L262`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L262) — Initializes cache for autoregressive cached decoding.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`NestedTensor`](utils.md#NestedTensor), [`BaseLayer`](base_layer.md#BaseLayer), [`KVState`](kv_cache/base_kv_cache.md#KVState), [`TransformerLayer`](attention.md#TransformerLayer), [`Output`](attention.md#BaseTransformerLayer.Output), [`nowrap`](module.md#nowrap), [`BaseStackedTransformerLayer`](attention.md#BaseStackedTransformerLayer), [`extend_step`](attention.md#StackedTransformerLayer.extend_step), [`extend_step`](attention.md#BottleNeckAdapterTransformerLayer.extend_step), [`extend_step`](attention.md#TransformerLayer.extend_step), [`forward`](attention.md#ParallelTransformerLayer.forward), [`forward`](attention.md#TransformerLayer.forward), [`forward`](attention.md#BottleNeckAdapterTransformerLayer.forward), [`forward`](attention.md#StackedTransformerLayer.forward), [`extend_step`](attention.md#RepeatedTransformerLayer.extend_step), [`init_states`](attention.md#StackedTransformerLayer.init_states), [`init_states`](attention.md#RepeatedTransformerLayer.init_states), [`forward`](attention.md#PipelinedTransformerLayer.forward), [`forward`](attention.md#RepeatedTransformerLayer.forward), [`init_states`](attention.md#BottleNeckAdapterTransformerLayer.init_states), [`init_states`](attention.md#TransformerLayer.init_states), [`BottleNeckAdapterTransformerLayer`](attention.md#BottleNeckAdapterTransformerLayer), [`ParallelTransformerLayer`](attention.md#ParallelTransformerLayer)
- used by: [`BaseLayer`](base_layer.md#BaseLayer), [`TransformerLayer`](attention.md#TransformerLayer), `test_build_remat_spec_neuron`, `test_update_layer_kwargs`, `test_stacked_transformer_with_non_uniform_layers`, [`_forward_for_mode`](attention.md#TransformerLayer._forward_for_mode), `test_stack_vs_pipeline`, `test_transformer_extend_step`, `test_passthrough_update_layer_kwargs`, `_test_decoder_with_transformer`, `test_build_remat_spec`, `_test_forward_vs_extend_step`, [`BaseStackedTransformerLayer`](attention.md#BaseStackedTransformerLayer), [`extend_step`](attention.md#BottleNeckAdapterTransformerLayer.extend_step), [`extend_step`](attention.md#TransformerLayer.extend_step), [`layer`](attention.md#BaseStackedTransformerLayer.Config.layer), [`forward`](attention.md#ParallelTransformerLayer.forward), [`forward`](attention.md#TransformerLayer.forward), [`Config`](attention.md#TransformerLayer.Config), [`_aggregate_layer_outputs`](attention.md#StackedTransformerLayer._aggregate_layer_outputs), [`Config`](attention.md#BaseStackedTransformerLayer.Config), `_aggregate_layer_outputs`, [`forward`](attention.md#BottleNeckAdapterTransformerLayer.forward), [`layer`](attention.md#BottleNeckAdapterTransformerLayer.Config.layer), [`layer`](attention.md#StackedTransformerLayer.Config.layer), [`_update_layer_kwargs`](attention.md#StackedTransformerLayer._update_layer_kwargs), [`update_data`](attention.md#update_data_with_skip_connection.update_data), `_update_layer_kwargs`, [`Config`](attention.md#BottleNeckAdapterTransformerLayer.Config), [`Config`](attention.md#ParallelTransformerLayer.Config), [`__call__`](attention.md#UpdateDataFn.__call__), `_aggregate_layer_outputs`, [`BottleNeckAdapterTransformerLayer`](attention.md#BottleNeckAdapterTransformerLayer), [`ParallelTransformerLayer`](attention.md#ParallelTransformerLayer)

### `BottleNeckAdapterTransformerLayer`  ·  implements/extends BaseTransformerLayer
- def: [`axlearn/common/attention.py:3683`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3683)
- doc: TransformerLayer with bottleneck adaptor for fine-tuning.
- signature: `class BottleNeckAdapterTransformerLayer(BaseTransformerLayer):`
- members:
  - `_forward_for_mode(self, *, mode: ForwardMode, data: Tensor, cached_states: Optional[NestedTensor] = None, **kwargs)` — [`L3716`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3716) — Computes transformer layer outputs and self/cross-attention probabilities.
  - `extend_step(self, cached_states: NestedTensor, data: Tensor, *, is_prefill: bool = False, **kwargs)` — [`L3791`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3791)
  - `forward(self, data: Tensor, **kwargs)` — [`L3764`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3764)
  - `init_states(self, *, batch_size: int, max_len: int, dtype: jnp.dtype)` — [`L3778`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3778) — Initializes cache for autoregressive cached decoding.
- protocol/private: `__init__`[`L3703`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3703)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`ForwardMode`](attention.md#ForwardMode), [`__init__`](base_layer.md#BaseLayer.__init__), [`BaseTransformerLayer`](attention.md#BaseTransformerLayer), [`vlog`](module.md#Module.vlog), [`Output`](attention.md#BaseTransformerLayer.Output), [`nowrap`](module.md#nowrap), [`PREFILL`](attention.md#ForwardMode.PREFILL), [`EXTEND_STEP`](attention.md#ForwardMode.EXTEND_STEP), [`input_dim`](attention.md#BaseTransformerLayer.Config.input_dim), [`FORWARD`](attention.md#ForwardMode.FORWARD), [`layer`](attention.md#BottleNeckAdapterTransformerLayer.Config.layer), [`Config`](attention.md#BottleNeckAdapterTransformerLayer.Config), [`adapter`](attention.md#BottleNeckAdapterTransformerLayer.Config.adapter), [`_next_power_of_two`](attention.md#_next_power_of_two), [`bottleneck_ratio`](attention.md#BottleNeckAdapterTransformerLayer.Config.bottleneck_ratio)
- used by: [`BaseTransformerLayer`](attention.md#BaseTransformerLayer), [`extend_step`](attention.md#BaseTransformerLayer.extend_step), [`forward`](attention.md#BaseTransformerLayer.forward), [`init_states`](attention.md#BaseTransformerLayer.init_states), `test_forward`

### `CausalAttentionLogitBiasLayer`  ·  implements/extends AttentionLogitBiasLayer
- def: [`axlearn/common/attention.py:4632`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4632)
- doc: Causal attention logit bias layer.
- signature: `class CausalAttentionLogitBiasLayer(AttentionLogitBiasLayer):`
- members:
  - `forward(self, *, segment_ids: Tensor, positions: Tensor)` — [`L4635`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4635) — Refer to AttentionLogitBiasLayer.forward for docstring.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`NEG_INF`](attention_bias.md#NEG_INF), [`apply_attention_logit_biases`](attention.md#apply_attention_logit_biases), [`forward`](attention.md#ALiBiAttentionLogitBiasLayer.forward), [`make_segment_mask`](attention_bias.md#make_segment_mask), [`AttentionLogitBiasLayer`](attention.md#AttentionLogitBiasLayer), [`ALiBiAttentionLogitBiasLayer`](attention.md#ALiBiAttentionLogitBiasLayer)
- used by: `test_alibi_attention_mask`, `test_causal_attention_mask_layer`, [`forward`](attention.md#ALiBiAttentionLogitBiasLayer.forward), [`AttentionLogitBiasLayer`](attention.md#AttentionLogitBiasLayer), [`Config`](attention.md#ALiBiAttentionLogitBiasLayer.Config), [`forward`](attention.md#AttentionLogitBiasLayer.forward), [`ALiBiAttentionLogitBiasLayer`](attention.md#ALiBiAttentionLogitBiasLayer)

### `Config`  ·  implements/extends Config
- def: [`axlearn/common/attention.py:4732`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4732)
- doc: Configures SymmetricALiBiAttentionLogitBiasLayer.
- signature: `class Config(FullAttentionLogitBiasLayer.Config):`
- members:
  - `activation` — [`L3153`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3153)
  - `adapter` — [`L3697`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3697)
  - `add_dead_neuron_summary` — [`L3190`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3190)
  - `add_value_rms_norm_summary` — [`L3197`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3197)
  - `atten_logit_cap` — [`L1750`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1750)
  - `attention` — [`L2800`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2800)
  - `attn_factor` — [`L1209`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1209)
  - `beta_fast` — [`L1204`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1204)
  - `beta_slow` — [`L1203`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1203)
  - `bias` — [`L472`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L472)
  - `bottleneck_ratio` — [`L3701`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3701)
  - `carry` — [`L4190`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4190)
  - `causal` — [`L1765`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1765)
  - `cross_attention` — [`L3432`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3432)
  - `data_merger` — [`L3986`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3986)
  - `dim` — [`L325`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L325)
  - `dim` — [`L447`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L447)
  - `dim` — [`L1120`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1120)
  - `dim` — [`L1374`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1374)
  - `dropout` — [`L1744`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1744)
  - `dropout` — [`L2803`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2803)
  - `dropout` — [`L3156`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3156)
  - `extrapolation_factor` — [`L1205`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1205)
  - `feed_forward` — [`L3433`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3433)
  - `feed_forward` — [`L3618`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3618)
  - `hidden_dim` — [`L1736`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1736)
  - `hidden_dim` — [`L3130`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3130)
  - `input_dim` — [`L188`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L188)
  - `input_dim` — [`L3126`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3126)
  - `input_linear` — [`L1581`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1581)
  - `input_linear` — [`L1740`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1740)
  - `k_partition_spec` — [`L1797`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1797)
  - `key_dim` — [`L673`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L673)
  - `key_dim` — [`L1733`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1733)
  - `key_partition_spec` — [`L743`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L743)
  - `key_scale` — [`L1587`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1587)
  - `key_scale` — [`L1748`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1748)
  - `kv_cache` — [`L1767`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1767)
  - `layer` — [`L739`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L739)
  - `layer` — [`L829`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L829)
  - `layer` — [`L877`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L877)
  - `layer` — [`L1002`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1002)
  - `layer` — [`L3693`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3693)
  - `layer` — [`L3929`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3929)
  - `layer` — [`L3981`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3981)
  - `linear1` — [`L3132`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3132)
  - `linear2` — [`L3136`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3136)
  - `logit_sink` — [`L1790`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1790)
  - `mask` — [`L1759`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1759)
  - `max_timescale` — [`L449`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L449)
  - `min_timescale` — [`L448`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L448)
  - `model_dim` — [`L469`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L469)
  - `norm` — [`L1476`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1476)
  - `norm` — [`L1537`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1537)
  - `norm` — [`L2797`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2797)
  - `norm` — [`L3140`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3140)
  - `norm` — [`L3616`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3616)
  - `num_heads` — [`L470`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L470)
  - `num_heads` — [`L677`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L677)
  - `num_heads` — [`L1738`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1738)
  - `num_heads` — [`L4693`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4693)
  - `num_heads` — [`L4735`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4735)
  - `num_kv_heads` — [`L805`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L805)
  - `num_kv_heads` — [`L1000`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1000)
  - `num_layers` — [`L3926`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3926)
  - `num_microbatches` — [`L4447`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4447)
  - `num_stages` — [`L4445`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4445)
  - `o_partition_spec` — [`L1804`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1804)
  - `original_max_seq_length` — [`L1200`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1200)
  - `output_dim` — [`L1735`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1735)
  - `output_linear` — [`L1742`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1742)
  - `output_partition_spec` — [`L1004`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1004)
  - `partial_rope_factor` — [`L1590`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1590)
  - `peak_stochastic_depth_rate` — [`L3930`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3930)
  - `per_dim_scale` — [`L1481`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1481)
  - `per_head_dim` — [`L471`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L471)
  - `per_head_dim` — [`L679`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L679)
  - `per_head_dim` — [`L1450`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1450)
  - `pipeline` — [`L4451`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4451)
  - `pos_emb_dim` — [`L2638`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2638)
  - `q_partition_spec` — [`L1793`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1793)
  - `query_dim` — [`L671`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L671)
  - `query_dim` — [`L1732`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1732)
  - `query_partition_spec` — [`L742`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L742)
  - `query_scale` — [`L1586`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1586)
  - `query_scale` — [`L1746`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1746)
  - `relative_pos_emb` — [`L2640`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2640)
  - `relative_pos_linear` — [`L2644`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2644)
  - `repeat` — [`L4340`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4340)
  - `residual_gate_init` — [`L2819`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2819)
  - `residual_gate_init` — [`L3200`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3200)
  - `residual_weight` — [`L3179`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3179)
  - `rope_pos_emb_layer` — [`L1578`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1578)
  - `rotary_value` — [`L1583`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1583)
  - `scale_factor` — [`L1479`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1479)
  - `scale_factor` — [`L1540`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1540)
  - `scale_kv_before_cache_update` — [`L1787`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1787)
  - `scale_position` — [`L2647`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2647)
  - `scaling_factor` — [`L1202`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1202)
  - `self_attention` — [`L3430`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3430)
  - `self_attention` — [`L3617`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3617)
  - `seq_len` — [`L2458`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2458)
  - `shape` — [`L326`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L326)
  - `source_dim` — [`L2795`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2795)
  - `stage` — [`L4449`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4449)
  - `stochastic_depth` — [`L2809`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2809)
  - `stochastic_depth` — [`L3163`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3163)
  - `structure` — [`L2817`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2817)
  - `structure` — [`L3176`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3176)
  - `subtract_seq_len_bias` — [`L2459`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2459)
  - `target_dim` — [`L2794`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2794)
  - `theta` — [`L1121`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1121)
  - `v_partition_spec` — [`L1801`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1801)
  - `value_dim` — [`L675`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L675)
  - `value_dim` — [`L1734`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1734)
  - `value_partition_spec` — [`L744`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L744)
- uses (calls/refs, reference-scoped): [`default_config`](config.md#Configurable.default_config), [`BaseLayer`](base_layer.md#BaseLayer), [`config_class`](config.md#config_class), [`REQUIRED`](config.md#REQUIRED), [`Required`](config.md#Required), [`InstantiableConfig`](config.md#InstantiableConfig), [`Config`](base_layer.md#BaseLayer.Config), [`MultiheadAttention`](attention.md#MultiheadAttention), [`BaseTransformerLayer`](attention.md#BaseTransformerLayer), [`TransformerLayer`](attention.md#TransformerLayer), [`MaskFnAttentionBias`](attention_bias.md#MaskFnAttentionBias), [`BaseQKVLinear`](attention.md#BaseQKVLinear), [`StackedTransformerLayer`](attention.md#StackedTransformerLayer), [`BaseKVCache`](kv_cache/base_kv_cache.md#BaseKVCache), [`KVCache`](kv_cache/kv_cache.md#KVCache), [`QKVLinear`](attention.md#QKVLinear), [`PerDimScale`](attention.md#PerDimScale), [`Dropout`](layers.md#Dropout), [`TransformerAttentionLayer`](attention.md#TransformerAttentionLayer), [`Config`](attention.md#MultiheadAttention.Config), [`ClassConfigBase`](config.md#ClassConfigBase), [`Config`](attention.md#PerDimScale.Config), [`NormPosition`](attention.md#NormPosition), [`FunctionConfigBase`](config.md#FunctionConfigBase), [`TransformerFeedForwardLayer`](attention.md#TransformerFeedForwardLayer), [`Config`](attention.md#BaseQKVLinear.Config), [`Config`](attention.md#BaseTransformerLayer.Config), [`MultiheadAttentionXL`](attention.md#MultiheadAttentionXL), [`ScaleQuery`](attention.md#ScaleQuery), [`RoFormerSinusoidalPositionalEmbedding`](attention.md#RoFormerSinusoidalPositionalEmbedding), [`BaseScaleQK`](attention.md#BaseScaleQK), [`MultiheadInputLinear`](attention.md#MultiheadInputLinear), [`PartitionSpecType`](utils.md#PartitionSpecType), [`Config`](attention.md#BaseScaleQK.Config), [`Config`](attention.md#TransformerFeedForwardLayer.Config), [`Config`](pipeline.md#Pipeline.Config), [`default_config`](attention.md#BaseMultiheadLinear.default_config), [`Config`](attention.md#MultiheadInputLinear.Config), [`ScaleFn`](attention.md#ScaleFn), [`Config`](attention.md#BaseMultiheadLinear.Config)  (+20 more)
- used by: [`_forward_for_mode`](attention.md#MultiheadAttention._forward_for_mode), [`_compute_attention`](flash_attention/layer.md#FlashAttention._compute_attention), [`_mask_tpl`](rattention/rattention.md#RAttention._mask_tpl), [`_mask_tpl`](attention.md#MultiheadAttention._mask_tpl), [`_forward_for_mode`](attention.md#_TransformerRepeat._forward_for_mode), `test_causal`, `_test_extend_step`, `test_sliding_window`, [`__init__`](attention.md#QKVLinear.__init__), `test_build_remat_spec_neuron`, [`__init__`](attention.md#TransformerFeedForwardLayer.__init__), `test_stack_vs_pipeline`, [`__init__`](attention.md#RoFormerQKVLinear.__init__), `test_repeated_layer_with_custom_carry`, `test_transformer_extend_step`, [`__init__`](attention.md#FusedGroupedQKVLinear.__init__), [`__init__`](attention.md#MultiheadAttentionXL.__init__), [`default_config`](flash_attention/layer.md#FlashAttention.default_config), `test_v2_structure`, `_test_prefill_states`, [`initialize_parameters_recursively`](attention.md#StackedTransformerLayer.initialize_parameters_recursively), [`__init__`](attention.md#FusedQKVLinear.__init__), `_test_decoder_with_transformer`, [`compute_rope_params`](attention.md#YaRNSinusoidalPositionalEmbedding.compute_rope_params), [`__init__`](attention.md#PipelinedTransformerLayer.__init__), [`__init__`](attention.md#TransformerAttentionLayer.__init__), `test_build_remat_spec`, `test_residual_gate`, `_convert_from_stacked_params`, [`__init__`](attention.md#QLinear.__init__), `test_forward_vs_extend_step`, `test_set_double_shard_weights_config`, [`_maybe_repeat_kv_heads`](flash_attention/layer.md#FlashAttention._maybe_repeat_kv_heads), `_test_forward_vs_extend_step`, [`initialize_parameters_recursively`](attention.md#PipelinedTransformerLayer.initialize_parameters_recursively), [`enable_sliding_window_attention`](kv_cache/sliding_window_kv_cache.md#enable_sliding_window_attention), [`set_attention_partition_specs`](attention.md#set_attention_partition_specs), [`set_double_shard_weights_config`](attention.md#set_double_shard_weights_config), [`__init__`](attention.md#BottleNeckAdapterTransformerLayer.__init__), [`__init__`](attention.md#ParallelTransformerLayer.__init__)  (+16 more)

### `ForwardMode`  ·  implements/extends Enum
- def: [`axlearn/common/attention.py:159`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L159) — documented in [axlearn-common-attention](../../../concepts/axlearn-common-attention.md)
- doc: ForwardMode describes the type of computation to be done in a forward pass through a layer.
- signature: `class ForwardMode(enum.Enum):`
- members:
  - `EXTEND_STEP` — [`L173`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L173)
  - `FORWARD` — [`L171`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L171)
  - `PREFILL` — [`L172`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L172)
- used by: [`_forward_for_mode`](attention.md#MultiheadAttention._forward_for_mode), [`_compute_attention`](flash_attention/layer.md#FlashAttention._compute_attention), [`_forward_for_mode`](rattention/rattention.md#RAttention._forward_for_mode), [`_forward_for_mode`](attention.md#_TransformerRepeat._forward_for_mode), `test_sigmoid_compute_attention`, [`_forward_for_mode`](attention.md#TransformerAttentionLayer._forward_for_mode), [`_forward_for_mode`](attention.md#TransformerLayer._forward_for_mode), `test_gqa_against_mha`, [`_compute_attention`](attention.md#MultiheadAttention._compute_attention), [`_forward_for_mode`](rattention/rattention.md#ResidualLinearAttention._forward_for_mode), [`_forward_for_mode`](attention.md#StackedTransformerLayer._forward_for_mode), [`_compute_attention`](attention.md#SigmoidAttention._compute_attention), [`extend_step`](decoder.md#Decoder.extend_step), [`extend_step`](attention.md#MultiheadAttention.extend_step), [`_forward_for_mode`](decoder.md#Decoder._forward_for_mode), [`extend_step`](rattention/rattention.md#RAttention.extend_step), [`extend_step`](attention.md#StackedTransformerLayer.extend_step), [`extend_step`](attention.md#TransformerAttentionLayer.extend_step), [`_forward_for_mode`](attention.md#BottleNeckAdapterTransformerLayer._forward_for_mode), [`extend_step`](attention.md#BottleNeckAdapterTransformerLayer.extend_step), [`extend_step`](attention.md#TransformerLayer.extend_step), [`extend_step`](attention.md#_TransformerRepeat.extend_step), [`extend_step`](rattention/rattention.md#ResidualLinearAttention.extend_step), [`forward`](attention.md#MultiheadAttention.forward), [`attention_thunk`](attention.md#TransformerAttentionLayer.attention_thunk), [`forward`](attention.md#TransformerLayer.forward), [`forward`](decoder.md#Decoder.forward), [`forward`](attention.md#BottleNeckAdapterTransformerLayer.forward), [`forward`](attention.md#StackedTransformerLayer.forward), [`forward`](attention.md#TransformerAttentionLayer.forward), [`forward`](attention.md#_TransformerRepeat.forward), [`forward`](rattention/rattention.md#ResidualLinearAttention.forward), [`layer_fn`](attention.md#_TransformerRepeat.layer_fn), [`_get_kv_cache_type`](flash_attention/layer.md#FlashAttention._get_kv_cache_type)

### `FullAttentionLogitBiasLayer`  ·  implements/extends AttentionLogitBiasLayer
- def: [`axlearn/common/attention.py:4644`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4644)
- doc: Full attention logit bias layer.
- signature: `class FullAttentionLogitBiasLayer(AttentionLogitBiasLayer):`
- members:
  - `forward(self, *, segment_ids: Tensor, positions: Tensor)` — [`L4647`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4647) — Refer to AttentionLogitBiasLayer.forward for docstring.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`make_segment_mask`](attention_bias.md#make_segment_mask), [`forward`](attention.md#SymmetricALiBiAttentionLogitBiasLayer.forward), [`AttentionLogitBiasLayer`](attention.md#AttentionLogitBiasLayer), [`SymmetricALiBiAttentionLogitBiasLayer`](attention.md#SymmetricALiBiAttentionLogitBiasLayer)
- used by: `test_full_attention_mask_layer`, [`forward`](attention.md#SymmetricALiBiAttentionLogitBiasLayer.forward), [`AttentionLogitBiasLayer`](attention.md#AttentionLogitBiasLayer), [`Config`](attention.md#SymmetricALiBiAttentionLogitBiasLayer.Config), [`forward`](attention.md#AttentionLogitBiasLayer.forward), [`SymmetricALiBiAttentionLogitBiasLayer`](attention.md#SymmetricALiBiAttentionLogitBiasLayer)

### `FusedGroupedQKVLinear`  ·  implements/extends BaseQKVLinear
- def: [`axlearn/common/attention.py:986`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L986)
- doc: Maps input query, key, and value to multi-headed query, key, and value using a fused weight.
- signature: `class FusedGroupedQKVLinear(BaseQKVLinear):`
- members:
  - `forward(self, query: Tensor, *, key: Optional[Tensor] = None, value: Optional[Tensor] = None, kv_state: Optional[Tensor] = None, query_positions: Optional[Tensor] = None)` — [`L1029`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1029) — See FusedQKVLinear for full docstring.
  - `num_kv_heads(self)` — [`L1026`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1026)
- protocol/private: `__init__`[`L1006`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1006)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`__init__`](base_layer.md#BaseLayer.__init__), [`BaseQKVLinear`](attention.md#BaseQKVLinear), [`with_sharding_constraint`](utils.md#with_sharding_constraint), [`FusedQKVLinear`](attention.md#FusedQKVLinear), `env`, [`Output`](attention.md#BaseQKVLinear.Output), [`num_heads`](attention.md#BaseQKVLinear.Config.num_heads), [`per_head_dim`](attention.md#BaseQKVLinear.Config.per_head_dim), [`query_dim`](attention.md#BaseQKVLinear.Config.query_dim), [`key_dim`](attention.md#BaseQKVLinear.Config.key_dim), [`value_dim`](attention.md#BaseQKVLinear.Config.value_dim), [`model_dim`](attention.md#BaseMultiheadLinear.Config.model_dim), [`num_heads`](attention.md#BaseMultiheadLinear.Config.num_heads), [`per_head_dim`](attention.md#BaseMultiheadLinear.Config.per_head_dim), [`Config`](attention.md#FusedGroupedQKVLinear.Config), [`layer`](attention.md#FusedGroupedQKVLinear.Config.layer), [`num_kv_heads`](attention.md#FusedGroupedQKVLinear.Config.num_kv_heads), [`key`](attention.md#BaseQKVLinear.Output.key), [`query`](attention.md#BaseQKVLinear.Output.query), [`value`](attention.md#BaseQKVLinear.Output.value)
- used by: `test_gqa_forward`, `test_causal`, `test_sliding_window`, [`BaseQKVLinear`](attention.md#BaseQKVLinear), `test_gqa_extend_step`, `test_qkv_equality`, `test_gqa_prefill_states`, `test_num_kv_heads`, [`forward`](attention.md#BaseQKVLinear.forward), `_convert_to_qkv_linear`, `test_fused_grouped_qkv_inside_shard_map`, [`num_kv_heads`](attention.md#BaseQKVLinear.num_kv_heads)

### `FusedQKVLinear`  ·  implements/extends BaseQKVLinear
- def: [`axlearn/common/attention.py:866`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L866)
- doc: Maps input query, key, and value to multi-headed query, key, and value using a fused weight.
- signature: `class FusedQKVLinear(BaseQKVLinear):`
- members:
  - `create_parameter_specs_recursively(self)` — [`L893`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L893)
  - `forward(self, query: Tensor, *, key: Optional[Tensor] = None, value: Optional[Tensor] = None, kv_state: Optional[KVState] = None, query_positions: Optional[Tensor] = None)` — [`L927`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L927) — Computes multi-head query, key, and value for the input query, key, value
  - `init(prng_key_i)` — [`L922`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L922)
  - `initialize_parameters_recursively(self, prng_key: Tensor, *, prebuilt: Optional[Nested[Optional[ParameterSpec]]] = None)` — [`L916`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L916)
  - `transform_factorization_spec(spec: Optional[FactorizationSpec])` — [`L896`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L896)
- protocol/private: `__init__`[`L879`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L879)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`ParameterSpec`](base_layer.md#ParameterSpec), [`shape`](utils.md#TensorSpec.shape), [`mesh_axes`](utils.md#TensorSpec.mesh_axes), [`dtype`](utils.md#TensorSpec.dtype), [`KVState`](kv_cache/base_kv_cache.md#KVState), [`__init__`](base_layer.md#BaseLayer.__init__), [`FanAxes`](param_init.md#FanAxes), [`child_context`](module.md#child_context), [`BaseQKVLinear`](attention.md#BaseQKVLinear), [`create_parameter_specs_recursively`](base_layer.md#BaseLayer.create_parameter_specs_recursively), [`VDict`](utils.md#VDict), [`FactorizationSpec`](base_layer.md#FactorizationSpec), [`split_prng_key`](utils.md#split_prng_key), [`factorization`](base_layer.md#ParameterSpec.factorization), [`Output`](attention.md#BaseQKVLinear.Output), [`num_heads`](attention.md#BaseQKVLinear.Config.num_heads), [`per_head_dim`](attention.md#BaseQKVLinear.Config.per_head_dim), [`query_dim`](attention.md#BaseQKVLinear.Config.query_dim), [`axes`](base_layer.md#FactorizationSpec.axes), [`NestedParameterSpec`](base_layer.md#NestedParameterSpec), [`key_dim`](attention.md#BaseQKVLinear.Config.key_dim), [`value_dim`](attention.md#BaseQKVLinear.Config.value_dim), [`fan_axes`](base_layer.md#ParameterSpec.fan_axes), [`_use_prebuilt_params`](base_layer.md#BaseLayer._use_prebuilt_params), [`model_dim`](attention.md#BaseMultiheadLinear.Config.model_dim), [`num_heads`](attention.md#BaseMultiheadLinear.Config.num_heads), [`per_head_dim`](attention.md#BaseMultiheadLinear.Config.per_head_dim), [`maybe_prepend_axis`](param_init.md#maybe_prepend_axis), [`Config`](attention.md#FusedQKVLinear.Config), [`layer`](attention.md#FusedQKVLinear.Config.layer), [`AxisType`](param_init.md#FanAxes.AxisType), [`key`](attention.md#BaseQKVLinear.Output.key), [`query`](attention.md#BaseQKVLinear.Output.query), [`value`](attention.md#BaseQKVLinear.Output.value)  (+2 more)
- used by: [`initialize_parameters_recursively`](base_layer.md#BaseLayer.initialize_parameters_recursively), `test_gqa_forward`, `test_causal`, `test_sliding_window`, [`BaseQKVLinear`](attention.md#BaseQKVLinear), [`create_parameter_specs_recursively`](base_layer.md#BaseLayer.create_parameter_specs_recursively), `test_qkv_equality`, `test_input_linear_variants`, `test_set_double_shard_weights_config`, `test_set_double_shard_weights_config_for_list_of_configs`, [`forward`](attention.md#FusedGroupedQKVLinear.forward), `test_num_kv_heads`, [`default_config`](attention.md#MultiheadAttentionXL.default_config), [`forward`](attention.md#BaseQKVLinear.forward), `test_set_attn_partition_specs`, `_convert_to_qkv_linear`, `test_invalid_key_value_combinations_raise`, `test_is_kv_sharing`, `combine_qkv`

### `GroupedQKVLinear`  ·  implements/extends QKVLinear
- def: [`axlearn/common/attention.py:792`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L792)
- doc: A variant of QKVLinear that supports configuring a different number of key, value
- signature: `class GroupedQKVLinear(QKVLinear):`
- members:
  - `num_kv_heads(self)` — [`L817`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L817)
- protocol/private: `__init__`[`L807`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L807)
- uses (calls/refs, reference-scoped): [`config`](config.md#Configurable.config), [`Module`](module.md#Module), [`__init__`](attention.md#QKVLinear.__init__), [`QKVLinear`](attention.md#QKVLinear), [`num_heads`](attention.md#BaseQKVLinear.Config.num_heads), [`Config`](attention.md#GroupedQKVLinear.Config), [`num_kv_heads`](attention.md#GroupedQKVLinear.Config.num_kv_heads)
- used by: [`_test_backward`](flash_attention/layer_test.md#TestFlashAttention._test_backward), `test_gqa_forward`, `test_causal`, `test_sliding_window`, [`test_logit_sink`](flash_attention/layer_test.md#TestFlashAttention.test_logit_sink), [`_prepare_layers`](flash_attention/layer_test.md#_prepare_layers), `test_gqa_against_mha`, `test_gqa_extend_step`, `test_qkv_equality`, `test_roformer_qkv_linear`, `test_gqa_prefill_states`, [`QKVLinear`](attention.md#QKVLinear), `test_num_kv_heads`, [`test_gqa_kv_heads`](flash_attention/layer_test.md#TestFlashAttention.test_gqa_kv_heads), [`num_kv_heads`](attention.md#BaseQKVLinear.num_kv_heads)

### `GroupedQueryAttention`  ·  implements/extends MultiheadAttention
- def: [`axlearn/common/attention.py:2388`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2388)
- doc: A Grouped-Query Attention (GQA) layer.
- signature: `class GroupedQueryAttention(MultiheadAttention):`
- members:
  - `_compute_context(self, probs: Tensor, v_proj: Tensor)` — [`L2428`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2428) — Compute attention context.
  - `_compute_logits(self, q_proj: Tensor, k_proj: Tensor)` — [`L2411`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2411) — Compute attention logits.
  - `num_kv_heads(self)` — [`L2408`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2408)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`MultiheadAttention`](attention.md#MultiheadAttention), [`FlashAttention`](flash_attention/layer.md#FlashAttention), [`_compute_logits`](attention.md#MultiheadAttention._compute_logits), [`_compute_context`](attention.md#MultiheadAttention._compute_context), [`compute_gqa_context`](attention.md#compute_gqa_context), [`compute_gqa_logits`](attention.md#compute_gqa_logits)
- used by: [`_test_backward`](flash_attention/layer_test.md#TestFlashAttention._test_backward), [`MultiheadAttention`](attention.md#MultiheadAttention), `test_gqa_forward`, `test_causal`, `_test_extend_step`, `test_sliding_window`, [`test_logit_sink`](flash_attention/layer_test.md#TestFlashAttention.test_logit_sink), [`_prepare_layers`](flash_attention/layer_test.md#_prepare_layers), `test_gqa_against_mha`, `test_gqa_extend_step`, `_test_prefill_states`, [`FlashAttention`](flash_attention/layer.md#FlashAttention), `test_gqa_prefill_states`, [`Config`](flash_attention/layer.md#FlashAttention.Config), [`layer`](flash_attention/layer_test.md#DummyModel.Config.layer), [`_compute_logits`](attention.md#MultiheadAttention._compute_logits), [`_compute_context`](attention.md#MultiheadAttention._compute_context)

### `LearnedPositionalEmbedding`  ·  implements/extends BaseLayer
- def: [`axlearn/common/attention.py:318`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L318)
- doc: TODO(ruoming): Remove LearnedPositionalEmbedding. We can just use the Embedding layer.
- signature: `class LearnedPositionalEmbedding(BaseLayer):`
- members:
  - `default_config(cls)` — [`L331`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L331)
  - `embeddings(self)` — [`L371`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L371) — Returns weights of shape cfg.shape + [dim].
  - `forward(self, positions: Tensor)` — [`L375`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L375) — Args:
- protocol/private: `_compute_fan_axes`[`L361`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L361), `_create_layer_parameter_specs`[`L352`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L352)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`default_config`](config.md#Configurable.default_config), [`config`](config.md#Configurable.config), [`BaseLayer`](base_layer.md#BaseLayer), [`ParameterSpec`](base_layer.md#ParameterSpec), [`shape`](utils.md#TensorSpec.shape), [`mesh_axes`](utils.md#TensorSpec.mesh_axes), [`FanAxes`](param_init.md#FanAxes), [`parameters`](base_layer.md#BaseLayer.parameters), [`DefaultInitializer`](param_init.md#DefaultInitializer), [`WeightInitializer`](param_init.md#WeightInitializer), [`in_axis`](param_init.md#FanAxes.in_axis), [`out_axis`](param_init.md#FanAxes.out_axis), [`PARAM_REGEXP_WEIGHT`](param_init.md#PARAM_REGEXP_WEIGHT), [`batch_axis`](param_init.md#FanAxes.batch_axis)
- used by: [`default_config`](config.md#Configurable.default_config), [`BaseLayer`](base_layer.md#BaseLayer), [`_create_layer_parameter_specs`](base_layer.md#BaseLayer._create_layer_parameter_specs), `_test_decoder_with_transformer`, `test_learned_positional_embedding_1d`, [`_compute_fan_axes`](base_layer.md#BaseLayer._compute_fan_axes)

### `MultiheadAttention`  ·  implements/extends BaseLayer
- def: [`axlearn/common/attention.py:1719`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1719) — documented in [axlearn-common-attention](../../../concepts/axlearn-common-attention.md)
- doc: A basic multi-head attention layer.
- signature: `class MultiheadAttention(BaseLayer):`
- members:
  - `_cap_logits(self, logits: Tensor)` — [`L2200`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2200) — Caps the logits with tanh.
  - `_compute_attention(self, *, mode: ForwardMode, q_proj: Tensor, kv_state: KVState, attention_logit_biases: BaseAttentionBias)` — [`L2112`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2112) — Computes attention context and probs.
  - `_compute_context(self, probs: Tensor, v_proj: Tensor)` — [`L2220`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2220) — Compute attention context.
  - `_compute_logits(self, q_proj: Tensor, k_proj: Tensor)` — [`L2208`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2208) — Compute attention logits.
  - `_forward_for_mode(self, *, mode: ForwardMode, query: Tensor, key: Optional[Tensor] = None, value: Optional[Tensor] = None, kv_state: Optional[KVState] = None, attention_logit_biases: Union[None, Tensor, BaseAttentionBias] = None, segment_ids: Optional[Tensor] = None, query_positions: Optional[Tensor] = None, cached_states: Optional[NestedTensor] = None, return_aux: Optional[set[str]] = None, page_pool: Optional[Nested[Tensor]] = None)` — [`L1901`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1901) — Computes attention for the given query, key, value, and attention logit biases. — documented in [axlearn-common-attention](../../../concepts/axlearn-common-attention.md)
  - `default_key_scale_config()` — [`L2328`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2328) — The config for the default function used to compute the key scale.
  - `default_query_scale_config()` — [`L2322`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2322) — The config for the default function used to compute the query scale.
  - `extend_step(self, cached_states: NestedTensor, query: Tensor, *, is_prefill: bool = False, key: Optional[Tensor] = None, value: Optional[Tensor] = None, kv_state: Optional[KVState] = None, segment_ids: Optional[Tensor] = None, attention_logit_biases: Optional[Tensor] = None, return_aux: Optional[set[str]] = None, page_pool: Optional[Nested[Tensor]] = None)` — [`L2259`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2259) — Computes the value vector given the query of the current step.
  - `forward(self, query: Tensor, *, key: Optional[Tensor] = None, value: Optional[Tensor] = None, kv_state: Optional[KVState] = None, attention_logit_biases: Optional[Tensor] = None, segment_ids: Optional[Tensor] = None, query_positions: Optional[Tensor] = None, return_aux: Optional[set[str]] = None)` — [`L2154`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2154) — Computes attention for the given query, key, value, and attention logit biases.
  - `hidden_dim(self)` — [`L1874`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1874)
  - `init_states(self, *, batch_size: int, max_len: int, dtype: jnp.dtype)` — [`L2233`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2233) — Initializes cache for autoregressive cached decoding.
  - `output_dim(self)` — [`L1869`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1869)
  - `per_head_dim(self)` — [`L1879`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1879)
- protocol/private: `__init__`[`L1806`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1806), `_create_layer_parameter_specs`[`L1850`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1850), `_mask_tpl`[`L1812`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1812), `_scale_qk`[`L2100`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2100)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`default_config`](config.md#Configurable.default_config), [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`BaseLayer`](base_layer.md#BaseLayer), [`Module`](module.md#Module), [`instantiate`](config.md#Configurable.Config.instantiate), [`_add_child`](module.md#Module._add_child), [`ParameterSpec`](base_layer.md#ParameterSpec), [`InstantiableConfig`](config.md#InstantiableConfig), [`BaseAttentionBias`](attention_bias.md#BaseAttentionBias), [`ForwardMode`](attention.md#ForwardMode), [`shape`](utils.md#TensorSpec.shape), [`_compute_attention`](flash_attention/layer.md#FlashAttention._compute_attention), [`mesh_axes`](utils.md#TensorSpec.mesh_axes), [`config_for_function`](config.md#config_for_function), [`dtype`](utils.md#TensorSpec.dtype), [`_forward_for_mode`](rattention/rattention.md#RAttention._forward_for_mode), [`KVState`](kv_cache/base_kv_cache.md#KVState), [`__init__`](base_layer.md#BaseLayer.__init__), [`MaskFnAttentionBias`](attention_bias.md#MaskFnAttentionBias), [`vlog`](module.md#Module.vlog), [`parameters`](base_layer.md#BaseLayer.parameters), [`CausalAttentionBias`](attention_bias.md#CausalAttentionBias), [`KVCache`](kv_cache/kv_cache.md#KVCache), [`nowrap`](module.md#nowrap), [`PREFILL`](attention.md#ForwardMode.PREFILL), [`_compute_attention`](attention.md#SigmoidAttention._compute_attention), [`_add_tensor_stats`](base_layer.md#BaseLayer._add_tensor_stats), [`default_config`](attention_bias.md#CausalAttentionBias.default_config), [`maybe_shard`](utils.md#maybe_shard), [`init_states`](rattention/rattention.md#RAttention.init_states), [`GroupedQueryAttention`](attention.md#GroupedQueryAttention), [`k_proj`](kv_cache/base_kv_cache.md#KVState.k_proj), [`v_proj`](kv_cache/base_kv_cache.md#KVState.v_proj), [`EXTEND_STEP`](attention.md#ForwardMode.EXTEND_STEP), [`extend_step`](rattention/rattention.md#RAttention.extend_step), [`value`](attention_bias.md#BaseAttentionBias.value), [`FORWARD`](attention.md#ForwardMode.FORWARD)  (+61 more)
- used by: [`BaseLayer`](base_layer.md#BaseLayer), [`_compute_attention`](flash_attention/layer.md#FlashAttention._compute_attention), [`_mask_tpl`](rattention/rattention.md#RAttention._mask_tpl), [`_forward_for_mode`](rattention/rattention.md#RAttention._forward_for_mode), `test_gqa_forward`, `test_causal`, `_test_extend_step`, `test_sliding_window`, `test_all_mask`, [`_create_layer_parameter_specs`](base_layer.md#BaseLayer._create_layer_parameter_specs), `test_extend_step`, `test_gqa_against_mha`, [`_test_extend_step`](flash_attention/layer_test.md#TestFlashAttention._test_extend_step), [`__init__`](attention.md#MultiheadAttentionXL.__init__), `_test_prefill_states`, `test_input_linear_variants`, `test_prefill_states`, [`_compute_attention`](attention.md#SigmoidAttention._compute_attention), `test_prescaled_kv_share`, `_scale_query_kwargs`, [`init_states`](rattention/rattention.md#RAttention.init_states), [`GroupedQueryAttention`](attention.md#GroupedQueryAttention), [`enable_sliding_window_attention`](kv_cache/sliding_window_kv_cache.md#enable_sliding_window_attention), [`set_attention_partition_specs`](attention.md#set_attention_partition_specs), `test_data_types`, [`_create_layer_parameter_specs`](attention.md#MultiheadAttentionXL._create_layer_parameter_specs), `test_add_tensor_stats`, `test_against_llama_for_attention`, `test_rope_self_attention`, [`MultiheadAttentionXL`](attention.md#MultiheadAttentionXL), `test_qkvo_partition_spec`, `test_segment_ids_vs_logit_biases_padding`, `test_set_attn_partition_specs`, [`__init__`](flash_attention/layer.md#FlashAttention.__init__), `test_logit_sink_with_attention_biases`, [`attention`](attention.md#TransformerAttentionLayer.Config.attention), `test_invalid_key_value_combinations_raise`, `test_logit_sink_basic_functionality`, `test_logit_sink_vs_no_logit_sink`, `test_logit_sink_with_different_dtypes`  (+12 more)

### `MultiheadAttentionXL`  ·  implements/extends MultiheadAttention
- def: [`axlearn/common/attention.py:2611`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2611)
- doc: Multi-head self-attention with relative positional embeddings.
- signature: `class MultiheadAttentionXL(MultiheadAttention):`
- members:
  - `default_config(cls)` — [`L2650`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2650)
  - `extend_step(self, cached_states: NestedTensor, query: Tensor, **kwargs)` — [`L2752`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2752)
  - `forward(self, query: Tensor, *, key: Optional[Tensor] = None, value: Optional[Tensor] = None, **kwargs)` — [`L2686`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2686)
- protocol/private: `__init__`[`L2659`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2659), `_compute_logits`[`L2723`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2723), `_create_layer_parameter_specs`[`L2676`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2676), `_scale_qk`[`L2698`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2698)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`default_config`](config.md#Configurable.default_config), [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`ParameterSpec`](base_layer.md#ParameterSpec), [`shape`](utils.md#TensorSpec.shape), [`mesh_axes`](utils.md#TensorSpec.mesh_axes), [`MultiheadAttention`](attention.md#MultiheadAttention), [`child_context`](module.md#child_context), [`parameters`](base_layer.md#BaseLayer.parameters), [`_create_layer_parameter_specs`](attention.md#MultiheadAttention._create_layer_parameter_specs), [`FusedQKVLinear`](attention.md#FusedQKVLinear), [`per_head_dim`](attention.md#MultiheadAttention.per_head_dim), [`input_linear`](attention.md#MultiheadAttention.Config.input_linear), [`clone`](config.md#ConfigBase.clone), [`constant_initializer`](param_init.md#constant_initializer), [`forward`](attention.md#MultiheadAttention.forward), [`initializer`](base_layer.md#ParameterSpec.initializer), [`__init__`](attention.md#MultiheadAttention.__init__), [`num_heads`](attention.md#MultiheadAttention.Config.num_heads), [`query_dim`](attention.md#MultiheadAttention.Config.query_dim), [`scale_position`](attention.md#MultiheadAttentionXL.Config.scale_position), [`ScalePosition`](attention.md#MultiheadAttentionXL.ScalePosition), [`Config`](attention.md#MultiheadAttentionXL.Config), [`Output`](attention.md#MultiheadAttention.Output), [`relative_pos_emb`](attention.md#MultiheadAttentionXL.Config.relative_pos_emb), [`relative_pos_linear`](attention.md#MultiheadAttentionXL.Config.relative_pos_linear), [`key_dim`](attention.md#MultiheadAttention.Config.key_dim), [`value_dim`](attention.md#MultiheadAttention.Config.value_dim), [`xl_attention_logits`](attention.md#xl_attention_logits), [`QUERY`](attention.md#MultiheadAttentionXL.ScalePosition.QUERY), [`LOGIT`](attention.md#MultiheadAttentionXL.ScalePosition.LOGIT), [`pos_emb_dim`](attention.md#MultiheadAttentionXL.Config.pos_emb_dim)
- used by: [`default_config`](config.md#Configurable.default_config), [`MultiheadAttention`](attention.md#MultiheadAttention), `test_per_dim_scale`, [`_create_layer_parameter_specs`](attention.md#MultiheadAttention._create_layer_parameter_specs), [`extend_step`](attention.md#MultiheadAttention.extend_step), `test_multihead_attention_xl`, [`forward`](attention.md#MultiheadAttention.forward), [`scale_position`](attention.md#MultiheadAttentionXL.Config.scale_position), [`_compute_logits`](attention.md#MultiheadAttention._compute_logits), [`_scale_qk`](attention.md#MultiheadAttention._scale_qk)

### `MultiheadInputLinear`  ·  implements/extends BaseMultiheadLinear
- def: [`axlearn/common/attention.py:509`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L509)
- doc: Multi-head input linear layer.
- signature: `class MultiheadInputLinear(BaseMultiheadLinear):`
- protocol/private: `_bias_spec`[`L523`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L523), `_compute_fan_axes`[`L531`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L531), `_einsum_expr`[`L519`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L519)
- uses (calls/refs, reference-scoped): [`config`](config.md#Configurable.config), [`ParameterSpec`](base_layer.md#ParameterSpec), [`shape`](utils.md#TensorSpec.shape), [`mesh_axes`](utils.md#TensorSpec.mesh_axes), [`FanAxes`](param_init.md#FanAxes), [`BaseMultiheadLinear`](attention.md#BaseMultiheadLinear), [`in_axis`](param_init.md#FanAxes.in_axis), [`out_axis`](param_init.md#FanAxes.out_axis)
- used by: `test_compute_fan_axes`, [`BaseMultiheadLinear`](attention.md#BaseMultiheadLinear), [`_compute_fan_axes`](attention.md#BaseMultiheadLinear._compute_fan_axes), `test_unique_config_classes`, [`_einsum_expr`](attention.md#BaseMultiheadLinear._einsum_expr), [`layer`](attention.md#FusedGroupedQKVLinear.Config.layer), [`layer`](attention.md#FusedQKVLinear.Config.layer), [`layer`](attention.md#QKVLinear.Config.layer), [`layer`](attention.md#QLinear.Config.layer)

### `MultiheadOutputLinear`  ·  implements/extends BaseMultiheadLinear
- def: [`axlearn/common/attention.py:538`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L538)
- doc: Multi-head output linear layer.
- signature: `class MultiheadOutputLinear(BaseMultiheadLinear):`
- protocol/private: `_bias_spec`[`L552`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L552), `_compute_fan_axes`[`L560`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L560), `_einsum_expr`[`L548`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L548)
- uses (calls/refs, reference-scoped): [`config`](config.md#Configurable.config), [`ParameterSpec`](base_layer.md#ParameterSpec), [`shape`](utils.md#TensorSpec.shape), [`mesh_axes`](utils.md#TensorSpec.mesh_axes), [`FanAxes`](param_init.md#FanAxes), [`BaseMultiheadLinear`](attention.md#BaseMultiheadLinear), [`in_axis`](param_init.md#FanAxes.in_axis), [`out_axis`](param_init.md#FanAxes.out_axis)
- used by: `test_compute_fan_axes`, [`BaseMultiheadLinear`](attention.md#BaseMultiheadLinear), [`_compute_fan_axes`](attention.md#BaseMultiheadLinear._compute_fan_axes), `test_unique_config_classes`, [`output_linear`](attention.md#MultiheadAttention.Config.output_linear), [`_einsum_expr`](attention.md#BaseMultiheadLinear._einsum_expr)

### `MultiheadRelativePositionLinear`  ·  implements/extends BaseMultiheadLinear
- def: [`axlearn/common/attention.py:2551`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2551)
- doc: Multi-head relative position linear layer.
- signature: `class MultiheadRelativePositionLinear(BaseMultiheadLinear):`
- protocol/private: `_bias_spec`[`L2559`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2559), `_compute_fan_axes`[`L2567`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2567), `_einsum_expr`[`L2555`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2555)
- uses (calls/refs, reference-scoped): [`config`](config.md#Configurable.config), [`ParameterSpec`](base_layer.md#ParameterSpec), [`shape`](utils.md#TensorSpec.shape), [`mesh_axes`](utils.md#TensorSpec.mesh_axes), [`FanAxes`](param_init.md#FanAxes), [`BaseMultiheadLinear`](attention.md#BaseMultiheadLinear), [`in_axis`](param_init.md#FanAxes.in_axis), [`out_axis`](param_init.md#FanAxes.out_axis)
- used by: `test_compute_fan_axes`, [`BaseMultiheadLinear`](attention.md#BaseMultiheadLinear), [`_compute_fan_axes`](attention.md#BaseMultiheadLinear._compute_fan_axes), [`_einsum_expr`](attention.md#BaseMultiheadLinear._einsum_expr), [`relative_pos_linear`](attention.md#MultiheadAttentionXL.Config.relative_pos_linear)

### `NormPosition`  ·  implements/extends Enum
- def: [`axlearn/common/attention.py:2761`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2761)
- doc: NormPosition is used for structure=v2 to indicate the positions of
- signature: `class NormPosition(enum.Enum):`
- members:
  - `IN_NORM` — [`L2779`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2779)
  - `OUT_NORM` — [`L2781`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2781)
  - `RES_NORM` — [`L2780`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2780)
- used by: [`_forward_for_mode`](attention.md#TransformerAttentionLayer._forward_for_mode), `test_v2_structure`, `test_residual_gate`, [`forward`](attention.md#TransformerFeedForwardLayer.forward), `test_v2_structure`, `test_residual_gate`, [`norm`](attention.md#TransformerAttentionLayer.Config.norm), [`norm`](attention.md#TransformerFeedForwardLayer.Config.norm)

### `Output`  ·  implements/extends NamedTuple
- def: [`axlearn/common/attention.py:2864`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2864)
- doc: Outputs of TransformerAttentionLayer.
- signature: `class Output(NamedTuple):`
- members:
  - `cross_attention_probs` — [`L221`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L221)
  - `data` — [`L218`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L218)
  - `data` — [`L1897`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1897)
  - `data` — [`L2875`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2875)
  - `key` — [`L685`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L685)
  - `kv_state` — [`L1899`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1899)
  - `kv_state` — [`L2877`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2877)
  - `probs` — [`L1898`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1898)
  - `probs` — [`L2876`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2876)
  - `query` — [`L683`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L683)
  - `self_attention_kv_state` — [`L220`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L220)
  - `self_attention_probs` — [`L219`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L219)
  - `value` — [`L687`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L687)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`KVState`](kv_cache/base_kv_cache.md#KVState)
- used by: [`_forward_for_mode`](attention.md#MultiheadAttention._forward_for_mode), [`_forward_for_mode`](rattention/rattention.md#RAttention._forward_for_mode), `test_update_layer_kwargs`, `test_stacked_transformer_with_non_uniform_layers`, [`_forward_for_mode`](attention.md#TransformerAttentionLayer._forward_for_mode), [`_forward_for_mode`](attention.md#TransformerLayer._forward_for_mode), [`forward`](attention.md#_TransformerPipeline.forward), `test_passthrough_update_layer_kwargs`, [`forward`](attention.md#QLinear.forward), [`extend_step`](attention.md#TransformerAttentionLayer.extend_step), `forward`, [`forward`](attention.md#FusedGroupedQKVLinear.forward), [`forward`](attention.md#FusedQKVLinear.forward), [`forward`](attention.md#ParallelTransformerLayer.forward), [`_aggregate_layer_outputs`](attention.md#StackedTransformerLayer._aggregate_layer_outputs), `_aggregate_layer_outputs`, [`forward`](attention.md#QKVLinear.forward), [`forward`](attention.md#TransformerAttentionLayer.forward), [`update_data`](attention.md#update_data_with_skip_connection.update_data), `_update_layer_kwargs`, [`layer_fn`](attention.md#_TransformerPipeline.layer_fn)

### `ParallelTransformerLayer`  ·  implements/extends BaseTransformerLayer
- def: [`axlearn/common/attention.py:3600`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3600)
- doc: A Transformer layer with parallel self-attention and feed-forward layers:
- signature: `class ParallelTransformerLayer(BaseTransformerLayer):`
- members:
  - `forward(self, *, data: Tensor, self_attention_logit_biases: Optional[Tensor] = None, target_segment_ids: Optional[Tensor] = None)` — [`L3637`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3637) — Computes transformer layer outputs and self/cross-attention probabilities.
- protocol/private: `__init__`[`L3622`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3622)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`__init__`](base_layer.md#BaseLayer.__init__), [`BaseTransformerLayer`](attention.md#BaseTransformerLayer), [`TransformerLayer`](attention.md#TransformerLayer), [`Output`](attention.md#BaseTransformerLayer.Output), [`input_dim`](attention.md#BaseTransformerLayer.Config.input_dim), [`self_attention`](attention.md#TransformerLayer.Config.self_attention), [`Config`](attention.md#TransformerLayer.Config), [`data`](attention.md#BaseTransformerLayer.Output.data), [`feed_forward`](attention.md#TransformerLayer.Config.feed_forward), [`self_attention_kv_state`](attention.md#BaseTransformerLayer.Output.self_attention_kv_state), [`Config`](attention.md#ParallelTransformerLayer.Config), [`self_attention_probs`](attention.md#BaseTransformerLayer.Output.self_attention_probs), [`cross_attention_probs`](attention.md#BaseTransformerLayer.Output.cross_attention_probs)
- used by: [`BaseTransformerLayer`](attention.md#BaseTransformerLayer), `test_with_golden_value`, [`forward`](attention.md#BaseTransformerLayer.forward)

### `PerDimScale`  ·  implements/extends BaseLayer
- def: [`axlearn/common/attention.py:1367`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1367)
- doc: A layer to scale individual dimensions of the input.
- signature: `class PerDimScale(BaseLayer):`
- members:
  - `default_config(cls)` — [`L1377`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1377)
  - `forward(self, x: Tensor)` — [`L1388`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1388) — Returns x * per_dim_scale.
- protocol/private: `_create_layer_parameter_specs`[`L1382`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1382)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`default_config`](config.md#Configurable.default_config), [`config`](config.md#Configurable.config), [`BaseLayer`](base_layer.md#BaseLayer), [`ParameterSpec`](base_layer.md#ParameterSpec), [`shape`](utils.md#TensorSpec.shape), [`mesh_axes`](utils.md#TensorSpec.mesh_axes), [`parameters`](base_layer.md#BaseLayer.parameters), [`Config`](attention.md#PerDimScale.Config), [`ConstantInitializer`](param_init.md#ConstantInitializer), [`param_init`](base_layer.md#BaseLayer.Config.param_init)
- used by: [`default_config`](config.md#Configurable.default_config), [`BaseLayer`](base_layer.md#BaseLayer), `test_gqa_forward`, `test_all_mask`, [`_create_layer_parameter_specs`](base_layer.md#BaseLayer._create_layer_parameter_specs), `test_extend_step`, `test_gqa_extend_step`, `test_per_dim_scale`, `test_gqa_prefill_states`, `test_input_linear_variants`, `test_prefill_states`, `test_scale_query`, `_scale_kwargs`, `test_data_types`, `test_per_dim_scale`, [`per_dim_scale`](attention.md#ScaleQuery.Config.per_dim_scale)

### `PipelinedTransformerLayer`  ·  implements/extends BaseStackedTransformerLayer
- def: [`axlearn/common/attention.py:4437`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4437)
- doc: An implementation of BaseStackedTransformerLayer with pipeline model parallelism.
- signature: `class PipelinedTransformerLayer(BaseStackedTransformerLayer):`
- members:
  - `forward(self, data: Tensor, **kwargs)` — [`L4480`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4480)
  - `initialize_parameters_recursively(self, prng_key: Tensor, *, prebuilt: Optional[Nested[Optional[ParameterSpec]]] = None)` — [`L4467`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4467)
- protocol/private: `__init__`[`L4453`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4453)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`ParameterSpec`](base_layer.md#ParameterSpec), [`__init__`](base_layer.md#BaseLayer.__init__), [`TransformerLayer`](attention.md#TransformerLayer), [`Output`](attention.md#BaseTransformerLayer.Output), [`input_dim`](attention.md#BaseTransformerLayer.Config.input_dim), [`BaseStackedTransformerLayer`](attention.md#BaseStackedTransformerLayer), [`split_prng_key`](utils.md#split_prng_key), [`layer`](attention.md#BaseStackedTransformerLayer.Config.layer), [`num_layers`](attention.md#BaseStackedTransformerLayer.Config.num_layers), [`Config`](attention.md#PipelinedTransformerLayer.Config), [`num_stages`](attention.md#PipelinedTransformerLayer.Config.num_stages), [`stage`](attention.md#PipelinedTransformerLayer.Config.stage), [`get_or_none`](utils.md#get_or_none), [`pipeline`](attention.md#PipelinedTransformerLayer.Config.pipeline), [`num_microbatches`](attention.md#PipelinedTransformerLayer.Config.num_microbatches)
- used by: [`initialize_parameters_recursively`](base_layer.md#BaseLayer.initialize_parameters_recursively), `_compare_layers`, `test_stack_vs_pipeline`, [`build_remat_spec`](attention.md#build_remat_spec), `_convert_from_stacked_params`, [`BaseStackedTransformerLayer`](attention.md#BaseStackedTransformerLayer), [`forward`](attention.md#BaseTransformerLayer.forward), `test_pipeline_return_aux`

### `QKVLinear`  ·  implements/extends BaseQKVLinear
- def: [`axlearn/common/attention.py:731`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L731)
- doc: Maps input query, key, and value to multi-headed output query, key, and value.
- signature: `class QKVLinear(BaseQKVLinear):`
- members:
  - `forward(self, query: Tensor, *, key: Optional[Tensor] = None, value: Optional[Tensor] = None, kv_state: Optional[Tensor] = None, query_positions: Optional[Tensor] = None)` — [`L762`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L762) — Computes attention for the given query, key, value.
- protocol/private: `__init__`[`L746`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L746)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`__init__`](base_layer.md#BaseLayer.__init__), [`BaseQKVLinear`](attention.md#BaseQKVLinear), [`clone`](config.md#ConfigBase.clone), [`GroupedQKVLinear`](attention.md#GroupedQKVLinear), [`Output`](attention.md#BaseQKVLinear.Output), [`num_heads`](attention.md#BaseQKVLinear.Config.num_heads), [`per_head_dim`](attention.md#BaseQKVLinear.Config.per_head_dim), [`query_dim`](attention.md#BaseQKVLinear.Config.query_dim), [`key_dim`](attention.md#BaseQKVLinear.Config.key_dim), [`num_kv_heads`](attention.md#BaseQKVLinear.num_kv_heads), [`value_dim`](attention.md#BaseQKVLinear.Config.value_dim), [`model_dim`](attention.md#BaseMultiheadLinear.Config.model_dim), [`num_heads`](attention.md#BaseMultiheadLinear.Config.num_heads), [`per_head_dim`](attention.md#BaseMultiheadLinear.Config.per_head_dim), [`Config`](attention.md#QKVLinear.Config), [`layer`](attention.md#QKVLinear.Config.layer), [`key`](attention.md#BaseQKVLinear.Output.key), [`param_partition_spec`](base_layer.md#BaseLayer.Config.param_partition_spec), [`query`](attention.md#BaseQKVLinear.Output.query), [`value`](attention.md#BaseQKVLinear.Output.value), [`key_partition_spec`](attention.md#QKVLinear.Config.key_partition_spec), [`query_partition_spec`](attention.md#QKVLinear.Config.query_partition_spec), [`value_partition_spec`](attention.md#QKVLinear.Config.value_partition_spec)
- used by: [`_test_backward`](flash_attention/layer_test.md#TestFlashAttention._test_backward), `test_gqa_forward`, `test_extend_step`, [`BaseQKVLinear`](attention.md#BaseQKVLinear), [`test_logit_sink`](flash_attention/layer_test.md#TestFlashAttention.test_logit_sink), [`_prepare_layers`](flash_attention/layer_test.md#_prepare_layers), [`__init__`](attention.md#RoFormerQKVLinear.__init__), `test_qkv_equality`, `test_roformer_qkv_linear`, `test_prefill_states`, `test_qlinear`, `test_set_double_shard_weights_config`, `test_set_double_shard_weights_config_for_list_of_configs`, `test_num_kv_heads`, [`input_linear`](attention.md#MultiheadAttention.Config.input_linear), [`forward`](attention.md#BaseQKVLinear.forward), [`GroupedQKVLinear`](attention.md#GroupedQKVLinear), `test_set_attn_partition_specs`, [`__init__`](attention.md#GroupedQKVLinear.__init__), [`test_gqa_kv_heads`](flash_attention/layer_test.md#TestFlashAttention.test_gqa_kv_heads), `test_is_kv_sharing`, [`input_linear`](attention.md#RoFormerQKVLinear.Config.input_linear), [`Config`](attention.md#GroupedQKVLinear.Config)

### `QLinear`  ·  implements/extends BaseQKVLinear
- def: [`axlearn/common/attention.py:821`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L821)
- doc: Maps input query to multi-headed output query. Assumes external KVState.
- signature: `class QLinear(BaseQKVLinear):`
- members:
  - `forward(self, query: Tensor, *, kv_state: KVState, key: Optional[Tensor] = None, value: Optional[Tensor] = None, query_positions: Optional[Tensor] = None)` — [`L844`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L844) — Computes projects for the given query. Uses {k,v}_proj from `kv_state`.
  - `is_kv_sharing(cls, cfg: Config)` — [`L832`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L832)
- protocol/private: `__init__`[`L835`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L835)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`KVState`](kv_cache/base_kv_cache.md#KVState), [`__init__`](base_layer.md#BaseLayer.__init__), [`BaseQKVLinear`](attention.md#BaseQKVLinear), [`k_proj`](kv_cache/base_kv_cache.md#KVState.k_proj), [`v_proj`](kv_cache/base_kv_cache.md#KVState.v_proj), [`Output`](attention.md#BaseQKVLinear.Output), [`num_heads`](attention.md#BaseQKVLinear.Config.num_heads), [`per_head_dim`](attention.md#BaseQKVLinear.Config.per_head_dim), [`query_dim`](attention.md#BaseQKVLinear.Config.query_dim), `forward`, [`model_dim`](attention.md#BaseMultiheadLinear.Config.model_dim), [`num_heads`](attention.md#BaseMultiheadLinear.Config.num_heads), [`per_head_dim`](attention.md#BaseMultiheadLinear.Config.per_head_dim), [`Config`](attention.md#QLinear.Config), [`layer`](attention.md#QLinear.Config.layer), `_QLinearWithKvUpdate`, [`key`](attention.md#BaseQKVLinear.Output.key), [`query`](attention.md#BaseQKVLinear.Output.query), [`value`](attention.md#BaseQKVLinear.Output.value)
- used by: `_test_extend_step`, `test_extend_step`, `test_update_layer_kwargs`, [`BaseQKVLinear`](attention.md#BaseQKVLinear), `test_repeated_layer_with_custom_carry`, `test_prescaled_kv_share`, `test_qlinear`, `test_self_attention_kv_state`, `test_kv_sharing`, `test_qlinear_kv_sharing`, [`forward`](attention.md#BaseQKVLinear.forward), `forward`, `test_is_kv_sharing`, [`is_kv_sharing`](attention.md#BaseQKVLinear.is_kv_sharing), `_QLinearWithKvUpdate`

### `RematRegexSavePatterns`  ·  implements/extends Enum
- def: [`axlearn/common/attention.py:4508`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4508)
- doc: Common regex patterns for saving tensors in attention and feedforward layers.
- signature: `class RematRegexSavePatterns(enum.Enum):`
- members:
  - `CONTEXT` — [`L4513`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4513)
  - `FEED_FORWARD` — [`L4523`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4523)
  - `FLASH_ATTENTION` — [`L4522`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4522)
  - `FLASH_CONTEXT` — [`L4521`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4521)
  - `GATE_ASSIGNMENT` — [`L4517`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4517)
  - `GATE_WEIGHTS` — [`L4516`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4516)
  - `INPUT` — [`L4525`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4525)
  - `LINEAR1_X` — [`L4514`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4514)
  - `LINEAR2_X` — [`L4515`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4515)
  - `MOE_GATING` — [`L4524`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4524)
  - `NATIVE_ATTENTION` — [`L4520`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4520)
  - `O_PROJ` — [`L4512`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4512)
  - `QKV_PROJ` — [`L4511`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4511)
- uses (calls/refs, reference-scoped): [`FLASH_ATTN_RESIDUAL_NAME`](flash_attention/remat.md#FLASH_ATTN_RESIDUAL_NAME)
- used by: `test_build_remat_spec_neuron`, [`build_remat_spec`](attention.md#build_remat_spec), `test_linear_remat`

### `RepeatedTransformerLayer`  ·  implements/extends BaseStackedTransformerLayer
- def: [`axlearn/common/attention.py:4328`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4328)
- doc: An implementation of BaseStackedTransformerLayer with a scan loop.
- signature: `class RepeatedTransformerLayer(BaseStackedTransformerLayer):`
- members:
  - `extend_step(self, cached_states: NestedTensor, data: Tensor, *, is_prefill: bool = False, **layer_kwargs)` — [`L4384`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4384)
  - `forward(self, data: Tensor, **layer_kwargs)` — [`L4362`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4362)
  - `init_states(self, *, batch_size: int, max_len: int, dtype: jnp.dtype)` — [`L4370`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4370) — Initializes cache for autoregressive cached decoding.
  - `initialize_parameters_recursively(self, prng_key: Tensor, *, prebuilt: Optional[NestedTensor] = None)` — [`L4351`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4351)
- protocol/private: `__init__`[`L4342`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4342)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`__init__`](base_layer.md#BaseLayer.__init__), [`TransformerLayer`](attention.md#TransformerLayer), [`Output`](attention.md#BaseTransformerLayer.Output), [`nowrap`](module.md#nowrap), [`VDict`](utils.md#VDict), [`input_dim`](attention.md#BaseTransformerLayer.Config.input_dim), [`BaseStackedTransformerLayer`](attention.md#BaseStackedTransformerLayer), [`layer`](attention.md#BaseStackedTransformerLayer.Config.layer), [`num_layers`](attention.md#BaseStackedTransformerLayer.Config.num_layers), [`get_or_none`](utils.md#get_or_none), [`repeat`](attention.md#RepeatedTransformerLayer.Config.repeat), [`Config`](attention.md#RepeatedTransformerLayer.Config)
- used by: [`initialize_parameters_recursively`](base_layer.md#BaseLayer.initialize_parameters_recursively), `_compare_layers`, `test_stack_vs_pipeline`, `test_repeated_layer_with_custom_carry`, `test_transformer_extend_step`, `_convert_from_stacked_params`, `test_transformer_prefill_states`, [`BaseStackedTransformerLayer`](attention.md#BaseStackedTransformerLayer), [`extend_step`](attention.md#BaseTransformerLayer.extend_step), [`forward`](attention.md#BaseTransformerLayer.forward), `test_initialize_parameters_recursively`, [`init_states`](attention.md#BaseTransformerLayer.init_states), `test_stack_vs_repeat_remat_everything_saveable`, `test_nested_stacked_decoding`, `test_stack_vs_repeat_with_build_remat_spec`, `test_stacked_decoding`, `test_stack_vs_repeat`

### `RoFormerQKVLinear`  ·  implements/extends BaseQKVLinear
- def: [`axlearn/common/attention.py:1568`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1568)
- doc: RoFormerQKVLinear class
- signature: `class RoFormerQKVLinear(BaseQKVLinear):`
- members:
  - `forward(self, query: Tensor, *, key: Optional[Tensor] = None, value: Optional[Tensor] = None, kv_state: Optional[KVState] = None, query_positions: Optional[Tensor] = None)` — [`L1650`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1650)
  - `is_kv_sharing(cls, cfg: Config)` — [`L1593`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1593)
  - `num_kv_heads(self)` — [`L1646`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1646) — Propagate num KV heads from input linear.
- protocol/private: `__init__`[`L1596`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1596)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`default_config`](config.md#Configurable.default_config), [`config`](config.md#Configurable.config), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`config_for_function`](config.md#config_for_function), [`KVState`](kv_cache/base_kv_cache.md#KVState), [`__init__`](base_layer.md#BaseLayer.__init__), [`BaseQKVLinear`](attention.md#BaseQKVLinear), [`QKVLinear`](attention.md#QKVLinear), [`klass`](config.md#Configurable.Config.klass), [`ScaleQuery`](attention.md#ScaleQuery), [`Output`](attention.md#BaseQKVLinear.Output), [`num_heads`](attention.md#BaseQKVLinear.Config.num_heads), [`per_head_dim`](attention.md#BaseQKVLinear.Config.per_head_dim), [`query_dim`](attention.md#BaseQKVLinear.Config.query_dim), [`Config`](attention.md#RoFormerQKVLinear.Config), [`constant_scale_fn`](attention.md#constant_scale_fn), [`key_dim`](attention.md#BaseQKVLinear.Config.key_dim), [`value_dim`](attention.md#BaseQKVLinear.Config.value_dim), [`input_linear`](attention.md#RoFormerQKVLinear.Config.input_linear), [`apply_rotary_position_embeddings`](attention.md#apply_rotary_position_embeddings), [`Config`](attention.md#QKVLinear.Config), [`ScaleKey`](attention.md#ScaleKey)
- used by: [`_mask_tpl`](rattention/rattention.md#RAttention._mask_tpl), `test_causal`, `test_sliding_window`, `test_extend_step`, [`BaseQKVLinear`](attention.md#BaseQKVLinear), `test_roformer_qkv_linear`, `test_prefill_states`, `test_qlinear`, `test_set_double_shard_weights_config`, `test_kv_sharing`, `test_qlinear_kv_sharing`, [`forward`](attention.md#BaseQKVLinear.forward), `test_against_llama_for_attention`, `test_rope_self_attention`, `test_default_rope_emb`, `test_set_attn_partition_specs`, [`num_kv_heads`](attention.md#BaseQKVLinear.num_kv_heads), `test_is_kv_sharing`, [`is_kv_sharing`](attention.md#BaseQKVLinear.is_kv_sharing)

### `RoFormerSinusoidalPositionalEmbedding`  ·  implements/extends BaseLayer
- def: [`axlearn/common/attention.py:1109`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1109)
- doc: Implementation of Rotary Position Embedding (RoPE).
- signature: `class RoFormerSinusoidalPositionalEmbedding(BaseLayer):`
- members:
  - `default_query_positions(self, max_seq_len: int)` — [`L1123`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1123) — Compute default `positions` value to be inputed into forward when `positions` is
  - `forward(self, *, positions: Optional[Tensor] = None, max_seq_len: Optional[int] = None)` — [`L1129`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1129) — TODO(bwzhang): 1. verify the performance under float32.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`BaseLayer`](base_layer.md#BaseLayer), [`forward`](attention.md#YaRNSinusoidalPositionalEmbedding.forward), [`YaRNSinusoidalPositionalEmbedding`](attention.md#YaRNSinusoidalPositionalEmbedding), [`_rotary_sinusoidal_positional_embeddings`](attention.md#_rotary_sinusoidal_positional_embeddings)
- used by: [`BaseLayer`](base_layer.md#BaseLayer), [`_mask_tpl`](rattention/rattention.md#RAttention._mask_tpl), `test_against_llama_for_apply_rotary_emb`, `test_against_llama_for_precompute_freqs_cis`, `test_default_rope_emb`, `test_rope_emb`, [`forward`](attention.md#YaRNSinusoidalPositionalEmbedding.forward), `test_rope_emb_no_pos`, [`Config`](attention.md#YaRNSinusoidalPositionalEmbedding.Config), [`rope_pos_emb_layer`](attention.md#RoFormerQKVLinear.Config.rope_pos_emb_layer), [`YaRNSinusoidalPositionalEmbedding`](attention.md#YaRNSinusoidalPositionalEmbedding)

### `ScaleKey`  ·  implements/extends BaseScaleQK
- def: [`axlearn/common/attention.py:1528`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1528)
- doc: Default implementation for scaling projected keys.
- signature: `class ScaleKey(BaseScaleQK):`
- members:
  - `default_scale_factor_config()` — [`L1562`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1562) — The config for the default function used to compute the key scale.
  - `forward(self, proj: Tensor, *, positions: Optional[Tensor])` — [`L1552`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1552) — Scales the projected keys.
- protocol/private: `__init__`[`L1542`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1542), `_scale_factor`[`L1545`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1545)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`InstantiableConfig`](config.md#InstantiableConfig), [`config_for_function`](config.md#config_for_function), [`__init__`](base_layer.md#BaseLayer.__init__), [`instantiate`](config.md#InstantiableConfig.instantiate), [`BaseScaleQK`](attention.md#BaseScaleQK), [`constant_scale_fn`](attention.md#constant_scale_fn), [`ScaleFn`](attention.md#ScaleFn), [`Config`](attention.md#ScaleKey.Config), [`per_head_dim`](attention.md#BaseScaleQK.Config.per_head_dim), [`scale_factor`](attention.md#ScaleKey.Config.scale_factor), [`norm`](attention.md#ScaleKey.Config.norm)
- used by: [`__init__`](attention.md#RoFormerQKVLinear.__init__), `test_roformer_qkv_linear`, `_scale_query_kwargs`, `_scale_kwargs`, [`BaseScaleQK`](attention.md#BaseScaleQK), [`key_scale`](attention.md#MultiheadAttention.Config.key_scale), [`forward`](attention.md#BaseScaleQK.forward)

### `ScalePosition`  ·  implements/extends Enum
- def: [`axlearn/common/attention.py:2628`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2628)
- signature: `class ScalePosition(Enum):`
- members:
  - `LOGIT` — [`L2630`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2630)
  - `QUERY` — [`L2632`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2632)
- uses (calls/refs, reference-scoped): [`config_class`](config.md#config_class)
- used by: `test_per_dim_scale`, [`default_config`](attention.md#MultiheadAttentionXL.default_config), [`_compute_logits`](attention.md#MultiheadAttentionXL._compute_logits), [`_scale_qk`](attention.md#MultiheadAttentionXL._scale_qk), [`scale_position`](attention.md#MultiheadAttentionXL.Config.scale_position)

### `ScaleQuery`  ·  implements/extends BaseScaleQK
- def: [`axlearn/common/attention.py:1467`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1467)
- doc: Default implementation for scaling projected queries.
- signature: `class ScaleQuery(BaseScaleQK):`
- members:
  - `apply_norm(self, proj: Tensor)` — [`L1495`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1495) — Applies the norm to projected queries if configured.
  - `apply_per_dim_scale(self, proj: Tensor)` — [`L1501`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1501) — Applies the per-dim scale to projected queries if configured.
  - `apply_scale_factor(self, proj: Tensor)` — [`L1509`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1509) — Applies the scale-factor to projected queries.
  - `default_scale_factor_config()` — [`L1522`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1522) — The config for the default function used to compute the query scale.
  - `forward(self, proj: Tensor, *, positions: Optional[Tensor])` — [`L1514`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1514) — Scales the projected queries.
- protocol/private: `__init__`[`L1483`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1483), `_scale_factor`[`L1486`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1486)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`InstantiableConfig`](config.md#InstantiableConfig), [`config_for_function`](config.md#config_for_function), [`__init__`](base_layer.md#BaseLayer.__init__), [`instantiate`](config.md#InstantiableConfig.instantiate), [`children`](module.md#Module.children), [`BaseScaleQK`](attention.md#BaseScaleQK), [`ScaleFn`](attention.md#ScaleFn), [`Config`](attention.md#ScaleQuery.Config), [`per_head_dim`](attention.md#BaseScaleQK.Config.per_head_dim), [`per_dim_scale`](attention.md#ScaleQuery.Config.per_dim_scale), [`scale_factor`](attention.md#ScaleQuery.Config.scale_factor), [`pow_scale_fn`](attention.md#pow_scale_fn), [`norm`](attention.md#ScaleQuery.Config.norm)
- used by: `test_gqa_forward`, `test_all_mask`, `test_extend_step`, `test_sigmoid_compute_attention`, `test_gqa_extend_step`, [`__init__`](attention.md#RoFormerQKVLinear.__init__), `test_per_dim_scale`, `test_roformer_qkv_linear`, `test_gqa_prefill_states`, `test_input_linear_variants`, `test_prefill_states`, `_scale_query_kwargs`, `_scale_kwargs`, `test_data_types`, [`BaseScaleQK`](attention.md#BaseScaleQK), [`query_scale`](attention.md#MultiheadAttention.Config.query_scale), [`forward`](attention.md#BaseScaleQK.forward)

### `SigmoidAttention`  ·  implements/extends MultiheadAttention
- def: [`axlearn/common/attention.py:2446`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2446)
- doc: A multi-head sigmoid-based attention layer, instead of softmax.
- signature: `class SigmoidAttention(MultiheadAttention):`
- members:
  - `_compute_attention(self, *, mode: ForwardMode, q_proj: Tensor, kv_state: KVState, attention_logit_biases: BaseAttentionBias)` — [`L2461`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2461) — See `MultiheadAttention._compute_attention` for details.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`BaseAttentionBias`](attention_bias.md#BaseAttentionBias), [`ForwardMode`](attention.md#ForwardMode), [`MultiheadAttention`](attention.md#MultiheadAttention), [`KVState`](kv_cache/base_kv_cache.md#KVState), [`vlog`](module.md#Module.vlog), [`value`](attention_bias.md#BaseAttentionBias.value), [`_remat_name`](base_layer.md#BaseLayer._remat_name), [`_compute_logits`](attention.md#MultiheadAttention._compute_logits), [`sigmoid_with_biases`](attention.md#sigmoid_with_biases), [`_compute_context`](attention.md#MultiheadAttention._compute_context), [`_cap_logits`](attention.md#MultiheadAttention._cap_logits)
- used by: [`MultiheadAttention`](attention.md#MultiheadAttention), `test_causal`, `test_sliding_window`, `test_sigmoid_compute_attention`, [`_compute_attention`](attention.md#MultiheadAttention._compute_attention)

### `SinusoidalPositionalEmbedding`  ·  implements/extends BaseLayer
- def: [`axlearn/common/attention.py:437`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L437)
- doc: Sinusoidal positional embeddings.
- signature: `class SinusoidalPositionalEmbedding(BaseLayer):`
- members:
  - `forward(self, positions: Tensor)` — [`L451`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L451) — Looks up positional embeddings by positions.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`BaseLayer`](base_layer.md#BaseLayer), [`Config`](attention.md#SinusoidalPositionalEmbedding.Config), [`dim`](attention.md#SinusoidalPositionalEmbedding.Config.dim), [`sinusoidal_positional_embeddings`](attention.md#sinusoidal_positional_embeddings), [`max_timescale`](attention.md#SinusoidalPositionalEmbedding.Config.max_timescale), [`min_timescale`](attention.md#SinusoidalPositionalEmbedding.Config.min_timescale)
- used by: [`BaseLayer`](base_layer.md#BaseLayer), [`relative_pos_emb`](attention.md#MultiheadAttentionXL.Config.relative_pos_emb)

### `StackedTransformerLayer`  ·  implements/extends BaseStackedTransformerLayer
- def: [`axlearn/common/attention.py:3972`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3972)
- doc: A simple implementation of BaseStackedTransformerLayer.
- signature: `class StackedTransformerLayer(BaseStackedTransformerLayer):`
- members:
  - `_aggregate_layer_outputs(self, layer_outputs: Sequence[BaseTransformerLayer.Output])` — [`L4132`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4132) — Aggregates outputs from the stack.
  - `_forward_for_mode(self, *, mode: ForwardMode, data: Tensor, cached_states: Optional[Nested[Tensor]] = None, **layer_kwargs)` — [`L4033`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4033) — Computes transformer stack outputs.
  - `_update_layer_kwargs(self, layer_kwargs: dict[str, Any], *, all_layer_outputs: list[BaseTransformerLayer.Output], external_self_attention_kv_state: Optional[KVState] = None)` — [`L4111`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4111) — Updates `layer_kwargs` using other args.
  - `extend_step(self, cached_states: list[NestedTensor], data: Tensor, *, is_prefill: bool = False, **layer_kwargs)` — [`L4159`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4159)
  - `forward(self, data: Tensor, **layer_kwargs)` — [`L4146`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4146)
  - `init_states(self, *, batch_size: int, max_len: int, dtype: jnp.dtype)` — [`L4093`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4093) — Initializes cache for autoregressive cached decoding.
  - `initialize_parameters_recursively(self, prng_key: Tensor, *, prebuilt: Optional[Nested[Optional[ParameterSpec]]] = None)` — [`L4019`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4019)
- protocol/private: `__init__`[`L3988`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3988), `_layers`[`L4002`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4002), `_update_data`[`L3991`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3991)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`REQUIRED`](config.md#REQUIRED), [`_add_child`](module.md#Module._add_child), [`ParameterSpec`](base_layer.md#ParameterSpec), [`ForwardMode`](attention.md#ForwardMode), [`maybe_instantiate`](config.md#maybe_instantiate), [`KVState`](kv_cache/base_kv_cache.md#KVState), [`__init__`](base_layer.md#BaseLayer.__init__), [`BaseTransformerLayer`](attention.md#BaseTransformerLayer), [`TransformerLayer`](attention.md#TransformerLayer), [`Output`](attention.md#BaseTransformerLayer.Output), [`nowrap`](module.md#nowrap), [`PREFILL`](attention.md#ForwardMode.PREFILL), [`EXTEND_STEP`](attention.md#ForwardMode.EXTEND_STEP), [`input_dim`](attention.md#BaseTransformerLayer.Config.input_dim), [`BaseStackedTransformerLayer`](attention.md#BaseStackedTransformerLayer), [`FORWARD`](attention.md#ForwardMode.FORWARD), [`split_prng_key`](utils.md#split_prng_key), [`clone`](config.md#ConfigBase.clone), [`num_layers`](attention.md#BaseStackedTransformerLayer.Config.num_layers), [`data`](attention.md#BaseTransformerLayer.Output.data), `_aggregate_layer_outputs`, [`layer`](attention.md#StackedTransformerLayer.Config.layer), [`self_attention_kv_state`](attention.md#BaseTransformerLayer.Output.self_attention_kv_state), [`get_or_none`](utils.md#get_or_none), `_update_layer_kwargs`, [`Config`](attention.md#StackedTransformerLayer.Config), [`data_merger`](attention.md#StackedTransformerLayer.Config.data_merger), `_aggregate_layer_outputs`, `NonUniformStack`, `_StackedTransformerLayerWithSkipConnection`, [`get_stochastic_depth_linear_rate`](layers.md#get_stochastic_depth_linear_rate), [`peak_stochastic_depth_rate`](attention.md#BaseStackedTransformerLayer.Config.peak_stochastic_depth_rate)
- used by: [`initialize_parameters_recursively`](base_layer.md#BaseLayer.initialize_parameters_recursively), `_compare_layers`, `test_update_layer_kwargs`, `test_stacked_transformer_with_non_uniform_layers`, `test_stack_vs_pipeline`, `test_transformer_extend_step`, [`build_remat_spec`](attention.md#build_remat_spec), `test_passthrough_update_layer_kwargs`, `_convert_from_stacked_params`, `test_transformer_prefill_states`, `test_skip_connection`, [`BaseStackedTransformerLayer`](attention.md#BaseStackedTransformerLayer), [`extend_step`](attention.md#BaseTransformerLayer.extend_step), [`forward`](attention.md#BaseTransformerLayer.forward), `test_kv_sharing`, `test_qlinear_kv_sharing`, [`init_states`](attention.md#BaseTransformerLayer.init_states), `test_stacked_transformer_with_seq_layer_cfgs`, `test_stack_vs_repeat_remat_everything_saveable`, `test_stochastic_depth`, [`stage`](attention.md#PipelinedTransformerLayer.Config.stage), [`transformer`](decoder.md#Decoder.Config.transformer), `test_nested_stacked_decoding`, `test_stack_vs_repeat_with_build_remat_spec`, `test_stacked_decoding`, `test_stack_vs_repeat`, `NonUniformStack`, `_StackedTransformerLayerWithSkipConnection`

### `SymmetricALiBiAttentionLogitBiasLayer`  ·  implements/extends FullAttentionLogitBiasLayer
- def: [`axlearn/common/attention.py:4719`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4719)
- doc: Symmetric full attention version of ALiBiAttentionLogitBiasLayer.
- signature: `class SymmetricALiBiAttentionLogitBiasLayer(FullAttentionLogitBiasLayer):`
- members:
  - `forward(self, *, segment_ids: Tensor, positions: Tensor)` — [`L4737`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4737)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`apply_attention_logit_biases`](attention.md#apply_attention_logit_biases), [`alibi_get_slopes`](attention.md#alibi_get_slopes), [`forward`](attention.md#FullAttentionLogitBiasLayer.forward), [`FullAttentionLogitBiasLayer`](attention.md#FullAttentionLogitBiasLayer)
- used by: `test_alibi_attention_mask`, [`forward`](attention.md#FullAttentionLogitBiasLayer.forward), [`FullAttentionLogitBiasLayer`](attention.md#FullAttentionLogitBiasLayer)

### `TransformerAttentionLayer`  ·  implements/extends BaseLayer
- def: [`axlearn/common/attention.py:2784`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2784)
- doc: A Transformer attention layer with normalization and a skip connection.
- signature: `class TransformerAttentionLayer(BaseLayer):`
- members:
  - `_forward_for_mode(self, *, mode: ForwardMode, target: Tensor, source: Optional[Union[Tensor, KVState]] = None, attention_logit_biases: Optional[Tensor] = None, segment_ids: Optional[Tensor] = None, target_positions: Optional[Tensor] = None, cached_states: Optional[NestedTensor] = None, return_aux: Optional[set[str]] = None, page_pool: Optional[Nested[Tensor]] = None)` — [`L2879`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2879) — Computes either self-attention or cross-attention for the given target and source.
  - `attention_thunk(target: Tensor)` — [`L2930`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2930)
  - `extend_step(self, cached_states: NestedTensor, target: Tensor, *, is_prefill: bool = False, source: Optional[Union[Tensor, KVState]] = None, segment_ids: Optional[Tensor] = None, attention_logit_biases: Optional[Tensor] = None, return_aux: Optional[set[str]] = None, page_pool: Optional[Nested[Tensor]] = None)` — [`L3059`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3059) — Computes the value vector given the query of the current step.
  - `forward(self, *, target: Tensor, source: Optional[Union[Tensor, KVState]] = None, attention_logit_biases: Optional[Tensor] = None, segment_ids: Optional[Tensor] = None, target_positions: Optional[Tensor] = None, return_aux: Optional[set[str]] = None)` — [`L3001`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3001) — Computes attention with target as query and source as key and value.
  - `init_states(self, *, batch_size: int, max_len: int, dtype: jnp.dtype)` — [`L3042`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3042) — Initializes cache for autoregressive cached decoding.
- protocol/private: `__init__`[`L2821`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2821), `_create_layer_parameter_specs`[`L2852`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2852)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`BaseLayer`](base_layer.md#BaseLayer), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`ParameterSpec`](base_layer.md#ParameterSpec), [`InstantiableConfig`](config.md#InstantiableConfig), [`ForwardMode`](attention.md#ForwardMode), [`shape`](utils.md#TensorSpec.shape), [`dtype`](utils.md#TensorSpec.dtype), [`KVState`](kv_cache/base_kv_cache.md#KVState), [`__init__`](base_layer.md#BaseLayer.__init__), [`parameters`](base_layer.md#BaseLayer.parameters), [`add_summary`](module.md#Module.add_summary), [`nowrap`](module.md#nowrap), [`PREFILL`](attention.md#ForwardMode.PREFILL), [`EXTEND_STEP`](attention.md#ForwardMode.EXTEND_STEP), [`FORWARD`](attention.md#ForwardMode.FORWARD), [`constant_initializer`](param_init.md#constant_initializer), [`initializer`](base_layer.md#ParameterSpec.initializer), [`NormPosition`](attention.md#NormPosition), [`weight_decay_scale`](base_layer.md#ParameterSpec.weight_decay_scale), [`norm`](attention.md#TransformerAttentionLayer.Config.norm), [`Config`](attention.md#TransformerAttentionLayer.Config), [`attention`](attention.md#TransformerAttentionLayer.Config.attention), [`target_dim`](attention.md#TransformerAttentionLayer.Config.target_dim), [`dropout`](attention.md#TransformerAttentionLayer.Config.dropout), [`stochastic_depth`](attention.md#TransformerAttentionLayer.Config.stochastic_depth), [`IN_NORM`](attention.md#NormPosition.IN_NORM), [`source_dim`](attention.md#TransformerAttentionLayer.Config.source_dim), [`Output`](attention.md#TransformerAttentionLayer.Output), [`OUT_NORM`](attention.md#NormPosition.OUT_NORM), [`RES_NORM`](attention.md#NormPosition.RES_NORM), [`structure`](attention.md#TransformerAttentionLayer.Config.structure), [`data`](attention.md#TransformerAttentionLayer.Output.data), [`kv_state`](attention.md#TransformerAttentionLayer.Output.kv_state), [`probs`](attention.md#TransformerAttentionLayer.Output.probs)
- used by: [`BaseLayer`](base_layer.md#BaseLayer), [`_create_layer_parameter_specs`](base_layer.md#BaseLayer._create_layer_parameter_specs), `test_transformer_extend_step`, `test_per_dim_scale`, `test_v2_structure`, `test_residual_gate`, `test_forward_vs_extend_step`, `test_set_double_shard_weights_config`, `test_transformer_prefill_states`, `test_set_double_shard_weights_config_for_list_of_configs`, `test_multihead_attention_xl`, `test_rope_self_attention`, [`self_attention`](attention.md#TransformerLayer.Config.self_attention), `test_against_roberta_attention`

### `TransformerFeedForwardLayer`  ·  implements/extends BaseLayer
- def: [`axlearn/common/attention.py:3119`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3119)
- doc: A Transformer feed-forward layer.
- signature: `class TransformerFeedForwardLayer(BaseLayer):`
- members:
  - `_get_activation(self, x: Tensor, activation_fn_name: str)` — [`L3383`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3383) — Applies activation function on 'x' and optionally counts the number of dead neurons.
  - `_linear2(x)` — [`L3277`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3277) — Applies linear2, optionally logging RMS norm of the output.
  - `forward(self, inputs: Tensor)` — [`L3274`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3274)
- protocol/private: `__init__`[`L3202`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3202), `_create_layer_parameter_specs`[`L3262`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3262), `_linear1_activation`[`L3360`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3360)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`BaseLayer`](base_layer.md#BaseLayer), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`ParameterSpec`](base_layer.md#ParameterSpec), [`InstantiableConfig`](config.md#InstantiableConfig), [`shape`](utils.md#TensorSpec.shape), [`dtype`](utils.md#TensorSpec.dtype), [`__init__`](base_layer.md#BaseLayer.__init__), [`parameters`](base_layer.md#BaseLayer.parameters), [`add_summary`](module.md#Module.add_summary), [`_add_tensor_stats`](base_layer.md#BaseLayer._add_tensor_stats), [`children`](module.md#Module.children), [`_remat_name`](base_layer.md#BaseLayer._remat_name), [`constant_initializer`](param_init.md#constant_initializer), [`initializer`](base_layer.md#ParameterSpec.initializer), [`NormPosition`](attention.md#NormPosition), [`weight_decay_scale`](base_layer.md#ParameterSpec.weight_decay_scale), [`norm`](attention.md#TransformerFeedForwardLayer.Config.norm), [`instantiate`](config.md#FunctionConfigBase.instantiate), [`Config`](attention.md#TransformerFeedForwardLayer.Config), [`input_dim`](attention.md#TransformerFeedForwardLayer.Config.input_dim), [`linear1`](attention.md#TransformerFeedForwardLayer.Config.linear1), [`dropout`](attention.md#TransformerFeedForwardLayer.Config.dropout), [`hidden_dim`](attention.md#TransformerFeedForwardLayer.Config.hidden_dim), [`linear2`](attention.md#TransformerFeedForwardLayer.Config.linear2), [`stochastic_depth`](attention.md#TransformerFeedForwardLayer.Config.stochastic_depth), [`IN_NORM`](attention.md#NormPosition.IN_NORM), [`structure`](attention.md#TransformerFeedForwardLayer.Config.structure), [`get_activation_fn`](layers.md#get_activation_fn), [`activation`](attention.md#TransformerFeedForwardLayer.Config.activation), [`OUT_NORM`](attention.md#NormPosition.OUT_NORM), [`RES_NORM`](attention.md#NormPosition.RES_NORM), [`add_value_rms_norm_summary`](attention.md#TransformerFeedForwardLayer.Config.add_value_rms_norm_summary)
- used by: [`BaseLayer`](base_layer.md#BaseLayer), [`_create_layer_parameter_specs`](base_layer.md#BaseLayer._create_layer_parameter_specs), `test_v2_structure`, [`set_double_shard_weights_config`](attention.md#set_double_shard_weights_config), `test_add_value_rms_norm_summary`, `test_linear_remat`, `test_residual_gate`, [`feed_forward`](attention.md#TransformerLayer.Config.feed_forward), `test_add_dead_neuron_summary`, [`set_feed_forward_partition_specs`](attention.md#set_feed_forward_partition_specs), [`adapter`](attention.md#BottleNeckAdapterTransformerLayer.Config.adapter), [`feed_forward`](attention.md#ParallelTransformerLayer.Config.feed_forward), `test_set_ffn_partition_specs`

### `TransformerLayer`  ·  implements/extends BaseTransformerLayer
- def: [`axlearn/common/attention.py:3418`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3418)
- doc: A Transformer layer.
- signature: `class TransformerLayer(BaseTransformerLayer):`
- members:
  - `_forward_for_mode(self, *, mode: ForwardMode, data: Tensor, self_attention_kv_state: Optional[KVState] = None, self_attention_logit_biases: Optional[Tensor] = None, cross_attention_data: Optional[Tensor] = None, cross_attention_logit_biases: Optional[Tensor] = None, target_segment_ids: Optional[Tensor] = None, target_positions: Optional[Tensor] = None, cached_states: Optional[NestedTensor] = None, return_aux: Optional[set[str]] = None, page_pool: Optional[Nested[Tensor]] = None)` — [`L3446`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3446) — Computes transformer layer outputs and self/cross-attention probabilities.
  - `extend_step(self, cached_states: NestedTensor, data: Tensor, *, is_prefill: bool = False, **kwargs)` — [`L3583`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3583)
  - `forward(self, data: Tensor, **kwargs)` — [`L3555`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3555)
  - `init_states(self, *, batch_size: int, max_len: int, dtype: jnp.dtype)` — [`L3566`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3566) — Initializes cache for autoregressive cached decoding.
- protocol/private: `__init__`[`L3435`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3435)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`ForwardMode`](attention.md#ForwardMode), [`KVState`](kv_cache/base_kv_cache.md#KVState), [`__init__`](base_layer.md#BaseLayer.__init__), [`BaseTransformerLayer`](attention.md#BaseTransformerLayer), [`vlog`](module.md#Module.vlog), [`Output`](attention.md#BaseTransformerLayer.Output), [`nowrap`](module.md#nowrap), [`PREFILL`](attention.md#ForwardMode.PREFILL), [`EXTEND_STEP`](attention.md#ForwardMode.EXTEND_STEP), [`input_dim`](attention.md#BaseTransformerLayer.Config.input_dim), `forward`, [`FORWARD`](attention.md#ForwardMode.FORWARD), [`_remat_name`](base_layer.md#BaseLayer._remat_name), [`self_attention`](attention.md#TransformerLayer.Config.self_attention), [`Config`](attention.md#TransformerLayer.Config), [`data`](attention.md#BaseTransformerLayer.Output.data), [`feed_forward`](attention.md#TransformerLayer.Config.feed_forward), [`self_attention_kv_state`](attention.md#BaseTransformerLayer.Output.self_attention_kv_state), [`add_module_output`](module.md#Module.add_module_output), [`cross_attention`](attention.md#TransformerLayer.Config.cross_attention), [`self_attention_probs`](attention.md#BaseTransformerLayer.Output.self_attention_probs), [`cross_attention_probs`](attention.md#BaseTransformerLayer.Output.cross_attention_probs), `DummyTransformerLayer`
- used by: [`_forward_for_mode`](attention.md#_TransformerRepeat._forward_for_mode), [`BaseTransformerLayer`](attention.md#BaseTransformerLayer), `test_build_remat_spec_neuron`, `test_update_layer_kwargs`, `test_stacked_transformer_with_non_uniform_layers`, `test_transformer_extend_step`, [`forward`](attention.md#_TransformerPipeline.forward), `test_passthrough_update_layer_kwargs`, `test_build_remat_spec`, `test_with_golden_value`, [`_forward_for_mode`](attention.md#StackedTransformerLayer._forward_for_mode), `test_set_double_shard_weights_config`, `test_self_attention_kv_state`, `test_transformer_prefill_states`, [`set_double_shard_weights_config`](attention.md#set_double_shard_weights_config), [`__init__`](attention.md#ParallelTransformerLayer.__init__), `test_set_double_shard_weights_config_for_list_of_configs`, `test_skip_connection`, [`extend_step`](attention.md#BaseTransformerLayer.extend_step), [`extend_step`](attention.md#StackedTransformerLayer.extend_step), `forward`, [`forward`](attention.md#BaseTransformerLayer.forward), `test_kv_sharing`, `test_qlinear_kv_sharing`, [`init_states`](attention.md#BaseTransformerLayer.init_states), [`extend_step`](attention.md#_TransformerRepeat.extend_step), [`layer`](attention.md#BaseStackedTransformerLayer.Config.layer), `test_against_roberta_layer`, `test_forward`, `test_stacked_transformer_with_seq_layer_cfgs`, [`forward`](attention.md#StackedTransformerLayer.forward), [`forward`](attention.md#_TransformerRepeat.forward), `_stack_config`, [`layer`](attention.md#BottleNeckAdapterTransformerLayer.Config.layer), [`layer`](attention.md#StackedTransformerLayer.Config.layer), [`extend_step`](attention.md#RepeatedTransformerLayer.extend_step), [`layer_fn`](attention.md#_TransformerPipeline.layer_fn), [`forward`](attention.md#PipelinedTransformerLayer.forward), [`forward`](attention.md#RepeatedTransformerLayer.forward), `test_decoding`  (+1 more)

### `UpdateDataFn`  ·  implements/extends Protocol
- def: [`axlearn/common/attention.py:3933`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3933)
- doc: A function for updating the constituent layers' input in a StackTransformerLayer.
- signature: `class UpdateDataFn(Protocol):`
- members:
  - `__call__(self, data: Tensor, all_layer_outputs: list[BaseTransformerLayer.Output])` — [`L3936`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3936) — Returns a new Tensor with the same shape as `data`, reflecting some desired updates.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`BaseTransformerLayer`](attention.md#BaseTransformerLayer), [`Output`](attention.md#BaseTransformerLayer.Output)
- used by: [`update_data_with_skip_connection`](attention.md#update_data_with_skip_connection), [`data_merger`](attention.md#StackedTransformerLayer.Config.data_merger)

### `YaRNSinusoidalPositionalEmbedding`  ·  implements/extends RoFormerSinusoidalPositionalEmbedding
- def: [`axlearn/common/attention.py:1183`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1183)
- doc: Scale RoPE with Scaling with YaRN.
- signature: `class YaRNSinusoidalPositionalEmbedding(RoFormerSinusoidalPositionalEmbedding):`
- members:
  - `build_rotary_sinusoidal_positional_embeddings(self, positions: Tensor, inv_freq: Tensor, mscale: float)` — [`L1258`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1258) — Generate the sin/cos positional embedding from prebuilt inverse frequencies.
  - `compute_rope_params(self)` — [`L1211`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1211) — Generates the inverse frequencies and scaling factors for the YaRN RoPE.
  - `forward(self, *, positions: Optional[Tensor] = None, max_seq_len: Optional[int] = None)` — [`L1277`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1277) — Computes the rotary sinusoidal positional embeddings with YaRN.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`RoFormerSinusoidalPositionalEmbedding`](attention.md#RoFormerSinusoidalPositionalEmbedding), [`Config`](attention.md#YaRNSinusoidalPositionalEmbedding.Config), [`original_max_seq_length`](attention.md#YaRNSinusoidalPositionalEmbedding.Config.original_max_seq_length), [`dim`](attention.md#RoFormerSinusoidalPositionalEmbedding.Config.dim), [`default_query_positions`](attention.md#RoFormerSinusoidalPositionalEmbedding.default_query_positions), [`_find_correction_dim`](attention.md#_find_correction_dim), [`scaling_factor`](attention.md#YaRNSinusoidalPositionalEmbedding.Config.scaling_factor), [`_linear_ramp_mask`](attention.md#_linear_ramp_mask), [`attn_factor`](attention.md#YaRNSinusoidalPositionalEmbedding.Config.attn_factor), [`beta_fast`](attention.md#YaRNSinusoidalPositionalEmbedding.Config.beta_fast), [`beta_slow`](attention.md#YaRNSinusoidalPositionalEmbedding.Config.beta_slow), [`extrapolation_factor`](attention.md#YaRNSinusoidalPositionalEmbedding.Config.extrapolation_factor), [`theta`](attention.md#RoFormerSinusoidalPositionalEmbedding.Config.theta)
- used by: [`RoFormerSinusoidalPositionalEmbedding`](attention.md#RoFormerSinusoidalPositionalEmbedding), [`forward`](attention.md#RoFormerSinusoidalPositionalEmbedding.forward), `test_yarn_emb_basic`, `test_yarn_emb_extend`

### `_TransformerPipeline`  ·  implements/extends Pipeline
- def: [`axlearn/common/attention.py:4401`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4401)
- doc: Transformer pipeline layer.
- signature: `class _TransformerPipeline(Pipeline):`
- members:
  - `forward(self, data: Tensor, *, return_aux: Optional[set[str]] = None, **kwargs)` — [`L4404`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4404)
  - `layer_fn(carry, _)` — [`L4420`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4420)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`_run`](pipeline.md#Pipeline._run), [`TransformerLayer`](attention.md#TransformerLayer), [`vlog`](module.md#Module.vlog), [`shapes`](utils.md#shapes), [`Output`](attention.md#BaseTransformerLayer.Output), [`data`](attention.md#BaseTransformerLayer.Output.data), [`Pipeline`](pipeline.md#Pipeline), [`_to_microbatches`](pipeline.md#Pipeline._to_microbatches), [`carry`](pipeline.md#Pipeline.Output.carry), [`ys`](pipeline.md#Pipeline.Output.ys), [`_from_microbatches`](pipeline.md#Pipeline._from_microbatches), [`Output`](pipeline.md#Pipeline.Output)
- used by: [`Pipeline`](pipeline.md#Pipeline), [`pipeline`](attention.md#PipelinedTransformerLayer.Config.pipeline)

### `_TransformerRepeat`  ·  implements/extends Repeat
- def: [`axlearn/common/attention.py:4176`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4176)
- doc: A Repeat layer with layer=TransformerLayer.
- signature: `class _TransformerRepeat(Repeat):`
- members:
  - `_forward_for_mode(self, *, mode: ForwardMode, data: Tensor, cached_states: Optional[Nested[Tensor]] = None, **layer_kwargs)` — [`L4192`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4192) — Computes transformer stack outputs.
  - `extend_step(self, cached_states: NestedTensor, data: Tensor, *, is_prefill: bool = False, **layer_kwargs)` — [`L4311`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4311)
  - `forward(self, data: Tensor, **layer_kwargs)` — [`L4279`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4279)
  - `init_states(self, *, batch_size: int, max_len: int, dtype: jnp.dtype)` — [`L4293`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4293) — Initializes cache for autoregressive cached decoding.
  - `layer_fn(carry, x_i)` — [`L4223`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4223)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`ForwardMode`](attention.md#ForwardMode), [`flatten_items`](utils.md#flatten_items), [`KVState`](kv_cache/base_kv_cache.md#KVState), [`TransformerLayer`](attention.md#TransformerLayer), [`shapes`](utils.md#shapes), [`Output`](attention.md#BaseTransformerLayer.Output), [`_run`](repeat.md#Repeat._run), [`nowrap`](module.md#nowrap), [`PREFILL`](attention.md#ForwardMode.PREFILL), [`k_proj`](kv_cache/base_kv_cache.md#KVState.k_proj), [`v_proj`](kv_cache/base_kv_cache.md#KVState.v_proj), [`EXTEND_STEP`](attention.md#ForwardMode.EXTEND_STEP), [`FORWARD`](attention.md#ForwardMode.FORWARD), [`key_positions`](kv_cache/base_kv_cache.md#KVState.key_positions), [`Repeat`](repeat.md#Repeat), [`Config`](attention.md#_TransformerRepeat.Config), [`num_layers`](repeat.md#Repeat.Config.num_layers), [`carry`](repeat.md#Repeat.Output.carry), [`ys`](repeat.md#Repeat.Output.ys), [`Output`](repeat.md#Repeat.Output), [`carry`](attention.md#_TransformerRepeat.Config.carry)
- used by: [`repeat`](attention.md#RepeatedTransformerLayer.Config.repeat), [`Repeat`](repeat.md#Repeat)

## Functions
- `_find_correction_dim(num_rotations, dim, theta=10000, max_position_embeddings=2048)` — [`L1168`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1168)
- `_linear_ramp_mask(low, high, dim)` — [`L1174`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1174)
- `_next_power_of_two(n: float)` — [`L3677`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3677)
- `_rotary_sinusoidal_positional_embeddings(*, positions: Tensor, dim: int, theta: float = 10000)` — [`L1078`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1078) — Generate the sin/cos positional embedding.
- `_save_and_offload_only_these_names_regex(*, names_which_can_be_saved: SavePattern, names_which_can_be_offloaded: SavePattern, offload_src: str, offload_dst: str)` — [`L4492`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4492)
- `alibi_get_slopes(num_heads: int)` — [`L4653`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4653) — Get the slopes for different attention heads defined in ALiBi paper.
- `apply_attention_logit_biases(logits: Tensor, attention_logit_biases: Optional[Tensor] = None)` — [`L567`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L567) — Applies `attention_logit_biases` on `logits`.
- `apply_rotary_position_embeddings(*, query: Tensor, key: Tensor, value: Tensor, sinusoidal_pos: Tensor, rotary_key: bool, rotary_value: bool)` — [`L1316`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1316) — This is a jax implementation (a copy) of the RoPE apply_rotary_position_embeddings.
- `build_remat_spec(stack_cfg: Union[BaseStackedTransformerLayer.Config, RepeatedConformerLayer.Config,], *, save_pattern: SavePattern = RematRegexSavePatterns.NATIVE_ATTENTION.value, offload_pattern: SavePattern = None, offload_dst: str = "pinned_host")` — [`L4528`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4528) — Configures how the Transformer or Conformer stack will save the linearization points.
- `compute_gqa_context(probs: Tensor, v_proj: Tensor)` — [`L2361`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2361) — Compute attention context.
- `compute_gqa_logits(q_proj: Tensor, k_proj: Tensor)` — [`L2334`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2334) — Compute attention logits.
- `compute_padding_biases(input_ids: Tensor, *, pad_token_id: Optional[int])` — [`L4615`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4615) — Compute the logits bias to disable attention to/from paddings.
- `constant_function(per_head_dim: int)` — [`L1416`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1416)
- `constant_scale_fn(value: float)` — [`L1403`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1403) — A constant scale function for `MultiheadAttention`.
- `get_slopes_power_of_2(n: int)` — [`L4668`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L4668)
- `pow_scale_fn(exp: float)` — [`L1423`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1423) — A scale function for `MultiheadAttention` that computes `per_head_dim ** exp`.
- `rel_pos_to_abs_pos(x: Tensor)` — [`L2497`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2497) — Converts a (T, relative_pos_offset) Tensor to a (T, abs_position) tensor.
- `rotate_half(x)` — [`L1352`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1352)
- `scale_fn(input_dim: int, *, scale: float)` — [`L3113`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3113)
- `scaled_hidden_dim(scale: float = 4)` — [`L3112`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3112)
- `set_attention_partition_specs(cfg: MultiheadAttention.Config, *, batch_axis_names: Union[str, Sequence[str]] = ("data", "fsdp"), fsdp_axis_names: Union[str, Sequence[str]] = "fsdp", tp_axis_names: Union[str, Sequence[str]] = "model", seq_axis_names: Union[str, Sequence[str]] = "seq", set_attn_activation_specs: bool = False)` — [`L3808`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3808) — Sets `cfg` to shard attention weights over both fsdp and tp axes.
- `set_double_shard_weights_config(cfg: Union[TransformerLayer.Config, Sequence[TransformerLayer.Config]], *, batch_axis_names: Union[str, Sequence[str]] = ("data", "fsdp"), fsdp_axis_names: Union[str, Sequence[str]] = "fsdp", tp_axis_names: Union[str, Sequence[str]] = "model", seq_axis_names: Union[str, Sequence[str]] = "seq", set_attn_activation_specs: bool = False)` — [`L3858`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3858) — Sets `cfg` to shard FFN and attention weights over both fsdp and tp axes.
- `set_feed_forward_partition_specs(cfg: TransformerFeedForwardLayer.Config, *, batch_axis_names: Union[str, Sequence[str]] = ("data", "fsdp"), fsdp_axis_names: Union[str, Sequence[str]] = "fsdp", tp_axis_names: Union[str, Sequence[str]] = "model", seq_axis_names: Union[str, Sequence[str]] = "seq")` — [`L3836`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3836) — Sets `cfg` to shard FFN weights over both fsdp and tp axes.
- `sigmoid_with_biases(logits: Tensor, attention_logit_biases: Optional[Tensor] = None)` — [`L636`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L636) — Computes sigmoid with optional masking.
- `sinusoidal_positional_embeddings(positions: Tensor, *, dim: int, min_timescale: float = 1, max_timescale: float = 10000)` — [`L387`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L387) — Sinusoidal positional embeddings.
- `softmax_with_biases(logits: Tensor, attention_logit_biases: Optional[Tensor] = None, logit_sink: Optional[Tensor] = None)` — [`L584`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L584) — Computes softmax with optional masking.
- `update_data(data: Tensor, all_layer_outputs: list[BaseTransformerLayer.Output])` — [`L3963`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3963)
- `update_data_with_skip_connection(skip_connections: dict[int, int])` — [`L3951`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L3951) — Creates a function that adds skip connection to the input data tensor.
- `xl_attention_logits(q_proj: Tensor, k_proj: Tensor, relative_pos_emb: Tensor, u: Tensor, v: Tensor)` — [`L2574`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L2574) — Computes Transformer XL self-attention logits.

## Module values
- `ScaleFn` — [`L1400`](../../../../../../raw/code/axlearn/axlearn/common/attention.py#L1400)

