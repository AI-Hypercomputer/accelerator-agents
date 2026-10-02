---
title: 'Module: axlearn/common/rattention/rattention.py'
type: catalog
provenance: extracted
module: axlearn/common/rattention/rattention.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.rattention.rattention`/
symbols:
  RAttention._mask_tpl: RAttention#_mask_tpl.
  RAttention._forward_for_mode: RAttention#_forward_for_mode().
  ResidualLinearAttention._create_layer_parameter_specs: ResidualLinearAttention#_create_layer_parameter_specs().
  ResidualLinearAttention._forward_for_mode: ResidualLinearAttention#_forward_for_mode().
  RAttention.init_states: RAttention#init_states().
  RAttention.extend_step: RAttention#extend_step().
  ResidualLinearAttention.__init__: ResidualLinearAttention#__init__().
  ResidualLinearAttention.extend_step: ResidualLinearAttention#extend_step().
  RAttention.__init__: RAttention#__init__().
  ResidualLinearAttention._get_linear_attention_impl: ResidualLinearAttention#_get_linear_attention_impl().
  ResidualLinearAttention._compute_init_state: ResidualLinearAttention#_compute_init_state().
  ResidualLinearAttention.forward: ResidualLinearAttention#forward().
  ResidualLinearAttention._compute_linear_attention_step: ResidualLinearAttention#_compute_linear_attention_step().
  ResidualLinearAttention._compute_linear_attention_parallel: ResidualLinearAttention#_compute_linear_attention_parallel().
  ResidualLinearAttention._prepare_linear_attention_inputs: ResidualLinearAttention#_prepare_linear_attention_inputs().
  ResidualLinearAttention.Config: ResidualLinearAttention#Config#
  RAttention.Config: RAttention#Config#
  RAttention.Config.residual_la: RAttention#Config#residual_la.
  ResidualLinearAttention.per_head_dim: ResidualLinearAttention#per_head_dim().
  ResidualLinearAttention.init_states: ResidualLinearAttention#init_states().
  RAttention.default_config: RAttention#default_config().
  apply_rotary_position_embeddings: apply_rotary_position_embeddings().
  ResidualLinearAttention._repeat_kv_heads: ResidualLinearAttention#_repeat_kv_heads().
  RAttention.Config.sliding_window_size: RAttention#Config#sliding_window_size.
  ResidualLinearAttention.Config.sliding_window_size: ResidualLinearAttention#Config#sliding_window_size.
  RAttention.Config.mixing_norm: RAttention#Config#mixing_norm.
  ResidualLinearAttention.Config.input_dim: ResidualLinearAttention#Config#input_dim.
  ResidualLinearAttention.Config.hidden_dim: ResidualLinearAttention#Config#hidden_dim.
  ResidualLinearAttention.Config.num_heads: ResidualLinearAttention#Config#num_heads.
  ResidualLinearAttention.Config.num_kv_heads: ResidualLinearAttention#Config#num_kv_heads.
  ResidualLinearAttention.default_config: ResidualLinearAttention#default_config().
  RAttention._repeat_kv_heads: RAttention#_repeat_kv_heads().
  ResidualLinearAttention: ResidualLinearAttention#
  ResidualLinearAttention.Config.feat_fn: ResidualLinearAttention#Config#feat_fn.
  RAttention: RAttention#
  RAttention.Config.rope_theta: RAttention#Config#rope_theta.
  ResidualLinearAttention.Config.chunk_size: ResidualLinearAttention#Config#chunk_size.
  ResidualLinearAttention.Config.use_learned_init: ResidualLinearAttention#Config#use_learned_init.
  ResidualLinearAttention.Config.use_qk_scale: ResidualLinearAttention#Config#use_qk_scale.
  ResidualLinearAttention.Config.dim_to_partition_spec: ResidualLinearAttention#Config#dim_to_partition_spec.
  ResidualLinearAttention.Config.output_partition_spec: ResidualLinearAttention#Config#output_partition_spec.
---
# Module: [`axlearn/common/rattention/rattention.py`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py)

## Classes
### `Config`  ·  implements/extends Config
- def: [`axlearn/common/rattention/rattention.py:487`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L487)
- doc: Config for RAttention layer.
- signature: `class Config(FlashAttention.Config):`
- members:
  - `chunk_size` — [`L101`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L101)
  - `dim_to_partition_spec` — [`L110`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L110)
  - `feat_fn` — [`L100`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L100)
  - `hidden_dim` — [`L94`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L94)
  - `input_dim` — [`L93`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L93)
  - `mixing_norm` — [`L500`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L500)
  - `num_heads` — [`L95`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L95)
  - `num_kv_heads` — [`L96`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L96)
  - `output_partition_spec` — [`L111`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L111)
  - `residual_la` — [`L497`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L497)
  - `rope_theta` — [`L498`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L498)
  - `sliding_window_size` — [`L98`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L98)
  - `sliding_window_size` — [`L496`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L496)
  - `use_learned_init` — [`L105`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L105)
  - `use_qk_scale` — [`L108`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L108)
- uses (calls/refs, reference-scoped): [`config_class`](../config.md#config_class), [`REQUIRED`](../config.md#REQUIRED), [`Required`](../config.md#Required), [`FlashAttention`](../flash_attention/layer.md#FlashAttention), [`BaseNormalizationLayer`](../layers.md#BaseNormalizationLayer), [`FeatureMap`](kernels/utils.md#FeatureMap), [`Config`](../flash_attention/layer.md#FlashAttention.Config), [`Config`](../layers.md#BaseNormalizationLayer.Config), [`Config`](rattention.md#ResidualLinearAttention.Config), [`ResidualLinearAttention`](rattention.md#ResidualLinearAttention)
- used by: [`_mask_tpl`](rattention.md#RAttention._mask_tpl), [`__init__`](rattention.md#RAttention.__init__), [`__init__`](rattention.md#ResidualLinearAttention.__init__)

### `RAttention`  ·  implements/extends FlashAttention
- def: [`axlearn/common/rattention/rattention.py:473`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L473)
- doc: RAttention layer with residual linear attention and sliding window attention.
- signature: `class RAttention(FlashAttention):`
- members:
  - `__init__(self, cfg: Config, *, parent: Module)` — [`L519`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L519) — Initializes the RAttention layer.
  - `_forward_for_mode(self, *, mode: ForwardMode, query: Tensor, key: Optional[Tensor] = None, value: Optional[Tensor] = None, kv_state: Optional[KVState] = None, attention_logit_biases: Union[None, Tensor, BaseAttentionBias] = None, segment_ids: Optional[Tensor] = None, query_positions: Optional[Tensor] = None, cached_states: Optional[NestedTensor] = None, return_aux: Optional[set[str]] = None, page_pool: Optional[Nested[Tensor]] = None)` — [`L599`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L599) — Forward function for RAttention. — documented in [axlearn-common-attention_bias](../../../../concepts/axlearn-common-attention_bias.md)
  - `default_config(cls)` — [`L503`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L503)
  - `extend_step(self, cached_states: NestedTensor, query: Tensor, *, is_prefill: bool = False, segment_ids: Optional[Tensor] = None, kv_state: Optional[KVState] = None, attention_logit_biases: Optional[Tensor] = None, return_aux: Optional[set[str]] = None, **kwargs)` — [`L793`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L793) — Computes incremental outputs.
  - `init_states(self, *, batch_size: int, max_len: int, dtype: jnp.dtype)` — [`L770`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L770) — Initializes decoding cache for SWA and residual LA.
- protocol/private: `_mask_tpl`[`L528`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L528), `_repeat_kv_heads`[`L766`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L766)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`Nested`](../utils.md#Nested), [`default_config`](../config.md#Configurable.default_config), [`config`](../config.md#Configurable.config), [`NestedTensor`](../utils.md#NestedTensor), [`BaseLayer`](../base_layer.md#BaseLayer), [`Module`](../module.md#Module), [`_add_child`](../module.md#Module._add_child), [`BaseAttentionBias`](../attention_bias.md#BaseAttentionBias), [`ForwardMode`](../attention.md#ForwardMode), [`_compute_attention`](../flash_attention/layer.md#FlashAttention._compute_attention), [`KVState`](../kv_cache/base_kv_cache.md#KVState), [`__init__`](../base_layer.md#BaseLayer.__init__), [`child_context`](../module.md#child_context), [`KVCache`](../kv_cache/kv_cache.md#KVCache), [`default_config`](../flash_attention/layer.md#FlashAttention.default_config), [`FlashAttention`](../flash_attention/layer.md#FlashAttention), [`RoFormerQKVLinear`](../attention.md#RoFormerQKVLinear), [`PREFILL`](../attention.md#ForwardMode.PREFILL), [`SlidingWindowAttentionBias`](../attention_bias.md#SlidingWindowAttentionBias), [`EXTEND_STEP`](../attention.md#ForwardMode.EXTEND_STEP), [`per_head_dim`](../attention.md#MultiheadAttention.per_head_dim), [`FORWARD`](../attention.md#ForwardMode.FORWARD), [`SegmentIdAttentionBias`](../attention_bias.md#SegmentIdAttentionBias), [`input_linear`](../attention.md#MultiheadAttention.Config.input_linear), [`key_positions`](../kv_cache/base_kv_cache.md#KVState.key_positions), [`clone`](../config.md#ConfigBase.clone), [`as_attention_bias`](../attention_bias.md#as_attention_bias), [`kv_cache`](../attention.md#MultiheadAttention.Config.kv_cache), [`Config`](../flash_attention/layer.md#FlashAttention.Config), [`klass`](../config.md#Configurable.Config.klass), [`default_config`](../attention_bias.md#SlidingWindowAttentionBias.default_config), [`RoFormerSinusoidalPositionalEmbedding`](../attention.md#RoFormerSinusoidalPositionalEmbedding), [`num_heads`](../attention.md#BaseQKVLinear.Config.num_heads), [`per_head_dim`](../attention.md#BaseQKVLinear.Config.per_head_dim), [`query_dim`](../attention.md#BaseQKVLinear.Config.query_dim), [`instantiate`](../config.md#ClassConfigBase.instantiate), [`num_heads`](../attention.md#MultiheadAttention.Config.num_heads), [`key_dim`](../attention.md#BaseQKVLinear.Config.key_dim), [`query_dim`](../attention.md#MultiheadAttention.Config.query_dim)  (+28 more)
- used by: [`_forward_for_mode`](../attention.md#MultiheadAttention._forward_for_mode), [`default_config`](../flash_attention/layer.md#FlashAttention.default_config), [`FlashAttention`](../flash_attention/layer.md#FlashAttention), [`init_states`](../attention.md#MultiheadAttention.init_states), [`extend_step`](../attention.md#MultiheadAttention.extend_step)

### `ResidualLinearAttention`  ·  implements/extends BaseLayer
- def: [`axlearn/common/rattention/rattention.py:72`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L72)
- doc: Residual Linear Attention layer.
- signature: `class ResidualLinearAttention(BaseLayer):`
- members:
  - `_compute_init_state(self, batch_size: int)` — [`L268`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L268) — Compute the initial state for linear attention.
  - `_compute_linear_attention_parallel(self, *, q_proj: Tensor, k_proj: Tensor, v_proj: Tensor, time_step: Optional[Tensor] = None)` — [`L290`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L290) — Parallel computation of linear attention, either for training or prefilling.
  - `_compute_linear_attention_step(self, *, q_proj: Tensor, k_proj: Tensor, v_proj: Tensor, time_step: Tensor, state: Tensor)` — [`L331`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L331) — Compute linear attention for a single step.
  - `_forward_for_mode(self, *, mode: ForwardMode, query: Tensor, qkv_proj: BaseQKVLinear.Output, cached_states: Optional[NestedTensor] = None, segment_ids: Optional[Tensor] = None, page_pool: Optional[Nested[Tensor]] = None)` — [`L381`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L381) — Forward function for linear attention.
  - `_get_linear_attention_impl(self, decoding_mode=False)` — [`L187`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L187) — Get the kernel for running linear attention.
  - `_prepare_linear_attention_inputs(self, q_proj: Tensor, k_proj: Tensor, v_proj: Tensor, time_step: Optional[Tensor] = None)` — [`L231`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L231) — Apply feat_fn functions to prepare q_proj, k_proj, v_proj for linear attention.
  - `default_config(cls)` — [`L114`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L114)
  - `extend_step(self, cached_states: Nested[Tensor], query: Tensor, qkv_proj: BaseQKVLinear.Output, *, is_prefill: bool = False, segment_ids: Optional[Tensor] = None, **kwargs)` — [`L446`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L446) — Computes incremental outputs.
  - `forward(self, query: Tensor, qkv_proj: BaseQKVLinear.Output)` — [`L428`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L428)
  - `init_states(self, *, batch_size: int, max_len: int, dtype: jnp.dtype)` — [`L439`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L439) — Initializes decoding cache.
  - `per_head_dim(self)` — [`L134`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L134) — Returns the dimension of each attention head.
- protocol/private: `__init__`[`L129`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L129), `_create_layer_parameter_specs`[`L139`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L139), `_repeat_kv_heads`[`L262`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L262)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`Nested`](../utils.md#Nested), [`default_config`](../config.md#Configurable.default_config), [`config`](../config.md#Configurable.config), [`NestedTensor`](../utils.md#NestedTensor), [`BaseLayer`](../base_layer.md#BaseLayer), [`Module`](../module.md#Module), [`instantiate`](../config.md#Configurable.Config.instantiate), [`ParameterSpec`](../base_layer.md#ParameterSpec), [`ForwardMode`](../attention.md#ForwardMode), [`shape`](../utils.md#TensorSpec.shape), [`mesh_axes`](../utils.md#TensorSpec.mesh_axes), [`dtype`](../utils.md#TensorSpec.dtype), [`__init__`](../base_layer.md#BaseLayer.__init__), [`FanAxes`](../param_init.md#FanAxes), [`BaseQKVLinear`](../attention.md#BaseQKVLinear), [`parameters`](../base_layer.md#BaseLayer.parameters), [`PREFILL`](../attention.md#ForwardMode.PREFILL), [`get_feature_map`](kernels/utils.md#get_feature_map), [`EXTEND_STEP`](../attention.md#ForwardMode.EXTEND_STEP), [`FORWARD`](../attention.md#ForwardMode.FORWARD), [`FeatureMap`](kernels/utils.md#FeatureMap), `env`, [`initializer`](../base_layer.md#ParameterSpec.initializer), [`weight_decay_scale`](../base_layer.md#ParameterSpec.weight_decay_scale), [`rearrange`](../ein_ops.md#rearrange), [`residual_linear_attention_linear_scan`](kernels/linear_attention_kernels.md#residual_linear_attention_linear_scan), [`residual_linear_attention_w_timestep`](kernels/linear_attention_kernels.md#residual_linear_attention_w_timestep), [`repeat`](../ein_ops.md#repeat), [`Output`](../attention.md#BaseQKVLinear.Output), [`fwd`](kernels/utils.md#FeatureMapFn.fwd), [`ConstantInitializer`](../param_init.md#ConstantInitializer), [`WeightInitializer`](../param_init.md#WeightInitializer), [`residual_linear_attention`](kernels/linear_attention_kernels.md#residual_linear_attention), [`fan_axes`](../base_layer.md#ParameterSpec.fan_axes), [`right_shift_and_zero_pad`](kernels/linear_attention_kernels.md#right_shift_and_zero_pad), [`in_axis`](../param_init.md#FanAxes.in_axis), [`out_axis`](../param_init.md#FanAxes.out_axis), [`Config`](rattention.md#ResidualLinearAttention.Config), [`get_current_abstract_or_physical_mesh`](../utils.md#get_current_abstract_or_physical_mesh)  (+5 more)
- used by: [`default_config`](../config.md#Configurable.default_config), [`BaseLayer`](../base_layer.md#BaseLayer), [`_create_layer_parameter_specs`](../base_layer.md#BaseLayer._create_layer_parameter_specs), [`residual_la`](rattention.md#RAttention.Config.residual_la)

## Functions
- `apply_rotary_position_embeddings(inputs: Tensor, sinusoidal_pos: Tensor)` — [`L51`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention.py#L51) — Applies rotary position embeddings to the inputs.

