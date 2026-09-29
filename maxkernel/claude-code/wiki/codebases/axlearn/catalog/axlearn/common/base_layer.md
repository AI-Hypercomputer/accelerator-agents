---
title: 'Module: axlearn/common/base_layer.py'
type: catalog
provenance: extracted
module: axlearn/common/base_layer.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.base_layer`/
symbols:
  BaseLayer: BaseLayer#
  ParameterSpec: ParameterSpec#
  BaseLayer.initialize_parameters_recursively: BaseLayer#initialize_parameters_recursively().
  BaseLayer.Config: BaseLayer#Config#
  BaseLayer.__init__: BaseLayer#__init__().
  BaseLayer._create_layer_parameter_specs: BaseLayer#_create_layer_parameter_specs().
  BaseLayer.create_parameter_specs_recursively: BaseLayer#create_parameter_specs_recursively().
  BaseLayer.parameters: BaseLayer#parameters().
  BaseLayer.maybe_call_with_remat: BaseLayer#maybe_call_with_remat().
  FactorizationSpec: FactorizationSpec#
  BaseLayer._add_tensor_stats: BaseLayer#_add_tensor_stats().
  BaseLayer._initialize_parameter: BaseLayer#_initialize_parameter().
  BaseLayer._compute_fan_axes: BaseLayer#_compute_fan_axes().
  CompositeTensorStats.add_stats: CompositeTensorStats#add_stats().
  ParameterSpec.initializer: ParameterSpec#initializer.
  BaseLayer._remat_name: BaseLayer#_remat_name().
  BaseLayer.apply_parameter_noise_recursively: BaseLayer#apply_parameter_noise_recursively().
  ParameterSpec.weight_decay_scale: ParameterSpec#weight_decay_scale.
  TensorStats: TensorStats#
  ParameterSpec.factorization: ParameterSpec#factorization.
  FactorizationSpec.axes: FactorizationSpec#axes.
  BaseLayer._param_noise: BaseLayer#_param_noise.
  BaseLayer.dtype: BaseLayer#dtype().
  NestedParameterSpec: NestedParameterSpec.
  TensorStats.add_stats: TensorStats#add_stats().
  DefaultTensorStats.Config.tensor_stats: DefaultTensorStats#Config#tensor_stats.
  BaseLayer._wrap_methods_with_auto_child_context: BaseLayer#_wrap_methods_with_auto_child_context().
  ParameterSpec.fan_axes: ParameterSpec#fan_axes.
  BaseLayer.Config.param_init: BaseLayer#Config#param_init.
  BaseLayer._maybe_wrap_with_remat: BaseLayer#_maybe_wrap_with_remat().
  BaseLayer._use_prebuilt_params: BaseLayer#_use_prebuilt_params().
  CompositeTensorStats.__init__: CompositeTensorStats#__init__().
  CompositeTensorStats.Config: CompositeTensorStats#Config#
  CompositeTensorStats._child_stats: CompositeTensorStats#_child_stats.
  ParameterNoise: ParameterNoise#
  RematSpec: RematSpec#
  ParameterNoise.apply: ParameterNoise#apply().
  TensorRMSNorm.add_stats: TensorRMSNorm#add_stats().
  TensorMaxAbs.add_stats: TensorMaxAbs#add_stats().
  DefaultTensorStats.Config: DefaultTensorStats#Config#
  BaseLayer._methods_to_wrap_for_auto_child_context: BaseLayer#_methods_to_wrap_for_auto_child_context().
  BaseLayer._add_activation_summary: BaseLayer#_add_activation_summary().
  CompositeTensorStats: CompositeTensorStats#
  BaseLayer.Config.param_noise: BaseLayer#Config#param_noise.
  BaseLayer.Config.tensor_stats: BaseLayer#Config#tensor_stats.
  DefaultTensorStats: DefaultTensorStats#
  BaseLayer.Config.param_partition_spec: BaseLayer#Config#param_partition_spec.
  CompositeTensorStats.Config.tensor_stats: CompositeTensorStats#Config#tensor_stats.
  BaseLayer.fn: BaseLayer#fn().
  BaseLayer.param_init: BaseLayer#param_init().
  RematSpec.policy: RematSpec#policy.
  ParameterSpec.fans: ParameterSpec#fans().
  BaseLayer.Config.remat_spec: BaseLayer#Config#remat_spec.
  TensorRMSNorm: TensorRMSNorm#
  TensorMaxAbs: TensorMaxAbs#
  NestedFactorizationSpec: NestedFactorizationSpec.
  BaseLayer._param_init: BaseLayer#_param_init.
  RematSpec.prevent_cse: RematSpec#prevent_cse.
  BaseLayer.Config.dtype: BaseLayer#Config#dtype.
  CompositeTensorStats.Config.inline_child_summaries: CompositeTensorStats#Config#inline_child_summaries.
  DefaultTensorStats.Config.inline_child_summaries: DefaultTensorStats#Config#inline_child_summaries.
  BaseLayer._remat_methods: BaseLayer#_remat_methods.
  no_remat: no_remat().
---
# Module: [`axlearn/common/base_layer.py`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py)

## Classes
### `BaseLayer`  ·  implements/extends Module
- def: [`axlearn/common/base_layer.py:418`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L418) — documented in [axlearn-common-attention](../../../concepts/axlearn-common-attention.md)
- doc: Base class for all neural network layers in AXLearn.
- signature: `class BaseLayer(Module):`
- members:
  - `_add_activation_summary(self, *, name: str, activations: Tensor, activation_paddings: Optional[Tensor] = None)` — [`L825`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L825) — Add activation summaries.
  - `_add_tensor_stats(self, name: str, value: Nested[Tensor])` — [`L803`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L803) — Adds tensor stats about `value`.
  - `_create_layer_parameter_specs(self)` — [`L613`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L613) — Subclasses can override this method to add layer parameters.
  - `_initialize_parameter(self, name: str, *, prng_key: Tensor, parameter_spec: ParameterSpec)` — [`L716`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L716) — Adds a parameter with the given name and shape.
  - `_maybe_wrap_with_remat(self, method_fn: Callable)` — [`L531`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L531) — Maybe wrap `method_fn` with jax remat.
  - `_remat_name(self, x: Tensor, name: str)` — [`L783`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L783) — Tags 'x' with 'name' using a custom jax.extend.core.Primitive, which
  - `apply_parameter_noise_recursively(self, prng_key: Tensor, params: NestedTensor)` — [`L747`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L747) — Applies parameter noise recursively on `params`.
  - `create_parameter_specs_recursively(self)` — [`L617`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L617)
  - `dtype(self)` — [`L599`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L599)
  - `fn(*args, **kwargs)` — [`L577`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L577) — Unlike module.method, fn returns (outputs, output_collection).
  - `initialize_parameters_recursively(self, prng_key: Tensor, *, prebuilt: Optional[Nested[Optional[ParameterSpec]]] = None)` — [`L651`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L651) — Initializes parameters with given ParameterSpecs for the prebuilt params.
  - `maybe_call_with_remat(module: Module, *args, **kwargs)` — [`L548`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L548)
  - `param_init(self)` — [`L606`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L606)
  - `parameters(self)` — [`L800`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L800)
- protocol/private: `__init__`[`L495`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L495), `_compute_fan_axes`[`L773`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L773), `_methods_to_wrap_for_auto_child_context`[`L513`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L513), `_param_init`[`L504`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L504), `_param_noise`[`L506`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L506), `_remat_methods`[`L511`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L511), `_use_prebuilt_params`[`L705`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L705), `_wrap_methods_with_auto_child_context`[`L520`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L520)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`default_config`](config.md#Configurable.default_config), [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`instantiate`](config.md#Configurable.Config.instantiate), [`_add_child`](module.md#Module._add_child), [`ParameterSpec`](base_layer.md#ParameterSpec), [`shape`](utils.md#TensorSpec.shape), [`flatten_items`](utils.md#flatten_items), [`mesh_axes`](utils.md#TensorSpec.mesh_axes), [`dtype`](utils.md#TensorSpec.dtype), [`maybe_instantiate`](config.md#maybe_instantiate), [`Config`](base_layer.md#BaseLayer.Config), [`MultiheadAttention`](attention.md#MultiheadAttention), [`FanAxes`](param_init.md#FanAxes), [`BaseTransformerLayer`](attention.md#BaseTransformerLayer), [`child_context`](module.md#child_context), [`BaseQKVLinear`](attention.md#BaseQKVLinear), [`vlog`](module.md#Module.vlog), [`BaseKVCache`](kv_cache/base_kv_cache.md#BaseKVCache), [`_create_layer_parameter_specs`](quantized_dot_general/layers.md#QuantizedDotGeneral._create_layer_parameter_specs), [`_create_layer_parameter_specs`](rattention/rattention.md#ResidualLinearAttention._create_layer_parameter_specs), [`WeightedSummary`](metrics.md#WeightedSummary), [`Config`](module.md#Module.Config), [`add_summary`](module.md#Module.add_summary), [`__init__`](module.md#Module.__init__), [`new_output_collection`](module.md#new_output_collection), [`initialize_parameters_recursively`](attention.md#StackedTransformerLayer.initialize_parameters_recursively), [`path`](module.md#Module.path), [`create_parameter_specs_recursively`](attention.md#FusedQKVLinear.create_parameter_specs_recursively), [`_create_layer_parameter_specs`](attention.md#MultiheadAttention._create_layer_parameter_specs), [`summaries`](module.md#OutputCollection.summaries), [`Initializer`](param_init.md#Initializer), [`nowrap`](module.md#nowrap), [`create_parameter_specs_recursively`](pipeline.md#Pipeline.create_parameter_specs_recursively), [`initialize_parameters_recursively`](pipeline.md#Pipeline.initialize_parameters_recursively), [`PerDimScale`](attention.md#PerDimScale), [`initialize_parameters_recursively`](attention.md#PipelinedTransformerLayer.initialize_parameters_recursively)  (+80 more)
- used by: [`Module`](module.md#Module), [`_forward_for_mode`](attention.md#MultiheadAttention._forward_for_mode), [`_compute_attention`](flash_attention/layer.md#FlashAttention._compute_attention), [`_test_backward`](flash_attention/layer_test.md#TestFlashAttention._test_backward), `_compare_layers`, [`MultiheadAttention`](attention.md#MultiheadAttention), `test_gqa_forward`, [`BaseTransformerLayer`](attention.md#BaseTransformerLayer), `_test_extend_step`, [`__init__`](attention.md#QKVLinear.__init__), `test_all_mask`, `test_build_remat_spec_neuron`, [`BaseQKVLinear`](attention.md#BaseQKVLinear), [`__init__`](attention.md#TransformerFeedForwardLayer.__init__), [`test_logit_sink`](flash_attention/layer_test.md#TestFlashAttention.test_logit_sink), `test_sigmoid_compute_attention`, [`BaseKVCache`](kv_cache/base_kv_cache.md#BaseKVCache), [`_forward_for_mode`](attention.md#TransformerAttentionLayer._forward_for_mode), [`_forward_for_mode`](attention.md#TransformerLayer._forward_for_mode), [`_create_layer_parameter_specs`](quantized_dot_general/layers.md#QuantizedDotGeneral._create_layer_parameter_specs), [`_prepare_layers`](flash_attention/layer_test.md#_prepare_layers), `test_gqa_against_mha`, [`_compute_attention`](attention.md#MultiheadAttention._compute_attention), [`__init__`](attention.md#RoFormerQKVLinear.__init__), [`__init__`](decoder.md#Decoder.__init__), `test_qkv_equality`, `test_transformer_extend_step`, [`__init__`](attention.md#FusedGroupedQKVLinear.__init__), `test_per_dim_scale`, `test_v2_structure`, `_test_prefill_states`, [`__init__`](attention.md#FusedQKVLinear.__init__), [`create_parameter_specs_recursively`](attention.md#FusedQKVLinear.create_parameter_specs_recursively), `test_compute_fan_axes`, `test_roformer_qkv_linear`, [`_forward_for_mode`](rattention/rattention.md#ResidualLinearAttention._forward_for_mode), `_test_decoder_with_transformer`, [`__init__`](attention.md#PipelinedTransformerLayer.__init__), [`__init__`](attention.md#TransformerAttentionLayer.__init__), `test_build_remat_spec`  (+166 more; 6 test-only)

### `CompositeTensorStats`  ·  implements/extends TensorStats
- def: [`axlearn/common/base_layer.py:312`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L312)
- doc: A TensorStats that combines multiple child TensorStats implementations.
- signature: `class CompositeTensorStats(TensorStats):`
- members:
  - `add_stats(self, name: str, value: Nested[Tensor])` — [`L363`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L363)
- protocol/private: `__init__`[`L356`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L356), `_child_stats`[`L359`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L359)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`child_context`](module.md#child_context), [`__init__`](module.md#Module.__init__), [`new_output_collection`](module.md#new_output_collection), [`summaries`](module.md#OutputCollection.summaries), [`output_collection`](module.md#InvocationContext.output_collection), [`get_invocation_context`](module.md#Module.get_invocation_context), [`TensorStats`](base_layer.md#TensorStats), [`Config`](base_layer.md#CompositeTensorStats.Config), [`DefaultTensorStats`](base_layer.md#DefaultTensorStats), [`tensor_stats`](base_layer.md#CompositeTensorStats.Config.tensor_stats)
- used by: [`TensorStats`](base_layer.md#TensorStats), [`add_stats`](base_layer.md#TensorStats.add_stats), [`Config`](base_layer.md#DefaultTensorStats.Config), [`DefaultTensorStats`](base_layer.md#DefaultTensorStats)

### `Config`  ·  implements/extends Config
- def: [`axlearn/common/base_layer.py:465`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L465)
- doc: Configuration for BaseLayer. These settings cascade down the layer hierarchy - child
- signature: `class Config(Module.Config):`
- members:
  - `dtype` — [`L472`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L472)
  - `inline_child_summaries` — [`L354`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L354)
  - `inline_child_summaries` — [`L415`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L415)
  - `param_init` — [`L475`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L475)
  - `param_noise` — [`L488`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L488)
  - `param_partition_spec` — [`L479`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L479)
  - `remat_spec` — [`L483`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L483)
  - `tensor_stats` — [`L348`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L348)
  - `tensor_stats` — [`L411`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L411)
  - `tensor_stats` — [`L493`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L493)
- uses (calls/refs, reference-scoped): [`default_config`](config.md#Configurable.default_config), [`config_class`](config.md#config_class), [`Module`](module.md#Module), [`Config`](module.md#Module.Config), [`Config`](config.md#Configurable.Config), [`TensorStats`](base_layer.md#TensorStats), [`DefaultInitializer`](param_init.md#DefaultInitializer), [`NestedParameterSpec`](base_layer.md#NestedParameterSpec), [`Config`](param_init.md#DefaultInitializer.Config), [`ParameterNoise`](base_layer.md#ParameterNoise), [`RematSpec`](base_layer.md#RematSpec), [`TensorMaxAbs`](base_layer.md#TensorMaxAbs), [`TensorRMSNorm`](base_layer.md#TensorRMSNorm)
- used by: [`__init__`](base_layer.md#BaseLayer.__init__), [`__init__`](attention.md#QKVLinear.__init__), `test_repeated_layer_with_custom_carry`, [`__init__`](layers.md#SqueezeExcitation.__init__), [`set_attention_partition_specs`](attention.md#set_attention_partition_specs), [`default_config`](attention.md#PerDimScale.default_config), `test_initialize_parameters_recursively`, [`Config`](attention.md#MultiheadAttention.Config), [`set_quantized_dot_general_recursively`](quantized_dot_general/layers.md#set_quantized_dot_general_recursively), [`Config`](attention.md#PerDimScale.Config), [`_is_initialized`](trainer.md#SpmdTrainer._is_initialized), [`Config`](attention.md#BaseQKVLinear.Config), [`Config`](attention.md#BaseTransformerLayer.Config), [`model`](trainer.md#SpmdTrainer.Config.model), [`_param_noise`](base_layer.md#BaseLayer._param_noise), [`Config`](attention.md#BaseScaleQK.Config), [`Config`](attention.md#TransformerFeedForwardLayer.Config), [`Config`](layers.md#BaseNormalizationLayer.Config), [`Config`](pipeline.md#Pipeline.Config), [`Config`](attention.md#TransformerAttentionLayer.Config), [`Config`](quantized_dot_general/layers.md#DenseGeneralBaseLayer.Config), [`stage`](attention.md#PipelinedTransformerLayer.Config.stage), [`Config`](quantized_dot_general/activation_clipping.md#BaseActivationClippingLayer.Config), [`Config`](quantized_dot_general/layers.md#QuantizedDotGeneral.Config), [`Config`](embedding.md#BaseEmbedding.Config), [`Config`](flash_attention/layer_test.md#DummyModel.Config), [`Config`](layers.md#Dropout.Config), [`Config`](layers.md#RedirectToSharedModule.Config), [`Config`](rattention/rattention.md#ResidualLinearAttention.Config), [`Config`](repeat.md#Repeat.Config), [`inner`](loss_metrics.md#ModalityLossMetrics.Config.inner), `Config`, [`Config`](attention.md#RoFormerSinusoidalPositionalEmbedding.Config), [`Config`](attention.md#SinusoidalPositionalEmbedding.Config), [`Config`](decoder.md#Decoder.Config), [`Config`](kv_cache/base_kv_cache.md#BaseKVCache.Config), [`_child_stats`](base_layer.md#CompositeTensorStats._child_stats), [`quantized_dot_general`](quantized_dot_general/layers.md#DenseGeneralBaseLayer.Config.quantized_dot_general), [`Config`](layers.md#BaseClassificationHead.Config), [`Config`](layers.md#DropToken.Config)  (+21 more; 1 test-only)

### `DefaultTensorStats`  ·  implements/extends CompositeTensorStats
- def: [`axlearn/common/base_layer.py:393`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L393)
- doc: Default tensor stats that compute RMS norm and max value.
- signature: `class DefaultTensorStats(CompositeTensorStats):`
- uses (calls/refs, reference-scoped): [`CompositeTensorStats`](base_layer.md#CompositeTensorStats)
- used by: `test_add_value_rms_norm_summary`, `test_linear_remat`, `test_add_tensor_stats`, [`CompositeTensorStats`](base_layer.md#CompositeTensorStats)

### `FactorizationSpec`
- def: [`axlearn/common/base_layer.py:131`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L131)
- doc: A FactorizationSpec describes how to factorize a parameter's gradient.
- signature: `class FactorizationSpec:`
- members:
  - `axes` — [`L155`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L155)
- used by: `test_all_mask`, [`mock_updates`](update_transformation_test.md#mock_updates), [`get_vx_spec`](factorized_rms.md#scale_by_factored_rms.get_vx_spec), [`test_param_specs`](update_transformation_test.md#UpdatesTest.test_param_specs), [`_create_layer_parameter_specs`](attention.md#BaseMultiheadLinear._create_layer_parameter_specs), [`_create_layer_parameter_specs`](layers.md#Linear._create_layer_parameter_specs), [`_create_layer_parameter_specs`](layers.md#MultiLinear._create_layer_parameter_specs), [`factorization`](base_layer.md#ParameterSpec.factorization), [`factorization_spec`](optimizer_base.md#OptParam.factorization_spec), [`_factored_dims`](factorized_rms.md#_factored_dims), [`transform_factorization_spec`](attention.md#FusedQKVLinear.transform_factorization_spec), [`transform_factorization_spec`](pipeline.md#Pipeline.transform_factorization_spec), [`transform_factorization_spec`](repeat.md#Repeat.transform_factorization_spec), [`NestedFactorizationSpec`](base_layer.md#NestedFactorizationSpec)

### `ParameterNoise`  ·  implements/extends Configurable
- def: [`axlearn/common/base_layer.py:263`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L263)
- doc: An interface for applying parameter noise.
- signature: `class ParameterNoise(Configurable):`
- members:
  - `apply(self, prng_key: Tensor, params: NestedTensor)` — [`L266`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L266) — To be implemented by subclasses.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`NestedTensor`](utils.md#NestedTensor), [`Configurable`](config.md#Configurable), [`apply`](layers.md#VariationalNoise.apply), [`VariationalNoise`](layers.md#VariationalNoise)
- used by: [`Configurable`](config.md#Configurable), [`Config`](layers.md#VariationalNoise.Config), [`param_noise`](base_layer.md#BaseLayer.Config.param_noise), [`VariationalNoise`](layers.md#VariationalNoise)

### `ParameterSpec`  ·  implements/extends TensorSpec
- def: [`axlearn/common/base_layer.py:165`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L165)
- doc: Specification of a layer parameter.
- signature: `class ParameterSpec(TensorSpec):`
- members:
  - `fans(self)` — [`L208`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L208) — Returns a dictionary with keys 'fan_in', 'fan_out', and 'fan_avg' containing
  - `factorization` — [`L204`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L204)
  - `fan_axes` — [`L205`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L205)
  - `initializer` — [`L203`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L203)
  - `weight_decay_scale` — [`L206`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L206)
- uses (calls/refs, reference-scoped): [`shape`](utils.md#TensorSpec.shape), [`FanAxes`](param_init.md#FanAxes), [`TensorSpec`](utils.md#TensorSpec), [`Initializer`](param_init.md#Initializer), [`FactorizationSpec`](base_layer.md#FactorizationSpec)
- used by: [`initialize_parameters_recursively`](base_layer.md#BaseLayer.initialize_parameters_recursively), `test_all_mask`, [`_create_layer_parameter_specs`](base_layer.md#BaseLayer._create_layer_parameter_specs), [`create_parameter_specs_recursively`](base_layer.md#BaseLayer.create_parameter_specs_recursively), [`_create_layer_parameter_specs`](quantized_dot_general/layers.md#QuantizedDotGeneral._create_layer_parameter_specs), [`_create_layer_parameter_specs`](rattention/rattention.md#ResidualLinearAttention._create_layer_parameter_specs), [`test_conditional_update_transformation`](update_transformation_test.md#ConditionalUpdateTransformationTest.test_conditional_update_transformation), [`partition_fn`](optimizers.md#skip_and_clip_by_global_norm.partition_fn), [`TensorSpec`](utils.md#TensorSpec), [`initialize_parameters_recursively`](attention.md#StackedTransformerLayer.initialize_parameters_recursively), [`create_parameter_specs_recursively`](attention.md#FusedQKVLinear.create_parameter_specs_recursively), [`_create_layer_parameter_specs`](attention.md#MultiheadAttention._create_layer_parameter_specs), [`_trainer_state_specs`](trainer.md#SpmdTrainer._trainer_state_specs), [`partition_fn`](optimizers.md#ema.partition_fn), [`initialize_parameters_recursively`](pipeline.md#Pipeline.initialize_parameters_recursively), [`get_vx_spec`](factorized_rms.md#scale_by_factored_rms.get_vx_spec), [`initialize_parameters_recursively`](attention.md#PipelinedTransformerLayer.initialize_parameters_recursively), [`partition_fn`](optimizers.md#param_ema.partition_fn), [`partition_fn`](optimizers.md#scale_by_lion.partition_fn), [`create_state_partition_specs`](update_transformation.md#ConditionalUpdateTransformation.create_state_partition_specs), [`_initialize_parameter`](base_layer.md#BaseLayer._initialize_parameter), [`_partition`](optimizers.md#adastar_optimizer.partition_fn._partition), [`_compute_fan_axes`](base_layer.md#BaseLayer._compute_fan_axes), [`test_param_specs`](update_transformation_test.md#UpdatesTest.test_param_specs), [`param_specs`](update_transformation.md#Updates.param_specs), [`_create_layer_parameter_specs`](attention.md#MultiheadAttentionXL._create_layer_parameter_specs), [`_create_layer_parameter_specs`](layers.md#BatchNorm._create_layer_parameter_specs), [`_create_layer_parameter_specs`](layers.md#MovingAverage._create_layer_parameter_specs), [`create_state_partition_specs`](learner.md#Learner.create_state_partition_specs), [`initialize_parameters_recursively`](attention.md#FusedQKVLinear.initialize_parameters_recursively), [`initialize_parameters_recursively`](repeat.md#Repeat.initialize_parameters_recursively), [`partition_fn`](optimizers.md#adam_partition.partition_fn), [`_create_layer_parameter_specs`](attention.md#BaseMultiheadLinear._create_layer_parameter_specs), [`_create_layer_parameter_specs`](attention.md#TransformerAttentionLayer._create_layer_parameter_specs), [`_create_layer_parameter_specs`](attention.md#TransformerFeedForwardLayer._create_layer_parameter_specs), [`_create_layer_parameter_specs`](layers.md#Linear._create_layer_parameter_specs), [`_create_layer_parameter_specs`](layers.md#MultiLinear._create_layer_parameter_specs), [`create_state_partition_specs`](learner.md#CompositeLearner.create_state_partition_specs), [`create_state_partition_specs`](learner_base.md#LearnerModule.create_state_partition_specs), [`get_scale_partition`](optimizers.md#ema.partition_fn.get_scale_partition)  (+26 more; 1 test-only)

### `RematSpec`
- def: [`axlearn/common/base_layer.py:235`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L235)
- doc: Configuration for rematerialization (remat) / checkpointing of layer computations.
- signature: `class RematSpec:`
- members:
  - `policy` — [`L260`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L260)
  - `prevent_cse` — [`L259`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L259)
- uses (calls/refs, reference-scoped): [`ConfigOr`](config.md#ConfigOr)
- used by: `test_stack_vs_pipeline`, [`build_remat_spec`](attention.md#build_remat_spec), `test_stack_vs_repeat_remat_everything_saveable`, [`remat_spec`](base_layer.md#BaseLayer.Config.remat_spec)

### `TensorMaxAbs`  ·  implements/extends TensorStats
- def: [`axlearn/common/base_layer.py:386`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L386)
- doc: Computes and logs the maximum absolute value of tensors.
- signature: `class TensorMaxAbs(TensorStats):`
- members:
  - `add_stats(self, name: str, value: Nested[Tensor])` — [`L389`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L389)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`add_summary`](module.md#Module.add_summary), [`TensorStats`](base_layer.md#TensorStats)
- used by: [`TensorStats`](base_layer.md#TensorStats), [`add_stats`](base_layer.md#TensorStats.add_stats), [`tensor_stats`](base_layer.md#DefaultTensorStats.Config.tensor_stats)

### `TensorRMSNorm`  ·  implements/extends TensorStats
- def: [`axlearn/common/base_layer.py:379`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L379)
- doc: Computes and logs the root mean square norm of tensors.
- signature: `class TensorRMSNorm(TensorStats):`
- members:
  - `add_stats(self, name: str, value: Nested[Tensor])` — [`L382`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L382)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`add_summary`](module.md#Module.add_summary), [`TensorStats`](base_layer.md#TensorStats)
- used by: [`TensorStats`](base_layer.md#TensorStats), [`add_stats`](base_layer.md#TensorStats.add_stats), [`tensor_stats`](base_layer.md#DefaultTensorStats.Config.tensor_stats)

### `TensorStats`  ·  implements/extends Module
- def: [`axlearn/common/base_layer.py:271`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L271)
- doc: An abstract Module to add summaries about the given Tensors.
- signature: `class TensorStats(Module):`
- members:
  - `add_stats(self, name: str, value: Nested[Tensor])` — [`L299`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L299) — Computes and adds summaries for the given tensor.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`Module`](module.md#Module), [`add_stats`](base_layer.md#CompositeTensorStats.add_stats), [`add_stats`](base_layer.md#TensorMaxAbs.add_stats), [`add_stats`](base_layer.md#TensorRMSNorm.add_stats), [`CompositeTensorStats`](base_layer.md#CompositeTensorStats), [`TensorMaxAbs`](base_layer.md#TensorMaxAbs), [`TensorRMSNorm`](base_layer.md#TensorRMSNorm)
- used by: [`Module`](module.md#Module), [`tensor_stats`](base_layer.md#DefaultTensorStats.Config.tensor_stats), [`Config`](base_layer.md#CompositeTensorStats.Config), [`CompositeTensorStats`](base_layer.md#CompositeTensorStats), [`tensor_stats`](base_layer.md#BaseLayer.Config.tensor_stats), [`tensor_stats`](base_layer.md#CompositeTensorStats.Config.tensor_stats), [`TensorMaxAbs`](base_layer.md#TensorMaxAbs), [`TensorRMSNorm`](base_layer.md#TensorRMSNorm)

## Functions
- `no_remat(fn: Callable)` — [`L871`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L871) — Decorator to exclude a method from rematerialization (remat/checkpoint).

## Module values
- `NestedFactorizationSpec` — [`L161`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L161)
- `NestedParameterSpec` — [`L231`](../../../../../../raw/code/axlearn/axlearn/common/base_layer.py#L231)

