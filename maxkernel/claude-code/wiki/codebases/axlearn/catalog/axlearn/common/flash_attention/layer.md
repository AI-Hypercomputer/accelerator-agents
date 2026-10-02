---
title: 'Module: axlearn/common/flash_attention/layer.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/layer.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.layer`/
symbols:
  FlashAttention._compute_attention: FlashAttention#_compute_attention().
  FlashAttention.default_config: FlashAttention#default_config().
  FlashAttention: FlashAttention#
  FlashAttention._maybe_repeat_kv_heads: FlashAttention#_maybe_repeat_kv_heads().
  FlashAttention.Config: FlashAttention#Config#
  FlashAttention.__init__: FlashAttention#__init__().
  FlashBlockSizeModifier.visit_fn: FlashBlockSizeModifier#visit_fn().
  FlashBlockSizeModifier.__call__: FlashBlockSizeModifier#__call__().
  BackendOverrideModifier.__call__: BackendOverrideModifier#__call__().
  BackendOverrideModifier.visit_fn: BackendOverrideModifier#visit_fn().
  FlashAttention.Config.mha_dim_to_partition_spec: FlashAttention#Config#mha_dim_to_partition_spec.
  FlashAttention._create_layer_parameter_specs: FlashAttention#_create_layer_parameter_specs().
  FlashAttention._logit_biases_spec: FlashAttention#_logit_biases_spec().
  FlashBlockSizeModifier.Config: FlashBlockSizeModifier#Config#
  BackendOverrideModifier.Config: BackendOverrideModifier#Config#
  default_mha_dim_to_partition_spec: default_mha_dim_to_partition_spec().
  default_output_dim_to_partition_spec: default_output_dim_to_partition_spec().
  FlashBlockSizeModifier.is_flash_config: FlashBlockSizeModifier#is_flash_config().
  BackendOverrideModifier.is_flash_config: BackendOverrideModifier#is_flash_config().
  FlashAttention.Config.backend_overrides: FlashAttention#Config#backend_overrides.
  FlashAttention._get_kv_cache_type: FlashAttention#_get_kv_cache_type().
  BackendOverrideModifier: BackendOverrideModifier#
  FlashAttention._backend: FlashAttention#_backend().
  FlashBlockSizeModifier.enter_fn: FlashBlockSizeModifier#enter_fn().
  BackendOverrideModifier.enter_fn: BackendOverrideModifier#enter_fn().
  FlashAttention.Config.tpu_block_size: FlashAttention#Config#tpu_block_size.
  FlashAttention.Config.output_dim_to_partition_spec: FlashAttention#Config#output_dim_to_partition_spec.
  FlashBlockSizeModifier: FlashBlockSizeModifier#
  FlashAttention.Config.gpu_block_size: FlashAttention#Config#gpu_block_size.
  FlashAttention.Config.softmax_scale: FlashAttention#Config#softmax_scale.
  FlashBlockSizeModifier.Config.tpu_block_size: FlashBlockSizeModifier#Config#tpu_block_size.
  FlashBlockSizeModifier.Config.gpu_block_size: FlashBlockSizeModifier#Config#gpu_block_size.
  BackendOverrideModifier.Config.backend_overrides: BackendOverrideModifier#Config#backend_overrides.
---
# Module: [`axlearn/common/flash_attention/layer.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py)

## Classes
### `BackendOverrideModifier`  ·  implements/extends ConfigModifier
- def: [`axlearn/common/flash_attention/layer.py:430`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L430)
- doc: Modifies the backend_overrides config of Flash Attention.
- signature: `class BackendOverrideModifier(ConfigModifier):`
- members:
  - `enter_fn(_, value, default_kv)` — [`L458`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L458)
  - `is_flash_config(cfg)` — [`L442`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L442)
  - `visit_fn(_, value)` — [`L445`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L445)
- protocol/private: `__call__`[`L439`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L439)
- uses (calls/refs, reference-scoped): [`config`](../config.md#Configurable.config), [`ConfigBase`](../config.md#ConfigBase), [`FlashAttention`](layer.md#FlashAttention), [`Config`](layer.md#FlashAttention.Config), [`visit`](../config.md#ConfigBase.visit), [`ConfigModifier`](../config.md#ConfigModifier), [`backend_overrides`](layer.md#FlashAttention.Config.backend_overrides)
- used by: [`test_backend_override_modifier`](layer_test.md#TestFlashAttention.test_backend_override_modifier), [`test_backend_override_modifier_ignores_none`](layer_test.md#TestFlashAttention.test_backend_override_modifier_ignores_none), [`ConfigModifier`](../config.md#ConfigModifier)

### `Config`  ·  implements/extends Config
- def: [`axlearn/common/flash_attention/layer.py:434`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L434)
- doc: Configures BackendOverrideModifier.
- signature: `class Config(ConfigModifier.Config):`
- members:
  - `backend_overrides` — [`L72`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L72)
  - `backend_overrides` — [`L437`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L437)
  - `gpu_block_size` — [`L57`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L57)
  - `gpu_block_size` — [`L408`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L408)
  - `mha_dim_to_partition_spec` — [`L66`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L66)
  - `output_dim_to_partition_spec` — [`L68`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L68)
  - `softmax_scale` — [`L76`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L76)
  - `tpu_block_size` — [`L54`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L54)
  - `tpu_block_size` — [`L407`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L407)
- uses (calls/refs, reference-scoped): [`config_class`](../config.md#config_class), [`Config`](../config.md#Configurable.Config), [`ConfigModifier`](../config.md#ConfigModifier)
- used by: [`_compute_attention`](layer.md#FlashAttention._compute_attention), [`default_config`](layer.md#FlashAttention.default_config), [`_maybe_repeat_kv_heads`](layer.md#FlashAttention._maybe_repeat_kv_heads), [`test_backend_override_modifier`](layer_test.md#TestFlashAttention.test_backend_override_modifier), [`test_backend_override_modifier_ignores_none`](layer_test.md#TestFlashAttention.test_backend_override_modifier_ignores_none), [`__init__`](layer.md#FlashAttention.__init__), [`visit_fn`](layer.md#FlashBlockSizeModifier.visit_fn), [`visit_fn`](layer.md#BackendOverrideModifier.visit_fn)

### `FlashAttention`  ·  implements/extends GroupedQueryAttention
- def: [`axlearn/common/flash_attention/layer.py:30`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L30) — documented in [axlearn-common-flash_attention-layer](../../../../concepts/axlearn-common-flash_attention-layer.md)
- doc: FlashAttention layer.
- signature: `class FlashAttention(GroupedQueryAttention):`
- members:
  - `_compute_attention(self, *, mode: ForwardMode, q_proj: Tensor, kv_state: KVState, attention_logit_biases: BaseAttentionBias)` — [`L190`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L190) — Computes attention context and probs. — documented in [axlearn-common-attention_bias](../../../../concepts/axlearn-common-attention_bias.md)
  - `_maybe_repeat_kv_heads(self, key_or_value: Tensor)` — [`L137`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L137) — Repeats key or value heads dim to be shardable. — documented in [axlearn-common-flash_attention-layer](../../../../concepts/axlearn-common-flash_attention-layer.md)
  - `default_config(cls)` — [`L108`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L108) — documented in [axlearn-common-flash_attention-layer](../../../../concepts/axlearn-common-flash_attention-layer.md)
- protocol/private: `__init__`[`L78`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L78), `_backend`[`L123`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L123), `_create_layer_parameter_specs`[`L94`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L94), `_get_kv_cache_type`[`L344`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L344), `_logit_biases_spec`[`L133`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L133)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`default_config`](../config.md#Configurable.default_config), [`config`](../config.md#Configurable.config), [`Module`](../module.md#Module), [`ParameterSpec`](../base_layer.md#ParameterSpec), [`BaseAttentionBias`](../attention_bias.md#BaseAttentionBias), [`ForwardMode`](../attention.md#ForwardMode), [`mesh_axes`](../utils.md#TensorSpec.mesh_axes), [`KVState`](../kv_cache/base_kv_cache.md#KVState), [`_compute_attention`](../attention.md#MultiheadAttention._compute_attention), [`parameters`](../base_layer.md#BaseLayer.parameters), [`flash_attention_implementation`](utils.md#flash_attention_implementation), [`_create_layer_parameter_specs`](../attention.md#MultiheadAttention._create_layer_parameter_specs), [`with_sharding_constraint`](../utils.md#with_sharding_constraint), [`maybe_shard`](../utils.md#maybe_shard), [`Dropout`](../layers.md#Dropout), [`GroupedQueryAttention`](../attention.md#GroupedQueryAttention), [`k_proj`](../kv_cache/base_kv_cache.md#KVState.k_proj), [`v_proj`](../kv_cache/base_kv_cache.md#KVState.v_proj), [`EXTEND_STEP`](../attention.md#ForwardMode.EXTEND_STEP), `env`, [`partition_spec`](../attention_bias.md#BaseAttentionBias.partition_spec), [`Config`](layer.md#FlashAttention.Config), [`__init__`](../attention.md#MultiheadAttention.__init__), [`num_heads`](../attention.md#MultiheadAttention.Config.num_heads), [`dropout`](../attention.md#MultiheadAttention.Config.dropout), [`is_training`](../module.md#Module.is_training), [`mha_dim_to_partition_spec`](layer.md#FlashAttention.Config.mha_dim_to_partition_spec), [`get_current_abstract_or_physical_mesh`](../utils.md#get_current_abstract_or_physical_mesh), [`default_config`](../rattention/rattention.md#RAttention.default_config), [`q_partition_spec`](../attention.md#MultiheadAttention.Config.q_partition_spec), [`split_prng_keys_for_shard_map`](common.md#split_prng_keys_for_shard_map), [`k_partition_spec`](../attention.md#MultiheadAttention.Config.k_partition_spec), [`page_indices`](../kv_cache/base_kv_cache.md#KVState.page_indices), [`backend_overrides`](layer.md#FlashAttention.Config.backend_overrides), [`v_partition_spec`](../attention.md#MultiheadAttention.Config.v_partition_spec), [`astype`](../attention_bias.md#BaseAttentionBias.astype), [`fn`](attention_types.md#FlashAttentionWithShardMapSpecs.fn), [`output_dim_to_partition_spec`](layer.md#FlashAttention.Config.output_dim_to_partition_spec), [`tpu_block_size`](layer.md#FlashAttention.Config.tpu_block_size)  (+7 more)
- used by: [`default_config`](../config.md#Configurable.default_config), [`_test_backward`](layer_test.md#TestFlashAttention._test_backward), [`_forward_for_mode`](../rattention/rattention.md#RAttention._forward_for_mode), [`test_logit_sink`](layer_test.md#TestFlashAttention.test_logit_sink), [`_prepare_layers`](layer_test.md#_prepare_layers), [`_compute_attention`](../attention.md#MultiheadAttention._compute_attention), [`_create_layer_parameter_specs`](../attention.md#MultiheadAttention._create_layer_parameter_specs), [`GroupedQueryAttention`](../attention.md#GroupedQueryAttention), [`extend_step`](../rattention/rattention.md#RAttention.extend_step), [`test_backend_override_modifier`](layer_test.md#TestFlashAttention.test_backend_override_modifier), [`test_backend_override_modifier_ignores_none`](layer_test.md#TestFlashAttention.test_backend_override_modifier_ignores_none), [`test_logit_sink_backward_pass`](layer_test.md#TestFlashAttention.test_logit_sink_backward_pass), [`test_logit_sink_numerical_stability`](layer_test.md#TestFlashAttention.test_logit_sink_numerical_stability), [`test_softmax_scale`](layer_test.md#TestFlashAttention.test_softmax_scale), [`test_dropout_support`](layer_test.md#TestFlashAttention.test_dropout_support), [`test_gqa_kv_heads`](layer_test.md#TestFlashAttention.test_gqa_kv_heads), [`visit_fn`](layer.md#FlashBlockSizeModifier.visit_fn), [`visit_fn`](layer.md#BackendOverrideModifier.visit_fn), [`Config`](../rattention/rattention.md#RAttention.Config), [`default_config`](../rattention/rattention.md#RAttention.default_config), [`test_create_layer_parameter_specs_with_logit_sink`](layer_test.md#TestFlashAttention.test_create_layer_parameter_specs_with_logit_sink), [`test_create_layer_parameter_specs_without_logit_sink`](layer_test.md#TestFlashAttention.test_create_layer_parameter_specs_without_logit_sink), [`test_logit_sink_parameter_initialization`](layer_test.md#TestFlashAttention.test_logit_sink_parameter_initialization), [`is_flash_config`](layer.md#BackendOverrideModifier.is_flash_config), [`is_flash_config`](layer.md#FlashBlockSizeModifier.is_flash_config), [`RAttention`](../rattention/rattention.md#RAttention)

### `FlashBlockSizeModifier`  ·  implements/extends ConfigModifier
- def: [`axlearn/common/flash_attention/layer.py:400`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L400)
- doc: Modifies the tpu_block_size or gpu_block_size config of FlashAttention.
- signature: `class FlashBlockSizeModifier(ConfigModifier):`
- members:
  - `enter_fn(_, value, default_kv)` — [`L423`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L423)
  - `is_flash_config(cfg)` — [`L414`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L414)
  - `visit_fn(_, value)` — [`L417`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L417)
- protocol/private: `__call__`[`L410`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L410)
- uses (calls/refs, reference-scoped): [`config`](../config.md#Configurable.config), [`ConfigBase`](../config.md#ConfigBase), [`FlashAttention`](layer.md#FlashAttention), [`Config`](layer.md#FlashAttention.Config), [`visit`](../config.md#ConfigBase.visit), [`ConfigModifier`](../config.md#ConfigModifier), [`tpu_block_size`](layer.md#FlashAttention.Config.tpu_block_size), [`gpu_block_size`](layer.md#FlashAttention.Config.gpu_block_size)
- used by: [`ConfigModifier`](../config.md#ConfigModifier)

## Functions
- `default_mha_dim_to_partition_spec(mesh_axis_names: Sequence[str])` — [`L353`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L353) — Builds a default FlashAttention mapping from tensor dims to partition specs for the MHA impl.
- `default_output_dim_to_partition_spec(mesh_axis_names: Sequence[str])` — [`L376`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer.py#L376) — Builds a default mapping from tensor dims to partition specs for the FlashAttention outputs.

