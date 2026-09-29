---
title: 'Module: axlearn/common/flash_attention/layer_test.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/layer_test.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.layer_test`/
symbols:
  TestFlashAttention._test_backward: TestFlashAttention#_test_backward().
  TestFlashAttention.test_logit_sink: TestFlashAttention#test_logit_sink().
  _prepare_layers: _prepare_layers().
  TestFlashAttention._test_extend_step: TestFlashAttention#_test_extend_step().
  TestFlashAttention._test_forward: TestFlashAttention#_test_forward().
  TestFlashAttention.test_shard_biases: TestFlashAttention#test_shard_biases().
  TestFlashAttention.test_backend_override_modifier: TestFlashAttention#test_backend_override_modifier().
  TestFlashAttention.test_backend_override_modifier_ignores_none: TestFlashAttention#test_backend_override_modifier_ignores_none().
  TestFlashAttention.test_logit_sink_numerical_stability: TestFlashAttention#test_logit_sink_numerical_stability().
  TestFlashAttention.test_logit_sink_backward_pass: TestFlashAttention#test_logit_sink_backward_pass().
  TestFlashAttention.test_softmax_scale: TestFlashAttention#test_softmax_scale().
  TestFlashAttention.test_incremental_prefill: TestFlashAttention#test_incremental_prefill().
  TestFlashAttention._ATTN_CONFIGS: TestFlashAttention#_ATTN_CONFIGS.
  DummyModel.__init__: DummyModel#__init__().
  TestFlashAttention.test_dropout_support: TestFlashAttention#test_dropout_support().
  TestFlashAttention.test_gqa_kv_heads: TestFlashAttention#test_gqa_kv_heads().
  DummyModel.Config.layer: DummyModel#Config#layer.
  jax_fn_mask: jax_fn_mask().
  _fake_inputs: _fake_inputs().
  DummyModel.Config: DummyModel#Config#
  TestFlashAttention.as_tensor_bias: TestFlashAttention#as_tensor_bias().
  TestFlashAttention.test_backend: TestFlashAttention#test_backend().
  DummyModel: DummyModel#
  TestFlashAttention.test_logit_sink_parameter_initialization: TestFlashAttention#test_logit_sink_parameter_initialization().
  TestFlashAttention.test_create_layer_parameter_specs_with_logit_sink: TestFlashAttention#test_create_layer_parameter_specs_with_logit_sink().
  TestFlashAttention.test_create_layer_parameter_specs_without_logit_sink: TestFlashAttention#test_create_layer_parameter_specs_without_logit_sink().
  TestFlashAttention.as_partition_spec: TestFlashAttention#as_partition_spec().
  TestFlashAttention._MESH_CONFIGS: TestFlashAttention#_MESH_CONFIGS.
  TestFlashAttention.loss_fn: TestFlashAttention#loss_fn().
  TestFlashAttention.test_forward_attn_types: TestFlashAttention#test_forward_attn_types().
  TestFlashAttention.test_forward_cross_attn: TestFlashAttention#test_forward_cross_attn().
  TestFlashAttention.test_forward_bias: TestFlashAttention#test_forward_bias().
  TestFlashAttention.test_forward_segment_ids: TestFlashAttention#test_forward_segment_ids().
  TestFlashAttention.test_forward_float32: TestFlashAttention#test_forward_float32().
  TestFlashAttention.test_forward_dropout: TestFlashAttention#test_forward_dropout().
  TestFlashAttention.test_forward_mesh: TestFlashAttention#test_forward_mesh().
  TestFlashAttention.test_backward_attn_types: TestFlashAttention#test_backward_attn_types().
  TestFlashAttention.test_backward_cross_attn: TestFlashAttention#test_backward_cross_attn().
  TestFlashAttention.test_backward_bias: TestFlashAttention#test_backward_bias().
  TestFlashAttention.test_backward_segment_ids: TestFlashAttention#test_backward_segment_ids().
  TestFlashAttention.test_backward_layer_bias: TestFlashAttention#test_backward_layer_bias().
  TestFlashAttention.test_backward_dropout: TestFlashAttention#test_backward_dropout().
  TestFlashAttention.test_backward_mesh: TestFlashAttention#test_backward_mesh().
  TestFlashAttention.test_extend_step: TestFlashAttention#test_extend_step().
  TestFlashAttention.test_extend_step_float32: TestFlashAttention#test_extend_step_float32().
  TestFlashAttention.test_extend_step_mesh: TestFlashAttention#test_extend_step_mesh().
  TestFlashAttention.loss: TestFlashAttention#loss().
  TestFlashAttention.extend_one_step: TestFlashAttention#extend_one_step().
  jax_fn_mask.mask: jax_fn_mask().mask().
  TestFlashAttention.test_dropout_support.OtherDropout: TestFlashAttention#test_dropout_support().OtherDropout#
  TestFlashAttention: TestFlashAttention#
  TestFlashAttention.test_extend_step_paged_mesh: TestFlashAttention#test_extend_step_paged_mesh().
  TestFlashAttention.use_smaller_gpu_block: TestFlashAttention#use_smaller_gpu_block().
  TestFlashAttention.check_finite: TestFlashAttention#check_finite().
  DummyModel.forward: DummyModel#forward().
---
# Module: [`axlearn/common/flash_attention/layer_test.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py)

## Classes
### `Config`  ·  implements/extends Config
- def: [`axlearn/common/flash_attention/layer_test.py:178`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L178)
- signature: `class Config(BaseLayer.Config):`
- members:
  - `layer` — [`L179`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L179)
- uses (calls/refs, reference-scoped): [`default_config`](../config.md#Configurable.default_config), [`BaseLayer`](../base_layer.md#BaseLayer), [`config_class`](../config.md#config_class), [`Config`](../base_layer.md#BaseLayer.Config), [`GroupedQueryAttention`](../attention.md#GroupedQueryAttention), [`Config`](../attention.md#MultiheadAttention.Config)
- used by: [`test_backend_override_modifier`](layer_test.md#TestFlashAttention.test_backend_override_modifier), [`test_backend_override_modifier_ignores_none`](layer_test.md#TestFlashAttention.test_backend_override_modifier_ignores_none), [`__init__`](layer_test.md#DummyModel.__init__)

### `DummyModel`  ·  implements/extends BaseLayer
- def: [`axlearn/common/flash_attention/layer_test.py:174`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L174)
- doc: A dummy model.
- signature: `class DummyModel(BaseLayer):`
- members:
  - `forward(self, *, query, key, value, attention_logit_biases, segment_ids)` — [`L186`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L186)
- protocol/private: `__init__`[`L181`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L181)
- uses (calls/refs, reference-scoped): [`config`](../config.md#Configurable.config), [`BaseLayer`](../base_layer.md#BaseLayer), [`Module`](../module.md#Module), [`_add_child`](../module.md#Module._add_child), [`__init__`](../base_layer.md#BaseLayer.__init__), [`layer`](layer_test.md#DummyModel.Config.layer), [`Config`](layer_test.md#DummyModel.Config)
- used by: [`BaseLayer`](../base_layer.md#BaseLayer), [`_test_backward`](layer_test.md#TestFlashAttention._test_backward), [`test_backend_override_modifier`](layer_test.md#TestFlashAttention.test_backend_override_modifier), [`test_backend_override_modifier_ignores_none`](layer_test.md#TestFlashAttention.test_backend_override_modifier_ignores_none)

### `OtherDropout`  ·  implements/extends Dropout
- def: [`axlearn/common/flash_attention/layer_test.py:292`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L292)
- signature: `class OtherDropout(Dropout):`
- uses (calls/refs, reference-scoped): [`Dropout`](../layers.md#Dropout)
- used by: [`Dropout`](../layers.md#Dropout), [`test_dropout_support`](layer_test.md#TestFlashAttention.test_dropout_support)

### `TestFlashAttention`  ·  implements/extends TestCase
- def: [`axlearn/common/flash_attention/layer_test.py:200`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L200)
- doc: Tests FlashAttention layer.
- signature: `class TestFlashAttention(TestCase):`
- members:
  - `_test_backward(self, *, batch, seq_len, num_heads, num_kv_heads, per_head_dim, mesh, mesh_axis_names, query_len_multiplier=1, attn_type="full", use_bias=False, use_segment_ids=False, set_layer_bias_recursively=False, dropout_rate=0)` — [`L608`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L608) — Shared backward test: compares gradients of FlashAttention against reference.
  - `_test_extend_step(self, *, batch, seq_len, num_heads, num_kv_heads, per_head_dim, mesh, mesh_axis_names, attn_type, dtype=jnp.bfloat16)` — [`L842`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L842) — Shared extend_step test: compares autoregressive decoding against prefill.
  - `_test_forward(self, *, batch, seq_len, num_heads, num_kv_heads, per_head_dim, mesh, mesh_axis_names, query_len_multiplier=1, attn_type="full", use_bias=False, use_segment_ids=False, input_dtype=jnp.bfloat16, dropout_rate=0)` — [`L431`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L431) — Shared forward test: compares FlashAttention output against reference.
  - `as_partition_spec(pytree: CompositeAttentionBias)` — [`L388`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L388)
  - `as_tensor_bias(bias: Tensor)` — [`L385`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L385)
  - `check_finite(x)` — [`L1396`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L1396)
  - `extend_one_step(params, inputs, layer)` — [`L995`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L995)
  - `loss(params, inputs, layer)` — [`L702`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L702)
  - `loss_fn(params)` — [`L1370`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L1370)
  - `test_backend(self, batch, seq_len, num_heads, num_kv_heads, per_head_dim, mesh, mesh_axis_names)` — [`L350`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L350)
  - `test_backend_override_modifier(self)` — [`L1472`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L1472) — Tests BackendOverrideModifier.
  - `test_backend_override_modifier_ignores_none(self)` — [`L1493`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L1493) — Tests that BackendOverrideModifier ignores overrides values of None.
  - `test_backward_attn_types(self, batch, seq_len, num_heads, num_kv_heads, per_head_dim, mesh, mesh_axis_names, attn_type)` — [`L749`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L749) — Tests backward with each mask type across attention configs.
  - `test_backward_bias(self)` — [`L784`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L784) — Tests backward with explicit attention bias tensor.
  - `test_backward_cross_attn(self, query_len_multiplier, attn_type)` — [`L776`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L776) — Tests backward with different Q/K sequence lengths.
  - `test_backward_dropout(self)` — [`L806`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L806) — Tests backward with dropout (needs segment_ids for gradient comparison).
  - `test_backward_layer_bias(self, set_layer_bias_recursively)` — [`L798`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L798) — Tests backward with/without recursive layer bias.
  - `test_backward_mesh(self, batch, seq_len, num_heads, num_kv_heads, per_head_dim, mesh, mesh_axis_names)` — [`L817`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L817) — Tests backward with various mesh sharding configurations.
  - `test_backward_segment_ids(self, attn_type)` — [`L789`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L789) — Tests backward with segment ID masking.
  - `test_create_layer_parameter_specs_with_logit_sink(self, bsnh_partition_spec, expected_mesh_axes, test_description)` — [`L1416`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L1416) — Tests _create_layer_parameter_specs with different partition spec lengths.
  - `test_create_layer_parameter_specs_without_logit_sink(self)` — [`L1448`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L1448) — Tests _create_layer_parameter_specs when logit_sink is disabled.
  - `test_dropout_support(self)` — [`L289`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L289) — Tests that FlashAttention errors out when custom dropout is used.
  - `test_extend_step(self, batch, seq_len, num_heads, num_kv_heads, per_head_dim, mesh, mesh_axis_names, attn_type)` — [`L1055`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L1055) — Tests extend_step with each KV cache type across attention configs.
  - `test_extend_step_float32(self, attn_type)` — [`L1079`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L1079) — Tests extend_step with float32 dtype.
  - `test_extend_step_mesh(self, batch, seq_len, num_heads, num_kv_heads, per_head_dim, mesh, mesh_axis_names)` — [`L1092`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L1092) — Tests extend_step with mesh sharding.
  - `test_extend_step_paged_mesh(self)` — [`L1115`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L1115) — Tests paged extend_step with model-only sharding.
  - `test_forward_attn_types(self, batch, seq_len, num_heads, num_kv_heads, per_head_dim, mesh, mesh_axis_names, attn_type)` — [`L516`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L516) — Tests forward with each mask type across attention configs (MHA, GQA, head dims).
  - `test_forward_bias(self)` — [`L551`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L551) — Tests forward with explicit attention bias tensor.
  - `test_forward_cross_attn(self, query_len_multiplier, attn_type)` — [`L543`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L543) — Tests forward with different Q/K sequence lengths (cross-attention shapes).
  - `test_forward_dropout(self)` — [`L572`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L572) — Tests forward with dropout (needs segment_ids for output comparison).
  - `test_forward_float32(self)` — [`L564`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L564) — Tests forward with float32 input dtype.
  - `test_forward_mesh(self, batch, seq_len, num_heads, num_kv_heads, per_head_dim, mesh, mesh_axis_names)` — [`L583`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L583) — Tests forward with various mesh sharding configurations.
  - `test_forward_segment_ids(self, attn_type)` — [`L556`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L556) — Tests forward with segment ID masking.
  - `test_gqa_kv_heads(self)` — [`L305`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L305) — Tests _maybe_repeat_kv_heads.
  - `test_incremental_prefill(self, batch, seq_len, num_heads, num_kv_heads, per_head_dim, mesh, mesh_axis_names, prefix_len, suffix_len)` — [`L1602`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L1602) — Tests that incremental prefill through FlashAttention layer produces same results as full
  - `test_logit_sink(self, batch, seq_len, num_heads, num_kv_heads, per_head_dim, mesh, mesh_axis_names, logit_sink, attn_type)` — [`L1135`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L1135) — Tests logit sink functionality in FlashAttention.
  - `test_logit_sink_backward_pass(self, logit_sink)` — [`L1343`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L1343) — Tests that gradients flow correctly through logit sink.
  - `test_logit_sink_numerical_stability(self)` — [`L1265`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L1265) — Tests that logit sink improves numerical stability with extreme logits.
  - `test_logit_sink_parameter_initialization(self)` — [`L1232`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L1232) — Tests that logit sink parameters are properly initialized.
  - `test_shard_biases(self, batch, seq_len, num_heads, num_kv_heads, per_head_dim, mesh, mesh_axis_names)` — [`L379`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L379)
  - `test_softmax_scale(self, softmax_scale)` — [`L1522`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L1522) — Tests that softmax_scale is correctly applied in flash attention.
  - `use_smaller_gpu_block(k_seq_len: int, q_seq_len: int)` — [`L658`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L658)
- protocol/private: `_ATTN_CONFIGS`[`L210`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L210), `_MESH_CONFIGS`[`L241`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L241)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`default_config`](../config.md#Configurable.default_config), [`instantiate`](../config.md#Configurable.Config.instantiate), [`functional`](../module.md#functional), [`initialize_parameters_recursively`](../base_layer.md#BaseLayer.initialize_parameters_recursively), [`MaskFnAttentionBias`](../attention_bias.md#MaskFnAttentionBias), [`_prepare_layers`](layer_test.md#_prepare_layers), [`CausalAttentionBias`](../attention_bias.md#CausalAttentionBias), [`KVCache`](../kv_cache/kv_cache.md#KVCache), [`default_config`](layer.md#FlashAttention.default_config), [`FlashAttention`](layer.md#FlashAttention), [`init_states`](../attention.md#MultiheadAttention.init_states), [`QKVLinear`](../attention.md#QKVLinear), [`default_config`](../attention_bias.md#CausalAttentionBias.default_config), [`SlidingWindowAttentionBias`](../attention_bias.md#SlidingWindowAttentionBias), [`Dropout`](../layers.md#Dropout), [`GroupedQueryAttention`](../attention.md#GroupedQueryAttention), [`CompositeAttentionBias`](../attention_bias.md#CompositeAttentionBias), [`SegmentIdAttentionBias`](../attention_bias.md#SegmentIdAttentionBias), [`set_bias_recursively`](../layers.md#set_bias_recursively), [`default_config`](../attention_bias.md#MaskFnAttentionBias.default_config), [`GroupedQKVLinear`](../attention.md#GroupedQKVLinear), [`default_config`](../attention_bias.md#SlidingWindowAttentionBias.default_config), [`PagedKVCache`](../kv_cache/paged_kv_cache.md#PagedKVCache), [`layer`](layer_test.md#DummyModel.Config.layer), [`jax_fn_mask`](layer_test.md#jax_fn_mask), [`_fake_inputs`](layer_test.md#_fake_inputs), [`Config`](layer_test.md#DummyModel.Config), [`TensorAttentionBias`](../attention_bias.md#TensorAttentionBias), [`DummyModel`](layer_test.md#DummyModel), [`default_mha_dim_to_partition_spec`](layer.md#default_mha_dim_to_partition_spec), [`default_output_dim_to_partition_spec`](layer.md#default_output_dim_to_partition_spec), [`backend_overrides`](layer.md#FlashAttention.Config.backend_overrides), [`BackendOverrideModifier`](layer.md#BackendOverrideModifier), [`SlidingWindowKVCache`](../kv_cache/sliding_window_kv_cache.md#SlidingWindowKVCache), [`OtherDropout`](layer_test.md#TestFlashAttention.test_dropout_support.OtherDropout)  (3 test-only)
- used by: (1 test-only callers)

## Functions
- `_fake_inputs(*, batch: int, num_heads: int, kv_len: int, query_len: int, hidden_dim: int, use_bias: bool, use_segment_ids: bool, input_dtype: jnp.dtype = jnp.bfloat16)` — [`L61`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L61)
- `_prepare_layers(*, num_heads, num_kv_heads, per_head_dim, mesh_axis_names, mask, kv_cache=KVCache.default_config(), inference=False, set_layer_bias_recursively=False, tpu_block_size=512, dropout_rate=0)` — [`L104`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L104)
- `jax_fn_mask(sliding_window_size: int)` — [`L166`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L166)
- `mask(query_position: Tensor, key_position: Tensor)` — [`L167`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/layer_test.py#L167)

