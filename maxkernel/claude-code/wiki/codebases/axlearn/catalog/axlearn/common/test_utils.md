---
title: 'Module: axlearn/common/test_utils.py'
type: catalog
provenance: extracted
module: axlearn/common/test_utils.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.test_utils`/
symbols:
  TestCase: TestCase#
  TestCase.assertNestedAllClose: TestCase#assertNestedAllClose().
  read_per_param_settings: read_per_param_settings().
  assert_allclose: assert_allclose().
  bind_module: bind_module().
  mock_trainer_config: mock_trainer_config().
  initialize_parameters_with_prebuilt: initialize_parameters_with_prebuilt().
  bind_layer: bind_layer().
  read_param_init_specs_recursively: read_param_init_specs_recursively().
  Tolerance: Tolerance#
  TestCase._compute_layer_outputs: TestCase#_compute_layer_outputs().
  _complete_param_init_spec_tree: _complete_param_init_spec_tree().
  Tolerance.rtol: Tolerance#rtol.
  Tolerance.atol: Tolerance#atol.
  TestCase.assertAllCloseWithOutliers: TestCase#assertAllCloseWithOutliers().
  read_param_init_specs_recursively.patched_init: read_param_init_specs_recursively().patched_init().
  is_supported_mesh_shape: is_supported_mesh_shape().
  TestCase.assertNestedAllCloseWithOutliers: TestCase#assertNestedAllCloseWithOutliers().
  _cast_ordered_dict: _cast_ordered_dict().
  TestCase.setUp: TestCase#setUp().
  ParamInitSpec: ParamInitSpec#
  ThirdPartyInitializer.Config: ThirdPartyInitializer#Config#
  ThirdPartyInitializer.debug_string: ThirdPartyInitializer#debug_string().
  set_threefry_partitionable: set_threefry_partitionable().
  as_local_tensor: as_local_tensor().
  DummyForwardModel.forward: DummyForwardModel#forward().
  TestWithTemporaryCWD._temp_root: TestWithTemporaryCWD#_temp_root.
  read_per_param_settings.patched_register_per_param_settings: read_per_param_settings().patched_register_per_param_settings().
  ParameterConversionFn.__call__: ParameterConversionFn#__call__().
  TestCase.tearDown: TestCase#tearDown().
  DummyForwardModel.predict: DummyForwardModel#predict().
  ThirdPartyInitializer.Config.library: ThirdPartyInitializer#Config#library.
  TestCase.assertNestedEqual: TestCase#assertNestedEqual().
  dummy_segments_positions: dummy_segments_positions().
  L: L.
  M: M.
  ParamInitSpec.initializer: ParamInitSpec#initializer.
  NestedParamInitSpec: NestedParamInitSpec.
  read_param_init_specs_recursively.patched_vmap: read_param_init_specs_recursively().patched_vmap().
  read_param_init_specs_recursively.patched_vmap.wrapped_fn: read_param_init_specs_recursively().patched_vmap().wrapped_fn().
  set_threefry_partitionable.decorator_set: set_threefry_partitionable().decorator_set().
  assert_not_allclose: assert_not_allclose().
  DummyForwardModel: DummyForwardModel#
  TestWithTemporaryCWD: TestWithTemporaryCWD#
  prng_impl: prng_impl().
  ThirdPartyInitializer: ThirdPartyInitializer#
  dummy_padding_mask: dummy_padding_mask().
  take_segment: take_segment().
  pytest_addoption_atomic: pytest_addoption_atomic().
  _PYTEST_OPT_REGISTERED: _PYTEST_OPT_REGISTERED.
  TestCase.data_dir: TestCase#data_dir().
  prng_impl.switch: prng_impl().switch().
  _complete_param_init_spec_tree.is_leaf: _complete_param_init_spec_tree().is_leaf().
  _complete_param_init_spec_tree.replace_keys: _complete_param_init_spec_tree().replace_keys().
  clean_hlo: clean_hlo().
  ParameterConversionFn: ParameterConversionFn#
  TestCase._jax_backend: TestCase#_jax_backend().
  ParamInitSpec.shape: ParamInitSpec#shape.
  ParamInitSpec.fan_axes: ParamInitSpec#fan_axes.
  read_per_param_settings.patched_init: read_per_param_settings().patched_init().
  temp_chdir: temp_chdir().
  set_threefry_partitionable.decorator_set.wrapper: set_threefry_partitionable().decorator_set().wrapper().
  is_supported_platform: is_supported_platform().
  TestWithTemporaryCWD.run: TestWithTemporaryCWD#run().
---
# Module: [`axlearn/common/test_utils.py`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py)

## Classes
### `Config`  ·  implements/extends Config
- def: [`axlearn/common/test_utils.py:384`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L384)
- signature: `class Config(Initializer.Config):`
- members:
  - `library` — [`L385`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L385)
- uses (calls/refs, reference-scoped): [`config_class`](config.md#config_class), [`REQUIRED`](config.md#REQUIRED), [`Required`](config.md#Required), [`Config`](config.md#Configurable.Config), [`Initializer`](param_init.md#Initializer)

### `DummyForwardModel`  ·  implements/extends BaseModel
- def: [`axlearn/common/test_utils.py:331`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L331)
- doc: A dummy model whose `forward` returns (0, input_batch\["aux"\]).
- signature: `class DummyForwardModel(BaseModel):`
- members:
  - `forward(self, input_batch: NestedTensor, **kwargs)` — [`L337`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L337)
  - `predict(self, input_batch: NestedTensor)` — [`L341`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L341)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`NestedTensor`](utils.md#NestedTensor), [`BaseModel`](base_model.md#BaseModel)
- used by: [`BaseModel`](base_model.md#BaseModel), [`forward`](base_model.md#BaseModel.forward)

### `ParamInitSpec`
- def: [`axlearn/common/test_utils.py:371`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L371)
- signature: `class ParamInitSpec:`
- members:
  - `fan_axes` — [`L374`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L374)
  - `initializer` — [`L373`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L373)
  - `shape` — [`L372`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L372)
- uses (calls/refs, reference-scoped): [`FanAxes`](param_init.md#FanAxes), [`Initializer`](param_init.md#Initializer)
- used by: (4 test-only callers)

### `ParameterConversionFn`  ·  implements/extends Protocol
- def: [`axlearn/common/test_utils.py:153`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L153)
- signature: `class ParameterConversionFn(Protocol):`
- members:
  - `__call__(self, src: Any, *, dst_layer: BaseLayer)` — [`L154`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L154) — Converts parameters from `src` to parameters for `dst_layer`.
- uses (calls/refs, reference-scoped): [`NestedTensor`](utils.md#NestedTensor), [`BaseLayer`](base_layer.md#BaseLayer)
- used by: (1 test-only callers)

### `TestCase`  ·  implements/extends TestCase
- def: [`axlearn/common/test_utils.py:163`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L163)
- doc: Base test class.
- signature: `class TestCase(parameterized.TestCase):`
- members:
  - `assertAllCloseWithOutliers(self, actual, desired, *, tolerance_map: dict[float, Tolerance])` — [`L277`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L277) — Like np.testing.assert_allclose, but allows outlier percentiles to be specified.
  - `assertNestedAllClose(self, a, b, atol=0.000001, rtol=0.001)` — [`L244`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L244)
  - `assertNestedAllCloseWithOutliers(self, actual: NestedTensor, desired: NestedTensor, *, tolerance_map: dict[float, Tolerance])` — [`L316`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L316) — A nested variant of `assertAllCloseWithOutliers`.
  - `assertNestedEqual(self, a, b)` — [`L264`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L264)
  - `data_dir(self)` — [`L167`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L167)
  - `setUp(self)` — [`L170`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L170)
  - `tearDown(self)` — [`L179`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L179)
- protocol/private: `_compute_layer_outputs`[`L183`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L183), `_jax_backend`[`L175`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L175)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`NestedTensor`](utils.md#NestedTensor), [`BaseLayer`](base_layer.md#BaseLayer), [`functional`](module.md#functional), [`initialize_parameters_recursively`](base_layer.md#BaseLayer.initialize_parameters_recursively), [`flatten_items`](utils.md#flatten_items), [`as_tensor`](utils.md#as_tensor), [`prune_empty`](utils.md#prune_empty), `BaseTransformerTest`, [`pop_data_dir`](utils.md#pop_data_dir), [`push_data_dir`](utils.md#push_data_dir), `ALiBiAttentionLogitBiasLayerTest`, `BottleNeckAdapterTransformerLayerTest`, `CausalAttentionLogitBiasLayerTest`, [`ConditionalUpdateTransformationTest`](update_transformation_test.md#ConditionalUpdateTransformationTest), `ConfigHelperTest`, `FullAttentionLogitBiasLayerTest`, [`GmmTest`](megablock/gmm_gpu_test.md#GmmTest), `LogitSinkTest`, `MultiheadAttentionTest`, `MultiheadLinearInitTest`, [`OverrideInplaceUpdateTransformationTest`](update_transformation_test.md#OverrideInplaceUpdateTransformationTest), `ParallelTransformerTest`, `PerDimScaleTest`, `PositionalEmbeddingTest`, `QKVLinearTest`, `RoFormerSinusoidalPositionalEmbeddingAgainstLLaMATest`, `RoFormerSinusoidalPositionalEmbeddingTest`, `ScaleFunctionsTest`, `ScaleKeyTest`, `ScaleQueryTest`, `SymmetricALiBiAttentionLogitBiasLayerTest`, [`TestFlashAttention`](flash_attention/layer_test.md#TestFlashAttention), `TransformerAttentionLayerTest`, `TransformerFeedForwardLayerTest`, `TransformerXLTest`, [`UpdateTransformationTest`](update_transformation_test.md#UpdateTransformationTest), [`UpdatesTest`](update_transformation_test.md#UpdatesTest), `YarnScaleRopeParametersTest`  (7 test-only)
- used by: [`test_wrapped_partitioned_gradient_transformation`](update_transformation_test.md#UpdateTransformationTest.test_wrapped_partitioned_gradient_transformation), [`_test_backward`](flash_attention/layer_test.md#TestFlashAttention._test_backward), `_compare_layers`, `test_gqa_forward`, `test_causal`, `_test_extend_step`, `test_sliding_window`, [`test_logit_sink`](flash_attention/layer_test.md#TestFlashAttention.test_logit_sink), `test_sigmoid_compute_attention`, `test_qkv_equality`, `test_repeated_layer_with_custom_carry`, [`_test_extend_step`](flash_attention/layer_test.md#TestFlashAttention._test_extend_step), `test_v2_structure`, [`_test_forward`](flash_attention/layer_test.md#TestFlashAttention._test_forward), `test_passthrough_update_layer_kwargs`, `test_input_linear_variants`, `test_with_golden_value`, `test_prescaled_kv_share`, `test_qlinear`, `test_scale_query`, `test_v2_structure`, `test_skip_connection`, `test_add_value_rms_norm_summary`, `test_multihead_attention_xl`, `test_pipeline_return_aux`, [`test_incremental_prefill`](flash_attention/layer_test.md#TestFlashAttention.test_incremental_prefill), [`test_mask`](update_transformation_test.md#UpdatesTest.test_mask), `test_scale_key`, [`test_gmm_lib`](megablock/gmm_gpu_test.md#GmmTest.test_gmm_lib), [`test_gmm_benchmark_fn`](megablock/gmm_gpu_test.md#GmmTest.test_gmm_benchmark_fn), `test_scale_query_key_dim_dependence`, [`test_param_values`](update_transformation_test.md#UpdatesTest.test_param_values), `test_scale_query_key`, `BaseTransformerTest`, [`check_state_and_model_params_equal`](update_transformation_test.md#ConditionalUpdateTransformationTest.check_state_and_model_params_equal), `ALiBiAttentionLogitBiasLayerTest`, `BottleNeckAdapterTransformerLayerTest`, `CausalAttentionLogitBiasLayerTest`, [`ConditionalUpdateTransformationTest`](update_transformation_test.md#ConditionalUpdateTransformationTest), `ConfigHelperTest`  (+23 more; 1 test-only)

### `TestWithTemporaryCWD`  ·  implements/extends TestCase
- def: [`axlearn/common/test_utils.py:345`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L345)
- doc: Run all tests in a temp directory to isolate from local env.
- signature: `class TestWithTemporaryCWD(TestCase):`
- members:
  - `run(self, result=None)` — [`L348`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L348)
- protocol/private: `_temp_root`[`L352`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L352)
- uses (calls/refs, reference-scoped): (2 test-only callers)
- used by: (1 test-only callers)

### `ThirdPartyInitializer`  ·  implements/extends Initializer
- def: [`axlearn/common/test_utils.py:379`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L379)
- doc: An stand-in initializer that indicates that initialization is delegated to a third party
- signature: `class ThirdPartyInitializer(Initializer):`
- members:
  - `debug_string(self, name: Optional[str] = None, shape: Optional[Shape] = None, axes: Optional[FanAxes] = None)` — [`L387`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L387)
- uses (calls/refs, reference-scoped): [`config`](config.md#Configurable.config), [`FanAxes`](param_init.md#FanAxes), [`Initializer`](param_init.md#Initializer), [`Shape`](param_init.md#Shape)
- used by: [`Initializer`](param_init.md#Initializer), [`debug_string`](param_init.md#Initializer.debug_string)

### `Tolerance`  ·  implements/extends NamedTuple
- def: [`axlearn/common/test_utils.py:158`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L158)
- signature: `class Tolerance(NamedTuple):`
- members:
  - `atol` — [`L160`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L160)
  - `rtol` — [`L159`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L159)
- used by: [`tolerances_map`](megablock/gmm_gpu_test.md#tolerances_map)  (2 test-only)

## Functions
- `_cast_ordered_dict(params: NestedTensor)` — [`L400`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L400)
- `_complete_param_init_spec_tree(params: NestedTensor, param_init_specs: list[ParamInitSpec], delegates: dict[str, ParamInitSpec])` — [`L406`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L406) — Completes the param_init_specs by replacing certain param paths with proxy Initializers.
- `as_local_tensor(x: Tensor)` — [`L130`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L130)
- `assert_allclose(actual, desired, atol=0.000001, rtol=0.001, err_msg="")` — [`L71`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L71)
- `assert_not_allclose(actual, desired, atol=0.000001, rtol=0.001)` — [`L97`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L97) — Assert that actual and desired are NOT close.
- `bind_layer(layer: ConfigOr[L], *, is_training: bool = True, prng_key: Optional[jax.random.PRNGKey] = None, state: Optional[Nested[Tensor]] = None)` — [`L831`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L831) — Creates a context in which `layer` has state initialized using
- `bind_module(module: ConfigOr[M], *, is_training: bool = True, prng_key: Optional[jax.random.PRNGKey] = None, state: Nested[Tensor])` — [`L776`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L776) — Creates a context in which `module` has `state`.`
- `clean_hlo(hlo: str)` — [`L136`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L136) — Returns a cleaned version of `hlo` with non-functional parts that may impact test reliability
- `decorator_set(fn)` — [`L930`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L930)
- `dummy_padding_mask(*, batch_size: int, max_seq_len: int)` — [`L616`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L616) — Builds a dummy attention mask where non-padding tokens are followed by padding tokens.
- `dummy_segments_positions(batch: int, seq_len: int, *, num_segments: int)` — [`L642`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L642) — Builds dummy segment IDs and corresponding positions.
- `initialize_parameters_with_prebuilt(layer: BaseLayer, *, prng_key: Tensor, prebuilt: Nested[Union[Tensor, ParameterSpec]])` — [`L874`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L874) — Initializes parameters with given prebuilt parameters.
- `is_leaf(v)` — [`L428`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L428)
- `is_supported_mesh_shape(mesh_shape: Sequence[int], devices: Optional[list[jax.Device]] = None)` — [`L119`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L119) — Checks if a function intended for a mesh shape is compatible with the current device(s).
- `is_supported_platform(target_platform: str)` — [`L106`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L106) — Checks if a function intended for a specific platform can be executed on the current one.
- `mock_trainer_config(input_config: InstantiableConfig, model_config: BaseModel.Config, mesh_axis_names: Sequence[str] = ("data", "model"))` — [`L732`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L732)
- `patched_init(self, name, *, prng_key, parameter_spec)` — [`L470`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L470)
- `patched_init(self, name, *, prng_key, parameter_spec)` — [`L534`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L534)
- `patched_register_per_param_settings(settings: NestedTree, *, description: str, path: Optional[str] = None)` — [`L546`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L546)
- `patched_vmap(fn, **vmap_kwargs)` — [`L494`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L494)
- `prng_impl(new_prng_impl: str)` — [`L358`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L358)
- `pytest_addoption_atomic(parser, option, **kwargs)` — [`L721`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L721) — This function allows pytest_addoption to be invoked multiple times atomically.
- `read_param_init_specs_recursively(layer: BaseLayer, *, delegates: Optional[dict[str, ParamInitSpec]] = None)` — [`L453`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L453) — Given a layer, returns all nested parameter initialization specs.
- `read_per_param_settings(module: Any, config_name: str, trainer_config: Optional[TrainerConfigFn] = None)` — [`L507`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L507) — Extracts per-param settings for the given trainer config.
- `replace_keys(v, mapping)` — [`L431`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L431)
- `set_threefry_partitionable(on: bool = False)` — [`L927`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L927) — Helper decorator to enable/disable threefry_partitionable.
- `switch(value)` — [`L361`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L361)
- `take_segment(inputs: Tensor, mask: Tensor, *, pad_value: int = 0)` — [`L697`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L697) — Selects elements from the input corresponding to the given segment.
- `temp_chdir(new_cwd: Union[pathlib.Path, str])` — [`L761`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L761) — Changes into a temp CWD only within the context.
- `wrapped_fn(*args, **kwargs)` — [`L495`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L495)
- `wrapper(*args, **kwargs)` — [`L932`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L932)

## Module values
- `L` — [`L771`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L771)
- `M` — [`L772`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L772)
- `NestedParamInitSpec` — [`L397`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L397)
- `_PYTEST_OPT_REGISTERED` — [`L68`](../../../../../../raw/code/axlearn/axlearn/common/test_utils.py#L68)

