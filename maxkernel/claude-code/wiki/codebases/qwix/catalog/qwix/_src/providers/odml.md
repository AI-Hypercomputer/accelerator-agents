---
title: 'Module: qwix/_src/providers/odml.py'
type: catalog
provenance: extracted
module: qwix/_src/providers/odml.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.providers.odml`/Odml
symbols:
  OdmlQatProvider._fake_quant: QatProvider#_fake_quant().
  OdmlConversionProvider._fake_quant: ConversionProvider#_fake_quant().
  OdmlQatProvider: QatProvider#
  OdmlQatProvider.get_intercept_map: QatProvider#get_intercept_map().
  OdmlQatProvider._update_and_get_quant_stat: QatProvider#_update_and_get_quant_stat().
  OdmlConversionProvider: ConversionProvider#
  OdmlConversionProvider._compute_static_scale_zero_point: ConversionProvider#_compute_static_scale_zero_point().
  OdmlQatProvider.process_model_inputs: QatProvider#process_model_inputs().
  OdmlQatProvider.get_interceptors: QatProvider#get_interceptors().
  OdmlQatProvider._init_rule: QatProvider#_init_rule().
  OdmlQatProvider._initial_run_complete: QatProvider#_initial_run_complete.
  OdmlConversionProvider._fake_quant_op: ConversionProvider#_fake_quant_op().
  OdmlQatProvider.nn_param: QatProvider#nn_param().
  OdmlConversionProvider._flatten_dot_general: ConversionProvider#_flatten_dot_general().
  OdmlQatProvider._ops: QatProvider#_ops.
  OdmlQatProvider.__init__: QatProvider#__init__().
  OdmlConversionProvider.__init__: ConversionProvider#__init__().
  OdmlConversionProvider.get_intercept_map: ConversionProvider#get_intercept_map().
  OdmlQatProvider._fixed_range_for_inputs: QatProvider#_fixed_range_for_inputs.
  OdmlQatProvider._fixed_range_for_outputs: QatProvider#_fixed_range_for_outputs.
  OdmlQatProvider._strict: QatProvider#_strict.
  OdmlConversionProvider._flatten_params: ConversionProvider#_flatten_params.
  OdmlConversionProvider._quant_stats: ConversionProvider#_quant_stats.
  OdmlConversionProvider._get_attributes: ConversionProvider#_get_attributes().
  OdmlQatProvider.process_model_output: QatProvider#process_model_output().
---
# Module: [`qwix/_src/providers/odml.py`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py)

## Classes
### `OdmlConversionProvider`  ·  implements/extends OdmlQatProvider
- def: [`qwix/_src/providers/odml.py:331`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L331)
- doc: Quantization provider for ODML conversion.
- signature: `class OdmlConversionProvider(OdmlQatProvider):`
- members:
  - `_compute_static_scale_zero_point(self, how: qarray.HowToQuantize, quant_stat_name: str)` — [`L441`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L441) — Statically compute the scale and zero point for weights or activations.
  - `_flatten_dot_general(self, *args, _dot_general, **kwargs)` — [`L385`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L385) — Flatten N-D weights to 2-D to support channelwise quantization.
  - `_get_attributes(self, *, scale: jax.Array | None, zp: jax.Array | None, dtype: jax.typing.DTypeLike, is_weight: bool)` — [`L456`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L456) — Return the attributes for the fake_quant composite.
  - `get_intercept_map(self)` — [`L376`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L376)
- protocol/private: `__init__`[`L364`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L364), `_fake_quant`[`L401`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L401), `_fake_quant_op`[`L431`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L431), `_flatten_params`[`L373`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L373), `_quant_stats`[`L374`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L374)
- uses (calls/refs, reference-scoped): [`QuantizationRule`](../qconfig.md#QuantizationRule), [`HowToQuantize`](../core/qarray.md#HowToQuantize), [`qtype`](../core/qarray.md#HowToQuantize.qtype), [`quantize`](../core/qarray.md#quantize), [`AuxDataKey`](odml_ops.md#AuxDataKey), [`dequantize`](../core/qarray.md#dequantize), [`tiled_axes`](../core/qarray.md#HowToQuantize.tiled_axes), [`get`](../aux_data.md#get), [`calibrate`](../core/qarray.md#calibrate), [`OdmlQatProvider`](odml.md#OdmlQatProvider), [`quantize_with_scale_zero_point`](../core/qarray.md#quantize_with_scale_zero_point), [`compute_scale_zero_point`](../core/qarray.md#compute_scale_zero_point), [`get_calibration`](../averaging.md#SimpleMovingAverage.get_calibration), [`get_intercept_map`](odml.md#OdmlQatProvider.get_intercept_map), [`SimpleMovingAverage`](../averaging.md#SimpleMovingAverage), [`WEIGHT_NAME`](odml_ops.md#AuxDataKey.WEIGHT_NAME), [`get_current_module_path`](../utils/flax_util.md#get_current_module_path), [`__init__`](odml.md#OdmlQatProvider.__init__)
- used by: [`_fake_quant`](odml.md#OdmlQatProvider._fake_quant), [`test_cnn_srq`](../../../integration_tests/odml_cnn_test.md#OdmlCnnTest.test_cnn_srq), [`test_srq`](../../../integration_tests/odml_coverage_test.md#OdmlTest.test_srq), [`OdmlQatProvider`](odml.md#OdmlQatProvider), [`test_drq`](../../../integration_tests/odml_coverage_test.md#OdmlTest.test_drq), [`test_nnx`](../../../integration_tests/odml_coverage_test.md#OdmlTest.test_nnx), [`test_partial_quantization_unet_succeeds`](../../../integration_tests/odml_coverage_test.md#OdmlTest.test_partial_quantization_unet_succeeds), [`test_weight_only`](../../../integration_tests/odml_coverage_test.md#OdmlTest.test_weight_only), [`test_vae_separate_export`](../../../integration_tests/odml_coverage_test.md#OdmlTest.test_vae_separate_export), [`get_intercept_map`](odml.md#OdmlQatProvider.get_intercept_map)  (8 test-only)

### `OdmlQatProvider`  ·  implements/extends QuantizationProvider
- def: [`qwix/_src/providers/odml.py:35`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L35) — documented in [qwix-_src-providers-odml](../../../../concepts/qwix-_src-providers-odml.md)
- doc: QAT provider for ODML.
- signature: `class OdmlQatProvider(qconfig.QuantizationProvider):`
- members:
  - `__init__(self, rules: Sequence[qconfig.QuantizationRule], *, disable_per_channel_weights: bool = False, fixed_range_for_inputs: tuple[float, float] | None = None, fixed_range_for_outputs: tuple[float, float] | None = None, strict: bool = True)` — [`L70`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L70) — Constructor.
  - `_fake_quant(self, array: jax.Array, how: qarray.HowToQuantize, quant_stat_name: str | None = None)` — [`L255`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L255) — Numerical operation used by intercepted model ops to fake-quantize tensors. — documented in [qwix-_src-providers-odml](../../../../concepts/qwix-_src-providers-odml.md)
  - `_init_rule(self, rule: qconfig.QuantizationRule)` — [`L119`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L119) — Set ODML specific default values.
  - `_update_and_get_quant_stat(self, name: str, calibration: averaging.Calibration, calibration_is_fixed_range: bool)` — [`L304`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L304) — Updates the running quantization statistics and returns the average. — documented in [qwix-_src-averaging](../../../../concepts/qwix-_src-averaging.md)
  - `get_intercept_map(self)` — [`L176`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L176) — Returns a map of function names to their intercepted implementations. — documented in [qwix-_src-providers-odml](../../../../concepts/qwix-_src-providers-odml.md)
  - `get_interceptors(self)` — [`L150`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L150) — Returns a list of interceptor factories. — documented in [qwix-_src-interception](../../../../concepts/qwix-_src-interception.md)
  - `nn_param(self, module: nn.Module, name: str, init_fn: Callable[..., Any], *init_args, unbox: bool = True, **init_kwargs)` — [`L129`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L129) — Intercepts nn.Module.param to associate weight_name aux_data.
  - `process_model_inputs(self, model: Any, model_args: Any, model_kwargs: Any)` — [`L197`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L197) — Prepares model activations for quantization metadata propagation. — documented in [qwix-_src-providers-odml](../../../../concepts/qwix-_src-providers-odml.md)
  - `process_model_output(self, method_name: str, model_output: Any)` — [`L240`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L240) — Quantize the output of the model.
- protocol/private: `_fixed_range_for_inputs`[`L99`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L99), `_fixed_range_for_outputs`[`L100`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L100), `_initial_run_complete`[`L242`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L242), `_ops`[`L102`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L102), `_strict`[`L101`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml.py#L101)
- uses (calls/refs, reference-scoped): [`QuantizationRule`](../qconfig.md#QuantizationRule), [`HowToQuantize`](../core/qarray.md#HowToQuantize), [`act_qtype`](../qconfig.md#QuantizationRule.act_qtype), [`qtype`](../core/qarray.md#HowToQuantize.qtype), [`AuxDataKey`](odml_ops.md#AuxDataKey), [`dequantize`](../core/qarray.md#dequantize), [`tiled_axes`](../core/qarray.md#HowToQuantize.tiled_axes), [`get`](../aux_data.md#get), [`calibration_method`](../core/qarray.md#HowToQuantize.calibration_method), [`_get_current_rule_and_op_id`](../qconfig.md#QuantizationProvider._get_current_rule_and_op_id), [`Interceptor`](../interception.md#Interceptor), [`get_all_ops`](odml_ops.md#get_all_ops), [`_fake_quant`](odml.md#OdmlConversionProvider._fake_quant), [`mapping`](../interception.md#Interceptor.mapping), [`calibrate`](../core/qarray.md#calibrate), [`act_static_scale`](../qconfig.md#QuantizationRule.act_static_scale), [`id`](../interception.md#Interceptor.id), [`QuantizationProvider`](../qconfig.md#QuantizationProvider), [`quantize_with_scale_zero_point`](../core/qarray.md#quantize_with_scale_zero_point), [`compute_scale_zero_point`](../core/qarray.md#compute_scale_zero_point), [`get_intercept_map`](../qconfig.md#QuantizationProvider.get_intercept_map), [`QuantizedOp`](odml_ops.md#QuantizedOp), [`get_calibration`](../averaging.md#SimpleMovingAverage.get_calibration), [`OdmlConversionProvider`](odml.md#OdmlConversionProvider), [`_init_rule`](../qconfig.md#QuantizationProvider._init_rule), [`act_calibration_method`](../qconfig.md#QuantizationRule.act_calibration_method), [`init`](../averaging.md#SimpleMovingAverage.init), [`update`](../averaging.md#SimpleMovingAverage.update), [`SimpleMovingAverage`](../averaging.md#SimpleMovingAverage), [`get_or_create_variable`](../utils/flax_util.md#get_or_create_variable), [`FIXED_RANGE`](odml_ops.md#AuxDataKey.FIXED_RANGE), [`WEIGHT_NAME`](odml_ops.md#AuxDataKey.WEIGHT_NAME), [`Calibration`](../averaging.md#Calibration.Calibration), [`clear`](../aux_data.md#clear), [`__init__`](../qconfig.md#QuantizationProvider.__init__), [`clip_to_calibration`](../core/qarray.md#clip_to_calibration), [`should_update_quant_stats`](../utils/flax_util.md#should_update_quant_stats), [`PRIMITIVE_BIND_KEY`](../interception.md#PRIMITIVE_BIND_KEY), [`get_intercept_map`](odml.md#OdmlConversionProvider.get_intercept_map), [`PrimitiveBindOp`](odml_ops.md#PrimitiveBindOp)  (+2 more)
- used by: [`test_cnn_srq`](../../../integration_tests/odml_cnn_test.md#OdmlCnnTest.test_cnn_srq), [`test_srq`](../../../integration_tests/odml_coverage_test.md#OdmlTest.test_srq), [`test_drq`](../../../integration_tests/odml_coverage_test.md#OdmlTest.test_drq), [`test_nnx`](../../../integration_tests/odml_coverage_test.md#OdmlTest.test_nnx), [`QuantizationProvider`](../qconfig.md#QuantizationProvider), [`get_intercept_map`](../qconfig.md#QuantizationProvider.get_intercept_map), [`test_partial_quantization_unet_succeeds`](../../../integration_tests/odml_coverage_test.md#OdmlTest.test_partial_quantization_unet_succeeds), [`test_vae_separate_export`](../../../integration_tests/odml_coverage_test.md#OdmlTest.test_vae_separate_export), [`OdmlConversionProvider`](odml.md#OdmlConversionProvider), [`_init_rule`](../qconfig.md#QuantizationProvider._init_rule), [`test_model_output_pytree`](../../../integration_tests/odml_coverage_test.md#OdmlTest.test_model_output_pytree), [`test_partial_quantization`](../../../integration_tests/odml_coverage_test.md#OdmlTest.test_partial_quantization), [`get_interceptors`](../qconfig.md#QuantizationProvider.get_interceptors), [`process_model_output`](../qconfig.md#QuantizationProvider.process_model_output), [`process_model_inputs`](../qconfig.md#QuantizationProvider.process_model_inputs), [`__init__`](odml.md#OdmlConversionProvider.__init__), [`get_intercept_map`](odml.md#OdmlConversionProvider.get_intercept_map)  (13 test-only)

