---
title: 'Module: qwix/_src/providers/qt.py'
type: catalog
provenance: extracted
module: qwix/_src/providers/qt.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.providers.qt`/Qt
symbols:
  QtProvider._create_dot_general_qt_config: Provider#_create_dot_general_qt_config().
  QtProvider._create_conv_general_qt_config: Provider#_create_conv_general_qt_config().
  QtProvider._create_ragged_dot_qt_config: Provider#_create_ragged_dot_qt_config().
  QtRule: Rule#
  QtProvider: Provider#
  QtProvider._update_and_get_quant_stat: Provider#_update_and_get_quant_stat().
  QtRule.bwd_qtype: Rule#bwd_qtype.
  QtProvider.conv_general_dilated: Provider#conv_general_dilated().
  QtProvider.get_intercept_map: Provider#get_intercept_map().
  QtProvider.dot_general: Provider#dot_general().
  QtProvider.ragged_dot: Provider#ragged_dot().
  QtRule.disable_channelwise_axes: Rule#disable_channelwise_axes.
  QtProvider.einsum: Provider#einsum().
  QtProvider._init_rule: Provider#_init_rule().
  QtProvider.custom_dot_general: Provider#custom_dot_general().
  QtRule.bwd_calibration_method: Rule#bwd_calibration_method.
  QtRule.additional_qt_config: Rule#additional_qt_config.
  QtRule.bwd_weight_grad_tile_size: Rule#bwd_weight_grad_tile_size.
  QtRule.bwd_stochastic_rounding: Rule#bwd_stochastic_rounding.
  QtRule.channelwise_noise_axes: Rule#channelwise_noise_axes.
---
# Module: [`qwix/_src/providers/qt.py`](../../../../../../../raw/code/qwix/qwix/_src/providers/qt.py)

## Classes
### `QtProvider`  ·  implements/extends QuantizationProvider
- def: [`qwix/_src/providers/qt.py:66`](../../../../../../../raw/code/qwix/qwix/_src/providers/qt.py#L66)
- doc: Quantization provider for Quantized Training (QT).
- signature: `class QtProvider(qconfig.QuantizationProvider):`
- members:
  - `_create_conv_general_qt_config(self, rule: qconfig.QuantizationRule, op_id: str, lhs: jax.Array, rhs: jax.Array)` — [`L263`](../../../../../../../raw/code/qwix/qwix/_src/providers/qt.py#L263) — Creates a ConvGeneralQtConfig for conv_general_dilated. — documented in [qwix-_src-providers-qt](../../../../concepts/qwix-_src-providers-qt.md)
  - `_create_dot_general_qt_config(self, rule: qconfig.QuantizationRule, op_id: str, lhs: jax.Array, rhs: jax.Array)` — [`L301`](../../../../../../../raw/code/qwix/qwix/_src/providers/qt.py#L301) — Creates a DotGeneralQtConfig for dot_general and einsum. — documented in [qwix-_src-providers-qt](../../../../concepts/qwix-_src-providers-qt.md)
  - `_create_ragged_dot_qt_config(self, rule: qconfig.QuantizationRule)` — [`L398`](../../../../../../../raw/code/qwix/qwix/_src/providers/qt.py#L398) — Creates a RaggedDotQtConfig for ragged_dot. — documented in [qwix-_src-providers-qt](../../../../concepts/qwix-_src-providers-qt.md)
  - `_update_and_get_quant_stat(self, name: str, batch_axes: tuple[int, ...], calibration: averaging.Calibration)` — [`L241`](../../../../../../../raw/code/qwix/qwix/_src/providers/qt.py#L241) — Updates the running quantization statistics and returns the average. — documented in [qwix-_src-averaging](../../../../concepts/qwix-_src-averaging.md)
  - `conv_general_dilated(self, lhs: jax.Array, rhs: jax.Array, window_strides: Sequence[int], padding: str | Sequence[tuple[int, int]], lhs_dilation: Sequence[int] | None = None, rhs_dilation: Sequence[int] | None = None, dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None, feature_group_count: int = 1, batch_group_count: int = 1, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, out_sharding=None)` — [`L152`](../../../../../../../raw/code/qwix/qwix/_src/providers/qt.py#L152) — QT conv_general_dilated. — documented in [qwix-_src-core-conv_general_qt](../../../../concepts/qwix-_src-core-conv_general_qt.md)
  - `custom_dot_general(lhs, rhs, dimension_numbers, precision, preferred_element_type, **kwargs)` — [`L123`](../../../../../../../raw/code/qwix/qwix/_src/providers/qt.py#L123) — documented in [qwix-_src-providers-qt](../../../../concepts/qwix-_src-providers-qt.md)
  - `dot_general(self, lhs: jax.Array, rhs: jax.Array, dimension_numbers: jax.lax.DotDimensionNumbers, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, *, out_sharding=None)` — [`L75`](../../../../../../../raw/code/qwix/qwix/_src/providers/qt.py#L75) — QT dot_general. — documented in [qwix-_src-core-dot_general_qt](../../../../concepts/qwix-_src-core-dot_general_qt.md)
  - `einsum(self, einsum_str: str, *operands: jax.Array, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, _dot_general: Callable[..., jax.Array] = jax.lax.dot_general, out_sharding=None)` — [`L99`](../../../../../../../raw/code/qwix/qwix/_src/providers/qt.py#L99) — QT einsum.
  - `get_intercept_map(self)` — [`L232`](../../../../../../../raw/code/qwix/qwix/_src/providers/qt.py#L232) — Used for interception.
  - `ragged_dot(self, lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, group_offset: jax.Array | None = None)` — [`L201`](../../../../../../../raw/code/qwix/qwix/_src/providers/qt.py#L201) — QT ragged_dot. — documented in [qwix-_src-providers-qt](../../../../concepts/qwix-_src-providers-qt.md)
- protocol/private: `_init_rule`[`L69`](../../../../../../../raw/code/qwix/qwix/_src/providers/qt.py#L69)
- uses (calls/refs, reference-scoped): [`weight_qtype`](../qconfig.md#QuantizationRule.weight_qtype), [`QuantizationRule`](../qconfig.md#QuantizationRule), [`act_qtype`](../qconfig.md#QuantizationRule.act_qtype), [`dot_general_qt`](../core/dot_general_qt.md#dot_general_qt), [`_get_current_rule_and_op_id`](../qconfig.md#QuantizationProvider._get_current_rule_and_op_id), [`tile_size`](../qconfig.md#QuantizationRule.tile_size), [`act_static_scale`](../qconfig.md#QuantizationRule.act_static_scale), [`QuantizationProvider`](../qconfig.md#QuantizationProvider), [`QtRule`](qt.md#QtRule), [`find_param`](../utils/flax_util.md#find_param), [`get_intercept_map`](../qconfig.md#QuantizationProvider.get_intercept_map), [`get_calibration`](../averaging.md#SimpleMovingAverage.get_calibration), [`weight_calibration_method`](../qconfig.md#QuantizationRule.weight_calibration_method), [`bwd_qtype`](qt.md#QtRule.bwd_qtype), [`_init_rule`](../qconfig.md#QuantizationProvider._init_rule), [`act_calibration_method`](../qconfig.md#QuantizationRule.act_calibration_method), [`init`](../averaging.md#SimpleMovingAverage.init), [`update`](../averaging.md#SimpleMovingAverage.update), [`DotGeneralQtConfig`](../core/dot_general_qt.md#DotGeneralQtConfig), [`SimpleMovingAverage`](../averaging.md#SimpleMovingAverage), [`lhs_qtype`](../core/dot_general_qt.md#DotGeneralQtConfig.lhs_qtype), [`rhs_qtype`](../core/dot_general_qt.md#DotGeneralQtConfig.rhs_qtype), [`get_noise_fn`](../core/stochastic_rounding.md#get_noise_fn), [`get_or_create_variable`](../utils/flax_util.md#get_or_create_variable), [`tile_size`](../core/dot_general_qt.md#DotGeneralQtConfig.tile_size), [`disable_channelwise_axes`](qt.md#QtRule.disable_channelwise_axes), [`ragged_dot_qt`](../core/ragged_dot_qt.md#ragged_dot_qt), [`make_rng`](../utils/flax_util.md#make_rng), [`RaggedDotQtConfig`](../core/ragged_dot_qt.md#RaggedDotQtConfig), [`Calibration`](../averaging.md#Calibration.Calibration), [`conv_general_qt`](../core/conv_general_qt.md#conv_general_qt), [`dlhs_grad_qtype`](../core/dot_general_qt.md#DotGeneralQtConfig.dlhs_grad_qtype), [`drhs_grad_qtype`](../core/dot_general_qt.md#DotGeneralQtConfig.drhs_grad_qtype), [`ConvGeneralQtConfig`](../core/conv_general_qt.md#ConvGeneralQtConfig), [`drhs_grad_qtype`](../core/ragged_dot_qt.md#RaggedDotQtConfig.drhs_grad_qtype), [`drhs_tile_size`](../core/dot_general_qt.md#DotGeneralQtConfig.drhs_tile_size), [`dlhs_stochastic_rounding_noise_fn`](../core/dot_general_qt.md#DotGeneralQtConfig.dlhs_stochastic_rounding_noise_fn), [`drhs_stochastic_rounding_noise_fn`](../core/dot_general_qt.md#DotGeneralQtConfig.drhs_stochastic_rounding_noise_fn), [`should_update_quant_stats`](../utils/flax_util.md#should_update_quant_stats), [`act_batch_axes`](../qconfig.md#QuantizationRule.act_batch_axes)  (+34 more)
- used by: [`test_srq`](../../../integration_tests/cnn_test.md#CnnTest.test_srq), [`test_srq`](../../../integration_tests/vae_qat_test.md#VaeQatTest.test_srq), [`QuantizationProvider`](../qconfig.md#QuantizationProvider), [`get_intercept_map`](../qconfig.md#QuantizationProvider.get_intercept_map), [`test_coverage`](../../../integration_tests/coverage_test.md#CoverageTest.test_coverage), [`test_drq`](../../../integration_tests/cnn_test.md#CnnTest.test_drq), [`test_drq`](../../../integration_tests/vae_qat_test.md#VaeQatTest.test_drq), [`_init_rule`](../qconfig.md#QuantizationProvider._init_rule)  (17 test-only)

### `QtRule`  ·  implements/extends QuantizationRule
- def: [`qwix/_src/providers/qt.py:32`](../../../../../../../raw/code/qwix/qwix/_src/providers/qt.py#L32) — documented in [qwix-_src-providers-qt](../../../../concepts/qwix-_src-providers-qt.md)
- doc: QuantizationRule with all settings specific to Quantized Training (QT).
- signature: `class QtRule(qconfig.QuantizationRule):`
- members:
  - `additional_qt_config` — [`L63`](../../../../../../../raw/code/qwix/qwix/_src/providers/qt.py#L63)
  - `bwd_calibration_method` — [`L41`](../../../../../../../raw/code/qwix/qwix/_src/providers/qt.py#L41)
  - `bwd_qtype` — [`L38`](../../../../../../../raw/code/qwix/qwix/_src/providers/qt.py#L38) — documented in [qwix-_src-providers-qt](../../../../concepts/qwix-_src-providers-qt.md)
  - `bwd_stochastic_rounding` — [`L54`](../../../../../../../raw/code/qwix/qwix/_src/providers/qt.py#L54)
  - `bwd_weight_grad_tile_size` — [`L47`](../../../../../../../raw/code/qwix/qwix/_src/providers/qt.py#L47)
  - `channelwise_noise_axes` — [`L58`](../../../../../../../raw/code/qwix/qwix/_src/providers/qt.py#L58)
  - `disable_channelwise_axes` — [`L50`](../../../../../../../raw/code/qwix/qwix/_src/providers/qt.py#L50) — documented in [qwix-_src-providers-qt](../../../../concepts/qwix-_src-providers-qt.md)
- uses (calls/refs, reference-scoped): [`QuantizationRule`](../qconfig.md#QuantizationRule)
- used by: [`QuantizationRule`](../qconfig.md#QuantizationRule), [`_create_dot_general_qt_config`](qt.md#QtProvider._create_dot_general_qt_config), [`_create_conv_general_qt_config`](qt.md#QtProvider._create_conv_general_qt_config), [`test_srq`](../../../integration_tests/cnn_test.md#CnnTest.test_srq), [`_create_ragged_dot_qt_config`](qt.md#QtProvider._create_ragged_dot_qt_config), [`restore_quantization_rules`](../utils/checkpoint_util.md#restore_quantization_rules), [`_init_rule`](qt.md#QtProvider._init_rule)  (10 test-only)

