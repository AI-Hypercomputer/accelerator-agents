---
title: 'Module: qwix/_src/providers/ptq.py'
type: catalog
provenance: extracted
module: qwix/_src/providers/ptq.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.providers.ptq`/
symbols:
  PtqProvider: PtqProvider#
  WithAux: WithAux#
  quantize_params: quantize_params().
  PtqProvider.dot_general: PtqProvider#dot_general().
  PtqProvider.einsum: PtqProvider#einsum().
  PtqProvider.conv_general_dilated: PtqProvider#conv_general_dilated().
  quantize_act: quantize_act().
  PtqProvider.get_intercept_map: PtqProvider#get_intercept_map().
  PtqProvider.asarray: PtqProvider#asarray().
  create_quantized_param: create_quantized_param().
  quantize_act.init: quantize_act().init().
  WithAux.array: WithAux#array.
  PtqProvider.__init__: PtqProvider#__init__().
  PtqProvider.dot: PtqProvider#dot().
  WithAux.how: WithAux#how.
  WithAux.astype: WithAux#astype().
  PtqProvider._qarray_module: PtqProvider#_qarray_module.
  WithAux.shape: WithAux#shape.
  ArrayTypeVar: ArrayTypeVar.
  WithAux.ndim: WithAux#ndim.
  WithAux.dtype: WithAux#dtype.
  PtqProvider.nn_param: PtqProvider#nn_param().
  WithAux.value: WithAux#value.
  WithAux.reshape: WithAux#reshape().
  PtqProvider.promote_dtype: PtqProvider#promote_dtype().
  PtqProvider._dot_general_fn: PtqProvider#_dot_general_fn.
  PtqProvider._einsum_fn: PtqProvider#_einsum_fn.
  PtqProvider._conv_general_dilated_fn: PtqProvider#_conv_general_dilated_fn.
  WithAux.__getitem__: WithAux#__getitem__.
---
# Module: [`qwix/_src/providers/ptq.py`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py)

## Classes
### `PtqProvider`
- def: [`qwix/_src/providers/ptq.py:83`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L83) — documented in [qwix-_src-providers-ptq](../../../../concepts/qwix-_src-providers-ptq.md)
- members:
  - `__init__(self, rules: Sequence[qconfig.QuantizationRule], *, disable_jit: bool = False, _qarray_module=qarray, _dot_general_fn=dot_general.dot_general, _einsum_fn=einsum.einsum, _conv_general_dilated_fn=conv_general.conv_general_dilated)` — [`L101`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L101) — Initializes the PTQ provider.
  - `asarray(self, a, dtype=None, order=None, **kwargs)` — [`L360`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L360) — Intercepts jax.numpy.asarray to correctly handle WithAux and QArray.
  - `conv_general_dilated(self, lhs: jax.Array, rhs: jax.Array | WithAux[qarray.QArray], window_strides: Sequence[int], padding: str | Sequence[tuple[int, int]], lhs_dilation: Sequence[int] | None = None, rhs_dilation: Sequence[int] | None = None, dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None, feature_group_count: int = 1, batch_group_count: int = 1, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, out_sharding=None)` — [`L247`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L247) — documented in [qwix-_src-providers-ptq](../../../../concepts/qwix-_src-providers-ptq.md)
  - `dot(self, a: jax.Array, b: jax.Array | WithAux[qarray.QArray], precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, out_sharding=None)` — [`L342`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L342) — Intercepts jax.numpy.dot.
  - `dot_general(self, lhs: jax.Array, rhs: jax.Array | WithAux[qarray.QArray], dimension_numbers: jax.lax.DotDimensionNumbers, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, *, out_sharding: jax.sharding.NamedSharding | None = None)` — [`L118`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L118) — documented in [qwix-_src-core-dot_general](../../../../concepts/qwix-_src-core-dot_general.md)
  - `einsum(self, einsum_str: str, *operands: jax.Array, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, _dot_general: Callable[..., jax.Array] = jax.lax.dot_general, out_sharding=None)` — [`L182`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L182) — documented in [qwix-_src-providers-ptq](../../../../concepts/qwix-_src-providers-ptq.md)
  - `get_intercept_map(self)` — [`L413`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L413) — Used for interception.
  - `nn_param(self, module: nn.Module, name: str, *args, **kwargs)` — [`L325`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L325) — Intercepts nn.Module.param to handle quantized params.
  - `promote_dtype(self, *args, **kwargs)` — [`L333`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L333) — Intercepts promote_dtype to handle quantized params.
  - `_conv_general_dilated_fn` — [`L116`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L116) — ---
  - `_dot_general_fn` — [`L114`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L114) — ---
  - `_einsum_fn` — [`L115`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L115) — ---
  - `_qarray_module` — [`L113`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L113) — ---
- uses (calls/refs, reference-scoped): [`weight_qtype`](../qconfig.md#QuantizationRule.weight_qtype), [`QuantizationRule`](../qconfig.md#QuantizationRule), [`QArray`](../core/qarray.md#QArray), [`act_qtype`](../qconfig.md#QuantizationRule.act_qtype), [`WithAux`](ptq.md#WithAux), [`dot_general`](../core/dot_general.md#dot_general), [`_get_current_rule_and_op_id`](../qconfig.md#QuantizationProvider._get_current_rule_and_op_id), [`tile_size`](../qconfig.md#QuantizationRule.tile_size), [`quantize_act`](ptq.md#quantize_act), [`dot_general`](lora.md#LoraProvider.dot_general), [`einsum`](lora.md#LoraProvider.einsum), [`einsum`](../core/einsum.md#einsum), [`get_how_to_quantize`](../core/einsum.md#get_how_to_quantize), [`conv_general_dilated`](lora.md#LoraProvider.conv_general_dilated), [`QuantizationProvider`](../qconfig.md#QuantizationProvider), [`find_param`](../utils/flax_util.md#find_param), [`get_how_to_quantize`](../core/dot_general.md#get_how_to_quantize), [`get_intercept_map`](../qconfig.md#QuantizationProvider.get_intercept_map), [`unbox`](../utils/flax_util.md#unbox), [`weight_calibration_method`](../qconfig.md#QuantizationRule.weight_calibration_method), [`create_quantized_param`](ptq.md#create_quantized_param), [`dot_general`](../../contrib/smooth_quant.md#SqInferenceProvider.dot_general), [`act_calibration_method`](../qconfig.md#QuantizationRule.act_calibration_method), [`get_how_to_quantize`](../core/conv_general.md#get_how_to_quantize), [`conv_general_dilated`](../core/conv_general.md#conv_general_dilated), [`array`](ptq.md#WithAux.array), [`LoraProvider`](lora.md#LoraProvider), [`dot_general`](../../contrib/awq.md#AwqInferenceProvider.dot_general), [`einsum`](../../contrib/awq.md#AwqInferenceProvider.einsum), [`dot`](../core/dot.md#dot), [`AwqInferenceProvider`](../../contrib/awq.md#AwqInferenceProvider), [`get_intercept_map`](../../contrib/awq.md#AwqInferenceProvider.get_intercept_map), [`astype`](../core/qarray.md#QArray.astype), [`SqInferenceProvider`](../../contrib/smooth_quant.md#SqInferenceProvider), [`__init__`](../qconfig.md#QuantizationProvider.__init__), [`get_intercept_map`](../../contrib/smooth_quant.md#SqInferenceProvider.get_intercept_map), [`dtype`](../core/qarray.md#QArray.dtype)
- used by: [`test_srq`](../../../integration_tests/cnn_test.md#CnnTest.test_srq), [`dot_general`](lora.md#LoraProvider.dot_general), [`einsum`](lora.md#LoraProvider.einsum), [`test_load_full_precision_weights`](../../../integration_tests/orbax_safetensors_test.md#OrbaxSafetensorsIntegrationTest.test_load_full_precision_weights), [`test_load_prequantized_checkpoints`](../../../integration_tests/orbax_safetensors_test.md#OrbaxSafetensorsIntegrationTest.test_load_prequantized_checkpoints), [`test_load_with_renaming`](../../../integration_tests/orbax_safetensors_test.md#OrbaxSafetensorsIntegrationTest.test_load_with_renaming), [`test_srq`](../../../integration_tests/vae_qat_test.md#VaeQatTest.test_srq), [`conv_general_dilated`](lora.md#LoraProvider.conv_general_dilated), [`get_intercept_map`](../qconfig.md#QuantizationProvider.get_intercept_map), [`test_coverage`](../../../integration_tests/coverage_test.md#CoverageTest.test_coverage), [`test_drq`](../../../integration_tests/cnn_test.md#CnnTest.test_drq), [`test_drq`](../../../integration_tests/vae_qat_test.md#VaeQatTest.test_drq), [`LoraProvider`](lora.md#LoraProvider), [`dot_general`](../../contrib/awq.md#AwqInferenceProvider.dot_general), [`einsum`](../../contrib/awq.md#AwqInferenceProvider.einsum), [`create_quantized_model`](../../../integration_tests/orbax_safetensors_test.md#OrbaxSafetensorsIntegrationTest.create_quantized_model), [`_create_ptq_model_and_abstract_quantized_params`](../../contrib/qep.md#_create_ptq_model_and_abstract_quantized_params), [`AwqInferenceProvider`](../../contrib/awq.md#AwqInferenceProvider), [`get_intercept_map`](../../contrib/awq.md#AwqInferenceProvider.get_intercept_map), [`SqInferenceProvider`](../../contrib/smooth_quant.md#SqInferenceProvider), [`__init__`](lora.md#LoraProvider.__init__), [`get_intercept_map`](../../contrib/smooth_quant.md#SqInferenceProvider.get_intercept_map), [`PtqProvider`](../../contrib/padded_ptq.md#PtqProvider)  (42 test-only)

### `WithAux`  ·  implements/extends Generic
- def: [`qwix/_src/providers/ptq.py:37`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L37) — documented in [qwix-_src-providers-ptq](../../../../concepts/qwix-_src-providers-ptq.md)
- doc: An array/QArray with auxiliary information.
- signature: `class WithAux(Generic[ArrayTypeVar]):`
- members:
  - `astype(self, dtype)` — [`L61`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L61)
  - `reshape(self, *shape)` — [`L65`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L65)
  - `array` — [`L51`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L51)
  - `dtype` — [`L59`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L59)
  - `how` — [`L52`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L52)
  - `ndim` — [`L57`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L57)
  - `shape` — [`L56`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L56)
  - `value` — [`L55`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L55) — documented in [qwix-_src-utils-flax_util](../../../../concepts/qwix-_src-utils-flax_util.md)
- protocol/private: `__getitem__`[`L58`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L58)
- uses (calls/refs, reference-scoped): [`HowToQuantize`](../core/qarray.md#HowToQuantize), [`unbox`](../utils/flax_util.md#unbox), [`update_boxed`](../utils/flax_util.md#update_boxed), [`WithAwqScale`](../../contrib/awq.md#WithAwqScale), [`WithSqScale`](../../contrib/smooth_quant.md#WithSqScale), [`ArrayTypeVar`](ptq.md#ArrayTypeVar)
- used by: [`quantize_params`](ptq.md#quantize_params), [`dot_general`](ptq.md#PtqProvider.dot_general), [`einsum`](ptq.md#PtqProvider.einsum), [`conv_general_dilated`](ptq.md#PtqProvider.conv_general_dilated), [`extract_calibrated_quant_context`](../../contrib/calibration.md#extract_calibrated_quant_context), [`_quantize_weight`](../../contrib/qep.md#_quantize_weight), [`dot_general`](lora.md#LoraProvider.dot_general), [`einsum`](lora.md#LoraProvider.einsum), [`quantize_params`](../../contrib/smooth_quant.md#quantize_params), [`_get_or_create_lora_params`](lora.md#_get_or_create_lora_params), [`conv_general_dilated`](lora.md#LoraProvider.conv_general_dilated), [`_quantize`](../../contrib/awq.md#quantize_params._quantize), [`asarray`](ptq.md#PtqProvider.asarray), [`quantize_params_with_calibration`](../../contrib/calibration.md#quantize_params_with_calibration), [`create_quantized_param`](ptq.md#create_quantized_param), [`dot_general`](../../contrib/smooth_quant.md#SqInferenceProvider.dot_general), [`init`](ptq.md#quantize_act.init), [`_apply_awq_scale`](../../contrib/awq.md#AwqInferenceProvider._apply_awq_scale), [`dot_general`](../../contrib/awq.md#AwqInferenceProvider.dot_general), [`WithAwqScale`](../../contrib/awq.md#WithAwqScale), [`WithSqScale`](../../contrib/smooth_quant.md#WithSqScale), [`dot`](ptq.md#PtqProvider.dot), [`einsum`](../../contrib/awq.md#AwqInferenceProvider.einsum), [`create_quantized_param`](../../contrib/padded_ptq.md#create_quantized_param), [`abs_w`](../../contrib/calibration.md#CalibratedQuantContext.abs_w), [`nn_param`](ptq.md#PtqProvider.nn_param)  (14 test-only)

## Functions
- `create_quantized_param(name: str, value: jax.Array, how: qarray.HowToQuantize, *, _qarray_module=qarray)` — [`L469`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L469) — Creates the quantized param and replaces the original param in the module.
- `init()` — [`L445`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L445) — documented in [qwix-_src-averaging](../../../../concepts/qwix-_src-averaging.md)
- `quantize_act(array: jax.Array, how: qarray.HowToQuantize, rule: qconfig.QuantizationRule, act_name: str, *, _qarray_module=qarray)` — [`L427`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L427) — Quantizes the input activation with support for static scale. — documented in [qwix-_src-core-qarray](../../../../concepts/qwix-_src-core-qarray.md)
- `quantize_params(params: Any, abstract_quantized_params: Any, quant_stats: Any = flax.core.FrozenDict(), *, allow_extra_params: bool = False, _qarray_module=qarray)` — [`L513`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L513) — Quantize the param tree for PTQ. — documented in [qwix-_src-providers-ptq](../../../../concepts/qwix-_src-providers-ptq.md)

## Module values
- `ArrayTypeVar` — [`L33`](../../../../../../../raw/code/qwix/qwix/_src/providers/ptq.py#L33)

