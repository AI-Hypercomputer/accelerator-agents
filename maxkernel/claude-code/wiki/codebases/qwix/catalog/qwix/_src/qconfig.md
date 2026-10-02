---
title: 'Module: qwix/_src/qconfig.py'
type: catalog
provenance: extracted
module: qwix/_src/qconfig.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.qconfig`/
symbols:
  QuantizationRule.weight_qtype: QuantizationRule#weight_qtype.
  QuantizationRule: QuantizationRule#
  QuantizationRule.act_qtype: QuantizationRule#act_qtype.
  QuantizationRule.module_path: QuantizationRule#module_path.
  QuantizationProvider._get_current_rule_and_op_id: QuantizationProvider#_get_current_rule_and_op_id().
  QuantizationRule.tile_size: QuantizationRule#tile_size.
  QuantizationRule.act_static_scale: QuantizationRule#act_static_scale.
  QuantizationProvider: QuantizationProvider#
  QuantizationProvider.get_intercept_map: QuantizationProvider#get_intercept_map().
  QuantizationRule.weight_calibration_method: QuantizationRule#weight_calibration_method.
  QuantizationRule.act_calibration_method: QuantizationRule#act_calibration_method.
  QuantizationProvider._init_rule: QuantizationProvider#_init_rule().
  QuantizationProvider.get_unused_rules: QuantizationProvider#get_unused_rules().
  QuantizationProvider.get_interceptors: QuantizationProvider#get_interceptors().
  QuantizationProvider.process_model_output: QuantizationProvider#process_model_output().
  QuantizationProvider.__init__: QuantizationProvider#__init__().
  QuantizationProvider._rules: QuantizationProvider#_rules.
  QuantizationProvider.process_model_inputs: QuantizationProvider#process_model_inputs().
  QuantizationRule.act_batch_axes: QuantizationRule#act_batch_axes.
  get_current_rule: get_current_rule().
  QuantizationRule.op_names: QuantizationRule#op_names.
  QuantizationProvider._rule_matches: QuantizationProvider#_rule_matches.
  QuantizationProvider._logged_ops: QuantizationProvider#_logged_ops.
  QuantizationProvider._initial_run_complete: QuantizationProvider#_initial_run_complete.
  QuantizationProvider.disable_jit: QuantizationProvider#disable_jit.
---
# Module: [`qwix/_src/qconfig.py`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py)

## Classes
### `QuantizationProvider`
- def: [`qwix/_src/qconfig.py:101`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L101) — documented in [qwix-contrib-gptq](../../../concepts/qwix-contrib-gptq.md)
- doc: Interface for model integration.
- signature: `class QuantizationProvider:`
- members:
  - `__init__(self, rules: Sequence[QuantizationRule], *, disable_jit: bool = False)` — [`L108`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L108) — Initialize the provider.
  - `_get_current_rule_and_op_id(self, op_name: str, *, only_rule: bool = False, repeated_call: bool = False)` — [`L189`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L189) — Returns the quantization rule and a unique op id for given op. — documented in [qwix-_src-providers-lora](../../../concepts/qwix-_src-providers-lora.md)
  - `_init_rule(self, rule: QuantizationRule)` — [`L123`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L123) — Validate and set default values for the rule.
  - `get_intercept_map(self)` — [`L137`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L137) — Returns the intercept map for interception.wrap_func_intercepted.
  - `get_interceptors(self)` — [`L158`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L158) — Returns a list of interceptor factories. — documented in [qwix-_src-interception](../../../concepts/qwix-_src-interception.md)
  - `get_unused_rules(self)` — [`L242`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L242) — Returns the quantization rules that did not match any operations.
  - `process_model_inputs(self, model: Any, model_args: Sequence[Any], model_kwargs: dict[str, Any])` — [`L173`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L173) — Process the model and its inputs before it is called.
  - `process_model_output(self, method_name: str, model_output: Any)` — [`L183`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L183) — Process the model output before it is returned.
  - `disable_jit` — [`L121`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L121)
- protocol/private: `_initial_run_complete`[`L120`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L120), `_logged_ops`[`L119`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L119), `_rule_matches`[`L117`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L117), `_rules`[`L118`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L118)
- uses (calls/refs, reference-scoped): [`QuantizationRule`](qconfig.md#QuantizationRule), [`act_qtype`](qconfig.md#QuantizationRule.act_qtype), [`module_path`](qconfig.md#QuantizationRule.module_path), [`get`](aux_data.md#get), [`Interceptor`](interception.md#Interceptor), [`mapping`](interception.md#Interceptor.mapping), [`act_static_scale`](qconfig.md#QuantizationRule.act_static_scale), [`id`](interception.md#Interceptor.id), [`OdmlQatProvider`](providers/odml.md#OdmlQatProvider), [`get_intercept_map`](providers/ptq.md#PtqProvider.get_intercept_map), [`QtProvider`](providers/qt.md#QtProvider), [`get_intercept_map`](providers/odml.md#OdmlQatProvider.get_intercept_map), [`act_calibration_method`](qconfig.md#QuantizationRule.act_calibration_method), [`process_model_inputs`](providers/odml.md#OdmlQatProvider.process_model_inputs), [`get_current_module`](utils/flax_util.md#get_current_module), [`get_interceptors`](providers/odml.md#OdmlQatProvider.get_interceptors), [`_init_rule`](providers/odml.md#OdmlQatProvider._init_rule), [`get_intercept_map`](providers/qt.md#QtProvider.get_intercept_map), `scope`, [`CalibrationProvider`](../contrib/calibration.md#CalibrationProvider), [`disable_interceptions`](interception.md#disable_interceptions), [`_init_rule`](providers/qt.md#QtProvider._init_rule), [`clear`](aux_data.md#clear), [`get_intercept_map`](../contrib/calibration.md#CalibrationProvider.get_intercept_map), [`get_current_module_path`](utils/flax_util.md#get_current_module_path), [`has_attribute`](interception.md#has_attribute), [`op_names`](qconfig.md#QuantizationRule.op_names), [`process_model_output`](providers/odml.md#OdmlQatProvider.process_model_output)  (3 test-only)
- used by: [`quantize_model`](model.md#quantize_model), [`PtqProvider`](providers/ptq.md#PtqProvider), [`dot_general`](providers/ptq.md#PtqProvider.dot_general), [`einsum`](providers/ptq.md#PtqProvider.einsum), [`compute_stats`](../contrib/smooth_quant.md#SqCalibrationProvider.compute_stats), [`conv_general_dilated`](providers/ptq.md#PtqProvider.conv_general_dilated), [`dot_general`](providers/lora.md#LoraProvider.dot_general), [`einsum`](providers/lora.md#LoraProvider.einsum), [`OdmlQatProvider`](providers/odml.md#OdmlQatProvider), [`conv_general_dilated`](providers/lora.md#LoraProvider.conv_general_dilated), [`get_intercept_map`](providers/ptq.md#PtqProvider.get_intercept_map), [`_collect_stats`](../contrib/qep.md#_CaptureProvider._collect_stats), [`QtProvider`](providers/qt.md#QtProvider), [`apply_lora_to_model`](providers/lora.md#apply_lora_to_model), [`get_intercept_map`](providers/odml.md#OdmlQatProvider.get_intercept_map), [`process_model_inputs`](providers/odml.md#OdmlQatProvider.process_model_inputs), [`dot_general`](../contrib/calibration.md#CalibrationProvider.dot_general), [`_apply_interceptors`](model.md#_apply_interceptors), [`__init__`](providers/ptq.md#PtqProvider.__init__), [`conv_general_dilated`](providers/qt.md#QtProvider.conv_general_dilated), [`_init_rule`](providers/odml.md#OdmlQatProvider._init_rule), [`_initial_run_complete`](providers/odml.md#OdmlQatProvider._initial_run_complete), [`get_intercept_map`](providers/qt.md#QtProvider.get_intercept_map), [`dot_general`](providers/qt.md#QtProvider.dot_general), [`quantize_linen_model`](model.md#quantize_linen_model), [`ragged_dot`](providers/qt.md#QtProvider.ragged_dot), [`CalibrationProvider`](../contrib/calibration.md#CalibrationProvider), [`einsum`](providers/qt.md#QtProvider.einsum), [`einsum`](../contrib/calibration.md#CalibrationProvider.einsum), [`quantize_nnx_model`](model.md#quantize_nnx_model), [`_init_rule`](providers/qt.md#QtProvider._init_rule), [`get_intercept_map`](../contrib/calibration.md#CalibrationProvider.get_intercept_map), [`__init__`](providers/odml.md#OdmlQatProvider.__init__), [`_input_transform`](model.md#_input_transform), [`_output_transform_nnx`](model.md#_output_transform_nnx), [`__init__`](../contrib/qep.md#_CaptureProvider.__init__)  (7 test-only)

### `QuantizationRule`
- def: [`qwix/_src/qconfig.py:32`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L32) — documented in [qwix-_src-providers-ptq](../../../concepts/qwix-_src-providers-ptq.md)
- members:
  - `act_batch_axes` — [`L92`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L92)
  - `act_calibration_method` — [`L85`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L85)
  - `act_qtype` — [`L54`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L54) — documented in [qwix-_src-providers-ptq](../../../concepts/qwix-_src-providers-ptq.md)
  - `act_static_scale` — [`L68`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L68) — documented in [qwix-_src-providers-qt](../../../concepts/qwix-_src-providers-qt.md)
  - `module_path` — [`L40`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L40) — documented in [qwix-_src-model](../../../concepts/qwix-_src-model.md)
  - `op_names` — [`L44`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L44)
  - `tile_size` — [`L60`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L60) — documented in [qwix-_src-providers-ptq](../../../concepts/qwix-_src-providers-ptq.md)
  - `weight_calibration_method` — [`L80`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L80)
  - `weight_qtype` — [`L51`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L51) — documented in [qwix-_src-providers-qt](../../../concepts/qwix-_src-providers-qt.md)
- uses (calls/refs, reference-scoped): [`QtRule`](providers/qt.md#QtRule), [`GptqRule`](../contrib/gptq.md#GptqRule), [`LoraRule`](providers/lora.md#LoraRule), [`AwqRule`](../contrib/awq.md#AwqRule), [`SqRule`](../contrib/smooth_quant.md#SqRule)
- used by: [`_create_dot_general_qt_config`](providers/qt.md#QtProvider._create_dot_general_qt_config), [`_create_conv_general_qt_config`](providers/qt.md#QtProvider._create_conv_general_qt_config), [`_maybe_fake_quant`](providers/odml_ops.md#QuantizedOp._maybe_fake_quant), [`__call__`](providers/odml_ops.md#DotEinsumConv.__call__), [`dot_general`](providers/ptq.md#PtqProvider.dot_general), [`_get_current_rule_and_op_id`](qconfig.md#QuantizationProvider._get_current_rule_and_op_id), [`einsum`](providers/ptq.md#PtqProvider.einsum), [`compute_stats`](../contrib/smooth_quant.md#SqCalibrationProvider.compute_stats), [`conv_general_dilated`](providers/ptq.md#PtqProvider.conv_general_dilated), [`test_srq`](../../integration_tests/cnn_test.md#CnnTest.test_srq), [`_fake_quant_output`](providers/odml_ops.md#QuantizedOp._fake_quant_output), [`quantize_act`](providers/ptq.md#quantize_act), [`test_cnn_srq`](../../integration_tests/odml_cnn_test.md#OdmlCnnTest.test_cnn_srq), [`test_load_full_precision_weights`](../../integration_tests/orbax_safetensors_test.md#OrbaxSafetensorsIntegrationTest.test_load_full_precision_weights), [`test_load_prequantized_checkpoints`](../../integration_tests/orbax_safetensors_test.md#OrbaxSafetensorsIntegrationTest.test_load_prequantized_checkpoints), [`test_load_with_renaming`](../../integration_tests/orbax_safetensors_test.md#OrbaxSafetensorsIntegrationTest.test_load_with_renaming), [`test_srq`](../../integration_tests/odml_coverage_test.md#OdmlTest.test_srq), [`test_srq`](../../integration_tests/vae_qat_test.md#VaeQatTest.test_srq), [`test_drq`](../../integration_tests/odml_coverage_test.md#OdmlTest.test_drq), [`test_nnx`](../../integration_tests/odml_coverage_test.md#OdmlTest.test_nnx), [`__call__`](providers/odml_ops.md#OnlyOutputOp.__call__), [`_create_ragged_dot_qt_config`](providers/qt.md#QtProvider._create_ragged_dot_qt_config), [`QtRule`](providers/qt.md#QtRule), [`__call__`](providers/odml_ops.md#Concatenate.__call__), [`test_coverage`](../../integration_tests/coverage_test.md#CoverageTest.test_coverage), [`test_drq`](../../integration_tests/cnn_test.md#CnnTest.test_drq), [`test_drq`](../../integration_tests/vae_qat_test.md#VaeQatTest.test_drq), [`__call__`](providers/odml_ops.md#OnlyInputOp.__call__), [`__call__`](providers/odml_ops.md#Take.__call__), [`test_load_prequantized_2d_blocksize_checkpoints`](../../integration_tests/orbax_safetensors_test.md#OrbaxSafetensorsIntegrationTest.test_load_prequantized_2d_blocksize_checkpoints), [`test_partial_quantization_unet_succeeds`](../../integration_tests/odml_coverage_test.md#OdmlTest.test_partial_quantization_unet_succeeds), [`test_weight_only`](../../integration_tests/odml_coverage_test.md#OdmlTest.test_weight_only), [`_fake_quant_inputs`](providers/odml_ops.md#UfuncCall._fake_quant_inputs), [`test_vae_separate_export`](../../integration_tests/odml_coverage_test.md#OdmlTest.test_vae_separate_export), [`restore_quantization_rules`](utils/checkpoint_util.md#restore_quantization_rules), [`init`](providers/ptq.md#quantize_act.init), [`GptqRule`](../contrib/gptq.md#GptqRule), [`_init_rule`](qconfig.md#QuantizationProvider._init_rule), [`test_model_output_pytree`](../../integration_tests/odml_coverage_test.md#OdmlTest.test_model_output_pytree), [`test_partial_quantization`](../../integration_tests/odml_coverage_test.md#OdmlTest.test_partial_quantization)  (+26 more; 100 test-only)

## Functions
- `get_current_rule(op_name: str)` — [`L95`](../../../../../../raw/code/qwix/qwix/_src/qconfig.py#L95) — Returns the current quantization rule if intercepted, or None otherwise.

