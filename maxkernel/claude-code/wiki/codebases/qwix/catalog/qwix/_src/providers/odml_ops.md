---
title: 'Module: qwix/_src/providers/odml_ops.py'
type: catalog
provenance: extracted
module: qwix/_src/providers/odml_ops.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.providers.odml_ops`/
symbols:
  AuxDataKey: AuxDataKey#
  QuantizedOp._maybe_fake_quant: QuantizedOp#_maybe_fake_quant().
  DotEinsumConv.__call__: DotEinsumConv#__call__().
  get_all_ops: get_all_ops().
  QuantizedOp._fake_quant_output: QuantizedOp#_fake_quant_output().
  FinalOutput.__call__: FinalOutput#__call__().
  Silu.__call__: Silu#__call__().
  _forward_metadata: _forward_metadata().
  OnlyOutputOp.__call__: OnlyOutputOp#__call__().
  Concatenate.__call__: Concatenate#__call__().
  QuantizedOp: QuantizedOp#
  OnlyInputOp.__call__: OnlyInputOp#__call__().
  Take.__call__: Take#__call__().
  BatchNorm.__call__: BatchNorm#__call__().
  UfuncCall._fake_quant_inputs: UfuncCall#_fake_quant_inputs().
  QuantizedOp._call_original_op: QuantizedOp#_call_original_op().
  QuantizedOp._inputs_have_activations: QuantizedOp#_inputs_have_activations().
  QuantizedOp._fake_quant_inputs: QuantizedOp#_fake_quant_inputs().
  QuantizedOp.__call__: QuantizedOp#__call__().
  AuxDataKey.IS_ACTIVATION: AuxDataKey#IS_ACTIVATION.
  QuantizedOp._op_name: QuantizedOp#_op_name.
  NoQuantOp.__call__: NoQuantOp#__call__().
  _copy_for_isolation: _copy_for_isolation().
  DotEinsumConv._get_how_to_quantize: DotEinsumConv#_get_how_to_quantize().
  _VALUE_DEPENDENT_METADATA: _VALUE_DEPENDENT_METADATA.
  QuantizedOp.input_idx: QuantizedOp#input_idx.
  UfuncCall._fake_quant_output: UfuncCall#_fake_quant_output().
  AuxDataKey.FQ_RULE: AuxDataKey#FQ_RULE.
  AuxDataKey.WEIGHT_NAME: AuxDataKey#WEIGHT_NAME.
  AuxDataKey.FIXED_RANGE: AuxDataKey#FIXED_RANGE.
  QuantizedOp._get_rule_and_op_id_fn: QuantizedOp#_get_rule_and_op_id_fn.
  UfuncCall._output_allow_fusion: UfuncCall#_output_allow_fusion.
  AuxDataKey.ALLOW_FUSION: AuxDataKey#ALLOW_FUSION.
  QuantizedOp.__init__: QuantizedOp#__init__().
  AuxDataKey.FQ_ARRAY: AuxDataKey#FQ_ARRAY.
  Dropout.__call__: Dropout#__call__().
  DotEinsumConv.__init__: DotEinsumConv#__init__().
  OnlyOutputOp: OnlyOutputOp#
  DotEinsumConv: DotEinsumConv#
  OnlyInputOp: OnlyInputOp#
  NoQuantOp: NoQuantOp#
  QuantizedOp.fixed_range_for_output: QuantizedOp#fixed_range_for_output.
  ModelInput.__call__: ModelInput#__call__().
  PrimitiveBindOp.__init__: PrimitiveBindOp#__init__().
  PrimitiveBindOp.__call__: PrimitiveBindOp#__call__().
  CustomJvpCall.__call__: CustomJvpCall#__call__().
  PrimitiveBindOp: PrimitiveBindOp#
  BatchNorm: BatchNorm#
  Softmax: Softmax#
  Take: Take#
  QuantizedOp._fake_quant_fn: QuantizedOp#_fake_quant_fn.
  GetRuleAndOpIdFn: GetRuleAndOpIdFn.
  FakeQuantFn: FakeQuantFn.
  ModelInput: ModelInput#
  FinalOutput: FinalOutput#
  Dropout: Dropout#
  Tanh: Tanh#
  UfuncCall: UfuncCall#
  Silu: Silu#
  CustomJvpCall: CustomJvpCall#
  ModelInput.__init__: ModelInput#__init__().
  FinalOutput.__init__: FinalOutput#__init__().
  NotAnActivationError: NotAnActivationError.
  UfuncCall._op_name: UfuncCall#_op_name.
  _VALUE_PRESERVING_PRIMITIVES: _VALUE_PRESERVING_PRIMITIVES.
  _LINEAR_ARITHMETIC_PRIMITIVES: _LINEAR_ARITHMETIC_PRIMITIVES.
  QuantizedOp._op_full_name: QuantizedOp#_op_full_name.
  FinalOutput.check_activation: FinalOutput#check_activation.
  DotEinsumConv.check_activation: DotEinsumConv#check_activation.
  DotEinsumConv.disable_per_channel_weights: DotEinsumConv#disable_per_channel_weights.
  OnlyInputOp.input_idx: OnlyInputOp#input_idx.
  OnlyOutputOp.input_idx: OnlyOutputOp#input_idx.
  NoQuantOp.input_idx: NoQuantOp#input_idx.
  Dropout.input_idx: Dropout#input_idx.
  Softmax.input_idx: Softmax#input_idx.
  Softmax.fixed_range_for_output: Softmax#fixed_range_for_output.
  Tanh.input_idx: Tanh#input_idx.
  Tanh.fixed_range_for_output: Tanh#fixed_range_for_output.
  UfuncCall.__call__: UfuncCall#__call__().
  Take.input_idx: Take#input_idx.
  CustomJvpCall.input_idx: CustomJvpCall#input_idx.
---
# Module: [`qwix/_src/providers/odml_ops.py`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py)

## Classes
### `AuxDataKey`  ·  implements/extends Enum, str
- def: [`qwix/_src/providers/odml_ops.py:119`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L119) — documented in [qwix-_src-providers-odml](../../../../concepts/qwix-_src-providers-odml.md)
- doc: Auxiliary data keys.
- signature: `class AuxDataKey(str, enum.Enum):`
- members:
  - `ALLOW_FUSION` — [`L135`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L135) — documented in [qwix-_src-providers-odml_ops](../../../../concepts/qwix-_src-providers-odml_ops.md)
  - `FIXED_RANGE` — [`L149`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L149) — documented in [qwix-_src-providers-odml_ops](../../../../concepts/qwix-_src-providers-odml_ops.md)
  - `FQ_ARRAY` — [`L131`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L131) — documented in [qwix-_src-providers-odml_ops](../../../../concepts/qwix-_src-providers-odml_ops.md)
  - `FQ_RULE` — [`L126`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L126) — documented in [qwix-_src-providers-odml_ops](../../../../concepts/qwix-_src-providers-odml_ops.md)
  - `IS_ACTIVATION` — [`L139`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L139) — documented in [qwix-_src-providers-odml_ops](../../../../concepts/qwix-_src-providers-odml_ops.md)
  - `WEIGHT_NAME` — [`L145`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L145) — documented in [qwix-_src-providers-odml_ops](../../../../concepts/qwix-_src-providers-odml_ops.md)
- used by: [`_maybe_fake_quant`](odml_ops.md#QuantizedOp._maybe_fake_quant), [`__call__`](odml_ops.md#DotEinsumConv.__call__), [`_fake_quant`](odml.md#OdmlQatProvider._fake_quant), [`_fake_quant_output`](odml_ops.md#QuantizedOp._fake_quant_output), [`_fake_quant`](odml.md#OdmlConversionProvider._fake_quant), [`__call__`](odml_ops.md#FinalOutput.__call__), [`__call__`](odml_ops.md#Silu.__call__), [`_forward_metadata`](odml_ops.md#_forward_metadata), [`__call__`](odml_ops.md#OnlyOutputOp.__call__), [`__call__`](odml_ops.md#Concatenate.__call__), [`__call__`](odml_ops.md#OnlyInputOp.__call__), [`__call__`](odml_ops.md#Take.__call__), [`__call__`](odml_ops.md#BatchNorm.__call__), [`_fake_quant_inputs`](odml_ops.md#UfuncCall._fake_quant_inputs), [`_inputs_have_activations`](odml_ops.md#QuantizedOp._inputs_have_activations), [`process_model_inputs`](odml.md#OdmlQatProvider.process_model_inputs), [`_copy_for_isolation`](odml_ops.md#_copy_for_isolation), [`_VALUE_DEPENDENT_METADATA`](odml_ops.md#_VALUE_DEPENDENT_METADATA), [`_fake_quant_output`](odml_ops.md#UfuncCall._fake_quant_output), [`_flatten_dot_general`](odml.md#OdmlConversionProvider._flatten_dot_general), [`nn_param`](odml.md#OdmlQatProvider.nn_param)  (1 test-only)

### `BatchNorm`
- def: [`qwix/_src/providers/odml_ops.py:683`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L683)
- protocol/private: `__call__`[`L686`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L686)
- uses (calls/refs, reference-scoped): [`AuxDataKey`](odml_ops.md#AuxDataKey), [`_maybe_fake_quant`](odml_ops.md#QuantizedOp._maybe_fake_quant), [`get`](../aux_data.md#get), [`_fake_quant_output`](odml_ops.md#QuantizedOp._fake_quant_output), [`QuantizedOp`](odml_ops.md#QuantizedOp), [`IS_ACTIVATION`](odml_ops.md#AuxDataKey.IS_ACTIVATION), [`FQ_RULE`](odml_ops.md#AuxDataKey.FQ_RULE), [`_get_rule_and_op_id_fn`](odml_ops.md#QuantizedOp._get_rule_and_op_id_fn), [`ALLOW_FUSION`](odml_ops.md#AuxDataKey.ALLOW_FUSION)
- used by: [`get_all_ops`](odml_ops.md#get_all_ops)

### `CustomJvpCall`
- def: [`qwix/_src/providers/odml_ops.py:954`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L954)
- members:
  - `input_idx` — [`L962`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L962)
- protocol/private: `__call__`[`L964`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L964)
- uses (calls/refs, reference-scoped): [`__call__`](odml_ops.md#OnlyOutputOp.__call__), [`_call_original_op`](odml_ops.md#QuantizedOp._call_original_op), [`OnlyOutputOp`](odml_ops.md#OnlyOutputOp)
- used by: [`get_all_ops`](odml_ops.md#get_all_ops)

### `DotEinsumConv`
- def: [`qwix/_src/providers/odml_ops.py:828`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L828)
- members:
  - `_get_how_to_quantize(self, for_lhs: bool, qtype: jax.typing.DTypeLike, calibration_method: str, args: Sequence[Any], kwargs: dict[str, Any])` — [`L844`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L844) — Get the HowToQuantize for the given op and arguments. — documented in [qwix-_src-core-dot_general](../../../../concepts/qwix-_src-core-dot_general.md)
  - `check_activation` — [`L832`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L832)
  - `disable_per_channel_weights` — [`L835`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L835)
- protocol/private: `__call__`[`L888`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L888), `__init__`[`L837`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L837)
- uses (calls/refs, reference-scoped): [`weight_qtype`](../qconfig.md#QuantizationRule.weight_qtype), [`HowToQuantize`](../core/qarray.md#HowToQuantize), [`act_qtype`](../qconfig.md#QuantizationRule.act_qtype), [`AuxDataKey`](odml_ops.md#AuxDataKey), [`_maybe_fake_quant`](odml_ops.md#QuantizedOp._maybe_fake_quant), [`get`](../aux_data.md#get), [`_fake_quant_output`](odml_ops.md#QuantizedOp._fake_quant_output), [`act_static_scale`](../qconfig.md#QuantizationRule.act_static_scale), [`get_how_to_quantize`](../core/einsum.md#get_how_to_quantize), [`get_how_to_quantize`](../core/dot_general.md#get_how_to_quantize), [`QuantizedOp`](odml_ops.md#QuantizedOp), [`weight_calibration_method`](../qconfig.md#QuantizationRule.weight_calibration_method), [`_call_original_op`](odml_ops.md#QuantizedOp._call_original_op), [`act_calibration_method`](../qconfig.md#QuantizationRule.act_calibration_method), [`get_how_to_quantize`](../core/conv_general.md#get_how_to_quantize), [`IS_ACTIVATION`](odml_ops.md#AuxDataKey.IS_ACTIVATION), [`_op_name`](odml_ops.md#QuantizedOp._op_name), [`input_idx`](odml_ops.md#QuantizedOp.input_idx), [`WEIGHT_NAME`](odml_ops.md#AuxDataKey.WEIGHT_NAME), [`_get_rule_and_op_id_fn`](odml_ops.md#QuantizedOp._get_rule_and_op_id_fn), [`ALLOW_FUSION`](odml_ops.md#AuxDataKey.ALLOW_FUSION), [`__init__`](odml_ops.md#QuantizedOp.__init__), [`_fake_quant_fn`](odml_ops.md#QuantizedOp._fake_quant_fn), [`NotAnActivationError`](odml_ops.md#NotAnActivationError)
- used by: [`get_all_ops`](odml_ops.md#get_all_ops)

### `Dropout`
- def: [`qwix/_src/providers/odml_ops.py:650`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L650)
- members:
  - `input_idx` — [`L658`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L658)
- protocol/private: `__call__`[`L660`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L660)
- uses (calls/refs, reference-scoped): [`_forward_metadata`](odml_ops.md#_forward_metadata), [`QuantizedOp`](odml_ops.md#QuantizedOp), [`_call_original_op`](odml_ops.md#QuantizedOp._call_original_op), [`input_idx`](odml_ops.md#QuantizedOp.input_idx)
- used by: [`get_all_ops`](odml_ops.md#get_all_ops)

### `FinalOutput`  ·  implements/extends QuantizedOp
- def: [`qwix/_src/providers/odml_ops.py:512`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L512)
- doc: A synthetic op for the model output.
- signature: `class FinalOutput(QuantizedOp):`
- members:
  - `check_activation` — [`L516`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L516)
- protocol/private: `__call__`[`L521`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L521), `__init__`[`L518`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L518)
- uses (calls/refs, reference-scoped): [`AuxDataKey`](odml_ops.md#AuxDataKey), [`_maybe_fake_quant`](odml_ops.md#QuantizedOp._maybe_fake_quant), [`get`](../aux_data.md#get), [`QuantizedOp`](odml_ops.md#QuantizedOp), [`IS_ACTIVATION`](odml_ops.md#AuxDataKey.IS_ACTIVATION), [`_op_name`](odml_ops.md#QuantizedOp._op_name), [`FQ_RULE`](odml_ops.md#AuxDataKey.FQ_RULE), [`FIXED_RANGE`](odml_ops.md#AuxDataKey.FIXED_RANGE), [`_get_rule_and_op_id_fn`](odml_ops.md#QuantizedOp._get_rule_and_op_id_fn), [`__init__`](odml_ops.md#QuantizedOp.__init__), [`fixed_range_for_output`](odml_ops.md#QuantizedOp.fixed_range_for_output), [`NotAnActivationError`](odml_ops.md#NotAnActivationError)
- used by: [`QuantizedOp`](odml_ops.md#QuantizedOp), [`_initial_run_complete`](odml.md#OdmlQatProvider._initial_run_complete)

### `ModelInput`  ·  implements/extends QuantizedOp
- def: [`qwix/_src/providers/odml_ops.py:498`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L498)
- doc: A synthetic op for the model input.
- signature: `class ModelInput(QuantizedOp):`
- protocol/private: `__call__`[`L505`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L505), `__init__`[`L501`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L501)
- uses (calls/refs, reference-scoped): [`_fake_quant_output`](odml_ops.md#QuantizedOp._fake_quant_output), [`QuantizedOp`](odml_ops.md#QuantizedOp), [`__init__`](odml_ops.md#QuantizedOp.__init__), [`clear`](../aux_data.md#clear)
- used by: [`QuantizedOp`](odml_ops.md#QuantizedOp), [`process_model_inputs`](odml.md#OdmlQatProvider.process_model_inputs)

### `NoQuantOp`
- def: [`qwix/_src/providers/odml_ops.py:480`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L480)
- members:
  - `input_idx` — [`L483`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L483)
- protocol/private: `__call__`[`L485`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L485)
- uses (calls/refs, reference-scoped): [`_fake_quant_output`](odml_ops.md#QuantizedOp._fake_quant_output), [`QuantizedOp`](odml_ops.md#QuantizedOp), [`_call_original_op`](odml_ops.md#QuantizedOp._call_original_op), [`_inputs_have_activations`](odml_ops.md#QuantizedOp._inputs_have_activations), [`_fake_quant_inputs`](odml_ops.md#QuantizedOp._fake_quant_inputs), [`_op_name`](odml_ops.md#QuantizedOp._op_name), [`_get_rule_and_op_id_fn`](odml_ops.md#QuantizedOp._get_rule_and_op_id_fn)
- used by: [`get_all_ops`](odml_ops.md#get_all_ops)

### `OnlyInputOp`
- def: [`qwix/_src/providers/odml_ops.py:446`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L446)
- members:
  - `input_idx` — [`L449`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L449)
- protocol/private: `__call__`[`L451`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L451)
- uses (calls/refs, reference-scoped): [`act_qtype`](../qconfig.md#QuantizationRule.act_qtype), [`AuxDataKey`](odml_ops.md#AuxDataKey), [`_fake_quant_output`](odml_ops.md#QuantizedOp._fake_quant_output), [`QuantizedOp`](odml_ops.md#QuantizedOp), [`_call_original_op`](odml_ops.md#QuantizedOp._call_original_op), [`_inputs_have_activations`](odml_ops.md#QuantizedOp._inputs_have_activations), [`_fake_quant_inputs`](odml_ops.md#QuantizedOp._fake_quant_inputs), [`_op_name`](odml_ops.md#QuantizedOp._op_name), [`_get_rule_and_op_id_fn`](odml_ops.md#QuantizedOp._get_rule_and_op_id_fn), [`FQ_ARRAY`](odml_ops.md#AuxDataKey.FQ_ARRAY)
- used by: [`get_all_ops`](odml_ops.md#get_all_ops), [`Take`](odml_ops.md#Take)

### `OnlyOutputOp`
- def: [`qwix/_src/providers/odml_ops.py:464`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L464)
- members:
  - `input_idx` — [`L467`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L467)
- protocol/private: `__call__`[`L469`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L469)
- uses (calls/refs, reference-scoped): [`act_qtype`](../qconfig.md#QuantizationRule.act_qtype), [`AuxDataKey`](odml_ops.md#AuxDataKey), [`get`](../aux_data.md#get), [`_fake_quant_output`](odml_ops.md#QuantizedOp._fake_quant_output), [`QuantizedOp`](odml_ops.md#QuantizedOp), [`_call_original_op`](odml_ops.md#QuantizedOp._call_original_op), [`_inputs_have_activations`](odml_ops.md#QuantizedOp._inputs_have_activations), [`_op_name`](odml_ops.md#QuantizedOp._op_name), [`input_idx`](odml_ops.md#QuantizedOp.input_idx), [`FQ_RULE`](odml_ops.md#AuxDataKey.FQ_RULE), [`_get_rule_and_op_id_fn`](odml_ops.md#QuantizedOp._get_rule_and_op_id_fn)
- used by: [`get_all_ops`](odml_ops.md#get_all_ops), [`__call__`](odml_ops.md#CustomJvpCall.__call__), [`CustomJvpCall`](odml_ops.md#CustomJvpCall)

### `PrimitiveBindOp`  ·  implements/extends QuantizedOp
- def: [`qwix/_src/providers/odml_ops.py:666`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L666)
- doc: Propagates metadata tags across JAX primitives.
- signature: `class PrimitiveBindOp(QuantizedOp):`
- protocol/private: `__call__`[`L677`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L677), `__init__`[`L669`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L669)
- uses (calls/refs, reference-scoped): [`_forward_metadata`](odml_ops.md#_forward_metadata), [`QuantizedOp`](odml_ops.md#QuantizedOp), [`_call_original_op`](odml_ops.md#QuantizedOp._call_original_op), [`__init__`](odml_ops.md#QuantizedOp.__init__), [`PRIMITIVE_BIND_KEY`](../interception.md#PRIMITIVE_BIND_KEY)
- used by: [`QuantizedOp`](odml_ops.md#QuantizedOp), [`get_interceptors`](odml.md#OdmlQatProvider.get_interceptors)  (1 test-only)

### `QuantizedOp`
- def: [`qwix/_src/providers/odml_ops.py:220`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L220) — documented in [qwix-_src-providers-odml](../../../../concepts/qwix-_src-providers-odml.md)
- members:
  - `__call__(self, *args, **kwargs)` — [`L269`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L269) — Quantize the op. — documented in [qwix-_src-providers-odml_ops](../../../../concepts/qwix-_src-providers-odml_ops.md)
  - `__init__(self, *, op_full_name: str, get_rule_and_op_id_fn: GetRuleAndOpIdFn, fake_quant_fn: FakeQuantFn, op_name: str | None = None, **kwargs)` — [`L233`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L233) — Create a generic op.
  - `_call_original_op(self, *args, **kwargs)` — [`L292`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L292) — Get the original function from op_full_name.
  - `_fake_quant_inputs(self, args: Sequence[Any], rule: qconfig.QuantizationRule | None, op_id: str)` — [`L300`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L300) — Fake quantize the inputs of the op. — documented in [qwix-_src-providers-odml_ops](../../../../concepts/qwix-_src-providers-odml_ops.md)
  - `_fake_quant_output(self, outputs: Any, rule: qconfig.QuantizationRule | None)` — [`L430`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L430) — Fake quantize an output activation, which is delayed to the next op. — documented in [qwix-_src-providers-odml_ops](../../../../concepts/qwix-_src-providers-odml_ops.md)
  - `_inputs_have_activations(self, args: Sequence[Any])` — [`L281`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L281) — Check if there's any activation in the inputs. — documented in [qwix-_src-providers-odml_ops](../../../../concepts/qwix-_src-providers-odml_ops.md)
  - `_maybe_fake_quant(self, array: jax.Array, rule: qconfig.QuantizationRule | None, quant_stat_name: str)` — [`L329`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L329) — Fake quantize the array based on the given rule. — documented in [qwix-_src-providers-odml_ops](../../../../concepts/qwix-_src-providers-odml_ops.md)
  - `fixed_range_for_output` — [`L231`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L231)
  - `input_idx` — [`L228`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L228) — documented in [qwix-_src-providers-odml_ops](../../../../concepts/qwix-_src-providers-odml_ops.md)
- protocol/private: `_fake_quant_fn`[`L258`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L258), `_get_rule_and_op_id_fn`[`L257`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L257), `_op_full_name`[`L256`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L256), `_op_name`[`L261`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L261)
- uses (calls/refs, reference-scoped): [`weight_qtype`](../qconfig.md#QuantizationRule.weight_qtype), [`QuantizationRule`](../qconfig.md#QuantizationRule), [`HowToQuantize`](../core/qarray.md#HowToQuantize), [`act_qtype`](../qconfig.md#QuantizationRule.act_qtype), [`qtype`](../core/qarray.md#HowToQuantize.qtype), [`AuxDataKey`](odml_ops.md#AuxDataKey), [`tiled_axes`](../core/qarray.md#HowToQuantize.tiled_axes), [`channelwise_axes`](../core/qarray.md#HowToQuantize.channelwise_axes), [`get`](../aux_data.md#get), [`calibration_method`](../core/qarray.md#HowToQuantize.calibration_method), [`act_static_scale`](../qconfig.md#QuantizationRule.act_static_scale), [`_fake_quant_inputs`](odml_ops.md#UfuncCall._fake_quant_inputs), [`act_calibration_method`](../qconfig.md#QuantizationRule.act_calibration_method), [`IS_ACTIVATION`](odml_ops.md#AuxDataKey.IS_ACTIVATION), [`_copy_for_isolation`](odml_ops.md#_copy_for_isolation), [`_fake_quant_output`](odml_ops.md#UfuncCall._fake_quant_output), [`FQ_RULE`](odml_ops.md#AuxDataKey.FQ_RULE), [`FIXED_RANGE`](odml_ops.md#AuxDataKey.FIXED_RANGE), [`WEIGHT_NAME`](odml_ops.md#AuxDataKey.WEIGHT_NAME), [`FQ_ARRAY`](odml_ops.md#AuxDataKey.FQ_ARRAY), [`act_batch_axes`](../qconfig.md#QuantizationRule.act_batch_axes), [`PrimitiveBindOp`](odml_ops.md#PrimitiveBindOp), [`FinalOutput`](odml_ops.md#FinalOutput), [`ModelInput`](odml_ops.md#ModelInput), [`FakeQuantFn`](odml_ops.md#FakeQuantFn), [`GetRuleAndOpIdFn`](odml_ops.md#GetRuleAndOpIdFn)
- used by: [`__call__`](odml_ops.md#DotEinsumConv.__call__), [`get_all_ops`](odml_ops.md#get_all_ops), [`__call__`](odml_ops.md#FinalOutput.__call__), [`__call__`](odml_ops.md#Silu.__call__), [`__call__`](odml_ops.md#OnlyOutputOp.__call__), [`__call__`](odml_ops.md#Concatenate.__call__), [`__call__`](odml_ops.md#OnlyInputOp.__call__), [`__call__`](odml_ops.md#Take.__call__), [`__call__`](odml_ops.md#BatchNorm.__call__), [`_fake_quant_inputs`](odml_ops.md#UfuncCall._fake_quant_inputs), [`get_intercept_map`](odml.md#OdmlQatProvider.get_intercept_map), [`__call__`](odml_ops.md#NoQuantOp.__call__), [`_get_how_to_quantize`](odml_ops.md#DotEinsumConv._get_how_to_quantize), [`_fake_quant_output`](odml_ops.md#UfuncCall._fake_quant_output), [`_output_allow_fusion`](odml_ops.md#UfuncCall._output_allow_fusion), [`__call__`](odml_ops.md#Dropout.__call__), [`__init__`](odml_ops.md#DotEinsumConv.__init__), [`DotEinsumConv`](odml_ops.md#DotEinsumConv), [`OnlyOutputOp`](odml_ops.md#OnlyOutputOp), [`NoQuantOp`](odml_ops.md#NoQuantOp), [`OnlyInputOp`](odml_ops.md#OnlyInputOp), [`__call__`](odml_ops.md#CustomJvpCall.__call__), [`__call__`](odml_ops.md#ModelInput.__call__), [`__call__`](odml_ops.md#PrimitiveBindOp.__call__), [`__init__`](odml_ops.md#PrimitiveBindOp.__init__), [`BatchNorm`](odml_ops.md#BatchNorm), [`PrimitiveBindOp`](odml_ops.md#PrimitiveBindOp), [`Softmax`](odml_ops.md#Softmax), [`Dropout`](odml_ops.md#Dropout), [`FinalOutput`](odml_ops.md#FinalOutput), [`ModelInput`](odml_ops.md#ModelInput), [`Silu`](odml_ops.md#Silu), [`Tanh`](odml_ops.md#Tanh), [`UfuncCall`](odml_ops.md#UfuncCall), [`__init__`](odml_ops.md#FinalOutput.__init__), [`__init__`](odml_ops.md#ModelInput.__init__)

### `Silu`
- def: [`qwix/_src/providers/odml_ops.py:813`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L813)
- members:
  - `__call__(self, x: jax.Array)` — [`L816`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L816) — QAT silu. — documented in [qwix-_src-providers-odml_ops](../../../../concepts/qwix-_src-providers-odml_ops.md)
- uses (calls/refs, reference-scoped): [`AuxDataKey`](odml_ops.md#AuxDataKey), [`_maybe_fake_quant`](odml_ops.md#QuantizedOp._maybe_fake_quant), [`get`](../aux_data.md#get), [`_fake_quant_output`](odml_ops.md#QuantizedOp._fake_quant_output), [`QuantizedOp`](odml_ops.md#QuantizedOp), [`_call_original_op`](odml_ops.md#QuantizedOp._call_original_op), [`IS_ACTIVATION`](odml_ops.md#AuxDataKey.IS_ACTIVATION), [`_op_name`](odml_ops.md#QuantizedOp._op_name), [`FIXED_RANGE`](odml_ops.md#AuxDataKey.FIXED_RANGE), [`_get_rule_and_op_id_fn`](odml_ops.md#QuantizedOp._get_rule_and_op_id_fn), [`fixed_range_for_output`](odml_ops.md#QuantizedOp.fixed_range_for_output), [`Softmax`](odml_ops.md#Softmax)
- used by: [`get_all_ops`](odml_ops.md#get_all_ops)

### `Softmax`
- def: [`qwix/_src/providers/odml_ops.py:700`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L700)
- members:
  - `fixed_range_for_output` — [`L706`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L706)
  - `input_idx` — [`L703`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L703)
- uses (calls/refs, reference-scoped): [`QuantizedOp`](odml_ops.md#QuantizedOp)
- used by: [`get_all_ops`](odml_ops.md#get_all_ops), [`__call__`](odml_ops.md#Silu.__call__)

### `Take`
- def: [`qwix/_src/providers/odml_ops.py:787`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L787)
- members:
  - `input_idx` — [`L790`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L790)
- protocol/private: `__call__`[`L792`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L792)
- uses (calls/refs, reference-scoped): [`act_qtype`](../qconfig.md#QuantizationRule.act_qtype), [`AuxDataKey`](odml_ops.md#AuxDataKey), [`_maybe_fake_quant`](odml_ops.md#QuantizedOp._maybe_fake_quant), [`_fake_quant_output`](odml_ops.md#QuantizedOp._fake_quant_output), [`_call_original_op`](odml_ops.md#QuantizedOp._call_original_op), [`_inputs_have_activations`](odml_ops.md#QuantizedOp._inputs_have_activations), [`_op_name`](odml_ops.md#QuantizedOp._op_name), [`_get_rule_and_op_id_fn`](odml_ops.md#QuantizedOp._get_rule_and_op_id_fn), [`FQ_ARRAY`](odml_ops.md#AuxDataKey.FQ_ARRAY), [`OnlyInputOp`](odml_ops.md#OnlyInputOp)
- used by: [`get_all_ops`](odml_ops.md#get_all_ops)

### `Tanh`
- def: [`qwix/_src/providers/odml_ops.py:709`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L709)
- members:
  - `fixed_range_for_output` — [`L715`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L715)
  - `input_idx` — [`L712`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L712)
- uses (calls/refs, reference-scoped): [`QuantizedOp`](odml_ops.md#QuantizedOp)
- used by: [`get_all_ops`](odml_ops.md#get_all_ops)

### `UfuncCall`
- def: [`qwix/_src/providers/odml_ops.py:718`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L718)
- members:
  - `_fake_quant_inputs(self, args: Sequence[Any], rule: qconfig.QuantizationRule | None, op_id: str)` — [`L728`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L728) — Fake quantize the inputs of the op. — documented in [qwix-_src-providers-odml_ops](../../../../concepts/qwix-_src-providers-odml_ops.md)
- protocol/private: `__call__`[`L721`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L721), `_fake_quant_output`[`L745`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L745), `_op_name`[`L723`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L723), `_output_allow_fusion`[`L724`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L724)
- uses (calls/refs, reference-scoped): [`QuantizationRule`](../qconfig.md#QuantizationRule), [`AuxDataKey`](odml_ops.md#AuxDataKey), [`get`](../aux_data.md#get), [`_fake_quant_output`](odml_ops.md#QuantizedOp._fake_quant_output), [`QuantizedOp`](odml_ops.md#QuantizedOp), [`_fake_quant_inputs`](odml_ops.md#QuantizedOp._fake_quant_inputs), [`__call__`](odml_ops.md#QuantizedOp.__call__), [`IS_ACTIVATION`](odml_ops.md#AuxDataKey.IS_ACTIVATION), [`input_idx`](odml_ops.md#QuantizedOp.input_idx), [`ALLOW_FUSION`](odml_ops.md#AuxDataKey.ALLOW_FUSION)
- used by: [`_fake_quant_output`](odml_ops.md#QuantizedOp._fake_quant_output), [`get_all_ops`](odml_ops.md#get_all_ops), [`_fake_quant_inputs`](odml_ops.md#QuantizedOp._fake_quant_inputs)

## Functions
- `_copy_for_isolation(original_array: jax.Array)` — [`L194`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L194) — Creates a copy of the array to isolate it from other branches.
- `_forward_metadata(inputs: Any, outputs: Any, primitive_name: str | None = None)` — [`L534`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L534) — Forwards metadata from inputs to outputs. — documented in [qwix-_src-providers-odml_ops](../../../../concepts/qwix-_src-providers-odml_ops.md)
- `get_all_ops()` — [`L33`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L33) — Get all the functions to intercept and the corresponding ops. — documented in [qwix-_src-providers-odml](../../../../concepts/qwix-_src-providers-odml.md)

## Module values
- `FakeQuantFn` — [`L191`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L191)
- `GetRuleAndOpIdFn` — [`L190`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L190)
- `NotAnActivationError` — [`L111`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L111)
- `_LINEAR_ARITHMETIC_PRIMITIVES` — [`L181`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L181)
- `_VALUE_DEPENDENT_METADATA` — [`L154`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L154) — documented in [qwix-_src-providers-odml_ops](../../../../concepts/qwix-_src-providers-odml_ops.md)
- `_VALUE_PRESERVING_PRIMITIVES` — [`L163`](../../../../../../../raw/code/qwix/qwix/_src/providers/odml_ops.py#L163)

