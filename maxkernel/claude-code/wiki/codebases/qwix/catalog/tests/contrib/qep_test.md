---
title: 'Module: tests/contrib/qep_test.py'
type: catalog
provenance: extracted
module: tests/contrib/qep_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests.contrib.qep_test`/
symbols:
  QepLinenTest.test_single_layer_qep_beats_ptq_and_matches_gptq: QepLinenTest#test_single_layer_qep_beats_ptq_and_matches_gptq().
  QepLinenTest._manual_two_stage_reference: QepLinenTest#_manual_two_stage_reference().
  QepLinenTest.test_quantize_params_without_correction_does_not_require_hessian_delta: QepLinenTest#test_quantize_params_without_correction_does_not_require_hessian_delta().
  QepLinenTest.test_exact_stagewise_matches_manual_two_stage_reference: QepLinenTest#test_exact_stagewise_matches_manual_two_stage_reference().
  QepLinenTest.test_infers_shared_input_branch_stage: QepLinenTest#test_infers_shared_input_branch_stage().
  QepLinenTest.test_no_matching_layers_raises: QepLinenTest#test_no_matching_layers_raises().
  QepLinenTest.test_non_reiterable_input_raises: QepLinenTest#test_non_reiterable_input_raises().
  QepLinenTest._make_ptq_model: QepLinenTest#_make_ptq_model().
  QepLinenTest._make_dense_model: QepLinenTest#_make_dense_model().
  _mae: _mae().
  QepLinenTest._make_branch_model: QepLinenTest#_make_branch_model().
  QepLinenTest._get_abs_quantized: QepLinenTest#_get_abs_quantized().
  QepLinenTest._make_dense_model.DenseModel: QepLinenTest#_make_dense_model().DenseModel#
  QepLinenTest._make_branch_model.BranchModel: QepLinenTest#_make_branch_model().BranchModel#
  QepLinenTest: QepLinenTest#
  QepLinenTest._make_dense_model.DenseModel.__call__: QepLinenTest#_make_dense_model().DenseModel#__call__().
  QepLinenTest._make_branch_model.BranchModel.__call__: QepLinenTest#_make_branch_model().BranchModel#__call__().
---
# Module: [`tests/contrib/qep_test.py`](../../../../../../raw/code/qwix/tests/contrib/qep_test.py)

## Classes
### `BranchModel`  ·  implements/extends Module
- def: [`tests/contrib/qep_test.py:50`](../../../../../../raw/code/qwix/tests/contrib/qep_test.py#L50)
- signature: `class BranchModel(nn.Module):`
- protocol/private: `__call__`[`L53`](../../../../../../raw/code/qwix/tests/contrib/qep_test.py#L53)
- used by: (1 test-only callers)

### `DenseModel`  ·  implements/extends Module
- def: [`tests/contrib/qep_test.py:35`](../../../../../../raw/code/qwix/tests/contrib/qep_test.py#L35)
- signature: `class DenseModel(nn.Module):`
- protocol/private: `__call__`[`L38`](../../../../../../raw/code/qwix/tests/contrib/qep_test.py#L38)
- used by: (1 test-only callers)

### `QepLinenTest`  ·  implements/extends TestCase
- def: [`tests/contrib/qep_test.py:32`](../../../../../../raw/code/qwix/tests/contrib/qep_test.py#L32)
- signature: `class QepLinenTest(parameterized.TestCase):`
- members:
  - `test_exact_stagewise_matches_manual_two_stage_reference(self)` — [`L123`](../../../../../../raw/code/qwix/tests/contrib/qep_test.py#L123)
  - `test_infers_shared_input_branch_stage(self)` — [`L135`](../../../../../../raw/code/qwix/tests/contrib/qep_test.py#L135)
  - `test_no_matching_layers_raises(self)` — [`L147`](../../../../../../raw/code/qwix/tests/contrib/qep_test.py#L147)
  - `test_non_reiterable_input_raises(self)` — [`L160`](../../../../../../raw/code/qwix/tests/contrib/qep_test.py#L160)
  - `test_quantize_params_without_correction_does_not_require_hessian_delta(self)` — [`L169`](../../../../../../raw/code/qwix/tests/contrib/qep_test.py#L169)
  - `test_single_layer_qep_beats_ptq_and_matches_gptq(self)` — [`L95`](../../../../../../raw/code/qwix/tests/contrib/qep_test.py#L95)
- protocol/private: `_get_abs_quantized`[`L67`](../../../../../../raw/code/qwix/tests/contrib/qep_test.py#L67), `_make_branch_model`[`L49`](../../../../../../raw/code/qwix/tests/contrib/qep_test.py#L49), `_make_dense_model`[`L34`](../../../../../../raw/code/qwix/tests/contrib/qep_test.py#L34), `_make_ptq_model`[`L62`](../../../../../../raw/code/qwix/tests/contrib/qep_test.py#L62), `_manual_two_stage_reference`[`L70`](../../../../../../raw/code/qwix/tests/contrib/qep_test.py#L70)
- uses (calls/refs, reference-scoped): [`weight_qtype`](../../qwix/_src/qconfig.md#QuantizationRule.weight_qtype), [`quantize_model`](../../qwix/_src/model.md#quantize_model), [`module_path`](../../qwix/_src/qconfig.md#QuantizationRule.module_path), [`PtqProvider`](../../qwix/_src/providers/ptq.md#PtqProvider), [`quantize`](../../qwix/contrib/qep.md#quantize), [`quantize_params`](../../qwix/_src/providers/ptq.md#quantize_params), [`QepRule`](../../qwix/contrib/qep.md#QepRule), [`GptqRule`](../../qwix/contrib/gptq.md#GptqRule), [`init`](../../qwix/_src/averaging.md#SimpleMovingAverage.init), [`update`](../../qwix/_src/averaging.md#SimpleMovingAverage.update), [`SimpleMovingAverage`](../../qwix/_src/averaging.md#SimpleMovingAverage), [`quantize_params`](../../qwix/contrib/gptq.md#quantize_params), [`GptqCalibrationProvider`](../../qwix/contrib/gptq.md#GptqCalibrationProvider), [`quantize_params`](../../qwix/contrib/qep.md#quantize_params), [`stages`](../../qwix/contrib/qep.md#QepResult.stages), [`compute_qep_stats`](../../qwix/contrib/qep_core.md#compute_qep_stats), [`params`](../../qwix/contrib/qep.md#QepResult.params), [`model`](../../qwix/contrib/qep.md#QepResult.model), [`module_paths`](../../qwix/contrib/qep.md#QepStage.module_paths)  (3 test-only)

## Functions
- `_mae(a, b)` — [`L28`](../../../../../../raw/code/qwix/tests/contrib/qep_test.py#L28)

