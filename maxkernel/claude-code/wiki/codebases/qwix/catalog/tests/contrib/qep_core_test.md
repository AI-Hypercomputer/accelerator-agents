---
title: 'Module: tests/contrib/qep_core_test.py'
type: catalog
provenance: extracted
module: tests/contrib/qep_core_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests.contrib.qep_core_test`/
symbols:
  QepCoreTest.test_qep_quantize_weight_matmul_accuracy: QepCoreTest#test_qep_quantize_weight_matmul_accuracy().
  QepCoreTest.test_compute_qep_stats_identical_inputs: QepCoreTest#test_compute_qep_stats_identical_inputs().
  rel_rmse: rel_rmse().
  QepCoreTest.test_compute_qep_stats_shapes: QepCoreTest#test_compute_qep_stats_shapes().
  QepCoreTest.test_weight_correct_identity_with_zero_delta: QepCoreTest#test_weight_correct_identity_with_zero_delta().
  QepCoreTest.test_weight_correct_zero_correction_factor: QepCoreTest#test_weight_correct_zero_correction_factor().
  QepCoreTest.test_weight_correct_reduces_output_error: QepCoreTest#test_weight_correct_reduces_output_error().
  QepCoreTest: QepCoreTest#
---
# Module: [`tests/contrib/qep_core_test.py`](../../../../../../raw/code/qwix/tests/contrib/qep_core_test.py)

## Classes
### `QepCoreTest`  ·  implements/extends TestCase
- def: [`tests/contrib/qep_core_test.py:31`](../../../../../../raw/code/qwix/tests/contrib/qep_core_test.py#L31)
- signature: `class QepCoreTest(parameterized.TestCase):`
- members:
  - `test_compute_qep_stats_identical_inputs(self)` — [`L46`](../../../../../../raw/code/qwix/tests/contrib/qep_core_test.py#L46) — Tests that hessian_delta is zero when inputs are identical.
  - `test_compute_qep_stats_shapes(self)` — [`L33`](../../../../../../raw/code/qwix/tests/contrib/qep_core_test.py#L33) — Tests that QEP stats have correct shapes.
  - `test_qep_quantize_weight_matmul_accuracy(self, groupsize, blocksize)` — [`L100`](../../../../../../raw/code/qwix/tests/contrib/qep_core_test.py#L100) — Tests that QEP (weight_correct + GPTQ) improves matmul accuracy.
  - `test_weight_correct_identity_with_zero_delta(self)` — [`L56`](../../../../../../raw/code/qwix/tests/contrib/qep_core_test.py#L56) — Tests that weight_correct with zero H_delta produces no correction.
  - `test_weight_correct_reduces_output_error(self)` — [`L74`](../../../../../../raw/code/qwix/tests/contrib/qep_core_test.py#L74) — Tests that weight correction reduces ||W @ X_float - W_corr @ X_q||.
  - `test_weight_correct_zero_correction_factor(self)` — [`L64`](../../../../../../raw/code/qwix/tests/contrib/qep_core_test.py#L64) — Tests that correction_factor=0.0 produces no correction.
- uses (calls/refs, reference-scoped): [`HowToQuantize`](../../qwix/_src/core/qarray.md#HowToQuantize), [`qtype`](../../qwix/_src/core/qarray.md#HowToQuantize.qtype), [`quantize`](../../qwix/_src/core/qarray.md#quantize), [`dequantize`](../../qwix/_src/core/qarray.md#dequantize), [`tiled_axes`](../../qwix/_src/core/qarray.md#HowToQuantize.tiled_axes), [`channelwise_axes`](../../qwix/_src/core/qarray.md#HowToQuantize.channelwise_axes), [`quantize_weight`](../../qwix/contrib/gptq_core.md#quantize_weight), [`weight_correct`](../../qwix/contrib/qep_core.md#weight_correct), [`compute_hessian`](../../qwix/contrib/gptq_core.md#compute_hessian), [`compute_qep_stats`](../../qwix/contrib/qep_core.md#compute_qep_stats)  (1 test-only)

## Functions
- `rel_rmse(x: jax.Array, y: jax.Array)` — [`L27`](../../../../../../raw/code/qwix/tests/contrib/qep_core_test.py#L27)

