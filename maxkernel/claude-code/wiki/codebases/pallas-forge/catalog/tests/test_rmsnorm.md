---
title: 'Module: tests/test_rmsnorm.py'
type: catalog
provenance: extracted
module: tests/test_rmsnorm.py
status: fresh
symbol_base: scip-python python pallas-forge 0.0.0 `tests.test_rmsnorm`/TestFusedRMSNormResidual#
symbols:
  TestFusedRMSNormResidual.test_basic_correctness: test_basic_correctness().
  TestFusedRMSNormResidual.test_unit_weight: test_unit_weight().
  TestFusedRMSNormResidual.test_output_shapes: test_output_shapes().
  TestFusedRMSNormResidual.test_zero_residual: test_zero_residual().
  TestFusedRMSNormResidual.test_eps_prevents_div_by_zero: test_eps_prevents_div_by_zero().
  TestFusedRMSNormResidual.test_bfloat16: test_bfloat16().
  TestFusedRMSNormResidual.test_shape_mismatch_raises: test_shape_mismatch_raises().
  TestFusedRMSNormResidual.test_weight_shape_mismatch_raises: test_weight_shape_mismatch_raises().
  TestFusedRMSNormResidual: ''
---
# Module: [`tests/test_rmsnorm.py`](../../../../../raw/code/pallas-forge/tests/test_rmsnorm.py)

## Classes
### `TestFusedRMSNormResidual`
- def: [`tests/test_rmsnorm.py:12`](../../../../../raw/code/pallas-forge/tests/test_rmsnorm.py#L12)
- doc: Correctness tests for fused_rmsnorm_residual.
- signature: `class TestFusedRMSNormResidual:`
- members:
  - `test_basic_correctness(self, rng_key)` — [`L15`](../../../../../raw/code/pallas-forge/tests/test_rmsnorm.py#L15) — Basic correctness against reference implementation.
  - `test_bfloat16(self, rng_key)` — [`L77`](../../../../../raw/code/pallas-forge/tests/test_rmsnorm.py#L77) — Should work with bfloat16 inputs.
  - `test_eps_prevents_div_by_zero(self, rng_key)` — [`L67`](../../../../../raw/code/pallas-forge/tests/test_rmsnorm.py#L67) — Epsilon should prevent division by zero for all-zero input.
  - `test_output_shapes(self, rng_key)` — [`L33`](../../../../../raw/code/pallas-forge/tests/test_rmsnorm.py#L33) — Output shapes should match input shapes.
  - `test_shape_mismatch_raises(self, rng_key)` — [`L89`](../../../../../raw/code/pallas-forge/tests/test_rmsnorm.py#L89) — Mismatched shapes should raise ValueError.
  - `test_unit_weight(self, rng_key)` — [`L55`](../../../../../raw/code/pallas-forge/tests/test_rmsnorm.py#L55) — With weight=1, output is just normalized x+residual.
  - `test_weight_shape_mismatch_raises(self, rng_key)` — [`L98`](../../../../../raw/code/pallas-forge/tests/test_rmsnorm.py#L98) — Wrong weight shape should raise ValueError.
  - `test_zero_residual(self, rng_key)` — [`L45`](../../../../../raw/code/pallas-forge/tests/test_rmsnorm.py#L45) — With zero residual, new_residual should equal x.
- uses (calls/refs, reference-scoped): [`fused_rmsnorm_residual`](../pallas_forge/kernels/rmsnorm.md#fused_rmsnorm_residual), [`rmsnorm_reference`](../pallas_forge/kernels/rmsnorm.md#rmsnorm_reference)

