---
title: 'Module: tests/_src/core/mxfp_dot_test.py'
type: catalog
provenance: extracted
module: tests/_src/core/mxfp_dot_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.core.mxfp_dot_test`/
symbols:
  MxfpDotTest.test_one_side_mxfp_fallback: MxfpDotTest#test_one_side_mxfp_fallback().
  MxfpNumericsTest.run_mxfp_test: MxfpNumericsTest#run_mxfp_test().
  MxfpDotTest.test_unflatten_from_3d: MxfpDotTest#test_unflatten_from_3d().
  MxfpDotTest.test_mxfp_dot_general_emulation_fallback: MxfpDotTest#test_mxfp_dot_general_emulation_fallback().
  MxfpDotTest.test_flatten_to_3d: MxfpDotTest#test_flatten_to_3d().
  MxfpDotTest.test_flatten_to_3d_with_broadcasting: MxfpDotTest#test_flatten_to_3d_with_broadcasting().
  MxfpNumericsTest.test_matmul_f32_baseline: MxfpNumericsTest#test_matmul_f32_baseline().
  MxfpNumericsTest.setUp: MxfpNumericsTest#setUp().
  MxfpNumericsTest.test_scaled_matmul_mxfp8: MxfpNumericsTest#test_scaled_matmul_mxfp8().
  MxfpNumericsTest.test_scaled_matmul_mxfp4: MxfpNumericsTest#test_scaled_matmul_mxfp4().
  MxfpNumericsTest.test_scaled_matmul_nvfp4: MxfpNumericsTest#test_scaled_matmul_nvfp4().
  local_quantize: local_quantize().
  MxfpNumericsTest._get_hlo: MxfpNumericsTest#_get_hlo().
  MxfpNumericsTest._generate_test_data: MxfpNumericsTest#_generate_test_data().
  reference_scaled_matmul: reference_scaled_matmul().
  MxfpNumericsTest._log_device_info: MxfpNumericsTest#_log_device_info().
  MxfpNumericsTest: MxfpNumericsTest#
  MxfpDotTest: MxfpDotTest#
---
# Module: [`tests/_src/core/mxfp_dot_test.py`](../../../../../../../raw/code/qwix/tests/_src/core/mxfp_dot_test.py)

## Classes
### `MxfpDotTest`  ·  implements/extends TestCase
- def: [`tests/_src/core/mxfp_dot_test.py:265`](../../../../../../../raw/code/qwix/tests/_src/core/mxfp_dot_test.py#L265)
- doc: Tests for mxfp_dot dispatcher and shape handling.
- signature: `class MxfpDotTest(absltest.TestCase):`
- members:
  - `test_flatten_to_3d(self)` — [`L268`](../../../../../../../raw/code/qwix/tests/_src/core/mxfp_dot_test.py#L268)
  - `test_flatten_to_3d_with_broadcasting(self)` — [`L279`](../../../../../../../raw/code/qwix/tests/_src/core/mxfp_dot_test.py#L279)
  - `test_mxfp_dot_general_emulation_fallback(self)` — [`L306`](../../../../../../../raw/code/qwix/tests/_src/core/mxfp_dot_test.py#L306)
  - `test_one_side_mxfp_fallback(self, mock_devices)` — [`L326`](../../../../../../../raw/code/qwix/tests/_src/core/mxfp_dot_test.py#L326)
  - `test_unflatten_from_3d(self)` — [`L290`](../../../../../../../raw/code/qwix/tests/_src/core/mxfp_dot_test.py#L290)
- uses (calls/refs, reference-scoped): [`QArray`](../../../qwix/_src/core/qarray.md#QArray), [`qvalue`](../../../qwix/_src/core/qarray.md#QArray.qvalue), [`scale`](../../../qwix/_src/core/qarray.md#QArray.scale), [`qtype`](../../../qwix/_src/core/qarray.md#QArray.qtype), [`_flatten_to_3d`](../../../qwix/_src/core/mxfp_dot.md#_flatten_to_3d), [`mxfp_dot_general`](../../../qwix/_src/core/mxfp_dot.md#mxfp_dot_general), [`_unflatten_from_3d`](../../../qwix/_src/core/mxfp_dot.md#_unflatten_from_3d), [`_get_primary_platform`](../../../qwix/_src/core/mxfp_dot.md#_get_primary_platform)

### `MxfpNumericsTest`  ·  implements/extends TestCase
- def: [`tests/_src/core/mxfp_dot_test.py:103`](../../../../../../../raw/code/qwix/tests/_src/core/mxfp_dot_test.py#L103)
- doc: Unit tests for MXFP numerical correctness and hardware acceleration.
- signature: `class MxfpNumericsTest(absltest.TestCase):`
- members:
  - `_generate_test_data(self, seed=123)` — [`L132`](../../../../../../../raw/code/qwix/tests/_src/core/mxfp_dot_test.py#L132) — Generates standard test data (lhs, rhs) in FP32.
  - `run_mxfp_test(self, mxfp_format, data_type, scale_type, block_size)` — [`L171`](../../../../../../../raw/code/qwix/tests/_src/core/mxfp_dot_test.py#L171) — Helper to run a specific MXFP configuration test.
  - `setUp(self)` — [`L106`](../../../../../../../raw/code/qwix/tests/_src/core/mxfp_dot_test.py#L106)
  - `test_matmul_f32_baseline(self)` — [`L144`](../../../../../../../raw/code/qwix/tests/_src/core/mxfp_dot_test.py#L144) — Sanity check for scaled_matmul with FP32 and unit scales.
  - `test_scaled_matmul_mxfp4(self)` — [`L248`](../../../../../../../raw/code/qwix/tests/_src/core/mxfp_dot_test.py#L248)
  - `test_scaled_matmul_mxfp8(self)` — [`L240`](../../../../../../../raw/code/qwix/tests/_src/core/mxfp_dot_test.py#L240)
  - `test_scaled_matmul_nvfp4(self)` — [`L256`](../../../../../../../raw/code/qwix/tests/_src/core/mxfp_dot_test.py#L256)
- protocol/private: `_get_hlo`[`L125`](../../../../../../../raw/code/qwix/tests/_src/core/mxfp_dot_test.py#L125), `_log_device_info`[`L110`](../../../../../../../raw/code/qwix/tests/_src/core/mxfp_dot_test.py#L110)
- uses (calls/refs, reference-scoped): (2 test-only callers)

## Functions
- `local_quantize(x, data_type, scale_type, block_size)` — [`L72`](../../../../../../../raw/code/qwix/tests/_src/core/mxfp_dot_test.py#L72) — Simplified version of quantize for testing.
- `reference_scaled_matmul(lhs, rhs, lhs_scale, rhs_scale)` — [`L46`](../../../../../../../raw/code/qwix/tests/_src/core/mxfp_dot_test.py#L46) — Reference implementation using JNP.

