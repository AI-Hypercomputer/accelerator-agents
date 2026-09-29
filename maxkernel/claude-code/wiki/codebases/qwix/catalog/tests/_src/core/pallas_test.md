---
title: 'Module: tests/_src/core/pallas_test.py'
type: catalog
provenance: extracted
module: tests/_src/core/pallas_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.core.pallas_test`/PallasTest#
symbols:
  PallasTest.test_pallas_batch_matmul: test_pallas_batch_matmul().
  PallasTest.dequantize_pallas: dequantize_pallas().
  PallasTest.test_update_block_specs_for_qarray: test_update_block_specs_for_qarray().
  PallasTest.test_pallas_dequantize: test_pallas_dequantize().
  PallasTest.pallas_batch_matmul: pallas_batch_matmul().
  PallasTest.test_pallas_dot_general: test_pallas_dot_general().
  PallasTest.test_transform_block_specs_for_tpu: test_transform_block_specs_for_tpu().
  PallasTest._make_array: _make_array().
  PallasTest.pallas_call_kwargs: pallas_call_kwargs.
  PallasTest.dequantize_kernel: dequantize_kernel().
  PallasTest.pallas_matmul_kernel: pallas_matmul_kernel().
  PallasTest.pl_kernel: pl_kernel().
  PallasTest.setUpClass: setUpClass().
  PallasTest: ''
  PallasTest._init: _init().
  PallasTest._write: _write().
---
# Module: [`tests/_src/core/pallas_test.py`](../../../../../../../raw/code/qwix/tests/_src/core/pallas_test.py)

## Classes
### `PallasTest`  ·  implements/extends TestCase
- def: [`tests/_src/core/pallas_test.py:28`](../../../../../../../raw/code/qwix/tests/_src/core/pallas_test.py#L28)
- signature: `class PallasTest(parameterized.TestCase):`
- members:
  - `dequantize_kernel(scalar_ref, q_ref, out_ref)` — [`L202`](../../../../../../../raw/code/qwix/tests/_src/core/pallas_test.py#L202)
  - `dequantize_pallas(q: qarray.QArray)` — [`L206`](../../../../../../../raw/code/qwix/tests/_src/core/pallas_test.py#L206)
  - `pallas_batch_matmul(x: qarray.QArray, y: qarray.QArray, *, bm: int = 128, bk: int = 128, bn: int = 128)` — [`L246`](../../../../../../../raw/code/qwix/tests/_src/core/pallas_test.py#L246)
  - `pallas_matmul_kernel(x_ref, y_ref, z_ref, acc_ref, *, nsteps)` — [`L232`](../../../../../../../raw/code/qwix/tests/_src/core/pallas_test.py#L232)
  - `pl_kernel(x, y, o)` — [`L358`](../../../../../../../raw/code/qwix/tests/_src/core/pallas_test.py#L358)
  - `setUpClass(cls)` — [`L31`](../../../../../../../raw/code/qwix/tests/_src/core/pallas_test.py#L31)
  - `test_pallas_batch_matmul(self)` — [`L229`](../../../../../../../raw/code/qwix/tests/_src/core/pallas_test.py#L229) — A basic example of using Qwix pallas_call to implement batch_matmul.
  - `test_pallas_dequantize(self, input_shape, tiled_axes, bs=(128, 128))` — [`L199`](../../../../../../../raw/code/qwix/tests/_src/core/pallas_test.py#L199) — Comprehensive tests for the pallas_call function.
  - `test_pallas_dot_general(self, *, lhs_shape, lhs_dtype=jnp.int8, lhs_scale_shape, lhs_scale_dtype=jnp.float32, rhs_shape, rhs_dtype=jnp.int8, rhs_scale_shape, rhs_scale_dtype=jnp.float32, dimension_numbers)` — [`L343`](../../../../../../../raw/code/qwix/tests/_src/core/pallas_test.py#L343) — Test what kind of dot_general can be called in pallas kernels.
  - `test_transform_block_specs_for_tpu(self, arg_shape, block_shape, expected_new_block_shape, expected_new_index_map)` — [`L120`](../../../../../../../raw/code/qwix/tests/_src/core/pallas_test.py#L120)
  - `test_update_block_specs_for_qarray(self, block_shape, qvalue_shape, scale_shape, expected_scale_block_shape, expected_scale_index_map)` — [`L65`](../../../../../../../raw/code/qwix/tests/_src/core/pallas_test.py#L65)
  - `pallas_call_kwargs` — [`L35`](../../../../../../../raw/code/qwix/tests/_src/core/pallas_test.py#L35)
- protocol/private: `_init`[`L235`](../../../../../../../raw/code/qwix/tests/_src/core/pallas_test.py#L235), `_make_array`[`L386`](../../../../../../../raw/code/qwix/tests/_src/core/pallas_test.py#L386), `_write`[`L243`](../../../../../../../raw/code/qwix/tests/_src/core/pallas_test.py#L243)
- uses (calls/refs, reference-scoped): [`QArray`](../../../qwix/_src/core/qarray.md#QArray), [`HowToQuantize`](../../../qwix/_src/core/qarray.md#HowToQuantize), [`qtype`](../../../qwix/_src/core/qarray.md#HowToQuantize.qtype), [`quantize`](../../../qwix/_src/core/qarray.md#quantize), [`qvalue`](../../../qwix/_src/core/qarray.md#QArray.qvalue), [`scale`](../../../qwix/_src/core/qarray.md#QArray.scale), [`dequantize`](../../../qwix/_src/core/qarray.md#dequantize), [`tiled_axes`](../../../qwix/_src/core/qarray.md#HowToQuantize.tiled_axes), [`channelwise_axes`](../../../qwix/_src/core/qarray.md#HowToQuantize.channelwise_axes), [`dot_general`](../../../qwix/_src/core/dot_general.md#dot_general), [`loop_dot_general`](../../../qwix/_src/core/dot_general.md#loop_dot_general), [`zero_point`](../../../qwix/_src/core/qarray.md#QArray.zero_point), [`qtype`](../../../qwix/_src/core/qarray.md#QArray.qtype), [`transform_block_specs_for_tpu`](../../../qwix/_src/core/pallas.md#transform_block_specs_for_tpu), [`pallas_call`](../../../qwix/_src/core/pallas.md#pallas_call), [`update_block_specs_for_qarray`](../../../qwix/_src/core/pallas.md#update_block_specs_for_qarray), [`_can_fit_tpu_requirements`](../../../qwix/_src/core/pallas.md#_can_fit_tpu_requirements)

