---
title: 'Module: tests/contrib/kernels/quantized_matmul_test.py'
type: catalog
provenance: extracted
module: tests/contrib/kernels/quantized_matmul_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests.contrib.kernels.quantized_matmul_test`/QuantizedMatmulTest#
symbols:
  QuantizedMatmulTest.test_kernel_dot_general: test_kernel_dot_general().
  QuantizedMatmulTest: ''
  QuantizedMatmulTest.test_can_use_qmm_1: test_can_use_qmm_1().
  QuantizedMatmulTest.test_can_use_qmm_2: test_can_use_qmm_2().
  QuantizedMatmulTest.generate_quantized_arrays: generate_quantized_arrays().
  QuantizedMatmulTest.jax_answer: jax_answer().
---
# Module: [`tests/contrib/kernels/quantized_matmul_test.py`](../../../../../../../raw/code/qwix/tests/contrib/kernels/quantized_matmul_test.py)

## Classes
### `QuantizedMatmulTest`  ·  implements/extends TestCase
- def: [`tests/contrib/kernels/quantized_matmul_test.py:9`](../../../../../../../raw/code/qwix/tests/contrib/kernels/quantized_matmul_test.py#L9)
- signature: `class QuantizedMatmulTest(parameterized.TestCase):`
- members:
  - `generate_quantized_arrays(data_shape, scale_shape, dtype, *, key: jax.Array)` — [`L30`](../../../../../../../raw/code/qwix/tests/contrib/kernels/quantized_matmul_test.py#L30)
  - `jax_answer(x: jax.Array, sx: jax.Array, y: jax.Array, sy: jax.Array)` — [`L12`](../../../../../../../raw/code/qwix/tests/contrib/kernels/quantized_matmul_test.py#L12)
  - `test_can_use_qmm_1(self)` — [`L69`](../../../../../../../raw/code/qwix/tests/contrib/kernels/quantized_matmul_test.py#L69)
  - `test_can_use_qmm_2(self)` — [`L91`](../../../../../../../raw/code/qwix/tests/contrib/kernels/quantized_matmul_test.py#L91)
  - `test_kernel_dot_general(self, m, k, n, sm, sk, sn, bm, bk, bn, dtype)` — [`L46`](../../../../../../../raw/code/qwix/tests/contrib/kernels/quantized_matmul_test.py#L46)
- uses (calls/refs, reference-scoped): [`quantized_matmul`](../../../qwix/contrib/kernels/quantized_matmul.md#quantized_matmul), [`can_use_qmm`](../../../qwix/contrib/kernels/quantized_matmul.md#can_use_qmm)

