---
title: 'Module: tests/contrib/kernels/lhs_fused_quantized_matmul_test.py'
type: catalog
provenance: extracted
module: tests/contrib/kernels/lhs_fused_quantized_matmul_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests.contrib.kernels.lhs_fused_quantized_matmul_test`/LHSFusedQuantizedMatmulTest#
symbols:
  LHSFusedQuantizedMatmulTest.test_kernel_dot_general: test_kernel_dot_general().
  LHSFusedQuantizedMatmulTest.jax_answer: jax_answer().
  LHSFusedQuantizedMatmulTest: ''
  LHSFusedQuantizedMatmulTest.generate_quantized_arrays: generate_quantized_arrays().
---
# Module: [`tests/contrib/kernels/lhs_fused_quantized_matmul_test.py`](../../../../../../../raw/code/qwix/tests/contrib/kernels/lhs_fused_quantized_matmul_test.py)

## Classes
### `LHSFusedQuantizedMatmulTest`  ·  implements/extends TestCase
- def: [`tests/contrib/kernels/lhs_fused_quantized_matmul_test.py:9`](../../../../../../../raw/code/qwix/tests/contrib/kernels/lhs_fused_quantized_matmul_test.py#L9)
- signature: `class LHSFusedQuantizedMatmulTest(parameterized.TestCase):`
- members:
  - `generate_quantized_arrays(data_shape, scale_shape, dtype, *, key: jax.Array)` — [`L35`](../../../../../../../raw/code/qwix/tests/contrib/kernels/lhs_fused_quantized_matmul_test.py#L35)
  - `jax_answer(x: jax.Array, y: jax.Array, sy: jax.Array, *, sm: int)` — [`L12`](../../../../../../../raw/code/qwix/tests/contrib/kernels/lhs_fused_quantized_matmul_test.py#L12)
  - `test_kernel_dot_general(self, m, k, n, sm, sk, sn, bm, bk, bn, dtype)` — [`L51`](../../../../../../../raw/code/qwix/tests/contrib/kernels/lhs_fused_quantized_matmul_test.py#L51)
- uses (calls/refs, reference-scoped): [`lhs_fused_quantized_matmul`](../../../qwix/contrib/kernels/lhs_fused_quantized_matmul.md#lhs_fused_quantized_matmul), [`quantize_a_tile`](../../../qwix/contrib/kernels/lhs_fused_quantized_matmul.md#quantize_a_tile)

