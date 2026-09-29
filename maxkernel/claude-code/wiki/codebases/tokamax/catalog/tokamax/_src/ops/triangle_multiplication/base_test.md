---
title: 'Module: tokamax/_src/ops/triangle_multiplication/base_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/triangle_multiplication/base_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.triangle_multiplication.base_test`/
symbols:
  TriangleMultiplicationTest.test_triangle_multiplication: TriangleMultiplicationTest#test_triangle_multiplication().
  TriangleMultiplicationTest.test_triangle_multiplication_grad: TriangleMultiplicationTest#test_triangle_multiplication_grad().
  TriangleMultiplicationTest.f: TriangleMultiplicationTest#f().
  _get_params: _get_params().
  TriangleMultiplicationTest: TriangleMultiplicationTest#
---
# Module: [`tokamax/_src/ops/triangle_multiplication/base_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/triangle_multiplication/base_test.py)

## Classes
### `TriangleMultiplicationTest`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/triangle_multiplication/base_test.py:40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/triangle_multiplication/base_test.py#L40)
- signature: `class TriangleMultiplicationTest(parameterized.TestCase):`
- members:
  - `f(params)` — [`L53`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/triangle_multiplication/base_test.py#L53)
  - `test_triangle_multiplication(self, triangle_type, dtype, precision)` — [`L47`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/triangle_multiplication/base_test.py#L47)
  - `test_triangle_multiplication_grad(self, triangle_type, dtype, precision)` — [`L68`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/triangle_multiplication/base_test.py#L68)
- uses (calls/refs, reference-scoped): [`TriangleMultiplication`](base.md#TriangleMultiplication), [`_get_params`](base_test.md#_get_params)

## Functions
- `_get_params(n, c, h, d, dtype)` — [`L24`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/triangle_multiplication/base_test.py#L24)

