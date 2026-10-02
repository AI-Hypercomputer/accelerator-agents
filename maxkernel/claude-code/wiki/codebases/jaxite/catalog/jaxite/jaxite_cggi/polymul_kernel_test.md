---
title: 'Module: jaxite/jaxite_cggi/polymul_kernel_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/polymul_kernel_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.polymul_kernel_test`/
symbols:
  PolymulKernelTest.test_i32_matmul_vs_reference: PolymulKernelTest#test_i32_matmul_vs_reference().
  PolymulKernelTest.test_many_seeds: PolymulKernelTest#test_many_seeds().
  PolymulKernelTest.test_vector_matrix_vs_reference: PolymulKernelTest#test_vector_matrix_vs_reference().
  random: random().
  _SEEDS: _SEEDS.
  PolymulKernelTest: PolymulKernelTest#
---
# Module: [`jaxite/jaxite_cggi/polymul_kernel_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/polymul_kernel_test.py)

## Classes
### `PolymulKernelTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_cggi/polymul_kernel_test.py:17`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/polymul_kernel_test.py#L17)
- signature: `class PolymulKernelTest(parameterized.TestCase):`
- members:
  - `test_i32_matmul_vs_reference(self, seed: int)` — [`L20`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/polymul_kernel_test.py#L20)
  - `test_many_seeds(self, seed: int)` — [`L38`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/polymul_kernel_test.py#L38)
  - `test_vector_matrix_vs_reference(self)` — [`L28`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/polymul_kernel_test.py#L28)
- uses (calls/refs, reference-scoped): [`negacyclic_vector_matrix_polymul`](polymul_kernel.md#negacyclic_vector_matrix_polymul), [`fallback_vector_matrix_polymul`](polymul_kernel.md#fallback_vector_matrix_polymul), [`random`](polymul_kernel_test.md#random), [`fallback_i32_matmul`](polymul_kernel.md#fallback_i32_matmul), [`i32_matmul`](polymul_kernel.md#i32_matmul), [`_SEEDS`](polymul_kernel_test.md#_SEEDS)

## Functions
- `random(shape, dtype=np.int32, high=2 ** 31 - 1)` — [`L11`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/polymul_kernel_test.py#L11)

## Module values
- `_SEEDS` — [`L8`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/polymul_kernel_test.py#L8)

