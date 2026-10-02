---
title: 'Module: recurrentgemma/torch/layers_test.py'
type: catalog
provenance: extracted
module: recurrentgemma/torch/layers_test.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.torch.layers_test`/
symbols:
  RGLRUTest.test_numerically_to_jax: RGLRUTest#test_numerically_to_jax().
  EinsumTest.test_numerically_to_jax: EinsumTest#test_numerically_to_jax().
  RMSNormTest.test_numerically_to_jax: RMSNormTest#test_numerically_to_jax().
  BlockDiagonalLinearTest.test_numerically_to_jax: BlockDiagonalLinearTest#test_numerically_to_jax().
  Conv1DTest.test_numerically_to_jax: Conv1DTest#test_numerically_to_jax().
  RMSNormTest: RMSNormTest#
  BlockDiagonalLinearTest: BlockDiagonalLinearTest#
  RGLRUTest: RGLRUTest#
  Conv1DTest: Conv1DTest#
  EinsumTest: EinsumTest#
---
# Module: [`recurrentgemma/torch/layers_test.py`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers_test.py)

## Classes
### `BlockDiagonalLinearTest`  ·  implements/extends TestCase
- def: [`recurrentgemma/torch/layers_test.py:62`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers_test.py#L62)
- signature: `class BlockDiagonalLinearTest(parameterized.TestCase):`
- members:
  - `test_numerically_to_jax(self, width: int, num_blocks: int, dtype: str, seed: int)` — [`L70`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers_test.py#L70)
- uses (calls/refs, reference-scoped): [`num_blocks`](../jax/layers.md#BlockDiagonalLinear.num_blocks), [`param_dtype`](../jax/layers.md#BlockDiagonalLinear.param_dtype), [`width_input`](../jax/layers.md#BlockDiagonalLinear.width_input), [`BlockDiagonalLinear`](../jax/layers.md#BlockDiagonalLinear), [`BlockDiagonalLinear`](layers.md#BlockDiagonalLinear)  (1 test-only)

### `Conv1DTest`  ·  implements/extends TestCase
- def: [`recurrentgemma/torch/layers_test.py:136`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers_test.py#L136)
- signature: `class Conv1DTest(parameterized.TestCase):`
- members:
  - `test_numerically_to_jax(self, width: int, temporal_width: int, seq_len: int, dtype: str, seed: int, num_unroll_steps: int = 2)` — [`L145`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers_test.py#L145)
- uses (calls/refs, reference-scoped): [`Conv1D`](../jax/layers.md#Conv1D), [`Conv1D`](layers.md#Conv1D), [`temporal_width`](../jax/layers.md#Conv1D.temporal_width), [`param_dtype`](../jax/layers.md#Conv1D.param_dtype), [`width`](../jax/layers.md#Conv1D.width)  (1 test-only)

### `EinsumTest`  ·  implements/extends TestCase
- def: [`recurrentgemma/torch/layers_test.py:175`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers_test.py#L175)
- signature: `class EinsumTest(parameterized.TestCase):`
- members:
  - `test_numerically_to_jax(self, inputs_shape: tuple[int, ...], w_shape: tuple[int, ...], b_shape: tuple[int, ...], eqn: str, dtype: str, seed: int)` — [`L211`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers_test.py#L211)
- uses (calls/refs, reference-scoped): [`param_dtype`](../jax/layers.md#Einsum.param_dtype), [`b_shape`](../jax/layers.md#Einsum.b_shape), [`eqn`](../jax/layers.md#Einsum.eqn), [`w_shape`](../jax/layers.md#Einsum.w_shape), [`Einsum`](../jax/layers.md#Einsum), [`Einsum`](layers.md#Einsum)  (1 test-only)

### `RGLRUTest`  ·  implements/extends TestCase
- def: [`recurrentgemma/torch/layers_test.py:96`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers_test.py#L96)
- signature: `class RGLRUTest(parameterized.TestCase):`
- members:
  - `test_numerically_to_jax(self, width: int, num_heads: int, seq_len: int, dtype: str, seed: int, num_unroll_steps: int = 2)` — [`L105`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers_test.py#L105)
- uses (calls/refs, reference-scoped): [`RGLRU`](../jax/layers.md#RGLRU), [`ScanType`](../common.md#ScanType), [`RGLRU`](layers.md#RGLRU), [`scan_type`](../jax/layers.md#RGLRU.scan_type), [`param_dtype`](../jax/layers.md#RGLRU.param_dtype), [`width`](../jax/layers.md#RGLRU.width), [`LINEAR_NATIVE`](../common.md#ScanType.LINEAR_NATIVE), [`num_heads`](../jax/layers.md#RGLRU.num_heads)  (1 test-only)

### `RMSNormTest`  ·  implements/extends TestCase
- def: [`recurrentgemma/torch/layers_test.py:28`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers_test.py#L28)
- signature: `class RMSNormTest(parameterized.TestCase):`
- members:
  - `test_numerically_to_jax(self, width: int, eps: float, dtype: str, seed: int)` — [`L36`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/layers_test.py#L36)
- uses (calls/refs, reference-scoped): [`param_dtype`](../jax/layers.md#RMSNorm.param_dtype), [`width`](../jax/layers.md#RMSNorm.width), [`RMSNorm`](../jax/layers.md#RMSNorm), [`RMSNorm`](layers.md#RMSNorm), [`eps`](../jax/layers.md#RMSNorm.eps)  (1 test-only)

