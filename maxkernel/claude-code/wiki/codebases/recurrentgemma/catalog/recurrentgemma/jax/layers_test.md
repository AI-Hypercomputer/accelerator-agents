---
title: 'Module: recurrentgemma/jax/layers_test.py'
type: catalog
provenance: extracted
module: recurrentgemma/jax/layers_test.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.jax.layers_test`/LayersTest#
symbols:
  LayersTest.test_rglu: test_rglu().
  LayersTest.test_einsum: test_einsum().
  LayersTest.test_scan: test_scan().
  LayersTest.test_rmsnorm: test_rmsnorm().
  LayersTest: ''
---
# Module: [`recurrentgemma/jax/layers_test.py`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers_test.py)

## Classes
### `LayersTest`  ·  implements/extends TestCase
- def: [`recurrentgemma/jax/layers_test.py:26`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers_test.py#L26)
- signature: `class LayersTest(parameterized.TestCase):`
- members:
  - `test_einsum(self, inputs_shape: tuple[int, int], w_shape: tuple[int, int, int], b_shape: tuple[int, int], eqn: str, expected_shape: tuple[int, int])` — [`L44`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers_test.py#L44)
  - `test_rglu(self, only_real: bool)` — [`L119`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers_test.py#L119)
  - `test_rmsnorm(self, x: float, expected: float)` — [`L69`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers_test.py#L69)
  - `test_scan(self, seq_len: int, dtype: str, scan_type: common.ScanType)` — [`L84`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/layers_test.py#L84)
- uses (calls/refs, reference-scoped): [`RGLRU`](layers.md#RGLRU), [`ScanType`](../common.md#ScanType), [`linear_scan`](scan.md#linear_scan), [`only_real`](layers.md#RGLRU.only_real), [`dtype`](layers.md#RGLRU.dtype), [`width`](layers.md#RGLRU.width), [`LINEAR_NATIVE`](../common.md#ScanType.LINEAR_NATIVE), [`width`](layers.md#RMSNorm.width), [`RMSNorm`](layers.md#RMSNorm), [`num_heads`](layers.md#RGLRU.num_heads), [`b_shape`](layers.md#Einsum.b_shape), [`eqn`](layers.md#Einsum.eqn), [`w_shape`](layers.md#Einsum.w_shape), [`Einsum`](layers.md#Einsum), [`ASSOCIATIVE_NATIVE`](../common.md#ScanType.ASSOCIATIVE_NATIVE)

