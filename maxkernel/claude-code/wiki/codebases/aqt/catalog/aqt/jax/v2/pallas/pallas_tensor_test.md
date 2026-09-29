---
title: 'Module: aqt/jax/v2/pallas/pallas_tensor_test.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/pallas/pallas_tensor_test.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.pallas.pallas_tensor_test`/
symbols:
  QTensor: QTensor.
  TransposedTensor: TransposedTensor.
  PallasTensorTest: PallasTensorTest#
  PallasTensorTest.test_qtensor_blockspec_correctness: PallasTensorTest#test_qtensor_blockspec_correctness().
  PallasTensorTest.test_transpose_for_memory_saving: PallasTensorTest#test_transpose_for_memory_saving().
---
# Module: [`aqt/jax/v2/pallas/pallas_tensor_test.py`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_tensor_test.py)

## Classes
### `PallasTensorTest`
- def: [`aqt/jax/v2/pallas/pallas_tensor_test.py:27`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_tensor_test.py#L27)
- signature: `class PallasTensorTest(parameterized.TestCase):`
- members:
  - `test_qtensor_blockspec_correctness(self, qvalue_shape, scale_shape, block_shape, expected_scale_block_shape, index_and_expected_index)` — [`L63`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_tensor_test.py#L63)
  - `test_transpose_for_memory_saving(self, tensor_shape, block_shape, expect_transpose, expected_permute_axes=None, expected_transposed_tensor_shape=None, expected_block_shape=None)` — [`L121`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_tensor_test.py#L121)

## Module values
- `QTensor` — [`L23`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_tensor_test.py#L23)
- `TransposedTensor` — [`L24`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_tensor_test.py#L24)

