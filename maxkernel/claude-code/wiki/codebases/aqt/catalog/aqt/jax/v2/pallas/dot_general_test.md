---
title: 'Module: aqt/jax/v2/pallas/dot_general_test.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/pallas/dot_general_test.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.pallas.dot_general_test`/
symbols:
  QTensor: QTensor.
  DequantMode: DequantMode.
  DotGeneralTest: DotGeneralTest#
  DotGeneralTest.test_quantized_matmul_error: DotGeneralTest#test_quantized_matmul_error().
  DotGeneralTest.test_dequantization_location: DotGeneralTest#test_dequantization_location().
---
# Module: [`aqt/jax/v2/pallas/dot_general_test.py`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/dot_general_test.py)

## Classes
### `DotGeneralTest`
- def: [`aqt/jax/v2/pallas/dot_general_test.py:31`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/dot_general_test.py#L31)
- signature: `class DotGeneralTest(parameterized.TestCase):`
- members:
  - `test_dequantization_location(self, lhs_shape=(2048, 512), rhs_shape=(512, 2048), lhs_calibration_axes=None, rhs_calibration_axes=None, dequant_mode_lhs=DequantMode.OUTPUT, dequant_mode_rhs=DequantMode.OUTPUT, dimension_numbers=(((1,), (0,)), ((), ())))` — [`L166`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/dot_general_test.py#L166)
  - `test_quantized_matmul_error(self, mkn_and_blk, quantize_lhs, quantize_rhs, quant_type)` — [`L45`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/dot_general_test.py#L45)

## Module values
- `DequantMode` — [`L28`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/dot_general_test.py#L28)
- `QTensor` — [`L27`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/dot_general_test.py#L27)

