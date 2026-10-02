---
title: 'Module: tests/_src/core/qarray_test.py'
type: catalog
provenance: extracted
module: tests/_src/core/qarray_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.core.qarray_test`/QArrayTest#
symbols:
  QArrayTest.test_array_methods: test_array_methods().
  QArrayTest.test_quantize_dequantize: test_quantize_dequantize().
  QArrayTest.test_quantize_nan_reproduction: test_quantize_nan_reproduction().
  QArrayTest.test_exact_quantization: test_exact_quantization().
  QArrayTest.test_broadcast_to_with: test_broadcast_to_with().
  QArrayTest.test_sparsify: test_sparsify().
  QArrayTest.test_get_tiled_axes: test_get_tiled_axes().
  QArrayTest.unsafe_compute_scale_zero_point: unsafe_compute_scale_zero_point().
  QArrayTest.test_mxfp_tile_size_validation: test_mxfp_tile_size_validation().
  QArrayTest._make_array: _make_array().
  QArrayTest.test_clip_gradient_to_calibration: test_clip_gradient_to_calibration().
  QArrayTest: ''
---
# Module: [`tests/_src/core/qarray_test.py`](../../../../../../../raw/code/qwix/tests/_src/core/qarray_test.py)

## Classes
### `QArrayTest`  ·  implements/extends TestCase
- def: [`tests/_src/core/qarray_test.py:28`](../../../../../../../raw/code/qwix/tests/_src/core/qarray_test.py#L28)
- signature: `class QArrayTest(parameterized.TestCase):`
- members:
  - `test_array_methods(self)` — [`L228`](../../../../../../../raw/code/qwix/tests/_src/core/qarray_test.py#L228)
  - `test_broadcast_to_with(self, input_shape, target_shape, is_qarray)` — [`L364`](../../../../../../../raw/code/qwix/tests/_src/core/qarray_test.py#L364)
  - `test_clip_gradient_to_calibration(self)` — [`L304`](../../../../../../../raw/code/qwix/tests/_src/core/qarray_test.py#L304)
  - `test_exact_quantization(self, with_error)` — [`L191`](../../../../../../../raw/code/qwix/tests/_src/core/qarray_test.py#L191)
  - `test_get_tiled_axes(self)` — [`L221`](../../../../../../../raw/code/qwix/tests/_src/core/qarray_test.py#L221)
  - `test_mxfp_tile_size_validation(self)` — [`L464`](../../../../../../../raw/code/qwix/tests/_src/core/qarray_test.py#L464)
  - `test_quantize_dequantize(self, array_shape: tuple[int, ...], qtype: jax.typing.DTypeLike, channelwise_axes: Collection[int], tiled_axes: Mapping[int, int], calibration_method: str, expected_mae: float)` — [`L137`](../../../../../../../raw/code/qwix/tests/_src/core/qarray_test.py#L137)
  - `test_quantize_nan_reproduction(self)` — [`L397`](../../../../../../../raw/code/qwix/tests/_src/core/qarray_test.py#L397)
  - `test_sparsify(self)` — [`L387`](../../../../../../../raw/code/qwix/tests/_src/core/qarray_test.py#L387)
  - `unsafe_compute_scale_zero_point(calibration, qtype)` — [`L429`](../../../../../../../raw/code/qwix/tests/_src/core/qarray_test.py#L429)
- protocol/private: `_make_array`[`L30`](../../../../../../../raw/code/qwix/tests/_src/core/qarray_test.py#L30)
- uses (calls/refs, reference-scoped): [`QArray`](../../../qwix/_src/core/qarray.md#QArray), [`HowToQuantize`](../../../qwix/_src/core/qarray.md#HowToQuantize), [`qtype`](../../../qwix/_src/core/qarray.md#HowToQuantize.qtype), [`quantize`](../../../qwix/_src/core/qarray.md#quantize), [`qvalue`](../../../qwix/_src/core/qarray.md#QArray.qvalue), [`scale`](../../../qwix/_src/core/qarray.md#QArray.scale), [`dequantize`](../../../qwix/_src/core/qarray.md#dequantize), [`tiled_axes`](../../../qwix/_src/core/qarray.md#HowToQuantize.tiled_axes), [`channelwise_axes`](../../../qwix/_src/core/qarray.md#HowToQuantize.channelwise_axes), [`calibration_method`](../../../qwix/_src/core/qarray.md#HowToQuantize.calibration_method), [`zero_point`](../../../qwix/_src/core/qarray.md#QArray.zero_point), [`compute_scale_zero_point`](../../../qwix/_src/core/qarray.md#compute_scale_zero_point), [`convert_to`](../../../qwix/_src/core/numerics.md#convert_to), [`quantize_api`](../../../qwix/_src/core/qarray.md#quantize_api), [`sparsify`](../../../qwix/_src/core/qarray.md#sparsify), [`call_with_generic_broadcast`](../../../qwix/_src/core/qarray.md#call_with_generic_broadcast), [`shape`](../../../qwix/_src/core/qarray.md#QArray.shape), [`get_tiled_axes`](../../../qwix/_src/core/qarray.md#get_tiled_axes), [`validate_qarray`](../../../qwix/_src/core/qarray.md#validate_qarray), [`get_symmetric_bound`](../../../qwix/_src/core/numerics.md#get_symmetric_bound), [`SparsityRule`](../../../qwix/_src/core/sparsity.md#SparsityRule), [`reshape`](../../../qwix/_src/core/qarray.md#QArray.reshape), [`clip_gradient_to_calibration`](../../../qwix/_src/core/qarray.md#clip_gradient_to_calibration), [`broadcast_to`](../../../qwix/_src/core/qarray.md#broadcast_to), [`ndim`](../../../qwix/_src/core/qarray.md#QArray.ndim), [`weight_sparsity_m`](../../../qwix/_src/core/sparsity.md#SparsityRule.weight_sparsity_m), [`astype`](../../../qwix/_src/core/qarray.md#QArray.astype), [`weight_sparsity_n`](../../../qwix/_src/core/sparsity.md#SparsityRule.weight_sparsity_n), [`USE_RECIPROCAL_FOR_QUANTIZATION`](../../../qwix/_src/core/qarray.md#USE_RECIPROCAL_FOR_QUANTIZATION), [`scale_tile_shape`](../../../qwix/_src/core/qarray.md#QArray.scale_tile_shape), [`zero_point_tile_shape`](../../../qwix/_src/core/qarray.md#QArray.zero_point_tile_shape), [`swapaxes`](../../../qwix/_src/core/qarray.md#QArray.swapaxes), [`transpose`](../../../qwix/_src/core/qarray.md#QArray.transpose), [`T`](../../../qwix/_src/core/qarray.md#QArray.T), [`get_asymmetric_bound`](../../../qwix/_src/core/numerics.md#get_asymmetric_bound), [`weight_sparsity_order`](../../../qwix/_src/core/sparsity.md#SparsityRule.weight_sparsity_order), [`mT`](../../../qwix/_src/core/qarray.md#QArray.mT)

