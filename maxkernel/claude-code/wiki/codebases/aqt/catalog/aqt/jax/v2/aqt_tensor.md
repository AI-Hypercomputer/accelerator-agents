---
title: 'Module: aqt/jax/v2/aqt_tensor.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/aqt_tensor.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.aqt_tensor`/
symbols:
  QTensor: QTensor#
  GradientFn: GradientFn.
  _MSG_NO_QVALUE: _MSG_NO_QVALUE.
  TilingState: TilingState.
  QTensor.qvalue: QTensor#qvalue.
  QTensor.sparsity_mask: QTensor#sparsity_mask.
  QTensor.scale: QTensor#scale.
  QTensor.scale_t: QTensor#scale_t.
  QTensor.bias: QTensor#bias.
  QTensor.dequant_dtype: QTensor#dequant_dtype.
  QTensor.tiling_state: QTensor#tiling_state.
  QTensor.dtype: QTensor#dtype().
  QTensor._validate_tiling_state: QTensor#_validate_tiling_state().
  QTensor.is_full: QTensor#is_full().
  QTensor.without_qvalue: QTensor#without_qvalue().
  QTensor.astype: QTensor#astype().
  QTensor.quant: QTensor#quant().
  QTensor.dequant: QTensor#dequant().
  QTensor.qvalue_astype: QTensor#qvalue_astype().
  QTensor.__getitem__: QTensor#__getitem__().
  QTensor.ndim: QTensor#ndim().
  QTensor.shape: QTensor#shape().
  QTensor.__len__: QTensor#__len__().
  zeros: zeros().
  zeros_with_scale: zeros_with_scale().
  partition_spec: partition_spec().
  dynamic_slice: dynamic_slice().
  dynamic_update_slice: dynamic_update_slice().
  update_frame: update_frame().
---
# Module: [`aqt/jax/v2/aqt_tensor.py`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py)

## Classes
### `QTensor`
- def: [`aqt/jax/v2/aqt_tensor.py:55`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L55) — documented in [aqt-jax-aqt_dot_general](../../../../concepts/aqt-jax-aqt_dot_general.md)
- doc: Quantized tensor.
- signature: `class QTensor:`
- members:
  - `__getitem__(self, idx: jax_typing.ArrayLike)` — [`L206`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L206) — Returns the indexed subtensor on the first axis.
  - `astype(self, dtype: jnp.dtype)` — [`L122`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L122)
  - `dequant(self)` — [`L161`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L161) — Dequantizes the QTensor into a jax array.
  - `dtype(self)` — [`L97`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L97)
  - `is_full(self)` — [`L115`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L115)
  - `ndim(self)` — [`L223`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L223)
  - `quant(self, x)` — [`L125`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L125) — Quantizes x into a new QTensor.
  - `qvalue_astype(self, dtype)` — [`L202`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L202)
  - `shape(self)` — [`L231`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L231)
  - `without_qvalue(self)` — [`L118`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L118) — Returns a copy of the QTensor without the qvalue.
  - `bias` — [`L83`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L83)
  - `dequant_dtype` — [`L87`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L87)
  - `qvalue` — [`L60`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L60)
  - `scale` — [`L68`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L68)
  - `scale_t` — [`L77`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L77)
  - `sparsity_mask` — [`L62`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L62)
  - `tiling_state` — [`L92`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L92)
- protocol/private: `__len__`[`L238`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L238), `_validate_tiling_state`[`L104`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L104)
- used by: [`__call__`](../aqt_dot_general.md#DotGeneralRaw.__call__), [`calibrate`](../aqt_dot_general.md#DefaultDotGeneralQuantizer.calibrate), [`calibrate`](aqt_quantizer.md#Quantizer.calibrate), [`_qtensor_dot_general`](../aqt_dot_general.md#_qtensor_dot_general), [`dg_core`](../aqt_dot_general.md#DotGeneral.dg_core), [`quant`](aqt_quantizer.md#Quantizer.quant), [`calculate_qvalue`](aqt_quantizer.md#Quantizer.calculate_qvalue), [`quant`](../aqt_dot_general.md#quant), [`QTensor`](pallas/dot_general.md#QTensor), [`__call__`](../aqt_dot_general.md#DotGeneralQuantizer.__call__), [`_maybe_dequant`](../aqt_dot_general.md#_qtensor_dot_general._maybe_dequant), [`MultiTensor`](../aqt_dot_general.md#MultiTensor), [`calculate_qvalue`](../aqt_dot_general.md#DefaultDotGeneralQuantizer.calculate_qvalue), [`_dg_core`](../aqt_dot_general.md#_dg_core), [`_postprocess_qtensor`](../aqt_dot_general.md#quant._postprocess_qtensor), [`dg_core_vjp_fwd`](../aqt_dot_general.md#dg_core_vjp_fwd), [`calculate_qvalue`](../aqt_dot_general.md#DotGeneralQuantizer.calculate_qvalue), [`calibrate`](../aqt_dot_general.md#DotGeneralQuantizer.calibrate), [`QTensor`](pallas/pallas_call.md#QTensor), [`_get_scale_t`](../aqt_dot_general.md#_get_scale_t), [`QTensor`](pallas/quantizer.md#QTensor), [`my_conv_general_dilated`](../aqt_conv_general.md#make_conv_general_dilated_with_qt.my_conv_general_dilated)

## Functions
- `dynamic_slice(operand: QTensor, start_indices: Sequence[int], slice_sizes: Sequence[int])` — [`L315`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L315) — Dynamically slices the value at start_indices using the given shape.
- `dynamic_update_slice(operand: QTensor, update: QTensor, start_indices: Sequence[int])` — [`L356`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L356) — Updates the value at start_indices with the given QTensor value.
- `partition_spec(partitions: Sequence[Any], calibration_axis: Sequence[utils.AxisIdx], dtype: jnp.dtype, *, use_bias: bool)` — [`L283`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L283) — Returns a QTensor filled with partition specs.
- `update_frame(operand: QTensor, frame: int, update: QTensor)` — [`L405`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L405) — Updates the value at frame with the given QTensor value.
- `zeros(shape: Sequence[int], *, container_dtype: jnp.dtype, dequant_dtype: jnp.dtype=jnp.bfloat16)` — [`L243`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L243) — documented in [aqt-jax-v2-aqt_tensor](../../../../concepts/aqt-jax-v2-aqt_tensor.md)
- `zeros_with_scale(shape: Sequence[int], calibration_axis: Sequence[utils.AxisIdx], *, container_dtype: jnp.dtype, scale_dtype: None | jnp.dtype=None, dequant_dtype: jnp.dtype=jnp.bfloat16)` — [`L258`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L258) — Initializes a QTensor with empty qvalue along with empty scale value. — documented in [aqt-jax-v2-aqt_tensor](../../../../concepts/aqt-jax-v2-aqt_tensor.md)

## Module values
- `GradientFn` — [`L36`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L36) — documented in [aqt-jax-aqt_dot_general](../../../../concepts/aqt-jax-aqt_dot_general.md)
- `TilingState` — [`L51`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L51)
- `_MSG_NO_QVALUE` — [`L37`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_tensor.py#L37)

