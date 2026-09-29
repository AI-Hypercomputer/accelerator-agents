---
title: 'Module: aqt/jax/v2/aqt_quantizer.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/aqt_quantizer.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.aqt_quantizer`/
symbols:
  Quantizer.calibrate: Quantizer#calibrate().
  Quantizer.quant: Quantizer#quant().
  quantizer_make: quantizer_make().
  Quantizer.calculate_qvalue: Quantizer#calculate_qvalue().
  Quantizer.numerics: Quantizer#numerics.
  Quantizer._calibrator: Quantizer#_calibrator.
  Quantizer.calibration: Quantizer#calibration.
  Quantizer.init_calibration: Quantizer#init_calibration().
  TilingState: TilingState.
  Quantizer: Quantizer#
  Quantizer.calib_shared_axes: Quantizer#calib_shared_axes.
  make_fake_quant: make_fake_quant().
  AbstractAqtCalibration: AbstractAqtCalibration.
  Quantizer.context: Quantizer#context.
  AbstractAqtNumerics: AbstractAqtNumerics.
  Quantizer.scale_stop_grad: Quantizer#scale_stop_grad.
  Quantizer.scale_dtype: Quantizer#scale_dtype.
  AxisTiling: AxisTiling.
  make_fake_quant.fake_quant: make_fake_quant().fake_quant().
---
# Module: [`aqt/jax/v2/aqt_quantizer.py`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_quantizer.py)

## Classes
### `Quantizer`
- def: [`aqt/jax/v2/aqt_quantizer.py:38`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_quantizer.py#L38) — documented in [aqt-jax-v2-aqt_quantizer](../../../../concepts/aqt-jax-v2-aqt_quantizer.md)
- doc: Configuration of quantization of one tensor.
- signature: `class Quantizer:`
- members:
  - `calculate_qvalue(self, x, qt: aqt_tensor.QTensor)` — [`L159`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_quantizer.py#L159) — Uses the quantization parameters in qt to quantize x. — documented in [aqt-jax-v2-aqt_quantizer](../../../../concepts/aqt-jax-v2-aqt_quantizer.md)
  - `calibrate(self, x, *, calibration_axes: None | Sequence[utils.AxisIdx], tiling_state: None | TilingState = None)` — [`L79`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_quantizer.py#L79) — Creates incomplete QTensor with only quantization parameters. — documented in [aqt-jax-v2-aqt_quantizer](../../../../concepts/aqt-jax-v2-aqt_quantizer.md)
  - `init_calibration(self)` — [`L58`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_quantizer.py#L58)
  - `quant(self, x, *, calibration_axes: None | Sequence[utils.AxisIdx], tiling_state: None | TilingState = None)` — [`L65`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_quantizer.py#L65) — The core quantizing function. — documented in [aqt-jax-v2-aqt_quantizer](../../../../concepts/aqt-jax-v2-aqt_quantizer.md)
  - `calib_shared_axes` — [`L42`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_quantizer.py#L42) — documented in [aqt-jax-v2-aqt_quantizer](../../../../concepts/aqt-jax-v2-aqt_quantizer.md)
  - `calibration` — [`L49`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_quantizer.py#L49) — documented in [aqt-jax-v2-aqt_quantizer](../../../../concepts/aqt-jax-v2-aqt_quantizer.md)
  - `context` — [`L54`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_quantizer.py#L54) — documented in [aqt-jax-v2-aqt_quantizer](../../../../concepts/aqt-jax-v2-aqt_quantizer.md)
  - `numerics` — [`L41`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_quantizer.py#L41) — documented in [aqt-jax-v2-utils](../../../../concepts/aqt-jax-v2-utils.md)
  - `scale_dtype` — [`L46`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_quantizer.py#L46) — documented in [aqt-jax-v2-utils](../../../../concepts/aqt-jax-v2-utils.md)
  - `scale_stop_grad` — [`L45`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_quantizer.py#L45) — documented in [aqt-jax-v2-utils](../../../../concepts/aqt-jax-v2-utils.md)
- protocol/private: `_calibrator`[`L52`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_quantizer.py#L52)
- uses (calls/refs, reference-scoped): [`QTensor`](aqt_tensor.md#QTensor), [`static_field`](utils.md#static_field), [`GradientFn`](aqt_tensor.md#GradientFn), [`AxisIdx`](utils.md#AxisIdx), [`Context`](utils.md#Context), [`TilingState`](aqt_quantizer.md#TilingState), [`AbstractAqtCalibration`](aqt_quantizer.md#AbstractAqtCalibration), [`AbstractAqtNumerics`](aqt_quantizer.md#AbstractAqtNumerics), [`NoNumerics`](numerics/no_numerics.md#NoNumerics), [`apply`](tiled_dot_general.md#TilingState.apply), [`to_tiled_axes_transposed`](tiled_dot_general.md#TilingState.to_tiled_axes_transposed)
- used by: [`quantizer_make`](aqt_quantizer.md#quantizer_make), [`lhs`](../aqt_dot_general.md#DefaultDotGeneralQuantizer.lhs), [`rhs`](../aqt_dot_general.md#DefaultDotGeneralQuantizer.rhs), [`lhs_mid`](../aqt_dot_general.md#DefaultDotGeneralQuantizer.lhs_mid), [`rhs_mid`](../aqt_dot_general.md#DefaultDotGeneralQuantizer.rhs_mid), [`make_fake_quant`](aqt_quantizer.md#make_fake_quant)

## Functions
- `fake_quant(x)` — [`L210`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_quantizer.py#L210)
- `make_fake_quant(quantizer: Quantizer, calibration_axes=None)` — [`L209`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_quantizer.py#L209)
- `quantizer_make(n_bits: None | int | fp8_numerics.FP8Dtype, preserve_max_val: bool = False, initialize_calibration: bool = True, scale_stop_grad: bool = True, scale_dtype: None | jnp.dtype = None)` — [`L179`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_quantizer.py#L179) — Makes Quantizer. — documented in [aqt-jax-v2-aqt_quantizer](../../../../concepts/aqt-jax-v2-aqt_quantizer.md)

## Module values
- `AbstractAqtCalibration` — [`L31`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_quantizer.py#L31) — documented in [aqt-jax-v2-aqt_quantizer](../../../../concepts/aqt-jax-v2-aqt_quantizer.md)
- `AbstractAqtNumerics` — [`L30`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_quantizer.py#L30) — documented in [aqt-jax-v2-aqt_quantizer](../../../../concepts/aqt-jax-v2-aqt_quantizer.md)
- `AxisTiling` — [`L33`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_quantizer.py#L33)
- `TilingState` — [`L34`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_quantizer.py#L34) — documented in [aqt-jax-v2-aqt_quantizer](../../../../concepts/aqt-jax-v2-aqt_quantizer.md)

