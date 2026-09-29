---
title: 'Module: aqt/jax/v2/calibration.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/calibration.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.calibration`/
symbols:
  Calibration: Calibration#
  ConstantCalibration: ConstantCalibration#
  AbsMaxCalibration: AbsMaxCalibration#
  ceil_to_po2: ceil_to_po2().
  Calibration.dtype: Calibration#dtype.
  Calibration.po2_scale: Calibration#po2_scale.
  Calibration.get_scale_and_bias_and_sparsity: Calibration#get_scale_and_bias_and_sparsity().
  Calibration.init_calibration: Calibration#init_calibration().
  ConstantCalibration.bound: ConstantCalibration#bound.
  ConstantCalibration.bias: ConstantCalibration#bias.
  ConstantCalibration.get_scale_and_bias_and_sparsity: ConstantCalibration#get_scale_and_bias_and_sparsity().
  AbsMaxCalibration.clipping_scale: AbsMaxCalibration#clipping_scale.
  AbsMaxCalibration.get_scale_and_bias_and_sparsity: AbsMaxCalibration#get_scale_and_bias_and_sparsity().
  AbsMeanCalibration: AbsMeanCalibration#
  AbsMeanCalibration.clipping_scale: AbsMeanCalibration#clipping_scale.
  AbsMeanCalibration.p: AbsMeanCalibration#p.
  AbsMeanCalibration.get_scale_and_bias_and_sparsity: AbsMeanCalibration#get_scale_and_bias_and_sparsity().
  SnrBasedAutoCalibration: SnrBasedAutoCalibration#
  SnrBasedAutoCalibration.auto_clip_search_config: SnrBasedAutoCalibration#auto_clip_search_config.
  SnrBasedAutoCalibration.get_scale_and_bias_and_sparsity: SnrBasedAutoCalibration#get_scale_and_bias_and_sparsity().
  SnrBasedAutoCalibration._update_best_clip_scales_and_max_snr: SnrBasedAutoCalibration#_update_best_clip_scales_and_max_snr().
  SnrBasedAutoCalibration._calculate_snr: SnrBasedAutoCalibration#_calculate_snr().
---
# Module: [`aqt/jax/v2/calibration.py`](../../../../../../../raw/code/aqt/aqt/jax/v2/calibration.py)

## Classes
### `AbsMaxCalibration`
- def: [`aqt/jax/v2/calibration.py:107`](../../../../../../../raw/code/aqt/aqt/jax/v2/calibration.py#L107) — documented in [aqt-jax-v2-aqt_quantizer](../../../../concepts/aqt-jax-v2-aqt_quantizer.md)
- doc: Simple max(abs(x)) calibration.
- signature: `class AbsMaxCalibration(Calibration):`
- members:
  - `get_scale_and_bias_and_sparsity(self, x: jnp.ndarray, shared_axes: None | Sequence[utils.AxisIdx], numerics_: numerics.AqtNumerics, context: None | utils.Context=None)` — [`L117`](../../../../../../../raw/code/aqt/aqt/jax/v2/calibration.py#L117) — Calibration.
  - `clipping_scale` — [`L115`](../../../../../../../raw/code/aqt/aqt/jax/v2/calibration.py#L115)
- used by: [`quantizer_make`](aqt_quantizer.md#quantizer_make)

### `AbsMeanCalibration`
- def: [`aqt/jax/v2/calibration.py:161`](../../../../../../../raw/code/aqt/aqt/jax/v2/calibration.py#L161)
- doc: Simple clipping_scale * mean(abs(x) ** p) ** (1 / p) calibration.
- signature: `class AbsMeanCalibration(Calibration):`
- members:
  - `get_scale_and_bias_and_sparsity(self, x: jnp.ndarray, shared_axes: None | Sequence[utils.AxisIdx], numerics_: numerics.AqtNumerics, context: None | utils.Context=None)` — [`L173`](../../../../../../../raw/code/aqt/aqt/jax/v2/calibration.py#L173) — Calibration.
  - `clipping_scale` — [`L170`](../../../../../../../raw/code/aqt/aqt/jax/v2/calibration.py#L170)
  - `p` — [`L171`](../../../../../../../raw/code/aqt/aqt/jax/v2/calibration.py#L171)

### `Calibration`
- def: [`aqt/jax/v2/calibration.py:34`](../../../../../../../raw/code/aqt/aqt/jax/v2/calibration.py#L34) — documented in [aqt-jax-v2-aqt_quantizer](../../../../concepts/aqt-jax-v2-aqt_quantizer.md)
- doc: Abstract class for scale and bias calibration.
- signature: `class Calibration(abc.ABC):`
- members:
  - `get_scale_and_bias_and_sparsity(self, x: jnp.ndarray, shared_axes: None | Sequence[utils.AxisIdx], numerics_: numerics.AqtNumerics, context: None | utils.Context=None)` — [`L44`](../../../../../../../raw/code/aqt/aqt/jax/v2/calibration.py#L44) — Returns the quantizaiton scale and bias for the given input tensor.
  - `init_calibration(self)` — [`L63`](../../../../../../../raw/code/aqt/aqt/jax/v2/calibration.py#L63)
  - `dtype` — [`L39`](../../../../../../../raw/code/aqt/aqt/jax/v2/calibration.py#L39)
  - `po2_scale` — [`L41`](../../../../../../../raw/code/aqt/aqt/jax/v2/calibration.py#L41)
- used by: [`AbstractAqtCalibration`](aqt_quantizer.md#AbstractAqtCalibration)

### `ConstantCalibration`
- def: [`aqt/jax/v2/calibration.py:68`](../../../../../../../raw/code/aqt/aqt/jax/v2/calibration.py#L68)
- doc: Calibration with a constant per-tensor or per-channel value.
- signature: `class ConstantCalibration(Calibration):`
- members:
  - `get_scale_and_bias_and_sparsity(self, x: jnp.ndarray, shared_axes: None | Sequence[utils.AxisIdx], numerics_: numerics.AqtNumerics, context: None | utils.Context=None)` — [`L74`](../../../../../../../raw/code/aqt/aqt/jax/v2/calibration.py#L74)
  - `bias` — [`L72`](../../../../../../../raw/code/aqt/aqt/jax/v2/calibration.py#L72)
  - `bound` — [`L71`](../../../../../../../raw/code/aqt/aqt/jax/v2/calibration.py#L71)
- used by: [`quant`](pallas/quantizer.md#quant)

### `SnrBasedAutoCalibration`
- def: [`aqt/jax/v2/calibration.py:198`](../../../../../../../raw/code/aqt/aqt/jax/v2/calibration.py#L198)
- doc: Automatically finds the best clipping scales based on SNR values.
- signature: `class SnrBasedAutoCalibration(Calibration):`
- members:
  - `_calculate_snr(self, x: jnp.ndarray, bound: jnp.ndarray, shared_axes: Sequence[utils.AxisIdx], numerics_: numerics.AqtNumerics, context: utils.Context)` — [`L346`](../../../../../../../raw/code/aqt/aqt/jax/v2/calibration.py#L346) — Calculates the quantization signal-to-noise ratio (SNR) for the given bound.
  - `_update_best_clip_scales_and_max_snr(self, current_clip_scales: jnp.ndarray, current_snr_values: jnp.ndarray, clip_scale: float, x: jnp.ndarray, abs_max: jnp.ndarray, shared_axes: Sequence[utils.AxisIdx], numerics_: numerics.AqtNumerics, context: utils.Context)` — [`L285`](../../../../../../../raw/code/aqt/aqt/jax/v2/calibration.py#L285) — Updates the best clip scales and max SNR values given a `clip_scale`.
  - `get_scale_and_bias_and_sparsity(self, x: jnp.ndarray, shared_axes: None | Sequence[utils.AxisIdx], numerics_: numerics.AqtNumerics, context: None | utils.Context=None)` — [`L217`](../../../../../../../raw/code/aqt/aqt/jax/v2/calibration.py#L217) — Produces the scale for quantization based on SNR values.
  - `auto_clip_search_config` — [`L215`](../../../../../../../raw/code/aqt/aqt/jax/v2/calibration.py#L215)

## Functions
- `ceil_to_po2(scale: jnp.ndarray)` — [`L25`](../../../../../../../raw/code/aqt/aqt/jax/v2/calibration.py#L25)

