---
title: 'Module: aqt/jax/v2/config.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/config.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.config`/
symbols:
  SKIP: SKIP.
  SkipT: SkipT.
  _split_key: _split_key().
  set_context: set_context().
  set_fwd_dequant_mode: set_fwd_dequant_mode().
  set_fwd_calibration_mode: set_fwd_calibration_mode().
  set_numerics: set_numerics().
  set_fwd_rhs_dtype_int2: set_fwd_rhs_dtype_int2().
  set_accumulator_dtype: set_accumulator_dtype().
  set_stochastic_rounding: set_stochastic_rounding().
  set_constant_calibration: set_constant_calibration().
  set_local_aqt: set_local_aqt().
  set_use_fwd_quant: set_use_fwd_quant().
  set_use_mid_quant: set_use_mid_quant().
  set_int_numerics_preserve_zero: set_int_numerics_preserve_zero().
  set_auto_calib_scale: set_auto_calib_scale().
  set_absmax_calib_scale: set_absmax_calib_scale().
  set_bits: set_bits().
  set_scale_and_bias_dtype: set_scale_and_bias_dtype().
  default_unquantized_config: default_unquantized_config().
  fully_quantized: fully_quantized().
  config_v3: config_v3().
  config_v4: config_v4().
  config_fwd_fp8: config_fwd_fp8().
  set_fwd_calibration: set_fwd_calibration().
---
# Module: [`aqt/jax/v2/config.py`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py)

## Functions
- `_split_key(key: None | jax.Array, num_splits: int)` — [`L61`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L61)
- `config_fwd_fp8(fwd_bits: fp8_numerics.FP8Dtype='e4m3')` — [`L791`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L791) — Configs for FP8 forward pass.
- `config_v3(*, fwd_bits: None | int=8, dlhs_bits: None | int=8, drhs_bits: None | int=None, use_dummy_static_bound: bool=False, rng_type: str='jax.uniform', dlhs_local_aqt: None | LocalAqt=None, drhs_local_aqt: None | LocalAqt=None, fwd_accumulator_dtype=jnp.int32, dlhs_accumulator_dtype=jnp.int32, drhs_accumulator_dtype=None)` — [`L661`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L661) — Fully Quantized Training.
- `config_v4(*, fwd_bits: None | int | fp8_numerics.FP8Dtype=8, dlhs_bits: None | int | fp8_numerics.FP8Dtype=8, drhs_bits: None | int | fp8_numerics.FP8Dtype=None, use_dummy_static_bound: bool=False, rng_type: str='jax.uniform', dlhs_local_aqt: None | LocalAqt=None, drhs_local_aqt: None | LocalAqt=None, fwd_accumulator_dtype: None | jnp.dtype | SkipT=SKIP, dlhs_accumulator_dtype: None | jnp.dtype | SkipT=SKIP, drhs_accumulator_dtype: None | jnp.dtype | SkipT=SKIP, dlhs_use_fwd_quant: None | bool | SkipT=SKIP, drhs_use_fwd_quant: None | bool | SkipT=SKIP, fwd_mid_alpha_both: SkipT | float=SKIP, dlhs_mid_alpha_both: SkipT | float=SKIP, drhs_mid_alpha_both: SkipT | float=SKIP)` — [`L724`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L724) — Version 4 of user-visible AQT config.
- `default_unquantized_config()` — [`L555`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L555) — Aqt config for floating-point dot general.
- `fully_quantized(*, fwd_bits: None | int=8, bwd_bits: None | int=8, use_fwd_quant: bool=True, use_stochastic_rounding: None | bool=True, vjp_lhs_stochastic_rounding: None | bool=None, vjp_rhs_stochastic_rounding: None | bool=None, use_dummy_static_bound: bool=False, dlhs_local_aqt: None | LocalAqt=None, drhs_local_aqt: None | LocalAqt=None)` — [`L598`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L598) — Fully Quantized Training.
- `set_absmax_calib_scale(cfg: DotGeneral, scale: float)` — [`L413`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L413) — Set clipping_scale and clip_gradient for AbsMaxCalibration quantizers.
- `set_accumulator_dtype(cfg: DotGeneral, fwd_dtype: None | jnp.dtype | SkipT, dlhs_dtype: None | jnp.dtype | SkipT, drhs_dtype: None | jnp.dtype | SkipT)` — [`L157`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L157)
- `set_auto_calib_scale(cfg: DotGeneral, auto_clip_search_config: utils.AutoScaleSearchConfig)` — [`L375`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L375) — Update `cfg`'s quantizers' calibration to use auto clipping search.
- `set_bits(cfg: DotGeneral, fwd_lhs_bit: None | int | fp8_numerics.FP8Dtype, fwd_rhs_bit: None | int | fp8_numerics.FP8Dtype, dlhs_lhs_bit: None | int | fp8_numerics.FP8Dtype, dlhs_rhs_bit: None | int | fp8_numerics.FP8Dtype, drhs_lhs_bit: None | int | fp8_numerics.FP8Dtype, drhs_rhs_bit: None | int | fp8_numerics.FP8Dtype)` — [`L463`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L463) — Set quant bits for dot_general. Overwrites with AbsMaxCalibration.
- `set_constant_calibration(cfg: DotGeneral, bound: float=1.0, bias: None | float=None)` — [`L228`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L228) — Sets the static bound for calibration.
- `set_context(cfg: DotGeneral, key: None | jax.Array, train_step: None | int, lhs_quant_mode: utils.QuantMode=utils.QuantMode.TRAIN, rhs_quant_mode: utils.QuantMode=utils.QuantMode.TRAIN)` — [`L66`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L66) — Set context with prng keys and train_steps for dot_general config.
- `set_fwd_calibration(cfg: DotGeneral, calibration_factory)` — [`L811`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L811) — Updates aqt_cfg for static range calibration.
- `set_fwd_calibration_mode(cfg: DotGeneral, *, lhs_calibration_mode: CalibrationMode | SkipT=SKIP, rhs_calibration_mode: CalibrationMode | SkipT=SKIP)` — [`L105`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L105)
- `set_fwd_dequant_mode(cfg: DotGeneral, *, lhs_dequant_mode: None | DequantMode=None, rhs_dequant_mode: None | DequantMode=None)` — [`L93`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L93)
- `set_fwd_rhs_dtype_int2(cfg: DotGeneral)` — [`L142`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L142) — A special setter for int2 weights.
- `set_int_numerics_preserve_zero(cfg: DotGeneral, preserve_zero: bool)` — [`L350`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L350) — Set preserve_zero for int_numerics.
- `set_local_aqt(cfg: DotGeneral, fwd_local_aqt: None | SkipT | LocalAqt, dlhs_local_aqt: None | SkipT | LocalAqt, drhs_local_aqt: None | SkipT | LocalAqt)` — [`L254`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L254)
- `set_numerics(cfg: DotGeneralRaw, lhs_numerics: numerics.AqtNumerics, rhs_numerics: numerics.AqtNumerics)` — [`L117`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L117) — Set numerics for DotGeneralRaw config.
- `set_scale_and_bias_dtype(cfg: DotGeneral, dtype: jnp.dtype)` — [`L515`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L515) — Set the dtype for all scales and biases in the given DotGeneral config.
- `set_stochastic_rounding(cfg: DotGeneral, vjp_lhs_stochastic_rounding: bool, vjp_rhs_stochastic_rounding: bool, implementation: str, noise_sharing_axes: Sequence[int]=())` — [`L171`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L171) — Configure stochastic rounding implementation.
- `set_use_fwd_quant(cfg: DotGeneral, dlhs_use_fwd_quant: None | bool | SkipT, drhs_use_fwd_quant: None | bool | SkipT)` — [`L268`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L268) — Enable resusing of fwd pass quantization for backprop.
- `set_use_mid_quant(cfg: DotGeneral, fwd_mid_alpha_both: SkipT | float, dlhs_mid_alpha_both: SkipT | float, drhs_mid_alpha_both: SkipT | float)` — [`L287`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L287) — Enable middle quantization. Variant of SmoothQuant / AWQ.

## Module values
- `SKIP` — [`L57`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L57)
- `SkipT` — [`L58`](../../../../../../../raw/code/aqt/aqt/jax/v2/config.py#L58)

