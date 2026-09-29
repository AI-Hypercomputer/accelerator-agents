---
title: 'Module: aqt/jax/v2/aqt_dot_general_test.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/aqt_dot_general_test.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.aqt_dot_general_test`/
symbols:
  _apply_po2_scale: _apply_po2_scale().
  test_jaxpr_dtype: test_jaxpr_dtype().
  rand_unif: rand_unif().
  test_eq: test_eq().
  _check_result_eq: _check_result_eq().
  fqt_param_dict: fqt_param_dict().
  _TrickyNumerics: _TrickyNumerics#
  _TrickyNumerics.dtype: _TrickyNumerics#dtype.
  _TrickyNumerics.get_dtype: _TrickyNumerics#get_dtype().
  _TrickyNumerics.get_quant_bound: _TrickyNumerics#get_quant_bound().
  _TrickyNumerics.fwd: _TrickyNumerics#fwd().
  _TrickyNumerics.vjp_fwd: _TrickyNumerics#vjp_fwd().
  _TrickyNumerics.vjp_bwd: _TrickyNumerics#vjp_bwd().
  _modify_dg: _modify_dg().
  _aqt_dg_full_lr_diff: _aqt_dg_full_lr_diff().
  _aqt_dg_full: _aqt_dg_full().
  _aqt_dg_raw_lr_diff: _aqt_dg_raw_lr_diff().
  _aqt_dg_raw: _aqt_dg_raw().
  AqtDotGeneralResearchTest: AqtDotGeneralResearchTest#
  AqtDotGeneralResearchTest.test_empty: AqtDotGeneralResearchTest#test_empty().
  AqtDotGeneralResearchTest.test_fq_noise: AqtDotGeneralResearchTest#test_fq_noise().
  AqtDotGeneralResearchTest.test_stochastic_rounding_noise: AqtDotGeneralResearchTest#test_stochastic_rounding_noise().
  AqtDotGeneralResearchTest.test_fake_quant: AqtDotGeneralResearchTest#test_fake_quant().
  AqtDotGeneralResearchTest.test_dot_general_calibration_with_contracting_axis: AqtDotGeneralResearchTest#test_dot_general_calibration_with_contracting_axis().
  AqtDotGeneralResearchTest.test_dot_general_calibration_with_remaining_axis: AqtDotGeneralResearchTest#test_dot_general_calibration_with_remaining_axis().
  AqtDotGeneralResearchTest.test_dot_general_calibrate_dequant_mode_mismatch: AqtDotGeneralResearchTest#test_dot_general_calibrate_dequant_mode_mismatch().
  AqtDotGeneralResearchTest.test_dot_general_prevent_fwd_quant_with_remaining_axis: AqtDotGeneralResearchTest#test_dot_general_prevent_fwd_quant_with_remaining_axis().
  AqtDotGeneralResearchTest.test_dot_general_equality_between_different_calibration_axes: AqtDotGeneralResearchTest#test_dot_general_equality_between_different_calibration_axes().
  AqtDotGeneralResearchTest.test_dynamic_context: AqtDotGeneralResearchTest#test_dynamic_context().
  AqtDotGeneralResearchTest.test_hardware_int8: AqtDotGeneralResearchTest#test_hardware_int8().
  AqtDotGeneralResearchTest.test_local_aqt: AqtDotGeneralResearchTest#test_local_aqt().
  AqtDotGeneralResearchTest.test_per_tensor: AqtDotGeneralResearchTest#test_per_tensor().
  AqtDotGeneralResearchTest.test_per_subchannel: AqtDotGeneralResearchTest#test_per_subchannel().
  AqtDotGeneralResearchTest.test_mid_quantization: AqtDotGeneralResearchTest#test_mid_quantization().
---
# Module: [`aqt/jax/v2/aqt_dot_general_test.py`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py)

## Classes
### `AqtDotGeneralResearchTest`
- def: [`aqt/jax/v2/aqt_dot_general_test.py:409`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L409)
- signature: `class AqtDotGeneralResearchTest(parameterized.TestCase):`
- members:
  - `test_dot_general_calibrate_dequant_mode_mismatch(self)` — [`L905`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L905)
  - `test_dot_general_calibration_with_contracting_axis(self, dg: aqt.DotGeneral, lhs_maxval=10.0, rhs_maxval=20.0, gra_maxval=30.0, dims=(((1,), (0,)), ((), ())), lhs_shape=(10, 20), rhs_shape=(20, 30), gra_shape=(10, 30), seed=0, dtype=jnp.float32, clip_gradient=False)` — [`L557`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L557)
  - `test_dot_general_calibration_with_remaining_axis(self, dg: config.DotGeneral | Callable[[], config.DotGeneral], lhs_maxval=10.0, rhs_maxval=20.0, gra_maxval=30.0, dims=(((1,), (0,)), ((), ())), lhs_shape=(10, 20), rhs_shape=(20, 30), gra_shape=(10, 30), seed=0, dtype=jnp.float32, clip_gradient=False)` — [`L761`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L761)
  - `test_dot_general_equality_between_different_calibration_axes(self, dg: config.DotGeneral)` — [`L972`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L972) — Check equality between different calibration axes.
  - `test_dot_general_prevent_fwd_quant_with_remaining_axis(self)` — [`L939`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L939) — If calibration axis is remaining_axis, use_fwd_quant should be False.
  - `test_dynamic_context(self)` — [`L1050`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L1050)
  - `test_empty(self)` — [`L411`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L411)
  - `test_fake_quant(self, bits=4, maxval=10.0, shape=(20, 1))` — [`L465`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L465)
  - `test_fq_noise(self, preserve_zero, prec, v, seed)` — [`L421`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L421)
  - `test_hardware_int8(self, seed=0)` — [`L1060`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L1060)
  - `test_local_aqt(self, shard_count, lhs, expected_product)` — [`L1091`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L1091)
  - `test_mid_quantization(self)` — [`L1169`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L1169)
  - `test_per_subchannel(self)` — [`L1146`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L1146)
  - `test_per_tensor(self)` — [`L1115`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L1115)
  - `test_stochastic_rounding_noise(self)` — [`L440`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L440)

### `_TrickyNumerics`
- def: [`aqt/jax/v2/aqt_dot_general_test.py:192`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L192)
- signature: `class _TrickyNumerics(numerics.AqtNumerics):`
- members:
  - `fwd(self, x, context)` — [`L202`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L202)
  - `get_dtype(self)` — [`L196`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L196)
  - `get_quant_bound(self)` — [`L199`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L199)
  - `vjp_bwd(self, res, grad)` — [`L210`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L210)
  - `vjp_fwd(self, x, context)` — [`L206`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L206)
  - `dtype` — [`L194`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L194)

## Functions
- `_apply_po2_scale(quantizer)` — [`L37`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L37)
- `_aqt_dg_full(dequant_mode: aqt.DequantMode, calibration_mode: aqt.CalibrationMode=aqt.CalibrationMode.CONTRACTING_AXIS, use_fwd_quant: None | bool=None, disable_rounding: bool=False, fwd_lhs_tricky_clip_and_round: bool=False, local_aqt: None | aqt.LocalAqt=None, use_mid_quant: bool=False, *, readonly_dg: aqt.DotGeneral, dims: jax.lax.DotDimensionNumbers, clip_gradient: bool=False)` — [`L342`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L342)
- `_aqt_dg_full_lr_diff(lhs_dequant_mode: aqt.DequantMode, rhs_dequant_mode: aqt.DequantMode, lhs_calibration_mode: aqt.CalibrationMode=aqt.CalibrationMode.CONTRACTING_AXIS, rhs_calibration_mode: aqt.CalibrationMode=aqt.CalibrationMode.CONTRACTING_AXIS, use_fwd_quant: None | bool=None, use_mid_quant: bool=False, disable_rounding: bool=False, fwd_lhs_tricky_clip_and_round: bool=False, local_aqt: None | aqt.LocalAqt=None, *, readonly_dg: aqt.DotGeneral, dims: jax.lax.DotDimensionNumbers, clip_gradient: bool=False)` — [`L310`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L310)
- `_aqt_dg_raw(dequant_mode: aqt.DequantMode, calibration_mode: aqt.CalibrationMode=aqt.CalibrationMode.CONTRACTING_AXIS, *, readonly_dg: aqt.DotGeneral, dims: jax.lax.DotDimensionNumbers)` — [`L392`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L392)
- `_aqt_dg_raw_lr_diff(lhs_dequant_mode: aqt.DequantMode, rhs_dequant_mode: aqt.DequantMode, lhs_calibration_mode: aqt.CalibrationMode=aqt.CalibrationMode.CONTRACTING_AXIS, rhs_calibration_mode: aqt.CalibrationMode=aqt.CalibrationMode.CONTRACTING_AXIS, *, readonly_dg: aqt.DotGeneral, dims: jax.lax.DotDimensionNumbers)` — [`L371`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L371)
- `_check_result_eq(dgs, *, lhs, rhs, gra)` — [`L146`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L146)
- `_modify_dg(readonly_dg: aqt.DotGeneral, *, lhs_dequant_mode: aqt.DequantMode=aqt.DequantMode.OUTPUT, rhs_dequant_mode: aqt.DequantMode=aqt.DequantMode.OUTPUT, lhs_calibration_mode: aqt.CalibrationMode=aqt.CalibrationMode.CONTRACTING_AXIS, rhs_calibration_mode: aqt.CalibrationMode=aqt.CalibrationMode.CONTRACTING_AXIS, use_fwd_quant: None | bool=None, disable_rounding: bool=False, fwd_lhs_tricky_clip_and_round: bool=False, local_aqt: None | aqt.LocalAqt=None, use_mid_quant: bool=False, clip_gradient: bool=False)` — [`L216`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L216)
- `fqt_param_dict(s, use_fwd_quant, **kwargs)` — [`L180`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L180)
- `rand_unif(shape, maxval, seed, dtype=jnp.float32)` — [`L97`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L97)
- `test_eq(name, a, b)` — [`L119`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L119)
- `test_jaxpr_dtype(f, dg_raws: list[aqt.DotGeneralRaw], float_dtype)` — [`L54`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_dot_general_test.py#L54) — Tests whether dot_generals in f conform to dtypes inside of dg_raws.

