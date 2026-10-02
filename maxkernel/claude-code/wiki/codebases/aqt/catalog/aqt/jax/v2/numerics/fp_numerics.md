---
title: 'Module: aqt/jax/v2/numerics/fp_numerics.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/numerics/fp_numerics.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.numerics.fp_numerics`/
symbols:
  fp_round: fp_round().
  FpNumericsConfig: FpNumericsConfig#
  e1m2: e1m2.
  e1m2_ocp: e1m2_ocp.
  e0m3_ocp: e0m3_ocp.
  e5m0_ocp: e5m0_ocp.
  e4m1_ocp: e4m1_ocp.
  e3m2_ocp: e3m2_ocp.
  e2m3_ocp: e2m3_ocp.
  e1m4_ocp: e1m4_ocp.
  e0m5_ocp: e0m5_ocp.
  e4m0_ocp: e4m0_ocp.
  e3m1_ocp: e3m1_ocp.
  e2m2_ocp: e2m2_ocp.
  e1m3_ocp: e1m3_ocp.
  e0m4_ocp: e0m4_ocp.
  e3m0: e3m0.
  e3m0_ocp: e3m0_ocp.
  e2m1: e2m1.
  e2m1_ocp: e2m1_ocp.
  e0m3: e0m3.
  e1m0: e1m0.
  fp_round_new: fp_round_new().
  fp_largest_representable: fp_largest_representable().
  FpNumerics: FpNumerics#
  float8_e4m3fn: float8_e4m3fn.
  float8_e5m2: float8_e5m2.
  float16: float16.
  RADIX4: RADIX4.
  float_repr: float_repr.
  FpNumericsConfig.nexp: FpNumericsConfig#nexp.
  FpNumericsConfig.minexp: FpNumericsConfig#minexp.
  FpNumericsConfig.nmant: FpNumericsConfig#nmant.
  FpNumericsConfig.has_subnormals: FpNumericsConfig#has_subnormals.
  FpNumericsConfig.has_two_nan: FpNumericsConfig#has_two_nan.
  FpNumericsConfig.has_naninf: FpNumericsConfig#has_naninf.
  FpNumericsConfig.radix: FpNumericsConfig#radix.
  radix2_round: radix2_round().
  FpNumerics.cfg: FpNumerics#cfg.
  FpNumerics.stochastic_rounding: FpNumerics#stochastic_rounding.
  FpNumerics.clip_gradient: FpNumerics#clip_gradient.
  FpNumerics.get_quant_bound: FpNumerics#get_quant_bound().
  FpNumerics.get_dtype: FpNumerics#get_dtype().
  FpNumerics.vjp_fwd: FpNumerics#vjp_fwd().
  FpNumerics.vjp_bwd: FpNumerics#vjp_bwd().
  radix4_round: radix4_round().
---
# Module: [`aqt/jax/v2/numerics/fp_numerics.py`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py)

## Classes
### `FpNumerics`
- def: [`aqt/jax/v2/numerics/fp_numerics.py:628`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L628)
- doc: Numerics for fp8.
- signature: `class FpNumerics(numerics.AqtNumerics):`
- members:
  - `get_dtype(self)` — [`L639`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L639)
  - `get_quant_bound(self)` — [`L636`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L636)
  - `vjp_bwd(self, res, grad)` — [`L652`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L652)
  - `vjp_fwd(self, x, context)` — [`L642`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L642)
  - `cfg` — [`L632`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L632)
  - `clip_gradient` — [`L634`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L634)
  - `stochastic_rounding` — [`L633`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L633)
- used by: `test_e1m2_vs_e0m3`

### `FpNumericsConfig`
- def: [`aqt/jax/v2/numerics/fp_numerics.py:26`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L26) — documented in [aqt-jax-v2-numerics-fp_numerics](../../../../../concepts/aqt-jax-v2-numerics-fp_numerics.md)
- signature: `class FpNumericsConfig:`
- members:
  - `has_naninf` — [`L32`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L32)
  - `has_subnormals` — [`L30`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L30)
  - `has_two_nan` — [`L31`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L31)
  - `minexp` — [`L28`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L28)
  - `nexp` — [`L27`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L27)
  - `nmant` — [`L29`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L29)
  - `radix` — [`L33`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L33)
- used by: `test_fp_round_old_vs_new`, `test_fp_round`, `fp_values`, `test_fp_some_fp_values`

## Functions
- `fp_largest_representable(cfg: FpNumericsConfig)` — [`L604`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L604) — Compute maximum representable FP number.
- `fp_round(x, *, cfg: FpNumericsConfig, key: jax.Array, stochastic_rounding: bool, test_noise_axis=None)` — [`L288`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L288) — Round function dispatcher. — documented in [aqt-jax-v2-numerics-fp_numerics](../../../../../concepts/aqt-jax-v2-numerics-fp_numerics.md)
- `fp_round_new(x, *, cfg: FpNumericsConfig, key: jax.Array, stochastic_rounding: bool, test_noise_axis=None)` — [`L474`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L474) — FP stochastic rounding for a given mantissa and exponent. Returns bf16.
- `radix2_round(x, *, cfg: FpNumericsConfig, key: jax.Array, stochastic_rounding: bool, test_noise_axis=None)` — [`L309`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L309) — FP stochastic rounding for a given mantissa and exponent. Returns bf16.
- `radix4_round(x, *, cfg: FpNumericsConfig, key: jax.Array, stochastic_rounding: bool, test_noise_axis=None, coeff: float=1.6)` — [`L661`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L661) — Round to nearest radix4 (4^n).

## Module values
- `RADIX4` — [`L277`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L277)
- `e0m3` — [`L214`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L214)
- `e0m3_ocp` — [`L224`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L224)
- `e0m4_ocp` — [`L141`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L141)
- `e0m5_ocp` — [`L88`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L88)
- `e1m0` — [`L237`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L237)
- `e1m2` — [`L194`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L194) — documented in [aqt-jax-v2-numerics-fp_numerics](../../../../../concepts/aqt-jax-v2-numerics-fp_numerics.md)
- `e1m2_ocp` — [`L204`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L204) — documented in [aqt-jax-v2-numerics-fp_numerics](../../../../../concepts/aqt-jax-v2-numerics-fp_numerics.md)
- `e1m3_ocp` — [`L131`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L131)
- `e1m4_ocp` — [`L78`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L78)
- `e2m1` — [`L174`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L174)
- `e2m1_ocp` — [`L184`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L184)
- `e2m2_ocp` — [`L121`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L121)
- `e2m3_ocp` — [`L68`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L68)
- `e3m0` — [`L154`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L154)
- `e3m0_ocp` — [`L164`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L164)
- `e3m1_ocp` — [`L111`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L111)
- `e3m2_ocp` — [`L58`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L58)
- `e4m0_ocp` — [`L101`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L101)
- `e4m1_ocp` — [`L48`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L48)
- `e5m0_ocp` — [`L38`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L38)
- `float16` — [`L267`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L267)
- `float8_e4m3fn` — [`L247`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L247)
- `float8_e5m2` — [`L257`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L257)
- `float_repr` — [`L22`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp_numerics.py#L22)

