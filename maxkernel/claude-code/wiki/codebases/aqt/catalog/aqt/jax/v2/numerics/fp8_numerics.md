---
title: 'Module: aqt/jax/v2/numerics/fp8_numerics.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/numerics/fp8_numerics.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.numerics.fp8_numerics`/
symbols:
  fp8_map: fp8_map.
  _convert_to_fp8dtype: _convert_to_fp8dtype().
  FP8Dtype: FP8Dtype.
  fp_mantissa_round: fp_mantissa_round().
  Fp8Numerics: Fp8Numerics#
  Fp8Numerics.dtype: Fp8Numerics#dtype.
  Fp8Numerics.exponent_bits: Fp8Numerics#exponent_bits.
  Fp8Numerics.mantissa_bits: Fp8Numerics#mantissa_bits.
  Fp8Numerics.stochastic_rounding: Fp8Numerics#stochastic_rounding.
  Fp8Numerics._get_edge_of_last_fp8_bucket: Fp8Numerics#_get_edge_of_last_fp8_bucket().
  Fp8Numerics.get_dtype: Fp8Numerics#get_dtype().
  Fp8Numerics.get_quant_bound: Fp8Numerics#get_quant_bound().
  Fp8Numerics.vjp_fwd: Fp8Numerics#vjp_fwd().
  Fp8Numerics.vjp_bwd: Fp8Numerics#vjp_bwd().
  round_to_nearest_even: round_to_nearest_even().
---
# Module: [`aqt/jax/v2/numerics/fp8_numerics.py`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics.py)

## Classes
### `Fp8Numerics`
- def: [`aqt/jax/v2/numerics/fp8_numerics.py:68`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics.py#L68)
- doc: Numerics for fp8.
- signature: `class Fp8Numerics(numerics.AqtNumerics):`
- members:
  - `get_dtype(self)` — [`L82`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics.py#L82)
  - `get_quant_bound(self)` — [`L85`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics.py#L85)
  - `vjp_bwd(self, res, grad)` — [`L125`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics.py#L125)
  - `vjp_fwd(self, x, context)` — [`L88`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics.py#L88)
  - `dtype` — [`L72`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics.py#L72)
  - `exponent_bits` — [`L75`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics.py#L75)
  - `mantissa_bits` — [`L76`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics.py#L76)
  - `stochastic_rounding` — [`L77`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics.py#L77)
- protocol/private: `_get_edge_of_last_fp8_bucket`[`L79`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics.py#L79)

## Functions
- `_convert_to_fp8dtype(dtype: jax.typing.DTypeLike)` — [`L27`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics.py#L27)
- `fp_mantissa_round(x, mantissa_bits, key: jax.Array)` — [`L36`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics.py#L36) — FP stochastic rounding for a given mantissa.
- `round_to_nearest_even(x: jnp.ndarray, dtype: jnp.dtype)` — [`L136`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics.py#L136)

## Module values
- `FP8Dtype` — [`L23`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics.py#L23)
- `fp8_map` — [`L24`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics.py#L24)

