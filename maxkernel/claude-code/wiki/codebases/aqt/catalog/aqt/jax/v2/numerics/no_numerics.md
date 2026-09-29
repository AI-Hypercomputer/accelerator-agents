---
title: 'Module: aqt/jax/v2/numerics/no_numerics.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/numerics/no_numerics.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.numerics.no_numerics`/NoNumerics#
symbols:
  NoNumerics: ''
  NoNumerics.noise_fn: noise_fn.
  NoNumerics.dtype: dtype.
  NoNumerics.get_dtype: get_dtype().
  NoNumerics.get_quant_bound: get_quant_bound().
  NoNumerics.vjp_fwd: vjp_fwd().
  NoNumerics.vjp_bwd: vjp_bwd().
---
# Module: [`aqt/jax/v2/numerics/no_numerics.py`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/no_numerics.py)

## Classes
### `NoNumerics`
- def: [`aqt/jax/v2/numerics/no_numerics.py:23`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/no_numerics.py#L23) — documented in [aqt-jax-v2-aqt_quantizer](../../../../../concepts/aqt-jax-v2-aqt_quantizer.md)
- doc: No quantization, use a native type such as bf16.
- signature: `class NoNumerics(numerics.AqtNumerics):`
- members:
  - `get_dtype(self)` — [`L33`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/no_numerics.py#L33)
  - `get_quant_bound(self)` — [`L36`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/no_numerics.py#L36)
  - `vjp_bwd(self, res, grad)` — [`L42`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/no_numerics.py#L42)
  - `vjp_fwd(self, x, context)` — [`L39`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/no_numerics.py#L39)
  - `dtype` — [`L29`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/no_numerics.py#L29)
  - `noise_fn` — [`L28`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/no_numerics.py#L28)
- used by: [`calibrate`](../aqt_quantizer.md#Quantizer.calibrate), [`calculate_qvalue`](../aqt_quantizer.md#Quantizer.calculate_qvalue)

