---
title: 'Module: aqt/jax/v2/numerics/numerics.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/numerics/numerics.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.numerics.numerics`/AqtNumerics#
symbols:
  AqtNumerics: ''
  AqtNumerics.get_dtype: get_dtype().
  AqtNumerics.get_quant_bound: get_quant_bound().
  AqtNumerics.vjp_fwd: vjp_fwd().
  AqtNumerics.vjp_bwd: vjp_bwd().
---
# Module: [`aqt/jax/v2/numerics/numerics.py`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/numerics.py)

## Classes
### `AqtNumerics`
- def: [`aqt/jax/v2/numerics/numerics.py:21`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/numerics.py#L21) — documented in [aqt-jax-v2-aqt_quantizer](../../../../../concepts/aqt-jax-v2-aqt_quantizer.md)
- doc: Abstract class for various quantization numerics.
- signature: `class AqtNumerics(abc.ABC):`
- members:
  - `get_dtype(self)` — [`L25`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/numerics.py#L25)
  - `get_quant_bound(self)` — [`L29`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/numerics.py#L29) — The width that the bound corresponds to in the quantization range.
  - `vjp_bwd(self, res, grad)` — [`L53`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/numerics.py#L53)
  - `vjp_fwd(self, x, context)` — [`L49`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/numerics.py#L49)
- used by: [`AbstractAqtNumerics`](../aqt_quantizer.md#AbstractAqtNumerics)

