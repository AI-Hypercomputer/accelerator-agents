---
title: 'Module: aqt/jax/v2/numerics/int_numerics.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/numerics/int_numerics.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.numerics.int_numerics`/IntSymmetric#
symbols:
  IntSymmetric: ''
  IntSymmetric.bits: bits.
  IntSymmetric.preserve_zero: preserve_zero.
  IntSymmetric.preserve_max_val: preserve_max_val.
  IntSymmetric.clip: clip.
  IntSymmetric.clip_gradient: clip_gradient.
  IntSymmetric.round: round.
  IntSymmetric.noise_fn: noise_fn.
  IntSymmetric.dtype: dtype.
  IntSymmetric.get_edge_of_last_int_bucket: get_edge_of_last_int_bucket().
  IntSymmetric.get_center_of_last_int_bucket: get_center_of_last_int_bucket().
  IntSymmetric.get_quant_bound: get_quant_bound().
  IntSymmetric._get_fwd_clip_bound: _get_fwd_clip_bound().
  IntSymmetric.get_dtype: get_dtype().
  IntSymmetric.vjp_fwd: vjp_fwd().
  IntSymmetric.vjp_bwd: vjp_bwd().
---
# Module: [`aqt/jax/v2/numerics/int_numerics.py`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/int_numerics.py)

## Classes
### `IntSymmetric`
- def: [`aqt/jax/v2/numerics/int_numerics.py:25`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/int_numerics.py#L25)
- doc: Symmetric numerics for sint8, sint4, binary, etc.
- signature: `class IntSymmetric(numerics.AqtNumerics):`
- members:
  - `get_center_of_last_int_bucket(self)` — [`L58`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/int_numerics.py#L58)
  - `get_dtype(self)` — [`L78`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/int_numerics.py#L78)
  - `get_edge_of_last_int_bucket(self)` — [`L51`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/int_numerics.py#L51)
  - `get_quant_bound(self)` — [`L61`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/int_numerics.py#L61)
  - `vjp_bwd(self, res, grad)` — [`L113`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/int_numerics.py#L113)
  - `vjp_fwd(self, x, context)` — [`L81`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/int_numerics.py#L81) — Forward pass.
  - `bits` — [`L28`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/int_numerics.py#L28)
  - `clip` — [`L35`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/int_numerics.py#L35)
  - `clip_gradient` — [`L36`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/int_numerics.py#L36)
  - `dtype` — [`L39`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/int_numerics.py#L39)
  - `noise_fn` — [`L38`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/int_numerics.py#L38)
  - `preserve_max_val` — [`L32`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/int_numerics.py#L32)
  - `preserve_zero` — [`L29`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/int_numerics.py#L29)
  - `round` — [`L37`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/int_numerics.py#L37)
- protocol/private: `_get_fwd_clip_bound`[`L67`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/int_numerics.py#L67)

