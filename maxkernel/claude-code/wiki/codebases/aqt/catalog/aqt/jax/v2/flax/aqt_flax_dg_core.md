---
title: 'Module: aqt/jax/v2/flax/aqt_flax_dg_core.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/flax/aqt_flax_dg_core.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.flax.aqt_flax_dg_core`/dg_core_flax_lifted().
symbols:
  dg_core_flax_lifted: ''
---
# Module: [`aqt/jax/v2/flax/aqt_flax_dg_core.py`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_dg_core.py)

## Functions
- `dg_core_flax_lifted(lhs: jnp.ndarray, rhs: jnp.ndarray, lhs_qt: None | aqt_tensor.QTensor, rhs_qt: None | aqt_tensor.QTensor, dimension_numbers: jax.lax.DotDimensionNumbers, mdl: nn.Module, cfg: aqt_dot_general.DotGeneral)` — [`L22`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_dg_core.py#L22) — dot_general with flax lifted custom_vjp applied on it.

