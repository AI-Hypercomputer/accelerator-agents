---
title: 'Module: jax/_src/state/utils.py'
type: catalog
provenance: extracted
module: jax/_src/state/utils.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.state.utils`/
symbols:
  hoist_consts_to_refs: hoist_consts_to_refs().
  bitcast: bitcast().
  eval_bitcast_shape: eval_bitcast_shape().
  hoist_consts_to_refs._hoist: hoist_consts_to_refs()._hoist().
  unsafe_map: unsafe_map.
  unsafe_zip: unsafe_zip.
---
# Module: [`jax/_src/state/utils.py`](../../../../../../../raw/code/jax/jax/_src/state/utils.py)

## Functions
- `_hoist(*consts_args)` — [`L65`](../../../../../../../raw/code/jax/jax/_src/state/utils.py#L65)
- `bitcast(x, dtype: DTypeLike)` — [`L83`](../../../../../../../raw/code/jax/jax/_src/state/utils.py#L83)
- `eval_bitcast_shape(x, dtype: DTypeLike)` — [`L110`](../../../../../../../raw/code/jax/jax/_src/state/utils.py#L110)
- `hoist_consts_to_refs(jaxpr: core.Jaxpr, *, index: int = 0, make_abstract_ref: Callable[[core.AbstractValue], AbstractRef] = lambda aval: AbstractRef(aval))` — [`L34`](../../../../../../../raw/code/jax/jax/_src/state/utils.py#L34) — Hoists the constants in the given jaxpr into invars.

## Module values
- `unsafe_map` — [`L30`](../../../../../../../raw/code/jax/jax/_src/state/utils.py#L30)
- `unsafe_zip` — [`L31`](../../../../../../../raw/code/jax/jax/_src/state/utils.py#L31)

