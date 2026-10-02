---
title: 'Module: jax/_src/deprecations.py'
type: catalog
provenance: extracted
module: jax/_src/deprecations.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.deprecations`/
symbols:
  register: register().
  warn: warn().
  _registered_deprecations._registered_deprecations: _registered_deprecations._registered_deprecations.
  is_accelerated: is_accelerated().
  accelerate: accelerate().
  deprecation_getattr: deprecation_getattr().
  unregister: unregister().
  DeprecationState: DeprecationState#
  DeprecationState.accelerated: DeprecationState#accelerated.
  deprecation_getattr.getattr: deprecation_getattr().getattr().
  accelerate_getattr_deprecation: accelerate_getattr_deprecation().
  is_accelerated_attribute: is_accelerated_attribute().
---
# Module: [`jax/_src/deprecations.py`](../../../../../../raw/code/jax/jax/_src/deprecations.py)

## Classes
### `DeprecationState`
- def: [`jax/_src/deprecations.py:89`](../../../../../../raw/code/jax/jax/_src/deprecations.py#L89)
- signature: `class DeprecationState:`
- members:
  - `accelerated` — [`L90`](../../../../../../raw/code/jax/jax/_src/deprecations.py#L90)
- used by: [`register`](deprecations.md#register), [`_registered_deprecations`](deprecations.md#_registered_deprecations._registered_deprecations), [`is_accelerated`](deprecations.md#is_accelerated), [`accelerate`](deprecations.md#accelerate)

## Functions
- `accelerate(deprecation_id: str)` — [`L105`](../../../../../../raw/code/jax/jax/_src/deprecations.py#L105)
- `accelerate_getattr_deprecation(module: ModuleType, *names: str)` — [`L62`](../../../../../../raw/code/jax/jax/_src/deprecations.py#L62) — Accelerate the deprecation of a module-level attribute.
- `deprecation_getattr(module, deprecations)` — [`L48`](../../../../../../raw/code/jax/jax/_src/deprecations.py#L48)
- `getattr(name)` — [`L50`](../../../../../../raw/code/jax/jax/_src/deprecations.py#L50)
- `is_accelerated(deprecation_id: str)` — [`L111`](../../../../../../raw/code/jax/jax/_src/deprecations.py#L111)
- `is_accelerated_attribute(module: ModuleType, name: str)` — [`L73`](../../../../../../raw/code/jax/jax/_src/deprecations.py#L73) — Returns true if given name is accelerated.
- `register(deprecation_id: str)` — [`L95`](../../../../../../raw/code/jax/jax/_src/deprecations.py#L95)
- `unregister(deprecation_id: str)` — [`L99`](../../../../../../raw/code/jax/jax/_src/deprecations.py#L99)
- `warn(deprecation_id: str, message: str, stacklevel: int, *, error_class: type[Exception] = ValueError)` — [`L117`](../../../../../../raw/code/jax/jax/_src/deprecations.py#L117) — Warns about a deprecation, or errors if the deprecation is accelerated.

## Module values
- `_registered_deprecations` — [`L92`](../../../../../../raw/code/jax/jax/_src/deprecations.py#L92)

