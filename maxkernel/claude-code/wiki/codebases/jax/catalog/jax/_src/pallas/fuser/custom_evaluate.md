---
title: 'Module: jax/_src/pallas/fuser/custom_evaluate.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/fuser/custom_evaluate.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.fuser.custom_evaluate`/
symbols:
  _custom_evaluate_jaxpr: _custom_evaluate_jaxpr().
  evaluate.wrapped: evaluate().wrapped().
  disallowed_primitives: disallowed_primitives.
  _custom_evaluate_jaxpr.read: _custom_evaluate_jaxpr().read().
  _custom_evaluate_jaxpr.write: _custom_evaluate_jaxpr().write().
  evaluate: evaluate().
  CustomEvaluateSettings: CustomEvaluateSettings#
  CustomEvaluateSettings.allow_transpose: CustomEvaluateSettings#allow_transpose.
---
# Module: [`jax/_src/pallas/fuser/custom_evaluate.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_evaluate.py)

## Classes
### `CustomEvaluateSettings`
- def: [`jax/_src/pallas/fuser/custom_evaluate.py:28`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_evaluate.py#L28)
- signature: `class CustomEvaluateSettings:`
- members:
  - `allow_transpose` — [`L29`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_evaluate.py#L29)
- used by: [`_custom_evaluate_jaxpr`](custom_evaluate.md#_custom_evaluate_jaxpr), [`wrapped`](custom_evaluate.md#evaluate.wrapped)

## Functions
- `_custom_evaluate_jaxpr(settings: CustomEvaluateSettings, jaxpr: core.Jaxpr, consts, *args)` — [`L47`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_evaluate.py#L47)
- `evaluate(f, *, allow_transpose: bool = True)` — [`L32`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_evaluate.py#L32)
- `read(v: core.Atom)` — [`L50`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_evaluate.py#L50)
- `wrapped(*args, **kwargs)` — [`L33`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_evaluate.py#L33)
- `write(v: core.Var, val: Any)` — [`L53`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_evaluate.py#L53)

## Module values
- `disallowed_primitives` — [`L44`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_evaluate.py#L44)

