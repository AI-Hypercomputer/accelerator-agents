---
title: 'Module: jax/_src/pallas/fuser/fuser_utils.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/fuser/fuser_utils.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.fuser.fuser_utils`/
symbols:
  discharge_state: discharge_state().
  _jaxpr_signature: _jaxpr_signature().
  _make_hashable: _make_hashable().
  make_jaxpr: make_jaxpr().
  _jaxpr_signature.get_effect_sig: _jaxpr_signature().get_effect_sig().
  get_write_indices: get_write_indices().
  _jaxpr_signature.get_var_sig: _jaxpr_signature().get_var_sig().
  compare_jaxprs: compare_jaxprs().
---
# Module: [`jax/_src/pallas/fuser/fuser_utils.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fuser_utils.py)

## Functions
- `_jaxpr_signature(jaxpr_obj)` — [`L78`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fuser_utils.py#L78)
- `_make_hashable(val)` — [`L50`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fuser_utils.py#L50)
- `compare_jaxprs(jaxpr1, jaxpr2)` — [`L124`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fuser_utils.py#L124) — Compares two JAXPRs for symbolic equivalence.
- `discharge_state(jaxpr: core.Jaxpr, *, allow_additional_outputs: bool = True, dce: bool = False)` — [`L142`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fuser_utils.py#L142) — Converts a stateful fusion jaxpr into a pure one.
- `get_effect_sig(e)` — [`L101`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fuser_utils.py#L101)
- `get_var_sig(v)` — [`L91`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fuser_utils.py#L91)
- `get_write_indices(jaxpr)` — [`L131`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fuser_utils.py#L131)
- `make_jaxpr(f, *args, **kwargs)` — [`L31`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fuser_utils.py#L31)

