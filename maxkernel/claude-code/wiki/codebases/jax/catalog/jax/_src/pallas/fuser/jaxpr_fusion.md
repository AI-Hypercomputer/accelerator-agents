---
title: 'Module: jax/_src/pallas/fuser/jaxpr_fusion.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/fuser/jaxpr_fusion.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.fuser.jaxpr_fusion`/
symbols:
  fuse_jaxpr: fuse_jaxpr().
  _construct_output_fusions: _construct_output_fusions().
  _construct_fusion_jaxpr: _construct_fusion_jaxpr().
  fuse.decorator.wrapper: fuse().decorator().wrapper().
  construct_input_fusion: construct_input_fusion().
  fuse: fuse().
  fuse.decorator: fuse().decorator().
  construct_input_fusion._fn: construct_input_fusion()._fn().
  _find_downstream: _find_downstream().
  _construct_output_fusions._fn: _construct_output_fusions()._fn().
  _fusible._fusible: _fusible._fusible.
  _construct_output_permutation: _construct_output_permutation().
---
# Module: [`jax/_src/pallas/fuser/jaxpr_fusion.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/jaxpr_fusion.py)

## Functions
- `_construct_fusion_jaxpr(candidate_values, jaxpr: jax_core.Jaxpr, outvars, *invars, **kwargs)` — [`L93`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/jaxpr_fusion.py#L93)
- `_construct_output_fusions(candidate_values, jaxpr, out_tree, fusion_eqn_index, fusion_eqn_outvars, fusion_eqn_out_tree, output_fusion_prefix, *, strict_mode: bool = True)` — [`L170`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/jaxpr_fusion.py#L170)
- `_construct_output_permutation(used: list[tuple[bool, ...]])` — [`L160`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/jaxpr_fusion.py#L160)
- `_find_downstream(jaxpr: jax_core.Jaxpr, in_used: Sequence[bool])` — [`L138`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/jaxpr_fusion.py#L138)
- `_fn()` — [`L131`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/jaxpr_fusion.py#L131)
- `_fn(jaxpr, vals, *args, **kwargs)` — [`L269`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/jaxpr_fusion.py#L269)
- `construct_input_fusion(candidate_values, jaxpr: jax_core.Jaxpr, outvars)` — [`L124`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/jaxpr_fusion.py#L124)
- `decorator(f)` — [`L57`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/jaxpr_fusion.py#L57)
- `fuse(f=None, *, resolve_fusion_dtypes: bool = True, debug: bool = False, strict_mode: bool = True)` — [`L35`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/jaxpr_fusion.py#L35) — Fuses a function into a single fusible.
- `fuse_jaxpr(jaxpr: jax_core.Jaxpr, out_tree: tree_util.PyTreeDef, consts, *args, strict_mode: bool = True)` — [`L293`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/jaxpr_fusion.py#L293)
- `wrapper(*args, **kwargs)` — [`L58`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/jaxpr_fusion.py#L58)

## Module values
- `_fusible` — [`L90`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/jaxpr_fusion.py#L90)

