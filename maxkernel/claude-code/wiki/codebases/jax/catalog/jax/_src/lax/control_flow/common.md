---
title: 'Module: jax/_src/lax/control_flow/common.py'
type: catalog
provenance: extracted
module: jax/_src/lax/control_flow/common.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.lax.control_flow.common`/
symbols:
  _pad_constvars: _pad_constvars().
  _check_tree_and_avals: _check_tree_and_avals().
  _avals_short: _avals_short().
  _merge_common_consts: _merge_common_consts().
  _dedup_consts: _dedup_consts().
  _make_closed_jaxpr: _make_closed_jaxpr().
  _prune_zeros: _prune_zeros().
  _check_tree: _check_tree().
  _typecheck_param: _typecheck_param().
  _pad_constvars.make_var: _pad_constvars().make_var().
  _show_diff: _show_diff().
  unsafe_map: unsafe_map.
---
# Module: [`jax/_src/lax/control_flow/common.py`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/common.py)

## Functions
- `_avals_short(avals)` — [`L161`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/common.py#L161)
- `_check_tree(func_name, expected_name, actual_tree, expected_tree, has_aux=False)` — [`L123`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/common.py#L123)
- `_check_tree_and_avals(what1, tree1, avals1, what2, tree2, avals2)` — [`L100`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/common.py#L100) — Raises TypeError if (tree1, avals1) does not match (tree2, avals2).
- `_dedup_consts(jaxpr, num_consts, const_ids)` — [`L83`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/common.py#L83)
- `_make_closed_jaxpr(traceable, in_avals: Sequence[core.AbstractValue], debug_info: core.DebugInfo)` — [`L145`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/common.py#L145)
- `_merge_common_consts(jaxprs: Sequence[core.ClosedJaxpr], all_consts: Sequence[Sequence[Any]])` — [`L52`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/common.py#L52)
- `_pad_constvars(jaxpr: core.ClosedJaxpr, num_consts: int, left: tuple[core.AvalQDD, ...], right: tuple[core.AbstractValue, ...])` — [`L71`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/common.py#L71)
- `_prune_zeros(ts)` — [`L141`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/common.py#L141)
- `_show_diff(array1, array2)` — [`L156`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/common.py#L156)
- `_typecheck_param(prim, param, name, msg_required, pred)` — [`L39`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/common.py#L39)
- `make_var(aq)` — [`L74`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/common.py#L74)

## Module values
- `unsafe_map` — [`L36`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/common.py#L36)

