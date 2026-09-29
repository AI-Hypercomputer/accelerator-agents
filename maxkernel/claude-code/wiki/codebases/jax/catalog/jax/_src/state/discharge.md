---
title: 'Module: jax/_src/state/discharge.py'
type: catalog
provenance: extracted
module: jax/_src/state/discharge.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.state.discharge`/
symbols:
  _eval_jaxpr_discharge_state: _eval_jaxpr_discharge_state().
  transform_swap_array: transform_swap_array().
  _convert_to_gather_arrays: _convert_to_gather_arrays().
  run_state_reference.wrapped: run_state_reference().wrapped().
  transform_array: transform_array().
  register_discharge_rule: register_discharge_rule().
  _addupdate_discharge: _addupdate_discharge().
  _run_state_jvp: _run_state_jvp().
  discharge_state: discharge_state().
  _pjit_state_discharge_rule: _pjit_state_discharge_rule().
  _call_primitive_discharge_rule: _call_primitive_discharge_rule().
  run_state.wrapped: run_state().wrapped().
  run_state_p: run_state_p.
  _is_trivial_indexer: _is_trivial_indexer().
  _maybe_convert_to_dynamic_slice: _maybe_convert_to_dynamic_slice().
  _index_array: _index_array().
  _discharge_state: _discharge_state().
  _closed_call_discharge_rule: _closed_call_discharge_rule().
  custom_vjp_call_discharge: custom_vjp_call_discharge().
  _run_state_impl: _run_state_impl().
  _perform_transpose_before_gather: _perform_transpose_before_gather().
  _run_state_to_lojax: _run_state_to_lojax().
  _run_state_abstract_eval: _run_state_abstract_eval().
  _cached_closed_jaxpr_discharge: _cached_closed_jaxpr_discharge().
  Environment.read: Environment#read().
  _maybe_convert_to_slice: _maybe_convert_to_slice().
  _initial_style_jaxpr: _initial_style_jaxpr().
  _run_state_discharge_rule: _run_state_discharge_rule().
  initial_style_jaxpr: initial_style_jaxpr().
  register_partial_discharge_rule: register_partial_discharge_rule().
  run_state: run_state().
  Environment.write: Environment#write().
  _get_discharge_rule: _get_discharge_rule().
  _swap_discharge_rule: _swap_discharge_rule().
  _addupdate_discharge_rule: _addupdate_discharge_rule().
  _maybe_transpose_before_gather: _maybe_transpose_before_gather().
  _discharge_rules._discharge_rules: _discharge_rules._discharge_rules.
  _partial_discharge_rules._partial_discharge_rules: _partial_discharge_rules._partial_discharge_rules.
  _eval_jaxpr_ad_error: _eval_jaxpr_ad_error().
  register_discharge_rule.register: register_discharge_rule().register().
  register_partial_discharge_rule.register: register_partial_discharge_rule().register().
  _get_discharge: _get_discharge().
  _swap_discharge: _swap_discharge().
  _eval_jaxpr_ad_error_jvp: _eval_jaxpr_ad_error_jvp().
  PyTreeDef: PyTreeDef.
  Environment.env: Environment#env.
  _default_initialization: _default_initialization().
  DischargeRule: DischargeRule#
  DischargeRule.__call__: DischargeRule#__call__().
  PartialDischargeRule.__call__: PartialDischargeRule#__call__().
  run_state_reference: run_state_reference().
  unsafe_map: unsafe_map.
  unsafe_zip: unsafe_zip.
  PartialDischargeRule: PartialDischargeRule#
  T: T.
  Environment: Environment#
  _convert_to_gather_arrays.get_idx_in_shape_after_indexing: _convert_to_gather_arrays().get_idx_in_shape_after_indexing().
  _run_state_is_high: _run_state_is_high().
---
# Module: [`jax/_src/state/discharge.py`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py)

## Classes
### `DischargeRule`  ·  implements/extends Protocol
- def: [`jax/_src/state/discharge.py:117`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L117)
- signature: `class DischargeRule(Protocol):`
- members:
  - `__call__(self, in_avals: Sequence[core.AbstractValue], out_avals: Sequence[core.AbstractValue], *args: Any, **params: Any)` — [`L119`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L119) — Discharge rule for a primitive.
- uses (calls/refs, reference-scoped): [`AbstractValue`](../core.md#AbstractValue)
- used by: [`_eval_jaxpr_discharge_state`](discharge.md#_eval_jaxpr_discharge_state), [`_discharge_rules`](discharge.md#_discharge_rules._discharge_rules), [`register`](discharge.md#register_discharge_rule.register)

### `Environment`
- def: [`jax/_src/state/discharge.py:104`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L104)
- signature: `class Environment:`
- members:
  - `read(self, v: core.Atom)` — [`L107`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L107)
  - `write(self, v: core.Var, val: Any)` — [`L113`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L113)
  - `env` — [`L105`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L105)
- uses (calls/refs, reference-scoped): [`Var`](../core.md#Var), [`Atom`](../core.md#Atom), [`val`](../core.md#Literal.val)
- used by: [`_eval_jaxpr_discharge_state`](discharge.md#_eval_jaxpr_discharge_state)

### `PartialDischargeRule`  ·  implements/extends Protocol
- def: [`jax/_src/state/discharge.py:157`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L157)
- doc: Discharge rule that supports selective discharging of `Ref` inputs.
- signature: `class PartialDischargeRule(Protocol):`
- protocol/private: `__call__`[`L166`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L166)
- uses (calls/refs, reference-scoped): [`AbstractValue`](../core.md#AbstractValue)
- used by: [`_partial_discharge_rules`](discharge.md#_partial_discharge_rules._partial_discharge_rules), [`register`](discharge.md#register_partial_discharge_rule.register)

## Functions
- `_addupdate_discharge(x, val, idx, tree)` — [`L585`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L585)
- `_addupdate_discharge_rule(in_avals: Sequence[core.AbstractValue], out_avals: Sequence[core.AbstractValue], x, val, *idx, tree)` — [`L577`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L577)
- `_cached_closed_jaxpr_discharge(closed_jaxpr: core.ClosedJaxpr)` — [`L618`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L618)
- `_call_primitive_discharge_rule(prim: core.Primitive, in_avals: Sequence[core.AbstractValue], _, *args, call_jaxpr: core.Jaxpr, **kwargs)` — [`L640`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L640)
- `_closed_call_discharge_rule(in_avals: Sequence[core.AbstractValue], _, *args, call_jaxpr: core.ClosedJaxpr)` — [`L626`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L626)
- `_convert_to_gather_arrays(indexer: indexing.NDIndexer)` — [`L384`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L384)
- `_default_initialization(x)` — [`L691`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L691)
- `_discharge_state(closed_jaxpr: core.ClosedJaxpr, should_discharge: tuple[bool, ...], lower: bool)` — [`L86`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L86)
- `_eval_jaxpr_ad_error(dis_jaxpr, consts, args)` — [`L932`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L932)
- `_eval_jaxpr_ad_error_jvp(*_)` — [`L935`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L935)
- `_eval_jaxpr_discharge_state(jaxpr: core.Jaxpr, should_discharge: Sequence[bool], consts: Sequence[Any], *args: Any)` — [`L186`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L186)
- `_get_discharge(x, idx, tree)` — [`L559`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L559)
- `_get_discharge_rule(in_avals: Sequence[core.AbstractValue], out_avals: Sequence[core.AbstractValue], x, *idx, tree)` — [`L432`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L432)
- `_index_array(x, indexer: indexing.NDIndexer)` — [`L441`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L441)
- `_initial_style_jaxpr(fun: Callable, in_tree: api_util.PyTreeDef, in_avals: Sequence[core.AbstractValue], debug: core.DebugInfo)` — [`L817`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L817)
- `_is_trivial_indexer(indexer: indexing.NDIndexer)` — [`L268`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L268) — Returns whether the indexer selects the entire shape.
- `_maybe_convert_to_dynamic_slice(indexer: indexing.NDIndexer)` — [`L302`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L302)
- `_maybe_convert_to_slice(indexer: indexing.NDIndexer)` — [`L280`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L280)
- `_maybe_transpose_before_gather(indexer: indexing.NDIndexer)` — [`L350`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L350)
- `_perform_transpose_before_gather(target_arr: Array, indexer: indexing.NDIndexer, transpose_order: tuple[int, ...])` — [`L369`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L369)
- `_pjit_state_discharge_rule(in_avals, out_avals, *args, jaxpr, in_shardings, out_shardings, in_layouts, out_layouts, **params)` — [`L872`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L872)
- `_run_state_abstract_eval(*avals: core.AbstractValue, jaxpr: core.Jaxpr, which_linear: tuple[bool, ...], is_initialized: tuple[bool, ...])` — [`L717`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L717)
- `_run_state_discharge_rule(in_avals: Sequence[core.AbstractValue], out_avals: Sequence[core.AbstractValue], *args: Any, jaxpr: core.Jaxpr, which_linear: Sequence[bool], is_initialized: tuple[bool, ...])` — [`L793`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L793)
- `_run_state_impl(*args: Any, jaxpr: core.Jaxpr, which_linear: tuple[bool, ...], is_initialized: tuple[bool, ...])` — [`L701`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L701)
- `_run_state_is_high(*_, jaxpr, **__)` — [`L674`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L674)
- `_run_state_jvp(primals: Sequence[Any], tangents: Sequence[Any], *, jaxpr: core.Jaxpr, which_linear: tuple[bool, ...], is_initialized: tuple[bool, ...])` — [`L755`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L755)
- `_run_state_to_lojax(*args, jaxpr, is_initialized, **params)` — [`L678`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L678)
- `_swap_discharge(x, val, idx, tree)` — [`L572`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L572)
- `_swap_discharge_rule(in_avals: Sequence[core.AbstractValue], out_avals: Sequence[core.AbstractValue], x, val, *idx, tree)` — [`L564`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L564)
- `custom_vjp_call_discharge(in_avals, out_avals, *args, call_jaxpr, fwd_jaxpr_thunk, bwd, out_trees, symbolic_zeros, num_consts)` — [`L916`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L916)
- `discharge_state(closed_jaxpr: core.ClosedJaxpr, *, should_discharge: bool | Sequence[bool] = True, lower: bool = True)` — [`L59`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L59) — Converts a stateful jaxpr into a pure one.
- `get_idx_in_shape_after_indexing(i)` — [`L399`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L399)
- `initial_style_jaxpr(fun: Callable, in_tree: PyTreeDef, in_avals: Sequence[core.AbstractValue], dbg: core.DebugInfo)` — [`L810`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L810)
- `register(f: DischargeRule)` — [`L150`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L150)
- `register(f: PartialDischargeRule)` — [`L181`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L181)
- `register_discharge_rule(prim: core.Primitive)` — [`L149`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L149)
- `register_partial_discharge_rule(prim: core.Primitive)` — [`L180`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L180)
- `run_state(f: Callable[..., None])` — [`L829`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L829)
- `run_state_reference(f: Callable[..., None])` — [`L849`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L849)
- `transform_array(x, transforms)` — [`L464`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L464)
- `transform_swap_array(x, transforms, val)` — [`L482`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L482)
- `wrapped(args)` — [`L830`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L830)
- `wrapped(args)` — [`L850`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L850)

## Module values
- `PyTreeDef` — [`L54`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L54)
- `T` — [`L828`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L828)
- `_discharge_rules` — [`L146`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L146)
- `_partial_discharge_rules` — [`L177`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L177)
- `run_state_p` — [`L671`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L671)
- `unsafe_map` — [`L52`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L52)
- `unsafe_zip` — [`L53`](../../../../../../../raw/code/jax/jax/_src/state/discharge.py#L53)

