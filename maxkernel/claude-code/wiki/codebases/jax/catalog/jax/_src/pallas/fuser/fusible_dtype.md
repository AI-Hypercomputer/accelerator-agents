---
title: 'Module: jax/_src/pallas/fuser/fusible_dtype.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/fuser/fusible_dtype.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.fuser.fusible_dtype`/
symbols:
  physicalize_interp: physicalize_interp().
  _pallas_call_physicalize_rule: _pallas_call_physicalize_rule().
  physicalize_closed_jaxpr: physicalize_closed_jaxpr().
  _custom_vjp_call_physicalize_rule: _custom_vjp_call_physicalize_rule().
  physicalize_jaxpr: physicalize_jaxpr().
  _physical_aval: _physical_aval().
  physicalize.wrapper: physicalize().wrapper().
  _while_rule: _while_rule().
  _unpack_dtype_push_rule: _unpack_dtype_push_rule().
  _unpack_dtype_pull_rule: _unpack_dtype_pull_rule().
  _assert_no_fusion_types: _assert_no_fusion_types().
  _core_map_rule: _core_map_rule().
  _mpmd_map_rule: _mpmd_map_rule().
  _run_scoped_rule: _run_scoped_rule().
  _run_state_rule: _run_state_rule().
  _scan_rule: _scan_rule().
  _physicalize_rules._physicalize_rules: _physicalize_rules._physicalize_rules.
  _is_fusion_type: _is_fusion_type().
  _pack_dtype_pull_rule: _pack_dtype_pull_rule().
  _unpack_dtype_eval_rule: _unpack_dtype_eval_rule().
  unpack_dtype_p: unpack_dtype_p.
  _cond_physicalize_rule: _cond_physicalize_rule().
  _fusible_physicalize_rule: _fusible_physicalize_rule().
  _swap_rule: _swap_rule().
  _get_rule: _get_rule().
  pack_dtype_p: pack_dtype_p.
  Context: Context#
  FusionDType: FusionDType#
  Context.avals_out: Context#avals_out.
  _phys_find_rule: _phys_find_rule().
  Context.avals_in: Context#avals_in.
  unpack_dtype_abstract_eval: unpack_dtype_abstract_eval().
  _pack_dtype_eval_rule: _pack_dtype_eval_rule().
  physicalize_interp.read_env: physicalize_interp().read_env().
  physicalize_interp.write_env: physicalize_interp().write_env().
  _physicalize_transform_bwd: _physicalize_transform_bwd().
  physicalize: physicalize().
  FusibleElementDType: FusibleElementDType#
  pack_dtype_abstract_eval: pack_dtype_abstract_eval().
  pack: pack().
  unpack: unpack().
  physicalize_jaxpr._flat_jaxpr_eval: physicalize_jaxpr()._flat_jaxpr_eval().
  FusionDType._register_fn: FusionDType#_register_fn().
  _physicalize_transform: _physicalize_transform().
  FusionDType._rules: FusionDType#_rules.
  FusionDType.type: FusionDType#type.
  FusionDType.register_op: FusionDType#register_op().
  FusionDType.get_op_rule: FusionDType#get_op_rule().
  unsafe_map: unsafe_map.
  unsafe_zip: unsafe_zip.
  FusionDType._op_registry: FusionDType#_op_registry.
  FusibleTyRules: FusibleTyRules#
  FusionDType.abstract_unpack: FusionDType#abstract_unpack().
  FusionDType.pull_block_spec_one_step: FusionDType#pull_block_spec_one_step().
  FusionDType.unpack_push_block_spec: FusionDType#unpack_push_block_spec().
  FusionDType.unpack_pull_block_spec: FusionDType#unpack_pull_block_spec().
  _pack_rule: _pack_rule().
  _unpack_rule: _unpack_rule().
  T: T.
  FusibleTyRules.allow_conversion: FusibleTyRules#allow_conversion.
  FusionDType.abstract_pack: FusionDType#abstract_pack().
  FusionDType.name: FusionDType#name().
---
# Module: [`jax/_src/pallas/fuser/fusible_dtype.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py)

## Classes
### `Context`
- def: [`jax/_src/pallas/fuser/fusible_dtype.py:208`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L208)
- signature: `class Context:`
- members:
  - `avals_in` — [`L209`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L209)
  - `avals_out` — [`L210`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L210)
- used by: [`physicalize_interp`](fusible_dtype.md#physicalize_interp), [`_pallas_call_physicalize_rule`](fusible_dtype.md#_pallas_call_physicalize_rule), [`_custom_vjp_call_physicalize_rule`](fusible_dtype.md#_custom_vjp_call_physicalize_rule), [`_while_rule`](fusible_dtype.md#_while_rule), [`_core_map_rule`](fusible_dtype.md#_core_map_rule), [`_mpmd_map_rule`](fusible_dtype.md#_mpmd_map_rule), [`_run_scoped_rule`](fusible_dtype.md#_run_scoped_rule), [`_run_state_rule`](fusible_dtype.md#_run_state_rule), [`_scan_rule`](fusible_dtype.md#_scan_rule), [`_cond_physicalize_rule`](fusible_dtype.md#_cond_physicalize_rule), [`_get_rule`](fusible_dtype.md#_get_rule), [`_swap_rule`](fusible_dtype.md#_swap_rule)

### `FusibleElementDType`
- def: [`jax/_src/pallas/fuser/fusible_dtype.py:84`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L84)
- uses (calls/refs, reference-scoped): [`extended`](../../dtypes.md#extended)
- used by: [`unpack_dtype_abstract_eval`](fusible_dtype.md#unpack_dtype_abstract_eval), [`pack_dtype_abstract_eval`](fusible_dtype.md#pack_dtype_abstract_eval), [`type`](fusible_dtype.md#FusionDType.type)

### `FusibleTyRules`
- def: [`jax/_src/pallas/fuser/fusible_dtype.py:88`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L88)
- signature: `class FusibleTyRules:`
- members:
  - `allow_conversion` — [`L89`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L89)
- used by: [`_rules`](fusible_dtype.md#FusionDType._rules)

### `FusionDType`  ·  implements/extends ExtendedDType, StrictABC
- def: [`jax/_src/pallas/fuser/fusible_dtype.py:92`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L92)
- doc: Base class for fusible extended dtypes.
- signature: `class FusionDType(dtypes.ExtendedDType, util.StrictABC):`
- members:
  - `abstract_pack(self, *xs)` — [`L104`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L104)
  - `abstract_unpack(self, x)` — [`L100`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L100)
  - `get_op_rule(cls, primitive)` — [`L115`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L115)
  - `name(self)` — [`L119`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L119)
  - `pull_block_spec_one_step(self, aval_out, *args, **kwargs)` — [`L123`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L123)
  - `register_op(cls, primitive)` — [`L108`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L108)
  - `unpack_pull_block_spec(self, aval_in, *args, **kwargs)` — [`L131`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L131)
  - `unpack_push_block_spec(self, aval_in, *args, **kwargs)` — [`L127`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L127)
  - `type` — [`L97`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L97)
- protocol/private: `_op_registry`[`L95`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L95), `_register_fn`[`L109`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L109), `_rules`[`L96`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L96)
- uses (calls/refs, reference-scoped): [`ExtendedDType`](../../dtypes.md#ExtendedDType), [`StrictABC`](../../util.md#StrictABC), [`FusibleElementDType`](fusible_dtype.md#FusibleElementDType), [`FusibleTyRules`](fusible_dtype.md#FusibleTyRules)
- used by: [`ExtendedDType`](../../dtypes.md#ExtendedDType), [`_physical_aval`](fusible_dtype.md#_physical_aval), [`_unpack_dtype_pull_rule`](fusible_dtype.md#_unpack_dtype_pull_rule), [`_unpack_dtype_push_rule`](fusible_dtype.md#_unpack_dtype_push_rule), [`_is_fusion_type`](fusible_dtype.md#_is_fusion_type), [`_pack_dtype_pull_rule`](fusible_dtype.md#_pack_dtype_pull_rule), [`_unpack_dtype_eval_rule`](fusible_dtype.md#_unpack_dtype_eval_rule), [`StrictABC`](../../util.md#StrictABC)

## Functions
- `_assert_no_fusion_types(avals: Sequence[core.AbstractValue])` — [`L295`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L295)
- `_cond_physicalize_rule(ctx: Context, *args, branches, **kwargs)` — [`L322`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L322)
- `_core_map_rule(ctx: Context, *args, jaxpr, **params)` — [`L385`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L385)
- `_custom_vjp_call_physicalize_rule(ctx: Context, *args, call_jaxpr, num_consts, fwd_jaxpr_thunk, bwd, **kwargs)` — [`L353`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L353)
- `_flat_jaxpr_eval(consts, args)` — [`L188`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L188)
- `_fusible_physicalize_rule(_, *consts_and_args, jaxpr, num_consts, in_tree, out_tree, func, **params)` — [`L573`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L573)
- `_get_rule(ctx: Context, ref, *args, tree)` — [`L516`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L516)
- `_is_fusion_type(aval: core.AbstractValue)` — [`L266`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L266) — Returns whether an aval is an array containing fusion types.
- `_mpmd_map_rule(ctx: Context, *args, jaxprs, meshes, external_meshes, **params)` — [`L397`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L397)
- `_pack_dtype_eval_rule(ctx: block_spec.KernelEvalContext, *args, dtype)` — [`L527`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L527)
- `_pack_dtype_pull_rule(ctx: block_spec.PullRuleContext, block_spec: pallas_core.BlockSpec, *, dtype: FusionDType)` — [`L532`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L532)
- `_pack_rule(_, *args, dtype)` — [`L491`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L491)
- `_pallas_call_physicalize_rule(ctx: Context, *args, jaxpr, grid_mapping: pallas_core.GridMapping, **kwargs)` — [`L300`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L300)
- `_phys_find_rule(primitive, avals: Sequence[core.AbstractValue])` — [`L275`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L275) — Finds the physicalization rule for a primitive.
- `_physical_aval(aval)` — [`L172`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L172)
- `_physicalize_transform(f, *args)` — [`L337`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L337)
- `_physicalize_transform_bwd(f, const_avals, *args)` — [`L347`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L347)
- `_run_scoped_rule(ctx: Context, *args, jaxpr, **params)` — [`L417`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L417)
- `_run_state_rule(ctx: Context, *args, jaxpr, which_linear, is_initialized)` — [`L370`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L370)
- `_scan_rule(ctx: Context, *args, jaxpr, **params)` — [`L430`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L430)
- `_swap_rule(ctx: Context, ref, val, *args, tree)` — [`L506`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L506)
- `_unpack_dtype_eval_rule(ctx: block_spec.KernelEvalContext, *args)` — [`L565`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L565)
- `_unpack_dtype_pull_rule(ctx: block_spec.PushRuleContext, block_specs: pallas_core.BlockSpec)` — [`L554`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L554)
- `_unpack_dtype_push_rule(ctx: block_spec.PushRuleContext, block_spec: pallas_core.BlockSpec)` — [`L543`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L543)
- `_unpack_rule(_, arg)` — [`L499`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L499)
- `_while_rule(ctx: Context, *args, body_jaxpr, cond_jaxpr, body_nconsts, cond_nconsts, **params)` — [`L440`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L440)
- `pack(*xs, dtype)` — [`L63`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L63)
- `pack_dtype_abstract_eval(*xs, dtype)` — [`L57`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L57)
- `physicalize(f)` — [`L135`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L135) — Runs a function that contains fusible extended dtypes.
- `physicalize_closed_jaxpr(jaxpr: core.ClosedJaxpr)` — [`L158`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L158) — Replaces all extended dtypes with physical types in a jaxpr.
- `physicalize_interp(jaxpr: core.Jaxpr, consts: Sequence[core.Value], *args: core.Value)` — [`L213`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L213) — Physicalizes a jaxpr by replacing fusible dtypes with physical types.
- `physicalize_jaxpr(jaxpr: core.Jaxpr)` — [`L185`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L185) — Replaces all extended dtypes with physical types in a jaxpr.
- `read_env(var: core.Atom)` — [`L220`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L220)
- `unpack(x)` — [`L80`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L80)
- `unpack_dtype_abstract_eval(x)` — [`L72`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L72)
- `wrapper(*args, **kwargs)` — [`L138`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L138)
- `write_env(var: core.Var, val: Any)` — [`L225`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L225)

## Module values
- `T` — [`L49`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L49)
- `_physicalize_rules` — [`L51`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L51)
- `pack_dtype_p` — [`L53`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L53)
- `unpack_dtype_p` — [`L67`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L67)
- `unsafe_map` — [`L46`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L46)
- `unsafe_zip` — [`L47`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible_dtype.py#L47)

