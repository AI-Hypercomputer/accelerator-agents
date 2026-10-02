---
title: 'Module: jax/_src/ad_checkpoint.py'
type: catalog
provenance: extracted
module: jax/_src/ad_checkpoint.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.ad_checkpoint`/
symbols:
  _insert_reduce_precision: _insert_reduce_precision().
  remat_partial_eval: remat_partial_eval().
  _saved_residuals: _saved_residuals().
  remat_p: remat_p.
  checkpoint.fun_remat: checkpoint().fun_remat().
  remat_dce: remat_dce().
  RematTraced.vjp_fwd: RematTraced#vjp_fwd().
  checkpoint_policies: checkpoint_policies.
  _transpose_jaxpr.transposed: _transpose_jaxpr().transposed().
  RematTraced.lin: RematTraced#lin().
  saved_residuals: saved_residuals().
  _remat_state_discharge_rule: _remat_state_discharge_rule().
  CheckpointName.remat: CheckpointName#remat().
  remat_jvp: remat_jvp().
  remat_transpose: remat_transpose().
  _trace_to_jaxpr: _trace_to_jaxpr().
  _transpose_jaxpr: _transpose_jaxpr().
  name_p: name_p.
  zip: zip.
  checkpoint_name: checkpoint_name().
  dce: dce().
  _dyn_args_fun: _dyn_args_fun().
  _remat_lowering: _remat_lowering().
  _remat_to_lojax: _remat_to_lojax().
  RematTraced.jvp: RematTraced#jvp().
  CustomRemat.jvp: CustomRemat#jvp().
  checkpoint: checkpoint().
  _saved_residuals.get_name: _saved_residuals().get_name().
  transpose_jaxpr: transpose_jaxpr().
  remat_vmap: remat_vmap().
  RematTraced.__init__: RematTraced#__init__().
  RematTraced.batch: RematTraced#batch().
  CheckpointName.__init__: CheckpointName#__init__().
  PrimalLeftTangentRight.__init__: PrimalLeftTangentRight#__init__().
  CustomRemat.__init__: CustomRemat#__init__().
  CustomRemat.remat: CustomRemat#remat().
  _remat3: _remat3().
  primal_left_tangent_right: primal_left_tangent_right().
  _remat_bind: _remat_bind().
  _dced: _dced().
  RematTraced.jaxpr: RematTraced#jaxpr.
  custom_remat.call: custom_remat().call().
  CustomRemat.rem_flat: CustomRemat#rem_flat().
  CheckpointName: CheckpointName#
  map: map.
  DotsSaveable.__call__: DotsSaveable#__call__().
  WrapHashably.__init__: WrapHashably#__init__().
  WrapHashably.__eq__: WrapHashably#__eq__().
  remat_impl: remat_impl().
  remat_abstract_eval: remat_abstract_eval().
  RematTraced.linearized: RematTraced#linearized().
  CheckpointName.jvp: CheckpointName#jvp().
  CheckpointName.vjp_fwd: CheckpointName#vjp_fwd().
  CheckpointName.lin: CheckpointName#lin().
  custom_remat: custom_remat().
  WrapHashably.val: WrapHashably#val.
  _dyn_args_fun_uncached: _dyn_args_fun_uncached().
  CheckpointName.name: CheckpointName#name.
  _remat_static_argnums: _remat_static_argnums().
  _dyn_args_fun_cached: _dyn_args_fun_cached.
  _has_effects: _has_effects().
  name_jvp: name_jvp().
  name_batcher: name_batcher().
  checkpoint_name3: checkpoint_name3().
  RematTraced: RematTraced#
  nothing_saveable: nothing_saveable().
  SaveOnlyTheseNames.__call__: SaveOnlyTheseNames#__call__().
  SaveAnyNamesButThese.__call__: SaveAnyNamesButThese#__call__().
  print_saved_residuals: print_saved_residuals().
  RematTraced.policy: RematTraced#policy.
  RematTraced.expand: RematTraced#expand().
  CustomRemat.expand: CustomRemat#expand().
  dots_saveable: dots_saveable.
  dots_with_no_batch_dims_saveable: dots_with_no_batch_dims_saveable.
  CustomRemat.jaxpr: CustomRemat#jaxpr.
  DotsSaveable: DotsSaveable#
  WrapHashably.hashable: WrapHashably#hashable.
  WrapHashably: WrapHashably#
  offload_dot_with_no_batch_dims: offload_dot_with_no_batch_dims().
  offload_dot_with_no_batch_dims.policy: offload_dot_with_no_batch_dims().policy().
  save_any_names_but_these: save_any_names_but_these().
  save_only_these_names: save_only_these_names().
  save_and_offload_only_these_names: save_and_offload_only_these_names().
  save_and_offload_only_these_names.policy: save_and_offload_only_these_names().policy().
  save_from_both_policies: save_from_both_policies().
  remat: remat().
  _dyn_args_fun_uncached.new_fun: _dyn_args_fun_uncached().new_fun().
  saved_residuals.f_: saved_residuals().f_().
  remat_partial_eval_custom_params_updater: remat_partial_eval_custom_params_updater().
  remat3: remat3().
  PrimalLeftTangentRight: PrimalLeftTangentRight#
  CustomRemat: CustomRemat#
  WrapHashably.hash: WrapHashably#hash.
  SaveAnyNamesButThese: SaveAnyNamesButThese#
  WrapHashably.__hash__: WrapHashably#__hash__().
  RematTraced.vjp_bwd: RematTraced#vjp_bwd().
  logger: logger.
  everything_saveable: everything_saveable().
  SaveOnlyTheseNames: SaveOnlyTheseNames#
  SaveOnlyTheseNames.saveable_names: SaveOnlyTheseNames#saveable_names.
  SaveAnyNamesButThese.names: SaveAnyNamesButThese#names.
  DotsSaveable.only_if_no_batch_dims: DotsSaveable#only_if_no_batch_dims.
  save_anything_except_these_names: save_anything_except_these_names().
  save_from_both_policies.policy: save_from_both_policies().policy().
  _remat_is_high: _remat_is_high().
  CustomRemat.f1: CustomRemat#f1.
  CustomRemat.f2_fbwd: CustomRemat#f2_fbwd.
  identity: identity().
  checkpoint_dots: checkpoint_dots.
  CheckpointName.expand: CheckpointName#expand().
  CheckpointName.vjp_bwd_retval: CheckpointName#vjp_bwd_retval().
  CheckpointName.linearized: CheckpointName#linearized().
  PrimalLeftTangentRight.expand: PrimalLeftTangentRight#expand().
  PrimalLeftTangentRight.lin: PrimalLeftTangentRight#lin().
  PrimalLeftTangentRight.linearized: PrimalLeftTangentRight#linearized().
  PrimalLeftTangentRight.vjp_fwd: PrimalLeftTangentRight#vjp_fwd().
  PrimalLeftTangentRight.vjp_bwd_retval: PrimalLeftTangentRight#vjp_bwd_retval().
  PrimalLeftTangentRight.jvp: PrimalLeftTangentRight#jvp().
  PrimalLeftTangentRight.batch: PrimalLeftTangentRight#batch().
  CustomRemat.lin: CustomRemat#lin().
  CustomRemat.linearized: CustomRemat#linearized().
  CustomRemat.vjp_fwd: CustomRemat#vjp_fwd().
  CustomRemat.vjp_bwd: CustomRemat#vjp_bwd().
---
# Module: [`jax/_src/ad_checkpoint.py`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py)

## Classes
### `CheckpointName`
- def: [`jax/_src/ad_checkpoint.py:1080`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1080)
- members:
  - `expand(self, x)` — [`L1089`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1089)
  - `jvp(self, primals, tangents)` — [`L1104`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1104)
  - `lin(self, nzs_in, x)` — [`L1114`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1114)
  - `linearized(self, _, g)` — [`L1117`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1117)
  - `remat(self, policy, x)` — [`L1092`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1092)
  - `vjp_bwd_retval(self, _, g)` — [`L1111`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1111)
  - `vjp_fwd(self, _nzs_in, x)` — [`L1108`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1108)
  - `name` — [`L1081`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1081)
- protocol/private: `__init__`[`L1083`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1083)
- uses (calls/refs, reference-scoped): [`VJPHiPrimitive`](hijax.md#VJPHiPrimitive), [`in_avals`](hijax.md#VJPHiPrimitive.in_avals), [`__init__`](hijax.md#VJPHiPrimitive.__init__), [`out_aval`](hijax.md#VJPHiPrimitive.out_aval), [`params`](hijax.md#VJPHiPrimitive.params), [`primal_left_tangent_right`](ad_checkpoint.md#primal_left_tangent_right), [`SaveAnyNamesButThese`](ad_checkpoint.md#SaveAnyNamesButThese), [`SaveOnlyTheseNames`](ad_checkpoint.md#SaveOnlyTheseNames), [`everything_saveable`](ad_checkpoint.md#everything_saveable), [`names`](ad_checkpoint.md#SaveAnyNamesButThese.names), [`saveable_names`](ad_checkpoint.md#SaveOnlyTheseNames.saveable_names)
- used by: [`expand`](hijax.md#VJPHiPrimitive.expand), [`jvp`](hijax.md#VJPHiPrimitive.jvp), [`vjp_fwd`](hijax.md#VJPHiPrimitive.vjp_fwd), [`vjp_bwd_retval`](hijax.md#VJPHiPrimitive.vjp_bwd_retval), [`get_name`](ad_checkpoint.md#_saved_residuals.get_name), [`lin`](hijax.md#VJPHiPrimitive.lin), [`linearized`](hijax.md#VJPHiPrimitive.linearized), [`remat`](hijax.md#VJPHiPrimitive.remat), [`checkpoint_name3`](ad_checkpoint.md#checkpoint_name3)

### `CustomRemat`
- def: [`jax/_src/ad_checkpoint.py:1169`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1169)
- members:
  - `expand(self, *args)` — [`L1181`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1181)
  - `jvp(self, primals, tangents)` — [`L1194`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1194)
  - `lin(self, nzs_in, *primals)` — [`L1199`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1199)
  - `linearized(self, res, *tangents)` — [`L1202`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1202)
  - `rem_flat(*args_flat)` — [`L1188`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1188)
  - `remat(self, policy, *args_flat)` — [`L1184`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1184)
  - `vjp_bwd(self, res, ybar)` — [`L1208`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1208)
  - `vjp_fwd(self, in_nzs, *args_flat)` — [`L1205`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1205)
  - `f1` — [`L1171`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1171)
  - `f2_fbwd` — [`L1172`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1172)
  - `jaxpr` — [`L1170`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1170)
- protocol/private: `__init__`[`L1174`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1174)
- uses (calls/refs, reference-scoped): [`tree_unflatten`](tree_util.md#tree_unflatten), [`ClosedJaxpr`](core.md#ClosedJaxpr), [`jaxpr_as_fun`](core.md#jaxpr_as_fun), [`tree_leaves_checked`](tree_util.md#tree_leaves_checked), [`VJPHiPrimitive`](hijax.md#VJPHiPrimitive), [`in_avals`](hijax.md#VJPHiPrimitive.in_avals), [`instantiate`](ad_util.md#instantiate), [`__init__`](hijax.md#VJPHiPrimitive.__init__), [`out_aval`](hijax.md#VJPHiPrimitive.out_aval), [`jvp`](api.md#jvp), [`params`](hijax.md#VJPHiPrimitive.params), [`map`](ad_checkpoint.md#map)
- used by: [`expand`](hijax.md#VJPHiPrimitive.expand), [`jvp`](hijax.md#VJPHiPrimitive.jvp), [`vjp_fwd`](hijax.md#VJPHiPrimitive.vjp_fwd), [`vjp_bwd`](hijax.md#VJPHiPrimitive.vjp_bwd), [`lin`](hijax.md#VJPHiPrimitive.lin), [`linearized`](hijax.md#VJPHiPrimitive.linearized), [`call`](ad_checkpoint.md#custom_remat.call), [`remat`](hijax.md#VJPHiPrimitive.remat)

### `DotsSaveable`
- def: [`jax/_src/ad_checkpoint.py:80`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L80)
- signature: `class DotsSaveable:`
- members:
  - `only_if_no_batch_dims` — [`L81`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L81)
- protocol/private: `__call__`[`L82`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L82)
- uses (calls/refs, reference-scoped): [`dot_general_p`](lax/lax.md#dot_general_p), [`conv_general_dilated_p`](lax/convolution.md#conv_general_dilated_p)
- used by: [`_dot_general_remat`](lax/lax.md#_dot_general_remat), [`dots_saveable`](ad_checkpoint.md#dots_saveable), [`dots_with_no_batch_dims_saveable`](ad_checkpoint.md#dots_with_no_batch_dims_saveable)

### `PrimalLeftTangentRight`  ·  implements/extends VJPHiPrimitive
- def: [`jax/_src/ad_checkpoint.py:1120`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1120)
- signature: `class PrimalLeftTangentRight(VJPHiPrimitive):`
- members:
  - `batch(self, axis_data, args, dims)` — [`L1145`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1145)
  - `expand(self, x, _x)` — [`L1127`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1127)
  - `jvp(self, primals, tangents)` — [`L1142`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1142)
  - `lin(self, nzs_in, x, _x)` — [`L1130`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1130)
  - `linearized(self, _, xdot, _xdot)` — [`L1133`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1133)
  - `vjp_bwd_retval(self, _, g)` — [`L1139`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1139)
  - `vjp_fwd(self, nzs_in, x, _x)` — [`L1136`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1136)
- protocol/private: `__init__`[`L1121`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1121)
- uses (calls/refs, reference-scoped): [`VJPHiPrimitive`](hijax.md#VJPHiPrimitive), [`in_avals`](hijax.md#VJPHiPrimitive.in_avals), [`__init__`](hijax.md#VJPHiPrimitive.__init__), [`out_aval`](hijax.md#VJPHiPrimitive.out_aval), [`params`](hijax.md#VJPHiPrimitive.params)
- used by: [`expand`](hijax.md#VJPHiPrimitive.expand), [`VJPHiPrimitive`](hijax.md#VJPHiPrimitive), [`jvp`](hijax.md#VJPHiPrimitive.jvp), [`vjp_fwd`](hijax.md#VJPHiPrimitive.vjp_fwd), [`vjp_bwd_retval`](hijax.md#VJPHiPrimitive.vjp_bwd_retval), [`batch`](hijax.md#VJPHiPrimitive.batch), [`lin`](hijax.md#VJPHiPrimitive.lin), [`linearized`](hijax.md#VJPHiPrimitive.linearized), [`primal_left_tangent_right`](ad_checkpoint.md#primal_left_tangent_right)

### `RematTraced`
- def: [`jax/_src/ad_checkpoint.py:1023`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1023)
- members:
  - `batch(self, axis_data, args, dims)` — [`L1073`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1073)
  - `expand(self, *args)` — [`L1033`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1033)
  - `jvp(self, primals, tangents)` — [`L1054`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1054)
  - `lin(self, nzs_in, *primals)` — [`L1059`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1059)
  - `linearized(self, primals_rem, *tangents)` — [`L1067`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1067)
  - `vjp_bwd(self, primals_rem, outgrad, *arg_accums)` — [`L1049`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1049)
  - `vjp_fwd(self, _nzs_in, *primals)` — [`L1037`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1037)
  - `jaxpr` — [`L1024`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1024)
  - `policy` — [`L1025`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1025)
- protocol/private: `__init__`[`L1027`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1027)
- uses (calls/refs, reference-scoped): [`jit`](api.md#jit), [`ClosedJaxpr`](core.md#ClosedJaxpr), [`tree_leaves`](tree_util.md#tree_leaves), [`jaxpr_as_fun`](core.md#jaxpr_as_fun), [`remat_transform`](interpreters/remat.md#remat_transform), [`float0`](dtypes.md#float0.float0), [`VJPHiPrimitive`](hijax.md#VJPHiPrimitive), [`in_avals`](hijax.md#VJPHiPrimitive.in_avals), [`instantiate`](ad_util.md#instantiate), [`optimization_barrier`](lax/lax.md#optimization_barrier), [`vjp`](api.md#vjp), [`__init__`](hijax.md#VJPHiPrimitive.__init__), [`out_aval`](hijax.md#VJPHiPrimitive.out_aval), [`dce`](ad_checkpoint.md#dce), [`zip`](ad_checkpoint.md#zip), [`jvp`](api.md#jvp), [`params`](hijax.md#VJPHiPrimitive.params), [`trace`](pjit.md#JitWrapped.trace), [`map`](ad_checkpoint.md#map), [`linearize`](api.md#linearize), [`outvars`](core.md#ClosedJaxpr.outvars)
- used by: [`expand`](hijax.md#VJPHiPrimitive.expand), [`jvp`](hijax.md#VJPHiPrimitive.jvp), [`vjp_fwd`](hijax.md#VJPHiPrimitive.vjp_fwd), [`vjp_bwd`](hijax.md#VJPHiPrimitive.vjp_bwd), [`batch`](hijax.md#VJPHiPrimitive.batch), [`lin`](hijax.md#VJPHiPrimitive.lin), [`linearized`](hijax.md#VJPHiPrimitive.linearized), [`_remat3`](ad_checkpoint.md#_remat3), [`_dced`](ad_checkpoint.md#_dced)

### `SaveAnyNamesButThese`
- def: [`jax/_src/ad_checkpoint.py:135`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L135)
- members:
  - `names` — [`L136`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L136)
- protocol/private: `__call__`[`L137`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L137)
- uses (calls/refs, reference-scoped): [`name_p`](ad_checkpoint.md#name_p)
- used by: [`remat`](ad_checkpoint.md#CheckpointName.remat), [`save_any_names_but_these`](ad_checkpoint.md#save_any_names_but_these)

### `SaveOnlyTheseNames`
- def: [`jax/_src/ad_checkpoint.py:127`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L127)
- signature: `class SaveOnlyTheseNames:`
- members:
  - `saveable_names` — [`L128`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L128)
- protocol/private: `__call__`[`L129`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L129)
- uses (calls/refs, reference-scoped): [`name_p`](ad_checkpoint.md#name_p)
- used by: [`remat`](ad_checkpoint.md#CheckpointName.remat), [`save_only_these_names`](ad_checkpoint.md#save_only_these_names)

### `WrapHashably`
- def: [`jax/_src/ad_checkpoint.py:423`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L423)
- members:
  - `hash` — [`L425`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L425)
  - `hashable` — [`L426`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L426)
  - `val` — [`L424`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L424)
- protocol/private: `__eq__`[`L438`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L438), `__hash__`[`L436`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L436), `__init__`[`L428`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L428)
- used by: [`_dyn_args_fun`](ad_checkpoint.md#_dyn_args_fun), [`_dyn_args_fun_uncached`](ad_checkpoint.md#_dyn_args_fun_uncached), [`_remat_static_argnums`](ad_checkpoint.md#_remat_static_argnums), [`new_fun`](ad_checkpoint.md#_dyn_args_fun_uncached.new_fun)

## Functions
- `_dced(jaxpr, out_tree, policy, res, *args)` — [`L1019`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1019)
- `_dyn_args_fun(fun: Callable, static_argnums: frozenset[int], static_args: tuple[WrapHashably, ...], nargs: int)` — [`L450`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L450)
- `_dyn_args_fun_uncached(fun: Callable, static_argnums: frozenset[int], static_args: tuple[WrapHashably, ...], nargs: int)` — [`L456`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L456)
- `_has_effects(effects)` — [`L854`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L854)
- `_insert_reduce_precision(jaxpr: core.Jaxpr, num_res: int)` — [`L693`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L693)
- `_remat3(policy, static_argnums, static_argnames, f, *args, **kwargs)` — [`L999`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L999)
- `_remat_bind(*args, jaxpr, prevent_cse, differentiated, policy)` — [`L581`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L581)
- `_remat_is_high(*_, jaxpr, **__)` — [`L891`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L891)
- `_remat_lowering(ctx: mlir.LoweringRuleContext, *args, jaxpr: core.Jaxpr, prevent_cse: bool, differentiated: bool, policy)` — [`L859`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L859)
- `_remat_state_discharge_rule(in_avals, out_avals, *args, jaxpr, **params)` — [`L969`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L969)
- `_remat_static_argnums(fun, static_argnums, args)` — [`L397`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L397)
- `_remat_to_lojax(*hi_args, jaxpr, **kwds)` — [`L896`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L896)
- `_saved_residuals(jaxpr: core.Jaxpr, arg_names: Sequence[str])` — [`L516`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L516)
- `_trace_to_jaxpr(fun: Callable, in_tree: PyTreeDef, in_avals: Sequence[core.AbstractValue], debug: core.DebugInfo)` — [`L470`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L470)
- `_transpose_jaxpr(jaxpr: core.ClosedJaxpr, in_lin: Sequence[bool], out_zeros: Sequence[bool])` — [`L778`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L778)
- `call(*args, **kwargs)` — [`L1156`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1156)
- `checkpoint(fun: Callable, *, prevent_cse: bool | Sequence[bool] = True, policy: Callable[..., bool] | None = None, static_argnums: int | tuple[int, ...] = (), static_argnames: str | Iterable[str] = ())` — [`L206`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L206) — Make ``fun`` recompute internal linearization points when differentiated.
- `checkpoint_name(x, name)` — [`L907`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L907) — Identifies a value with a name within :func:`jax.checkpoint`.
- `checkpoint_name3(name, x)` — [`L990`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L990)
- `custom_remat(f, f1, f2, fbwd, *, static_argnums=(), static_argnames=())` — [`L1153`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1153)
- `dce(traced, policy)` — [`L1011`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1011)
- `everything_saveable(*_, **__)` — [`L67`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L67) — The default strategy, as if ``jax.checkpoint`` were not being used at all.
- `f_(*args)` — [`L498`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L498)
- `fun_remat(*args, **kwargs)` — [`L365`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L365)
- `get_name(eqn)` — [`L543`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L543)
- `identity(x)` — [`L61`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L61)
- `name_batcher(args, dims, *, name)` — [`L962`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L962)
- `name_jvp(primals, tangents, *, name)` — [`L955`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L955)
- `new_fun(*dyn_args, **kwargs)` — [`L458`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L458)
- `nothing_saveable(*_, **__)` — [`L73`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L73) — Rematerialize everything, as if a custom policy were not being used at all.
- `offload_dot_with_no_batch_dims(offload_src, offload_dst)` — [`L99`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L99) — Same as ``dots_with_no_batch_dims_saveable``, but offload to CPU memory
- `policy(prim, *_, **params)` — [`L104`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L104)
- `policy(prim, *_, **params)` — [`L161`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L161)
- `policy(prim, *args, **params)` — [`L174`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L174)
- `primal_left_tangent_right(x, _x)` — [`L1148`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L1148)
- `print_saved_residuals(f, *args, **kwargs)` — [`L571`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L571)
- `remat(fun: Callable, *, prevent_cse: bool = True, policy: Callable[..., bool] | None = None, static_argnums: int | tuple[int, ...] = ())` — [`L386`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L386) — Alias of :func:`jax.checkpoint`.
- `remat3(f=None,, policy=None, static_argnums=(), static_argnames=())` — [`L993`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L993)
- `remat_abstract_eval(*args, jaxpr, prevent_cse, differentiated, policy)` — [`L593`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L593)
- `remat_dce(used_outputs: list[bool], eqn: core.JaxprEqn)` — [`L832`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L832)
- `remat_impl(*args, jaxpr, prevent_cse, differentiated, policy)` — [`L588`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L588)
- `remat_jvp(primals, tangents, jaxpr, prevent_cse, differentiated, policy)` — [`L597`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L597)
- `remat_partial_eval(trace: pe.JaxprTrace, *tracers: core.Tracer, jaxpr: core.Jaxpr, prevent_cse, **params)` — [`L615`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L615)
- `remat_partial_eval_custom_params_updater(*args)` — [`L729`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L729)
- `remat_transpose(out_cts, *args, jaxpr, prevent_cse, **params)` — [`L743`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L743)
- `remat_vmap(axis_data, args, dims, *, jaxpr, **params)` — [`L819`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L819)
- `save_and_offload_only_these_names(*, names_which_can_be_saved, names_which_can_be_offloaded, offload_src, offload_dst)` — [`L146`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L146) — Same as ``save_only_these_names``, but offload to CPU memory instead of
- `save_any_names_but_these(*names_not_to_save)` — [`L121`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L121) — Save only named values, i.e. any outputs of `checkpoint_name`, excluding
- `save_anything_except_these_names(*names_not_to_save)` — [`L117`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L117) — Save any values (not just named ones) excluding the names given.
- `save_from_both_policies(policy_1, policy_2)` — [`L170`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L170) — Logical OR of the given policies.
- `save_only_these_names(*names_which_can_be_saved)` — [`L142`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L142) — Save only named values, and only among the names given.
- `saved_residuals(f: Callable, *args, **kwargs)` — [`L494`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L494)
- `transpose_jaxpr(jaxpr: core.ClosedJaxpr, in_linear: bool | Sequence[bool], out_zeros: bool | Sequence[bool])` — [`L768`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L768)
- `transposed(*args_flat)` — [`L786`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L786)

## Module values
- `_dyn_args_fun_cached` — [`L465`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L465)
- `checkpoint_dots` — [`L95`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L95)
- `checkpoint_policies` — [`L188`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L188)
- `dots_saveable` — [`L95`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L95)
- `dots_with_no_batch_dims_saveable` — [`L96`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L96)
- `logger` — [`L63`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L63)
- `map` — [`L59`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L59)
- `name_p` — [`L113`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L113)
- `remat_p` — [`L578`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L578)
- `zip` — [`L60`](../../../../../../raw/code/jax/jax/_src/ad_checkpoint.py#L60)

