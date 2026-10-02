---
title: 'Module: jax/_src/pjit.py'
type: catalog
provenance: extracted
module: jax/_src/pjit.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pjit`/
symbols:
  _trace_for_jit: _trace_for_jit().
  _parse_jit_arguments: _parse_jit_arguments().
  _cpp_pjit: _cpp_pjit().
  jit_p: jit_p.
  with_sharding_constraint: with_sharding_constraint().
  _pjit_lowering: _pjit_lowering().
  reshard: reshard().
  _get_new_mesh: _get_new_mesh().
  _infer_params: _infer_params().
  _pjit_partial_eval: _pjit_partial_eval().
  _resolve_in_shardings: _resolve_in_shardings().
  _sharding_constraint_impl: _sharding_constraint_impl().
  _run_python_pjit: _run_python_pjit().
  _sharding_constraint_hlo_lowering: _sharding_constraint_hlo_lowering().
  _pjit_batcher_for_sharding: _pjit_batcher_for_sharding().
  _pjit_linearize: _pjit_linearize().
  _pjit_transpose_fancy: _pjit_transpose_fancy().
  _auto_axes.decorator: _auto_axes().decorator().
  _pjit_lower_jaxpr_to_fun: _pjit_lower_jaxpr_to_fun().
  dce_jaxpr_pjit_rule: dce_jaxpr_pjit_rule().
  reshard_p: reshard_p.
  _pjit_call_impl_python: _pjit_call_impl_python().
  _pjit_pp_rule: _pjit_pp_rule().
  _explicit_axes.decorator: _explicit_axes().decorator().
  check_shardings_are_auto: check_shardings_are_auto().
  sharding_constraint_p: sharding_constraint_p.
  _sharding_constraint_batcher: _sharding_constraint_batcher().
  _get_fastpath_data: _get_fastpath_data().
  get_ctx_mesh: get_ctx_mesh().
  _process_in_axis_resources: _process_in_axis_resources().
  _check_and_canonicalize_out_shardings: _check_and_canonicalize_out_shardings().
  pjit_staging_rule: pjit_staging_rule().
  assert_shardings_equal: assert_shardings_equal().
  jit_trace: jit_trace().
  with_layout_constraint: with_layout_constraint().
  layout_constraint_p: layout_constraint_p.
  _cpp_pjit.cache_miss: _cpp_pjit().cache_miss().
  _resolve_and_lower: _resolve_and_lower().
  const_args_layouts: const_args_layouts().
  _pjit_call_impl.call_impl_cache_miss: _pjit_call_impl().call_impl_cache_miss().
  _pjit_batcher: _pjit_batcher().
  _pjit_partial_eval.keep_where: _pjit_partial_eval().keep_where().
  _split_layout_and_sharding: _split_layout_and_sharding().
  _create_sharding_for_array: _create_sharding_for_array().
  auto_axes: auto_axes().
  convert_to_metaty: convert_to_metaty().
  pjit_check_aval_sharding: pjit_check_aval_sharding().
  _infer_input_type: _infer_input_type().
  _to_lojax: _to_lojax().
  _pjit_call_impl: _pjit_call_impl().
  const_args_shardings: const_args_shardings().
  _pjit_jvp: _pjit_jvp().
  _pjit_remat: _pjit_remat().
  _infer_params_cached: _infer_params_cached().
  _resolve_in_layouts: _resolve_in_layouts().
  pjit: pjit().
  finalize_arg_sharding: finalize_arg_sharding().
  _pjit_linearize.tangent_fun: _pjit_linearize().tangent_fun().
  _dce_jaxpr_pjit: _dce_jaxpr_pjit().
  _reshard_jvp_rule: _reshard_jvp_rule().
  _reshard_transpose_fancy: _reshard_transpose_fancy().
  get_unconstrained_dims: get_unconstrained_dims().
  make_jit: make_jit().
  PjitSharding: PjitSharding.
  hashable_pytree: hashable_pytree().
  jit_evict_fn: jit_evict_fn().
  _transpose_jaxpr_fancy: _transpose_jaxpr_fancy().
  _sharding_constraint_abstract_eval: _sharding_constraint_abstract_eval().
  _reshard_abstract_eval: _reshard_abstract_eval().
  _reshard_linearize: _reshard_linearize().
  _reshard_linearize.linearized: _reshard_linearize().linearized().
  _layout_constraint_impl: _layout_constraint_impl().
  _layout_constraint_batcher: _layout_constraint_batcher().
  JitWrapped.trace: JitWrapped#trace().
  _get_cpp_global_cache: _get_cpp_global_cache().
  create_meta_ty: create_meta_ty().
  PjitParams.consts: PjitParams#consts.
  InferParamsCacheEntry.pjit_params: InferParamsCacheEntry#pjit_params.
  check_aval_layout_compatibility: check_aval_layout_compatibility().
  MeshInfo.new: MeshInfo#new.
  PjitInfo: PjitInfo#
  jit_lower: jit_lower().
  jit_eval_shape: jit_eval_shape().
  _create_sharding_with_device_backend: _create_sharding_with_device_backend().
  _qdd_cache_index: _qdd_cache_index().
  _lojax_expand_params: _lojax_expand_params().
  _resolve_out_layouts: _resolve_out_layouts().
  _get_jaxpr_as_fun: _get_jaxpr_as_fun().
  _pjit_lower: _pjit_lower().
  _pjit_typecheck: _pjit_typecheck().
  PjitLoweringResult.const_args_and_avals: PjitLoweringResult#const_args_and_avals.
  _pjit_linearize._filter_zeros: _pjit_linearize()._filter_zeros().
  _pjit_partial_eval_custom_params_updater: _pjit_partial_eval_custom_params_updater().
  _transpose_jaxpr_fancy.transposed: _transpose_jaxpr_fancy().transposed().
  _reshard_impl: _reshard_impl().
  _reshard_batcher: _reshard_batcher().
  PjitInfo.in_shardings_treedef: PjitInfo#in_shardings_treedef.
  PjitInfo.out_shardings_treedef: PjitInfo#out_shardings_treedef.
  PjitInfo.in_layouts_treedef: PjitInfo#in_layouts_treedef.
  PjitInfo.out_layouts_treedef: PjitInfo#out_layouts_treedef.
  _cpp_pjit_cache_fun_only: _cpp_pjit_cache_fun_only.
  _cpp_pjit_cache_explicit_attributes: _cpp_pjit_cache_explicit_attributes.
  PjitInfo.static_argnums: PjitInfo#static_argnums.
  PjitInfo.static_argnames: PjitInfo#static_argnames.
  PjitInfo.device: PjitInfo#device.
  PjitInfo.backend: PjitInfo#backend.
  JitWrapped: JitWrapped#
  _lojax_expand_params.expand: _lojax_expand_params().expand().
  dce_jaxpr_pjit_rule.keep_where: dce_jaxpr_pjit_rule().keep_where().
  unsafe_zip: unsafe_zip.
  PjitInfo.inline: PjitInfo#inline.
  InferParamsCacheEntry: InferParamsCacheEntry#
  _seen_qdds_get: _seen_qdds_get().
  MeshInfo.prev: MeshInfo#prev.
  _auto_axes: _auto_axes().
  _explicit_axes: _explicit_axes().
  PjitInfo.donate_argnums: PjitInfo#donate_argnums.
  PjitInfo.donate_argnames: PjitInfo#donate_argnames.
  PjitInfo.use_resource_env: PjitInfo#use_resource_env.
  _pjit_linearize.keep_where: _pjit_linearize().keep_where().
  PjitParams: PjitParams#
  PjitParams.in_tree: PjitParams#in_tree.
  PjitParams.out_tree: PjitParams#out_tree.
  _qdd_cache_update: _qdd_cache_update().
  _converted_mutables_add_params: _converted_mutables_add_params().
  _pjit_abstract_eval: _pjit_abstract_eval().
  PjitLoweringResult.output_treedef: PjitLoweringResult#output_treedef.
  PjitLoweringResult.effects: PjitLoweringResult#effects.
  _add_res_to_params: _add_res_to_params().
  _pp_reshard: _pp_reshard().
  explicit_axes: explicit_axes().
  PjitInfo.fun_sourceinfo: PjitInfo#fun_sourceinfo.
  PjitInfo.in_shardings_leaves: PjitInfo#in_shardings_leaves.
  PjitInfo.out_shardings_leaves: PjitInfo#out_shardings_leaves.
  PjitInfo.in_layouts_leaves: PjitInfo#in_layouts_leaves.
  PjitInfo.out_layouts_leaves: PjitInfo#out_layouts_leaves.
  PjitInfo.compiler_options_kvs: PjitInfo#compiler_options_kvs.
  _need_to_rebuild_with_fdo: _need_to_rebuild_with_fdo().
  _identity_fn: _identity_fn().
  PjitParams.in_avals: PjitParams#in_avals.
  IgnoreKey.__eq__: IgnoreKey#__eq__().
  MetaTy.shape: MetaTy#shape().
  MetaTy.ndim: MetaTy#ndim().
  unsafe_map: unsafe_map.
  PjitInfo.fun_signature: PjitInfo#fun_signature.
  PjitInfo.user_specified_in_shardings: PjitInfo#user_specified_in_shardings.
  PjitInfo.keep_unused: PjitInfo#keep_unused.
  make_jit_cpp_cache: make_jit_cpp_cache().
  PjitParams.params: PjitParams#params.
  _pgle_profiler_dict: _pgle_profiler_dict.
  MetaTy.aval: MetaTy#aval.
  PjitLoweringResult: PjitLoweringResult#
  _pjit_jvp._filter_zeros: _pjit_jvp()._filter_zeros().
  MeshInfo: MeshInfo#
  MeshInfo.axes: MeshInfo#axes.
  _seen_qdds: _seen_qdds.
  IgnoreKey: IgnoreKey#
  _is_high: _is_high().
  MetaTy: MetaTy#
  pjit_forwarding_rule: pjit_forwarding_rule().
  PjitLoweringResult.flat_output_types: PjitLoweringResult#flat_output_types.
  PjitLoweringResult.symbol_ref: PjitLoweringResult#symbol_ref.
  PjitLoweringResult.wrapped_name: PjitLoweringResult#wrapped_name.
  _reshard_hlo_lowering: _reshard_hlo_lowering().
  _layout_constraint_hlo_lowering: _layout_constraint_hlo_lowering().
  PjitInfo.__hash__: PjitInfo#__hash__().
  PjitInfo.__eq__: PjitInfo#__eq__().
  PjitParams.arg_names: PjitParams#arg_names.
  JitWrapped.eval_shape: JitWrapped#eval_shape().
  IgnoreKey.val: IgnoreKey#val.
  IgnoreKey.__hash__: IgnoreKey#__hash__().
  MetaTy.sharding: MetaTy#sharding.
  MetaTy.format: MetaTy#format.
  MetaTy.committed: MetaTy#committed.
  MetaTy.is_np_array: MetaTy#is_np_array.
  MetaTy.replace: MetaTy#replace.
  PjitLoweringResult.func: PjitLoweringResult#func.
---
# Module: [`jax/_src/pjit.py`](../../../../../../raw/code/jax/jax/_src/pjit.py)

## Classes
### `IgnoreKey`
- def: [`jax/_src/pjit.py:833`](../../../../../../raw/code/jax/jax/_src/pjit.py#L833)
- signature: `class IgnoreKey:`
- members:
  - `val` — [`L834`](../../../../../../raw/code/jax/jax/_src/pjit.py#L834)
- protocol/private: `__eq__`[`L837`](../../../../../../raw/code/jax/jax/_src/pjit.py#L837), `__hash__`[`L835`](../../../../../../raw/code/jax/jax/_src/pjit.py#L835)

### `InferParamsCacheEntry`
- def: [`jax/_src/pjit.py:592`](../../../../../../raw/code/jax/jax/_src/pjit.py#L592)
- signature: `class InferParamsCacheEntry:`
- members:
  - `pjit_params` — [`L593`](../../../../../../raw/code/jax/jax/_src/pjit.py#L593)
- uses (calls/refs, reference-scoped): [`weakref_lru_cache`](util.md#weakref_lru_cache), [`PjitParams`](pjit.md#PjitParams)
- used by: [`_infer_params`](pjit.md#_infer_params), [`_infer_params_cached`](pjit.md#_infer_params_cached)

### `JitWrapped`  ·  implements/extends Wrapped
- def: [`jax/_src/pjit.py:673`](../../../../../../raw/code/jax/jax/_src/pjit.py#L673)
- signature: `class JitWrapped(stages.Wrapped):`
- members:
  - `eval_shape(self, *args, **kwargs)` — [`L675`](../../../../../../raw/code/jax/jax/_src/pjit.py#L675) — See ``jax.eval_shape``.
  - `trace(self, *args, **kwargs)` — [`L679`](../../../../../../raw/code/jax/jax/_src/pjit.py#L679)
- used by: [`jit`](api.md#jit), [`__call__`](hijax.md#custom_vjp3.__call__), [`jvp`](hijax.md#CustomVJPTraced.jvp), [`vjp_fwd`](ad_checkpoint.md#RematTraced.vjp_fwd), [`lin`](ad_checkpoint.md#RematTraced.lin), [`make_jaxpr_f`](api.md#make_jaxpr.make_jaxpr_f), [`_compile_fn`](pallas/mosaic_gpu/torch.md#_compile_fn), [`eval_shape`](api.md#eval_shape), [`pjit`](pjit.md#pjit), [`_run_jaxpr`](pallas/mosaic/interpret/thread_map.md#_run_jaxpr)

### `MeshInfo`
- def: [`jax/_src/pjit.py:2377`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2377)
- signature: `class MeshInfo:`
- members:
  - `axes` — [`L2380`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2380)
  - `new` — [`L2379`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2379)
  - `prev` — [`L2378`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2378)
- uses (calls/refs, reference-scoped): [`AbstractMesh`](mesh.md#AbstractMesh)
- used by: [`_get_new_mesh`](pjit.md#_get_new_mesh), [`decorator`](pjit.md#_auto_axes.decorator), [`decorator`](pjit.md#_explicit_axes.decorator)

### `MetaTy`
- def: [`jax/_src/pjit.py:1110`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1110)
- signature: `class MetaTy:`
- members:
  - `ndim(self)` — [`L1124`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1124)
  - `shape(self)` — [`L1120`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1120)
  - `aval` — [`L1111`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1111)
  - `committed` — [`L1114`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1114)
  - `format` — [`L1113`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1113)
  - `is_np_array` — [`L1115`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1115)
  - `replace` — [`L1117`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1117)
  - `sharding` — [`L1112`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1112)
- used by: [`create_meta_ty`](pjit.md#create_meta_ty)

### `PjitInfo`  ·  implements/extends NamedTuple
- def: [`jax/_src/pjit.py:94`](../../../../../../raw/code/jax/jax/_src/pjit.py#L94)
- doc: Things that we know about a jit instance before it is called.
- signature: `class PjitInfo(NamedTuple):`
- members:
  - `backend` — [`L118`](../../../../../../raw/code/jax/jax/_src/pjit.py#L118)
  - `compiler_options_kvs` — [`L122`](../../../../../../raw/code/jax/jax/_src/pjit.py#L122)
  - `device` — [`L117`](../../../../../../raw/code/jax/jax/_src/pjit.py#L117)
  - `donate_argnames` — [`L116`](../../../../../../raw/code/jax/jax/_src/pjit.py#L116)
  - `donate_argnums` — [`L115`](../../../../../../raw/code/jax/jax/_src/pjit.py#L115)
  - `fun_signature` — [`L101`](../../../../../../raw/code/jax/jax/_src/pjit.py#L101)
  - `fun_sourceinfo` — [`L100`](../../../../../../raw/code/jax/jax/_src/pjit.py#L100)
  - `in_layouts_leaves` — [`L110`](../../../../../../raw/code/jax/jax/_src/pjit.py#L110)
  - `in_layouts_treedef` — [`L109`](../../../../../../raw/code/jax/jax/_src/pjit.py#L109)
  - `in_shardings_leaves` — [`L106`](../../../../../../raw/code/jax/jax/_src/pjit.py#L106)
  - `in_shardings_treedef` — [`L105`](../../../../../../raw/code/jax/jax/_src/pjit.py#L105)
  - `inline` — [`L120`](../../../../../../raw/code/jax/jax/_src/pjit.py#L120)
  - `keep_unused` — [`L119`](../../../../../../raw/code/jax/jax/_src/pjit.py#L119)
  - `out_layouts_leaves` — [`L112`](../../../../../../raw/code/jax/jax/_src/pjit.py#L112)
  - `out_layouts_treedef` — [`L111`](../../../../../../raw/code/jax/jax/_src/pjit.py#L111)
  - `out_shardings_leaves` — [`L108`](../../../../../../raw/code/jax/jax/_src/pjit.py#L108)
  - `out_shardings_treedef` — [`L107`](../../../../../../raw/code/jax/jax/_src/pjit.py#L107)
  - `static_argnames` — [`L114`](../../../../../../raw/code/jax/jax/_src/pjit.py#L114)
  - `static_argnums` — [`L113`](../../../../../../raw/code/jax/jax/_src/pjit.py#L113)
  - `use_resource_env` — [`L121`](../../../../../../raw/code/jax/jax/_src/pjit.py#L121)
  - `user_specified_in_shardings` — [`L104`](../../../../../../raw/code/jax/jax/_src/pjit.py#L104)
- protocol/private: `__eq__`[`L128`](../../../../../../raw/code/jax/jax/_src/pjit.py#L128), `__hash__`[`L125`](../../../../../../raw/code/jax/jax/_src/pjit.py#L125)
- uses (calls/refs, reference-scoped): [`PyTreeDef`](tree_util.md#PyTreeDef), [`Inline`](api.md#Inline)
- used by: [`_trace_for_jit`](pjit.md#_trace_for_jit), [`_parse_jit_arguments`](pjit.md#_parse_jit_arguments), [`_cpp_pjit`](pjit.md#_cpp_pjit), [`_infer_params`](pjit.md#_infer_params), [`cache_miss`](pjit.md#_cpp_pjit.cache_miss), [`_infer_params_cached`](pjit.md#_infer_params_cached)

### `PjitLoweringResult`
- def: [`jax/_src/pjit.py:1342`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1342)
- signature: `class PjitLoweringResult:`
- members:
  - `const_args_and_avals` — [`L1346`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1346)
  - `effects` — [`L1347`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1347)
  - `flat_output_types` — [`L1344`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1344)
  - `func` — [`L1343`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1343)
  - `output_treedef` — [`L1345`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1345)
  - `symbol_ref` — [`L1348`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1348)
  - `wrapped_name` — [`L1349`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1349)
- uses (calls/refs, reference-scoped): [`ArrayLike`](basearray.md#ArrayLike), [`AbstractValue`](core.md#AbstractValue), [`PyTreeDef`](tree_util.md#PyTreeDef), [`Effect`](core.md#Effect)
- used by: [`_pjit_lowering`](pjit.md#_pjit_lowering), [`_pjit_lower_jaxpr_to_fun`](pjit.md#_pjit_lower_jaxpr_to_fun)

### `PjitParams`
- def: [`jax/_src/pjit.py:455`](../../../../../../raw/code/jax/jax/_src/pjit.py#L455)
- members:
  - `arg_names` — [`L465`](../../../../../../raw/code/jax/jax/_src/pjit.py#L465)
  - `consts` — [`L458`](../../../../../../raw/code/jax/jax/_src/pjit.py#L458)
  - `in_avals` — [`L462`](../../../../../../raw/code/jax/jax/_src/pjit.py#L462)
  - `in_tree` — [`L463`](../../../../../../raw/code/jax/jax/_src/pjit.py#L463)
  - `out_tree` — [`L464`](../../../../../../raw/code/jax/jax/_src/pjit.py#L464)
  - `params` — [`L461`](../../../../../../raw/code/jax/jax/_src/pjit.py#L461)
- uses (calls/refs, reference-scoped): [`ArrayLike`](basearray.md#ArrayLike), [`AbstractValue`](core.md#AbstractValue), [`PyTreeDef`](tree_util.md#PyTreeDef)
- used by: [`_trace_for_jit`](pjit.md#_trace_for_jit), [`_infer_params`](pjit.md#_infer_params), [`jit_trace`](pjit.md#jit_trace), [`InferParamsCacheEntry`](pjit.md#InferParamsCacheEntry)

## Functions
- `_add_res_to_params(num_res, in_shardings, out_shardings, in_layouts, out_layouts, donated_invars, **params)` — [`L1680`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1680)
- `_auto_axes(fun, *, axes_, out_sharding)` — [`L2433`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2433)
- `_check_and_canonicalize_out_shardings(out_shardings_treedef, out_shardings_leaves, out_layouts_treedef, out_layouts_leaves, out_tree, out_avals, debug_info: core.DebugInfo, device_or_backend_set)` — [`L779`](../../../../../../raw/code/jax/jax/_src/pjit.py#L779)
- `_converted_mutables_add_params(n, *, donated_invars, in_shardings, in_layouts, **params)` — [`L914`](../../../../../../raw/code/jax/jax/_src/pjit.py#L914)
- `_cpp_pjit(fun: Callable, jit_info: PjitInfo)` — [`L250`](../../../../../../raw/code/jax/jax/_src/pjit.py#L250)
- `_create_sharding_for_array(mesh, x, name, api_name)` — [`L716`](../../../../../../raw/code/jax/jax/_src/pjit.py#L716)
- `_create_sharding_with_device_backend(device, backend)` — [`L733`](../../../../../../raw/code/jax/jax/_src/pjit.py#L733)
- `_dce_jaxpr_pjit(jaxpr: core.ClosedJaxpr, used_outputs: tuple[bool, ...])` — [`L1942`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1942)
- `_explicit_axes(fun, *, axes, in_sharding)` — [`L2465`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2465)
- `_filter_zeros(is_nz_l, l)` — [`L1553`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1553)
- `_filter_zeros(is_nz_l, l)` — [`L1649`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1649)
- `_get_cpp_global_cache(contains_explicit_attributes: bool)` — [`L243`](../../../../../../raw/code/jax/jax/_src/pjit.py#L243)
- `_get_fastpath_data(executable, out_tree, args_flat, out_flat, effects, consts_for_constvars, pgle_profiler, const_args: Sequence[ArrayLike])` — [`L189`](../../../../../../raw/code/jax/jax/_src/pjit.py#L189)
- `_get_jaxpr_as_fun(jaxpr, in_shardings, out_shardings, in_layouts, out_layouts, donated_invars, ctx_mesh, name, keep_unused, inline, compiler_options_kvs)` — [`L1202`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1202)
- `_get_new_mesh(axes: str | tuple[str, ...] | None, axis_type: mesh_lib.AxisType, name: str, shardings=None)` — [`L2382`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2382)
- `_identity_fn(x)` — [`L2131`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2131)
- `_infer_input_type(fun: Callable, dbg_fn: Callable[[], core.DebugInfo], explicit_args)` — [`L642`](../../../../../../raw/code/jax/jax/_src/pjit.py#L642)
- `_infer_params(fun: Callable, ji: PjitInfo, args: tuple[Any, ...], kwargs: dict[str, Any])` — [`L618`](../../../../../../raw/code/jax/jax/_src/pjit.py#L618)
- `_infer_params_cached(fun: Callable, jit_info: PjitInfo, signature: jax_jit.ArgumentSignature, in_avals: tuple[core.AbstractValue, ...], ctx_mesh: mesh_lib.Mesh)` — [`L596`](../../../../../../raw/code/jax/jax/_src/pjit.py#L596)
- `_is_high(*_, jaxpr, **__)` — [`L888`](../../../../../../raw/code/jax/jax/_src/pjit.py#L888)
- `_layout_constraint_batcher(axis_data, vals_in, dims_in, layout)` — [`L2529`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2529)
- `_layout_constraint_hlo_lowering(ctx, x_node, *, layout)` — [`L2521`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2521)
- `_layout_constraint_impl(x, *, layout)` — [`L2511`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2511)
- `_lojax_expand_params(in_avals_, out_avals, donated_invars, in_shardings, in_layouts, out_shardings, out_layouts, **params)` — [`L923`](../../../../../../raw/code/jax/jax/_src/pjit.py#L923)
- `_need_to_rebuild_with_fdo(pgle_profiler)` — [`L185`](../../../../../../raw/code/jax/jax/_src/pjit.py#L185)
- `_parse_jit_arguments(fun: Callable, *, in_shardings: Any, out_shardings: Any, static_argnums: int | Sequence[int] | None, static_argnames: str | Iterable[str] | None, donate_argnums: int | Sequence[int] | None, donate_argnames: str | Iterable[str] | None, keep_unused: bool, device: xc.Device | None, backend: str | None, inline: bool | api.Inline, compiler_options: dict[str, Any] | None, use_resource_env: bool)` — [`L346`](../../../../../../raw/code/jax/jax/_src/pjit.py#L346) — Parses the arguments to jit/pjit.
- `_pjit_abstract_eval(*args, jaxpr, out_shardings, **_)` — [`L1336`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1336)
- `_pjit_batcher(axis_data, vals_in, dims_in: tuple[int, ...], jaxpr: core.ClosedJaxpr, in_shardings, out_shardings, in_layouts, out_layouts, donated_invars, ctx_mesh, name, keep_unused, inline, compiler_options_kvs)` — [`L1465`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1465)
- `_pjit_batcher_for_sharding(s, dim: int, spmd_axis_name: tuple[str, ...] | None, mesh, ndim: int)` — [`L1508`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1508)
- `_pjit_call_impl(*args, jaxpr: core.ClosedJaxpr, in_shardings, out_shardings, in_layouts, out_layouts, donated_invars, ctx_mesh, name, keep_unused, inline, compiler_options_kvs)` — [`L1215`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1215)
- `_pjit_call_impl_python(*args, jaxpr: core.ClosedJaxpr, in_shardings, out_shardings, in_layouts, out_layouts, donated_invars, ctx_mesh, name, keep_unused, inline, compiler_options_kvs)` — [`L1145`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1145)
- `_pjit_jvp(primals_in, tangents_in, jaxpr, in_shardings, out_shardings, in_layouts, out_layouts, donated_invars, ctx_mesh, name, keep_unused, inline, compiler_options_kvs)` — [`L1545`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1545)
- `_pjit_linearize(is_vjp, nzs, *primals_in, jaxpr, in_shardings, out_shardings, in_layouts, out_layouts, donated_invars, ctx_mesh, name, keep_unused, inline, compiler_options_kvs)` — [`L1579`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1579)
- `_pjit_lower(jaxpr: core.ClosedJaxpr, in_shardings, out_shardings, in_layouts: pxla.MaybeLayout, out_layouts: pxla.MaybeLayout, donated_invars, ctx_mesh, name: str, keep_unused: bool, inline: bool, compiler_options_kvs: tuple[tuple[str, Any], ...], *, lowering_platforms: tuple[str, ...] | None, lowering_parameters: mlir.LoweringParameters, pgle_profiler: profiler.PGLEProfiler | None)` — [`L1256`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1256)
- `_pjit_lower_jaxpr_to_fun(ctx: mlir.LoweringRuleContext, name: str, jaxpr: core.ClosedJaxpr, in_shardings, out_shardings, in_layouts, out_layouts)` — [`L1352`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1352)
- `_pjit_lowering(ctx: mlir.LoweringRuleContext, *args, name: str, jaxpr: core.ClosedJaxpr, in_shardings, out_shardings, in_layouts, out_layouts, donated_invars, ctx_mesh, keep_unused, inline, compiler_options_kvs)` — [`L1392`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1392)
- `_pjit_partial_eval(trace: pe.JaxprTrace, *in_tracers, jaxpr: core.ClosedJaxpr, in_shardings, out_shardings, in_layouts, out_layouts, donated_invars, ctx_mesh, name, keep_unused, inline, compiler_options_kvs)` — [`L1697`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1697)
- `_pjit_partial_eval_custom_params_updater(unks_in: Sequence[bool], inst_in: Sequence[bool], kept_outs_known: Sequence[bool], kept_outs_staged: Sequence[bool], num_res_out: int, num_res_in: int, params_known: dict, params_staged: dict)` — [`L1827`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1827)
- `_pjit_pp_rule(eqn: core.JaxprEqn, context: core.JaxprPpContext, settings: core.JaxprPpSettings)` — [`L1984`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1984)
- `_pjit_remat(policy, *args, jaxpr, **params)` — [`L1672`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1672)
- `_pjit_transpose_fancy(cts_in, *args, jaxpr, in_shardings, out_shardings, in_layouts, out_layouts, donated_invars, ctx_mesh, name, keep_unused, inline, compiler_options_kvs)` — [`L1878`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1878)
- `_pjit_typecheck(ctx_factory, *in_atoms, jaxpr, **params)` — [`L1330`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1330)
- `_pp_reshard(eqn, ctx, settings)` — [`L2370`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2370)
- `_process_in_axis_resources(in_shardings_treedef, in_shardings_leaves, in_layouts_treedef, in_layouts_leaves, in_avals, in_tree, dbg: core.DebugInfo, device_or_backend_set, kws)` — [`L748`](../../../../../../raw/code/jax/jax/_src/pjit.py#L748)
- `_qdd_cache_index(fun, in_type)` — [`L816`](../../../../../../raw/code/jax/jax/_src/pjit.py#L816)
- `_qdd_cache_update(fun, in_type, i, consts, aval_qdds)` — [`L825`](../../../../../../raw/code/jax/jax/_src/pjit.py#L825)
- `_reshard_abstract_eval(aval, *, dst_sharding, concrete_mesh)` — [`L2298`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2298)
- `_reshard_batcher(axis_data, vals_in, dims_in, dst_sharding, concrete_mesh)` — [`L2356`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2356)
- `_reshard_hlo_lowering(ctx, x_node, *, dst_sharding, concrete_mesh)` — [`L2351`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2351)
- `_reshard_impl(x, *, dst_sharding, concrete_mesh)` — [`L2305`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2305)
- `_reshard_jvp_rule(primals, tangents, *, dst_sharding, concrete_mesh)` — [`L2315`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2315)
- `_reshard_linearize(is_vjp, nzs, x, *, dst_sharding, concrete_mesh)` — [`L2327`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2327)
- `_reshard_transpose_fancy(ct, x, *, dst_sharding, concrete_mesh)` — [`L2340`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2340)
- `_resolve_and_lower(args, jaxpr: core.ClosedJaxpr, in_shardings, out_shardings, in_layouts, out_layouts, donated_invars, ctx_mesh, name, keep_unused, inline, lowering_platforms, lowering_parameters, pgle_profiler, compiler_options_kvs)` — [`L1090`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1090)
- `_resolve_in_layouts(args, jit_in_layouts, resolved_in_shardings, in_avals)` — [`L952`](../../../../../../raw/code/jax/jax/_src/pjit.py#L952)
- `_resolve_in_shardings(args, pjit_in_shardings: Sequence[PjitSharding])` — [`L1035`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1035)
- `_resolve_out_layouts(out_layouts, out_shardings, out_avals)` — [`L1010`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1010)
- `_run_python_pjit(p, args_flat, fun: Callable, args, kwargs)` — [`L132`](../../../../../../raw/code/jax/jax/_src/pjit.py#L132)
- `_seen_qdds_get(fun, in_type)` — [`L811`](../../../../../../raw/code/jax/jax/_src/pjit.py#L811)
- `_sharding_constraint_abstract_eval(x_aval, *, sharding, layout, context_mesh, unconstrained_dims)` — [`L2179`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2179)
- `_sharding_constraint_batcher(axis_data, vals_in, dims_in, sharding, layout, context_mesh, unconstrained_dims)` — [`L2230`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2230)
- `_sharding_constraint_hlo_lowering(ctx, x_node, *, sharding, layout, context_mesh, unconstrained_dims)` — [`L2187`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2187)
- `_sharding_constraint_impl(x, sharding, layout, context_mesh, unconstrained_dims)` — [`L2133`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2133)
- `_split_layout_and_sharding(entries)` — [`L325`](../../../../../../raw/code/jax/jax/_src/pjit.py#L325)
- `_to_lojax(*hi_args, jaxpr, **params)` — [`L892`](../../../../../../raw/code/jax/jax/_src/pjit.py#L892)
- `_trace_for_jit(fun: Callable, ji: PjitInfo, ctx_mesh: mesh_lib.Mesh, dbg: core.DebugInfo, avals, args, kwargs)` — [`L468`](../../../../../../raw/code/jax/jax/_src/pjit.py#L468)
- `_transpose_jaxpr_fancy(jaxpr, in_tree, in_avals, specs)` — [`L1926`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1926)
- `assert_shardings_equal(x_aval, user_sharding: NamedSharding)` — [`L2032`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2032)
- `auto_axes(f=None,, *, axes: str | tuple[str, ...] | None = None, out_sharding=None)` — [`L2426`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2426)
- `cache_miss(*args, **kwargs)` — [`L253`](../../../../../../raw/code/jax/jax/_src/pjit.py#L253)
- `call_impl_cache_miss(*args_, **kwargs_)` — [`L1219`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1219)
- `check_aval_layout_compatibility(layouts, flat_avals, names: Sequence[str], what_aval: str)` — [`L867`](../../../../../../raw/code/jax/jax/_src/pjit.py#L867)
- `check_shardings_are_auto(s: Sharding)` — [`L2019`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2019)
- `const_args_layouts(const_args: Sequence[ArrayLike], avals: Sequence[core.AbstractValue], shardings: Sequence[PjitSharding])` — [`L1456`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1456)
- `const_args_shardings(const_args: Sequence[Array | np.ndarray])` — [`L1451`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1451)
- `convert_to_metaty(arg)` — [`L1131`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1131)
- `create_meta_ty(aval, arg_sharding, arg_format, arg_committed, is_np_array)` — [`L1128`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1128)
- `dce_jaxpr_pjit_rule(used_outputs: list[bool], eqn: core.JaxprEqn)` — [`L1949`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1949)
- `decorator(*args, **kwargs)` — [`L2435`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2435)
- `decorator(*args, **kwargs)` — [`L2467`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2467)
- `expand(lol, stuff)` — [`L933`](../../../../../../raw/code/jax/jax/_src/pjit.py#L933)
- `explicit_axes(f=None,, *, axes: str | tuple[str, ...] | None = None, in_sharding=None)` — [`L2458`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2458)
- `finalize_arg_sharding(arg_s, committed)` — [`L1022`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1022)
- `get_ctx_mesh(use_resource_env)` — [`L602`](../../../../../../raw/code/jax/jax/_src/pjit.py#L602)
- `get_unconstrained_dims(sharding: NamedSharding)` — [`L2541`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2541)
- `hashable_pytree(pytree)` — [`L709`](../../../../../../raw/code/jax/jax/_src/pjit.py#L709)
- `jit_eval_shape(jit_func, *args, **kwargs)` — [`L316`](../../../../../../raw/code/jax/jax/_src/pjit.py#L316)
- `jit_evict_fn(self)` — [`L319`](../../../../../../raw/code/jax/jax/_src/pjit.py#L319)
- `jit_lower(jit_func, *args, **kwargs)` — [`L312`](../../../../../../raw/code/jax/jax/_src/pjit.py#L312)
- `jit_trace(jit_func, *args, **kwargs)` — [`L306`](../../../../../../raw/code/jax/jax/_src/pjit.py#L306)
- `keep_where(l, should_keep)` — [`L1596`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1596)
- `keep_where(l, should_keep)` — [`L1711`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1711)
- `keep_where(xs, keeps)` — [`L1957`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1957)
- `linearized(residuals, tangent)` — [`L2332`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2332)
- `make_jit(fun: Callable, *, in_shardings: Any, out_shardings: Any, static_argnums: int | Sequence[int] | None, static_argnames: str | Iterable[str] | None, donate_argnums: int | Sequence[int] | None, donate_argnames: str | Iterable[str] | None, keep_unused: bool, device: xc.Device | None, backend: str | None, inline: bool | api.Inline, compiler_options: dict[str, Any] | None, use_resource_env: bool)` — [`L430`](../../../../../../raw/code/jax/jax/_src/pjit.py#L430) — jit() and pjit() are thin wrappers around this function.
- `make_jit_cpp_cache(capacity)` — [`L226`](../../../../../../raw/code/jax/jax/_src/pjit.py#L226)
- `pjit(fun: Callable, in_shardings: Any = UNSPECIFIED, out_shardings: Any = UNSPECIFIED, static_argnums: int | Sequence[int] | None = None, static_argnames: str | Iterable[str] | None = None, donate_argnums: int | Sequence[int] | None = None, donate_argnames: str | Iterable[str] | None = None, keep_unused: bool = False, device: xc.Device | None = None, backend: str | None = None, inline: bool = False, compiler_options: dict[str, Any] | None = None)` — [`L686`](../../../../../../raw/code/jax/jax/_src/pjit.py#L686) — `jax.experimental.pjit.pjit` has been deprecated. Please use `jax.jit`.
- `pjit_check_aval_sharding(shardings, flat_avals, names: Sequence[str], what_aval: str, allow_uneven_sharding: bool)` — [`L841`](../../../../../../raw/code/jax/jax/_src/pjit.py#L841)
- `pjit_forwarding_rule(eqn)` — [`L1324`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1324)
- `pjit_staging_rule(trace, source_info, *args, **params)` — [`L1282`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1282)
- `reshard(xs, out_shardings)` — [`L2275`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2275)
- `tangent_fun(residuals, *tangents)` — [`L1629`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1629)
- `transposed(*in_flat)` — [`L1928`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1928)
- `with_layout_constraint(x, layouts)` — [`L2490`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2490)
- `with_sharding_constraint(x, shardings)` — [`L2050`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2050) — Mechanism to constrain the sharding of an Array inside a jitted computation

## Module values
- `PjitSharding` — [`L91`](../../../../../../raw/code/jax/jax/_src/pjit.py#L91)
- `_cpp_pjit_cache_explicit_attributes` — [`L240`](../../../../../../raw/code/jax/jax/_src/pjit.py#L240)
- `_cpp_pjit_cache_fun_only` — [`L233`](../../../../../../raw/code/jax/jax/_src/pjit.py#L233)
- `_pgle_profiler_dict` — [`L1106`](../../../../../../raw/code/jax/jax/_src/pjit.py#L1106)
- `_seen_qdds` — [`L809`](../../../../../../raw/code/jax/jax/_src/pjit.py#L809)
- `jit_p` — [`L883`](../../../../../../raw/code/jax/jax/_src/pjit.py#L883)
- `layout_constraint_p` — [`L2506`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2506)
- `reshard_p` — [`L2295`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2295)
- `sharding_constraint_p` — [`L2174`](../../../../../../raw/code/jax/jax/_src/pjit.py#L2174)
- `unsafe_map` — [`L86`](../../../../../../raw/code/jax/jax/_src/pjit.py#L86)
- `unsafe_zip` — [`L87`](../../../../../../raw/code/jax/jax/_src/pjit.py#L87)

