---
title: 'Module: jax/_src/core.py'
type: catalog
provenance: extracted
module: jax/_src/core.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.core`/
symbols:
  Primitive.bind: Primitive#bind().
  ShapedArray: ShapedArray#
  Primitive: Primitive#
  typeof: typeof().
  AbstractValue: AbstractValue#
  ShapedArray.shape: ShapedArray#shape.
  ShapedArray.dtype: ShapedArray#dtype.
  ClosedJaxpr.jaxpr: ClosedJaxpr#jaxpr.
  ClosedJaxpr: ClosedJaxpr#
  Jaxpr: Jaxpr#
  Var.aval: Var#aval.
  Primitive.multiple_results: Primitive#multiple_results.
  ShapeDtypeStruct: ShapeDtypeStruct#
  JaxprEqn.params: JaxprEqn#params.
  Primitive.def_effectful_abstract_eval: Primitive#def_effectful_abstract_eval().
  _check_jaxpr: _check_jaxpr().
  eval_jaxpr: eval_jaxpr().
  JaxprEqn.invars: JaxprEqn#invars.
  ClosedJaxpr.consts: ClosedJaxpr#consts.
  Primitive.def_abstract_eval: Primitive#def_abstract_eval().
  Primitive.def_impl: Primitive#def_impl().
  Jaxpr.invars: Jaxpr#invars().
  JaxprEqn.primitive: JaxprEqn#primitive.
  Var: Var#
  auto_insert_reshard: auto_insert_reshard().
  Tracer: Tracer#
  JaxprEqn.outvars: JaxprEqn#outvars.
  concrete_or_error: concrete_or_error().
  Jaxpr.outvars: Jaxpr#outvars().
  _pp_eqn: _pp_eqn().
  Jaxpr.constvars: Jaxpr#constvars().
  ShapedArray.sharding: ShapedArray#sharding.
  MemorySpace: MemorySpace#
  standard_vma_rule: standard_vma_rule().
  canonicalize_shape: canonicalize_shape().
  JaxprPpContext: JaxprPpContext#
  ClosedJaxpr.out_avals: ClosedJaxpr#out_avals().
  JaxprEqn: JaxprEqn#
  Jaxpr.eqns: Jaxpr#eqns().
  JaxprEqn.source_info: JaxprEqn#source_info.
  jaxpr_as_fun: jaxpr_as_fun().
  pp_jaxpr_skeleton: pp_jaxpr_skeleton().
  ShardingTypeError: ShardingTypeError#
  pvary: pvary().
  pp_toplevel_jaxpr: pp_toplevel_jaxpr().
  Atom: Atom.
  ShapedArray.update: ShapedArray#update().
  pp_eqn_rules.pp_eqn_rules: pp_eqn_rules.pp_eqn_rules.
  ShapedArray.mat: ShapedArray#mat().
  typematch: typematch().
  Tracer.aval: Tracer#aval.
  ShapedArray.ndim: ShapedArray#ndim.
  trace_ctx: trace_ctx.
  ShapeDtypeStruct.like: ShapeDtypeStruct#like().
  definitely_equal: definitely_equal().
  ShapedArray.memory_space: ShapedArray#memory_space.
  is_constant_dim: is_constant_dim().
  JaxprTypeError: JaxprTypeError#
  positional_effects: positional_effects().
  pp_vars: pp_vars().
  JaxprEqn.replace: JaxprEqn#replace().
  shaped_abstractify: shaped_abstractify().
  Jaxpr.debug_info: Jaxpr#debug_info().
  Primitive.is_high: Primitive#is_high().
  AbstractValue.str_short: AbstractValue#str_short().
  jaxpr_const_args: jaxpr_const_args().
  ref_p: ref_p.
  DebugInfo: DebugInfo.
  Jaxpr.__init__: Jaxpr#__init__().
  new_jaxpr_eqn: new_jaxpr_eqn().
  Tracer._pretty_print: Tracer#_pretty_print().
  _check_jaxpr.write: _check_jaxpr().write().
  pp_var: pp_var().
  aval_method: aval_method.
  pp_jaxprs: pp_jaxprs().
  Jaxpr.effects: Jaxpr#effects().
  ShapedArray.__new__: ShapedArray#__new__().
  Primitive.to_lojax: Primitive#to_lojax().
  AxisName: AxisName.
  empty_ref_p: empty_ref_p.
  JaxprPpSettings: JaxprPpSettings#
  pp_eqn: pp_eqn().
  ClosedJaxpr.in_avals: ClosedJaxpr#in_avals().
  update_aval_with_sharding: update_aval_with_sharding().
  check_jaxpr: check_jaxpr().
  pp_kv_pairs: pp_kv_pairs().
  Jaxpr.replace: Jaxpr#replace().
  physical_aval: physical_aval().
  NamedAxisEffect: NamedAxisEffect#
  _check_jaxpr.read: _check_jaxpr().read().
  Literal.val: Literal#val.
  getu: getu().
  pytype_aval_mappings.pytype_aval_mappings: pytype_aval_mappings.pytype_aval_mappings.
  Tracer._error_repr: Tracer#_error_repr().
  closed_call_p.closed_call_p: closed_call_p.closed_call_p.
  pp_jaxpr: pp_jaxpr().
  ShapedArray.str_short: ShapedArray#str_short().
  set_current_trace: set_current_trace.
  physical_element_aval: physical_element_aval().
  ShapeDtypeStruct.shape: ShapeDtypeStruct#shape.
  pp_shared_jaxpr: pp_shared_jaxpr().
  JaxprEqn.ctx: JaxprEqn#ctx.
  AbstractValue.to_ct_aval: AbstractValue#to_ct_aval().
  getr: getr().
  ShapedArray.weak_type: ShapedArray#weak_type.
  ShapedArray.to_ct_aval: ShapedArray#to_ct_aval().
  ShapeDtypeStruct.__init__: ShapeDtypeStruct#__init__().
  is_concrete: is_concrete().
  ManualAxisType: ManualAxisType#
  _check_call: _check_call().
  pp_kv_pair: pp_kv_pair().
  Tracer._origin_msg: Tracer#_origin_msg().
  is_constant_shape: is_constant_shape().
  concretization_function_error: concretization_function_error().
  insert_reduced_reshard: insert_reduced_reshard().
  freeze_p: freeze_p.
  reduced_vary_cast: reduced_vary_cast().
  Primitive.bind_with_trace: Primitive#bind_with_trace().
  ShapedArray.to_tangent_aval: ShapedArray#to_tangent_aval().
  ShapedArray.update_manual_axis_type: ShapedArray#update_manual_axis_type().
  Ref: Ref#
  _ref_abstract_eval: _ref_abstract_eval().
  ShapeDtypeStruct.update: ShapeDtypeStruct#update().
  custom_typechecks.custom_typechecks: custom_typechecks.custom_typechecks.
  Effect: Effect.
  check_avals_context_mesh: check_avals_context_mesh().
  concrete_dim_or_error: concrete_dim_or_error().
  accum_grad_in_ref_p: accum_grad_in_ref_p.
  definitely_equal_shape: definitely_equal_shape().
  AbstractValue.to_tangent_aval: AbstractValue#to_tangent_aval().
  ShapedArray.update_weak_type: ShapedArray#update_weak_type().
  order_wrt_mesh: order_wrt_mesh().
  _map_shaped_array: _map_shaped_array().
  _unmap_shaped_array: _unmap_shaped_array().
  AvalQDD: AvalQDD#
  free_ref_p: free_ref_p.
  unmapped_aval: unmapped_aval().
  abstract_token.abstract_token: abstract_token.abstract_token.
  JaxprEqnContextManager: JaxprEqnContextManager#
  primal_dtype_to_tangent_dtype: primal_dtype_to_tangent_dtype().
  ShapeDtypeStruct.__eq__: ShapeDtypeStruct#__eq__().
  pp_eqns: pp_eqns().
  pp_jaxpr_eqn_range: pp_jaxpr_eqn_range().
  is_dim: is_dim().
  typecompat: typecompat().
  shard_aval: shard_aval().
  unshard_aval: unshard_aval().
  get_cur_mesh_sharding: get_cur_mesh_sharding().
  mapped_aval: mapped_aval().
  ShapeDtypeStruct.sharding: ShapeDtypeStruct#sharding().
  ShapeDtypeStruct.__repr__: ShapeDtypeStruct#__repr__().
  Effects: Effects.
  MemorySpace.Host: MemorySpace#Host.
  str_short_aval: str_short_aval().
  aval_mismatch_extra: aval_mismatch_extra().
  check_type: check_type().
  ShapeDtypeStruct.dtype: ShapeDtypeStruct#dtype.
  last_used: last_used().
  JaxprEqn.__init__: JaxprEqn#__init__().
  _ref_impl: _ref_impl().
  ShapeDtypeStruct.__hash__: ShapeDtypeStruct#__hash__().
  aval_mapping_handlers.aval_mapping_handlers: aval_mapping_handlers.aval_mapping_handlers.
  AbstractValue.lo_ty: AbstractValue#lo_ty().
  _dropvars: _dropvars().
  eval_jaxpr.write: eval_jaxpr().write().
  stage_p: stage_p.
  cur_qdd: cur_qdd().
  max_dim: max_dim().
  eqn_effects: eqn_effects().
  InconclusiveDimensionOperation: InconclusiveDimensionOperation#
  Primitive.name: Primitive#name.
  check_integer_conversion: check_integer_conversion().
  _invalid_shape_error: _invalid_shape_error().
  MemorySpace.Device: MemorySpace#Device.
  get_sharding: get_sharding().
  filter_named_axis_effects: filter_named_axis_effects().
  Trace: Trace#
  extend_axis_env_nd: extend_axis_env_nd.
  JaxprEqnContextManager.__enter__: JaxprEqnContextManager#__enter__().
  JaxprEqnContextManager.__exit__: JaxprEqnContextManager#__exit__().
  JaxprEqnContext.__new__: JaxprEqnContext#__new__().
  Primitive.is_effectful: Primitive#is_effectful.
  Primitive.abstract_eval: Primitive#abstract_eval().
  Primitive.get_bind_params: Primitive#get_bind_params().
  Tracer.__getattr__: Tracer#__getattr__().
  AxisEnv.extend_pure: AxisEnv#extend_pure().
  SetCurrentTraceContextManager.__exit__: SetCurrentTraceContextManager#__exit__().
  get_axis_env: get_axis_env().
  eval_context: eval_context().
  ShapedArray.__getnewargs_ex__: ShapedArray#__getnewargs_ex__().
  freeze: freeze().
  jaxprs_in_params: jaxprs_in_params().
  get_mat: get_mat().
  canonicalize_slice: canonicalize_slice().
  remove_named_axis_effects: remove_named_axis_effects().
  pp_aval: pp_aval().
  ClosedJaxpr.effects: ClosedJaxpr#effects.
  AxisEnv.axis_sizes: AxisEnv#axis_sizes.
  TracingContext.is_top_level: TracingContext#is_top_level().
  dimension_as_value: dimension_as_value().
  ClosedJaxpr.in_aval_qdds: ClosedJaxpr#in_aval_qdds().
  Var.pretty_print: Var#pretty_print().
  EvalTrace.process_primitive: EvalTrace#process_primitive().
  TracingContext.trace: TracingContext#trace().
  _replace_jaxpr_effects: _replace_jaxpr_effects().
  pp_jaxpr_eqn_range.eqns_fn: pp_jaxpr_eqn_range().eqns_fn().
  Jaxpr.in_aval_qdds: Jaxpr#in_aval_qdds().
  Jaxpr.final_aval_qdds: Jaxpr#final_aval_qdds().
  JaxprEqnContext.__repr__: JaxprEqnContext#__repr__().
  AxisEnv.pop_pure: AxisEnv#pop_pure().
  trace_state_clean: trace_state_clean().
  ensure_no_leaks: ensure_no_leaks().
  AvalQDD.qdd: AvalQDD#qdd.
  modify_spec_for_auto_manual: modify_spec_for_auto_manual().
  ManualAxisType.vur: ManualAxisType#vur().
  empty_ref: empty_ref().
  divide_shape_sizes: divide_shape_sizes().
  min_dim: min_dim().
  escaped_tracer_error: escaped_tracer_error().
  Tracer.__init__: Tracer#__init__().
  get_referent: get_referent().
  new_ref: new_ref().
  canonicalize_slice.clamp_index: canonicalize_slice().clamp_index().
  call: call.
  mapped_leading_aval: mapped_leading_aval().
  check_jaxpr.ctx_factory: check_jaxpr().ctx_factory().
  JaxprPpContext.suggest_same_var_names: JaxprPpContext#suggest_same_var_names().
  ClosedJaxpr.replace: ClosedJaxpr#replace().
  TracingContext.axis_env: TracingContext#axis_env().
  AbstractValue.is_high: AbstractValue#is_high().
  AvalQDD.aval: AvalQDD#aval.
  free_ref: free_ref().
  dilate_dim: dilate_dim().
  replace_jaxpr_effects: replace_jaxpr_effects().
  resolve_input_effects: resolve_input_effects().
  Var.__init__: Var#__init__().
  eqn_params_const_args: eqn_params_const_args().
  check_scalar_conversion: check_scalar_conversion().
  TracingContext.reset: TracingContext#reset().
  TracingContext.set_trace: TracingContext#set_trace().
  AbstractValue.update: AbstractValue#update().
  _canonicalize_dimension: _canonicalize_dimension().
  _maybe_modify_sharding: _maybe_modify_sharding().
  ManualAxisType.update: ManualAxisType#update().
  _ref_to_lojax: _ref_to_lojax().
  is_symbolic_dim: is_symbolic_dim().
  unmapped_leading_aval: unmapped_leading_aval().
  _sds_aval_mapping: _sds_aval_mapping().
  CustomPpEqnRule: CustomPpEqnRule.
  pp_effect: pp_effect().
  ClosedJaxpr.final_aval_qdds: ClosedJaxpr#final_aval_qdds().
  Literal.aval: Literal#aval.
  Primitive.impl: Primitive#impl().
  Tracer.to_concrete_value: Tracer#to_concrete_value().
  TracingContext.set_axis_env: TracingContext#set_axis_env().
  mem_space_to_kind: mem_space_to_kind().
  NamedAxisEffect.name: NamedAxisEffect#name.
  typecheck: typecheck().
  clean_up_dead_vars: clean_up_dead_vars().
  subjaxprs: subjaxprs().
  ClosedJaxpr.__init__: ClosedJaxpr#__init__().
  Literal.pretty_print: Literal#pretty_print().
  is_literalable: is_literalable().
  Tracer.__bool__: Tracer#__bool__().
  Tracer.__int__: Tracer#__int__().
  Tracer.__hex__: Tracer#__hex__().
  Tracer.__oct__: Tracer#__oct__().
  Tracer.__index__: Tracer#__index__().
  AxisEnv.explicit_mesh_axis_names: AxisEnv#explicit_mesh_axis_names.
  AxisEnv.add_spmd_axis_names: AxisEnv#add_spmd_axis_names().
  AxisEnv.add_explicit_mesh_axis_names: AxisEnv#add_explicit_mesh_axis_names().
  AxisEnv.remove_explicit_mesh_axis_names: AxisEnv#remove_explicit_mesh_axis_names().
  TakeCurrentTraceContextManager.__enter__: TakeCurrentTraceContextManager#__enter__().
  SetCurrentTraceContextManager: SetCurrentTraceContextManager#
  SetCurrentTraceContextManager.__enter__: SetCurrentTraceContextManager#__enter__().
  ExtendAxisEnvNdContextManager.__enter__: ExtendAxisEnvNdContextManager#__enter__().
  AddSpmdAxisNamesContextManager.__enter__: AddSpmdAxisNamesContextManager#__enter__().
  AddExplicitMeshAxisNamesContextManager.__enter__: AddExplicitMeshAxisNamesContextManager#__enter__().
  RemoveExplicitMeshAxisNamesContextManager.__enter__: RemoveExplicitMeshAxisNamesContextManager#__enter__().
  valid_jaxtype: valid_jaxtype().
  to_concrete_value: to_concrete_value().
  AvalQDD.to_tangent_aval: AvalQDD#to_tangent_aval().
  physical_shape: physical_shape().
  ShapedArray.strip_weak_type: ShapedArray#strip_weak_type().
  check_unreduced_args: check_unreduced_args().
  ArrayRefImpl.__init__: ArrayRefImpl#__init__().
  _empty_ref_abstract_eval: _empty_ref_abstract_eval().
  is_empty_shape: is_empty_shape().
  CallPrimitive: CallPrimitive#
  ClosedCallPrimitive.get_bind_params: ClosedCallPrimitive#get_bind_params().
  JaxprPpContext.__init__: JaxprPpContext#__init__().
  JaxprEqnContext.manager: JaxprEqnContext#manager().
  JaxprEqn.effects: JaxprEqn#effects.
  Tracer._trace: Tracer#_trace.
  trace_state: trace_state.
  maybe_find_leaked_tracers: maybe_find_leaked_tracers().
  leaked_tracer_error: leaked_tracer_error().
  AbstractValue.strip_weak_type: AbstractValue#strip_weak_type().
  cur_aval_qdd: cur_aval_qdd().
  ManualAxisType.reduced: ManualAxisType#reduced.
  empty_mat: empty_mat.
  _empty_sharding: _empty_sharding().
  pvary_p: pvary_p.
  AbstractToken: AbstractToken#
  evaluate_shape: evaluate_shape().
  ShapeDtypeStruct.format: ShapeDtypeStruct#format().
  get_opaque_trace_state: get_opaque_trace_state().
  unsafe_am_i_under_a_jit: unsafe_am_i_under_a_jit().
  unsafe_am_i_under_a_vmap: unsafe_am_i_under_a_vmap().
  join_effects: join_effects().
  GenericEffect: GenericEffect#
  ManualAxisType.varying: ManualAxisType#varying.
  AxisSize: AxisSize.
  AbstractFuture.inner_aval: AbstractFuture#inner_aval.
  DropVar: DropVar#
  mem_kind_to_space: mem_kind_to_space().
  reduced_vary_cast_p: reduced_vary_cast_p.
  Ref._aval: Ref#_aval.
  _empty_ref_to_lojax: _empty_ref_to_lojax().
  _check_closed_call: _check_closed_call().
  trace_state_strong_ref: trace_state_strong_ref.
  axis_env_state: axis_env_state.
  AbstractValue.has_qdd: AbstractValue#has_qdd().
  AbstractValue.normalize: AbstractValue#normalize().
  AvalQDD.read_loval_in: AvalQDD#read_loval_in().
  stride_dim: stride_dim().
  ShapeDtypeStruct.manual_axis_type: ShapeDtypeStruct#manual_axis_type.
  literalable_types.literalable_types: literalable_types.literalable_types.
  ClosedJaxpr.map_jaxpr: ClosedJaxpr#map_jaxpr().
  JaxprEqnContext: JaxprEqnContext#
  JaxprEqn.__repr__: JaxprEqn#__repr__().
  gensym: gensym.
  Trace.stage_value: Trace#stage_value().
  Trace.process_primitive: Trace#process_primitive().
  Trace.process_call: Trace#process_call().
  Trace.process_custom_jvp_call: Trace#process_custom_jvp_call().
  Trace.process_custom_vjp_call: Trace#process_custom_vjp_call().
  Tracer.__dlpack__: Tracer#__dlpack__().
  Tracer.tolist: Tracer#tolist().
  Tracer.tobytes: Tracer#tobytes().
  Tracer.committed: Tracer#committed().
  Tracer.addressable_shards: Tracer#addressable_shards().
  Tracer.addressable_data: Tracer#addressable_data().
  Tracer.delete: Tracer#delete().
  Tracer.devices: Tracer#devices().
  Tracer.global_shards: Tracer#global_shards().
  Tracer.is_deleted: Tracer#is_deleted().
  Tracer.is_fully_addressable: Tracer#is_fully_addressable().
  Tracer.is_fully_replicated: Tracer#is_fully_replicated().
  Tracer.on_device_size_in_bytes: Tracer#on_device_size_in_bytes().
  Tracer.traceback: Tracer#traceback().
  Tracer.unsafe_buffer_pointer: Tracer#unsafe_buffer_pointer().
  EvalTrace.stage_value: EvalTrace#stage_value().
  EvalTrace.process_call: EvalTrace#process_call().
  AxisEnv.spmd_axis_names: AxisEnv#spmd_axis_names.
  eval_trace: eval_trace.
  TakeCurrentTraceContextManager.__exit__: TakeCurrentTraceContextManager#__exit__().
  ExtendAxisEnvNdContextManager.__exit__: ExtendAxisEnvNdContextManager#__exit__().
  AddSpmdAxisNamesContextManager.__exit__: AddSpmdAxisNamesContextManager#__exit__().
  AddExplicitMeshAxisNamesContextManager.__exit__: AddExplicitMeshAxisNamesContextManager#__exit__().
  RemoveExplicitMeshAxisNamesContextManager.__exit__: RemoveExplicitMeshAxisNamesContextManager#__exit__().
  reset_trace_state: reset_trace_state().
  AvalQDD.lo_ty: AvalQDD#lo_ty().
  ManualAxisType.__repr__: ManualAxisType#__repr__().
  ManualAxisType.__getnewargs_ex__: ManualAxisType#__getnewargs_ex__().
  ManualAxisType.to_ct_mat: ManualAxisType#to_ct_mat().
  ManualAxisType.invarying: ManualAxisType#invarying().
  ShapedArray._create: ShapedArray#_create().
  ShapedArray.nospec: ShapedArray#nospec().
  RefMeta.__instancecheck__: RefMeta#__instancecheck__().
  Ref.__init__: Ref#__init__().
  Ref.__repr__: Ref#__repr__().
  freeze_abstract_eval: freeze_abstract_eval().
  cancel_divide_tracers: cancel_divide_tracers().
  dim_value_aval: dim_value_aval().
  no_effects.no_effects: no_effects.no_effects.
  QuasiDynamicData: QuasiDynamicData#
  Jaxpr.out_avals: Jaxpr#out_avals().
  Jaxpr.pretty_print: Jaxpr#pretty_print().
  ClosedJaxpr.invars: ClosedJaxpr#invars.
  is_hoistable: is_hoistable().
  eval_jaxpr.read: eval_jaxpr().read().
  check_bool_conversion: check_bool_conversion().
  check_eval_args: check_eval_args().
  AxisEnv: AxisEnv#
  top_axis_env: top_axis_env.
  AddSpmdAxisNamesContextManager: AddSpmdAxisNamesContextManager#
  AddExplicitMeshAxisNamesContextManager: AddExplicitMeshAxisNamesContextManager#
  RemoveExplicitMeshAxisNamesContextManager: RemoveExplicitMeshAxisNamesContextManager#
  full_lower: full_lower().
  AbstractValue.update_weak_type: AbstractValue#update_weak_type().
  canonicalize_dim: canonicalize_dim().
  ManualAxisType.unreduced: ManualAxisType#unreduced.
  definitely_equal_one_of_dim: definitely_equal_one_of_dim().
  call_impl: call_impl().
  subst_input_effects: subst_input_effects().
  ShapeDtypeStruct._dll: ShapeDtypeStruct#_dll.
  nonempty_axis_env: nonempty_axis_env().
  unsafe_get_axis_names: unsafe_get_axis_names().
  Jaxpr._debug_info: Jaxpr#_debug_info.
  Var.initial_qdd: Var#initial_qdd.
  Primitive.def_bind_with_trace: Primitive#def_bind_with_trace().
  no_axis_name: no_axis_name.
  Ref._refs: Ref#_refs.
  internal_mutable_array_effect: internal_mutable_array_effect.
  JaxprPpContext.var_names: JaxprPpContext#var_names.
  JaxprPpContext.shared_jaxprs: JaxprPpContext#shared_jaxprs.
  call_p.call_p: call_p.call_p.
  config_ext: config_ext.
  _TRACER_ERROR_NUM_TRACEBACK_FRAMES: _TRACER_ERROR_NUM_TRACEBACK_FRAMES.
  current_jaxpr_eqn_context: current_jaxpr_eqn_context().
  Primitive.ref_primitive: Primitive#ref_primitive.
  Primitive.skip_canonicalization: Primitive#skip_canonicalization.
  ExtendAxisEnvNdContextManager: ExtendAxisEnvNdContextManager#
  dedup_referents: dedup_referents().
  AbstractValue.update_manual_axis_type: AbstractValue#update_manual_axis_type().
  ShapedArray.manual_axis_type: ShapedArray#manual_axis_type.
  _axis_types_dict: _axis_types_dict().
  _vma_ur_str: _vma_ur_str().
  InternalMutableArrayEffect: InternalMutableArrayEffect#
  _freeze_to_lojax: _freeze_to_lojax().
  replace_tracer_for_error_message: replace_tracer_for_error_message().
  _TempAxisName.id: _TempAxisName#id.
  MutableTypecheckVal: MutableTypecheckVal#
  MutableTypecheckVal.mutable_qdd: MutableTypecheckVal#mutable_qdd.
  check_eqn: check_eqn().
  _check_sharding: _check_sharding().
  ShapeDtypeStruct.weak_type: ShapeDtypeStruct#weak_type.
  ShapeDtypeStruct.is_ref: ShapeDtypeStruct#is_ref.
  unsafe_get_current_trace: unsafe_get_current_trace().
  Var.final_qdd: Var#final_qdd.
  take_current_trace: take_current_trace.
  remove_explicit_mesh_axis_names: remove_explicit_mesh_axis_names.
  concretization_function_error.error: concretization_function_error().error().
  AbstractFuture: AbstractFuture#
  canonicalize_slice.convert_to_index: canonicalize_slice().convert_to_index().
  ClosedCallPrimitive: ClosedCallPrimitive#
  unsafe_get_trace_stack: unsafe_get_trace_stack().
  Jaxpr.in_avals: Jaxpr#in_avals().
  ClosedJaxpr.outvars: ClosedJaxpr#outvars.
  Var.__repr__: Var#__repr__().
  DropVar.__init__: DropVar#__init__().
  Literal.__init__: Literal#__init__().
  Literal.hash: Literal#hash().
  Primitive.def_effectful_abstract_eval2: Primitive#def_effectful_abstract_eval2().
  Tracer._is_traced_array: Tracer#_is_traced_array().
  Tracer.sharding: Tracer#sharding().
  Tracer.device: Tracer#device().
  Tracer.__float__: Tracer#__float__().
  Tracer.__complex__: Tracer#__complex__().
  Tracer.__repr__: Tracer#__repr__().
  TraceTag: TraceTag#
  AxisEnv.__new__: AxisEnv#__new__().
  ExtendAxisEnvNdContextManager.__init__: ExtendAxisEnvNdContextManager#__init__().
  AddSpmdAxisNamesContextManager.__init__: AddSpmdAxisNamesContextManager#__init__().
  AddExplicitMeshAxisNamesContextManager.__init__: AddExplicitMeshAxisNamesContextManager#__init__().
  RemoveExplicitMeshAxisNamesContextManager.__init__: RemoveExplicitMeshAxisNamesContextManager#__init__().
  AbstractValue.dec_rank: AbstractValue#dec_rank().
  AbstractValue.inc_rank: AbstractValue#inc_rank().
  AbstractValue.shard: AbstractValue#shard().
  AbstractValue.unshard: AbstractValue#unshard().
  AbstractValue.vspace_add: AbstractValue#vspace_add().
  MutableQuasiDynamicData.__repr__: MutableQuasiDynamicData#__repr__().
  AvalQDD.read_loval: AvalQDD#read_loval().
  AvalQDD.read_loval_out: AvalQDD#read_loval_out().
  AvalQDD.new_from_loval: AvalQDD#new_from_loval().
  ShapedArray.__repr__: ShapedArray#__repr__().
  ShapedArray.__str__: ShapedArray#__str__().
  Ref.at: Ref#at().
  _free_ref_abstract_eval: _free_ref_abstract_eval().
  _free_ref_impl: _free_ref_impl().
  _freeze_impl: _freeze_impl().
  accum_grad_in_ref: accum_grad_in_ref().
  Token.block_until_ready: Token#block_until_ready().
  AbstractFuture.__eq__: AbstractFuture#__eq__().
  CallPrimitive.get_bind_params: CallPrimitive#get_bind_params().
  _TempAxisName.__eq__: _TempAxisName#__eq__().
  _TempAxisName.__lt__: _TempAxisName#__lt__().
  OpaqueTraceState.__eq__: OpaqueTraceState#__eq__().
  axis_frame: axis_frame().
  Jaxpr._constvars: Jaxpr#_constvars.
  Jaxpr._invars: Jaxpr#_invars.
  Jaxpr._outvars: Jaxpr#_outvars.
  Jaxpr._eqns: Jaxpr#_eqns.
  Jaxpr._effects: Jaxpr#_effects.
  Jaxpr.is_high: Jaxpr#is_high().
  Trace.__init__: Trace#__init__().
  Tracer._contents: Tracer#_contents().
  EvalTrace: EvalTrace#
  ensure_compile_time_eval: ensure_compile_time_eval().
  same_referent: same_referent().
  OutputType: OutputType.
  _dtype_object: _dtype_object().
  get_token_aval: get_token_aval().
  Token._buf: Token#_buf.
  dim_value_dtype: dim_value_dtype().
  cmp_shape_shd_mat_memsp: cmp_shape_shd_mat_memsp().
  find_top_trace: find_top_trace().
  unsafe_map: unsafe_map.
  ClosedJaxpr.debug_info: ClosedJaxpr#debug_info.
  Primitive.ref_allocating: Primitive#ref_allocating.
  _aval_property: _aval_property().
  aval_property: aval_property.
  TakeCurrentTraceContextManager: TakeCurrentTraceContextManager#
  Value: Value.
  Token: Token#
  ShapeDtypeStruct._sharding: ShapeDtypeStruct#_sharding.
  JaxprPpSettings.print_shapes: JaxprPpSettings#print_shapes.
  literalable_scalar_types.literalable_scalar_types: literalable_scalar_types.literalable_scalar_types.
  Jaxpr.__str__: Jaxpr#__str__().
  ClosedJaxpr._jaxpr: ClosedJaxpr#_jaxpr.
  ClosedJaxpr.pretty_print: ClosedJaxpr#pretty_print().
  JaxprEqnContext.cur_abstract_mesh: JaxprEqnContext#cur_abstract_mesh.
  JaxprEqnContext.remove_size_one_mesh_axis: JaxprEqnContext#remove_size_one_mesh_axis.
  JaxprEqnContext._create: JaxprEqnContext#_create().
  _effect_free_abstract_eval: _effect_free_abstract_eval().
  _effect_free_abstract_eval.abstract_eval_: _effect_free_abstract_eval().abstract_eval_().
  _generic_effectful_abstract_eval: _generic_effectful_abstract_eval().
  _generic_effectful_abstract_eval.abstract_eval_: _generic_effectful_abstract_eval().abstract_eval_().
  Trace.invalidate: Trace#invalidate().
  TracerBase: TracerBase.
  TracerMeta: TracerMeta.
  Tracer._short_repr: Tracer#_short_repr().
  AxisEnv._create: AxisEnv#_create().
  TracingContext.is_empty: TracingContext#is_empty().
  _why_alive: _why_alive().
  AbstractValue.lo_ty_qdd: AbstractValue#lo_ty_qdd().
  aval_qdd_from_current_val: aval_qdd_from_current_val().
  _dtype_object_types: _dtype_object_types.
  _make_lengths_same: _make_lengths_same().
  ManualAxisType._create: ManualAxisType#_create().
  ManualAxisType.empty: ManualAxisType#empty().
  ShapedArray._stripped_weak_type: ShapedArray#_stripped_weak_type.
  ShapedArray.broadcast: ShapedArray#broadcast.
  ShapedArray.transpose: ShapedArray#transpose.
  ShapedArray.reshape: ShapedArray#reshape.
  bint: bint#
  bint.name: bint#name().
  ArrayRefImpl._aval: ArrayRefImpl#_aval.
  ArrayRefImpl._buf: ArrayRefImpl#_buf.
  AbstractFuture.manual_axis_type: AbstractFuture#manual_axis_type().
  _cancel_divide: _cancel_divide().
  evaluate_shape.eval_one_dim: evaluate_shape().eval_one_dim().
  MutableTypecheckVal.aval: MutableTypecheckVal#aval.
  TraceType: TraceType.
  MutableQuasiDynamicData: MutableQuasiDynamicData#
  MemorySpace.Any: MemorySpace#Any.
  _create_str: _create_str().
  JaxprPpContext.shared_jaxpr_names: JaxprPpContext#shared_jaxpr_names.
  shard_aval_handlers: shard_aval_handlers.
  unshard_aval_handlers: unshard_aval_handlers.
  Jaxpr.__repr__: Jaxpr#__repr__.
  Jaxpr._repr_pretty_: Jaxpr#_repr_pretty_().
  ClosedJaxpr.literals: ClosedJaxpr#literals.
  ClosedJaxpr.__str__: ClosedJaxpr#__str__().
  ClosedJaxpr.__repr__: ClosedJaxpr#__repr__().
  ClosedJaxpr._repr_pretty_: ClosedJaxpr#_repr_pretty_().
  JaxprEqnContextManager.__init__: JaxprEqnContextManager#__init__().
  DropVar.pretty_print: DropVar#pretty_print().
  Literal.__repr__: Literal#__repr__().
  Primitive.__init__: Primitive#__init__().
  Primitive.__repr__: Primitive#__repr__().
  GenericEffect.prim: GenericEffect#prim.
  Trace.is_valid: Trace#is_valid().
  Tracer._line_info: Tracer#_line_info.
  Tracer.dtype: Tracer#dtype.
  Tracer.ndim: Tracer#ndim.
  Tracer.size: Tracer#size.
  Tracer.shape: Tracer#shape.
  Tracer.__jax_array__: Tracer#__jax_array__().
  Tracer.__array__: Tracer#__array__().
  Tracer.__iter__: Tracer#__iter__().
  Tracer.__len__: Tracer#__len__().
  Tracer.at: Tracer#at().
  Tracer.__reduce__: Tracer#__reduce__().
  Tracer.__setitem__: Tracer#__setitem__().
  Tracer.__array_module__: Tracer#__array_module__().
  EvalTrace.process_custom_jvp_call: EvalTrace#process_custom_jvp_call().
  EvalTrace.process_custom_vjp_call: EvalTrace#process_custom_vjp_call().
  TraceTag.__hash__: TraceTag#__hash__().
  TraceTag.__eq__: TraceTag#__eq__().
  AxisEnv.axis_size: AxisEnv#axis_size().
  AxisEnv.axis_exists: AxisEnv#axis_exists().
  AxisEnv.axis_names: AxisEnv#axis_names().
  SetCurrentTraceContextManager.__init__: SetCurrentTraceContextManager#__init__().
  add_spmd_axis_names: add_spmd_axis_names.
  add_explicit_mesh_axis_names: add_explicit_mesh_axis_names.
  AbstractValue.at_least_vspace: AbstractValue#at_least_vspace().
  AbstractValue.leading_axis_spec: AbstractValue#leading_axis_spec().
  InputType: InputType.
  MutableQuasiDynamicData.__init__: MutableQuasiDynamicData#__init__().
  MutableQuasiDynamicData.update: MutableQuasiDynamicData#update().
  AvalMutableQDD.aval: AvalMutableQDD#aval.
  AvalMutableQDD.mutable_qdd: AvalMutableQDD#mutable_qdd.
  ManualAxisType.__new__: ManualAxisType#__new__().
  ShapedArray._len: ShapedArray#_len().
  ShapedArray._bool: ShapedArray#_bool.
  ShapedArray._int: ShapedArray#_int.
  ShapedArray._float: ShapedArray#_float.
  ShapedArray._complex: ShapedArray#_complex.
  ShapedArray._hex: ShapedArray#_hex.
  ShapedArray._oct: ShapedArray#_oct.
  ShapedArray._index: ShapedArray#_index.
  bint.type: bint#type().
  bint.__str__: bint#__str__().
  Ref.__getitem__: Ref#__getitem__().
  Ref.__setitem__: Ref#__setitem__().
  Ref.__len__: Ref#__len__().
  Ref.addupdate: Ref#addupdate().
  Ref.unsafe_buffer_pointer: Ref#unsafe_buffer_pointer().
  Token.__init__: Token#__init__().
  AbstractFuture.__hash__: AbstractFuture#__hash__().
  AbstractFuture.str_short: AbstractFuture#str_short().
  AbstractFuture.done: AbstractFuture#done().
  AbstractFuture.shape: AbstractFuture#shape().
  AbstractFuture.dtype: AbstractFuture#dtype().
  AbstractFuture.sharding: AbstractFuture#sharding().
  AbstractFuture.mat: AbstractFuture#mat().
  dim_constant: dim_constant().
  _TempAxisName.__repr__: _TempAxisName#__repr__().
  _TempAxisName.__hash__: _TempAxisName#__hash__().
  ShapeDtypeStruct.__len__: ShapeDtypeStruct#__len__().
  ShapeDtypeStruct.__str__: ShapeDtypeStruct#__str__.
  unsafe_zip: unsafe_zip.
  Jaxpr._is_high: Jaxpr#_is_high.
  ClosedJaxpr.is_high: ClosedJaxpr#is_high.
  MutableQuasiDynamicData.cur_val: MutableQuasiDynamicData#cur_val.
  primal_sharding_to_cotangent_sharding: primal_sharding_to_cotangent_sharding().
  Ref.shape: Ref#shape.
  Ref.ndim: Ref#ndim.
  _TempAxisName: _TempAxisName#
  JaxprPpSettings.source_info: JaxprPpSettings#source_info.
  JaxprPpSettings.name_stack: JaxprPpSettings#name_stack.
  JaxprPpSettings.custom_pp_eqn_rules: JaxprPpSettings#custom_pp_eqn_rules.
  JaxprPpSettings.print_effects: JaxprPpSettings#print_effects.
  OpaqueTraceState: OpaqueTraceState#
  OpaqueTraceState._trace_ref: OpaqueTraceState#_trace_ref.
  PyTree: PyTree.
  InitialResultPaths: InitialResultPaths.
  initial_result_paths: initial_result_paths.
  ClosedJaxpr._consts: ClosedJaxpr#_consts.
  ClosedJaxpr.eqns: ClosedJaxpr#eqns.
  JaxprEqnContext.compute_type: JaxprEqnContext#compute_type.
  JaxprEqnContext.threefry_partitionable: JaxprEqnContext#threefry_partitionable.
  JaxprEqnContext.xla_metadata: JaxprEqnContext#xla_metadata.
  Primitive.call_primitive: Primitive#call_primitive.
  _str_abstractify: _str_abstractify().
  Tracer.__slots__: Tracer#__slots__.
  Tracer.full_lower: Tracer#full_lower().
  Tracer.get_referent: Tracer#get_referent().
  ParamDict: ParamDict.
  TRACER_LEAK_DEBUGGER_WARNING: TRACER_LEAK_DEBUGGER_WARNING.
  _why_alive_container_info: _why_alive_container_info().
  AbstractValue.raise_val2: AbstractValue#raise_val2().
  MutableQuasiDynamicData.init_val: MutableQuasiDynamicData#init_val.
  AvalMutableQDD: AvalMutableQDD#
  _check_divisibility: _check_divisibility().
  get_memory_space: get_memory_space().
  ShapedArray.lower_val: ShapedArray#lower_val().
  ShapedArray.raise_val: ShapedArray#raise_val().
  ShapedArray.lo_ty: ShapedArray#lo_ty().
  ShapedArray._iter: ShapedArray#_iter.
  _get_shape_sharding_str: _get_shape_sharding_str().
  bint.bound: bint#bound.
  RefMeta: RefMeta#
  Ref.size: Ref#size.
  Ref.dtype: Ref#dtype.
  ArrayRefImpl: ArrayRefImpl#
  SomeTracer: SomeTracer#
  AvalMapHandlerPair: AvalMapHandlerPair.
  _encode_digits_alphabetic: _encode_digits_alphabetic().
  TracingContext: TracingContext#
  identity: identity().
  EffectTypeSet: EffectTypeSet.
  Jaxpr.__slots__: Jaxpr#__slots__.
  ClosedJaxpr.__slots__: ClosedJaxpr#__slots__.
  ClosedJaxpr.constvars: ClosedJaxpr#constvars.
  JaxprEqnContextManager.__slots__: JaxprEqnContextManager#__slots__.
  JaxprEqnContext.__slots__: JaxprEqnContext#__slots__.
  JaxprEqn.__slots__: JaxprEqn#__slots__.
  Var.__slots__: Var#__slots__.
  DropVar.__repr__: DropVar#__repr__().
  Literal.__slots__: Literal#__slots__.
  Literal.__hash__: Literal#__hash__.
  Trace.__slots__: Trace#__slots__.
  Trace.__repr__: Trace#__repr__().
  Trace.full_raise: Trace#full_raise().
  Trace.main: Trace#main().
  Tracer.__array_priority__: Tracer#__array_priority__.
  Tracer.__hash__: Tracer#__hash__.
  Tracer.dimension_as_value: Tracer#dimension_as_value.
  Tracer.__reversed__: Tracer#__reversed__().
  EvalTrace.cur_qdd: EvalTrace#cur_qdd().
  AxisEnv.__slots__: AxisEnv#__slots__.
  TracingContext.__slots__: TracingContext#__slots__.
  TakeCurrentTraceContextManager.__slots__: TakeCurrentTraceContextManager#__slots__.
  SetCurrentTraceContextManager.__slots__: SetCurrentTraceContextManager#__slots__.
  ExtendAxisEnvNdContextManager.__slots__: ExtendAxisEnvNdContextManager#__slots__.
  AddSpmdAxisNamesContextManager.__slots__: AddSpmdAxisNamesContextManager#__slots__.
  AddExplicitMeshAxisNamesContextManager.__slots__: AddExplicitMeshAxisNamesContextManager#__slots__.
  RemoveExplicitMeshAxisNamesContextManager.__slots__: RemoveExplicitMeshAxisNamesContextManager#__slots__.
  AbstractValue.__slots__: AbstractValue#__slots__.
  AbstractValue.__repr__: AbstractValue#__repr__().
  AbstractValue.lower_val2: AbstractValue#lower_val2().
  AvalQDD.is_high: AvalQDD#is_high.
  AvalQDD.has_qdd: AvalQDD#has_qdd.
  AvalQDD.is_writer: AvalQDD#is_writer.
  MemorySpace.__repr__: MemorySpace#__repr__().
  MemorySpace.__hash__: MemorySpace#__hash__.
  ManualAxisType.__slots__: ManualAxisType#__slots__.
  ShapedArray.__slots__: ShapedArray#__slots__.
  ShapedArray.array_abstraction_level: ShapedArray#array_abstraction_level.
  ShapedArray.size: ShapedArray#size.
  Ref.aval: Ref#aval.
  Ref.sharding: Ref#sharding.
  Ref.format: Ref#format.
  Ref._committed: Ref#_committed.
  Ref.committed: Ref#committed.
  array_ref_effect: array_ref_effect.
  AbstractToken.str_short: AbstractToken#str_short().
  AbstractToken.to_tangent_aval: AbstractToken#to_tangent_aval().
  AbstractToken.to_ct_aval: AbstractToken#to_ct_aval().
  AbstractFuture.__init__: AbstractFuture#__init__().
  AbstractFuture.done_fun: AbstractFuture#done_fun.
  AbstractFuture.ndim: AbstractFuture#ndim.
  AbstractFuture.size: AbstractFuture#size.
  SomeTracer.__slots__: SomeTracer#__slots__.
  SomeTracer.__repr__: SomeTracer#__repr__().
  CallPrimitive.multiple_results: CallPrimitive#multiple_results.
  CallPrimitive.call_primitive: CallPrimitive#call_primitive.
  CallPrimitive.skip_canonicalization: CallPrimitive#skip_canonicalization.
  CallPrimitive.bind_with_trace: CallPrimitive#bind_with_trace().
  _TempAxisName.__init__: _TempAxisName#__init__().
  ShapeDtypeStruct.__slots__: ShapeDtypeStruct#__slots__.
  ShapeDtypeStruct.__setattr__: ShapeDtypeStruct#__setattr__().
  ShapeDtypeStruct.size: ShapeDtypeStruct#size.
  ShapeDtypeStruct.ndim: ShapeDtypeStruct#ndim.
  OpaqueTraceState.__init__: OpaqueTraceState#__init__().
---
# Module: [`jax/_src/core.py`](../../../../../../raw/code/jax/jax/_src/core.py)

## Classes
### `AbstractFuture`  ·  implements/extends AbstractValue
- def: [`jax/_src/core.py:3020`](../../../../../../raw/code/jax/jax/_src/core.py#L3020)
- signature: `class AbstractFuture(AbstractValue):`
- members:
  - `done(tracer)` — [`L3039`](../../../../../../raw/code/jax/jax/_src/core.py#L3039)
  - `dtype(self)` — [`L3050`](../../../../../../raw/code/jax/jax/_src/core.py#L3050)
  - `manual_axis_type(self)` — [`L3064`](../../../../../../raw/code/jax/jax/_src/core.py#L3064)
  - `mat(self)` — [`L3071`](../../../../../../raw/code/jax/jax/_src/core.py#L3071)
  - `shape(self)` — [`L3043`](../../../../../../raw/code/jax/jax/_src/core.py#L3043)
  - `sharding(self)` — [`L3057`](../../../../../../raw/code/jax/jax/_src/core.py#L3057)
  - `str_short(self, short_dtypes=False, mesh_axis_types=False)` — [`L3032`](../../../../../../raw/code/jax/jax/_src/core.py#L3032)
  - `done_fun` — [`L3023`](../../../../../../raw/code/jax/jax/_src/core.py#L3023)
  - `inner_aval` — [`L3022`](../../../../../../raw/code/jax/jax/_src/core.py#L3022)
  - `ndim` — [`L3035`](../../../../../../raw/code/jax/jax/_src/core.py#L3035)
  - `size` — [`L3036`](../../../../../../raw/code/jax/jax/_src/core.py#L3036)
- protocol/private: `__eq__`[`L3025`](../../../../../../raw/code/jax/jax/_src/core.py#L3025), `__hash__`[`L3029`](../../../../../../raw/code/jax/jax/_src/core.py#L3029), `__init__`[`L3021`](../../../../../../raw/code/jax/jax/_src/core.py#L3021)
- uses (calls/refs, reference-scoped): [`AbstractValue`](core.md#AbstractValue), [`aval_method`](core.md#aval_method)
- used by: [`AbstractValue`](core.md#AbstractValue), [`str_short`](core.md#AbstractValue.str_short), [`_async_done_abstract_eval`](lax/parallel.md#_async_done_abstract_eval), [`_async_start_abstract_eval`](lax/parallel.md#_async_start_abstract_eval)

### `AbstractToken`  ·  implements/extends AbstractValue
- def: [`jax/_src/core.py:2997`](../../../../../../raw/code/jax/jax/_src/core.py#L2997)
- signature: `class AbstractToken(AbstractValue):`
- members:
  - `str_short(self, short_dtypes=False, mesh_axis_types=False)` — [`L2998`](../../../../../../raw/code/jax/jax/_src/core.py#L2998)
  - `to_ct_aval(self)` — [`L3000`](../../../../../../raw/code/jax/jax/_src/core.py#L3000)
  - `to_tangent_aval(self)` — [`L2999`](../../../../../../raw/code/jax/jax/_src/core.py#L2999)
- uses (calls/refs, reference-scoped): [`AbstractValue`](core.md#AbstractValue)
- used by: [`AbstractValue`](core.md#AbstractValue), [`str_short`](core.md#AbstractValue.str_short), [`_check_sharding`](api.md#_check_sharding), [`to_ct_aval`](core.md#AbstractValue.to_ct_aval), [`to_tangent_aval`](core.md#AbstractValue.to_tangent_aval), [`abstract_token`](core.md#abstract_token.abstract_token), [`aval_mapping_handlers`](core.md#aval_mapping_handlers.aval_mapping_handlers), [`AbstractToken`](abstract_arrays.md#AbstractToken)

### `AbstractValue`
- def: [`jax/_src/core.py:1770`](../../../../../../raw/code/jax/jax/_src/core.py#L1770) — documented in [jax-_src-core](../../../concepts/jax-_src-core.md)
- members:
  - `at_least_vspace(self)` — [`L1788`](../../../../../../raw/code/jax/jax/_src/core.py#L1788)
  - `dec_rank(self, size, spec)` — [`L1822`](../../../../../../raw/code/jax/jax/_src/core.py#L1822)
  - `has_qdd(self)` — [`L1778`](../../../../../../raw/code/jax/jax/_src/core.py#L1778)
  - `inc_rank(self, size, spec)` — [`L1825`](../../../../../../raw/code/jax/jax/_src/core.py#L1825)
  - `is_high(self)` — [`L1774`](../../../../../../raw/code/jax/jax/_src/core.py#L1774)
  - `leading_axis_spec(self)` — [`L1828`](../../../../../../raw/code/jax/jax/_src/core.py#L1828)
  - `lo_ty(self)` — [`L1813`](../../../../../../raw/code/jax/jax/_src/core.py#L1813)
  - `lo_ty_qdd(self, qdd)` — [`L1816`](../../../../../../raw/code/jax/jax/_src/core.py#L1816)
  - `lower_val2(self, hi_val)` — [`L1844`](../../../../../../raw/code/jax/jax/_src/core.py#L1844)
  - `normalize(self)` — [`L1807`](../../../../../../raw/code/jax/jax/_src/core.py#L1807)
  - `raise_val2(self, lo_vals_ft)` — [`L1841`](../../../../../../raw/code/jax/jax/_src/core.py#L1841)
  - `shard(self, mesh, manual_axes, check_vma, spec)` — [`L1831`](../../../../../../raw/code/jax/jax/_src/core.py#L1831)
  - `str_short(self, short_dtypes=False, mesh_axis_types=False)` — [`L1819`](../../../../../../raw/code/jax/jax/_src/core.py#L1819)
  - `strip_weak_type(self)` — [`L1804`](../../../../../../raw/code/jax/jax/_src/core.py#L1804)
  - `to_ct_aval(self)` — [`L1784`](../../../../../../raw/code/jax/jax/_src/core.py#L1784)
  - `to_tangent_aval(self)` — [`L1781`](../../../../../../raw/code/jax/jax/_src/core.py#L1781)
  - `unshard(self, mesh, check_vma, spec)` — [`L1834`](../../../../../../raw/code/jax/jax/_src/core.py#L1834)
  - `update(self, **kwargs)` — [`L1810`](../../../../../../raw/code/jax/jax/_src/core.py#L1810)
  - `update_manual_axis_type(self, mat)` — [`L1801`](../../../../../../raw/code/jax/jax/_src/core.py#L1801)
  - `update_weak_type(self, weak_type)` — [`L1798`](../../../../../../raw/code/jax/jax/_src/core.py#L1798)
  - `vspace_add(self, x, y)` — [`L1837`](../../../../../../raw/code/jax/jax/_src/core.py#L1837)
- protocol/private: `__repr__`[`L1791`](../../../../../../raw/code/jax/jax/_src/core.py#L1791), `__slots__`[`L1771`](../../../../../../raw/code/jax/jax/_src/core.py#L1771)
- uses (calls/refs, reference-scoped): [`update`](core.md#ShapedArray.update), [`HiType`](hijax.md#HiType), [`update`](state/types.md#AbstractRef.update), [`str_short`](core.md#ShapedArray.str_short), [`to_ct_aval`](core.md#ShapedArray.to_ct_aval), [`to_tangent_aval`](core.md#ShapedArray.to_tangent_aval), [`update_manual_axis_type`](core.md#ShapedArray.update_manual_axis_type), [`add_jaxvals`](ad_util.md#add_jaxvals), [`update_weak_type`](core.md#ShapedArray.update_weak_type), [`unmapped_aval`](core.md#unmapped_aval), [`shard_aval`](core.md#shard_aval), [`unshard_aval`](core.md#unshard_aval), [`mapped_aval`](core.md#mapped_aval), [`AbstractLinVal`](state/types.md#AbstractLinVal), [`lo_ty_qdd`](hijax.md#MutableHiType.lo_ty_qdd), [`str_short`](state/types.md#AbstractRef.str_short), [`strip_weak_type`](core.md#ShapedArray.strip_weak_type), [`AbstractToken`](core.md#AbstractToken), [`to_tangent_aval`](hijax.md#MutableHiType.to_tangent_aval), [`AbstractSemaphore`](pallas/mosaic/core.md#AbstractSemaphore), [`update_manual_axis_type`](state/types.md#AbstractRef.update_manual_axis_type), [`update_weak_type`](state/types.md#AbstractRef.update_weak_type), [`MutableHiType`](hijax.md#MutableHiType), [`is_high`](state/types.md#AbstractRef.is_high), [`dec_rank`](hijax.md#HiType.dec_rank), [`inc_rank`](hijax.md#HiType.inc_rank), [`shard`](hijax.md#HiType.shard), [`to_ct_aval`](state/types.md#AbstractRef.to_ct_aval), [`unshard`](hijax.md#HiType.unshard), [`AbstractFuture`](core.md#AbstractFuture), [`lo_ty`](state/types.md#AbstractRef.lo_ty), [`to_ct_aval`](hijax.md#HiType.to_ct_aval), [`to_ct_aval`](hijax.md#MutableHiType.to_ct_aval), [`to_tangent_aval`](state/types.md#AbstractRef.to_tangent_aval), [`lo_ty`](hijax.md#HiType.lo_ty), [`to_tangent_aval`](hijax.md#HiType.to_tangent_aval), [`vspace_add`](hijax.md#HiType.vspace_add), [`leading_axis_spec`](hijax.md#HiType.leading_axis_spec), [`str_short`](core.md#AbstractFuture.str_short), [`lo_ty`](core.md#ShapedArray.lo_ty)  (+6 more)
- used by: [`AbstractRef`](state/types.md#AbstractRef), [`ShapedArray`](core.md#ShapedArray), [`interpret_pallas_call`](pallas/mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call), [`aval`](core.md#Var.aval), [`_pallas_call_batching_rule`](pallas/pallas_call.md#_pallas_call_batching_rule), [`to_block_mapping`](pallas/core.md#BlockSpec.to_block_mapping), [`get_grid_mapping`](pallas/core.md#get_grid_mapping), [`interpret_pallas_call`](pallas/mosaic_gpu/interpret/interpret_pallas_call.md#interpret_pallas_call), [`_pallas_call_state_discharge_rule`](pallas/pallas_call.md#_pallas_call_state_discharge_rule), [`pallas_call_hlo_interpret`](pallas/hlo_interpreter.md#pallas_call_hlo_interpret), [`_lower_to_custom_call`](pallas/mosaic/pallas_call_registration.md#_lower_to_custom_call), [`pallas_call_lowering`](pallas/triton/pallas_call_registration.md#pallas_call_lowering), [`_pallas_call_to_lojax`](pallas/pallas_call.md#_pallas_call_to_lojax), [`_flatten_bwd`](custom_derivatives.md#_flatten_bwd), [`_get_memory_space_from_aval`](pallas/mosaic/pallas_call_registration.md#_get_memory_space_from_aval), [`_pallas_call_jvp_rule`](pallas/pallas_call.md#_pallas_call_jvp_rule), [`get_ref_and_transforms`](state/primitives.md#get_ref_and_transforms), [`create`](pallas/mosaic/pipeline.md#BufferedRef.create), [`mpmd_map_tpu_lowering_rule`](pallas/mosaic/pallas_call_registration.md#mpmd_map_tpu_lowering_rule), [`pallas_call_lowering`](pallas/mosaic_gpu/pallas_call_registration.md#pallas_call_lowering), [`remat_partial_eval`](ad_checkpoint.md#remat_partial_eval), [`_extract_indirect_offsets_from_indices`](pallas/mosaic/sc_lowering.md#_extract_indirect_offsets_from_indices), [`pallas_call_tpu_lowering_rule`](pallas/mosaic/pallas_call_registration.md#pallas_call_tpu_lowering_rule), [`pass_scalars_as_refs`](pallas/mosaic/core.md#pass_scalars_as_refs), [`_commute_transform`](pallas/mosaic_gpu/lowering.md#_commute_transform), [`_shard_map_staging`](shard_map.md#_shard_map_staging), [`transform_type`](state/types.md#Transform.transform_type), [`_batch_with_explicit_loop`](pallas/pallas_call.md#_batch_with_explicit_loop), [`_saved_residuals`](ad_checkpoint.md#_saved_residuals), [`typematch`](core.md#typematch), [`__call__`](hijax.md#custom_vjp3.__call__), [`aval`](core.md#Tracer.aval), [`undo`](state/types.md#Transform.undo), [`_get_sds`](pallas/core.md#_get_sds), [`_scan_to_lojax`](lax/control_flow/loops.md#_scan_to_lojax), [`_shard_map_to_lojax`](shard_map.md#_shard_map_to_lojax), [`_sharding_constraint_impl`](pjit.md#_sharding_constraint_impl), [`transform_type`](state/indexing.md#NDIndexer.transform_type), [`shaped_abstractify`](core.md#shaped_abstractify), [`jaxpr_const_args`](core.md#jaxpr_const_args)  (+198 more)

### `AddExplicitMeshAxisNamesContextManager`
- def: [`jax/_src/core.py:1524`](../../../../../../raw/code/jax/jax/_src/core.py#L1524)
- signature: `class AddExplicitMeshAxisNamesContextManager:`
- protocol/private: `__enter__`[`L1530`](../../../../../../raw/code/jax/jax/_src/core.py#L1530), `__exit__`[`L1536`](../../../../../../raw/code/jax/jax/_src/core.py#L1536), `__init__`[`L1527`](../../../../../../raw/code/jax/jax/_src/core.py#L1527), `__slots__`[`L1525`](../../../../../../raw/code/jax/jax/_src/core.py#L1525)
- uses (calls/refs, reference-scoped): [`trace_ctx`](core.md#trace_ctx), [`AxisName`](core.md#AxisName), [`axis_env`](core.md#TracingContext.axis_env), [`set_axis_env`](core.md#TracingContext.set_axis_env)
- used by: [`add_explicit_mesh_axis_names`](core.md#add_explicit_mesh_axis_names)

### `AddSpmdAxisNamesContextManager`
- def: [`jax/_src/core.py:1507`](../../../../../../raw/code/jax/jax/_src/core.py#L1507)
- signature: `class AddSpmdAxisNamesContextManager:`
- protocol/private: `__enter__`[`L1513`](../../../../../../raw/code/jax/jax/_src/core.py#L1513), `__exit__`[`L1518`](../../../../../../raw/code/jax/jax/_src/core.py#L1518), `__init__`[`L1510`](../../../../../../raw/code/jax/jax/_src/core.py#L1510), `__slots__`[`L1508`](../../../../../../raw/code/jax/jax/_src/core.py#L1508)
- uses (calls/refs, reference-scoped): [`trace_ctx`](core.md#trace_ctx), [`AxisName`](core.md#AxisName), [`axis_env`](core.md#TracingContext.axis_env), [`set_axis_env`](core.md#TracingContext.set_axis_env)
- used by: [`add_spmd_axis_names`](core.md#add_spmd_axis_names)

### `ArrayRefImpl`
- def: [`jax/_src/core.py:2820`](../../../../../../raw/code/jax/jax/_src/core.py#L2820)
- signature: `class ArrayRefImpl:`
- protocol/private: `__init__`[`L2824`](../../../../../../raw/code/jax/jax/_src/core.py#L2824), `_aval`[`L2821`](../../../../../../raw/code/jax/jax/_src/core.py#L2821), `_buf`[`L2822`](../../../../../../raw/code/jax/jax/_src/core.py#L2822)
- uses (calls/refs, reference-scoped): [`Array`](basearray.md#Array), [`AbstractRef`](state/types.md#AbstractRef), [`ShapedArray`](core.md#ShapedArray), [`AbstractValue`](core.md#AbstractValue)
- used by: [`_ref_impl`](core.md#_ref_impl)

### `AvalMutableQDD`
- def: [`jax/_src/core.py:2049`](../../../../../../raw/code/jax/jax/_src/core.py#L2049)
- signature: `class AvalMutableQDD:`
- members:
  - `aval` — [`L2050`](../../../../../../raw/code/jax/jax/_src/core.py#L2050)
  - `mutable_qdd` — [`L2051`](../../../../../../raw/code/jax/jax/_src/core.py#L2051)
- uses (calls/refs, reference-scoped): [`AbstractValue`](core.md#AbstractValue), [`MutableQuasiDynamicData`](core.md#MutableQuasiDynamicData)
- used by: [`_check_jaxpr`](core.md#_check_jaxpr)

### `AvalQDD`
- def: [`jax/_src/core.py:2022`](../../../../../../raw/code/jax/jax/_src/core.py#L2022)
- members:
  - `lo_ty(self)` — [`L2030`](../../../../../../raw/code/jax/jax/_src/core.py#L2030)
  - `new_from_loval(self, *lovals)` — [`L2042`](../../../../../../raw/code/jax/jax/_src/core.py#L2042)
  - `read_loval(self, val)` — [`L2033`](../../../../../../raw/code/jax/jax/_src/core.py#L2033)
  - `read_loval_in(self, val)` — [`L2036`](../../../../../../raw/code/jax/jax/_src/core.py#L2036)
  - `read_loval_out(self, val)` — [`L2039`](../../../../../../raw/code/jax/jax/_src/core.py#L2039)
  - `to_tangent_aval(self)` — [`L2045`](../../../../../../raw/code/jax/jax/_src/core.py#L2045)
  - `aval` — [`L2024`](../../../../../../raw/code/jax/jax/_src/core.py#L2024)
  - `has_qdd` — [`L2026`](../../../../../../raw/code/jax/jax/_src/core.py#L2026)
  - `is_high` — [`L2023`](../../../../../../raw/code/jax/jax/_src/core.py#L2023)
  - `is_writer` — [`L2028`](../../../../../../raw/code/jax/jax/_src/core.py#L2028)
  - `qdd` — [`L2025`](../../../../../../raw/code/jax/jax/_src/core.py#L2025)
- uses (calls/refs, reference-scoped): [`AbstractValue`](core.md#AbstractValue), [`to_tangent_aval`](core.md#AbstractValue.to_tangent_aval), [`QuasiDynamicData`](core.md#QuasiDynamicData), [`lo_ty_qdd`](core.md#AbstractValue.lo_ty_qdd)
- used by: [`_trace_for_jit`](pjit.md#_trace_for_jit), [`_check_jaxpr`](core.md#_check_jaxpr), [`cond`](lax/control_flow/conditionals.md#cond), [`_scan_to_lojax`](lax/control_flow/loops.md#_scan_to_lojax), [`_while_to_lojax`](lax/control_flow/loops.md#_while_to_lojax), [`write`](core.md#_check_jaxpr.write), [`_pjit_transpose_fancy`](pjit.md#_pjit_transpose_fancy), [`_cond_transpose_fancy`](lax/control_flow/conditionals.md#_cond_transpose_fancy), [`_pad_constvars`](lax/control_flow/common.md#_pad_constvars), [`in_aval_qdds`](core.md#ClosedJaxpr.in_aval_qdds), [`abstract_eval`](hijax.md#NewBox.abstract_eval), [`final_aval_qdds`](core.md#Jaxpr.final_aval_qdds), [`in_aval_qdds`](core.md#Jaxpr.in_aval_qdds), [`final_aval_qdds`](core.md#ClosedJaxpr.final_aval_qdds), [`cur_aval_qdd`](core.md#cur_aval_qdd), [`abstract_eval`](hijax.md#NewLog.abstract_eval)

### `AxisEnv`
- def: [`jax/_src/core.py:1337`](../../../../../../raw/code/jax/jax/_src/core.py#L1337)
- signature: `class AxisEnv:`
- members:
  - `add_explicit_mesh_axis_names(self, axis_names)` — [`L1388`](../../../../../../raw/code/jax/jax/_src/core.py#L1388)
  - `add_spmd_axis_names(self, axis_names)` — [`L1383`](../../../../../../raw/code/jax/jax/_src/core.py#L1383)
  - `axis_exists(self, axis_name)` — [`L1364`](../../../../../../raw/code/jax/jax/_src/core.py#L1364)
  - `axis_names(self)` — [`L1367`](../../../../../../raw/code/jax/jax/_src/core.py#L1367)
  - `axis_size(self, axis_name)` — [`L1358`](../../../../../../raw/code/jax/jax/_src/core.py#L1358)
  - `extend_pure(self, name_size_pairs)` — [`L1376`](../../../../../../raw/code/jax/jax/_src/core.py#L1376)
  - `pop_pure(self, axis_name)` — [`L1370`](../../../../../../raw/code/jax/jax/_src/core.py#L1370)
  - `remove_explicit_mesh_axis_names(self, axis_names)` — [`L1392`](../../../../../../raw/code/jax/jax/_src/core.py#L1392)
  - `axis_sizes` — [`L1341`](../../../../../../raw/code/jax/jax/_src/core.py#L1341)
  - `explicit_mesh_axis_names` — [`L1343`](../../../../../../raw/code/jax/jax/_src/core.py#L1343)
  - `spmd_axis_names` — [`L1342`](../../../../../../raw/code/jax/jax/_src/core.py#L1342)
- protocol/private: `__new__`[`L1354`](../../../../../../raw/code/jax/jax/_src/core.py#L1354), `__slots__`[`L1338`](../../../../../../raw/code/jax/jax/_src/core.py#L1338), `_create`[`L1347`](../../../../../../raw/code/jax/jax/_src/core.py#L1347)
- uses (calls/refs, reference-scoped): [`FrozenDict`](frozen_dict.md#FrozenDict), [`AxisName`](core.md#AxisName), [`weak_value_interner`](util.md#weak_value_interner), [`no_axis_name`](core.md#no_axis_name)
- used by: [`top_axis_env`](core.md#top_axis_env)

### `CallPrimitive`  ·  implements/extends Primitive
- def: [`jax/_src/core.py:3329`](../../../../../../raw/code/jax/jax/_src/core.py#L3329)
- signature: `class CallPrimitive(Primitive):`
- members:
  - `bind_with_trace(self, trace, args, avals, params,)` — [`L3334`](../../../../../../raw/code/jax/jax/_src/core.py#L3334)
  - `get_bind_params(self, params)` — [`L3339`](../../../../../../raw/code/jax/jax/_src/core.py#L3339)
  - `call_primitive` — [`L3331`](../../../../../../raw/code/jax/jax/_src/core.py#L3331)
  - `multiple_results` — [`L3330`](../../../../../../raw/code/jax/jax/_src/core.py#L3330)
  - `skip_canonicalization` — [`L3332`](../../../../../../raw/code/jax/jax/_src/core.py#L3332)
- uses (calls/refs, reference-scoped): [`Primitive`](core.md#Primitive), [`eval_jaxpr`](core.md#eval_jaxpr), [`get_bind_params`](core.md#ClosedCallPrimitive.get_bind_params), [`ClosedCallPrimitive`](core.md#ClosedCallPrimitive)
- used by: [`Primitive`](core.md#Primitive), [`_deprecations`](../core.md#_deprecations), [`bind_with_trace`](core.md#Primitive.bind_with_trace), [`get_bind_params`](core.md#Primitive.get_bind_params), [`call_p`](core.md#call_p.call_p), [`ClosedCallPrimitive`](core.md#ClosedCallPrimitive), [`CallPrimitive`](../core.md#CallPrimitive)

### `ClosedCallPrimitive`  ·  implements/extends CallPrimitive
- def: [`jax/_src/core.py:3357`](../../../../../../raw/code/jax/jax/_src/core.py#L3357)
- signature: `class ClosedCallPrimitive(CallPrimitive):`
- members:
  - `get_bind_params(self, params)` — [`L3358`](../../../../../../raw/code/jax/jax/_src/core.py#L3358)
- uses (calls/refs, reference-scoped): [`jaxpr`](core.md#ClosedJaxpr.jaxpr), [`ClosedJaxpr`](core.md#ClosedJaxpr), [`eval_jaxpr`](core.md#eval_jaxpr), [`consts`](core.md#ClosedJaxpr.consts), [`CallPrimitive`](core.md#CallPrimitive)
- used by: [`closed_call_p`](core.md#closed_call_p.closed_call_p), [`_check_call`](core.md#_check_call), [`CallPrimitive`](core.md#CallPrimitive), [`get_bind_params`](core.md#CallPrimitive.get_bind_params)

### `ClosedJaxpr`
- def: [`jax/_src/core.py:247`](../../../../../../raw/code/jax/jax/_src/core.py#L247)
- members:
  - `final_aval_qdds(self)` — [`L281`](../../../../../../raw/code/jax/jax/_src/core.py#L281)
  - `in_aval_qdds(self)` — [`L276`](../../../../../../raw/code/jax/jax/_src/core.py#L276)
  - `in_avals(self)` — [`L272`](../../../../../../raw/code/jax/jax/_src/core.py#L272)
  - `map_jaxpr(self, f)` — [`L289`](../../../../../../raw/code/jax/jax/_src/core.py#L289)
  - `out_avals(self)` — [`L286`](../../../../../../raw/code/jax/jax/_src/core.py#L286)
  - `pretty_print(self, *, source_info=False, print_shapes=True, name_stack=False, custom_pp_eqn_rules=True, print_effects=False, **kwargs)` — [`L300`](../../../../../../raw/code/jax/jax/_src/core.py#L300)
  - `replace(self, *, jaxpr=None, consts=None)` — [`L292`](../../../../../../raw/code/jax/jax/_src/core.py#L292)
  - `consts` — [`L254`](../../../../../../raw/code/jax/jax/_src/core.py#L254)
  - `constvars` — [`L257`](../../../../../../raw/code/jax/jax/_src/core.py#L257)
  - `debug_info` — [`L262`](../../../../../../raw/code/jax/jax/_src/core.py#L262)
  - `effects` — [`L261`](../../../../../../raw/code/jax/jax/_src/core.py#L261)
  - `eqns` — [`L260`](../../../../../../raw/code/jax/jax/_src/core.py#L260)
  - `invars` — [`L258`](../../../../../../raw/code/jax/jax/_src/core.py#L258)
  - `is_high` — [`L263`](../../../../../../raw/code/jax/jax/_src/core.py#L263)
  - `jaxpr` — [`L253`](../../../../../../raw/code/jax/jax/_src/core.py#L253)
  - `literals` — [`L255`](../../../../../../raw/code/jax/jax/_src/core.py#L255)
  - `outvars` — [`L259`](../../../../../../raw/code/jax/jax/_src/core.py#L259)
- protocol/private: `__init__`[`L265`](../../../../../../raw/code/jax/jax/_src/core.py#L265), `__repr__`[`L298`](../../../../../../raw/code/jax/jax/_src/core.py#L298), `__slots__`[`L248`](../../../../../../raw/code/jax/jax/_src/core.py#L248), `__str__`[`L297`](../../../../../../raw/code/jax/jax/_src/core.py#L297), `_consts`[`L251`](../../../../../../raw/code/jax/jax/_src/core.py#L251), `_jaxpr`[`L250`](../../../../../../raw/code/jax/jax/_src/core.py#L250), `_repr_pretty_`[`L311`](../../../../../../raw/code/jax/jax/_src/core.py#L311)
- uses (calls/refs, reference-scoped): [`AbstractValue`](core.md#AbstractValue), [`Jaxpr`](core.md#Jaxpr), [`constvars`](core.md#Jaxpr.constvars), [`AvalQDD`](core.md#AvalQDD)
- used by: [`_interpret_jaxpr`](pallas/mosaic/interpret/interpret_pallas_call.md#_interpret_jaxpr), [`lower_jaxpr_into_pipelined_module`](pallas/mosaic/lowering.md#lower_jaxpr_into_pipelined_module), [`_trace_for_jit`](pjit.md#_trace_for_jit), [`to_block_mapping`](pallas/core.md#BlockSpec.to_block_mapping), [`_run_scoped_lowering_rule`](pallas/mosaic_gpu/lowering.md#_run_scoped_lowering_rule), [`_export_lowered`](export/_export.md#_export_lowered), [`scan`](lax/control_flow/loops.md#scan), [`_pallas_call_state_discharge_rule`](pallas/pallas_call.md#_pallas_call_state_discharge_rule), [`cond`](lax/control_flow/conditionals.md#cond), [`while_loop`](lax/control_flow/loops.md#while_loop), [`wrapper`](pallas/mpmd.md#_mpmd_map.wrapper), [`wrapped`](pallas/mosaic/pipeline.md#emit_pipeline.wrapped), [`_pallas_call_to_lojax`](pallas/pallas_call.md#_pallas_call_to_lojax), [`_scan_partial_eval_custom`](lax/control_flow/loops.md#_scan_partial_eval_custom), [`_run_state_lowering_rule`](pallas/mosaic_gpu/lowering.md#_run_state_lowering_rule), [`_pallas_call_jvp_rule`](pallas/pallas_call.md#_pallas_call_jvp_rule), [`__call__`](custom_partitioning.md#custom_partitioning.__call__), [`_pjit_lowering`](pjit.md#_pjit_lowering), [`_switch_internal`](lax/control_flow/conditionals.md#_switch_internal), [`_rewrite_jaxpr_for_lowering`](pallas/mosaic/pallas_call_registration.md#_rewrite_jaxpr_for_lowering), [`pass_scalars_as_refs`](pallas/mosaic/core.md#pass_scalars_as_refs), [`_shard_map_staging`](shard_map.md#_shard_map_staging), [`jaxpr_as_fun`](core.md#jaxpr_as_fun), [`_scan_lowering_rule`](pallas/mosaic_gpu/lowering.md#_scan_lowering_rule), [`_scan_partial_eval`](lax/control_flow/loops.md#_scan_partial_eval), [`_estimate_resources`](pallas/mosaic_gpu/lowering.md#_estimate_resources), [`cost_estimate_jaxpr`](pallas/cost_estimate.md#cost_estimate_jaxpr), [`_batch_block_mapping`](pallas/pallas_call.md#_batch_block_mapping), [`_pjit_partial_eval`](pjit.md#_pjit_partial_eval), [`_scan_dce_rule`](lax/control_flow/loops.md#_scan_dce_rule), [`_trace_kernel_to_jaxpr`](pallas/pallas_call.md#_trace_kernel_to_jaxpr), [`wrapped_fwd`](custom_derivatives.md#optimize_remat_of_custom_vjp_fwd.wrapped_fwd), [`_custom_jvp_call_eval_rule`](pallas/fuser/block_spec.md#_custom_jvp_call_eval_rule), [`_custom_vjp_call_eval_rule`](pallas/fuser/block_spec.md#_custom_vjp_call_eval_rule), [`_reduce_window`](lax/windowed_reductions.md#_reduce_window), [`positional_effects`](core.md#positional_effects), [`_custom_vjp_call_dce`](custom_derivatives.md#_custom_vjp_call_dce), [`_eval_index_map`](pallas/triton/lowering.md#_eval_index_map), [`_scan_to_lojax`](lax/control_flow/loops.md#_scan_to_lojax), [`_scan_transpose_fancy`](lax/control_flow/loops.md#_scan_transpose_fancy)  (+176 more)

### `DropVar`  ·  implements/extends Var
- def: [`jax/_src/core.py:531`](../../../../../../raw/code/jax/jax/_src/core.py#L531)
- signature: `class DropVar(Var):`
- members:
  - `pretty_print(self, context: JaxprPpContext, *, print_dtype: bool = True)` — [`L535`](../../../../../../raw/code/jax/jax/_src/core.py#L535)
- protocol/private: `__init__`[`L532`](../../../../../../raw/code/jax/jax/_src/core.py#L532), `__repr__`[`L534`](../../../../../../raw/code/jax/jax/_src/core.py#L534)
- uses (calls/refs, reference-scoped): [`AbstractValue`](core.md#AbstractValue), [`Var`](core.md#Var), [`JaxprPpContext`](core.md#JaxprPpContext), [`__init__`](core.md#Var.__init__)
- used by: [`Var`](core.md#Var), [`_deprecations`](../core.md#_deprecations), [`_partial_eval_jaxpr_custom_rule`](shard_map.md#_partial_eval_jaxpr_custom_rule), [`write`](core.md#_check_jaxpr.write), [`pp_var`](core.md#pp_var), [`_swap_pp_rule`](pallas/primitives.md#_swap_pp_rule), [`pretty_print`](core.md#Var.pretty_print), [`DropVar`](../core.md#DropVar)

### `EvalTrace`  ·  implements/extends Trace
- def: [`jax/_src/core.py:1277`](../../../../../../raw/code/jax/jax/_src/core.py#L1277)
- signature: `class EvalTrace(Trace):`
- members:
  - `cur_qdd(self, x)` — [`L1313`](../../../../../../raw/code/jax/jax/_src/core.py#L1313)
  - `process_call(self, primitive, f, tracers, params,)` — [`L1295`](../../../../../../raw/code/jax/jax/_src/core.py#L1295)
  - `process_custom_jvp_call(self, primitive, fun, jvp, tracers,, **_)` — [`L1303`](../../../../../../raw/code/jax/jax/_src/core.py#L1303)
  - `process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers,, **_)` — [`L1308`](../../../../../../raw/code/jax/jax/_src/core.py#L1308)
  - `process_primitive(self, primitive, args, params,)` — [`L1284`](../../../../../../raw/code/jax/jax/_src/core.py#L1284)
  - `stage_value(self, val)` — [`L1279`](../../../../../../raw/code/jax/jax/_src/core.py#L1279)
- uses (calls/refs, reference-scoped): [`Array`](basearray.md#Array), [`set_current_trace`](core.md#set_current_trace), [`call_impl_with_key_reuse_checks`](../experimental/key_reuse/_core.md#call_impl_with_key_reuse_checks), [`stage_p`](core.md#stage_p), [`Trace`](core.md#Trace), [`debug_key_reuse`](config.md#debug_key_reuse), [`check_eval_args`](core.md#check_eval_args), [`full_lower`](core.md#full_lower)
- used by: [`Trace`](core.md#Trace), [`eval_trace`](core.md#eval_trace), [`process_call`](core.md#Trace.process_call), [`process_custom_jvp_call`](core.md#Trace.process_custom_jvp_call), [`process_custom_vjp_call`](core.md#Trace.process_custom_vjp_call), [`process_primitive`](core.md#Trace.process_primitive), [`stage_value`](core.md#Trace.stage_value)

### `ExtendAxisEnvNdContextManager`
- def: [`jax/_src/core.py:1491`](../../../../../../raw/code/jax/jax/_src/core.py#L1491)
- signature: `class ExtendAxisEnvNdContextManager:`
- protocol/private: `__enter__`[`L1497`](../../../../../../raw/code/jax/jax/_src/core.py#L1497), `__exit__`[`L1501`](../../../../../../raw/code/jax/jax/_src/core.py#L1501), `__init__`[`L1494`](../../../../../../raw/code/jax/jax/_src/core.py#L1494), `__slots__`[`L1492`](../../../../../../raw/code/jax/jax/_src/core.py#L1492)
- uses (calls/refs, reference-scoped): [`trace_ctx`](core.md#trace_ctx), [`AxisName`](core.md#AxisName), [`axis_env`](core.md#TracingContext.axis_env), [`set_axis_env`](core.md#TracingContext.set_axis_env)
- used by: [`extend_axis_env_nd`](core.md#extend_axis_env_nd)

### `GenericEffect`  ·  implements/extends Effect
- def: [`jax/_src/core.py:763`](../../../../../../raw/code/jax/jax/_src/core.py#L763)
- signature: `class GenericEffect(Effect):`
- members:
  - `prim` — [`L764`](../../../../../../raw/code/jax/jax/_src/core.py#L764)
- uses (calls/refs, reference-scoped): [`Primitive`](core.md#Primitive), [`Effect`](core.md#Effect)
- used by: [`abstract_eval_`](core.md#_generic_effectful_abstract_eval.abstract_eval_)

### `InconclusiveDimensionOperation`
- def: [`jax/_src/core.py:3078`](../../../../../../raw/code/jax/jax/_src/core.py#L3078)
- uses (calls/refs, reference-scoped): [`set_module`](util.md#set_module)
- used by: [`_deprecations`](../core.md#_deprecations), [`fori_loop`](lax/control_flow/loops.md#fori_loop), [`definitely_equal`](core.md#definitely_equal), [`_arange_dynamic`](numpy/lax_numpy.md#_arange_dynamic), [`build_padding`](numpy/lax_numpy.md#_pad_symmetric_or_reflect.build_padding), [`canonicalize_slice`](core.md#canonicalize_slice), [`divide_shape_sizes`](core.md#divide_shape_sizes), [`clamp_index`](core.md#canonicalize_slice.clamp_index), [`_top_k_abstract_eval`](lax/lax.md#_top_k_abstract_eval), [`InconclusiveDimensionOperation`](../core.md#InconclusiveDimensionOperation)

### `InternalMutableArrayEffect`  ·  implements/extends Effect
- def: [`jax/_src/core.py:2834`](../../../../../../raw/code/jax/jax/_src/core.py#L2834)
- signature: `class InternalMutableArrayEffect(effects.Effect):`
- used by: [`kernel_local_effects`](pallas/core.md#kernel_local_effects.kernel_local_effects), [`internal_mutable_array_effect`](core.md#internal_mutable_array_effect), [`_has_effects`](ad_checkpoint.md#_has_effects), [`name`](state/types.md#AccumEffect.name)

### `Jaxpr`
- def: [`jax/_src/core.py:98`](../../../../../../raw/code/jax/jax/_src/core.py#L98)
- signature: `class Jaxpr:`
- members:
  - `__init__(self, constvars: Sequence[Var], invars: Sequence[Var], outvars: Sequence[Atom], eqns: Sequence[JaxprEqn], effects: Effects = no_effects, debug_info: DebugInfo = None, is_high: bool = False)` — [`L156`](../../../../../../raw/code/jax/jax/_src/core.py#L156) — Args:
  - `constvars(self)` — [`L111`](../../../../../../raw/code/jax/jax/_src/core.py#L111)
  - `debug_info(self)` — [`L131`](../../../../../../raw/code/jax/jax/_src/core.py#L131)
  - `effects(self)` — [`L127`](../../../../../../raw/code/jax/jax/_src/core.py#L127)
  - `eqns(self)` — [`L123`](../../../../../../raw/code/jax/jax/_src/core.py#L123)
  - `final_aval_qdds(self)` — [`L148`](../../../../../../raw/code/jax/jax/_src/core.py#L148)
  - `in_aval_qdds(self)` — [`L143`](../../../../../../raw/code/jax/jax/_src/core.py#L143)
  - `in_avals(self)` — [`L139`](../../../../../../raw/code/jax/jax/_src/core.py#L139)
  - `invars(self)` — [`L115`](../../../../../../raw/code/jax/jax/_src/core.py#L115)
  - `is_high(self)` — [`L135`](../../../../../../raw/code/jax/jax/_src/core.py#L135)
  - `out_avals(self)` — [`L153`](../../../../../../raw/code/jax/jax/_src/core.py#L153)
  - `outvars(self)` — [`L119`](../../../../../../raw/code/jax/jax/_src/core.py#L119)
  - `pretty_print(self, *, source_info=False, print_shapes=True, custom_pp_eqn_rules=True, name_stack=False, print_effects: bool = False, **kwargs)` — [`L194`](../../../../../../raw/code/jax/jax/_src/core.py#L194)
  - `replace(self, **kwargs)` — [`L206`](../../../../../../raw/code/jax/jax/_src/core.py#L206)
- protocol/private: `__repr__`[`L192`](../../../../../../raw/code/jax/jax/_src/core.py#L192), `__slots__`[`L99`](../../../../../../raw/code/jax/jax/_src/core.py#L99), `__str__`[`L189`](../../../../../../raw/code/jax/jax/_src/core.py#L189), `_constvars`[`L102`](../../../../../../raw/code/jax/jax/_src/core.py#L102), `_debug_info`[`L107`](../../../../../../raw/code/jax/jax/_src/core.py#L107), `_effects`[`L106`](../../../../../../raw/code/jax/jax/_src/core.py#L106), `_eqns`[`L105`](../../../../../../raw/code/jax/jax/_src/core.py#L105), `_invars`[`L103`](../../../../../../raw/code/jax/jax/_src/core.py#L103), `_is_high`[`L108`](../../../../../../raw/code/jax/jax/_src/core.py#L108), `_outvars`[`L104`](../../../../../../raw/code/jax/jax/_src/core.py#L104), `_repr_pretty_`[`L203`](../../../../../../raw/code/jax/jax/_src/core.py#L203)
- uses (calls/refs, reference-scoped): [`AbstractValue`](core.md#AbstractValue), [`aval`](core.md#Var.aval), [`Var`](core.md#Var), [`JaxprEqn`](core.md#JaxprEqn), [`pp_toplevel_jaxpr`](core.md#pp_toplevel_jaxpr), [`Atom`](core.md#Atom), [`DebugInfo`](core.md#DebugInfo), [`enable_checks`](config.md#enable_checks), [`AvalQDD`](core.md#AvalQDD), [`Effects`](core.md#Effects), [`format`](pretty_printer.md#Doc.format), [`no_effects`](core.md#no_effects.no_effects), [`initial_qdd`](core.md#Var.initial_qdd), [`final_qdd`](core.md#Var.final_qdd)
- used by: [`interpret_pallas_call`](pallas/mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call), [`jaxpr_subcomp`](pallas/mosaic/lowering.md#jaxpr_subcomp), [`lower_jaxpr_into_pipelined_module`](pallas/mosaic/lowering.md#lower_jaxpr_into_pipelined_module), [`lower_jaxpr_to_module`](pallas/mosaic_gpu/lowering.md#lower_jaxpr_to_module), [`lower_jaxpr_to_mosaic_gpu`](pallas/mosaic_gpu/lowering.md#lower_jaxpr_to_mosaic_gpu), [`_pallas_call_batching_rule`](pallas/pallas_call.md#_pallas_call_batching_rule), [`_run_scoped_lowering_rule`](pallas/mosaic_gpu/lowering.md#_run_scoped_lowering_rule), [`_check_jaxpr`](core.md#_check_jaxpr), [`eval_jaxpr`](core.md#eval_jaxpr), [`_eval_jaxpr_discharge_state`](state/discharge.md#_eval_jaxpr_discharge_state), [`_pull_block_transform`](pallas/fuser/block_spec.md#_pull_block_transform), [`jaxpr_type_signature`](../experimental/key_reuse/_core.md#jaxpr_type_signature), [`interpret_pallas_call`](pallas/mosaic_gpu/interpret/interpret_pallas_call.md#interpret_pallas_call), [`_pallas_call_state_discharge_rule`](pallas/pallas_call.md#_pallas_call_state_discharge_rule), [`lower_pipelined_jaxpr_to_module`](pallas/mosaic_gpu/lowering.md#lower_pipelined_jaxpr_to_module), [`_run_scoped_resource_estimator`](pallas/mosaic_gpu/lowering.md#_run_scoped_resource_estimator), [`pallas_call_hlo_interpret`](pallas/hlo_interpreter.md#pallas_call_hlo_interpret), [`_body`](pallas/mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call._execute_grid_for_core._body), [`kernel_fn`](pallas/fuser/block_spec.md#make_kernel_function.kernel_fn), [`lower_jaxpr_to_triton_ir`](pallas/triton/lowering.md#lower_jaxpr_to_triton_ir), [`pallas_call_lowering`](pallas/triton/pallas_call_registration.md#pallas_call_lowering), [`physicalize_interp`](pallas/fuser/fusible_dtype.md#physicalize_interp), [`_custom_evaluate_jaxpr`](pallas/fuser/custom_evaluate.md#_custom_evaluate_jaxpr), [`_dimension_semantics`](pallas/mosaic/lowering.md#MosaicGridMapping._dimension_semantics), [`wrapper`](pallas/mpmd.md#_mpmd_map.wrapper), [`_partial_eval_jaxpr_custom_rule`](shard_map.md#_partial_eval_jaxpr_custom_rule), [`lower_jaxpr_into_unpipelined_module`](pallas/mosaic/lowering.md#lower_jaxpr_into_unpipelined_module), [`eval_jaxpr_recursive`](pallas/hlo_interpreter.md#eval_jaxpr_recursive), [`lower_jaxpr_to_triton_module`](pallas/triton/lowering.md#lower_jaxpr_to_triton_module), [`_pallas_call_to_lojax`](pallas/pallas_call.md#_pallas_call_to_lojax), [`_run_state_lowering_rule`](pallas/mosaic_gpu/lowering.md#_run_state_lowering_rule), [`_insert_reduce_precision`](ad_checkpoint.md#_insert_reduce_precision), [`_pallas_call_jvp_rule`](pallas/pallas_call.md#_pallas_call_jvp_rule), [`_push_block_spec_jaxpr`](pallas/fuser/block_spec.md#_push_block_spec_jaxpr), [`mpmd_map_tpu_lowering_rule`](pallas/mosaic/pallas_call_registration.md#mpmd_map_tpu_lowering_rule), [`pallas_call_lowering`](pallas/mosaic_gpu/pallas_call_registration.md#pallas_call_lowering), [`remat_partial_eval`](ad_checkpoint.md#remat_partial_eval), [`_jaxpr_call_lowering_rule`](pallas/mosaic_gpu/lowering.md#_jaxpr_call_lowering_rule), [`pallas_call_tpu_lowering_rule`](pallas/mosaic/pallas_call_registration.md#pallas_call_tpu_lowering_rule), [`_rewrite_jaxpr_for_lowering`](pallas/mosaic/pallas_call_registration.md#_rewrite_jaxpr_for_lowering)  (+107 more)

### `JaxprEqn`
- def: [`jax/_src/core.py:423`](../../../../../../raw/code/jax/jax/_src/core.py#L423)
- members:
  - `replace(self, invars: list[Atom] | None = None, outvars: list[Var] | None = None, primitive: Primitive | None = None, params: dict[str, Any] | None = None, effects: Effects | None = None, source_info: source_info_util.SourceInfo | None = None, ctx: JaxprEqnContext | None = None)` — [`L455`](../../../../../../raw/code/jax/jax/_src/core.py#L455)
  - `ctx` — [`L436`](../../../../../../raw/code/jax/jax/_src/core.py#L436)
  - `effects` — [`L428`](../../../../../../raw/code/jax/jax/_src/core.py#L428)
  - `invars` — [`L424`](../../../../../../raw/code/jax/jax/_src/core.py#L424)
  - `outvars` — [`L425`](../../../../../../raw/code/jax/jax/_src/core.py#L425)
  - `params` — [`L427`](../../../../../../raw/code/jax/jax/_src/core.py#L427)
  - `primitive` — [`L426`](../../../../../../raw/code/jax/jax/_src/core.py#L426)
  - `source_info` — [`L435`](../../../../../../raw/code/jax/jax/_src/core.py#L435)
- protocol/private: `__init__`[`L442`](../../../../../../raw/code/jax/jax/_src/core.py#L442), `__repr__`[`L452`](../../../../../../raw/code/jax/jax/_src/core.py#L452), `__slots__`[`L439`](../../../../../../raw/code/jax/jax/_src/core.py#L439)
- uses (calls/refs, reference-scoped): [`Primitive`](core.md#Primitive), [`Var`](core.md#Var), [`JaxprPpContext`](core.md#JaxprPpContext), [`Atom`](core.md#Atom), [`SourceInfo`](source_info_util.md#SourceInfo), [`JaxprPpSettings`](core.md#JaxprPpSettings), [`pp_eqn`](core.md#pp_eqn), [`Effects`](core.md#Effects), [`JaxprEqnContext`](core.md#JaxprEqnContext)
- used by: [`jaxpr_subcomp`](pallas/mosaic/lowering.md#jaxpr_subcomp), [`lower_jaxpr_to_mosaic_gpu`](pallas/mosaic_gpu/lowering.md#lower_jaxpr_to_mosaic_gpu), [`_check_jaxpr`](core.md#_check_jaxpr), [`eval_jaxpr`](core.md#eval_jaxpr), [`_eval_jaxpr_discharge_state`](state/discharge.md#_eval_jaxpr_discharge_state), [`_pull_block_transform`](pallas/fuser/block_spec.md#_pull_block_transform), [`jaxpr_type_signature`](../experimental/key_reuse/_core.md#jaxpr_type_signature), [`kernel_fn`](pallas/fuser/block_spec.md#make_kernel_function.kernel_fn), [`lower_jaxpr_to_triton_ir`](pallas/triton/lowering.md#lower_jaxpr_to_triton_ir), [`physicalize_interp`](pallas/fuser/fusible_dtype.md#physicalize_interp), [`_custom_evaluate_jaxpr`](pallas/fuser/custom_evaluate.md#_custom_evaluate_jaxpr), [`_partial_eval_jaxpr_custom_rule`](shard_map.md#_partial_eval_jaxpr_custom_rule), [`eval_jaxpr_recursive`](pallas/hlo_interpreter.md#eval_jaxpr_recursive), [`_scan_partial_eval_custom`](lax/control_flow/loops.md#_scan_partial_eval_custom), [`_insert_reduce_precision`](ad_checkpoint.md#_insert_reduce_precision), [`_pp_eqn`](core.md#_pp_eqn), [`_push_block_spec_jaxpr`](pallas/fuser/block_spec.md#_push_block_spec_jaxpr), [`compute_usage`](pallas/fuser/block_spec.md#compute_usage), [`eqns`](core.md#Jaxpr.eqns), [`fuse_jaxpr`](pallas/fuser/jaxpr_fusion.md#fuse_jaxpr), [`pp_toplevel_jaxpr`](core.md#pp_toplevel_jaxpr), [`_estimate_resources`](pallas/mosaic_gpu/lowering.md#_estimate_resources), [`_saved_residuals`](ad_checkpoint.md#_saved_residuals), [`pattern_match_scan_to_fori_loop`](pallas/utils.md#pattern_match_scan_to_fori_loop), [`_scan_dce_rule`](lax/control_flow/loops.md#_scan_dce_rule), [`_shard_map_dce`](shard_map.md#_shard_map_dce), [`_custom_vjp_call_dce`](custom_derivatives.md#_custom_vjp_call_dce), [`jaxpr_const_args`](core.md#jaxpr_const_args), [`_remat_opt_dce`](custom_derivatives.md#_remat_opt_dce), [`__init__`](core.md#Jaxpr.__init__), [`new_jaxpr_eqn`](core.md#new_jaxpr_eqn), [`_while_partial_eval_custom`](lax/control_flow/loops.md#_while_partial_eval_custom), [`pp_eqn`](core.md#pp_eqn), [`_custom_jvp_call_dce`](custom_derivatives.md#_custom_jvp_call_dce), [`_semaphore_signal_pp_eqn`](pallas/primitives.md#_semaphore_signal_pp_eqn), [`dce_jaxpr_pjit_rule`](pjit.md#dce_jaxpr_pjit_rule), [`remat_dce`](ad_checkpoint.md#remat_dce), [`_copy_smem_to_gmem_pp_eqn`](pallas/mosaic_gpu/primitives.md#_copy_smem_to_gmem_pp_eqn), [`_pjit_pp_rule`](pjit.md#_pjit_pp_rule), [`_cond_dce_rule`](lax/control_flow/conditionals.md#_cond_dce_rule)  (+24 more)

### `JaxprEqnContext`
- def: [`jax/_src/core.py:368`](../../../../../../raw/code/jax/jax/_src/core.py#L368)
- signature: `class JaxprEqnContext:`
- members:
  - `manager(self)` — [`L408`](../../../../../../raw/code/jax/jax/_src/core.py#L408)
  - `compute_type` — [`L374`](../../../../../../raw/code/jax/jax/_src/core.py#L374)
  - `cur_abstract_mesh` — [`L377`](../../../../../../raw/code/jax/jax/_src/core.py#L377)
  - `remove_size_one_mesh_axis` — [`L378`](../../../../../../raw/code/jax/jax/_src/core.py#L378)
  - `threefry_partitionable` — [`L375`](../../../../../../raw/code/jax/jax/_src/core.py#L375)
  - `xla_metadata` — [`L376`](../../../../../../raw/code/jax/jax/_src/core.py#L376)
- protocol/private: `__new__`[`L393`](../../../../../../raw/code/jax/jax/_src/core.py#L393), `__repr__`[`L411`](../../../../../../raw/code/jax/jax/_src/core.py#L411), `__slots__`[`L370`](../../../../../../raw/code/jax/jax/_src/core.py#L370), `_create`[`L382`](../../../../../../raw/code/jax/jax/_src/core.py#L382)
- uses (calls/refs, reference-scoped): [`AbstractMesh`](mesh.md#AbstractMesh), [`get_abstract_mesh`](mesh.md#get_abstract_mesh), [`immutable`](util.md#immutable), [`JaxprEqnContextManager`](core.md#JaxprEqnContextManager), [`remove_size_one_mesh_axis_from_type`](config.md#remove_size_one_mesh_axis_from_type), [`weak_value_interner`](util.md#weak_value_interner), [`threefry_partitionable`](config.md#threefry_partitionable), [`compute_on_context_manager`](config.md#compute_on_context_manager)
- used by: [`jaxpr_subcomp`](pallas/mosaic/lowering.md#jaxpr_subcomp), [`_check_jaxpr`](core.md#_check_jaxpr), [`eval_jaxpr`](core.md#eval_jaxpr), [`_eval_jaxpr_discharge_state`](state/discharge.md#_eval_jaxpr_discharge_state), [`physicalize_interp`](pallas/fuser/fusible_dtype.md#physicalize_interp), [`_custom_evaluate_jaxpr`](pallas/fuser/custom_evaluate.md#_custom_evaluate_jaxpr), [`eval_jaxpr_recursive`](pallas/hlo_interpreter.md#eval_jaxpr_recursive), [`replace`](core.md#JaxprEqn.replace), [`ctx`](core.md#JaxprEqn.ctx), [`current_jaxpr_eqn_context`](core.md#current_jaxpr_eqn_context)

### `JaxprEqnContextManager`
- def: [`jax/_src/core.py:335`](../../../../../../raw/code/jax/jax/_src/core.py#L335)
- signature: `class JaxprEqnContextManager:`
- protocol/private: `__enter__`[`L343`](../../../../../../raw/code/jax/jax/_src/core.py#L343), `__exit__`[`L358`](../../../../../../raw/code/jax/jax/_src/core.py#L358), `__init__`[`L340`](../../../../../../raw/code/jax/jax/_src/core.py#L340), `__slots__`[`L336`](../../../../../../raw/code/jax/jax/_src/core.py#L336)
- uses (calls/refs, reference-scoped): [`remove_size_one_mesh_axis_from_type`](config.md#remove_size_one_mesh_axis_from_type), [`abstract_mesh_context_manager`](config.md#abstract_mesh_context_manager), [`threefry_partitionable`](config.md#threefry_partitionable), [`compute_on_context_manager`](config.md#compute_on_context_manager), [`xla_metadata_context_manager`](config.md#xla_metadata_context_manager)
- used by: [`manager`](core.md#JaxprEqnContext.manager)

### `JaxprPpContext`
- def: [`jax/_src/core.py:4096`](../../../../../../raw/code/jax/jax/_src/core.py#L4096)
- members:
  - `suggest_same_var_names(self, for_vars: Sequence[Atom], like_vars: Sequence[Atom])` — [`L4110`](../../../../../../raw/code/jax/jax/_src/core.py#L4110) — Suggests the names for `for_vars` to match those of `like_vars`.
  - `shared_jaxpr_names` — [`L4100`](../../../../../../raw/code/jax/jax/_src/core.py#L4100)
  - `shared_jaxprs` — [`L4099`](../../../../../../raw/code/jax/jax/_src/core.py#L4099)
  - `var_names` — [`L4097`](../../../../../../raw/code/jax/jax/_src/core.py#L4097)
- protocol/private: `__init__`[`L4102`](../../../../../../raw/code/jax/jax/_src/core.py#L4102)
- uses (calls/refs, reference-scoped): [`Jaxpr`](core.md#Jaxpr), [`Var`](core.md#Var), [`Atom`](core.md#Atom), [`pretty_print`](core.md#Var.pretty_print), [`_encode_digits_alphabetic`](core.md#_encode_digits_alphabetic)
- used by: [`_check_jaxpr`](core.md#_check_jaxpr), [`_deprecations`](../core.md#_deprecations), [`_pp_eqn`](core.md#_pp_eqn), [`pp_ref_transforms`](state/primitives.md#pp_ref_transforms), [`pp_jaxpr_skeleton`](core.md#pp_jaxpr_skeleton), [`pp_toplevel_jaxpr`](core.md#pp_toplevel_jaxpr), [`_trace_kernel_to_jaxpr`](pallas/pallas_call.md#_trace_kernel_to_jaxpr), [`pp_vars`](core.md#pp_vars), [`pp_var`](core.md#pp_var), [`pp_jaxprs`](core.md#pp_jaxprs), [`pp_eqn`](core.md#pp_eqn), [`_semaphore_signal_pp_eqn`](pallas/primitives.md#_semaphore_signal_pp_eqn), [`pp_kv_pairs`](core.md#pp_kv_pairs), [`_copy_smem_to_gmem_pp_eqn`](pallas/mosaic_gpu/primitives.md#_copy_smem_to_gmem_pp_eqn), [`_pjit_pp_rule`](pjit.md#_pjit_pp_rule), [`pp_jaxpr`](core.md#pp_jaxpr), [`_copy_gmem_to_smem_pp_eqn`](pallas/mosaic_gpu/primitives.md#_copy_gmem_to_smem_pp_eqn), [`_dma_start_pp_eqn`](pallas/mosaic/primitives.md#_dma_start_pp_eqn), [`_jaxpr_call_pp_eqn`](pallas/primitives.md#_jaxpr_call_pp_eqn), [`_pp_slice`](state/indexing.md#_pp_slice), [`_semaphore_wait_pp_eqn`](pallas/primitives.md#_semaphore_wait_pp_eqn), [`pp_shared_jaxpr`](core.md#pp_shared_jaxpr), [`pretty_print`](state/types.md#Transform.pretty_print), [`_pretty_print`](state/types.md#RefEffect._pretty_print), [`pretty_print`](state/indexing.md#NDIndexer.pretty_print), [`pp_kv_pair`](core.md#pp_kv_pair), [`_dma_wait_pp_eqn`](pallas/mosaic/primitives.md#_dma_wait_pp_eqn), [`_wgmma_ref_pp_eqn`](pallas/mosaic_gpu/primitives.md#_wgmma_ref_pp_eqn), [`_barrier_arrive_pp_eqn`](pallas/mosaic_gpu/primitives.md#_barrier_arrive_pp_eqn), [`_barrier_test_pp_eqn`](pallas/mosaic_gpu/primitives.md#_barrier_test_pp_eqn), [`_barrier_wait_pp_eqn`](pallas/mosaic_gpu/primitives.md#_barrier_wait_pp_eqn), [`_core_map_discharge_rule`](pallas/core.md#_core_map_discharge_rule), [`pp_eqns`](core.md#pp_eqns), [`pp_jaxpr_eqn_range`](core.md#pp_jaxpr_eqn_range), [`_custom_jvp_call_pp_rule`](custom_derivatives.md#_custom_jvp_call_pp_rule), [`_custom_vjp_call_pp_rule`](custom_derivatives.md#_custom_vjp_call_pp_rule), [`check_type`](core.md#check_type), [`_error_if_non_ref_consts`](pallas/mpmd.md#_error_if_non_ref_consts), [`pp_aval`](core.md#pp_aval), [`_pp_transforms`](state/primitives.md#_pp_transforms)  (+15 more)

### `JaxprPpSettings`
- def: [`jax/_src/core.py:4079`](../../../../../../raw/code/jax/jax/_src/core.py#L4079)
- members:
  - `custom_pp_eqn_rules` — [`L4083`](../../../../../../raw/code/jax/jax/_src/core.py#L4083)
  - `name_stack` — [`L4082`](../../../../../../raw/code/jax/jax/_src/core.py#L4082)
  - `print_effects` — [`L4084`](../../../../../../raw/code/jax/jax/_src/core.py#L4084)
  - `print_shapes` — [`L4080`](../../../../../../raw/code/jax/jax/_src/core.py#L4080)
  - `source_info` — [`L4081`](../../../../../../raw/code/jax/jax/_src/core.py#L4081)
- used by: [`_check_jaxpr`](core.md#_check_jaxpr), [`_deprecations`](../core.md#_deprecations), [`_pp_eqn`](core.md#_pp_eqn), [`pp_jaxpr_skeleton`](core.md#pp_jaxpr_skeleton), [`pp_toplevel_jaxpr`](core.md#pp_toplevel_jaxpr), [`pp_jaxprs`](core.md#pp_jaxprs), [`pp_eqn`](core.md#pp_eqn), [`_semaphore_signal_pp_eqn`](pallas/primitives.md#_semaphore_signal_pp_eqn), [`pp_kv_pairs`](core.md#pp_kv_pairs), [`_copy_smem_to_gmem_pp_eqn`](pallas/mosaic_gpu/primitives.md#_copy_smem_to_gmem_pp_eqn), [`_pjit_pp_rule`](pjit.md#_pjit_pp_rule), [`pp_jaxpr`](core.md#pp_jaxpr), [`_copy_gmem_to_smem_pp_eqn`](pallas/mosaic_gpu/primitives.md#_copy_gmem_to_smem_pp_eqn), [`_dma_start_pp_eqn`](pallas/mosaic/primitives.md#_dma_start_pp_eqn), [`_jaxpr_call_pp_eqn`](pallas/primitives.md#_jaxpr_call_pp_eqn), [`_semaphore_wait_pp_eqn`](pallas/primitives.md#_semaphore_wait_pp_eqn), [`pp_shared_jaxpr`](core.md#pp_shared_jaxpr), [`pp_kv_pair`](core.md#pp_kv_pair), [`_dma_wait_pp_eqn`](pallas/mosaic/primitives.md#_dma_wait_pp_eqn), [`_wgmma_ref_pp_eqn`](pallas/mosaic_gpu/primitives.md#_wgmma_ref_pp_eqn), [`_barrier_arrive_pp_eqn`](pallas/mosaic_gpu/primitives.md#_barrier_arrive_pp_eqn), [`_barrier_test_pp_eqn`](pallas/mosaic_gpu/primitives.md#_barrier_test_pp_eqn), [`_barrier_wait_pp_eqn`](pallas/mosaic_gpu/primitives.md#_barrier_wait_pp_eqn), [`pp_eqns`](core.md#pp_eqns), [`pp_jaxpr_eqn_range`](core.md#pp_jaxpr_eqn_range), [`_custom_jvp_call_pp_rule`](custom_derivatives.md#_custom_jvp_call_pp_rule), [`_custom_vjp_call_pp_rule`](custom_derivatives.md#_custom_vjp_call_pp_rule), [`check_type`](core.md#check_type), [`ctx_factory`](core.md#check_jaxpr.ctx_factory), [`CustomPpEqnRule`](core.md#CustomPpEqnRule), [`__repr__`](core.md#JaxprEqn.__repr__), [`JaxprPpSettings`](../core.md#JaxprPpSettings)

### `JaxprTypeError`  ·  implements/extends TypeError
- def: [`jax/_src/core.py:3560`](../../../../../../raw/code/jax/jax/_src/core.py#L3560)
- signature: `class JaxprTypeError(TypeError):`
- uses (calls/refs, reference-scoped): [`set_module`](util.md#set_module)
- used by: [`_check_jaxpr`](core.md#_check_jaxpr), [`_deprecations`](../core.md#_deprecations), [`write`](core.md#_check_jaxpr.write), [`check_jaxpr`](core.md#check_jaxpr), [`read`](core.md#_check_jaxpr.read), [`_scan_typecheck`](lax/control_flow/loops.md#_scan_typecheck), [`_shard_map_typecheck`](shard_map.md#_shard_map_typecheck), [`_check_call`](core.md#_check_call), [`_cond_typecheck`](lax/control_flow/conditionals.md#_cond_typecheck), [`_while_loop_abstract_eval`](lax/control_flow/loops.md#_while_loop_abstract_eval), [`_check_closed_call`](core.md#_check_closed_call), [`_typecheck_param`](lax/control_flow/common.md#_typecheck_param), [`JaxprTypeError`](../core.md#JaxprTypeError)

### `ManualAxisType`
- def: [`jax/_src/core.py:2334`](../../../../../../raw/code/jax/jax/_src/core.py#L2334)
- signature: `class ManualAxisType:`
- members:
  - `empty(self)` — [`L2387`](../../../../../../raw/code/jax/jax/_src/core.py#L2387)
  - `invarying(self, mesh)` — [`L2390`](../../../../../../raw/code/jax/jax/_src/core.py#L2390)
  - `to_ct_mat(self)` — [`L2383`](../../../../../../raw/code/jax/jax/_src/core.py#L2383)
  - `update(self, **kwargs)` — [`L2374`](../../../../../../raw/code/jax/jax/_src/core.py#L2374)
  - `vur(self)` — [`L2395`](../../../../../../raw/code/jax/jax/_src/core.py#L2395)
  - `reduced` — [`L2339`](../../../../../../raw/code/jax/jax/_src/core.py#L2339)
  - `unreduced` — [`L2338`](../../../../../../raw/code/jax/jax/_src/core.py#L2338)
  - `varying` — [`L2337`](../../../../../../raw/code/jax/jax/_src/core.py#L2337)
- protocol/private: `__getnewargs_ex__`[`L2370`](../../../../../../raw/code/jax/jax/_src/core.py#L2370), `__new__`[`L2350`](../../../../../../raw/code/jax/jax/_src/core.py#L2350), `__repr__`[`L2366`](../../../../../../raw/code/jax/jax/_src/core.py#L2366), `__slots__`[`L2335`](../../../../../../raw/code/jax/jax/_src/core.py#L2335), `_create`[`L2343`](../../../../../../raw/code/jax/jax/_src/core.py#L2343)
- uses (calls/refs, reference-scoped): [`immutable`](util.md#immutable), [`weak_value_interner`](util.md#weak_value_interner), [`empty_mat`](core.md#empty_mat)
- used by: [`process_shard_map`](shard_map.md#ShardMapTrace.process_shard_map), [`broadcast_to`](lax/lax.md#broadcast_to), [`_unshard_shaped_array`](shard_map.md#_unshard_shaped_array), [`_inout_vma_error`](shard_map.md#_inout_vma_error), [`_shard_shaped_array`](shard_map.md#_shard_shaped_array), [`__new__`](core.md#ShapedArray.__new__), [`__init__`](shard_map.md#ShardMapTracer.__init__), [`to_concrete_value`](shard_map.md#ShardMapTracer.to_concrete_value), [`_attention_out_aval`](cudnn/fused_attention_stablehlo.md#_attention_out_aval), [`__init__`](core.md#ShapeDtypeStruct.__init__), [`__str__`](shard_map.md#ShardMapTracer.__str__), [`_axis_index_effectful_abstract_eval`](lax/parallel.md#_axis_index_effectful_abstract_eval), [`_unmentioned`](shard_map.md#_unmentioned), [`_valid_repeats`](shard_map.md#_valid_repeats), [`manual_rule`](lax/utils.md#manual_rule), [`empty_mat`](core.md#empty_mat), [`used_axis_names`](shard_map.md#used_axis_names), [`_spec_to_mat`](shard_map.md#_spec_to_mat), [`mat`](shard_map.md#ShardMapTracer.mat)

### `MemorySpace`
- def: [`jax/_src/core.py:2178`](../../../../../../raw/code/jax/jax/_src/core.py#L2178)
- members:
  - `Any` — [`L2181`](../../../../../../raw/code/jax/jax/_src/core.py#L2181)
  - `Device` — [`L2179`](../../../../../../raw/code/jax/jax/_src/core.py#L2179)
  - `Host` — [`L2180`](../../../../../../raw/code/jax/jax/_src/core.py#L2180)
- protocol/private: `__hash__`[`L2188`](../../../../../../raw/code/jax/jax/_src/core.py#L2188), `__repr__`[`L2183`](../../../../../../raw/code/jax/jax/_src/core.py#L2183)
- used by: [`to_block_mapping`](pallas/core.md#BlockSpec.to_block_mapping), [`_get_memory_space_from_aval`](pallas/mosaic/pallas_call_registration.md#_get_memory_space_from_aval), [`device_put`](api.md#device_put), [`memory_space_to_tpu_memory_space`](pallas/mosaic/core.md#memory_space_to_tpu_memory_space), [`broadcast_to`](lax/lax.md#broadcast_to), [`_get_sds`](pallas/core.md#_get_sds), [`_pallas_call_abstract_eval`](pallas/pallas_call.md#_pallas_call_abstract_eval), [`_check_sharding`](api.md#_check_sharding), [`__new__`](core.md#ShapedArray.__new__), [`_memory_space_to_mosaic_attribute`](pallas/mosaic/lowering.md#_memory_space_to_mosaic_attribute), [`TPU_MEMORY_SPACE_IDXS`](pallas/mosaic/interpret/interpret_pallas_call.md#TPU_MEMORY_SPACE_IDXS.TPU_MEMORY_SPACE_IDXS), [`_mpmd_map_fallback_lowering`](pallas/mpmd.md#_mpmd_map_fallback_lowering), [`with_memory_space_constraint`](pallas/mosaic/primitives.md#with_memory_space_constraint), [`call_shape_dtype_sharding_rule`](lax/utils.md#call_shape_dtype_sharding_rule), [`_resolve_memory_spaces`](pallas/mosaic/pallas_call_registration.md#_resolve_memory_spaces), [`host_idx`](pallas/mosaic/interpret/interpret_pallas_call.md#host_idx), [`_deserialize_aval`](export/serialization.md#_deserialize_aval), [`_ref_abstract_eval`](core.md#_ref_abstract_eval), [`_remove_memory_space_abstract_eval`](pallas/mosaic/interpret/interpret_pallas_call.md#_remove_memory_space_abstract_eval), [`_get_memory_spaces_from_avals`](pallas/mosaic/pallas_call_registration.md#_get_memory_spaces_from_avals), [`check_avals_context_mesh`](core.md#check_avals_context_mesh), [`get_memory_space_aval`](pallas/core.md#get_memory_space_aval), [`__init__`](state/types.md#AbstractRef.__init__), [`_aval_to_ref_aval`](pallas/mpmd.md#_aval_to_ref_aval), [`str_short_aval`](core.md#str_short_aval), [`TPUMemorySpace`](pallas/mosaic/lowering.md#TPUMemorySpace), [`AnyMemorySpace`](pallas/mosaic/lowering.md#AnyMemorySpace), [`_memory_space_to_enum`](export/serialization.md#_memory_space_to_enum), [`mem_space_to_kind`](core.md#mem_space_to_kind), [`_default_memory_space_rule`](lax/utils.md#_default_memory_space_rule), [`mem_kind_to_space`](core.md#mem_kind_to_space), [`__getattr__`](pallas/mosaic/core.md#MemorySpace.__getattr__), [`_deprecations`](../experimental/pallas/tpu.md#_deprecations), [`HOST`](../experimental/pallas/tpu.md#HOST), [`HOST`](pallas/mosaic/lowering.md#HOST), [`HOST`](../experimental/pallas/__init__.md#HOST), [`memory`](../__init__.md#memory)

### `MutableQuasiDynamicData`
- def: [`jax/_src/core.py:2007`](../../../../../../raw/code/jax/jax/_src/core.py#L2007)
- signature: `class MutableQuasiDynamicData:`
- members:
  - `update(self, val)` — [`L2012`](../../../../../../raw/code/jax/jax/_src/core.py#L2012)
  - `cur_val` — [`L2010`](../../../../../../raw/code/jax/jax/_src/core.py#L2010)
  - `init_val` — [`L2009`](../../../../../../raw/code/jax/jax/_src/core.py#L2009)
- protocol/private: `__init__`[`L2008`](../../../../../../raw/code/jax/jax/_src/core.py#L2008), `__repr__`[`L2015`](../../../../../../raw/code/jax/jax/_src/core.py#L2015)
- uses (calls/refs, reference-scoped): [`QuasiDynamicData`](core.md#QuasiDynamicData)
- used by: [`write`](core.md#_check_jaxpr.write), [`mutable_qdd`](core.md#MutableTypecheckVal.mutable_qdd), [`mutable_qdd`](core.md#AvalMutableQDD.mutable_qdd)

### `MutableTypecheckVal`
- def: [`jax/_src/core.py:3613`](../../../../../../raw/code/jax/jax/_src/core.py#L3613)
- signature: `class MutableTypecheckVal:`
- members:
  - `aval` — [`L3614`](../../../../../../raw/code/jax/jax/_src/core.py#L3614)
  - `mutable_qdd` — [`L3615`](../../../../../../raw/code/jax/jax/_src/core.py#L3615)
- uses (calls/refs, reference-scoped): [`AbstractValue`](core.md#AbstractValue), [`weakref_lru_cache`](util.md#weakref_lru_cache), [`MutableQuasiDynamicData`](core.md#MutableQuasiDynamicData)
- used by: [`_check_jaxpr`](core.md#_check_jaxpr), [`write`](core.md#_check_jaxpr.write), [`read`](core.md#_check_jaxpr.read), [`_check_call`](core.md#_check_call), [`check_type`](core.md#check_type)

### `NamedAxisEffect`  ·  implements/extends Effect
- def: [`jax/_src/core.py:3462`](../../../../../../raw/code/jax/jax/_src/core.py#L3462)
- doc: A side-effect introducing a new named axis into the current scope.
- signature: `class NamedAxisEffect(effects.Effect):`
- members:
  - `name` — [`L3464`](../../../../../../raw/code/jax/jax/_src/core.py#L3464)
- uses (calls/refs, reference-scoped): [`AxisName`](core.md#AxisName)
- used by: [`_check_jaxpr`](core.md#_check_jaxpr), [`_lower_to_custom_call`](pallas/mosaic/pallas_call_registration.md#_lower_to_custom_call), [`_semaphore_signal_abstract_eval`](pallas/primitives.md#_semaphore_signal_abstract_eval), [`_axis_index_effectful_abstract_eval`](lax/parallel.md#_axis_index_effectful_abstract_eval), [`_core_map_abstract_eval`](pallas/core.md#_core_map_abstract_eval), [`_mpmd_map_abstract_eval`](pallas/mpmd.md#_mpmd_map_abstract_eval), [`_allreduce_effectful_abstract_eval`](lax/parallel.md#_allreduce_effectful_abstract_eval), [`_get_dma_effects`](pallas/mosaic/primitives.md#_get_dma_effects), [`_get_nonlocal_axis_names`](pallas/mosaic/lowering.md#MosaicGridMapping._get_nonlocal_axis_names), [`filter_named_axis_effects`](core.md#filter_named_axis_effects), [`_all_to_all_effectful_abstract_eval`](lax/parallel.md#_all_to_all_effectful_abstract_eval), [`_psum_invariant_abstract_eval`](lax/parallel.md#_psum_invariant_abstract_eval), [`_all_gather_effectful_abstract_eval`](lax/parallel.md#_all_gather_effectful_abstract_eval), [`_psend_abstract_eval`](lax/parallel.md#_psend_abstract_eval), [`_reduce_scatter_effectful_abstract_eval`](lax/parallel.md#_reduce_scatter_effectful_abstract_eval), [`_all_gather_invariant_effectful_abstract_eval`](lax/parallel.md#_all_gather_invariant_effectful_abstract_eval), [`_ragged_all_to_all_effectful_abstract_eval`](lax/parallel.md#_ragged_all_to_all_effectful_abstract_eval), [`_unreduced_psum_abstract_eval`](lax/parallel.md#_unreduced_psum_abstract_eval), [`_all_gather_reduced_effectful_abstract_eval`](lax/parallel.md#_all_gather_reduced_effectful_abstract_eval), [`_has_effects`](ad_checkpoint.md#_has_effects), [`_precv_abstract_eval`](lax/parallel.md#_precv_abstract_eval), [`_unreduced_reduce_scatter_effectful_abstract_eval`](lax/parallel.md#_unreduced_reduce_scatter_effectful_abstract_eval)

### `OpaqueTraceState`
- def: [`jax/_src/core.py:4362`](../../../../../../raw/code/jax/jax/_src/core.py#L4362)
- signature: `class OpaqueTraceState:`
- protocol/private: `__eq__`[`L4366`](../../../../../../raw/code/jax/jax/_src/core.py#L4366), `__init__`[`L4363`](../../../../../../raw/code/jax/jax/_src/core.py#L4363), `_trace_ref`[`L4364`](../../../../../../raw/code/jax/jax/_src/core.py#L4364)
- used by: [`get_opaque_trace_state`](core.md#get_opaque_trace_state)

### `Primitive`
- def: [`jax/_src/core.py:639`](../../../../../../raw/code/jax/jax/_src/core.py#L639) — documented in [jax-_src-core](../../../concepts/jax-_src-core.md)
- members:
  - `abstract_eval(self, *args, **params)` — [`L738`](../../../../../../raw/code/jax/jax/_src/core.py#L738)
  - `bind(self, *args, **params)` — [`L659`](../../../../../../raw/code/jax/jax/_src/core.py#L659) — documented in [jax-_src-core](../../../concepts/jax-_src-core.md)
  - `bind_with_trace(self, trace, args, avals, params,)` — [`L708`](../../../../../../raw/code/jax/jax/_src/core.py#L708)
  - `def_abstract_eval(self, abstract_eval)` — [`L718`](../../../../../../raw/code/jax/jax/_src/core.py#L718)
  - `def_bind_with_trace(self, bind_with_trace)` — [`L730`](../../../../../../raw/code/jax/jax/_src/core.py#L730)
  - `def_effectful_abstract_eval(self, effectful_abstract_eval)` — [`L722`](../../../../../../raw/code/jax/jax/_src/core.py#L722)
  - `def_effectful_abstract_eval2(self, abstract_eval)` — [`L726`](../../../../../../raw/code/jax/jax/_src/core.py#L726)
  - `def_impl(self, impl)` — [`L714`](../../../../../../raw/code/jax/jax/_src/core.py#L714)
  - `get_bind_params(self, params)` — [`L742`](../../../../../../raw/code/jax/jax/_src/core.py#L742)
  - `impl(self, *args, **params)` — [`L734`](../../../../../../raw/code/jax/jax/_src/core.py#L734)
  - `is_high(self, *avals, **params)` — [`L750`](../../../../../../raw/code/jax/jax/_src/core.py#L750)
  - `to_lojax(self, *args, **params)` — [`L745`](../../../../../../raw/code/jax/jax/_src/core.py#L745)
  - `call_primitive` — [`L644`](../../../../../../raw/code/jax/jax/_src/core.py#L644)
  - `is_effectful` — [`L651`](../../../../../../raw/code/jax/jax/_src/core.py#L651)
  - `multiple_results` — [`L642`](../../../../../../raw/code/jax/jax/_src/core.py#L642)
  - `name` — [`L640`](../../../../../../raw/code/jax/jax/_src/core.py#L640)
  - `ref_allocating` — [`L650`](../../../../../../raw/code/jax/jax/_src/core.py#L650)
  - `ref_primitive` — [`L646`](../../../../../../raw/code/jax/jax/_src/core.py#L646)
  - `skip_canonicalization` — [`L648`](../../../../../../raw/code/jax/jax/_src/core.py#L648)
- protocol/private: `__init__`[`L653`](../../../../../../raw/code/jax/jax/_src/core.py#L653), `__repr__`[`L656`](../../../../../../raw/code/jax/jax/_src/core.py#L656)
- uses (calls/refs, reference-scoped): [`ShapedArray`](core.md#ShapedArray), [`typeof`](core.md#typeof), [`NamedSharding`](named_sharding.md#NamedSharding), [`PartitionSpec`](partition_spec.md#PartitionSpec), [`ClosedJaxpr`](core.md#ClosedJaxpr), [`Jaxpr`](core.md#Jaxpr), [`Tracer`](core.md#Tracer), [`get_abstract_mesh`](mesh.md#get_abstract_mesh), [`sharding`](core.md#ShapedArray.sharding), [`reshard`](pjit.md#reshard), [`ndim`](core.md#ShapedArray.ndim), [`trace_ctx`](core.md#trace_ctx), [`str_short`](core.md#ShapedArray.str_short), [`set_current_trace`](core.md#set_current_trace), [`HiPrimitive`](hijax.md#HiPrimitive), [`get_bind_params`](custom_derivatives.md#CustomVJPCallPrimitive.get_bind_params), [`to_lojax`](hijax.md#HiPrimitive.to_lojax), [`abstract_eval`](hijax.md#HiPrimitive.abstract_eval), [`_any_axis_manual`](mesh.md#BaseMesh._any_axis_manual), [`trace`](core.md#TracingContext.trace), [`_are_all_axes_auto_or_manual`](mesh.md#BaseMesh._are_all_axes_auto_or_manual), [`escaped_tracer_error`](core.md#escaped_tracer_error), [`is_high`](hijax.md#HiPrimitive.is_high), [`canonicalize_value`](dtypes.md#canonicalize_value), [`set_trace`](core.md#TracingContext.set_trace), [`CallPrimitive`](core.md#CallPrimitive), [`get_bind_params`](custom_derivatives.md#CustomJVPCallPrimitive.get_bind_params), [`_trace`](core.md#Tracer._trace), [`get_bind_params`](shard_map.md#ShardMapPrimitive.get_bind_params), [`get_bind_params`](core.md#CallPrimitive.get_bind_params), [`to_lojax`](custom_derivatives.md#CustomJVPCallPrimitive.to_lojax), [`to_lojax`](custom_derivatives.md#CustomVJPCallPrimitive.to_lojax), [`is_high`](core.md#Jaxpr.is_high), [`CustomJVPCallPrimitive`](custom_derivatives.md#CustomJVPCallPrimitive), [`CustomVJPCallPrimitive`](custom_derivatives.md#CustomVJPCallPrimitive), [`ShardMapPrimitive`](shard_map.md#ShardMapPrimitive), [`_effect_free_abstract_eval`](core.md#_effect_free_abstract_eval), [`_generic_effectful_abstract_eval`](core.md#_generic_effectful_abstract_eval), [`bind_with_trace`](shard_map.md#ShardMapPrimitive.bind_with_trace), [`bind_with_trace`](core.md#CallPrimitive.bind_with_trace)  (+6 more)
- used by: [`register_lowering_rule`](pallas/mosaic_gpu/lowering.md#register_lowering_rule), [`interpret_pallas_call`](pallas/mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call), [`select`](lax/lax.md#select), [`jaxpr_subcomp`](pallas/mosaic/lowering.md#jaxpr_subcomp), [`mul`](lax/lax.md#mul), [`register_lowering_rule`](pallas/mosaic/lowering.md#register_lowering_rule), [`lower_jaxpr_to_mosaic_gpu`](pallas/mosaic_gpu/lowering.md#lower_jaxpr_to_mosaic_gpu), [`_pallas_call_batching_rule`](pallas/pallas_call.md#_pallas_call_batching_rule), [`_check_jaxpr`](core.md#_check_jaxpr), [`eval_jaxpr`](core.md#eval_jaxpr), [`key_reuse_signatures`](../experimental/key_reuse/_core.md#key_reuse_signatures.key_reuse_signatures), [`scan_p`](lax/control_flow/loops.md#scan_p), [`_eval_jaxpr_discharge_state`](state/discharge.md#_eval_jaxpr_discharge_state), [`primitive`](core.md#JaxprEqn.primitive), [`cond_p`](lax/control_flow/conditionals.md#cond_p), [`_pull_block_transform`](pallas/fuser/block_spec.md#_pull_block_transform), [`broadcast_in_dim`](lax/lax.md#broadcast_in_dim), [`add`](lax/lax.md#add), [`scan`](lax/control_flow/loops.md#scan), [`_pallas_call_state_discharge_rule`](pallas/pallas_call.md#_pallas_call_state_discharge_rule), [`cond`](lax/control_flow/conditionals.md#cond), [`div`](lax/lax.md#div), [`while_p`](lax/control_flow/loops.md#while_p), [`io_callback`](callback.md#io_callback), [`_convert_element_type`](lax/lax.md#_convert_element_type), [`done_p`](lax/parallel.md#done_p), [`kernel_fn`](pallas/fuser/block_spec.md#make_kernel_function.kernel_fn), [`while_loop`](lax/control_flow/loops.md#while_loop), [`broadcast_in_dim_p`](lax/lax.md#broadcast_in_dim_p), [`lower_jaxpr_to_triton_ir`](pallas/triton/lowering.md#lower_jaxpr_to_triton_ir), [`bitwise_or`](lax/lax.md#bitwise_or), [`wrapped`](shard_map.md#_shard_map.wrapped), [`axis_index`](lax/parallel.md#axis_index), [`physicalize_interp`](pallas/fuser/fusible_dtype.md#physicalize_interp), [`_custom_evaluate_jaxpr`](pallas/fuser/custom_evaluate.md#_custom_evaluate_jaxpr), [`broadcasted_iota`](lax/lax.md#broadcasted_iota), [`wrapper`](pallas/mpmd.md#_mpmd_map.wrapper), [`bitwise_and`](lax/lax.md#bitwise_and), [`jit_p`](pjit.md#jit_p), [`sub`](lax/lax.md#sub)  (+916 more)

### `QuasiDynamicData`
- def: [`jax/_src/core.py:2018`](../../../../../../raw/code/jax/jax/_src/core.py#L2018)
- uses (calls/refs, reference-scoped): [`BoxTypeState`](hijax.md#BoxTypeState)
- used by: [`qdd`](core.md#AvalQDD.qdd), [`QDD`](hijax.md#QDD), [`initial_qdd`](core.md#Var.initial_qdd), [`final_qdd`](core.md#Var.final_qdd), [`__init__`](core.md#MutableQuasiDynamicData.__init__)

### `Ref`
- def: [`jax/_src/core.py:2775`](../../../../../../raw/code/jax/jax/_src/core.py#L2775)
- doc: Mutable array reference.
- signature: `class Ref(metaclass=RefMeta):`
- members:
  - `addupdate(self, x, idx=())` — [`L2809`](../../../../../../raw/code/jax/jax/_src/core.py#L2809)
  - `at(self)` — [`L2818`](../../../../../../raw/code/jax/jax/_src/core.py#L2818)
  - `unsafe_buffer_pointer(self)` — [`L2815`](../../../../../../raw/code/jax/jax/_src/core.py#L2815)
  - `aval` — [`L2799`](../../../../../../raw/code/jax/jax/_src/core.py#L2799)
  - `committed` — [`L2814`](../../../../../../raw/code/jax/jax/_src/core.py#L2814)
  - `dtype` — [`L2803`](../../../../../../raw/code/jax/jax/_src/core.py#L2803)
  - `format` — [`L2813`](../../../../../../raw/code/jax/jax/_src/core.py#L2813)
  - `ndim` — [`L2802`](../../../../../../raw/code/jax/jax/_src/core.py#L2802)
  - `shape` — [`L2800`](../../../../../../raw/code/jax/jax/_src/core.py#L2800)
  - `sharding` — [`L2812`](../../../../../../raw/code/jax/jax/_src/core.py#L2812)
  - `size` — [`L2801`](../../../../../../raw/code/jax/jax/_src/core.py#L2801)
- protocol/private: `__getitem__`[`L2806`](../../../../../../raw/code/jax/jax/_src/core.py#L2806), `__init__`[`L2787`](../../../../../../raw/code/jax/jax/_src/core.py#L2787), `__len__`[`L2808`](../../../../../../raw/code/jax/jax/_src/core.py#L2808), `__repr__`[`L2793`](../../../../../../raw/code/jax/jax/_src/core.py#L2793), `__setitem__`[`L2807`](../../../../../../raw/code/jax/jax/_src/core.py#L2807), `_aval`[`L2784`](../../../../../../raw/code/jax/jax/_src/core.py#L2784), `_committed`[`L2814`](../../../../../../raw/code/jax/jax/_src/core.py#L2814), `_refs`[`L2785`](../../../../../../raw/code/jax/jax/_src/core.py#L2785)
- uses (calls/refs, reference-scoped): [`AbstractRef`](state/types.md#AbstractRef), [`AbstractValue`](core.md#AbstractValue), [`is_high`](core.md#AbstractValue.is_high), [`RefMeta`](core.md#RefMeta), [`PyTree`](core.md#PyTree)
- used by: [`Device`](../__init__.md#Device), [`wrapper`](pallas/mpmd.md#_mpmd_map.wrapper), [`ref_swap`](state/primitives.md#ref_swap), [`ref_get`](state/primitives.md#ref_get), [`fetch_and_add`](pallas/mosaic/sc_primitives.md#fetch_and_add), [`ref_set`](state/primitives.md#ref_set), [`ref_addupdate`](state/primitives.md#ref_addupdate), [`sync_copy`](pallas/mosaic/pipeline.md#sync_copy), [`__all__`](../ref.md#__all__), [`_window_ref_at`](pallas/mosaic/pipeline.md#BufferedRef._window_ref_at), [`pjit_staging_rule`](pjit.md#pjit_staging_rule), [`_ref_impl`](core.md#_ref_impl), [`freeze`](core.md#freeze), [`free_ref`](core.md#free_ref), [`_ref_to_lojax`](core.md#_ref_to_lojax), [`zeros_like_abstract_ref`](state/types.md#zeros_like_abstract_ref), [`raise_val`](state/types.md#AbstractRef.raise_val), [`_empty_ref_to_lojax`](core.md#_empty_ref_to_lojax), [`REF`](pallas/mosaic/pipeline.md#REF), [`new_ref`](ref.md#new_ref)

### `RefMeta`  ·  implements/extends type
- def: [`jax/_src/core.py:2769`](../../../../../../raw/code/jax/jax/_src/core.py#L2769)
- signature: `class RefMeta(type):`
- protocol/private: `__instancecheck__`[`L2770`](../../../../../../raw/code/jax/jax/_src/core.py#L2770)
- uses (calls/refs, reference-scoped): [`AbstractRef`](state/types.md#AbstractRef), [`Tracer`](core.md#Tracer), [`aval`](core.md#Tracer.aval)
- used by: [`Ref`](core.md#Ref)

### `RemoveExplicitMeshAxisNamesContextManager`
- def: [`jax/_src/core.py:1542`](../../../../../../raw/code/jax/jax/_src/core.py#L1542)
- signature: `class RemoveExplicitMeshAxisNamesContextManager:`
- protocol/private: `__enter__`[`L1548`](../../../../../../raw/code/jax/jax/_src/core.py#L1548), `__exit__`[`L1554`](../../../../../../raw/code/jax/jax/_src/core.py#L1554), `__init__`[`L1545`](../../../../../../raw/code/jax/jax/_src/core.py#L1545), `__slots__`[`L1543`](../../../../../../raw/code/jax/jax/_src/core.py#L1543)
- uses (calls/refs, reference-scoped): [`trace_ctx`](core.md#trace_ctx), [`AxisName`](core.md#AxisName), [`axis_env`](core.md#TracingContext.axis_env), [`set_axis_env`](core.md#TracingContext.set_axis_env)
- used by: [`remove_explicit_mesh_axis_names`](core.md#remove_explicit_mesh_axis_names)

### `SetCurrentTraceContextManager`
- def: [`jax/_src/core.py:1466`](../../../../../../raw/code/jax/jax/_src/core.py#L1466)
- signature: `class SetCurrentTraceContextManager:`
- protocol/private: `__enter__`[`L1473`](../../../../../../raw/code/jax/jax/_src/core.py#L1473), `__exit__`[`L1477`](../../../../../../raw/code/jax/jax/_src/core.py#L1477), `__init__`[`L1469`](../../../../../../raw/code/jax/jax/_src/core.py#L1469), `__slots__`[`L1467`](../../../../../../raw/code/jax/jax/_src/core.py#L1467)
- uses (calls/refs, reference-scoped): [`trace_ctx`](core.md#trace_ctx), [`trace`](core.md#TracingContext.trace), [`set_trace`](core.md#TracingContext.set_trace), [`leaked_tracer_error`](core.md#leaked_tracer_error), [`maybe_find_leaked_tracers`](core.md#maybe_find_leaked_tracers), [`check_tracer_leaks`](config.md#check_tracer_leaks)
- used by: [`set_current_trace`](core.md#set_current_trace)

### `ShapeDtypeStruct`
- def: [`jax/_src/core.py:3844`](../../../../../../raw/code/jax/jax/_src/core.py#L3844)
- members:
  - `format(self)` — [`L3951`](../../../../../../raw/code/jax/jax/_src/core.py#L3951)
  - `like(cls, x)` — [`L3913`](../../../../../../raw/code/jax/jax/_src/core.py#L3913)
  - `sharding(self)` — [`L3936`](../../../../../../raw/code/jax/jax/_src/core.py#L3936)
  - `update(self, **kwargs)` — [`L3985`](../../../../../../raw/code/jax/jax/_src/core.py#L3985)
  - `dtype` — [`L3858`](../../../../../../raw/code/jax/jax/_src/core.py#L3858)
  - `is_ref` — [`L3863`](../../../../../../raw/code/jax/jax/_src/core.py#L3863)
  - `manual_axis_type` — [`L3862`](../../../../../../raw/code/jax/jax/_src/core.py#L3862)
  - `ndim` — [`L3910`](../../../../../../raw/code/jax/jax/_src/core.py#L3910)
  - `shape` — [`L3857`](../../../../../../raw/code/jax/jax/_src/core.py#L3857)
  - `size` — [`L3909`](../../../../../../raw/code/jax/jax/_src/core.py#L3909)
  - `weak_type` — [`L3861`](../../../../../../raw/code/jax/jax/_src/core.py#L3861)
- protocol/private: `__eq__`[`L3972`](../../../../../../raw/code/jax/jax/_src/core.py#L3972), `__hash__`[`L3981`](../../../../../../raw/code/jax/jax/_src/core.py#L3981), `__init__`[`L3865`](../../../../../../raw/code/jax/jax/_src/core.py#L3865), `__len__`[`L3954`](../../../../../../raw/code/jax/jax/_src/core.py#L3954), `__repr__`[`L3960`](../../../../../../raw/code/jax/jax/_src/core.py#L3960), `__setattr__`[`L3898`](../../../../../../raw/code/jax/jax/_src/core.py#L3898), `__slots__`[`L3854`](../../../../../../raw/code/jax/jax/_src/core.py#L3854), `__str__`[`L3970`](../../../../../../raw/code/jax/jax/_src/core.py#L3970), `_dll`[`L3860`](../../../../../../raw/code/jax/jax/_src/core.py#L3860), `_sharding`[`L3859`](../../../../../../raw/code/jax/jax/_src/core.py#L3859)
- uses (calls/refs, reference-scoped): [`AbstractRef`](state/types.md#AbstractRef), [`issubdtype`](dtypes.md#issubdtype), [`NamedSharding`](named_sharding.md#NamedSharding), [`PartitionSpec`](partition_spec.md#PartitionSpec), [`shape`](core.md#ShapedArray.shape), [`dtype`](core.md#ShapedArray.dtype), [`Sharding`](sharding.md#Sharding), [`extended`](dtypes.md#extended), [`sharding`](core.md#ShapedArray.sharding), [`ArrayImpl`](array.md#ArrayImpl), [`set_module`](util.md#set_module), [`mat`](core.md#ShapedArray.mat), [`Format`](layout.md#Format), [`shaped_abstractify`](core.md#shaped_abstractify), [`HiType`](hijax.md#HiType), [`empty`](mesh.md#Mesh.empty), [`weak_type`](core.md#ShapedArray.weak_type), [`ManualAxisType`](core.md#ManualAxisType), [`get_concrete_mesh`](mesh.md#get_concrete_mesh), [`format`](array.md#ArrayImpl.format), [`AutoLayoutSingleton`](layout.md#AutoLayoutSingleton), [`_check_sharding`](core.md#_check_sharding)
- used by: [`_interpret_jaxpr`](pallas/mosaic/interpret/interpret_pallas_call.md#_interpret_jaxpr), [`interpret_pallas_call`](pallas/mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call), [`Device`](../__init__.md#Device), [`lower_jaxpr_to_module`](pallas/mosaic_gpu/lowering.md#lower_jaxpr_to_module), [`_run_scoped_lowering_rule`](pallas/mosaic_gpu/lowering.md#_run_scoped_lowering_rule), [`_splash_attention_bwd_dkv`](../experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.md#_splash_attention_bwd_dkv), [`_splash_attention_bwd_dq`](../experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.md#_splash_attention_bwd_dq), [`_allocate_for_aval`](pallas/mosaic_gpu/interpret/jaxpr_interpret.md#JaxprInterpreter._allocate_for_aval), [`mixed_matmul_kernel`](../experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.md#mixed_matmul_kernel), [`matmul_kernel`](../experimental/pallas/ops/gpu/blackwell_matmul_mgpu.md#matmul_kernel), [`all_gather_lhs_matmul`](../experimental/pallas/ops/gpu/collective_matmul_mgpu.md#all_gather_lhs_matmul), [`gmm`](../experimental/pallas/ops/tpu/megablox/gmm.md#gmm), [`tgmm`](../experimental/pallas/ops/tpu/megablox/gmm.md#tgmm), [`paged_attention`](../experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.md#paged_attention), [`_flash_attention_bwd_dkv`](../experimental/pallas/ops/tpu/flash_attention.md#_flash_attention_bwd_dkv), [`_flash_attention_bwd_dq`](../experimental/pallas/ops/tpu/flash_attention.md#_flash_attention_bwd_dq), [`ragged_paged_attention`](../experimental/pallas/ops/tpu/ragged_paged_attention/kernel.md#ragged_paged_attention), [`_flash_attention_impl`](../experimental/pallas/ops/tpu/flash_attention.md#_flash_attention_impl), [`_attention_bwd`](../experimental/pallas/ops/gpu/attention_mgpu.md#_attention_bwd), [`attention_with_pipeline_emitter`](../experimental/pallas/ops/gpu/attention_mgpu.md#attention_with_pipeline_emitter), [`_mha_backward`](../experimental/pallas/ops/gpu/attention.md#_mha_backward), [`ragged_dot_kernel`](../experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.md#ragged_dot_kernel), [`scratch_view`](pallas/mosaic_gpu/lowering.md#ModuleContext.scratch_view), [`matmul`](../experimental/pallas/ops/tpu/matmul.md#matmul), [`_kernel_to_module`](../experimental/mosaic/gpu/core.md#_kernel_to_module), [`mha`](../experimental/pallas/ops/gpu/attention.md#mha), [`transposed_ragged_dot`](../experimental/pallas/ops/gpu/transposed_ragged_dot_mgpu.md#transposed_ragged_dot), [`paged_attention_unbatched`](../experimental/pallas/ops/gpu/paged_attention.md#paged_attention_unbatched), [`_reduce_lowering_rule`](pallas/mosaic_gpu/lowering.md#_reduce_lowering_rule), [`all_gather`](../experimental/pallas/ops/gpu/all_gather_mgpu.md#all_gather), [`alloc_tmem`](pallas/mosaic_gpu/lowering.md#ModuleContext.alloc_tmem), [`reduce_scatter`](../experimental/pallas/ops/gpu/reduce_scatter_mgpu.md#reduce_scatter), [`_attention_forward`](../experimental/pallas/ops/gpu/attention_mgpu.md#_attention_forward), [`_get_sds`](pallas/core.md#_get_sds), [`TOKEN_SHAPE_DTYPE`](pallas/mosaic/interpret/interpret_pallas_call.md#TOKEN_SHAPE_DTYPE), [`from_array`](pallas/mosaic_gpu/interpret/shared_memory.md#HostAllocationRequest.from_array), [`philox_4x32_kernel`](../experimental/pallas/ops/tpu/random/philox.md#philox_4x32_kernel), [`_deallocate_barrier`](pallas/mosaic_gpu/interpret/gpu_callbacks.md#_deallocate_barrier), [`_store_slice_to_kernel_input`](pallas/mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call._execute_grid_for_core._body._store_slice_to_kernel_input), [`matmul`](../experimental/pallas/ops/gpu/hopper_matmul_mgpu.md#matmul)  (+34 more)

### `ShapedArray`
- def: [`jax/_src/core.py:2406`](../../../../../../raw/code/jax/jax/_src/core.py#L2406) — documented in [jax-_src-core](../../../concepts/jax-_src-core.md)
- members:
  - `lo_ty(self)` — [`L2455`](../../../../../../raw/code/jax/jax/_src/core.py#L2455)
  - `lower_val(self, val)` — [`L2453`](../../../../../../raw/code/jax/jax/_src/core.py#L2453)
  - `mat(self)` — [`L2458`](../../../../../../raw/code/jax/jax/_src/core.py#L2458)
  - `nospec(self, mesh, check_vma, all_names)` — [`L2549`](../../../../../../raw/code/jax/jax/_src/core.py#L2549)
  - `raise_val(self, val)` — [`L2454`](../../../../../../raw/code/jax/jax/_src/core.py#L2454)
  - `str_short(self, short_dtypes=False, mesh_axis_types=False)` — [`L2513`](../../../../../../raw/code/jax/jax/_src/core.py#L2513)
  - `strip_weak_type(self)` — [`L2537`](../../../../../../raw/code/jax/jax/_src/core.py#L2537)
  - `to_ct_aval(self)` — [`L2506`](../../../../../../raw/code/jax/jax/_src/core.py#L2506)
  - `to_tangent_aval(self)` — [`L2501`](../../../../../../raw/code/jax/jax/_src/core.py#L2501)
  - `update(self, shape=None, dtype=None, weak_type=None, **kwargs)` — [`L2461`](../../../../../../raw/code/jax/jax/_src/core.py#L2461)
  - `update_manual_axis_type(self, mat)` — [`L2524`](../../../../../../raw/code/jax/jax/_src/core.py#L2524)
  - `update_weak_type(self, weak_type)` — [`L2531`](../../../../../../raw/code/jax/jax/_src/core.py#L2531)
  - `array_abstraction_level` — [`L2410`](../../../../../../raw/code/jax/jax/_src/core.py#L2410)
  - `broadcast` — [`L2481`](../../../../../../raw/code/jax/jax/_src/core.py#L2481)
  - `dtype` — [`L2413`](../../../../../../raw/code/jax/jax/_src/core.py#L2413)
  - `manual_axis_type` — [`L2416`](../../../../../../raw/code/jax/jax/_src/core.py#L2416)
  - `memory_space` — [`L2417`](../../../../../../raw/code/jax/jax/_src/core.py#L2417)
  - `ndim` — [`L2476`](../../../../../../raw/code/jax/jax/_src/core.py#L2476)
  - `reshape` — [`L2483`](../../../../../../raw/code/jax/jax/_src/core.py#L2483)
  - `shape` — [`L2412`](../../../../../../raw/code/jax/jax/_src/core.py#L2412)
  - `sharding` — [`L2415`](../../../../../../raw/code/jax/jax/_src/core.py#L2415)
  - `size` — [`L2477`](../../../../../../raw/code/jax/jax/_src/core.py#L2477)
  - `transpose` — [`L2482`](../../../../../../raw/code/jax/jax/_src/core.py#L2482)
  - `weak_type` — [`L2414`](../../../../../../raw/code/jax/jax/_src/core.py#L2414)
- protocol/private: `__getnewargs_ex__`[`L2486`](../../../../../../raw/code/jax/jax/_src/core.py#L2486), `__new__`[`L2434`](../../../../../../raw/code/jax/jax/_src/core.py#L2434), `__repr__`[`L2493`](../../../../../../raw/code/jax/jax/_src/core.py#L2493), `__slots__`[`L2408`](../../../../../../raw/code/jax/jax/_src/core.py#L2408), `__str__`[`L2497`](../../../../../../raw/code/jax/jax/_src/core.py#L2497), `_bool`[`L2558`](../../../../../../raw/code/jax/jax/_src/core.py#L2558), `_complex`[`L2561`](../../../../../../raw/code/jax/jax/_src/core.py#L2561), `_create`[`L2422`](../../../../../../raw/code/jax/jax/_src/core.py#L2422), `_float`[`L2560`](../../../../../../raw/code/jax/jax/_src/core.py#L2560), `_hex`[`L2562`](../../../../../../raw/code/jax/jax/_src/core.py#L2562), `_index`[`L2564`](../../../../../../raw/code/jax/jax/_src/core.py#L2564), `_int`[`L2559`](../../../../../../raw/code/jax/jax/_src/core.py#L2559), `_iter`[`L2484`](../../../../../../raw/code/jax/jax/_src/core.py#L2484), `_len`[`L2518`](../../../../../../raw/code/jax/jax/_src/core.py#L2518), `_oct`[`L2563`](../../../../../../raw/code/jax/jax/_src/core.py#L2563), `_stripped_weak_type`[`L2418`](../../../../../../raw/code/jax/jax/_src/core.py#L2418)
- uses (calls/refs, reference-scoped): [`AbstractValue`](core.md#AbstractValue), [`PartitionSpec`](partition_spec.md#PartitionSpec), [`mesh`](named_sharding.md#NamedSharding.mesh), [`MemorySpace`](core.md#MemorySpace), [`canonicalize_shape`](core.md#canonicalize_shape), [`aval_method`](core.md#aval_method), [`ManualAxisType`](core.md#ManualAxisType), [`concretization_function_error`](core.md#concretization_function_error), [`immutable`](util.md#immutable), [`order_wrt_mesh`](core.md#order_wrt_mesh), [`primal_dtype_to_tangent_dtype`](core.md#primal_dtype_to_tangent_dtype), [`str_short_aval`](core.md#str_short_aval), [`Device`](core.md#MemorySpace.Device), [`get_sharding`](core.md#get_sharding), [`get_mat`](core.md#get_mat), [`weak_value_interner`](util.md#weak_value_interner), [`_empty_sharding`](core.md#_empty_sharding), [`empty_mat`](core.md#empty_mat), [`_dtype_object`](core.md#_dtype_object), [`empty`](core.md#ManualAxisType.empty), [`primal_sharding_to_cotangent_sharding`](core.md#primal_sharding_to_cotangent_sharding), [`get_memory_space`](core.md#get_memory_space)
- used by: [`bind`](core.md#Primitive.bind), [`interpret_pallas_call`](pallas/mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call), [`lower_jaxpr_into_pipelined_module`](pallas/mosaic/lowering.md#lower_jaxpr_into_pipelined_module), [`lower_jaxpr_to_module`](pallas/mosaic_gpu/lowering.md#lower_jaxpr_to_module), [`_pallas_call_batching_rule`](pallas/pallas_call.md#_pallas_call_batching_rule), [`to_block_mapping`](pallas/core.md#BlockSpec.to_block_mapping), [`get_grid_mapping`](pallas/core.md#get_grid_mapping), [`_eval_jaxpr_discharge_state`](state/discharge.md#_eval_jaxpr_discharge_state), [`_dot_general_lowering`](pallas/triton/lowering.md#_dot_general_lowering), [`_call_exported_lowering`](export/_export.md#_call_exported_lowering), [`_allocate_for_aval`](pallas/mosaic_gpu/interpret/jaxpr_interpret.md#JaxprInterpreter._allocate_for_aval), [`_pallas_call_state_discharge_rule`](pallas/pallas_call.md#_pallas_call_state_discharge_rule), [`auto_insert_reshard`](core.md#auto_insert_reshard), [`lower_pipelined_jaxpr_to_module`](pallas/mosaic_gpu/lowering.md#lower_pipelined_jaxpr_to_module), [`pallas_call_hlo_interpret`](pallas/hlo_interpreter.md#pallas_call_hlo_interpret), [`io_callback`](callback.md#io_callback), [`_convert_element_type`](lax/lax.md#_convert_element_type), [`_check_block_mappings`](pallas/mosaic/lowering.md#_check_block_mappings), [`_compute_offsets_from_indices`](pallas/triton/lowering.md#_compute_offsets_from_indices), [`_masked_load_lowering_rule`](pallas/triton/lowering.md#_masked_load_lowering_rule), [`pallas_call_lowering`](pallas/triton/pallas_call_registration.md#pallas_call_lowering), [`make_array_from_callback`](array.md#make_array_from_callback), [`_load_lowering_rule`](pallas/mosaic/lowering.md#_load_lowering_rule), [`wrapper`](pallas/mpmd.md#_mpmd_map.wrapper), [`_load_lowering_rule`](pallas/mosaic/sc_lowering.md#_load_lowering_rule), [`_store_lowering_rule`](pallas/mosaic/sc_lowering.md#_store_lowering_rule), [`lower_jaxpr_into_unpipelined_module`](pallas/mosaic/lowering.md#lower_jaxpr_into_unpipelined_module), [`wrapped`](pallas/mosaic/pipeline.md#emit_pipeline.wrapped), [`_core_map_lowering_rule`](pallas/mosaic_gpu/lowering.md#_core_map_lowering_rule), [`_mpmd_map_lowering_rule`](pallas/mosaic_gpu/lowering.md#_mpmd_map_lowering_rule), [`_run_state_lowering_rule`](pallas/mosaic_gpu/lowering.md#_run_state_lowering_rule), [`aval_to_ir_type`](pallas/mosaic/lowering.md#aval_to_ir_type), [`_allocate_buffers_for_outputs`](pallas/mosaic_gpu/interpret/interpret_pallas_call.md#_allocate_buffers_for_outputs), [`_get_memory_space_from_aval`](pallas/mosaic/pallas_call_registration.md#_get_memory_space_from_aval), [`_handle_array_process_allgather`](../experimental/multihost_utils.md#_handle_array_process_allgather), [`_insert_reduce_precision`](ad_checkpoint.md#_insert_reduce_precision), [`get_ref_and_transforms`](state/primitives.md#get_ref_and_transforms), [`create`](pallas/mosaic/pipeline.md#BufferedRef.create), [`mpmd_map_tpu_lowering_rule`](pallas/mosaic/pallas_call_registration.md#mpmd_map_tpu_lowering_rule), [`_barrier_arrive_lowering`](pallas/mosaic_gpu/primitives.md#_barrier_arrive_lowering)  (+348 more)

### `ShardingTypeError`  ·  implements/extends Exception
- def: [`jax/_src/core.py:2174`](../../../../../../raw/code/jax/jax/_src/core.py#L2174)
- signature: `class ShardingTypeError(Exception):`
- used by: [`bincount`](numpy/lax_numpy.md#bincount), [`_cumulative_reduction`](numpy/reductions.md#_cumulative_reduction), [`roll`](numpy/lax_numpy.md#roll), [`repeat`](numpy/lax_numpy.md#repeat), [`call_shape_dtype_sharding_rule`](lax/utils.md#call_shape_dtype_sharding_rule), [`call_sharding_rule`](lax/utils.md#call_sharding_rule), [`_conv_general_dilated_sharding_rule`](lax/convolution.md#_conv_general_dilated_sharding_rule), [`_scatter_sharding_rule`](lax/slicing.md#_scatter_sharding_rule), [`default_nary_reduced_rule`](lax/lax.md#default_nary_reduced_rule), [`broadcasting_sharding_rule`](lax/lax.md#broadcasting_sharding_rule), [`_dot_general_sharding_rule`](lax/lax.md#_dot_general_sharding_rule), [`_get_sharding_for_varying_out_shape`](lax/slicing.md#_get_sharding_for_varying_out_shape), [`_gather_sharding_rule`](lax/slicing.md#_gather_sharding_rule), [`_mul_ur_rule`](lax/lax.md#_mul_ur_rule), [`_top_k_abstract_eval`](lax/lax.md#_top_k_abstract_eval), [`_resolve_mesh`](lax/slicing.md#_resolve_mesh), [`raise_reshape_error`](lax/lax.md#raise_reshape_error), [`_add_ur_rule`](lax/lax.md#_add_ur_rule), [`_approx_top_k_abstract_eval`](lax/ann.md#_approx_top_k_abstract_eval), [`_dot_general_unreduced_rule`](lax/lax.md#_dot_general_unreduced_rule), [`_concatenate_reduced_rule`](lax/lax.md#_concatenate_reduced_rule), [`_concatenate_unreduced_rule`](lax/lax.md#_concatenate_unreduced_rule), [`_concatenate_sharding_rule`](lax/lax.md#_concatenate_sharding_rule), [`_dus_reduced_rule`](lax/slicing.md#_dus_reduced_rule), [`_dus_unreduced_rule`](lax/slicing.md#_dus_unreduced_rule), [`_ragged_dot_general_sharding_rule`](lax/lax.md#_ragged_dot_general_sharding_rule), [`_reduce_sum_unreduced_rule`](lax/lax.md#_reduce_sum_unreduced_rule), [`_select_and_gather_add_sharding_rule`](lax/windowed_reductions.md#_select_and_gather_add_sharding_rule), [`_select_sharding_rule`](lax/lax.md#_select_sharding_rule), [`_stack_sharding_rule`](lax/lax.md#_stack_sharding_rule), [`_unstack_sharding_rule`](lax/lax.md#_unstack_sharding_rule), [`_check_specs_match`](lax/lax.md#_check_specs_match), [`_cumred_sharding_rule`](lax/control_flow/loops.md#_cumred_sharding_rule), [`_dynamic_update_slice_sharding_rule`](lax/slicing.md#_dynamic_update_slice_sharding_rule), [`_sort_abstract_eval`](lax/lax.md#_sort_abstract_eval), [`_tile_abstract_eval`](lax/lax.md#_tile_abstract_eval)

### `SomeTracer`
- def: [`jax/_src/core.py:3276`](../../../../../../raw/code/jax/jax/_src/core.py#L3276)
- signature: `class SomeTracer:`
- protocol/private: `__repr__`[`L3278`](../../../../../../raw/code/jax/jax/_src/core.py#L3278), `__slots__`[`L3277`](../../../../../../raw/code/jax/jax/_src/core.py#L3277)
- used by: [`replace_tracer_for_error_message`](core.md#replace_tracer_for_error_message)

### `TakeCurrentTraceContextManager`
- def: [`jax/_src/core.py:1452`](../../../../../../raw/code/jax/jax/_src/core.py#L1452)
- signature: `class TakeCurrentTraceContextManager:`
- protocol/private: `__enter__`[`L1455`](../../../../../../raw/code/jax/jax/_src/core.py#L1455), `__exit__`[`L1460`](../../../../../../raw/code/jax/jax/_src/core.py#L1460), `__slots__`[`L1453`](../../../../../../raw/code/jax/jax/_src/core.py#L1453)
- uses (calls/refs, reference-scoped): [`trace_ctx`](core.md#trace_ctx), [`trace`](core.md#TracingContext.trace), [`set_trace`](core.md#TracingContext.set_trace)
- used by: [`take_current_trace`](core.md#take_current_trace)

### `Token`
- def: [`jax/_src/core.py:3008`](../../../../../../raw/code/jax/jax/_src/core.py#L3008)
- signature: `class Token:`
- members:
  - `block_until_ready(self)` — [`L3014`](../../../../../../raw/code/jax/jax/_src/core.py#L3014)
- protocol/private: `__init__`[`L3012`](../../../../../../raw/code/jax/jax/_src/core.py#L3012), `_buf`[`L3011`](../../../../../../raw/code/jax/jax/_src/core.py#L3011)
- uses (calls/refs, reference-scoped): [`Array`](basearray.md#Array), [`block_until_ready`](basearray.md#Array.block_until_ready)
- used by: [`wrapper`](array.md#_token_global_result_handler.wrapper)

### `Trace`
- def: [`jax/_src/core.py:828`](../../../../../../raw/code/jax/jax/_src/core.py#L828)
- signature: `class Trace:`
- members:
  - `full_raise(self, x)` — [`L874`](../../../../../../raw/code/jax/jax/_src/core.py#L874)
  - `invalidate(self)` — [`L847`](../../../../../../raw/code/jax/jax/_src/core.py#L847)
  - `is_valid(self)` — [`L850`](../../../../../../raw/code/jax/jax/_src/core.py#L850)
  - `main(self)` — [`L879`](../../../../../../raw/code/jax/jax/_src/core.py#L879)
  - `process_call(self, call_primitive, f, tracers, params,)` — [`L856`](../../../../../../raw/code/jax/jax/_src/core.py#L856)
  - `process_custom_jvp_call(self, primitive, fun, jvp, tracers,, *, symbolic_zeros)` — [`L861`](../../../../../../raw/code/jax/jax/_src/core.py#L861)
  - `process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers,, *, out_trees, symbolic_zeros)` — [`L867`](../../../../../../raw/code/jax/jax/_src/core.py#L867)
  - `process_primitive(self, primitive, tracers, params,)` — [`L844`](../../../../../../raw/code/jax/jax/_src/core.py#L844)
  - `stage_value(self, val)` — [`L837`](../../../../../../raw/code/jax/jax/_src/core.py#L837) — Lifts a value into a trace.
- protocol/private: `__init__`[`L831`](../../../../../../raw/code/jax/jax/_src/core.py#L831), `__repr__`[`L853`](../../../../../../raw/code/jax/jax/_src/core.py#L853), `__slots__`[`L829`](../../../../../../raw/code/jax/jax/_src/core.py#L829)
- uses (calls/refs, reference-scoped): [`process_primitive`](shard_map.md#ShardMapTrace.process_primitive), [`process_primitive`](interpreters/remat.md#RematTrace.process_primitive), [`process_custom_jvp_call`](interpreters/remat.md#RematTrace.process_custom_jvp_call), [`process_custom_jvp_call`](shard_map.md#ShardMapTrace.process_custom_jvp_call), [`process_custom_vjp_call`](interpreters/remat.md#RematTrace.process_custom_vjp_call), [`process_custom_vjp_call`](shard_map.md#ShardMapTrace.process_custom_vjp_call), [`process_primitive`](core.md#EvalTrace.process_primitive), [`process_call`](core.md#EvalTrace.process_call), [`process_call`](interpreters/remat.md#RematTrace.process_call), [`stage_value`](core.md#EvalTrace.stage_value), [`stage_value`](interpreters/remat.md#RematTrace.stage_value), [`stage_value`](shard_map.md#ShardMapTrace.stage_value), [`EvalTrace`](core.md#EvalTrace), [`process_call`](shard_map.md#ShardMapTrace.process_call), [`process_custom_jvp_call`](core.md#EvalTrace.process_custom_jvp_call), [`process_custom_vjp_call`](core.md#EvalTrace.process_custom_vjp_call)
- used by: [`__init__`](shard_map.md#ShardMapTrace.__init__), [`ensure_no_leaks`](core.md#ensure_no_leaks), [`ShardMapTrace`](shard_map.md#ShardMapTrace), [`maybe_find_leaked_tracers`](core.md#maybe_find_leaked_tracers), [`trace_state`](core.md#trace_state), [`RematTrace`](interpreters/remat.md#RematTrace), [`EvalTrace`](core.md#EvalTrace), [`__init__`](interpreters/remat.md#RematTrace.__init__)

### `TraceTag`
- def: [`jax/_src/core.py:1316`](../../../../../../raw/code/jax/jax/_src/core.py#L1316)
- signature: `class TraceTag:`
- protocol/private: `__eq__`[`L1328`](../../../../../../raw/code/jax/jax/_src/core.py#L1328), `__hash__`[`L1326`](../../../../../../raw/code/jax/jax/_src/core.py#L1326)
- used by: [`remat_transform`](interpreters/remat.md#remat_transform), [`remat_subtrace`](interpreters/remat.md#remat_subtrace), [`_remat_jaxpr`](interpreters/remat.md#_remat_jaxpr)

### `Tracer`
- def: [`jax/_src/core.py:957`](../../../../../../raw/code/jax/jax/_src/core.py#L957)
- members:
  - `addressable_data(self, index)` — [`L1206`](../../../../../../raw/code/jax/jax/_src/core.py#L1206)
  - `addressable_shards(self)` — [`L1065`](../../../../../../raw/code/jax/jax/_src/core.py#L1065)
  - `at(self)` — [`L1071`](../../../../../../raw/code/jax/jax/_src/core.py#L1071)
  - `committed(self)` — [`L1049`](../../../../../../raw/code/jax/jax/_src/core.py#L1049)
  - `delete(self)` — [`L1211`](../../../../../../raw/code/jax/jax/_src/core.py#L1211)
  - `device(self)` — [`L1056`](../../../../../../raw/code/jax/jax/_src/core.py#L1056)
  - `devices(self)` — [`L1216`](../../../../../../raw/code/jax/jax/_src/core.py#L1216)
  - `full_lower(self)` — [`L1019`](../../../../../../raw/code/jax/jax/_src/core.py#L1019)
  - `get_referent(self)` — [`L1076`](../../../../../../raw/code/jax/jax/_src/core.py#L1076)
  - `global_shards(self)` — [`L1222`](../../../../../../raw/code/jax/jax/_src/core.py#L1222)
  - `is_deleted(self)` — [`L1227`](../../../../../../raw/code/jax/jax/_src/core.py#L1227)
  - `is_fully_addressable(self)` — [`L1233`](../../../../../../raw/code/jax/jax/_src/core.py#L1233)
  - `is_fully_replicated(self)` — [`L1239`](../../../../../../raw/code/jax/jax/_src/core.py#L1239)
  - `on_device_size_in_bytes(self)` — [`L1244`](../../../../../../raw/code/jax/jax/_src/core.py#L1244)
  - `sharding(self)` — [`L1040`](../../../../../../raw/code/jax/jax/_src/core.py#L1040)
  - `to_concrete_value(self)` — [`L1035`](../../../../../../raw/code/jax/jax/_src/core.py#L1035)
  - `tobytes(self, order="C")` — [`L1012`](../../../../../../raw/code/jax/jax/_src/core.py#L1012)
  - `tolist(self)` — [`L1007`](../../../../../../raw/code/jax/jax/_src/core.py#L1007)
  - `traceback(self)` — [`L1250`](../../../../../../raw/code/jax/jax/_src/core.py#L1250)
  - `unsafe_buffer_pointer(self)` — [`L1255`](../../../../../../raw/code/jax/jax/_src/core.py#L1255)
  - `aval` — [`L964`](../../../../../../raw/code/jax/jax/_src/core.py#L964)
  - `dimension_as_value` — [`L974`](../../../../../../raw/code/jax/jax/_src/core.py#L974)
  - `dtype` — [`L966`](../../../../../../raw/code/jax/jax/_src/core.py#L966)
  - `ndim` — [`L967`](../../../../../../raw/code/jax/jax/_src/core.py#L967)
  - `shape` — [`L969`](../../../../../../raw/code/jax/jax/_src/core.py#L969)
  - `size` — [`L968`](../../../../../../raw/code/jax/jax/_src/core.py#L968)
- protocol/private: `__array__`[`L995`](../../../../../../raw/code/jax/jax/_src/core.py#L995), `__array_module__`[`L1139`](../../../../../../raw/code/jax/jax/_src/core.py#L1139), `__array_priority__`[`L958`](../../../../../../raw/code/jax/jax/_src/core.py#L958), `__bool__`[`L1079`](../../../../../../raw/code/jax/jax/_src/core.py#L1079), `__complex__`[`L1099`](../../../../../../raw/code/jax/jax/_src/core.py#L1099), `__dlpack__`[`L1002`](../../../../../../raw/code/jax/jax/_src/core.py#L1002), `__float__`[`L1093`](../../../../../../raw/code/jax/jax/_src/core.py#L1093), `__getattr__`[`L1144`](../../../../../../raw/code/jax/jax/_src/core.py#L1144), `__hash__`[`L960`](../../../../../../raw/code/jax/jax/_src/core.py#L960), `__hex__`[`L1105`](../../../../../../raw/code/jax/jax/_src/core.py#L1105), `__index__`[`L1119`](../../../../../../raw/code/jax/jax/_src/core.py#L1119), `__init__`[`L986`](../../../../../../raw/code/jax/jax/_src/core.py#L986), `__int__`[`L1086`](../../../../../../raw/code/jax/jax/_src/core.py#L1086), `__iter__`[`L1022`](../../../../../../raw/code/jax/jax/_src/core.py#L1022), `__jax_array__`[`L980`](../../../../../../raw/code/jax/jax/_src/core.py#L980), `__len__`[`L1030`](../../../../../../raw/code/jax/jax/_src/core.py#L1030), `__oct__`[`L1112`](../../../../../../raw/code/jax/jax/_src/core.py#L1112), `__reduce__`[`L1127`](../../../../../../raw/code/jax/jax/_src/core.py#L1127), `__repr__`[`L1193`](../../../../../../raw/code/jax/jax/_src/core.py#L1193), `__reversed__`[`L1027`](../../../../../../raw/code/jax/jax/_src/core.py#L1027), `__setitem__`[`L1133`](../../../../../../raw/code/jax/jax/_src/core.py#L1133), `__slots__`[`L959`](../../../../../../raw/code/jax/jax/_src/core.py#L959), `_contents`[`L1196`](../../../../../../raw/code/jax/jax/_src/core.py#L1196), `_error_repr`[`L990`](../../../../../../raw/code/jax/jax/_src/core.py#L990), `_is_traced_array`[`L999`](../../../../../../raw/code/jax/jax/_src/core.py#L999), `_line_info`[`L963`](../../../../../../raw/code/jax/jax/_src/core.py#L963), `_origin_msg`[`L1202`](../../../../../../raw/code/jax/jax/_src/core.py#L1202), `_pretty_print`[`L1178`](../../../../../../raw/code/jax/jax/_src/core.py#L1178), `_short_repr`[`L1175`](../../../../../../raw/code/jax/jax/_src/core.py#L1175), `_trace`[`L962`](../../../../../../raw/code/jax/jax/_src/core.py#L962)
- uses (calls/refs, reference-scoped): [`ShapedArray`](core.md#ShapedArray), [`AbstractValue`](core.md#AbstractValue), [`text`](pretty_printer.md#text), [`Doc`](pretty_printer.md#Doc), [`concat`](pretty_printer.md#concat), [`SourceInfo`](source_info_util.md#SourceInfo), [`str_short`](core.md#AbstractValue.str_short), [`aval_method`](core.md#aval_method), [`ConcretizationTypeError`](errors.md#ConcretizationTypeError), [`to_concrete_value`](shard_map.md#ShardMapTracer.to_concrete_value), [`is_concrete`](core.md#is_concrete), [`enable_checks`](config.md#enable_checks), [`join`](pretty_printer.md#join), [`brk`](pretty_printer.md#brk), [`check_integer_conversion`](core.md#check_integer_conversion), [`format`](pretty_printer.md#Doc.format), [`check_scalar_conversion`](core.md#check_scalar_conversion), [`group`](pretty_printer.md#group), [`nest`](pretty_printer.md#nest), [`TracerArrayConversionError`](errors.md#TracerArrayConversionError), [`check_bool_conversion`](core.md#check_bool_conversion), [`_aval_property`](core.md#_aval_property), [`aval_property`](core.md#aval_property), [`TracerBase`](core.md#TracerBase), [`TracerMeta`](core.md#TracerMeta), [`TraceType`](core.md#TraceType)
- used by: [`bind`](core.md#Primitive.bind), [`full_like`](lax/lax.md#full_like), [`_trace_for_jit`](pjit.md#_trace_for_jit), [`array`](numpy/array_constructors.md#array), [`_convert_element_type`](lax/lax.md#_convert_element_type), [`concrete_or_error`](core.md#concrete_or_error), [`wrapped`](pallas/mosaic/pipeline.md#emit_pipeline.wrapped), [`empty_like`](lax/lax.md#empty_like), [`remat_partial_eval`](ad_checkpoint.md#remat_partial_eval), [`at`](basearray.md#Array.at), [`__call__`](hijax.md#custom_vjp3.__call__), [`definitely_equal`](core.md#definitely_equal), [`is_constant_dim`](core.md#is_constant_dim), [`_psum`](lax/parallel.md#_psum), [`global_array_to_host_local_array`](pmap.md#global_array_to_host_local_array), [`to_val_mat_pair`](shard_map.md#ShardMapTrace.to_val_mat_pair), [`aval_method`](core.md#aval_method), [`make_array_from_single_device_arrays`](array.md#make_array_from_single_device_arrays), [`__init__`](shard_map.md#ShardMapTracer.__init__), [`_decorator`](lax/lax.md#composite._decorator), [`_reduction_jaxpr`](lax/lax.md#_reduction_jaxpr), [`to_concrete_value`](shard_map.md#ShardMapTracer.to_concrete_value), [`_while_partial_eval`](lax/control_flow/loops.md#_while_partial_eval), [`host_local_array_to_global_array`](pmap.md#host_local_array_to_global_array), [`__str__`](shard_map.md#ShardMapTracer.__str__), [`_trace_composite_to_jaxpr`](lax/lax.md#_trace_composite_to_jaxpr), [`_maybe_perturbed`](custom_derivatives.md#_maybe_perturbed), [`_variadic_reduction_jaxpr`](lax/lax.md#_variadic_reduction_jaxpr), [`sharding`](basearray.md#Array.sharding), [`_device_get`](api.md#_device_get), [`_infer_src_sharding`](api.md#_infer_src_sharding), [`_invalid_shape_error`](core.md#_invalid_shape_error), [`ShardMapTracer`](shard_map.md#ShardMapTracer), [`convert_to_metaty`](pjit.md#convert_to_metaty), [`get_data`](array.md#make_array_from_callback.get_data), [`dimension_as_value`](core.md#dimension_as_value), [`_dyn_args_fun`](ad_checkpoint.md#_dyn_args_fun), [`RematTracer`](interpreters/remat.md#RematTracer), [`__init__`](errors.md#TracerBoolConversionError.__init__), [`device`](basearray.md#Array.device)  (+36 more)

### `TracingContext`
- def: [`jax/_src/core.py:1418`](../../../../../../raw/code/jax/jax/_src/core.py#L1418)
- members:
  - `axis_env(self)` — [`L1432`](../../../../../../raw/code/jax/jax/_src/core.py#L1432)
  - `is_empty(self)` — [`L1438`](../../../../../../raw/code/jax/jax/_src/core.py#L1438)
  - `is_top_level(self)` — [`L1435`](../../../../../../raw/code/jax/jax/_src/core.py#L1435)
  - `reset()` — [`L1422`](../../../../../../raw/code/jax/jax/_src/core.py#L1422)
  - `set_axis_env(self, axis_env)` — [`L1446`](../../../../../../raw/code/jax/jax/_src/core.py#L1446)
  - `set_trace(self, trace)` — [`L1441`](../../../../../../raw/code/jax/jax/_src/core.py#L1441)
  - `trace(self)` — [`L1428`](../../../../../../raw/code/jax/jax/_src/core.py#L1428)
- protocol/private: `__slots__`[`L1419`](../../../../../../raw/code/jax/jax/_src/core.py#L1419)
- uses (calls/refs, reference-scoped): [`trace_state`](core.md#trace_state), [`axis_env_state`](core.md#axis_env_state), [`trace_state_strong_ref`](core.md#trace_state_strong_ref), [`eval_trace`](core.md#eval_trace), [`top_axis_env`](core.md#top_axis_env), [`config_ext`](core.md#config_ext)
- used by: [`bind`](core.md#Primitive.bind), [`_trace_for_jit`](pjit.md#_trace_for_jit), [`_cpp_pjit`](pjit.md#_cpp_pjit), [`trace_ctx`](core.md#trace_ctx), [`_shard_map_to_lojax`](shard_map.md#_shard_map_to_lojax), [`__exit__`](core.md#SetCurrentTraceContextManager.__exit__), [`get_axis_env`](core.md#get_axis_env), [`trace_state_clean`](core.md#trace_state_clean), [`__enter__`](core.md#AddExplicitMeshAxisNamesContextManager.__enter__), [`__enter__`](core.md#AddSpmdAxisNamesContextManager.__enter__), [`__enter__`](core.md#ExtendAxisEnvNdContextManager.__enter__), [`__enter__`](core.md#RemoveExplicitMeshAxisNamesContextManager.__enter__), [`__enter__`](core.md#SetCurrentTraceContextManager.__enter__), [`__enter__`](core.md#TakeCurrentTraceContextManager.__enter__), [`get_opaque_trace_state`](core.md#get_opaque_trace_state), [`unsafe_am_i_under_a_jit`](core.md#unsafe_am_i_under_a_jit), [`unsafe_am_i_under_a_vmap`](core.md#unsafe_am_i_under_a_vmap), [`__exit__`](core.md#AddExplicitMeshAxisNamesContextManager.__exit__), [`__exit__`](core.md#AddSpmdAxisNamesContextManager.__exit__), [`__exit__`](core.md#ExtendAxisEnvNdContextManager.__exit__), [`__exit__`](core.md#RemoveExplicitMeshAxisNamesContextManager.__exit__), [`__exit__`](core.md#TakeCurrentTraceContextManager.__exit__), [`reset_trace_state`](core.md#reset_trace_state), [`nonempty_axis_env`](core.md#nonempty_axis_env), [`unsafe_get_axis_names`](core.md#unsafe_get_axis_names), [`unsafe_get_current_trace`](core.md#unsafe_get_current_trace), [`axis_frame`](core.md#axis_frame)

### `Var`
- def: [`jax/_src/core.py:502`](../../../../../../raw/code/jax/jax/_src/core.py#L502)
- members:
  - `pretty_print(self, context: JaxprPpContext, *, print_dtype: bool = True)` — [`L520`](../../../../../../raw/code/jax/jax/_src/core.py#L520)
  - `aval` — [`L505`](../../../../../../raw/code/jax/jax/_src/core.py#L505)
  - `final_qdd` — [`L509`](../../../../../../raw/code/jax/jax/_src/core.py#L509)
  - `initial_qdd` — [`L508`](../../../../../../raw/code/jax/jax/_src/core.py#L508)
- protocol/private: `__init__`[`L511`](../../../../../../raw/code/jax/jax/_src/core.py#L511), `__repr__`[`L517`](../../../../../../raw/code/jax/jax/_src/core.py#L517), `__slots__`[`L503`](../../../../../../raw/code/jax/jax/_src/core.py#L503)
- uses (calls/refs, reference-scoped): [`AbstractValue`](core.md#AbstractValue), [`JaxprPpContext`](core.md#JaxprPpContext), [`str_short`](core.md#AbstractValue.str_short), [`DropVar`](core.md#DropVar), [`QuasiDynamicData`](core.md#QuasiDynamicData), [`var_names`](core.md#JaxprPpContext.var_names), [`pretty_print`](core.md#DropVar.pretty_print)
- used by: [`interpret_pallas_call`](pallas/mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call), [`jaxpr_subcomp`](pallas/mosaic/lowering.md#jaxpr_subcomp), [`lower_jaxpr_into_pipelined_module`](pallas/mosaic/lowering.md#lower_jaxpr_into_pipelined_module), [`lower_jaxpr_to_mosaic_gpu`](pallas/mosaic_gpu/lowering.md#lower_jaxpr_to_mosaic_gpu), [`_pallas_call_batching_rule`](pallas/pallas_call.md#_pallas_call_batching_rule), [`_run_scoped_lowering_rule`](pallas/mosaic_gpu/lowering.md#_run_scoped_lowering_rule), [`_check_jaxpr`](core.md#_check_jaxpr), [`eval_jaxpr`](core.md#eval_jaxpr), [`_eval_jaxpr_discharge_state`](state/discharge.md#_eval_jaxpr_discharge_state), [`invars`](core.md#Jaxpr.invars), [`_pull_block_transform`](pallas/fuser/block_spec.md#_pull_block_transform), [`_pallas_call_state_discharge_rule`](pallas/pallas_call.md#_pallas_call_state_discharge_rule), [`_run_scoped_resource_estimator`](pallas/mosaic_gpu/lowering.md#_run_scoped_resource_estimator), [`outvars`](core.md#JaxprEqn.outvars), [`_body`](pallas/mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call._execute_grid_for_core._body), [`kernel_fn`](pallas/fuser/block_spec.md#make_kernel_function.kernel_fn), [`lower_jaxpr_to_triton_ir`](pallas/triton/lowering.md#lower_jaxpr_to_triton_ir), [`physicalize_interp`](pallas/fuser/fusible_dtype.md#physicalize_interp), [`_custom_evaluate_jaxpr`](pallas/fuser/custom_evaluate.md#_custom_evaluate_jaxpr), [`_dimension_semantics`](pallas/mosaic/lowering.md#MosaicGridMapping._dimension_semantics), [`_partial_eval_jaxpr_custom_rule`](shard_map.md#_partial_eval_jaxpr_custom_rule), [`lower_jaxpr_into_unpipelined_module`](pallas/mosaic/lowering.md#lower_jaxpr_into_unpipelined_module), [`eval_jaxpr_recursive`](pallas/hlo_interpreter.md#eval_jaxpr_recursive), [`lower_jaxpr_to_triton_module`](pallas/triton/lowering.md#lower_jaxpr_to_triton_module), [`_scan_partial_eval_custom`](lax/control_flow/loops.md#_scan_partial_eval_custom), [`_run_state_lowering_rule`](pallas/mosaic_gpu/lowering.md#_run_state_lowering_rule), [`_insert_reduce_precision`](ad_checkpoint.md#_insert_reduce_precision), [`constvars`](core.md#Jaxpr.constvars), [`_push_block_spec_jaxpr`](pallas/fuser/block_spec.md#_push_block_spec_jaxpr), [`remat_partial_eval`](ad_checkpoint.md#remat_partial_eval), [`_rewrite_jaxpr_for_lowering`](pallas/mosaic/pallas_call_registration.md#_rewrite_jaxpr_for_lowering), [`_shard_map_lowering_shardy`](shard_map.md#_shard_map_lowering_shardy), [`compute_usage`](pallas/fuser/block_spec.md#compute_usage), [`pass_scalars_as_refs`](pallas/mosaic/core.md#pass_scalars_as_refs), [`Atom`](core.md#Atom), [`_estimate_resources`](pallas/mosaic_gpu/lowering.md#_estimate_resources), [`_saved_residuals`](ad_checkpoint.md#_saved_residuals), [`_kernel`](pallas/mosaic_gpu/interpret/interpret_pallas_call.md#interpret_pallas_call._kernel), [`pattern_match_scan_to_fori_loop`](pallas/utils.md#pattern_match_scan_to_fori_loop), [`_construct_output_fusions`](pallas/fuser/jaxpr_fusion.md#_construct_output_fusions)  (+72 more)

### `_TempAxisName`
- def: [`jax/_src/core.py:3443`](../../../../../../raw/code/jax/jax/_src/core.py#L3443)
- signature: `class _TempAxisName:`
- members:
  - `id` — [`L3446`](../../../../../../raw/code/jax/jax/_src/core.py#L3446)
- protocol/private: `__eq__`[`L3454`](../../../../../../raw/code/jax/jax/_src/core.py#L3454), `__hash__`[`L3451`](../../../../../../raw/code/jax/jax/_src/core.py#L3451), `__init__`[`L3445`](../../../../../../raw/code/jax/jax/_src/core.py#L3445), `__lt__`[`L3457`](../../../../../../raw/code/jax/jax/_src/core.py#L3457), `__repr__`[`L3448`](../../../../../../raw/code/jax/jax/_src/core.py#L3448)

### `bint`  ·  implements/extends ExtendedDType
- def: [`jax/_src/core.py:2752`](../../../../../../raw/code/jax/jax/_src/core.py#L2752)
- signature: `class bint(dtypes.ExtendedDType):`
- members:
  - `name(self)` — [`L2760`](../../../../../../raw/code/jax/jax/_src/core.py#L2760)
  - `type(self)` — [`L2756`](../../../../../../raw/code/jax/jax/_src/core.py#L2756)
  - `bound` — [`L2753`](../../../../../../raw/code/jax/jax/_src/core.py#L2753)
- protocol/private: `__str__`[`L2763`](../../../../../../raw/code/jax/jax/_src/core.py#L2763)
- uses (calls/refs, reference-scoped): [`extended`](dtypes.md#extended), [`ExtendedDType`](dtypes.md#ExtendedDType)
- used by: [`ExtendedDType`](dtypes.md#ExtendedDType), [`body`](pallas/hlo_interpreter.md#pallas_call_hlo_interpret.body), [`type`](dtypes.md#ExtendedDType.type)

## Functions
- `_aval_property(name)` — [`L941`](../../../../../../raw/code/jax/jax/_src/core.py#L941)
- `_axis_types_dict(mesh)` — [`L2580`](../../../../../../raw/code/jax/jax/_src/core.py#L2580)
- `_cancel_divide(num, denom)` — [`L3145`](../../../../../../raw/code/jax/jax/_src/core.py#L3145)
- `_canonicalize_dimension(dim: DimSize)` — [`L2119`](../../../../../../raw/code/jax/jax/_src/core.py#L2119)
- `_check_call(ctx_factory, prim, in_atoms, params)` — [`L3780`](../../../../../../raw/code/jax/jax/_src/core.py#L3780)
- `_check_closed_call(_, *in_atoms, call_jaxpr)` — [`L3565`](../../../../../../raw/code/jax/jax/_src/core.py#L3565)
- `_check_divisibility(sharding, shape)` — [`L2257`](../../../../../../raw/code/jax/jax/_src/core.py#L2257)
- `_check_jaxpr(ctx_factory: Callable[[], tuple[JaxprPpContext, JaxprPpSettings]], jaxpr: Jaxpr)` — [`L3629`](../../../../../../raw/code/jax/jax/_src/core.py#L3629)
- `_check_sharding(sharding, shape)` — [`L3835`](../../../../../../raw/code/jax/jax/_src/core.py#L3835)
- `_create_str(x, prefix)` — [`L2599`](../../../../../../raw/code/jax/jax/_src/core.py#L2599)
- `_dropvars(jaxpr: Jaxpr)` — [`L3618`](../../../../../../raw/code/jax/jax/_src/core.py#L3618)
- `_dtype_object(dtype)` — [`L2116`](../../../../../../raw/code/jax/jax/_src/core.py#L2116)
- `_effect_free_abstract_eval(abstract_eval)` — [`L757`](../../../../../../raw/code/jax/jax/_src/core.py#L757)
- `_empty_ref_abstract_eval(*, ty, memory_space)` — [`L2914`](../../../../../../raw/code/jax/jax/_src/core.py#L2914)
- `_empty_ref_to_lojax(*, ty, memory_space)` — [`L2905`](../../../../../../raw/code/jax/jax/_src/core.py#L2905)
- `_empty_sharding(ndim)` — [`L2401`](../../../../../../raw/code/jax/jax/_src/core.py#L2401)
- `_encode_digits_alphabetic(n: int)` — [`L4086`](../../../../../../raw/code/jax/jax/_src/core.py#L4086)
- `_free_ref_abstract_eval(ref_aval)` — [`L2933`](../../../../../../raw/code/jax/jax/_src/core.py#L2933)
- `_free_ref_impl(ref)` — [`L2940`](../../../../../../raw/code/jax/jax/_src/core.py#L2940)
- `_freeze_impl(ref)` — [`L2984`](../../../../../../raw/code/jax/jax/_src/core.py#L2984)
- `_freeze_to_lojax(ref)` — [`L2972`](../../../../../../raw/code/jax/jax/_src/core.py#L2972)
- `_generic_effectful_abstract_eval(abstract_eval, prim)` — [`L769`](../../../../../../raw/code/jax/jax/_src/core.py#L769)
- `_get_shape_sharding_str(shape, spec)` — [`L2567`](../../../../../../raw/code/jax/jax/_src/core.py#L2567)
- `_invalid_shape_error(shape: Shape, context: str = "")` — [`L2158`](../../../../../../raw/code/jax/jax/_src/core.py#L2158)
- `_make_lengths_same(sharding, ndim)` — [`L2215`](../../../../../../raw/code/jax/jax/_src/core.py#L2215)
- `_map_shaped_array(size: int, axis: int | None, aval: ShapedArray)` — [`L3402`](../../../../../../raw/code/jax/jax/_src/core.py#L3402)
- `_maybe_modify_sharding(sharding, ndim)` — [`L2243`](../../../../../../raw/code/jax/jax/_src/core.py#L2243)
- `_pp_eqn(eqn: JaxprEqn, context: JaxprPpContext, settings: JaxprPpSettings, params: Sequence[str] | None = None)` — [`L4197`](../../../../../../raw/code/jax/jax/_src/core.py#L4197)
- `_ref_abstract_eval(init_aval, *, memory_space: Any, kind: Any)` — [`L2873`](../../../../../../raw/code/jax/jax/_src/core.py#L2873)
- `_ref_impl(init_val, *, memory_space: Any, kind: Any)` — [`L2886`](../../../../../../raw/code/jax/jax/_src/core.py#L2886)
- `_ref_to_lojax(init_val, *, memory_space, kind)` — [`L2865`](../../../../../../raw/code/jax/jax/_src/core.py#L2865)
- `_replace_jaxpr_effects(jaxpr: ClosedJaxpr, effects: frozenset[Effect])` — [`L3490`](../../../../../../raw/code/jax/jax/_src/core.py#L3490)
- `_sds_aval_mapping(x)` — [`L4006`](../../../../../../raw/code/jax/jax/_src/core.py#L4006)
- `_str_abstractify(x)` — [`L936`](../../../../../../raw/code/jax/jax/_src/core.py#L936)
- `_unmap_shaped_array(size: int, axis: int | None, explicit_mesh_axis, aval: ShapedArray)` — [`L3414`](../../../../../../raw/code/jax/jax/_src/core.py#L3414)
- `_vma_ur_str(mat, spec_unreduced, spec_reduced, mesh)` — [`L2607`](../../../../../../raw/code/jax/jax/_src/core.py#L2607)
- `_why_alive(ignore_ids: set[int], x: Any)` — [`L1617`](../../../../../../raw/code/jax/jax/_src/core.py#L1617)
- `_why_alive_container_info(container, obj_id)` — [`L1646`](../../../../../../raw/code/jax/jax/_src/core.py#L1646)
- `abstract_eval_(*args, **kwargs)` — [`L758`](../../../../../../raw/code/jax/jax/_src/core.py#L758)
- `abstract_eval_(*args, **kwargs)` — [`L770`](../../../../../../raw/code/jax/jax/_src/core.py#L770)
- `accum_grad_in_ref(x)` — [`L2987`](../../../../../../raw/code/jax/jax/_src/core.py#L2987)
- `auto_insert_reshard(*args)` — [`L2708`](../../../../../../raw/code/jax/jax/_src/core.py#L2708)
- `aval_mismatch_extra(a1: AbstractValue, a2: AbstractValue)` — [`L3539`](../../../../../../raw/code/jax/jax/_src/core.py#L3539)
- `aval_qdd_from_current_val(aval, x)` — [`L2065`](../../../../../../raw/code/jax/jax/_src/core.py#L2065)
- `axis_frame(axis_name)` — [`L4404`](../../../../../../raw/code/jax/jax/_src/core.py#L4404)
- `call_impl(f: lu.WrappedFun, *args, **params)` — [`L3347`](../../../../../../raw/code/jax/jax/_src/core.py#L3347)
- `cancel_divide_tracers(num, denom)` — [`L3133`](../../../../../../raw/code/jax/jax/_src/core.py#L3133)
- `canonicalize_dim(d: DimSize, context: str = "")` — [`L2147`](../../../../../../raw/code/jax/jax/_src/core.py#L2147) — Canonicalizes and checks for errors in a user-provided shape dimension value.
- `canonicalize_shape(shape: Shape, context: str = "")` — [`L2130`](../../../../../../raw/code/jax/jax/_src/core.py#L2130) — Canonicalizes and checks for errors in a user-provided shape value.
- `canonicalize_slice(s: slice, axis_size: DimSize)` — [`L3212`](../../../../../../raw/code/jax/jax/_src/core.py#L3212) — Computes the start index, step, and size of the slice `x[s]`.
- `check_avals_context_mesh(avals, prim_name)` — [`L803`](../../../../../../raw/code/jax/jax/_src/core.py#L803)
- `check_bool_conversion(arr: Array)` — [`L925`](../../../../../../raw/code/jax/jax/_src/core.py#L925)
- `check_eqn(prim, in_avals, params)` — [`L3771`](../../../../../../raw/code/jax/jax/_src/core.py#L3771)
- `check_eval_args(args)` — [`L1270`](../../../../../../raw/code/jax/jax/_src/core.py#L1270)
- `check_integer_conversion(arr: Array)` — [`L920`](../../../../../../raw/code/jax/jax/_src/core.py#L920)
- `check_jaxpr(jaxpr: Jaxpr)` — [`L3572`](../../../../../../raw/code/jax/jax/_src/core.py#L3572) — Checks well-formedness of a jaxpr.
- `check_scalar_conversion(arr: Array)` — [`L914`](../../../../../../raw/code/jax/jax/_src/core.py#L914)
- `check_type(ctx_factory: Callable[[], tuple[JaxprPpContext, JaxprPpSettings]], env: dict[Var, Atom | MutableTypecheckVal], ty: AbstractValue)` — [`L3764`](../../../../../../raw/code/jax/jax/_src/core.py#L3764)
- `check_unreduced_args(args, axes, name)` — [`L2673`](../../../../../../raw/code/jax/jax/_src/core.py#L2673)
- `clamp_index(i: DimSize, which: str)` — [`L3242`](../../../../../../raw/code/jax/jax/_src/core.py#L3242)
- `clean_up_dead_vars(eqn: JaxprEqn, env: dict[Var, Any], last_used: dict[Var, JaxprEqn | None])` — [`L4326`](../../../../../../raw/code/jax/jax/_src/core.py#L4326) — Remove all eqn.invars from env if eqn is the last time they were used.
- `cmp_shape_shd_mat_memsp(t1, t2)` — [`L3525`](../../../../../../raw/code/jax/jax/_src/core.py#L3525)
- `concrete_dim_or_error(val: Any, context="")` — [`L1994`](../../../../../../raw/code/jax/jax/_src/core.py#L1994) — Like concrete_or_error(operator.index), allowing symbolic dimensions.
- `concrete_or_error(force: Any, val: Any, context="")` — [`L1981`](../../../../../../raw/code/jax/jax/_src/core.py#L1981) — Like force(val), but gives the context in the error message.
- `concretization_function_error(fun, suggest_astype=False)` — [`L1963`](../../../../../../raw/code/jax/jax/_src/core.py#L1963)
- `convert_to_index(d: DimSize)` — [`L3224`](../../../../../../raw/code/jax/jax/_src/core.py#L3224)
- `ctx_factory()` — [`L3584`](../../../../../../raw/code/jax/jax/_src/core.py#L3584)
- `cur_aval_qdd(x)` — [`L2060`](../../../../../../raw/code/jax/jax/_src/core.py#L2060)
- `cur_qdd(x)` — [`L2053`](../../../../../../raw/code/jax/jax/_src/core.py#L2053)
- `current_jaxpr_eqn_context()` — [`L420`](../../../../../../raw/code/jax/jax/_src/core.py#L420)
- `dedup_referents(itr: Iterable[Any])` — [`L1755`](../../../../../../raw/code/jax/jax/_src/core.py#L1755)
- `definitely_equal(x, y)` — [`L1758`](../../../../../../raw/code/jax/jax/_src/core.py#L1758)
- `definitely_equal_one_of_dim(d1: DimSize, dlist: Sequence[DimSize])` — [`L3107`](../../../../../../raw/code/jax/jax/_src/core.py#L3107)
- `definitely_equal_shape(s1: Shape, s2: Shape)` — [`L3110`](../../../../../../raw/code/jax/jax/_src/core.py#L3110) — Check that two shapes are guaranteed to be element-wise equal.
- `dilate_dim(d: DimSize, dilation: DimSize)` — [`L3158`](../../../../../../raw/code/jax/jax/_src/core.py#L3158) — max(0, 1 + dilation * (d - 1)).
- `dim_constant(ct: int)` — [`L3316`](../../../../../../raw/code/jax/jax/_src/core.py#L3316)
- `dim_value_aval()` — [`L3324`](../../../../../../raw/code/jax/jax/_src/core.py#L3324)
- `dim_value_dtype()` — [`L3312`](../../../../../../raw/code/jax/jax/_src/core.py#L3312) — The dtype to be used for dimension values.
- `dimension_as_value(d: DimSize)` — [`L3200`](../../../../../../raw/code/jax/jax/_src/core.py#L3200) — Turns a dimension size into a JAX array.
- `divide_shape_sizes(s1: Shape, s2: Shape)` — [`L3119`](../../../../../../raw/code/jax/jax/_src/core.py#L3119) — Returns an integer "i" s.t., i * size(s2) == size(s1).
- `empty_ref(ty, memory_space=None)` — [`L2896`](../../../../../../raw/code/jax/jax/_src/core.py#L2896)
- `ensure_compile_time_eval()` — [`L1675`](../../../../../../raw/code/jax/jax/_src/core.py#L1675) — Context manager to ensure evaluation at trace/compile time (or error).
- `ensure_no_leaks(trace: Trace)` — [`L1585`](../../../../../../raw/code/jax/jax/_src/core.py#L1585)
- `eqn_effects(jaxpr, invars)` — [`L3809`](../../../../../../raw/code/jax/jax/_src/core.py#L3809)
- `eqn_params_const_args(params)` — [`L629`](../../../../../../raw/code/jax/jax/_src/core.py#L629)
- `eqns_fn()` — [`L4296`](../../../../../../raw/code/jax/jax/_src/core.py#L4296)
- `error(self, arg)` — [`L1971`](../../../../../../raw/code/jax/jax/_src/core.py#L1971)
- `escaped_tracer_error(tracer, detail=None)` — [`L882`](../../../../../../raw/code/jax/jax/_src/core.py#L882)
- `eval_context()` — [`L1738`](../../../../../../raw/code/jax/jax/_src/core.py#L1738)
- `eval_jaxpr(jaxpr: Jaxpr, consts, *args, propagate_source_info=True)` — [`L776`](../../../../../../raw/code/jax/jax/_src/core.py#L776)
- `eval_one_dim(d: DimSize)` — [`L3304`](../../../../../../raw/code/jax/jax/_src/core.py#L3304)
- `evaluate_shape(shape: Shape, dim_vars: Sequence[str], *dim_values: Array)` — [`L3290`](../../../../../../raw/code/jax/jax/_src/core.py#L3290) — Evaluates a shape possibly containing non-constants.
- `filter_named_axis_effects(effects: Effects, names: Collection[AxisName])` — [`L3472`](../../../../../../raw/code/jax/jax/_src/core.py#L3472)
- `find_top_trace(_)` — [`L4387`](../../../../../../raw/code/jax/jax/_src/core.py#L4387)
- `free_ref(ref: Ref)` — [`L2921`](../../../../../../raw/code/jax/jax/_src/core.py#L2921) — Invalidate a given reference.
- `freeze(ref: Ref)` — [`L2943`](../../../../../../raw/code/jax/jax/_src/core.py#L2943) — Invalidate a given reference and return its final value.
- `freeze_abstract_eval(ref_aval)` — [`L2980`](../../../../../../raw/code/jax/jax/_src/core.py#L2980)
- `full_lower(val)` — [`L1743`](../../../../../../raw/code/jax/jax/_src/core.py#L1743)
- `get_axis_env()` — [`L1560`](../../../../../../raw/code/jax/jax/_src/core.py#L1560)
- `get_cur_mesh_sharding(spec=None)` — [`L2191`](../../../../../../raw/code/jax/jax/_src/core.py#L2191)
- `get_mat(mat, mesh)` — [`L2304`](../../../../../../raw/code/jax/jax/_src/core.py#L2304)
- `get_memory_space(memory_space)` — [`L2328`](../../../../../../raw/code/jax/jax/_src/core.py#L2328)
- `get_opaque_trace_state(convention=None)` — [`L4372`](../../../../../../raw/code/jax/jax/_src/core.py#L4372)
- `get_referent(x: Any)` — [`L1749`](../../../../../../raw/code/jax/jax/_src/core.py#L1749)
- `get_sharding(sharding, shape)` — [`L2273`](../../../../../../raw/code/jax/jax/_src/core.py#L2273) — Modifies and checks the sharding.
- `get_token_aval()` — [`L3004`](../../../../../../raw/code/jax/jax/_src/core.py#L3004)
- `getr(aval)` — [`L2205`](../../../../../../raw/code/jax/jax/_src/core.py#L2205)
- `getu(aval)` — [`L2195`](../../../../../../raw/code/jax/jax/_src/core.py#L2195)
- `identity(x)` — [`L84`](../../../../../../raw/code/jax/jax/_src/core.py#L84)
- `insert_reduced_reshard(args)` — [`L2686`](../../../../../../raw/code/jax/jax/_src/core.py#L2686)
- `is_concrete(x)` — [`L1954`](../../../../../../raw/code/jax/jax/_src/core.py#L1954)
- `is_constant_dim(d: DimSize)` — [`L3089`](../../../../../../raw/code/jax/jax/_src/core.py#L3089)
- `is_constant_shape(s: Shape)` — [`L3103`](../../../../../../raw/code/jax/jax/_src/core.py#L3103)
- `is_dim(v: Any)` — [`L3100`](../../../../../../raw/code/jax/jax/_src/core.py#L3100)
- `is_empty_shape(s: Shape)` — [`L3155`](../../../../../../raw/code/jax/jax/_src/core.py#L3155)
- `is_hoistable(v: Literal)` — [`L604`](../../../../../../raw/code/jax/jax/_src/core.py#L604)
- `is_literalable(x: Any, for_ad: bool = False)` — [`L584`](../../../../../../raw/code/jax/jax/_src/core.py#L584)
- `is_symbolic_dim(v: Any)` — [`L3081`](../../../../../../raw/code/jax/jax/_src/core.py#L3081) — Checks if a value is a symbolic dimension used for shape polymorphism.
- `jaxpr_as_fun(closed_jaxpr: ClosedJaxpr, *args)` — [`L318`](../../../../../../raw/code/jax/jax/_src/core.py#L318)
- `jaxpr_const_args(jaxpr: Jaxpr)` — [`L609`](../../../../../../raw/code/jax/jax/_src/core.py#L609)
- `jaxprs_in_params(params)` — [`L229`](../../../../../../raw/code/jax/jax/_src/core.py#L229)
- `join_effects(*effects: Effects)` — [`L226`](../../../../../../raw/code/jax/jax/_src/core.py#L226)
- `last_used(jaxpr: Jaxpr)` — [`L4316`](../../../../../../raw/code/jax/jax/_src/core.py#L4316) — Returns a mapping from every var in jaxpr to what equation uses it last.
- `leaked_tracer_error(name: str, t, tracers: list[Tracer])` — [`L1610`](../../../../../../raw/code/jax/jax/_src/core.py#L1610)
- `mapped_aval(size: AxisSize, axis, aval: AbstractValue)` — [`L3374`](../../../../../../raw/code/jax/jax/_src/core.py#L3374)
- `mapped_leading_aval(size, aval)` — [`L3384`](../../../../../../raw/code/jax/jax/_src/core.py#L3384)
- `max_dim(d1: DimSize, d2: DimSize)` — [`L3188`](../../../../../../raw/code/jax/jax/_src/core.py#L3188) — Like max(d1, d2) but for both constant and symbolic dimensions.
- `maybe_find_leaked_tracers(trace: Trace)` — [`L1598`](../../../../../../raw/code/jax/jax/_src/core.py#L1598) — Find the leaked tracers holding a reference to the Trace
- `mem_kind_to_space(mem_kind: str | None)` — [`L1865`](../../../../../../raw/code/jax/jax/_src/core.py#L1865)
- `mem_space_to_kind(mem_space: Any)` — [`L1871`](../../../../../../raw/code/jax/jax/_src/core.py#L1871) — Converts a memory space to its corresponding XLA memory kind string.
- `min_dim(d1: DimSize, d2: DimSize)` — [`L3176`](../../../../../../raw/code/jax/jax/_src/core.py#L3176) — Like min(d1, d2) but for both constant and symbolic dimensions.
- `modify_spec_for_auto_manual(spec, mesh)` — [`L2225`](../../../../../../raw/code/jax/jax/_src/core.py#L2225)
- `new_jaxpr_eqn(invars, outvars, primitive, params, effects, source_info=None, ctx=None)` — [`L477`](../../../../../../raw/code/jax/jax/_src/core.py#L477)
- `new_ref(init_val: Any, *, memory_space: Any = None, kind: Any = None)` — [`L2841`](../../../../../../raw/code/jax/jax/_src/core.py#L2841) — Create a mutable array reference with initial value ``init_val``.
- `nonempty_axis_env()` — [`L4377`](../../../../../../raw/code/jax/jax/_src/core.py#L4377)
- `order_wrt_mesh(mesh, x)` — [`L2604`](../../../../../../raw/code/jax/jax/_src/core.py#L2604)
- `physical_aval(aval: ShapedArray)` — [`L2088`](../../../../../../raw/code/jax/jax/_src/core.py#L2088)
- `physical_element_aval(edtype: dtypes.ExtendedDType)` — [`L2109`](../../../../../../raw/code/jax/jax/_src/core.py#L2109)
- `physical_shape(logical_shape, dtype)` — [`L2105`](../../../../../../raw/code/jax/jax/_src/core.py#L2105)
- `positional_effects(jaxpr)` — [`L3816`](../../../../../../raw/code/jax/jax/_src/core.py#L3816)
- `pp_aval(a: AbstractValue, context: JaxprPpContext)` — [`L4142`](../../../../../../raw/code/jax/jax/_src/core.py#L4142)
- `pp_effect(effect: Effect, context: JaxprPpContext)` — [`L4309`](../../../../../../raw/code/jax/jax/_src/core.py#L4309)
- `pp_eqn(eqn: JaxprEqn, context: JaxprPpContext, settings: JaxprPpSettings)` — [`L4189`](../../../../../../raw/code/jax/jax/_src/core.py#L4189)
- `pp_eqns(eqns: Sequence[JaxprEqn], context: JaxprPpContext, settings: JaxprPpSettings)` — [`L4216`](../../../../../../raw/code/jax/jax/_src/core.py#L4216)
- `pp_jaxpr(jaxpr: Jaxpr, context: JaxprPpContext, settings: JaxprPpSettings)` — [`L4270`](../../../../../../raw/code/jax/jax/_src/core.py#L4270)
- `pp_jaxpr_eqn_range(jaxpr: Jaxpr, lo: int, hi: int, context: JaxprPpContext, settings: JaxprPpSettings)` — [`L4291`](../../../../../../raw/code/jax/jax/_src/core.py#L4291)
- `pp_jaxpr_skeleton(jaxpr: Jaxpr, eqns_fn, context: JaxprPpContext, settings: JaxprPpSettings)` — [`L4222`](../../../../../../raw/code/jax/jax/_src/core.py#L4222)
- `pp_jaxprs(jaxprs: Sequence[ClosedJaxpr | Jaxpr], context: JaxprPpContext, settings: JaxprPpSettings)` — [`L4281`](../../../../../../raw/code/jax/jax/_src/core.py#L4281)
- `pp_kv_pair(k: str, v: Any, context: JaxprPpContext, settings: JaxprPpSettings)` — [`L4162`](../../../../../../raw/code/jax/jax/_src/core.py#L4162)
- `pp_kv_pairs(kv_pairs, context: JaxprPpContext, settings: JaxprPpSettings)` — [`L4178`](../../../../../../raw/code/jax/jax/_src/core.py#L4178)
- `pp_shared_jaxpr(name: str, jaxpr: Jaxpr, context: JaxprPpContext, settings: JaxprPpSettings)` — [`L4252`](../../../../../../raw/code/jax/jax/_src/core.py#L4252)
- `pp_toplevel_jaxpr(jaxpr_to_print: Jaxpr, *, source_info: bool = False, print_shapes: bool = True, custom_pp_eqn_rules: bool = True, name_stack: bool = False, print_effects: bool = False)` — [`L4020`](../../../../../../raw/code/jax/jax/_src/core.py#L4020)
- `pp_var(v: Var | Literal, context: JaxprPpContext, *, print_literal_dtype: bool = True, is_binder: bool = False)` — [`L4131`](../../../../../../raw/code/jax/jax/_src/core.py#L4131)
- `pp_vars(vs: Sequence[Atom], context: JaxprPpContext, *, separator="", print_shapes: bool = False, is_binder: bool = False)` — [`L4145`](../../../../../../raw/code/jax/jax/_src/core.py#L4145)
- `primal_dtype_to_tangent_dtype(primal_dtype)` — [`L2620`](../../../../../../raw/code/jax/jax/_src/core.py#L2620)
- `primal_sharding_to_cotangent_sharding(sharding)` — [`L2628`](../../../../../../raw/code/jax/jax/_src/core.py#L2628)
- `pvary(x, axis_name)` — [`L2634`](../../../../../../raw/code/jax/jax/_src/core.py#L2634)
- `read(v: Atom)` — [`L777`](../../../../../../raw/code/jax/jax/_src/core.py#L777)
- `read(x: Atom)` — [`L3635`](../../../../../../raw/code/jax/jax/_src/core.py#L3635)
- `reduced_vary_cast(x, axis_name)` — [`L2657`](../../../../../../raw/code/jax/jax/_src/core.py#L2657)
- `remove_named_axis_effects(jaxpr: Jaxpr, names: Collection[AxisName])` — [`L3479`](../../../../../../raw/code/jax/jax/_src/core.py#L3479)
- `replace_jaxpr_effects(jaxpr: ClosedJaxpr, effects: Effects)` — [`L3486`](../../../../../../raw/code/jax/jax/_src/core.py#L3486)
- `replace_tracer_for_error_message(obj)` — [`L3280`](../../../../../../raw/code/jax/jax/_src/core.py#L3280)
- `reset_trace_state()` — [`L1567`](../../../../../../raw/code/jax/jax/_src/core.py#L1567) — Resets the global trace state and returns True if it was already clean.
- `resolve_input_effects(effs, invars)` — [`L488`](../../../../../../raw/code/jax/jax/_src/core.py#L488)
- `same_referent(x: Any, y: Any)` — [`L1752`](../../../../../../raw/code/jax/jax/_src/core.py#L1752)
- `shaped_abstractify(x)` — [`L1907`](../../../../../../raw/code/jax/jax/_src/core.py#L1907)
- `shard_aval(mesh, manual_axes, check_vma, spec, aval: AbstractValue)` — [`L4338`](../../../../../../raw/code/jax/jax/_src/core.py#L4338)
- `standard_vma_rule(prim_name, *avals, **kwargs)` — [`L2736`](../../../../../../raw/code/jax/jax/_src/core.py#L2736)
- `str_short_aval(shape, dtype, mesh, spec, mat, memory_space, short_dtypes=False, mesh_axis_types=False)` — [`L2588`](../../../../../../raw/code/jax/jax/_src/core.py#L2588)
- `stride_dim(d: DimSize, window_size: DimSize, window_stride: DimSize)` — [`L3167`](../../../../../../raw/code/jax/jax/_src/core.py#L3167) — max(0, (d - window_size) // window_stride + 1)
- `subjaxprs(jaxpr: Jaxpr)` — [`L239`](../../../../../../raw/code/jax/jax/_src/core.py#L239) — Generator for all subjaxprs found in the params of jaxpr.eqns.
- `subst_input_effects(effs, env)` — [`L3812`](../../../../../../raw/code/jax/jax/_src/core.py#L3812)
- `to_concrete_value(x)` — [`L1957`](../../../../../../raw/code/jax/jax/_src/core.py#L1957)
- `trace_state_clean()` — [`L1564`](../../../../../../raw/code/jax/jax/_src/core.py#L1564)
- `typecheck(aval: AbstractValue, x)` — [`L3495`](../../../../../../raw/code/jax/jax/_src/core.py#L3495)
- `typecompat(aval_ref: AbstractValue, aval: AbstractValue)` — [`L3498`](../../../../../../raw/code/jax/jax/_src/core.py#L3498) — Determine whether `aval` conforms to `aval_ref`. Ignores weak_type.
- `typematch(t1: AbstractValue, t2: AbstractValue, no_dtype_check: bool = False)` — [`L3505`](../../../../../../raw/code/jax/jax/_src/core.py#L3505) — Determine whether `t1` and `t2` are equivalent. Ignores weak_type.
- `typeof(x: Any)` — [`L1935`](../../../../../../raw/code/jax/jax/_src/core.py#L1935) — Return the JAX type (i.e. :class:`AbstractValue`) of the input. — documented in [jax-_src-core](../../../concepts/jax-_src-core.md)
- `unmapped_aval(size: AxisSize, axis: int | None, aval: AbstractValue, explicit_mesh_axis=None)` — [`L3388`](../../../../../../raw/code/jax/jax/_src/core.py#L3388)
- `unmapped_leading_aval(size, aval)` — [`L3399`](../../../../../../raw/code/jax/jax/_src/core.py#L3399)
- `unsafe_am_i_under_a_jit()` — [`L4380`](../../../../../../raw/code/jax/jax/_src/core.py#L4380)
- `unsafe_am_i_under_a_vmap()` — [`L4383`](../../../../../../raw/code/jax/jax/_src/core.py#L4383)
- `unsafe_get_axis_names()` — [`L4400`](../../../../../../raw/code/jax/jax/_src/core.py#L4400)
- `unsafe_get_current_trace()` — [`L4391`](../../../../../../raw/code/jax/jax/_src/core.py#L4391)
- `unsafe_get_trace_stack(trace)` — [`L4394`](../../../../../../raw/code/jax/jax/_src/core.py#L4394)
- `unshard_aval(mesh, check_vma, spec, aval: AbstractValue)` — [`L4347`](../../../../../../raw/code/jax/jax/_src/core.py#L4347)
- `update_aval_with_sharding(aval, sharding, mat=None)` — [`L1889`](../../../../../../raw/code/jax/jax/_src/core.py#L1889)
- `valid_jaxtype(x)` — [`L1853`](../../../../../../raw/code/jax/jax/_src/core.py#L1853)
- `write(v: Var, val: Any)` — [`L780`](../../../../../../raw/code/jax/jax/_src/core.py#L780)
- `write(v: Var, a: AvalQDD)` — [`L3655`](../../../../../../raw/code/jax/jax/_src/core.py#L3655)

## Module values
- `Atom` — [`L637`](../../../../../../raw/code/jax/jax/_src/core.py#L637)
- `AvalMapHandlerPair` — [`L3430`](../../../../../../raw/code/jax/jax/_src/core.py#L3430)
- `AxisName` — [`L1332`](../../../../../../raw/code/jax/jax/_src/core.py#L1332)
- `AxisSize` — [`L2766`](../../../../../../raw/code/jax/jax/_src/core.py#L2766)
- `CustomPpEqnRule` — [`L4213`](../../../../../../raw/code/jax/jax/_src/core.py#L4213)
- `DebugInfo` — [`L94`](../../../../../../raw/code/jax/jax/_src/core.py#L94)
- `Effect` — [`L88`](../../../../../../raw/code/jax/jax/_src/core.py#L88)
- `EffectTypeSet` — [`L90`](../../../../../../raw/code/jax/jax/_src/core.py#L90)
- `Effects` — [`L89`](../../../../../../raw/code/jax/jax/_src/core.py#L89)
- `InitialResultPaths` — [`L95`](../../../../../../raw/code/jax/jax/_src/core.py#L95)
- `InputType` — [`L1847`](../../../../../../raw/code/jax/jax/_src/core.py#L1847)
- `OutputType` — [`L1848`](../../../../../../raw/code/jax/jax/_src/core.py#L1848)
- `ParamDict` — [`L1331`](../../../../../../raw/code/jax/jax/_src/core.py#L1331)
- `PyTree` — [`L75`](../../../../../../raw/code/jax/jax/_src/core.py#L75)
- `TRACER_LEAK_DEBUGGER_WARNING` — [`L1575`](../../../../../../raw/code/jax/jax/_src/core.py#L1575)
- `TraceType` — [`L826`](../../../../../../raw/code/jax/jax/_src/core.py#L826)
- `TracerBase` — [`L951`](../../../../../../raw/code/jax/jax/_src/core.py#L951)
- `TracerMeta` — [`L952`](../../../../../../raw/code/jax/jax/_src/core.py#L952)
- `Value` — [`L1851`](../../../../../../raw/code/jax/jax/_src/core.py#L1851)
- `_TRACER_ERROR_NUM_TRACEBACK_FRAMES` — [`L78`](../../../../../../raw/code/jax/jax/_src/core.py#L78)
- `_dtype_object_types` — [`L2114`](../../../../../../raw/code/jax/jax/_src/core.py#L2114)
- `abstract_token` — [`L3001`](../../../../../../raw/code/jax/jax/_src/core.py#L3001)
- `accum_grad_in_ref_p` — [`L2990`](../../../../../../raw/code/jax/jax/_src/core.py#L2990)
- `add_explicit_mesh_axis_names` — [`L1539`](../../../../../../raw/code/jax/jax/_src/core.py#L1539)
- `add_spmd_axis_names` — [`L1521`](../../../../../../raw/code/jax/jax/_src/core.py#L1521)
- `array_ref_effect` — [`L2836`](../../../../../../raw/code/jax/jax/_src/core.py#L2836)
- `aval_mapping_handlers` — [`L3431`](../../../../../../raw/code/jax/jax/_src/core.py#L3431)
- `aval_method` — [`L1265`](../../../../../../raw/code/jax/jax/_src/core.py#L1265)
- `aval_property` — [`L1264`](../../../../../../raw/code/jax/jax/_src/core.py#L1264)
- `axis_env_state` — [`L1410`](../../../../../../raw/code/jax/jax/_src/core.py#L1410)
- `call` — [`L3353`](../../../../../../raw/code/jax/jax/_src/core.py#L3353)
- `call_p` — [`L3352`](../../../../../../raw/code/jax/jax/_src/core.py#L3352)
- `closed_call_p` — [`L3366`](../../../../../../raw/code/jax/jax/_src/core.py#L3366)
- `config_ext` — [`L73`](../../../../../../raw/code/jax/jax/_src/core.py#L73)
- `custom_typechecks` — [`L3563`](../../../../../../raw/code/jax/jax/_src/core.py#L3563)
- `empty_mat` — [`L2398`](../../../../../../raw/code/jax/jax/_src/core.py#L2398)
- `empty_ref_p` — [`L2899`](../../../../../../raw/code/jax/jax/_src/core.py#L2899)
- `eval_trace` — [`L1396`](../../../../../../raw/code/jax/jax/_src/core.py#L1396)
- `extend_axis_env_nd` — [`L1504`](../../../../../../raw/code/jax/jax/_src/core.py#L1504)
- `free_ref_p` — [`L2926`](../../../../../../raw/code/jax/jax/_src/core.py#L2926)
- `freeze_p` — [`L2968`](../../../../../../raw/code/jax/jax/_src/core.py#L2968)
- `gensym` — [`L525`](../../../../../../raw/code/jax/jax/_src/core.py#L525)
- `initial_result_paths` — [`L96`](../../../../../../raw/code/jax/jax/_src/core.py#L96)
- `internal_mutable_array_effect` — [`L2836`](../../../../../../raw/code/jax/jax/_src/core.py#L2836)
- `literalable_scalar_types` — [`L582`](../../../../../../raw/code/jax/jax/_src/core.py#L582)
- `literalable_types` — [`L581`](../../../../../../raw/code/jax/jax/_src/core.py#L581)
- `no_axis_name` — [`L1334`](../../../../../../raw/code/jax/jax/_src/core.py#L1334)
- `no_effects` — [`L91`](../../../../../../raw/code/jax/jax/_src/core.py#L91)
- `pp_eqn_rules` — [`L4214`](../../../../../../raw/code/jax/jax/_src/core.py#L4214)
- `pvary_p` — [`L2652`](../../../../../../raw/code/jax/jax/_src/core.py#L2652)
- `pytype_aval_mappings` — [`L934`](../../../../../../raw/code/jax/jax/_src/core.py#L934)
- `reduced_vary_cast_p` — [`L2669`](../../../../../../raw/code/jax/jax/_src/core.py#L2669)
- `ref_p` — [`L2859`](../../../../../../raw/code/jax/jax/_src/core.py#L2859)
- `remove_explicit_mesh_axis_names` — [`L1557`](../../../../../../raw/code/jax/jax/_src/core.py#L1557)
- `set_current_trace` — [`L1489`](../../../../../../raw/code/jax/jax/_src/core.py#L1489)
- `shard_aval_handlers` — [`L4335`](../../../../../../raw/code/jax/jax/_src/core.py#L4335)
- `stage_p` — [`L1275`](../../../../../../raw/code/jax/jax/_src/core.py#L1275)
- `take_current_trace` — [`L1463`](../../../../../../raw/code/jax/jax/_src/core.py#L1463)
- `top_axis_env` — [`L1397`](../../../../../../raw/code/jax/jax/_src/core.py#L1397)
- `trace_ctx` — [`L1449`](../../../../../../raw/code/jax/jax/_src/core.py#L1449)
- `trace_state` — [`L1400`](../../../../../../raw/code/jax/jax/_src/core.py#L1400)
- `trace_state_strong_ref` — [`L1406`](../../../../../../raw/code/jax/jax/_src/core.py#L1406)
- `unsafe_map` — [`L71`](../../../../../../raw/code/jax/jax/_src/core.py#L71)
- `unsafe_zip` — [`L70`](../../../../../../raw/code/jax/jax/_src/core.py#L70)
- `unshard_aval_handlers` — [`L4336`](../../../../../../raw/code/jax/jax/_src/core.py#L4336)

