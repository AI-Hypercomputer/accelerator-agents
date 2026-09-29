---
title: 'Module: jax/core.py'
type: catalog
provenance: extracted
module: jax/core.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.core`/
symbols:
  _deprecations: _deprecations.
  CallPrimitive: CallPrimitive.
  DebugInfo: DebugInfo.
  DropVar: DropVar.
  Effect: Effect.
  Effects: Effects.
  InconclusiveDimensionOperation: InconclusiveDimensionOperation.
  JaxprPpContext: JaxprPpContext.
  JaxprPpSettings: JaxprPpSettings.
  JaxprTypeError: JaxprTypeError.
  OutputType: OutputType.
  abstract_token: abstract_token.
  aval_mapping_handlers: aval_mapping_handlers.
  call: call.
  check_jaxpr: check_jaxpr.
  concrete_or_error: concrete_or_error.
  concretization_function_error: concretization_function_error.
  custom_typechecks: custom_typechecks.
  find_top_trace: find_top_trace.
  gensym: gensym.
  get_opaque_trace_state: get_opaque_trace_state.
  is_concrete: is_concrete.
  is_constant_dim: is_constant_dim.
  is_constant_shape: is_constant_shape.
  jaxprs_in_params: jaxprs_in_params.
  literalable_types: literalable_types.
  new_jaxpr_eqn: new_jaxpr_eqn.
  no_axis_name: no_axis_name.
  no_effects: no_effects.
  nonempty_axis_env_DO_NOT_USE: nonempty_axis_env_DO_NOT_USE.
  primal_dtype_to_tangent_dtype: primal_dtype_to_tangent_dtype.
  pytype_aval_mappings: pytype_aval_mappings.
  trace_ctx: trace_ctx.
  unsafe_am_i_under_a_jit_DO_NOT_USE: unsafe_am_i_under_a_jit_DO_NOT_USE.
  unsafe_am_i_under_a_vmap_DO_NOT_USE: unsafe_am_i_under_a_vmap_DO_NOT_USE.
  unsafe_get_axis_names_DO_NOT_USE: unsafe_get_axis_names_DO_NOT_USE.
  valid_jaxtype: valid_jaxtype.
---
# Module: [`jax/core.py`](../../../../../raw/code/jax/jax/core.py)

## Module values
- `CallPrimitive` — [`L185`](../../../../../raw/code/jax/jax/core.py#L185)
- `DebugInfo` — [`L186`](../../../../../raw/code/jax/jax/core.py#L186)
- `DropVar` — [`L187`](../../../../../raw/code/jax/jax/core.py#L187)
- `Effect` — [`L188`](../../../../../raw/code/jax/jax/core.py#L188)
- `Effects` — [`L189`](../../../../../raw/code/jax/jax/core.py#L189)
- `InconclusiveDimensionOperation` — [`L190`](../../../../../raw/code/jax/jax/core.py#L190)
- `JaxprPpContext` — [`L191`](../../../../../raw/code/jax/jax/core.py#L191)
- `JaxprPpSettings` — [`L192`](../../../../../raw/code/jax/jax/core.py#L192)
- `JaxprTypeError` — [`L193`](../../../../../raw/code/jax/jax/core.py#L193)
- `OutputType` — [`L194`](../../../../../raw/code/jax/jax/core.py#L194)
- `_deprecations` — [`L35`](../../../../../raw/code/jax/jax/core.py#L35)
- `abstract_token` — [`L195`](../../../../../raw/code/jax/jax/core.py#L195)
- `aval_mapping_handlers` — [`L196`](../../../../../raw/code/jax/jax/core.py#L196)
- `call` — [`L197`](../../../../../raw/code/jax/jax/core.py#L197)
- `check_jaxpr` — [`L198`](../../../../../raw/code/jax/jax/core.py#L198)
- `concrete_or_error` — [`L199`](../../../../../raw/code/jax/jax/core.py#L199)
- `concretization_function_error` — [`L200`](../../../../../raw/code/jax/jax/core.py#L200)
- `custom_typechecks` — [`L201`](../../../../../raw/code/jax/jax/core.py#L201)
- `find_top_trace` — [`L202`](../../../../../raw/code/jax/jax/core.py#L202)
- `gensym` — [`L203`](../../../../../raw/code/jax/jax/core.py#L203)
- `get_opaque_trace_state` — [`L204`](../../../../../raw/code/jax/jax/core.py#L204)
- `is_concrete` — [`L205`](../../../../../raw/code/jax/jax/core.py#L205)
- `is_constant_dim` — [`L206`](../../../../../raw/code/jax/jax/core.py#L206)
- `is_constant_shape` — [`L207`](../../../../../raw/code/jax/jax/core.py#L207)
- `jaxprs_in_params` — [`L208`](../../../../../raw/code/jax/jax/core.py#L208)
- `literalable_types` — [`L209`](../../../../../raw/code/jax/jax/core.py#L209)
- `new_jaxpr_eqn` — [`L210`](../../../../../raw/code/jax/jax/core.py#L210)
- `no_axis_name` — [`L211`](../../../../../raw/code/jax/jax/core.py#L211)
- `no_effects` — [`L212`](../../../../../raw/code/jax/jax/core.py#L212)
- `nonempty_axis_env_DO_NOT_USE` — [`L213`](../../../../../raw/code/jax/jax/core.py#L213)
- `primal_dtype_to_tangent_dtype` — [`L214`](../../../../../raw/code/jax/jax/core.py#L214)
- `pytype_aval_mappings` — [`L215`](../../../../../raw/code/jax/jax/core.py#L215)
- `trace_ctx` — [`L216`](../../../../../raw/code/jax/jax/core.py#L216)
- `unsafe_am_i_under_a_jit_DO_NOT_USE` — [`L217`](../../../../../raw/code/jax/jax/core.py#L217)
- `unsafe_am_i_under_a_vmap_DO_NOT_USE` — [`L218`](../../../../../raw/code/jax/jax/core.py#L218)
- `unsafe_get_axis_names_DO_NOT_USE` — [`L219`](../../../../../raw/code/jax/jax/core.py#L219)
- `valid_jaxtype` — [`L220`](../../../../../raw/code/jax/jax/core.py#L220)

