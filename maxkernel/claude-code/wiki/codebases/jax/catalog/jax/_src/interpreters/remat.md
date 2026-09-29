---
title: 'Module: jax/_src/interpreters/remat.py'
type: catalog
provenance: extracted
module: jax/_src/interpreters/remat.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.interpreters.remat`/
symbols:
  remat_transform: remat_transform().
  remat_subtrace: remat_subtrace().
  _remat_jaxpr: _remat_jaxpr().
  RematTrace.process_primitive: RematTrace#process_primitive().
  RematTrace.to_val_tracer_pair: RematTrace#to_val_tracer_pair().
  RematTrace.process_custom_jvp_call: RematTrace#process_custom_jvp_call().
  RematTrace.process_custom_vjp_call: RematTrace#process_custom_vjp_call().
  map: map.
  RematTracer: RematTracer#
  rules.rules: rules.rules.
  remat_transform.f_rem: remat_transform().f_rem().
  RematTrace.stage_value: RematTrace#stage_value().
  RematTrace.process_call: RematTrace#process_call().
  RematTrace: RematTrace#
  reduce_precision: reduce_precision().
  remat_jaxpr: remat_jaxpr().
  RematTracer.__init__: RematTracer#__init__().
  RematTrace.parent_trace: RematTrace#parent_trace.
  RematTrace.jaxpr_trace: RematTrace#jaxpr_trace.
  _remat_jaxpr.new_arg: _remat_jaxpr().new_arg().
  zip: zip.
  RematTracer._trace: RematTracer#_trace.
  RematTrace.__init__: RematTrace#__init__().
  RematTrace.tag: RematTrace#tag.
  reduce_precision_handlers.reduce_precision_handlers: reduce_precision_handlers.reduce_precision_handlers.
  RematTracer.val: RematTracer#val.
  RematTracer.tracer: RematTracer#tracer.
  RematTrace.policy: RematTrace#policy.
  RematTrace.requires_low: RematTrace#requires_low.
---
# Module: [`jax/_src/interpreters/remat.py`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py)

## Classes
### `RematTrace`
- def: [`jax/_src/interpreters/remat.py:74`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L74)
- members:
  - `process_call(self, call_primitive, f, tracers, params)` — [`L112`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L112)
  - `process_custom_jvp_call(self, prim, fun, jvp, tracers,, *, symbolic_zeros)` — [`L116`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L116)
  - `process_custom_vjp_call(self, prim, f, fwd, bwd, tracers,, *, out_trees, symbolic_zeros)` — [`L125`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L125)
  - `process_primitive(self, prim, tracers, params,)` — [`L95`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L95)
  - `stage_value(self, val)` — [`L90`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L90)
  - `to_val_tracer_pair(self, x)` — [`L84`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L84)
  - `jaxpr_trace` — [`L78`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L78)
  - `parent_trace` — [`L77`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L77)
  - `policy` — [`L79`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L79)
  - `requires_low` — [`L80`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L80)
  - `tag` — [`L82`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L82)
- protocol/private: `__init__`[`L75`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L75)
- uses (calls/refs, reference-scoped): [`unzip2`](../util.md#unzip2), [`set_current_trace`](../core.md#set_current_trace), [`Trace`](../core.md#Trace), [`map`](remat.md#map), [`RematTracer`](remat.md#RematTracer), [`rules`](remat.md#rules.rules), [`__init__`](../core.md#Trace.__init__), [`_trace`](remat.md#RematTracer._trace), [`tracer`](remat.md#RematTracer.tracer), [`val`](remat.md#RematTracer.val)
- used by: [`remat_transform`](remat.md#remat_transform), [`remat_subtrace`](remat.md#remat_subtrace), [`_remat_jaxpr`](remat.md#_remat_jaxpr), [`process_call`](../core.md#Trace.process_call), [`process_custom_jvp_call`](../core.md#Trace.process_custom_jvp_call), [`process_custom_vjp_call`](../core.md#Trace.process_custom_vjp_call), [`process_primitive`](../core.md#Trace.process_primitive), [`stage_value`](../core.md#Trace.stage_value), [`_trace`](remat.md#RematTracer._trace)

### `RematTracer`
- def: [`jax/_src/interpreters/remat.py:66`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L66)
- members:
  - `tracer` — [`L72`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L72)
  - `val` — [`L71`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L71)
- protocol/private: `__init__`[`L69`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L69), `_trace`[`L67`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L67)
- uses (calls/refs, reference-scoped): [`typeof`](../core.md#typeof), [`Tracer`](../core.md#Tracer), [`__init__`](../core.md#Tracer.__init__), [`RematTrace`](remat.md#RematTrace)
- used by: [`remat_transform`](remat.md#remat_transform), [`_shard_map_remat`](../shard_map.md#_shard_map_remat), [`remat_subtrace`](remat.md#remat_subtrace), [`process_primitive`](remat.md#RematTrace.process_primitive), [`to_val_tracer_pair`](remat.md#RematTrace.to_val_tracer_pair), [`process_custom_jvp_call`](remat.md#RematTrace.process_custom_jvp_call), [`process_custom_vjp_call`](remat.md#RematTrace.process_custom_vjp_call), [`stage_value`](remat.md#RematTrace.stage_value), [`new_arg`](remat.md#_remat_jaxpr.new_arg)

## Functions
- `_remat_jaxpr(jaxpr, policy)` — [`L177`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L177)
- `f_rem(res, *args)` — [`L60`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L60)
- `new_arg(a)` — [`L185`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L185)
- `reduce_precision(x)` — [`L164`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L164)
- `remat_jaxpr(jaxpr, policy)` — [`L173`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L173)
- `remat_subtrace(f: Callable, tag: core.TraceTag, policy, debug_info: core.DebugInfo, args)` — [`L134`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L134)
- `remat_transform(policy, f, *args)` — [`L39`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L39)

## Module values
- `map` — [`L31`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L31)
- `reduce_precision_handlers` — [`L170`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L170)
- `rules` — [`L169`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L169)
- `zip` — [`L32`](../../../../../../../raw/code/jax/jax/_src/interpreters/remat.py#L32)

