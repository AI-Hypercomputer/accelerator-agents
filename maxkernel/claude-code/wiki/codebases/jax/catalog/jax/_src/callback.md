---
title: 'Module: jax/_src/callback.py'
type: catalog
provenance: extracted
module: jax/_src/callback.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.callback`/
symbols:
  io_callback: io_callback().
  _callback_op_sharding: _callback_op_sharding().
  emit_python_callback: emit_python_callback().
  _get_sdy_array_list_for_callbacks: _get_sdy_array_list_for_callbacks().
  receive_from_host: receive_from_host().
  pure_callback: pure_callback().
  send_to_host: send_to_host().
  pure_callback_p: pure_callback_p.
  pure_callback_impl: pure_callback_impl().
  io_callback_p: io_callback_p.
  io_callback_impl: io_callback_impl().
  io_callback_abstract_eval: io_callback_abstract_eval().
  _OrderedIOEffect: _OrderedIOEffect.
  emit_python_callback._wrapped_callback: emit_python_callback()._wrapped_callback().
  pure_callback_lowering: pure_callback_lowering().
  _make_array_shape: _make_array_shape().
  _emit_tpu_python_callback: _emit_tpu_python_callback().
  _FlatCallback.__call__: _FlatCallback#__call__().
  io_callback_batching_rule: io_callback_batching_rule().
  io_callback_lowering: io_callback_lowering().
  pure_callback_abstract_eval: pure_callback_abstract_eval().
  _aval_to_xla_shape: _aval_to_xla_shape().
  _IOEffect: _IOEffect.
  io_callback_batching_rule._batch_fun: io_callback_batching_rule()._batch_fun().
  _FlatCallback: _FlatCallback#
  is_empty_shape: is_empty_shape().
  _xla_shape_handlers._xla_shape_handlers: _xla_shape_handlers._xla_shape_handlers.
  OrderedIOEffect: OrderedIOEffect#
  _check_shape_dtype: _check_shape_dtype().
  io_callback_lowering._callback: io_callback_lowering()._callback().
  _FlatCallback.in_tree: _FlatCallback#in_tree.
  pure_callback_lowering._callback: pure_callback_lowering()._callback().
  IOEffect: IOEffect#
  unsafe_map: unsafe_map.
  unsafe_zip: unsafe_zip.
  logger: logger.
  _XLA_HOST_TRANSFER_PJRT_RENDEZVOUS_HANDLER_NAME: _XLA_HOST_TRANSFER_PJRT_RENDEZVOUS_HANDLER_NAME.
  _FlatCallback.callback_func: _FlatCallback#callback_func.
  pure_callback_jvp_rule: pure_callback_jvp_rule().
  pure_callback_transpose_rule: pure_callback_transpose_rule().
  io_callback_jvp_rule: io_callback_jvp_rule().
  io_callback_transpose_rule: io_callback_transpose_rule().
  IOEffect.__str__: IOEffect#__str__.
  OrderedIOEffect.__str__: OrderedIOEffect#__str__.
---
# Module: [`jax/_src/callback.py`](../../../../../../raw/code/jax/jax/_src/callback.py)

## Classes
### `IOEffect`  ·  implements/extends Effect
- def: [`jax/_src/callback.py:402`](../../../../../../raw/code/jax/jax/_src/callback.py#L402)
- signature: `class IOEffect(effects.Effect):`
- protocol/private: `__str__`[`L403`](../../../../../../raw/code/jax/jax/_src/callback.py#L403)
- used by: [`_OrderedIOEffect`](callback.md#_OrderedIOEffect), [`_IOEffect`](callback.md#_IOEffect)

### `OrderedIOEffect`  ·  implements/extends Effect
- def: [`jax/_src/callback.py:405`](../../../../../../raw/code/jax/jax/_src/callback.py#L405)
- signature: `class OrderedIOEffect(effects.Effect):`
- protocol/private: `__str__`[`L406`](../../../../../../raw/code/jax/jax/_src/callback.py#L406)
- used by: [`_OrderedIOEffect`](callback.md#_OrderedIOEffect)

### `_FlatCallback`
- def: [`jax/_src/callback.py:58`](../../../../../../raw/code/jax/jax/_src/callback.py#L58)
- doc: A Python function callable with flat arguments and results.
- signature: `class _FlatCallback:`
- members:
  - `callback_func` — [`L66`](../../../../../../raw/code/jax/jax/_src/callback.py#L66)
  - `in_tree` — [`L67`](../../../../../../raw/code/jax/jax/_src/callback.py#L67)
- protocol/private: `__call__`[`L69`](../../../../../../raw/code/jax/jax/_src/callback.py#L69)
- uses (calls/refs, reference-scoped): [`Array`](basearray.md#Array), [`tree_unflatten`](tree_util.md#tree_unflatten), [`PyTreeDef`](tree_util.md#PyTreeDef), [`tree_leaves`](tree_util.md#tree_leaves)
- used by: [`io_callback`](callback.md#io_callback), [`pure_callback`](callback.md#pure_callback), [`io_callback_impl`](callback.md#io_callback_impl), [`pure_callback_impl`](callback.md#pure_callback_impl), [`io_callback_abstract_eval`](callback.md#io_callback_abstract_eval), [`pure_callback_lowering`](callback.md#pure_callback_lowering), [`pure_callback_abstract_eval`](callback.md#pure_callback_abstract_eval)

## Functions
- `_aval_to_xla_shape(aval: core.AbstractValue)` — [`L674`](../../../../../../raw/code/jax/jax/_src/callback.py#L674)
- `_batch_fun(batched_args)` — [`L482`](../../../../../../raw/code/jax/jax/_src/callback.py#L482)
- `_callback(*flat_args)` — [`L224`](../../../../../../raw/code/jax/jax/_src/callback.py#L224)
- `_callback(*flat_args)` — [`L492`](../../../../../../raw/code/jax/jax/_src/callback.py#L492)
- `_callback_op_sharding(axis_context, sharding: Sharding | None, avals_out)` — [`L151`](../../../../../../raw/code/jax/jax/_src/callback.py#L151)
- `_check_shape_dtype(shape_dtype)` — [`L254`](../../../../../../raw/code/jax/jax/_src/callback.py#L254)
- `_emit_tpu_python_callback(backend: xc.Client, ctx: mlir.LoweringRuleContext, callback, token: Any | None, operands: Sequence[ir.Value], operand_avals: Sequence[core.ShapedArray], operand_shapes: Sequence[xc.Shape], result_avals: Sequence[core.ShapedArray], result_shapes: Sequence[xc.Shape], *, returns_token: bool, sharding: SdyArrayList | xc.OpSharding | None = None)` — [`L692`](../../../../../../raw/code/jax/jax/_src/callback.py#L692)
- `_get_sdy_array_list_for_callbacks(avals: Sequence[core.ShapedArray])` — [`L138`](../../../../../../raw/code/jax/jax/_src/callback.py#L138) — Returns an SdyArrayList with `max(1, len(avals))` replicated shardings.
- `_make_array_shape(aval: core.ShapedArray)` — [`L683`](../../../../../../raw/code/jax/jax/_src/callback.py#L683)
- `_wrapped_callback(*args)` — [`L815`](../../../../../../raw/code/jax/jax/_src/callback.py#L815)
- `emit_python_callback(ctx: mlir.LoweringRuleContext, callback, token: Any | None, operands: Sequence[ir.Value], operand_avals: Sequence[core.ShapedArray], result_avals: Sequence[core.ShapedArray], *, has_side_effect: bool, returns_token: bool = True, partitioned: bool = False, sharding: SdyArrayList | xc.OpSharding | None = None)` — [`L765`](../../../../../../raw/code/jax/jax/_src/callback.py#L765) — Emits MLIR that calls back to a provided Python function.
- `io_callback(callback: Callable[..., Any], result_shape_dtypes: Any, *args: Any, sharding: Sharding | None = None, ordered: bool = False, **kwargs: Any)` — [`L538`](../../../../../../raw/code/jax/jax/_src/callback.py#L538) — Calls an impure Python callback.
- `io_callback_abstract_eval(*avals, callback: _FlatCallback, result_avals, sharding: Sharding | None, ordered: bool)` — [`L448`](../../../../../../raw/code/jax/jax/_src/callback.py#L448)
- `io_callback_batching_rule(args, dims, callback, result_avals, sharding, ordered)` — [`L472`](../../../../../../raw/code/jax/jax/_src/callback.py#L472)
- `io_callback_impl(*args, result_avals, callback: _FlatCallback, sharding: Sharding | None, ordered: bool)` — [`L418`](../../../../../../raw/code/jax/jax/_src/callback.py#L418)
- `io_callback_jvp_rule(*args, **kwargs)` — [`L460`](../../../../../../raw/code/jax/jax/_src/callback.py#L460)
- `io_callback_lowering(ctx, *args, callback, sharding, ordered, **params)` — [`L491`](../../../../../../raw/code/jax/jax/_src/callback.py#L491)
- `io_callback_transpose_rule(*args, **kwargs)` — [`L466`](../../../../../../raw/code/jax/jax/_src/callback.py#L466)
- `is_empty_shape(s: core.Shape)` — [`L589`](../../../../../../raw/code/jax/jax/_src/callback.py#L589)
- `pure_callback(callback: Callable[..., Any], result_shape_dtypes: Any, *args: Any, sharding: Sharding | None = None, vmap_method: str | None = None, **kwargs: Any)` — [`L261`](../../../../../../raw/code/jax/jax/_src/callback.py#L261) — Calls a pure Python callback. Works under :func:`jit`/:func:`~vmap`/etc.
- `pure_callback_abstract_eval(*avals, callback: _FlatCallback, result_avals, sharding: Sharding | None, vmap_method: str | None)` — [`L104`](../../../../../../raw/code/jax/jax/_src/callback.py#L104)
- `pure_callback_impl(*args, result_avals, callback: _FlatCallback, sharding: Sharding | None, vmap_method: str | None)` — [`L74`](../../../../../../raw/code/jax/jax/_src/callback.py#L74)
- `pure_callback_jvp_rule(*args, **kwargs)` — [`L115`](../../../../../../raw/code/jax/jax/_src/callback.py#L115)
- `pure_callback_lowering(ctx, *args, callback: _FlatCallback, sharding: Sharding | None, **params)` — [`L221`](../../../../../../raw/code/jax/jax/_src/callback.py#L221)
- `pure_callback_transpose_rule(*args, **kwargs)` — [`L125`](../../../../../../raw/code/jax/jax/_src/callback.py#L125)
- `receive_from_host(ctx: mlir.ModuleContext, channel: int, token: ir.Value[hlo.TokenType], out_aval: core.ShapedArray, name: str | None = None, *, sharding: SdyArrayList | xc.OpSharding | None = None)` — [`L632`](../../../../../../raw/code/jax/jax/_src/callback.py#L632)
- `send_to_host(ctx: mlir.ModuleContext, channel: int, token: ir.Value[hlo.TokenType], operand: Any, name: str | None = None, *, sharding: SdyArrayList | xc.OpSharding | None = None)` — [`L596`](../../../../../../raw/code/jax/jax/_src/callback.py#L596)

## Module values
- `_IOEffect` — [`L408`](../../../../../../raw/code/jax/jax/_src/callback.py#L408)
- `_OrderedIOEffect` — [`L409`](../../../../../../raw/code/jax/jax/_src/callback.py#L409)
- `_XLA_HOST_TRANSFER_PJRT_RENDEZVOUS_HANDLER_NAME` — [`L593`](../../../../../../raw/code/jax/jax/_src/callback.py#L593)
- `_xla_shape_handlers` — [`L680`](../../../../../../raw/code/jax/jax/_src/callback.py#L680)
- `io_callback_p` — [`L398`](../../../../../../raw/code/jax/jax/_src/callback.py#L398)
- `logger` — [`L46`](../../../../../../raw/code/jax/jax/_src/callback.py#L46)
- `pure_callback_p` — [`L49`](../../../../../../raw/code/jax/jax/_src/callback.py#L49)
- `unsafe_map` — [`L53`](../../../../../../raw/code/jax/jax/_src/callback.py#L53)
- `unsafe_zip` — [`L54`](../../../../../../raw/code/jax/jax/_src/callback.py#L54)

