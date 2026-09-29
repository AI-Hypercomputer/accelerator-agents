---
title: 'Module: jax/_src/lax/windowed_reductions.py'
type: catalog
provenance: extracted
module: jax/_src/lax/windowed_reductions.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.lax.windowed_reductions`/
symbols:
  _reduce_window: _reduce_window().
  select_and_gather_add_p: select_and_gather_add_p.
  _get_monoid_window_reducer: _get_monoid_window_reducer().
  select_and_scatter_add_p: select_and_scatter_add_p.
  reduce_window_max_p: reduce_window_max_p.
  _select_and_gather_add_lowering: _select_and_gather_add_lowering().
  reduce_window_sum_p: reduce_window_sum_p.
  _select_and_scatter_add_impl: _select_and_scatter_add_impl().
  reduce_window_min_p: reduce_window_min_p.
  _reduce_window_prod: _reduce_window_prod().
  _reduce_window_logaddexp: _reduce_window_logaddexp().
  _reduce_window_abstract_eval_rule: _reduce_window_abstract_eval_rule().
  _select_and_gather_add_transpose: _select_and_gather_add_transpose().
  _select_and_gather_add: _select_and_gather_add().
  _select_and_scatter_add: _select_and_scatter_add().
  _select_and_scatter: _select_and_scatter().
  reduce_window_jvp: reduce_window_jvp().
  _reduce_window_sum_transpose_rule: _reduce_window_sum_transpose_rule().
  _select_and_scatter_lower: _select_and_scatter_lower().
  select_and_scatter_p: select_and_scatter_p.
  _reduce_window_sum: _reduce_window_sum().
  reduce_window_p: reduce_window_p.
  _reduce_window_chooser_jvp_rule: _reduce_window_chooser_jvp_rule().
  _reduce_window_min: _reduce_window_min().
  _reduce_window_max: _reduce_window_max().
  reduce_window_shape_tuple: reduce_window_shape_tuple().
  zip: zip.
  _generic_reduce_window_batch_rule: _generic_reduce_window_batch_rule().
  _select_and_gather_add_using_variadic_reducewindow: _select_and_gather_add_using_variadic_reducewindow().
  reduce_window_sharding_rule: reduce_window_sharding_rule().
  _common_reduce_window_shape_rule: _common_reduce_window_shape_rule().
  _select_and_scatter_add_jvp: _select_and_scatter_add_jvp().
  _select_and_gather_add_lowering.reducer_body: _select_and_gather_add_lowering().reducer_body().
  _select_and_gather_add_jvp: _select_and_gather_add_jvp().
  _reduce_window_min_batch_rule: _reduce_window_min_batch_rule.
  reduce_window: reduce_window().
  reduce_window_jvp.wrapper: reduce_window_jvp().wrapper().
  _generic_reduce_window_lower: _generic_reduce_window_lower().
  _reduce_window_sum_shape_rule: _reduce_window_sum_shape_rule().
  _select_and_scatter_add_transpose: _select_and_scatter_add_transpose().
  _select_and_scatter_add_batch_rule: _select_and_scatter_add_batch_rule().
  _select_and_gather_add_sharding_rule: _select_and_gather_add_sharding_rule().
  _select_and_gather_add_batching_rule: _select_and_gather_add_batching_rule().
  map: map.
  _select_and_gather_add_lowering.pack: _select_and_gather_add_lowering().pack().
  _reduce_window_batch_rule: _reduce_window_batch_rule().
  _select_and_scatter_shape_rule: _select_and_scatter_shape_rule().
  _select_and_gather_add_shape_rule: _select_and_gather_add_shape_rule().
  _select_and_gather_add_using_variadic_reducewindow.reducer: _select_and_gather_add_using_variadic_reducewindow().reducer().
  _reduce_window_lower: _reduce_window_lower().
  _select_and_gather_add_lowering._broadcast_scalar_const: _select_and_gather_add_lowering()._broadcast_scalar_const().
  _select_and_gather_add_lowering.fst: _select_and_gather_add_lowering().fst().
  _select_and_gather_add_lowering.snd: _select_and_gather_add_lowering().snd().
  _generic_reduce_window_lower.reducer_body: _generic_reduce_window_lower().reducer_body().
  _select_and_scatter_sharding_rule: _select_and_scatter_sharding_rule().
  _select_and_scatter_add_shape_rule: _select_and_scatter_add_shape_rule().
  _select_and_scatter_add_sharding_rule: _select_and_scatter_add_sharding_rule().
---
# Module: [`jax/_src/lax/windowed_reductions.py`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py)

## Functions
- `_broadcast_scalar_const(x, aval_out)` — [`L921`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L921)
- `_common_reduce_window_shape_rule(operand, window_dimensions, window_strides, padding, base_dilation, window_dilation)` — [`L590`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L590)
- `_generic_reduce_window_batch_rule(batched_args, batch_dims, *, jaxpr, consts, window_dimensions, window_strides, padding, base_dilation, window_dilation)` — [`L386`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L386)
- `_generic_reduce_window_lower(ctx: mlir.LoweringRuleContext, *args, jaxpr, consts, window_dimensions, window_strides, padding, base_dilation, window_dilation)` — [`L475`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L475)
- `_get_monoid_window_reducer(monoid_op, xs: Sequence[Array])` — [`L179`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L179)
- `_reduce_window(operand, init_value, computation, window_dimensions: core.Shape, window_strides: Sequence[int] | None, padding: str | Sequence[tuple[int, int]], base_dilation: Sequence[int] | None = None, window_dilation: Sequence[int] | None = None)` — [`L47`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L47)
- `_reduce_window_abstract_eval_rule(*avals, jaxpr, consts, window_dimensions, window_strides, padding, base_dilation, window_dilation)` — [`L350`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L350)
- `_reduce_window_batch_rule(reduce_window, batched_args, bdims, *, window_dimensions, window_strides, padding, base_dilation, window_dilation)` — [`L546`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L546)
- `_reduce_window_chooser_jvp_rule(prim, g, operand, *, window_dimensions, window_strides, padding, base_dilation, window_dilation)` — [`L580`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L580)
- `_reduce_window_logaddexp(operand: Array, window_dimensions: core.Shape, window_strides: Sequence[int], padding: Sequence[tuple[int, int]], base_dilation: Sequence[int] | None = None, window_dilation: Sequence[int] | None = None)` — [`L263`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L263)
- `_reduce_window_lower(reduce_op, init_value, ctx, operand, *, window_dimensions, window_strides, padding, base_dilation, window_dilation)` — [`L661`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L661)
- `_reduce_window_max(operand: Array, window_dimensions: core.Shape, window_strides: Sequence[int], padding: Sequence[tuple[int, int]], base_dilation: Sequence[int] | None = None, window_dilation: Sequence[int] | None = None)` — [`L233`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L233)
- `_reduce_window_min(operand: Array, window_dimensions: core.Shape, window_strides: Sequence[int], padding: Sequence[tuple[int, int]], base_dilation: Sequence[int] | None = None, window_dilation: Sequence[int] | None = None)` — [`L248`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L248)
- `_reduce_window_prod(operand: Array, window_dimensions: core.Shape, window_strides: Sequence[int], padding: Sequence[tuple[int, int]], base_dilation: Sequence[int] | None = None, window_dilation: Sequence[int] | None = None)` — [`L214`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L214)
- `_reduce_window_sum(operand: Array, window_dimensions: core.Shape, window_strides: Sequence[int], padding: Sequence[tuple[int, int]], base_dilation: Sequence[int] | None = None, window_dilation: Sequence[int] | None = None)` — [`L199`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L199)
- `_reduce_window_sum_shape_rule(operand, *, window_dimensions, window_strides, padding, base_dilation, window_dilation)` — [`L518`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L518)
- `_reduce_window_sum_transpose_rule(cotangent, operand, *, window_dimensions, window_strides, padding, base_dilation, window_dilation)` — [`L527`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L527)
- `_select_and_gather_add(tangents: Array, operand: Array, select_prim: core.Primitive, window_dimensions: core.Shape, window_strides: Sequence[int], padding: Sequence[tuple[int, int]], base_dilation: Sequence[int], window_dilation: Sequence[int])` — [`L311`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L311) — Extracts the tangent corresponding to the minimum or maximum element in
- `_select_and_gather_add_batching_rule(batched_args, batch_dims, *, select_prim, window_dimensions, window_strides, padding, base_dilation, window_dilation)` — [`L1087`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L1087)
- `_select_and_gather_add_jvp(primals, tangents, *, select_prim, window_dimensions, window_strides, padding, base_dilation, window_dilation)` — [`L1045`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L1045)
- `_select_and_gather_add_lowering(ctx: mlir.LoweringRuleContext, tangents, operand, *, select_prim, window_dimensions, window_strides, padding, base_dilation, window_dilation, max_bits=64)` — [`L904`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L904)
- `_select_and_gather_add_shape_rule(tangents, operand, *, select_prim, window_dimensions, window_strides, padding, base_dilation, window_dilation)` — [`L882`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L882)
- `_select_and_gather_add_sharding_rule(tangents, operand, *, select_prim, window_dimensions, window_strides, padding, base_dilation, window_dilation)` — [`L893`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L893)
- `_select_and_gather_add_transpose(t, tangents, operand, *, select_prim, window_dimensions, window_strides, padding, base_dilation, window_dilation)` — [`L1062`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L1062)
- `_select_and_gather_add_using_variadic_reducewindow(tangents, operand, *, select_prim, window_dimensions, window_strides, padding, base_dilation, window_dilation)` — [`L1027`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L1027)
- `_select_and_scatter(operand: Array, select: Callable, window_dimensions: core.Shape, window_strides: Sequence[int], padding: Sequence[tuple[int, int]], source: Array, init_value: Array, scatter: Callable)` — [`L283`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L283)
- `_select_and_scatter_add(source: Array, operand: Array, select_prim: core.Primitive, window_dimensions: core.Shape, window_strides: Sequence[int], padding: Sequence[tuple[int, int]])` — [`L300`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L300)
- `_select_and_scatter_add_batch_rule(batched_args, batch_dims, *, select_prim, window_dimensions, window_strides, padding)` — [`L816`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L816)
- `_select_and_scatter_add_impl(source, operand, *, select_prim, window_dimensions, window_strides, padding, expand_padding)` — [`L845`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L845)
- `_select_and_scatter_add_jvp(primals, tangents, *, select_prim, window_dimensions, window_strides, padding)` — [`L788`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L788)
- `_select_and_scatter_add_shape_rule(source, operand, *, select_prim, window_dimensions, window_strides, padding)` — [`L778`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L778)
- `_select_and_scatter_add_sharding_rule(source, operand, *, select_prim, window_dimensions, window_strides, padding)` — [`L783`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L783)
- `_select_and_scatter_add_transpose(t, source, operand, *, select_prim, window_dimensions, window_strides, padding)` — [`L805`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L805)
- `_select_and_scatter_lower(ctx: mlir.LoweringRuleContext, operand, source, init_value, *, select_jaxpr: core.Jaxpr, select_consts, scatter_jaxpr: core.Jaxpr, scatter_consts, window_dimensions, window_strides, padding)` — [`L726`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L726)
- `_select_and_scatter_shape_rule(operand, source, init_value, *, select_jaxpr, select_consts, scatter_jaxpr, scatter_consts, window_dimensions, window_strides, padding)` — [`L704`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L704)
- `_select_and_scatter_sharding_rule(operand, source, init_value, *, select_jaxpr, select_consts, scatter_jaxpr, scatter_consts, window_dimensions, window_strides, padding)` — [`L716`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L716)
- `fst(t)` — [`L946`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L946)
- `pack(a, b, ab_aval)` — [`L934`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L934)
- `reduce_window(operand: Any, init_value: Any, computation: Callable, window_dimensions: core.Shape, window_strides: Sequence[int] | None = None, padding: str | Sequence[tuple[int, int]] = "VALID", base_dilation: Sequence[int] | None = None, window_dilation: Sequence[int] | None = None)` — [`L116`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L116) — Reduction over padded windows.
- `reduce_window_jvp(primals, tangents, window_dimensions, window_strides, padding, base_dilation, window_dilation, jaxpr, consts)` — [`L426`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L426)
- `reduce_window_shape_tuple(operand_shape, window_dimensions, window_strides, padding, base_dilation=None, window_dilation=None)` — [`L628`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L628)
- `reduce_window_sharding_rule(operand, window_dimensions, window_strides, padding, base_dilation, window_dilation)` — [`L564`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L564)
- `reducer(x, y)` — [`L1030`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L1030)
- `reducer_body(reducer: ir.Block)` — [`L489`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L489)
- `reducer_body(reducer: ir.Block)` — [`L1004`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L1004)
- `snd(t, t_aval)` — [`L954`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L954)
- `wrapper(left, right)` — [`L450`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L450)

## Module values
- `_reduce_window_min_batch_rule` — [`L655`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L655)
- `map` — [`L43`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L43)
- `reduce_window_max_p` — [`L639`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L639)
- `reduce_window_min_p` — [`L648`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L648)
- `reduce_window_p` — [`L423`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L423)
- `reduce_window_sum_p` — [`L572`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L572)
- `select_and_gather_add_p` — [`L1107`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L1107)
- `select_and_scatter_add_p` — [`L833`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L833)
- `select_and_scatter_p` — [`L721`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L721)
- `zip` — [`L44`](../../../../../../../raw/code/jax/jax/_src/lax/windowed_reductions.py#L44)

