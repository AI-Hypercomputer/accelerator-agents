---
title: 'Module: jax/_src/state/primitives.py'
type: catalog
provenance: extracted
module: jax/_src/state/primitives.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.state.primitives`/
symbols:
  get_ref_and_transforms: get_ref_and_transforms().
  pp_ref_transforms: pp_ref_transforms().
  get_p: get_p.
  swap_p: swap_p.
  ref_swap: ref_swap().
  ref_get: ref_get().
  _batch_indexer: _batch_indexer().
  addupdate_p: addupdate_p.
  _addupdate_abstract_eval: _addupdate_abstract_eval().
  ref_set: ref_set().
  ref_addupdate: ref_addupdate().
  _swap_abstract_eval: _swap_abstract_eval().
  _swap_vmap: _swap_vmap().
  _addupdate_vmap: _addupdate_vmap().
  _swap_pp_rule: _swap_pp_rule().
  _get_vmap: _get_vmap().
  _get_pp_rule: _get_pp_rule().
  _addupdate_pp_rule: _addupdate_pp_rule().
  _get_abstract_eval: _get_abstract_eval().
  _maybe_implicit_cast: _maybe_implicit_cast().
  _ref_lin.mut_lin: _ref_lin().mut_lin().
  broadcast_to: broadcast_to().
  _pp_transforms: _pp_transforms().
  _swap_jvp: _swap_jvp().
  Indexer: Indexer.
  _get_to_lojax: _get_to_lojax().
  _swap_to_lojax: _swap_to_lojax().
  _get_jvp: _get_jvp().
  _get_transpose_fancy: _get_transpose_fancy().
  _swap_transpose_fancy: _swap_transpose_fancy().
  _ref_jvp: _ref_jvp().
  create_linear_p: create_linear_p.
  pin_p: pin_p.
  unpin_p: unpin_p.
  shapeof: shapeof().
  pp_ref_var: pp_ref_var.
  addupdate_jvp_rule: addupdate_jvp_rule().
  broadcast_to_p: broadcast_to_p.
  _broadcast_to_impl: _broadcast_to_impl().
  _ref_lin: _ref_lin().
  _empty_ref_lin: _empty_ref_lin().
  _broadcast_to_abstract_eval: _broadcast_to_abstract_eval().
  pin: pin().
  unpin: unpin().
  addupdate_transpose_fancy: addupdate_transpose_fancy().
  _array_ref_batched: _array_ref_batched().
  _freeze_batched: _freeze_batched().
  _empty_ref_jvp: _empty_ref_jvp().
  _empty_ref_lin.lin: _empty_ref_lin().lin().
  HijaxType: HijaxType.
  create_linear: create_linear().
  _create_linear_abstract_eval: _create_linear_abstract_eval().
  _pin_abstract_eval: _pin_abstract_eval().
  _unpin_abstract_eval: _unpin_abstract_eval().
  unsafe_zip: unsafe_zip.
  _free_ref_jvp: _free_ref_jvp().
  unsafe_map: unsafe_map.
  _state_partial_eval_custom: _state_partial_eval_custom().
  _array_ref_partial_eval_custom: _array_ref_partial_eval_custom().
  _addupdate_partial_eval_custom: _addupdate_partial_eval_custom().
  _lower_create_linear: _lower_create_linear().
  _lower_pin: _lower_pin().
  _lower_unpin: _lower_unpin().
  _linval_to_mlir_type: _linval_to_mlir_type().
---
# Module: [`jax/_src/state/primitives.py`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py)

## Functions
- `_addupdate_abstract_eval(ref_aval: AbstractRef, val_aval: core.AbstractValue, *args: Any, tree)` — [`L451`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L451)
- `_addupdate_partial_eval_custom(saveable, unks_in, inst_in, eqn)` — [`L692`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L692)
- `_addupdate_pp_rule(eqn, context, settings)` — [`L561`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L561)
- `_addupdate_vmap(axis_data, batched_args, batched_dims, *, tree)` — [`L979`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L979)
- `_array_ref_batched(axis_data, vals_in, dims_in, memory_space, kind)` — [`L657`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L657)
- `_array_ref_partial_eval_custom(saveable, unks_in, inst_in, eqn)` — [`L645`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L645)
- `_batch_indexer(indexer: indexing.NDIndexer, dims, axis_size: int, ref_shape: tuple[int, ...], ref_dim: int | batching.NotMapped, idx_is_batched: bool)` — [`L707`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L707) — Converts a batched indexer into an unbatched one.
- `_broadcast_to_abstract_eval(aval, *, shape)` — [`L1094`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1094)
- `_broadcast_to_impl(a, *, shape)` — [`L1089`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1089)
- `_create_linear_abstract_eval(*, ty, memory_space)` — [`L1161`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1161)
- `_empty_ref_jvp(primals, tangents, *, ty, memory_space)` — [`L1131`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1131)
- `_empty_ref_lin(_is_vjp, nzs_in, *, ty, memory_space)` — [`L1138`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1138)
- `_free_ref_jvp(primals, tangents)` — [`L1146`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1146)
- `_freeze_batched(axis_data, vals_in, dims_in)` — [`L669`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L669)
- `_get_abstract_eval(ref_aval: AbstractRef, *args, tree)` — [`L395`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L395)
- `_get_jvp(primals: list[Any], tangents: list[Any], **params: Any)` — [`L577`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L577)
- `_get_pp_rule(eqn, context, settings)` — [`L524`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L524)
- `_get_to_lojax(ref, *idx, tree)` — [`L78`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L78)
- `_get_transpose_fancy(g, ref_, *idx, tree)` — [`L619`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L619)
- `_get_vmap(batched_args, batched_dims, *, tree)` — [`L816`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L816)
- `_linval_to_mlir_type(a)` — [`L1235`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1235)
- `_lower_create_linear(ctx)` — [`L1165`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1165)
- `_lower_pin(ctx, x_op, *, to)` — [`L1186`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1186)
- `_lower_unpin(ctx, x_op)` — [`L1223`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1223)
- `_maybe_implicit_cast(dtype, value)` — [`L264`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L264)
- `_pin_abstract_eval(aval, *, to)` — [`L1181`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1181)
- `_pp_transforms(context: core.JaxprPpContext, transforms: tuple[Transform, ...])` — [`L493`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L493)
- `_ref_jvp(primals, tangents, *, memory_space, kind)` — [`L1103`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1103)
- `_ref_lin(_is_vjp, nzs, x, *, memory_space, kind)` — [`L1113`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1113)
- `_state_partial_eval_custom(saveable, unks_in, inst_in, eqn)` — [`L675`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L675)
- `_swap_abstract_eval(ref_aval: AbstractRef, val_aval: core.AbstractValue, *args: Any, tree)` — [`L412`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L412)
- `_swap_jvp(primals: list[Any], tangents: list[Any], **params: Any)` — [`L588`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L588)
- `_swap_pp_rule(eqn, context, settings)` — [`L538`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L538)
- `_swap_to_lojax(ref, val, *idx, tree)` — [`L185`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L185)
- `_swap_transpose_fancy(g, ref_, x, *idx, **params)` — [`L627`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L627)
- `_swap_vmap(axis_data, batched_args, batched_dims, *, tree)` — [`L888`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L888)
- `_unpin_abstract_eval(aval)` — [`L1219`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1219)
- `addupdate_jvp_rule(primals: list[Any], tangents: list[Any], **params: Any)` — [`L607`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L607)
- `addupdate_transpose_fancy(cts_in, ref_, x, *idx, **params)` — [`L637`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L637)
- `broadcast_to(a: Array, shape: tuple[int, ...])` — [`L1069`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1069) — Broadcasts an array to a new shape.
- `create_linear(ty, memory_space=None)` — [`L1156`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1156)
- `get_ref_and_transforms(ref_or_view: Any, idx: Indexer | tuple[Indexer, ...] | None, function_name: str)` — [`L91`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L91) — documented in [jax-_src-state-types](../../../../concepts/jax-_src-state-types.md)
- `lin(_)` — [`L1140`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1140)
- `mut_lin(_, x_dot)` — [`L1116`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1116)
- `pin(x, *, to=None)` — [`L1176`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1176)
- `pp_ref_transforms(context: core.JaxprPpContext, ref, transforms=())` — [`L504`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L504)
- `ref_addupdate(ref: core.Ref | TransformedRef, idx: Indexer | tuple[Indexer, ...] | None, x: ArrayLike | HijaxType)` — [`L341`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L341) — Add to an element in an Ref in-place.
- `ref_get(ref: core.Ref | TransformedRef, idx: Indexer | tuple[Indexer, ...] | None = None)` — [`L120`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L120) — Read a value from an Ref.
- `ref_set(ref: core.Ref | TransformedRef, idx: Indexer | tuple[Indexer, ...] | None, value: ArrayLike | HijaxType)` — [`L278`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L278) — Set a value in an Ref in-place.
- `ref_swap(ref: core.Ref | TransformedRef, idx: Indexer | tuple[Indexer, ...] | None, value: ArrayLike | HijaxType, _function_name: str = "ref_swap")` — [`L202`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L202) — Update an array value inplace while returning the previous value.
- `shapeof(x)` — [`L813`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L813)
- `unpin(x)` — [`L1214`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1214)

## Module values
- `HijaxType` — [`L53`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L53)
- `Indexer` — [`L88`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L88)
- `addupdate_p` — [`L335`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L335)
- `broadcast_to_p` — [`L1067`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1067)
- `create_linear_p` — [`L1158`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1158)
- `get_p` — [`L73`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L73)
- `pin_p` — [`L1178`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1178)
- `pp_ref_var` — [`L489`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L489)
- `swap_p` — [`L180`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L180)
- `unpin_p` — [`L1216`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L1216)
- `unsafe_map` — [`L60`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L60)
- `unsafe_zip` — [`L61`](../../../../../../../raw/code/jax/jax/_src/state/primitives.py#L61)

