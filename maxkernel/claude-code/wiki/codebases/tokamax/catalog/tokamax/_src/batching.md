---
title: 'Module: tokamax/_src/batching.py'
type: catalog
provenance: extracted
module: tokamax/_src/batching.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.batching`/
symbols:
  capture_batched_args.wrapped: capture_batched_args().wrapped().
  vmap_split.vmapped: vmap_split().vmapped().
  BatchedShapeDtype: BatchedShapeDtype#
  capture_batched_args.wrapped.vmap_rule: capture_batched_args().wrapped().vmap_rule().
  vmap_split: vmap_split().
  vmap_maybe_bcast: vmap_maybe_bcast().
  get_vmap_axis_sizes: get_vmap_axis_sizes().
  BatchedShapeDtype.vmap_axes: BatchedShapeDtype#vmap_axes.
  vmap_maybe_bcast.vmapped: vmap_maybe_bcast().vmapped().
  _P: _P.
  capture_batched_args.wrapped.vmap_rule.vmap_fn_flat: capture_batched_args().wrapped().vmap_rule().vmap_fn_flat().
  BatchedShapeDtype.__eq__: BatchedShapeDtype#__eq__().
  BatchedShapeDtype.vmap_shape: BatchedShapeDtype#vmap_shape().
  capture_batched_args: capture_batched_args().
  capture_batched_args.wrapped.fn_flat: capture_batched_args().wrapped().fn_flat().
  _T: _T.
  vmap_split.vmapped.arg_axis: vmap_split().vmapped().arg_axis().
  BatchedShapeDtype.vmap_axis_sizes: BatchedShapeDtype#vmap_axis_sizes().
  BatchedShapeDtype.__repr__: BatchedShapeDtype#__repr__().
  BatchedShapeDtype.__init__: BatchedShapeDtype#__init__().
  BatchedShapeDtype.__str__: BatchedShapeDtype#__str__.
  BatchedShapeDtype.__hash__: BatchedShapeDtype#__hash__().
  _zip: _zip.
  _broadcast_prefix: _broadcast_prefix().
  capture_batched_args.bind: capture_batched_args().bind().
  vmap_maybe_bcast.vmapped.arg_axis: vmap_maybe_bcast().vmapped().arg_axis().
  _split_dim: _split_dim().
  _unique_not_none_value: _unique_not_none_value().
  BatchedShapeDtype.__slots__: BatchedShapeDtype#__slots__.
---
# Module: [`tokamax/_src/batching.py`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py)

## Classes
### `BatchedShapeDtype`  ·  implements/extends ShapeDtypeStruct
- def: [`tokamax/_src/batching.py:102`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L102)
- doc: A container for the shape, dtype, and vmap axes of an array.
- signature: `class BatchedShapeDtype(jax.ShapeDtypeStruct):`
- members:
  - `vmap_axis_sizes(self)` — [`L124`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L124)
  - `vmap_shape(self)` — [`L115`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L115) — Returns the shape of the array with the `vmap` axes added.
  - `vmap_axes` — [`L108`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L108)
- protocol/private: `__eq__`[`L135`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L135), `__hash__`[`L142`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L142), `__init__`[`L110`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L110), `__repr__`[`L130`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L130), `__slots__`[`L105`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L105), `__str__`[`L133`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L133)
- used by: [`test_dot_product_attention`](ops/attention/api_test.md#DotProductAttentionTest.test_dot_product_attention), [`wrapped`](batching.md#capture_batched_args.wrapped), [`vmap_rule`](batching.md#capture_batched_args.wrapped.vmap_rule), [`_bsd`](ops/op_test.md#_bsd), [`get_vmap_axis_sizes`](batching.md#get_vmap_axis_sizes), [`convert_batched`](benchmarking.md#standardize_function.convert_batched), [`BatchedShapeDtype`](ops/attention/arg_specs.md#BatchedShapeDtype), [`BatchedShapeDtype`](ops/gated_linear_unit/arg_specs.md#BatchedShapeDtype)

## Functions
- `_broadcast_prefix(prefix_tree, full_tree)` — [`L33`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L33)
- `_split_dim(x, axis, num_parts)` — [`L67`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L67)
- `_unique_not_none_value(*args)` — [`L156`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L156)
- `arg_axis(x, axis)` — [`L51`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L51)
- `arg_axis(x, axis)` — [`L89`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L89)
- `bind(*args, **kwargs)` — [`L164`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L164)
- `capture_batched_args(fn: Callable[..., Any])` — [`L161`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L161) — Captures the batched arguments shapes and passes them to the function.
- `fn_flat(*arrays, batched_args=batched_args)` — [`L188`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L188)
- `get_vmap_axis_sizes(values)` — [`L147`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L147)
- `vmap_fn_flat(*arrays, batched_args)` — [`L204`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L204)
- `vmap_maybe_bcast(f: Callable[_P, _T], in_axes: Any)` — [`L42`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L42) — `vmap`s `f` over (possibly broadcast) axes of its arguments.
- `vmap_rule(axis_size, in_batched, *args)` — [`L193`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L193)
- `vmap_split(f: Callable[_P, _T], in_axes: Any, *, num_parts: int)` — [`L78`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L78) — `vmap`s `f` over (possibly broadcast) parts of axes of its arguments.
- `vmapped(*args: _P.args, **kwargs: _P.kwargs)` — [`L46`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L46)
- `vmapped(*args: _P.args, **kwargs: _P.kwargs)` — [`L84`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L84)
- `wrapped(*args, batched_args=None, **kwargs)` — [`L172`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L172)

## Module values
- `_P` — [`L28`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L28)
- `_T` — [`L29`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L29)
- `_zip` — [`L30`](../../../../../../raw/code/tokamax/tokamax/_src/batching.py#L30)

