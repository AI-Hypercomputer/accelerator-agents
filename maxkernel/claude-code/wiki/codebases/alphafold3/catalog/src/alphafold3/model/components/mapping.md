---
title: 'Module: src/alphafold3/model/components/mapping.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/components/mapping.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.components.mapping`/
symbols:
  sharded_apply.mapped_fn: sharded_apply().mapped_fn().
  sharded_apply: sharded_apply().
  sharded_apply.mapped_fn.compute_shard: sharded_apply().mapped_fn().compute_shard().
  inference_subbatch: inference_subbatch().
  sharded_map: sharded_map().
  PytreeJaxArray: PytreeJaxArray.
  _set_docstring: _set_docstring().
  sharded_apply.mapped_fn.apply_fun_to_slice: sharded_apply().mapped_fn().apply_fun_to_slice().
  _expand_axes: _expand_axes().
  Pytree: Pytree.
  T: T.
  _maybe_slice: _maybe_slice().
  _maybe_get_size: _maybe_get_size().
  _set_docstring.wrapped: _set_docstring().wrapped().
  sharded_apply.mapped_fn.scan_iteration: sharded_apply().mapped_fn().scan_iteration().
  partial: partial.
  PROXY: PROXY.
  sharded_apply.mapped_fn.make_output_shape: sharded_apply().mapped_fn().make_output_shape().
  sharded_apply.mapped_fn.dynamic_update_slice_in_dim: sharded_apply().mapped_fn().dynamic_update_slice_in_dim().
  sharded_apply.mapped_fn.allocate_buffer: sharded_apply().mapped_fn().allocate_buffer().
  inference_subbatch.run_module: inference_subbatch().run_module().
---
# Module: [`src/alphafold3/model/components/mapping.py`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/mapping.py)

## Functions
- `_expand_axes(axes, values, name="sharded_apply")` — [`L56`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/mapping.py#L56) — documented in [alphafold3-model-components-mapping](../../../../../concepts/alphafold3-model-components-mapping.md)
- `_maybe_get_size(array, axis)` — [`L49`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/mapping.py#L49)
- `_maybe_slice(array, i, slice_size, axis)` — [`L40`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/mapping.py#L40) — documented in [alphafold3-model-components-mapping](../../../../../concepts/alphafold3-model-components-mapping.md)
- `_set_docstring(docstr: str)` — [`L96`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/mapping.py#L96) — Decorator for setting the docstring of a function.
- `allocate_buffer(dtype, shape)` — [`L209`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/mapping.py#L209) — documented in [alphafold3-model-components-mapping](../../../../../concepts/alphafold3-model-components-mapping.md)
- `apply_fun_to_slice(slice_start, slice_size)` — [`L159`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/mapping.py#L159) — documented in [alphafold3-model-components-mapping](../../../../../concepts/alphafold3-model-components-mapping.md)
- `compute_shard(outputs, slice_start, slice_size)` — [`L198`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/mapping.py#L198) — documented in [alphafold3-model-components-mapping](../../../../../concepts/alphafold3-model-components-mapping.md)
- `dynamic_update_slice_in_dim(full_array, update, axis, i)` — [`L195`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/mapping.py#L195) — documented in [alphafold3-model-components-mapping](../../../../../concepts/alphafold3-model-components-mapping.md)
- `inference_subbatch(module: Callable[..., PytreeJaxArray], subbatch_size: int, batched_args: Sequence[PytreeJaxArray], nonbatched_args: Sequence[PytreeJaxArray], input_subbatch_dim: int = 0, output_subbatch_dim: int | None = None)` — [`L226`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/mapping.py#L226) — Run through subbatches (like batch apply but with split and concat). — documented in [alphafold3-model-components-mapping](../../../../../concepts/alphafold3-model-components-mapping.md)
- `make_output_shape(axis, shard_shape, remainder_shape)` — [`L182`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/mapping.py#L182) — documented in [alphafold3-model-components-mapping](../../../../../concepts/alphafold3-model-components-mapping.md)
- `mapped_fn(*args, **kwargs)` — [`L146`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/mapping.py#L146) — documented in [alphafold3-model-components-mapping](../../../../../concepts/alphafold3-model-components-mapping.md)
- `run_module(*batched_args)` — [`L244`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/mapping.py#L244)
- `scan_iteration(outputs, i)` — [`L203`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/mapping.py#L203) — documented in [alphafold3-model-components-mapping](../../../../../concepts/alphafold3-model-components-mapping.md)
- `sharded_apply(fun: Callable[..., PytreeJaxArray], shard_size: int | None = 1, in_axes: int | Pytree = 0, out_axes: int | Pytree = 0, new_out_axes: bool = False)` — [`L106`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/mapping.py#L106) — Sharded apply. — documented in [alphafold3-model-components-mapping](../../../../../concepts/alphafold3-model-components-mapping.md)
- `sharded_map(fun: Callable[..., PytreeJaxArray], shard_size: int | None = 1, in_axes: int | Pytree = 0, out_axes: int | Pytree = 0)` — [`L64`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/mapping.py#L64) — Sharded vmap. — documented in [alphafold3-model-components-mapping](../../../../../concepts/alphafold3-model-components-mapping.md)
- `wrapped(fun: T)` — [`L99`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/mapping.py#L99)

## Module values
- `PROXY` — [`L35`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/mapping.py#L35) — documented in [alphafold3-model-components-mapping](../../../../../concepts/alphafold3-model-components-mapping.md)
- `Pytree` — [`L31`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/mapping.py#L31)
- `PytreeJaxArray` — [`L32`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/mapping.py#L32)
- `T` — [`L37`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/mapping.py#L37)
- `partial` — [`L34`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/mapping.py#L34)

