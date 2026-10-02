---
title: 'Module: jaxite/jaxite_cggi/jax_helpers.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/jax_helpers.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.jax_helpers`/
symbols:
  batch_vmap.g: batch_vmap().g().
  _tree_map_multi_output: _tree_map_multi_output().
  batch_vmap: batch_vmap().
  get_tpu_version: get_tpu_version().
  tree_map: tree_map.
  F: F.
  batch_vmap.preprocess: batch_vmap().preprocess().
  batch_vmap.postprocess: batch_vmap().postprocess().
  batch_vmap.g.f2: batch_vmap().g().f2().
  tree_flatten: tree_flatten.
  tree_unflatten: tree_unflatten.
  _lax_map: _lax_map().
---
# Module: [`jaxite/jaxite_cggi/jax_helpers.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/jax_helpers.py)

## Functions
- `_lax_map(f, *xs)` — [`L23`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/jax_helpers.py#L23) — Like lax.map, but supports multiple arguments like the built-in map.
- `_tree_map_multi_output(f, *args)` — [`L14`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/jax_helpers.py#L14) — Like tree_map, but for functions that return tuples.
- `batch_vmap(f: F, in_axes: int | None | Sequence[Any] = 0, out_axes: Any = 0, *, batch_size: int)` — [`L33`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/jax_helpers.py#L33) — jax.vmap, but looping when the batch dimension exceeds batch_size. — documented in [jaxite-jaxite_cggi-rgsw](../../../concepts/jaxite-jaxite_cggi-rgsw.md)
- `f2(*args)` — [`L79`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/jax_helpers.py#L79)
- `g(*args)` — [`L56`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/jax_helpers.py#L56)
- `get_tpu_version()` — [`L98`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/jax_helpers.py#L98) — Returns the numeric version of the TPU, or -1 if not on TPU.
- `postprocess(x_loop, x_tail, out_axis)` — [`L50`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/jax_helpers.py#L50)
- `preprocess(x, in_axis)` — [`L42`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/jax_helpers.py#L42)

## Module values
- `F` — [`L30`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/jax_helpers.py#L30)
- `tree_flatten` — [`L9`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/jax_helpers.py#L9)
- `tree_map` — [`L11`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/jax_helpers.py#L11)
- `tree_unflatten` — [`L10`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/jax_helpers.py#L10)

