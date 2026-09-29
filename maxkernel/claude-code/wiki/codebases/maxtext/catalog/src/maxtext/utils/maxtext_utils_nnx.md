---
title: 'Module: src/maxtext/utils/maxtext_utils_nnx.py'
type: catalog
provenance: extracted
module: src/maxtext/utils/maxtext_utils_nnx.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.utils.maxtext_utils_nnx`/
symbols:
  nnx_sync_moveaxis._op: nnx_sync_moveaxis()._op().
  nnx_remove_scan_axis._op: nnx_remove_scan_axis()._op().
  nnx_add_scan_axis._op: nnx_add_scan_axis()._op().
  create_nnx_sharded_model: create_nnx_sharded_model().
  nnx_update_sharding_meta: nnx_update_sharding_meta().
  nnx_ensure_scan_leading_axis: nnx_ensure_scan_leading_axis().
  nnx_sync_moveaxis: nnx_sync_moveaxis().
  nnx_remove_scan_axis: nnx_remove_scan_axis().
  nnx_add_scan_axis: nnx_add_scan_axis().
  nnx_extract_named_sharding: nnx_extract_named_sharding().
  create_nnx_sharded_model.create_sharded_state: create_nnx_sharded_model().create_sharded_state().
  nnx_ensure_scan_leading_axis._op: nnx_ensure_scan_leading_axis()._op().
  nnx_sync_moveaxis._op.move_fn: nnx_sync_moveaxis()._op().move_fn().
  nnx_remove_scan_axis._op.remove_fn: nnx_remove_scan_axis()._op().remove_fn().
  nnx_add_scan_axis._op.add_fn: nnx_add_scan_axis()._op().add_fn().
  create_nnx_rngs: create_nnx_rngs().
  get_partition_spec_nnx: get_partition_spec_nnx().
  set_named_sharding_nnx: set_named_sharding_nnx().
  move_memory_to_host: move_memory_to_host().
  move_memory_to_device: move_memory_to_device().
---
# Module: [`src/maxtext/utils/maxtext_utils_nnx.py`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils_nnx.py)

## Functions
- `_op(x)` — [`L186`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils_nnx.py#L186)
- `_op(x)` — [`L226`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils_nnx.py#L226)
- `_op(x)` — [`L251`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils_nnx.py#L251)
- `_op(x)` — [`L270`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils_nnx.py#L270)
- `add_fn(l)` — [`L274`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils_nnx.py#L274)
- `create_nnx_rngs(config: pyconfig.HyperParameters, model_mode: str = MODEL_MODE_TRAIN, rng_key: jax.Array | None = None)` — [`L29`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils_nnx.py#L29) — Create NNX Rngs
- `create_nnx_sharded_model(abstract_model: nnx.Module, init_fn: Callable, mesh: Mesh | None = None, named_sharding: nnx.State | None = None)` — [`L142`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils_nnx.py#L142) — Create the model with the given sharding.
- `create_sharded_state()` — [`L173`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils_nnx.py#L173)
- `get_partition_spec_nnx(named_sharding: nnx.State)` — [`L78`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils_nnx.py#L78) — Get mesh partition spec from named sharding.
- `move_fn(l)` — [`L236`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils_nnx.py#L236)
- `move_memory_to_device(path: tuple[str, ...], x: NamedSharding)` — [`L125`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils_nnx.py#L125) — Change the memory_kind of the NamedSharding to "device". This function can be
- `move_memory_to_host(path: tuple[str, ...], x: NamedSharding)` — [`L108`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils_nnx.py#L108) — Change the memory_kind of the NamedSharding to "pinned_host". This function can be
- `nnx_add_scan_axis(tree, name="layers", pos=0)` — [`L267`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils_nnx.py#L267) — Adds the given scan axis to the PartitionSpec at the specified position.
- `nnx_ensure_scan_leading_axis(tree, length)` — [`L183`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils_nnx.py#L183) — Broadcasts scalar-like variables to have a leading scan axis.
- `nnx_extract_named_sharding(abstract_state: nnx.State)` — [`L55`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils_nnx.py#L55) — Get named sharding from NNX abstract state.
- `nnx_remove_scan_axis(tree, name="layers")` — [`L248`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils_nnx.py#L248) — Removes the given scan axis from the PartitionSpec.
- `nnx_sync_moveaxis(tree, from_axis, to_axis)` — [`L221`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils_nnx.py#L221) — Moves an axis in both values and sharding metadata of nnx.Variables.
- `nnx_update_sharding_meta(variable, transform_fn)` — [`L202`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils_nnx.py#L202) — Generic helper to apply a list transformation to all sharding-related metadata.
- `remove_fn(l)` — [`L255`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils_nnx.py#L255)
- `set_named_sharding_nnx(abstract_state: nnx.State, named_sharding: nnx.State)` — [`L95`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils_nnx.py#L95) — Set named sharding to NNX abstract state.

