---
title: 'Module: simply/utils/sharding.py'
type: catalog
provenance: extracted
module: simply/utils/sharding.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.sharding`/
symbols:
  with_sharding_constraint: with_sharding_constraint().
  set_mesh: set_mesh().
  reduce_across_hosts: reduce_across_hosts().
  sum_across_hosts: sum_across_hosts().
  MultihostData.load_async: MultihostData#load_async().
  partition_spec: partition_spec().
  get_partition_size: get_partition_size().
  mesh_sharding: mesh_sharding().
  _inner_partition_with_minimum_redundancy: _inner_partition_with_minimum_redundancy().
  batch_partition_with_minimum_redundancy: batch_partition_with_minimum_redundancy().
  MultihostData.save: MultihostData#save().
  _local_pytrees_to_global: _local_pytrees_to_global().
  multihost_sharded: multihost_sharded().
  partition_with_minimum_redundancy: partition_with_minimum_redundancy().
  create_mesh: create_mesh().
  pytree_ragged_stack_allgather: pytree_ragged_stack_allgather().
  named_sharding: named_sharding().
  get_partition_axis: get_partition_axis().
  get_array_sharding: get_array_sharding().
  set_default_mesh_shape: set_default_mesh_shape().
  MultihostData.global_data: MultihostData#global_data.
  MultihostData.local_data: MultihostData#local_data.
  MultihostData.load: MultihostData#load().
  PartitionAnnotation: PartitionAnnotation.
  get_default_mesh: get_default_mesh().
  NOT_ANNOTATED: NOT_ANNOTATED.
  max_across_hosts: max_across_hosts().
  MultihostData: MultihostData#
  mesh_context: mesh_context().
  MultihostData.snapshot: MultihostData#snapshot().
  DEFAULT_AXIS_NAMES: DEFAULT_AXIS_NAMES.
  _preduce: _preduce().
  reduce_across_hosts.pre_jit: reduce_across_hosts().pre_jit().
  reduce_across_hosts.post_jit: reduce_across_hosts().post_jit().
  _local_pytrees_to_global.pad_to_global: _local_pytrees_to_global().pad_to_global().
---
# Module: [`simply/utils/sharding.py`](../../../../../../raw/code/simply/simply/utils/sharding.py)

## Classes
### `MultihostData`
- def: [`simply/utils/sharding.py:518`](../../../../../../raw/code/simply/simply/utils/sharding.py#L518)
- doc: Multihost data.
- signature: `class MultihostData:`
- members:
  - `load(cls, load_dir: epath.PathLike)` — [`L603`](../../../../../../raw/code/simply/simply/utils/sharding.py#L603) — Loads multi-host data from local_dir.
  - `load_async(cls, load_dir: epath.PathLike)` — [`L569`](../../../../../../raw/code/simply/simply/utils/sharding.py#L569) — Loads multi-host data from local_dir.
  - `save(self, save_dir: epath.PathLike)` — [`L529`](../../../../../../raw/code/simply/simply/utils/sharding.py#L529) — Saves multi-host data.
  - `snapshot(self, snapshot_dir: epath.PathLike)` — [`L556`](../../../../../../raw/code/simply/simply/utils/sharding.py#L556) — Snapshots multi-host data.
  - `global_data` — [`L526`](../../../../../../raw/code/simply/simply/utils/sharding.py#L526)
  - `local_data` — [`L527`](../../../../../../raw/code/simply/simply/utils/sharding.py#L527)
- uses (calls/refs, reference-scoped): [`PyTree`](common.md#PyTree.PyTree), [`concatenate_pytrees`](pytree.md#concatenate_pytrees), [`load_pytree_from`](pytree.md#load_pytree_from), [`save_pytree_to`](pytree.md#save_pytree_to), [`multihost_sharded`](sharding.md#multihost_sharded)
- used by: [`test_multihost_data`](sharding_test.md#ShardingTest.test_multihost_data)

## Functions
- `_inner_partition_with_minimum_redundancy(shape: tuple[int, ...], mesh_axis_sizes: tuple[int, ...], cache: MutableMapping[tuple[tuple[int, ...], tuple[int, ...]], Sequence[Sequence[int]]])` — [`L402`](../../../../../../raw/code/simply/simply/utils/sharding.py#L402) — Fits partition to a shape.
- `_local_pytrees_to_global(abstract_pytree: common.PyTree, local_pytrees: Sequence[common.PyTree], num_per_process: np.ndarray, global_batch_size: int)` — [`L312`](../../../../../../raw/code/simply/simply/utils/sharding.py#L312) — See pytree_ragged_stack_allgather.
- `_preduce(x: jax.Array, reduce_op: Callable[..., jax.Array])` — [`L268`](../../../../../../raw/code/simply/simply/utils/sharding.py#L268) — documented in [simply-utils-sharding](../../../concepts/simply-utils-sharding.md)
- `batch_partition_with_minimum_redundancy(shapes: Sequence[Sequence[int]], mesh_axis_names: Sequence[str], mesh_axis_sizes: Sequence[int])` — [`L456`](../../../../../../raw/code/simply/simply/utils/sharding.py#L456) — Finds partitions for a batch of shapes with minimum redundancy. — documented in [simply-utils-checkpoint_lib](../../../concepts/simply-utils-checkpoint_lib.md)
- `create_mesh(mesh_shape: Sequence[int] | Mapping[str, int] | None = None, dcn_mesh_shape: Sequence[int] | Mapping[str, int] | None = None, axis_names: Sequence[str] | None = None)` — [`L98`](../../../../../../raw/code/simply/simply/utils/sharding.py#L98) — Creates mesh for the current device set. — documented in [simply-utils-sharding](../../../concepts/simply-utils-sharding.md)
- `get_array_sharding(array: jax.Array)` — [`L187`](../../../../../../raw/code/simply/simply/utils/sharding.py#L187) — Returns the sharding of the array.
- `get_default_mesh()` — [`L154`](../../../../../../raw/code/simply/simply/utils/sharding.py#L154) — Returns the default mesh for the current device set.
- `get_partition_axis(partition: PartitionAnnotation, axis: int)` — [`L198`](../../../../../../raw/code/simply/simply/utils/sharding.py#L198)
- `get_partition_size(partition: str | Sequence[str] | None)` — [`L206`](../../../../../../raw/code/simply/simply/utils/sharding.py#L206)
- `max_across_hosts(in_tree: common.PyTree)` — [`L307`](../../../../../../raw/code/simply/simply/utils/sharding.py#L307) — Sums data across all hosts. — documented in [simply-utils-sharding](../../../concepts/simply-utils-sharding.md)
- `mesh_context(mesh_shape: Sequence[int], dcn_mesh_shape: Sequence[int] | None = None, axis_names: Sequence[str] | None = None)` — [`L45`](../../../../../../raw/code/simply/simply/utils/sharding.py#L45)
- `mesh_sharding(pspec: common.PartitionAnnotation = None, mesh: js.Mesh | None = None)` — [`L164`](../../../../../../raw/code/simply/simply/utils/sharding.py#L164)
- `multihost_sharded(batch: Sequence[Any], process_index: int = -1, process_count: int = 0)` — [`L386`](../../../../../../raw/code/simply/simply/utils/sharding.py#L386) — Shards a batch across multiple hosts.
- `named_sharding(pspec: common.PartitionAnnotation = None)` — [`L173`](../../../../../../raw/code/simply/simply/utils/sharding.py#L173)
- `pad_to_global(x)` — [`L337`](../../../../../../raw/code/simply/simply/utils/sharding.py#L337)
- `partition_spec(partition: PartitionAnnotation | js.PartitionSpec)` — [`L218`](../../../../../../raw/code/simply/simply/utils/sharding.py#L218) — documented in [simply-utils-sharding](../../../concepts/simply-utils-sharding.md)
- `partition_with_minimum_redundancy(shape: Sequence[int], mesh_axis_names: Sequence[str], mesh_axis_sizes: Sequence[int])` — [`L507`](../../../../../../raw/code/simply/simply/utils/sharding.py#L507)
- `post_jit(x)` — [`L291`](../../../../../../raw/code/simply/simply/utils/sharding.py#L291)
- `pre_jit(x)` — [`L285`](../../../../../../raw/code/simply/simply/utils/sharding.py#L285)
- `pytree_ragged_stack_allgather(abstract_pytree: common.PyTree, local_pytrees: Sequence[common.PyTree], num_per_process: np.ndarray, global_batch_size: int)` — [`L349`](../../../../../../raw/code/simply/simply/utils/sharding.py#L349) — Combines pytrees local to each process into a global one by stacking. — documented in [simply-utils-sharding](../../../concepts/simply-utils-sharding.md)
- `reduce_across_hosts(in_tree: common.PyTree, reduce_op: Callable[..., jax.Array])` — [`L272`](../../../../../../raw/code/simply/simply/utils/sharding.py#L272) — Reduces data across all hosts. — documented in [simply-utils-sharding](../../../concepts/simply-utils-sharding.md)
- `set_default_mesh_shape(mesh_shape: Sequence[int] | Mapping[str, int], dcn_mesh_shape: Sequence[int] | Mapping[str, int] | None = None, axis_names: Sequence[str] | None = None)` — [`L58`](../../../../../../raw/code/simply/simply/utils/sharding.py#L58) — Sets the default mesh shape for the current thread context.
- `set_mesh(mesh_shape: Sequence[int] | Mapping[str, int], dcn_mesh_shape: Sequence[int] | Mapping[str, int] | None = None, axis_names: Sequence[str] | None = None)` — [`L84`](../../../../../../raw/code/simply/simply/utils/sharding.py#L84) — documented in [simply-utils-sharding](../../../concepts/simply-utils-sharding.md)
- `sum_across_hosts(in_tree: common.PyTree)` — [`L302`](../../../../../../raw/code/simply/simply/utils/sharding.py#L302) — Sums data across all hosts. — documented in [simply-serving-page_batcher](../../../concepts/simply-serving-page_batcher.md)
- `with_sharding_constraint(x: jax.Array, partition: js.Sharding | js.PartitionSpec | PartitionAnnotation)` — [`L230`](../../../../../../raw/code/simply/simply/utils/sharding.py#L230) — An extension of jax.lax.with_sharding_constraint. — documented in [simply-utils-module](../../../concepts/simply-utils-module.md)

## Module values
- `DEFAULT_AXIS_NAMES` — [`L41`](../../../../../../raw/code/simply/simply/utils/sharding.py#L41) — documented in [simply-utils-sharding](../../../concepts/simply-utils-sharding.md)
- `NOT_ANNOTATED` — [`L39`](../../../../../../raw/code/simply/simply/utils/sharding.py#L39) — documented in [simply-utils-sharding](../../../concepts/simply-utils-sharding.md)
- `PartitionAnnotation` — [`L38`](../../../../../../raw/code/simply/simply/utils/sharding.py#L38)

