---
title: 'Module: simply/utils/sharding_test.py'
type: catalog
provenance: extracted
module: simply/utils/sharding_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.sharding_test`/ShardingTest#
symbols:
  ShardingTest.test_multihost_data: test_multihost_data().
  ShardingTest._f: _f().
  ShardingTest.test_mesh_context: test_mesh_context().
  ShardingTest.test_sharding: test_sharding().
  ShardingTest.test_get_array_sharding: test_get_array_sharding().
  ShardingTest.test_pytree_ragged_stack_allgather: test_pytree_ragged_stack_allgather().
  ShardingTest.test_multihost_sharded: test_multihost_sharded().
  ShardingTest.test_partition_with_minimum_redundancy: test_partition_with_minimum_redundancy().
  ShardingTest.test_partition_spec: test_partition_spec().
  ShardingTest: ''
---
# Module: [`simply/utils/sharding_test.py`](../../../../../../raw/code/simply/simply/utils/sharding_test.py)

## Classes
### `ShardingTest`  ·  implements/extends TestCase
- def: [`simply/utils/sharding_test.py:25`](../../../../../../raw/code/simply/simply/utils/sharding_test.py#L25)
- signature: `class ShardingTest(absltest.TestCase):`
- members:
  - `test_get_array_sharding(self)` — [`L170`](../../../../../../raw/code/simply/simply/utils/sharding_test.py#L170)
  - `test_mesh_context(self)` — [`L27`](../../../../../../raw/code/simply/simply/utils/sharding_test.py#L27)
  - `test_multihost_data(self)` — [`L139`](../../../../../../raw/code/simply/simply/utils/sharding_test.py#L139)
  - `test_multihost_sharded(self)` — [`L99`](../../../../../../raw/code/simply/simply/utils/sharding_test.py#L99)
  - `test_partition_spec(self)` — [`L161`](../../../../../../raw/code/simply/simply/utils/sharding_test.py#L161)
  - `test_partition_with_minimum_redundancy(self)` — [`L119`](../../../../../../raw/code/simply/simply/utils/sharding_test.py#L119)
  - `test_pytree_ragged_stack_allgather(self)` — [`L52`](../../../../../../raw/code/simply/simply/utils/sharding_test.py#L52)
  - `test_sharding(self)` — [`L33`](../../../../../../raw/code/simply/simply/utils/sharding_test.py#L33)
- protocol/private: `_f`[`L182`](../../../../../../raw/code/simply/simply/utils/sharding_test.py#L182)
- uses (calls/refs, reference-scoped): [`with_sharding_constraint`](sharding.md#with_sharding_constraint), [`partition_spec`](sharding.md#partition_spec), [`mesh_sharding`](sharding.md#mesh_sharding), [`save`](sharding.md#MultihostData.save), [`_local_pytrees_to_global`](sharding.md#_local_pytrees_to_global), [`multihost_sharded`](sharding.md#multihost_sharded), [`partition_with_minimum_redundancy`](sharding.md#partition_with_minimum_redundancy), [`get_array_sharding`](sharding.md#get_array_sharding), [`global_data`](sharding.md#MultihostData.global_data), [`local_data`](sharding.md#MultihostData.local_data), [`load`](sharding.md#MultihostData.load), [`get_default_mesh`](sharding.md#get_default_mesh), [`MultihostData`](sharding.md#MultihostData), [`mesh_context`](sharding.md#mesh_context), [`snapshot`](sharding.md#MultihostData.snapshot)

