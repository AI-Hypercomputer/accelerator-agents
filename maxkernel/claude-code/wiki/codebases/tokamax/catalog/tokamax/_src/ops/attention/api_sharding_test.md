---
title: 'Module: tokamax/_src/ops/attention/api_sharding_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/api_sharding_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.api_sharding_test`/
symbols:
  ApiShardingTest.test_dot_product_attention_sharding: ApiShardingTest#test_dot_product_attention_sharding().
  Mesh: Mesh.
  NamedSharding: NamedSharding.
  PartitionSpec: PartitionSpec.
  ApiShardingTest.f: ApiShardingTest#f().
  ApiShardingTest: ApiShardingTest#
---
# Module: [`tokamax/_src/ops/attention/api_sharding_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_sharding_test.py)

## Classes
### `ApiShardingTest`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/attention/api_sharding_test.py:31`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_sharding_test.py#L31)
- signature: `class ApiShardingTest(parameterized.TestCase):`
- members:
  - `f(q, k, v)` — [`L80`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_sharding_test.py#L80)
  - `test_dot_product_attention_sharding(self, implementation, shard_axis)` — [`L37`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_sharding_test.py#L37)
- uses (calls/refs, reference-scoped): [`dot_product_attention`](api.md#dot_product_attention), [`IMPLEMENTATIONS`](api.md#IMPLEMENTATIONS.IMPLEMENTATIONS), [`Mesh`](api_sharding_test.md#Mesh), [`NamedSharding`](api_sharding_test.md#NamedSharding), [`PartitionSpec`](api_sharding_test.md#PartitionSpec)

## Module values
- `Mesh` — [`L26`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_sharding_test.py#L26)
- `NamedSharding` — [`L27`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_sharding_test.py#L27)
- `PartitionSpec` — [`L28`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_sharding_test.py#L28)

