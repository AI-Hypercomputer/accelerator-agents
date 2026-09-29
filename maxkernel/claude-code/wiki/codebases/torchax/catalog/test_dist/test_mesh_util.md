---
title: 'Module: test_dist/test_mesh_util.py'
type: catalog
provenance: extracted
module: test_dist/test_mesh_util.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `test_dist.test_mesh_util`/
symbols:
  TestModule.a: TestModule#a.
  MeshUtilTest.setUp: MeshUtilTest#setUp().
  MeshUtilTest.test_sharder_call: MeshUtilTest#test_sharder_call().
  MeshUtilTest.test_sharder_call_no_shardable: MeshUtilTest#test_sharder_call_no_shardable().
  MeshUtilTest.test_init_module_sharded.TestModule: MeshUtilTest#test_init_module_sharded().TestModule#
  MeshUtilTest: MeshUtilTest#
  MeshUtilTest.test_init_module_sharded: MeshUtilTest#test_init_module_sharded().
  MeshUtilTest.test_init_module_sharded.TestModule.__init__: MeshUtilTest#test_init_module_sharded().TestModule#__init__().
---
# Module: [`test_dist/test_mesh_util.py`](../../../../../raw/code/torchax/test_dist/test_mesh_util.py)

## Classes
### `MeshUtilTest`  ·  implements/extends TestCase
- def: [`test_dist/test_mesh_util.py:25`](../../../../../raw/code/torchax/test_dist/test_mesh_util.py#L25)
- signature: `class MeshUtilTest(unittest.TestCase):`
- members:
  - `setUp(self)` — [`L26`](../../../../../raw/code/torchax/test_dist/test_mesh_util.py#L26)
  - `test_init_module_sharded(self)` — [`L29`](../../../../../raw/code/torchax/test_dist/test_mesh_util.py#L29)
  - `test_sharder_call(self)` — [`L41`](../../../../../raw/code/torchax/test_dist/test_mesh_util.py#L41) — Test the __call__ method produces the correct PartitionSpec.
  - `test_sharder_call_no_shardable(self)` — [`L50`](../../../../../raw/code/torchax/test_dist/test_mesh_util.py#L50) — Test __call__ when no dimension is shardable.
- uses (calls/refs, reference-scoped): [`enable_globally`](../torchax/__init__.md#enable_globally), [`SingleAxisSharder`](../torchax/mesh_util.md#SingleAxisSharder)

### `TestModule`  ·  implements/extends Module
- def: [`test_dist/test_mesh_util.py:30`](../../../../../raw/code/torchax/test_dist/test_mesh_util.py#L30)
- signature: `class TestModule(torch.nn.Module):`
- members:
  - `a` — [`L33`](../../../../../raw/code/torchax/test_dist/test_mesh_util.py#L33)
- protocol/private: `__init__`[`L31`](../../../../../raw/code/torchax/test_dist/test_mesh_util.py#L31)
- uses (calls/refs, reference-scoped): [`initialize_model_sharded`](../torchax/mesh_util.md#Mesh.initialize_model_sharded), [`fsdp_mesh`](../torchax/mesh_util.md#Mesh.fsdp_mesh), [`Mesh`](../torchax/mesh_util.md#Mesh), `bias`

