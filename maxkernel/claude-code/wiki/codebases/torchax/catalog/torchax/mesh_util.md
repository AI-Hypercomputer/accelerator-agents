---
title: 'Module: torchax/mesh_util.py'
type: catalog
provenance: extracted
module: torchax/mesh_util.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `torchax.mesh_util`/
symbols:
  Mesh.initialize_model_sharded: Mesh#initialize_model_sharded().
  SingleAxisSharder.__call__: SingleAxisSharder#__call__().
  Mesh.jax_mesh: Mesh#jax_mesh.
  SingleAxisSharder: SingleAxisSharder#
  Mesh.shard_model: Mesh#shard_model().
  Mesh.fsdp_mesh: Mesh#fsdp_mesh().
  Mesh.model_initializer: Mesh#model_initializer().
  SingleAxisSharder.axis_size: SingleAxisSharder#axis_size.
  Mesh: Mesh#
  Mesh._sharder: Mesh#_sharder.
  _shard_first_multiple_of: _shard_first_multiple_of().
  SingleAxisSharder.axis_name: SingleAxisSharder#axis_name.
  SingleAxisSharder.replicate_unshardable: SingleAxisSharder#replicate_unshardable.
  SingleAxisSharder.__init__: SingleAxisSharder#__init__().
  Mesh.__init__: Mesh#__init__().
---
# Module: [`torchax/mesh_util.py`](../../../../../raw/code/torchax/torchax/mesh_util.py)

## Classes
### `Mesh`
- def: [`torchax/mesh_util.py:109`](../../../../../raw/code/torchax/torchax/mesh_util.py#L109)
- doc: A helper class that wraps `jax.sharding.Mesh` object.
- signature: `class Mesh:`
- members:
  - `__init__(self, jax_mesh, sharder=None)` — [`L146`](../../../../../raw/code/torchax/torchax/mesh_util.py#L146) — Initializes the Mesh helper.
  - `fsdp_mesh(cls, axis_name="fsdp")` — [`L125`](../../../../../raw/code/torchax/torchax/mesh_util.py#L125) — Creates a Mesh instance suitable for 1D FSDP-style sharding.
  - `initialize_model_sharded(self, model_class, init_args, init_kwargs=None, override_sharder=None)` — [`L169`](../../../../../raw/code/torchax/torchax/mesh_util.py#L169) — Initializes a PyTorch model with its parameters sharded across the mesh.
  - `model_initializer()` — [`L216`](../../../../../raw/code/torchax/torchax/mesh_util.py#L216)
  - `shard_model(self, model, override_sharder=None)` — [`L229`](../../../../../raw/code/torchax/torchax/mesh_util.py#L229)
  - `jax_mesh` — [`L161`](../../../../../raw/code/torchax/torchax/mesh_util.py#L161)
- protocol/private: `_sharder`[`L167`](../../../../../raw/code/torchax/torchax/mesh_util.py#L167)
- uses (calls/refs, reference-scoped): [`default_env`](__init__.md#default_env), [`jax_jit`](interop.md#jax_jit), [`disable_temporarily`](__init__.md#disable_temporarily), [`SingleAxisSharder`](mesh_util.md#SingleAxisSharder)
- used by: (1 test-only callers)

### `SingleAxisSharder`
- def: [`torchax/mesh_util.py:55`](../../../../../raw/code/torchax/torchax/mesh_util.py#L55)
- doc: A callable object that generates PartitionSpecs for single-axis sharding.
- signature: `class SingleAxisSharder:`
- members:
  - `__call__(self, name, shapedtype)` — [`L81`](../../../../../raw/code/torchax/torchax/mesh_util.py#L81) — Generates a PartitionSpec for a given tensor name and shaped type.
  - `__init__(self, axis_name, axis_size, replicate_unshardable=False)` — [`L68`](../../../../../raw/code/torchax/torchax/mesh_util.py#L68) — Initializes the SingleAxisSharder.
  - `axis_name` — [`L77`](../../../../../raw/code/torchax/torchax/mesh_util.py#L77)
  - `axis_size` — [`L78`](../../../../../raw/code/torchax/torchax/mesh_util.py#L78)
  - `replicate_unshardable` — [`L79`](../../../../../raw/code/torchax/torchax/mesh_util.py#L79)
- uses (calls/refs, reference-scoped): [`_shard_first_multiple_of`](mesh_util.md#_shard_first_multiple_of)
- used by: [`jax_mesh`](mesh_util.md#Mesh.jax_mesh), [`fsdp_mesh`](mesh_util.md#Mesh.fsdp_mesh)  (2 test-only)

## Functions
- `_shard_first_multiple_of(axis_name, shape, multiple_of)` — [`L23`](../../../../../raw/code/torchax/torchax/mesh_util.py#L23) — Creates a PartitionSpec to shard the first dimension divisible by a number.

