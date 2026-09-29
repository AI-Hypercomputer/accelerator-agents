---
title: 'Module: jax/experimental/pallas/ops/tpu/all_gather.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/tpu/all_gather.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.tpu.all_gather`/
symbols:
  ag_kernel: ag_kernel().
  all_gather.ag_local: all_gather().ag_local().
  all_gather: all_gather().
  get_neighbor: get_neighbor().
  P: P.
---
# Module: [`jax/experimental/pallas/ops/tpu/all_gather.py`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/all_gather.py)

## Functions
- `ag_kernel(x_ref, o_ref, send_sem, recv_sem, *, axis_name: str, mesh: jax.sharding.Mesh)` — [`L61`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/all_gather.py#L61)
- `ag_local(x_shard)` — [`L133`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/all_gather.py#L133)
- `all_gather(x, *, mesh: jax.sharding.Mesh, axis_name: str | Sequence[str], memory_space: pltpu.MemorySpace = pltpu.VMEM)` — [`L122`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/all_gather.py#L122)
- `get_neighbor(idx: jax.Array, mesh: jax.sharding.Mesh, axis_name: str, *, direction: str)` — [`L41`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/all_gather.py#L41) — Helper function that computes the mesh indices of a neighbor.

## Module values
- `P` — [`L38`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/all_gather.py#L38)

