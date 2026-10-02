---
title: 'Module: tpu_inference/layers/common/utils.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/common/utils.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.common.utils`/
symbols:
  cpu_mesh_context: cpu_mesh_context().
  general_device_put: general_device_put().
  slice_sharded_tensor_for_concatenation: slice_sharded_tensor_for_concatenation().
  reorder_concatenated_tensor_for_sharding: reorder_concatenated_tensor_for_sharding().
  cpu_mesh: cpu_mesh().
  _cpu_mesh: _cpu_mesh.
  general_device_put._put: general_device_put()._put().
  truncate_sharded_tensor: truncate_sharded_tensor().
---
# Module: [`tpu_inference/layers/common/utils.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/utils.py)

## Functions
- `_put(t)` — [`L140`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/utils.py#L140)
- `cpu_mesh()` — [`L171`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/utils.py#L171)
- `cpu_mesh_context()` — [`L178`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/utils.py#L178) — A context to enforce using CPU mesh, used for loading weights on CPU.
- `general_device_put(tensor: jax.Array, sharding: Sharding, *, layout: Layout | None = None, source_mesh: Mesh | None = None)` — [`L128`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/utils.py#L128) — Put a tensor onto devices with the given sharding.
- `reorder_concatenated_tensor_for_sharding(concatenated_tensor: jax.Array, split_sizes: list[int], n_shards: int, dim: int)` — [`L28`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/utils.py#L28) — Reorder a replicated concatenated tensor such that when sharded on multiple chips, each shard is a concatenation of the shards of the individual tensors.
- `slice_sharded_tensor_for_concatenation(sharded_tensor: jax.Array, split_sizes: list[int], n_shards: int)` — [`L67`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/utils.py#L67) — Slice the input tensor which is sharded on multiple chips (on the last dim) into individual tensors with the same sharding.
- `truncate_sharded_tensor(sharded_tensor: jax.Array, truncate_size: int, n_shards: int)` — [`L106`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/utils.py#L106) — Truncate each shard of a sharded tensor to a specified size.

## Module values
- `_cpu_mesh` — [`L25`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/utils.py#L25)

