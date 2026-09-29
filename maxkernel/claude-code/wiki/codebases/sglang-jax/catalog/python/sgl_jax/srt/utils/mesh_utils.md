---
title: 'Module: python/sgl_jax/srt/utils/mesh_utils.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/utils/mesh_utils.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.utils.mesh_utils`/
symbols:
  create_device_mesh: create_device_mesh().
  fill_unspecified_parallelism: fill_unspecified_parallelism().
  default_mesh_axes: default_mesh_axes.
---
# Module: [`python/sgl_jax/srt/utils/mesh_utils.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/mesh_utils.py)

## Functions
- `create_device_mesh(ici_parallelism: MutableSequence[int], dcn_parallelism: MutableSequence[int], devices=None, device_indexes: list[int] | None = None, num_slices: int = 1, allow_split_physical_axes: bool = True, use_explicit_sharding: bool = True, mesh_axes: Sequence[str] = default_mesh_axes)` — [`L15`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/mesh_utils.py#L15) — Create a device mesh
- `fill_unspecified_parallelism(parallelism: MutableSequence[int], num_devices: int)` — [`L68`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/mesh_utils.py#L68)

## Module values
- `default_mesh_axes` — [`L9`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/mesh_utils.py#L9)

