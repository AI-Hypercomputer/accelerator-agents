---
title: 'Module: easydel/layers/_sharding.py'
type: catalog
provenance: extracted
module: easydel/layers/_sharding.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.layers._sharding`/
symbols:
  resolve_safe_sharding: resolve_safe_sharding().
  _sanitize_spec_for_shape: _sanitize_spec_for_shape().
  _is_valid_mesh: _is_valid_mesh().
  _mesh_partition_product: _mesh_partition_product().
  pick_mesh: pick_mesh().
  _mesh_axis_size: _mesh_axis_size().
  _resolve_partition_spec: _resolve_partition_spec().
  _coerce_partition_spec: _coerce_partition_spec().
---
# Module: [`easydel/layers/_sharding.py`](../../../../../../raw/code/EasyDeL/easydel/layers/_sharding.py)

## Functions
- `_coerce_partition_spec(spec: tp.Any)` — [`L84`](../../../../../../raw/code/EasyDeL/easydel/layers/_sharding.py#L84)
- `_is_valid_mesh(mesh: tp.Any)` — [`L25`](../../../../../../raw/code/EasyDeL/easydel/layers/_sharding.py#L25)
- `_mesh_axis_size(mesh: tp.Any, axis_name: str)` — [`L33`](../../../../../../raw/code/EasyDeL/easydel/layers/_sharding.py#L33)
- `_mesh_partition_product(mesh: tp.Any, axis_spec: tp.Any)` — [`L47`](../../../../../../raw/code/EasyDeL/easydel/layers/_sharding.py#L47)
- `_resolve_partition_spec(*, partition_manager: tp.Any, axes: tp.Any, shape: tuple[int, ...], mode: str = common_types.MODE_TRAIN)` — [`L60`](../../../../../../raw/code/EasyDeL/easydel/layers/_sharding.py#L60)
- `_sanitize_spec_for_shape(*, spec: tp.Any, shape: tuple[int, ...], mesh: tp.Any)` — [`L95`](../../../../../../raw/code/EasyDeL/easydel/layers/_sharding.py#L95)
- `pick_mesh(*, partition_manager: tp.Any | None = None, mesh: tp.Any | None = None)` — [`L114`](../../../../../../raw/code/EasyDeL/easydel/layers/_sharding.py#L114)
- `resolve_safe_sharding(*, axes: tp.Any, shape: tuple[int, ...], partition_manager: tp.Any | None = None, mesh: tp.Any | None = None, mode: str = common_types.MODE_TRAIN)` — [`L145`](../../../../../../raw/code/EasyDeL/easydel/layers/_sharding.py#L145) — Resolve sharding axes and drop non-divisible mesh axes to EMPTY/None.

