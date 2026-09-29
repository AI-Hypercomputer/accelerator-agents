---
title: 'Module: graphcast/grid_mesh_connectivity.py'
type: catalog
provenance: extracted
module: graphcast/grid_mesh_connectivity.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.grid_mesh_connectivity`/
symbols:
  in_mesh_triangle_indices: in_mesh_triangle_indices().
  radius_query_indices: radius_query_indices().
  _grid_lat_lon_to_coordinates: _grid_lat_lon_to_coordinates().
---
# Module: [`graphcast/grid_mesh_connectivity.py`](../../../../../raw/code/graphcast/graphcast/grid_mesh_connectivity.py)

## Functions
- `_grid_lat_lon_to_coordinates(grid_latitude: np.ndarray, grid_longitude: np.ndarray)` — [`L22`](../../../../../raw/code/graphcast/graphcast/grid_mesh_connectivity.py#L22) — Lat [num_lat] lon [num_lon] to 3d coordinates [num_lat, num_lon, 3].
- `in_mesh_triangle_indices(*, grid_latitude: np.ndarray, grid_longitude: np.ndarray, mesh: icosahedral_mesh.TriangularMesh)` — [`L89`](../../../../../raw/code/graphcast/graphcast/grid_mesh_connectivity.py#L89) — Returns mesh-grid edge indices for grid points contained in mesh triangles. — documented in [graphcast](../../concepts/graphcast.md)
- `radius_query_indices(*, grid_latitude: np.ndarray, grid_longitude: np.ndarray, mesh: icosahedral_mesh.TriangularMesh, radius: float)` — [`L40`](../../../../../raw/code/graphcast/graphcast/grid_mesh_connectivity.py#L40) — Returns mesh-grid edge indices for radius query.

