---
title: 'Module: graphcast/grid_mesh_connectivity_test.py'
type: catalog
provenance: extracted
module: graphcast/grid_mesh_connectivity_test.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.grid_mesh_connectivity_test`/GridMeshConnectivityTest#
symbols:
  GridMeshConnectivityTest.test_radius_query_indices_smoke: test_radius_query_indices_smoke().
  GridMeshConnectivityTest.test_in_mesh_triangle_indices_smoke: test_in_mesh_triangle_indices_smoke().
  GridMeshConnectivityTest.test_grid_lat_lon_to_coordinates: test_grid_lat_lon_to_coordinates().
  GridMeshConnectivityTest: ''
---
# Module: [`graphcast/grid_mesh_connectivity_test.py`](../../../../../raw/code/graphcast/graphcast/grid_mesh_connectivity_test.py)

## Classes
### `GridMeshConnectivityTest`  ·  implements/extends TestCase
- def: [`graphcast/grid_mesh_connectivity_test.py:22`](../../../../../raw/code/graphcast/graphcast/grid_mesh_connectivity_test.py#L22)
- signature: `class GridMeshConnectivityTest(absltest.TestCase):`
- members:
  - `test_grid_lat_lon_to_coordinates(self)` — [`L24`](../../../../../raw/code/graphcast/graphcast/grid_mesh_connectivity_test.py#L24)
  - `test_in_mesh_triangle_indices_smoke(self)` — [`L61`](../../../../../raw/code/graphcast/graphcast/grid_mesh_connectivity_test.py#L61)
  - `test_radius_query_indices_smoke(self)` — [`L50`](../../../../../raw/code/graphcast/graphcast/grid_mesh_connectivity_test.py#L50)
- uses (calls/refs, reference-scoped): [`get_hierarchy_of_triangular_meshes_for_sphere`](icosahedral_mesh.md#get_hierarchy_of_triangular_meshes_for_sphere), [`in_mesh_triangle_indices`](grid_mesh_connectivity.md#in_mesh_triangle_indices), [`radius_query_indices`](grid_mesh_connectivity.md#radius_query_indices), [`_grid_lat_lon_to_coordinates`](grid_mesh_connectivity.md#_grid_lat_lon_to_coordinates)

