---
title: 'Module: graphcast/icosahedral_mesh.py'
type: catalog
provenance: extracted
module: graphcast/icosahedral_mesh.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.icosahedral_mesh`/
symbols:
  TriangularMesh.vertices: TriangularMesh#vertices.
  _two_split_unit_sphere_triangle_faces: _two_split_unit_sphere_triangle_faces().
  get_hierarchy_of_triangular_meshes_for_sphere: get_hierarchy_of_triangular_meshes_for_sphere().
  TriangularMesh: TriangularMesh#
  TriangularMesh.faces: TriangularMesh#faces.
  _ChildVerticesBuilder._create_child_vertex: _ChildVerticesBuilder#_create_child_vertex().
  _ChildVerticesBuilder.get_new_child_vertex_index: _ChildVerticesBuilder#get_new_child_vertex_index().
  merge_meshes: merge_meshes().
  get_icosahedron: get_icosahedron().
  faces_to_edges: faces_to_edges().
  get_last_triangular_mesh_for_sphere: get_last_triangular_mesh_for_sphere().
  _ChildVerticesBuilder.get_all_vertices: _ChildVerticesBuilder#get_all_vertices().
  _ChildVerticesBuilder._child_vertices_index_mapping: _ChildVerticesBuilder#_child_vertices_index_mapping.
  _ChildVerticesBuilder._all_vertices_list: _ChildVerticesBuilder#_all_vertices_list.
  _ChildVerticesBuilder._get_child_vertex_key: _ChildVerticesBuilder#_get_child_vertex_key().
  _ChildVerticesBuilder._parent_vertices: _ChildVerticesBuilder#_parent_vertices.
  _ChildVerticesBuilder: _ChildVerticesBuilder#
  _ChildVerticesBuilder.__init__: _ChildVerticesBuilder#__init__().
---
# Module: [`graphcast/icosahedral_mesh.py`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh.py)

## Classes
### `TriangularMesh`  ·  implements/extends NamedTuple
- def: [`graphcast/icosahedral_mesh.py:23`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh.py#L23)
- doc: Data structure for triangular meshes.
- signature: `class TriangularMesh(NamedTuple):`
- members:
  - `faces` — [`L34`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh.py#L34) — documented in [graphcast](../../concepts/graphcast.md)
  - `vertices` — [`L33`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh.py#L33) — documented in [graphcast](../../concepts/graphcast.md)
- used by: [`_init_mesh_graph`](graphcast.md#GraphCast._init_mesh_graph), [`_init_mesh_graph`](denoiser.md#_DenoiserArchitecture._init_mesh_graph), [`_init_mesh_properties`](denoiser.md#_DenoiserArchitecture._init_mesh_properties), [`_init_mesh_properties`](graphcast.md#GraphCast._init_mesh_properties), [`_two_split_unit_sphere_triangle_faces`](icosahedral_mesh.md#_two_split_unit_sphere_triangle_faces), [`get_hierarchy_of_triangular_meshes_for_sphere`](icosahedral_mesh.md#get_hierarchy_of_triangular_meshes_for_sphere), [`in_mesh_triangle_indices`](grid_mesh_connectivity.md#in_mesh_triangle_indices), [`_permute_mesh_to_banded`](denoiser.md#_permute_mesh_to_banded), [`radius_query_indices`](grid_mesh_connectivity.md#radius_query_indices), [`test_get_hierarchy_of_triangular_meshes_for_sphere`](icosahedral_mesh_test.md#IcosahedralMeshTest.test_get_hierarchy_of_triangular_meshes_for_sphere), [`test_merge_meshes`](icosahedral_mesh_test.md#IcosahedralMeshTest.test_merge_meshes), [`get_icosahedron`](icosahedral_mesh.md#get_icosahedron), [`merge_meshes`](icosahedral_mesh.md#merge_meshes), [`get_last_triangular_mesh_for_sphere`](icosahedral_mesh.md#get_last_triangular_mesh_for_sphere)

### `_ChildVerticesBuilder`
- def: [`graphcast/icosahedral_mesh.py:214`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh.py#L214)
- members:
  - `_create_child_vertex(self, parent_vertex_indices)` — [`L231`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh.py#L231) — Creates a new vertex.
  - `get_all_vertices(self)` — [`L254`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh.py#L254) — Returns an array with old vertices.
  - `get_new_child_vertex_index(self, parent_vertex_indices)` — [`L246`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh.py#L246) — Returns index for a child vertex, creating it if necessary.
- protocol/private: `__init__`[`L217`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh.py#L217), `_all_vertices_list`[`L226`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh.py#L226), `_child_vertices_index_mapping`[`L223`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh.py#L223), `_get_child_vertex_key`[`L228`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh.py#L228), `_parent_vertices`[`L224`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh.py#L224)
- used by: [`_two_split_unit_sphere_triangle_faces`](icosahedral_mesh.md#_two_split_unit_sphere_triangle_faces)

## Functions
- `_two_split_unit_sphere_triangle_faces(triangular_mesh: TriangularMesh)` — [`L173`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh.py#L173) — Splits each triangular face into 4 triangles keeping the orientation.
- `faces_to_edges(faces: np.ndarray)` — [`L259`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh.py#L259) — Transforms polygonal faces to sender and receiver indices.
- `get_hierarchy_of_triangular_meshes_for_sphere(splits: int)` — [`L59`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh.py#L59) — Returns a sequence of meshes, each with triangularization sphere.
- `get_icosahedron()` — [`L91`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh.py#L91) — Returns a regular icosahedral mesh with circumscribed unit sphere.
- `get_last_triangular_mesh_for_sphere(splits: int)` — [`L284`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh.py#L284)
- `merge_meshes(mesh_list: Sequence[TriangularMesh])` — [`L37`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh.py#L37) — Merges all meshes into one. Assumes the last mesh is the finest.

