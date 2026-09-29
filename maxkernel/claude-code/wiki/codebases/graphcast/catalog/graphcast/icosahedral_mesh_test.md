---
title: 'Module: graphcast/icosahedral_mesh_test.py'
type: catalog
provenance: extracted
module: graphcast/icosahedral_mesh_test.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.icosahedral_mesh_test`/
symbols:
  IcosahedralMeshTest.test_get_hierarchy_of_triangular_meshes_for_sphere: IcosahedralMeshTest#test_get_hierarchy_of_triangular_meshes_for_sphere().
  IcosahedralMeshTest.test_merge_meshes: IcosahedralMeshTest#test_merge_meshes().
  IcosahedralMeshTest.test_icosahedron: IcosahedralMeshTest#test_icosahedron().
  _assert_valid_mesh: _assert_valid_mesh().
  IcosahedralMeshTest.test_faces_to_edges: IcosahedralMeshTest#test_faces_to_edges().
  _get_mesh_spec: _get_mesh_spec().
  _assert_positive_face_orientation: _assert_positive_face_orientation().
  IcosahedralMeshTest: IcosahedralMeshTest#
---
# Module: [`graphcast/icosahedral_mesh_test.py`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh_test.py)

## Classes
### `IcosahedralMeshTest`  ·  implements/extends TestCase
- def: [`graphcast/icosahedral_mesh_test.py:35`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh_test.py#L35)
- signature: `class IcosahedralMeshTest(parameterized.TestCase):`
- members:
  - `test_faces_to_edges(self)` — [`L73`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh_test.py#L73)
  - `test_get_hierarchy_of_triangular_meshes_for_sphere(self, splits)` — [`L43`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh_test.py#L43)
  - `test_icosahedron(self)` — [`L37`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh_test.py#L37)
  - `test_merge_meshes(self, splits)` — [`L63`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh_test.py#L63)
- uses (calls/refs, reference-scoped): [`vertices`](icosahedral_mesh.md#TriangularMesh.vertices), [`get_hierarchy_of_triangular_meshes_for_sphere`](icosahedral_mesh.md#get_hierarchy_of_triangular_meshes_for_sphere), [`faces`](icosahedral_mesh.md#TriangularMesh.faces), [`get_icosahedron`](icosahedral_mesh.md#get_icosahedron), [`merge_meshes`](icosahedral_mesh.md#merge_meshes), [`faces_to_edges`](icosahedral_mesh.md#faces_to_edges), [`_assert_valid_mesh`](icosahedral_mesh_test.md#_assert_valid_mesh), [`_get_mesh_spec`](icosahedral_mesh_test.md#_get_mesh_spec)

## Functions
- `_assert_positive_face_orientation(vertices, faces)` — [`L108`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh_test.py#L108)
- `_assert_valid_mesh(mesh, num_expected_vertices, num_expected_faces)` — [`L95`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh_test.py#L95)
- `_get_mesh_spec(splits: int)` — [`L23`](../../../../../raw/code/graphcast/graphcast/icosahedral_mesh_test.py#L23) — Returns size of the final icosahedral mesh resulting from the splitting.

