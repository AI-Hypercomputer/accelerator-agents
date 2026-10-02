---
title: 'Module: src/alphafold3/jax/geometry/rigid_matrix_vector.py'
type: catalog
provenance: extracted
module: src/alphafold3/jax/geometry/rigid_matrix_vector.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.jax.geometry.rigid_matrix_vector`/
symbols:
  Rigid3Array.from_point_alignment: Rigid3Array#from_point_alignment().
  Rigid3Array.inverse: Rigid3Array#inverse().
  Rigid3Array.rotation: Rigid3Array#rotation.
  Rigid3Array.translation: Rigid3Array#translation.
  _compute_covariance_matrix: _compute_covariance_matrix().
  Rigid3Array.apply_to_point: Rigid3Array#apply_to_point().
  Rigid3Array.__matmul__: Rigid3Array#__matmul__().
  Rigid3Array.apply_inverse_to_point: Rigid3Array#apply_inverse_to_point().
  Rigid3Array.compose_rotation: Rigid3Array#compose_rotation().
  Rigid3Array.identity: Rigid3Array#identity().
  Rigid3Array.scale_translation: Rigid3Array#scale_translation().
  Rigid3Array.to_array: Rigid3Array#to_array().
  Rigid3Array.from_array: Rigid3Array#from_array().
  Rigid3Array: Rigid3Array#
  Rigid3Array.__getstate__: Rigid3Array#__getstate__().
  Rigid3Array.from_array4x4: Rigid3Array#from_array4x4().
  Rigid3Array.compute_center: Rigid3Array#compute_center().
  Float.Float: Float.Float.
  VERSION.VERSION: VERSION.VERSION.
  Rigid3Array.__setstate__: Rigid3Array#__setstate__().
---
# Module: [`src/alphafold3/jax/geometry/rigid_matrix_vector.py`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rigid_matrix_vector.py)

## Classes
### `Rigid3Array`
- def: [`src/alphafold3/jax/geometry/rigid_matrix_vector.py:109`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rigid_matrix_vector.py#L109) — documented in [alphafold3-jax-geometry-rigid_matrix_vector](../../../../../concepts/alphafold3-jax-geometry-rigid_matrix_vector.md)
- doc: Rigid Transformation, i.e. element of special euclidean group.
- signature: `class Rigid3Array:`
- members:
  - `apply_inverse_to_point(self, point: vector.Vec3Array)` — [`L130`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rigid_matrix_vector.py#L130) — Apply inverse Rigid3Array transform to point. — documented in [alphafold3-jax-geometry-rigid_matrix_vector](../../../../../concepts/alphafold3-jax-geometry-rigid_matrix_vector.md)
  - `apply_to_point(self, point: vector.Vec3Array)` — [`L126`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rigid_matrix_vector.py#L126) — Apply Rigid3Array transform to point. — documented in [alphafold3-jax-geometry-rigid_matrix_vector](../../../../../concepts/alphafold3-jax-geometry-rigid_matrix_vector.md)
  - `compose_rotation(self, other_rotation: rotation_matrix.Rot3Array)` — [`L135`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rigid_matrix_vector.py#L135)
  - `compute_center(value)` — [`L205`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rigid_matrix_vector.py#L205)
  - `from_array(cls, array)` — [`L160`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rigid_matrix_vector.py#L160)
  - `from_array4x4(cls, array: jnp.ndarray)` — [`L166`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rigid_matrix_vector.py#L166) — Construct Rigid3Array from homogeneous 4x4 array.
  - `from_point_alignment(cls, points_to: vector.Vec3Array, points_from: vector.Vec3Array, weights: Float | None = None, epsilon: float = 0.000001)` — [`L181`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rigid_matrix_vector.py#L181) — Constructs Rigid3Array by finding transform aligning points. — documented in [alphafold3-jax-geometry-rigid_matrix_vector](../../../../../concepts/alphafold3-jax-geometry-rigid_matrix_vector.md)
  - `identity(cls, shape: Any, dtype: jnp.dtype = jnp.float32)` — [`L143`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rigid_matrix_vector.py#L143) — Return identity Rigid3Array of given shape.
  - `inverse(self)` — [`L120`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rigid_matrix_vector.py#L120) — Return Rigid3Array corresponding to inverse transform. — documented in [alphafold3-jax-geometry-rigid_matrix_vector](../../../../../concepts/alphafold3-jax-geometry-rigid_matrix_vector.md)
  - `scale_translation(self, factor: Float)` — [`L150`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rigid_matrix_vector.py#L150) — Scale translation in Rigid3Array by 'factor'.
  - `to_array(self)` — [`L154`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rigid_matrix_vector.py#L154)
  - `rotation` — [`L112`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rigid_matrix_vector.py#L112) — documented in [alphafold3-jax-geometry-rigid_matrix_vector](../../../../../concepts/alphafold3-jax-geometry-rigid_matrix_vector.md)
  - `translation` — [`L113`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rigid_matrix_vector.py#L113) — documented in [alphafold3-jax-geometry-rigid_matrix_vector](../../../../../concepts/alphafold3-jax-geometry-rigid_matrix_vector.md)
- protocol/private: `__getstate__`[`L226`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rigid_matrix_vector.py#L226), `__matmul__`[`L115`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rigid_matrix_vector.py#L115), `__setstate__`[`L229`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rigid_matrix_vector.py#L229)
- uses (calls/refs, reference-scoped): [`Vec3Array`](vector.md#Vec3Array), [`apply_to_point`](rotation_matrix.md#Rot3Array.apply_to_point), [`inverse`](rotation_matrix.md#Rot3Array.inverse), [`to_array`](rotation_matrix.md#Rot3Array.to_array), [`from_svd`](rotation_matrix.md#Rot3Array.from_svd), [`_compute_covariance_matrix`](rigid_matrix_vector.md#_compute_covariance_matrix), [`Rot3Array`](rotation_matrix.md#Rot3Array), [`apply_inverse_to_point`](rotation_matrix.md#Rot3Array.apply_inverse_to_point), [`to_array`](vector.md#Vec3Array.to_array), [`from_array`](vector.md#Vec3Array.from_array), [`from_array`](rotation_matrix.md#Rot3Array.from_array), [`StructOfArray`](struct_of_array.md#StructOfArray), [`Float`](rigid_matrix_vector.md#Float.Float), [`VERSION`](rigid_matrix_vector.md#VERSION.VERSION), [`identity`](rotation_matrix.md#Rot3Array.identity), [`weighted_mean`](utils.md#weighted_mean), [`zeros`](vector.md#Vec3Array.zeros)
- used by: [`construct_input`](../../model/network/template_modules.md#SingleTemplateEmbedding.construct_input), [`Rigid3Array`](__init__.md#Rigid3Array)

## Functions
- `_compute_covariance_matrix(row_values: vector.Vec3Array, col_values: vector.Vec3Array, weights: jnp.ndarray, epsilon=0.000001)` — [`L39`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rigid_matrix_vector.py#L39) — Compute covariance matrix. — documented in [alphafold3-jax-geometry-rigid_matrix_vector](../../../../../concepts/alphafold3-jax-geometry-rigid_matrix_vector.md)

## Module values
- `Float` — [`L32`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rigid_matrix_vector.py#L32)
- `VERSION` — [`L34`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rigid_matrix_vector.py#L34)

