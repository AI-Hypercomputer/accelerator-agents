---
title: 'Module: src/alphafold3/jax/geometry/vector.py'
type: catalog
provenance: extracted
module: src/alphafold3/jax/geometry/vector.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.jax.geometry.vector`/
symbols:
  Vec3Array: Vec3Array#
  Vec3Array.x: Vec3Array#x.
  Vec3Array.y: Vec3Array#y.
  Vec3Array.z: Vec3Array#z.
  Vec3Array.dot: Vec3Array#dot().
  Vec3Array.cross: Vec3Array#cross().
  dihedral_angle: dihedral_angle().
  Float.Float: Float.Float.
  Vec3Array.__getstate__: Vec3Array#__getstate__().
  square_euclidean_distance: square_euclidean_distance().
  Vec3Array.to_array: Vec3Array#to_array().
  dot: dot().
  cross: cross().
  euclidean_distance: euclidean_distance().
  Vec3Array.norm: Vec3Array#norm().
  Vec3Array.__post_init__: Vec3Array#__post_init__().
  norm: norm().
  Vec3Array.normalized: Vec3Array#normalized().
  Vec3Array.from_array: Vec3Array#from_array().
  normalized: normalized().
  random_gaussian_vector: random_gaussian_vector().
  VERSION.VERSION: VERSION.VERSION.
  Vec3Array.__mul__: Vec3Array#__mul__().
  Vec3Array.__rmul__: Vec3Array#__rmul__().
  Vec3Array.__truediv__: Vec3Array#__truediv__().
  Vec3Array.norm2: Vec3Array#norm2().
  Vec3Array.zeros: Vec3Array#zeros().
  Vec3Array.__add__: Vec3Array#__add__().
  Vec3Array.__sub__: Vec3Array#__sub__().
  Vec3Array.__neg__: Vec3Array#__neg__().
  Vec3Array.__pos__: Vec3Array#__pos__().
  Vec3Array.__setstate__: Vec3Array#__setstate__().
---
# Module: [`src/alphafold3/jax/geometry/vector.py`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py)

## Classes
### `Vec3Array`
- def: [`src/alphafold3/jax/geometry/vector.py:38`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L38) — documented in [alphafold3-jax-geometry-rigid_matrix_vector](../../../../../concepts/alphafold3-jax-geometry-rigid_matrix_vector.md)
- doc: Vec3Array in 3 dimensional Space implemented as struct of arrays.
- signature: `class Vec3Array:`
- members:
  - `cross(self, other: Self)` — [`L87`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L87) — Compute cross product between 'self' and 'other'. — documented in [alphafold3-jax-geometry-vector](../../../../../concepts/alphafold3-jax-geometry-vector.md)
  - `dot(self, other: Self)` — [`L94`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L94) — Compute dot product between 'self' and 'other'. — documented in [alphafold3-jax-geometry-vector](../../../../../concepts/alphafold3-jax-geometry-vector.md)
  - `from_array(cls, array)` — [`L126`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L126)
  - `norm(self, epsilon: float = 0.000001)` — [`L98`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L98) — Compute Norm of Vec3Array, clipped to epsilon.
  - `norm2(self)` — [`L106`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L106)
  - `normalized(self, epsilon: float = 0.000001)` — [`L109`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L109) — Return unit vector with optional clipping.
  - `to_array(self)` — [`L122`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L122) — documented in [alphafold3-jax-geometry-vector](../../../../../concepts/alphafold3-jax-geometry-vector.md)
  - `zeros(cls, shape, dtype=jnp.float32)` — [`L114`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L114) — Return Vec3Array corresponding to zeros of given shape.
  - `x` — [`L51`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L51) — documented in [alphafold3-jax-geometry-vector](../../../../../concepts/alphafold3-jax-geometry-vector.md)
  - `y` — [`L52`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L52) — documented in [alphafold3-jax-geometry-vector](../../../../../concepts/alphafold3-jax-geometry-vector.md)
  - `z` — [`L53`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L53) — documented in [alphafold3-jax-geometry-vector](../../../../../concepts/alphafold3-jax-geometry-vector.md)
- protocol/private: `__add__`[`L66`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L66), `__getstate__`[`L129`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L129), `__mul__`[`L72`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L72), `__neg__`[`L81`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L81), `__pos__`[`L84`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L84), `__post_init__`[`L55`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L55), `__rmul__`[`L75`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L75), `__setstate__`[`L135`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L135), `__sub__`[`L69`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L69), `__truediv__`[`L78`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L78)
- uses (calls/refs, reference-scoped): [`Float`](vector.md#Float.Float), [`unstack`](utils.md#unstack), [`VERSION`](vector.md#VERSION.VERSION)
- used by: [`apply_to_point`](rotation_matrix.md#Rot3Array.apply_to_point), [`construct_input`](../../model/network/template_modules.md#SingleTemplateEmbedding.construct_input), [`__matmul__`](rotation_matrix.md#Rot3Array.__matmul__), [`from_two_vectors`](rotation_matrix.md#Rot3Array.from_two_vectors), [`from_point_alignment`](rigid_matrix_vector.md#Rigid3Array.from_point_alignment), [`dihedral_angle`](vector.md#dihedral_angle), [`translation`](rigid_matrix_vector.md#Rigid3Array.translation), [`_compute_covariance_matrix`](rigid_matrix_vector.md#_compute_covariance_matrix), [`apply_to_point`](rigid_matrix_vector.md#Rigid3Array.apply_to_point), [`apply_inverse_to_point`](rigid_matrix_vector.md#Rigid3Array.apply_inverse_to_point), [`from_array`](rigid_matrix_vector.md#Rigid3Array.from_array), [`identity`](rigid_matrix_vector.md#Rigid3Array.identity), [`to_array`](rigid_matrix_vector.md#Rigid3Array.to_array), [`square_euclidean_distance`](vector.md#square_euclidean_distance), [`apply_inverse_to_point`](rotation_matrix.md#Rot3Array.apply_inverse_to_point), [`cross`](vector.md#cross), [`dot`](vector.md#dot), [`euclidean_distance`](vector.md#euclidean_distance), [`norm`](vector.md#norm), [`from_array4x4`](rigid_matrix_vector.md#Rigid3Array.from_array4x4), [`normalized`](vector.md#normalized), [`random_gaussian_vector`](vector.md#random_gaussian_vector), [`Vec3Array`](__init__.md#Vec3Array)

## Functions
- `cross(vector1: Vec3Array, vector2: Vec3Array)` — [`L168`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L168) — documented in [alphafold3-jax-geometry-vector](../../../../../concepts/alphafold3-jax-geometry-vector.md)
- `dihedral_angle(a: Vec3Array, b: Vec3Array, c: Vec3Array, d: Vec3Array)` — [`L200`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L200) — Computes torsion angle for a quadruple of points.
- `dot(vector1: Vec3Array, vector2: Vec3Array)` — [`L164`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L164) — documented in [alphafold3-jax-geometry-vector](../../../../../concepts/alphafold3-jax-geometry-vector.md)
- `euclidean_distance(vec1: Vec3Array, vec2: Vec3Array, epsilon: float = 0.000001)` — [`L180`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L180) — Computes euclidean distance between 'vec1' and 'vec2'.
- `norm(vector: Vec3Array, epsilon: float = 0.000001)` — [`L172`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L172) — documented in [alphafold3-jax-geometry-vector](../../../../../concepts/alphafold3-jax-geometry-vector.md)
- `normalized(vector: Vec3Array, epsilon: float = 0.000001)` — [`L176`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L176) — documented in [alphafold3-jax-geometry-vector](../../../../../concepts/alphafold3-jax-geometry-vector.md)
- `random_gaussian_vector(shape, key, dtype=jnp.float32)` — [`L229`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L229)
- `square_euclidean_distance(vec1: Vec3Array, vec2: Vec3Array, epsilon: float = 0.000001)` — [`L142`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L142) — Computes square of euclidean distance between 'vec1' and 'vec2'.

## Module values
- `Float` — [`L32`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L32)
- `VERSION` — [`L34`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/vector.py#L34)

