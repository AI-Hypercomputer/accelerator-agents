---
title: 'Module: src/alphafold3/jax/geometry/rotation_matrix.py'
type: catalog
provenance: extracted
module: src/alphafold3/jax/geometry/rotation_matrix.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.jax.geometry.rotation_matrix`/
symbols:
  Rot3Array.apply_to_point: Rot3Array#apply_to_point().
  Rot3Array.__matmul__: Rot3Array#__matmul__().
  Rot3Array.inverse: Rot3Array#inverse().
  Rot3Array.to_array: Rot3Array#to_array().
  Rot3Array.from_two_vectors: Rot3Array#from_two_vectors().
  Rot3Array.from_svd: Rot3Array#from_svd().
  Rot3Array: Rot3Array#
  Rot3Array.apply_inverse_to_point: Rot3Array#apply_inverse_to_point().
  MATRIX_SVD_QUAT_FACTORS: MATRIX_SVD_QUAT_FACTORS.
  Rot3Array.random_uniform: Rot3Array#random_uniform().
  Rot3Array.__getstate__: Rot3Array#__getstate__().
  Rot3Array.from_array: Rot3Array#from_array().
  Rot3Array.xx: Rot3Array#xx.
  Rot3Array.xy: Rot3Array#xy.
  Rot3Array.xz: Rot3Array#xz.
  Rot3Array.yx: Rot3Array#yx.
  Rot3Array.yy: Rot3Array#yy.
  Rot3Array.yz: Rot3Array#yz.
  Rot3Array.zx: Rot3Array#zx.
  Rot3Array.zy: Rot3Array#zy.
  Rot3Array.zz: Rot3Array#zz.
  Rot3Array.__setstate__: Rot3Array#__setstate__().
  largest_evec: largest_evec().
  Rot3Array.from_quaternion: Rot3Array#from_quaternion().
  COMPONENTS.COMPONENTS: COMPONENTS.COMPONENTS.
  make_matrix_svd_factors: make_matrix_svd_factors().
  largest_evec_jvp: largest_evec_jvp().
  Rot3Array.identity: Rot3Array#identity().
  VERSION.VERSION: VERSION.VERSION.
  Rot3Array.__array_ufunc__: Rot3Array#__array_ufunc__.
---
# Module: [`src/alphafold3/jax/geometry/rotation_matrix.py`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py)

## Classes
### `Rot3Array`
- def: [`src/alphafold3/jax/geometry/rotation_matrix.py:118`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L118) — documented in [alphafold3-jax-geometry-rigid_matrix_vector](../../../../../concepts/alphafold3-jax-geometry-rigid_matrix_vector.md)
- doc: Rot3Array Matrix in 3 dimensional Space implemented as struct of arrays.
- signature: `class Rot3Array:`
- members:
  - `__matmul__(self, other: Self)` — [`L153`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L153) — Composes two Rot3Arrays. — documented in [alphafold3-jax-geometry-rotation_matrix](../../../../../concepts/alphafold3-jax-geometry-rotation_matrix.md)
  - `apply_inverse_to_point(self, point: vector.Vec3Array)` — [`L149`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L149) — Applies inverse Rot3Array to point. — documented in [alphafold3-jax-geometry-rotation_matrix](../../../../../concepts/alphafold3-jax-geometry-rotation_matrix.md)
  - `apply_to_point(self, point: vector.Vec3Array)` — [`L141`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L141) — Applies Rot3Array to point. — documented in [alphafold3-jax-geometry-rotation_matrix](../../../../../concepts/alphafold3-jax-geometry-rotation_matrix.md)
  - `from_array(cls, array: jnp.ndarray)` — [`L191`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L191) — Construct Rot3Array Matrix from array of shape. [..., 3, 3].
  - `from_quaternion(cls, w: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray, normalize: bool = True, epsilon: float = 0.000001)` — [`L209`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L209) — Construct Rot3Array from components of quaternion.
  - `from_svd(cls, mat: jnp.ndarray, use_quat_formula: bool = True)` — [`L237`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L237) — Constructs Rot3Array from arbitrary array of shape [3 * 3] using SVD. — documented in [alphafold3-jax-geometry-rigid_matrix_vector](../../../../../concepts/alphafold3-jax-geometry-rigid_matrix_vector.md)
  - `from_two_vectors(cls, e0: vector.Vec3Array, e1: vector.Vec3Array)` — [`L168`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L168) — Construct Rot3Array from two Vectors. — documented in [alphafold3-jax-geometry-vector](../../../../../concepts/alphafold3-jax-geometry-vector.md)
  - `identity(cls, shape: Any, dtype: jnp.dtype = jnp.float32)` — [`L161`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L161) — Returns identity of given shape.
  - `inverse(self)` — [`L133`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L133) — Returns inverse of Rot3Array. — documented in [alphafold3-jax-geometry-rigid_matrix_vector](../../../../../concepts/alphafold3-jax-geometry-rigid_matrix_vector.md)
  - `random_uniform(cls, key, shape, dtype=jnp.float32)` — [`L287`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L287) — Samples uniform random Rot3Array according to Haar Measure.
  - `to_array(self)` — [`L197`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L197) — Convert Rot3Array to array of shape [..., 3, 3]. — documented in [alphafold3-jax-geometry-rotation_matrix](../../../../../concepts/alphafold3-jax-geometry-rotation_matrix.md)
  - `xx` — [`L121`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L121) — documented in [alphafold3-jax-geometry-rotation_matrix](../../../../../concepts/alphafold3-jax-geometry-rotation_matrix.md)
  - `xy` — [`L122`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L122) — documented in [alphafold3-jax-geometry-rotation_matrix](../../../../../concepts/alphafold3-jax-geometry-rotation_matrix.md)
  - `xz` — [`L123`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L123)
  - `yx` — [`L124`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L124)
  - `yy` — [`L125`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L125)
  - `yz` — [`L126`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L126)
  - `zx` — [`L127`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L127)
  - `zy` — [`L128`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L128)
  - `zz` — [`L129`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L129) — documented in [alphafold3-jax-geometry-rotation_matrix](../../../../../concepts/alphafold3-jax-geometry-rotation_matrix.md)
- protocol/private: `__array_ufunc__`[`L131`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L131), `__getstate__`[`L293`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L293), `__setstate__`[`L296`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L296)
- uses (calls/refs, reference-scoped): [`Vec3Array`](vector.md#Vec3Array), [`x`](vector.md#Vec3Array.x), [`y`](vector.md#Vec3Array.y), [`z`](vector.md#Vec3Array.z), [`dot`](vector.md#Vec3Array.dot), [`cross`](vector.md#Vec3Array.cross), [`MATRIX_SVD_QUAT_FACTORS`](rotation_matrix.md#MATRIX_SVD_QUAT_FACTORS), [`normalized`](vector.md#Vec3Array.normalized), [`unstack`](utils.md#unstack), [`COMPONENTS`](rotation_matrix.md#COMPONENTS.COMPONENTS), [`largest_evec`](rotation_matrix.md#largest_evec), [`VERSION`](rotation_matrix.md#VERSION.VERSION)
- used by: [`from_point_alignment`](rigid_matrix_vector.md#Rigid3Array.from_point_alignment), [`inverse`](rigid_matrix_vector.md#Rigid3Array.inverse), [`rotation`](rigid_matrix_vector.md#Rigid3Array.rotation), [`apply_to_point`](rigid_matrix_vector.md#Rigid3Array.apply_to_point), [`make_backbone_rigid`](../../model/network/template_modules.md#make_backbone_rigid), [`apply_inverse_to_point`](rigid_matrix_vector.md#Rigid3Array.apply_inverse_to_point), [`compose_rotation`](rigid_matrix_vector.md#Rigid3Array.compose_rotation), [`from_array`](rigid_matrix_vector.md#Rigid3Array.from_array), [`identity`](rigid_matrix_vector.md#Rigid3Array.identity), [`to_array`](rigid_matrix_vector.md#Rigid3Array.to_array), [`from_array4x4`](rigid_matrix_vector.md#Rigid3Array.from_array4x4), [`Rot3Array`](__init__.md#Rot3Array)

## Functions
- `largest_evec(m)` — [`L78`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L78)
- `largest_evec_jvp(primals, tangents)` — [`L83`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L83) — jvp for largest eigenvector.
- `make_matrix_svd_factors()` — [`L41`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L41) — Generates factors for converting 3x3 matrix to symmetric 4x4 matrix.

## Module values
- `COMPONENTS` — [`L33`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L33)
- `MATRIX_SVD_QUAT_FACTORS` — [`L114`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L114)
- `VERSION` — [`L38`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/rotation_matrix.py#L38)

