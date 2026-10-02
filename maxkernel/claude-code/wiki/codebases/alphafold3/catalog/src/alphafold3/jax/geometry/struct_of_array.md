---
title: 'Module: src/alphafold3/jax/geometry/struct_of_array.py'
type: catalog
provenance: extracted
module: src/alphafold3/jax/geometry/struct_of_array.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.jax.geometry.struct_of_array`/
symbols:
  StructOfArray.__call__: StructOfArray#__call__().
  get_array_fields: get_array_fields().
  flatten: flatten().
  StructOfArray.unflatten: StructOfArray#unflatten().
  get_metadata_fields: get_metadata_fields().
  StructOfArray: StructOfArray#
  get_item: get_item().
  post_init: post_init().
  make_metadata_class: make_metadata_class().
  get_fields: get_fields().
  get_shape: get_shape().
  get_len: get_len().
  get_dtype: get_dtype().
  replace: replace().
  StructOfArray.same_dtype: StructOfArray#same_dtype.
  StructOfArray.__init__: StructOfArray#__init__().
---
# Module: [`src/alphafold3/jax/geometry/struct_of_array.py`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/struct_of_array.py)

## Classes
### `StructOfArray`
- def: [`src/alphafold3/jax/geometry/struct_of_array.py:199`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/struct_of_array.py#L199) — documented in [alphafold3-jax-geometry-rigid_matrix_vector](../../../../../concepts/alphafold3-jax-geometry-rigid_matrix_vector.md)
- doc: Class Decorator for Struct Of Arrays.
- signature: `class StructOfArray:`
- members:
  - `unflatten(aux, data)` — [`L219`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/struct_of_array.py#L219)
  - `same_dtype` — [`L203`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/struct_of_array.py#L203)
- protocol/private: `__call__`[`L205`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/struct_of_array.py#L205), `__init__`[`L202`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/struct_of_array.py#L202)
- uses (calls/refs, reference-scoped): [`get_array_fields`](struct_of_array.md#get_array_fields), [`flatten`](struct_of_array.md#flatten), [`get_metadata_fields`](struct_of_array.md#get_metadata_fields), [`get_item`](struct_of_array.md#get_item), [`make_metadata_class`](struct_of_array.md#make_metadata_class), [`post_init`](struct_of_array.md#post_init), [`get_dtype`](struct_of_array.md#get_dtype), [`get_len`](struct_of_array.md#get_len), [`get_shape`](struct_of_array.md#get_shape), [`replace`](struct_of_array.md#replace)
- used by: [`Rigid3Array`](rigid_matrix_vector.md#Rigid3Array), [`MATRIX_SVD_QUAT_FACTORS`](rotation_matrix.md#MATRIX_SVD_QUAT_FACTORS), [`VERSION`](vector.md#VERSION.VERSION), [`StructOfArray`](__init__.md#StructOfArray)

## Functions
- `flatten(instance)` — [`L143`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/struct_of_array.py#L143) — Flatten Struct of Array instance.
- `get_array_fields(cls, return_values=False)` — [`L183`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/struct_of_array.py#L183)
- `get_dtype(instance)` — [`L63`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/struct_of_array.py#L63) — Returns Dtype for given instance of dataclass.
- `get_fields(cls_or_instance, filterfn, return_values=False)` — [`L172`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/struct_of_array.py#L172)
- `get_item(instance, key)` — [`L27`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/struct_of_array.py#L27)
- `get_len(instance)` — [`L53`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/struct_of_array.py#L53) — Returns length for given instance of dataclass.
- `get_metadata_fields(cls, return_values=False)` — [`L191`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/struct_of_array.py#L191)
- `get_shape(instance)` — [`L42`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/struct_of_array.py#L42) — Returns Shape for given instance of dataclass.
- `make_metadata_class(cls)` — [`L159`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/struct_of_array.py#L159)
- `post_init(instance)` — [`L92`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/struct_of_array.py#L92) — Validate instance has same shapes & dtypes.
- `replace(instance, **kwargs)` — [`L88`](../../../../../../../../raw/code/alphafold3/src/alphafold3/jax/geometry/struct_of_array.py#L88)

