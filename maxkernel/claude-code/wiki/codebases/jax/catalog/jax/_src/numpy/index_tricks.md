---
title: 'Module: jax/_src/numpy/index_tricks.py'
type: catalog
provenance: extracted
module: jax/_src/numpy/index_tricks.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.numpy.index_tricks`/
symbols:
  _AxisConcat.__getitem__: _AxisConcat#__getitem__().
  _Mgrid.__getitem__: _Mgrid#__getitem__().
  _make_1d_grid_from_slice: _make_1d_grid_from_slice().
  _Ogrid.__getitem__: _Ogrid#__getitem__().
  _AxisConcat: _AxisConcat#
  export: export.
  ogrid: ogrid.
  _IndexType: _IndexType.
  RClass: RClass#
  CClass: CClass#
  mgrid: mgrid.
  r_: r_.
  c_: c_.
  _Mgrid: _Mgrid#
  _Ogrid: _Ogrid#
  _AxisConcat.axis: _AxisConcat#axis.
  _AxisConcat.ndmin: _AxisConcat#ndmin.
  _AxisConcat.trans1d: _AxisConcat#trans1d.
  _AxisConcat.op_name: _AxisConcat#op_name.
  __all__: __all__.
  _AxisConcat.__len__: _AxisConcat#__len__().
  RClass.axis: RClass#axis.
  RClass.ndmin: RClass#ndmin.
  RClass.trans1d: RClass#trans1d.
  RClass.op_name: RClass#op_name.
  CClass.axis: CClass#axis.
  CClass.ndmin: CClass#ndmin.
  CClass.trans1d: CClass#trans1d.
  CClass.op_name: CClass#op_name.
  s_: s_.
  index_exp: index_exp.
---
# Module: [`jax/_src/numpy/index_tricks.py`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py)

## Classes
### `CClass`  ·  implements/extends _AxisConcat
- def: [`jax/_src/numpy/index_tricks.py:289`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L289)
- doc: Concatenate slices, scalars and array-like objects along the last axis.
- signature: `class CClass(_AxisConcat):`
- members:
  - `axis` — [`L328`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L328)
  - `ndmin` — [`L329`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L329)
  - `op_name` — [`L331`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L331)
  - `trans1d` — [`L330`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L330)
- uses (calls/refs, reference-scoped): [`_AxisConcat`](index_tricks.md#_AxisConcat)
- used by: [`_AxisConcat`](index_tricks.md#_AxisConcat), [`c_`](index_tricks.md#c_), [`c_`](../../numpy/__init__.pyi.md#c_.c_)

### `RClass`  ·  implements/extends _AxisConcat
- def: [`jax/_src/numpy/index_tricks.py:216`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L216)
- doc: Concatenate slices, scalars and array-like objects along the first axis.
- signature: `class RClass(_AxisConcat):`
- members:
  - `axis` — [`L280`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L280)
  - `ndmin` — [`L281`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L281)
  - `op_name` — [`L283`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L283)
  - `trans1d` — [`L282`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L282)
- uses (calls/refs, reference-scoped): [`_AxisConcat`](index_tricks.md#_AxisConcat)
- used by: [`_AxisConcat`](index_tricks.md#_AxisConcat), [`r_`](index_tricks.md#r_), [`r_`](../../numpy/__init__.pyi.md#r_.r_)

### `_AxisConcat`
- def: [`jax/_src/numpy/index_tricks.py:144`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L144)
- doc: Concatenates slices, scalars and array-like objects along a given axis.
- signature: `class _AxisConcat:`
- members:
  - `axis` — [`L146`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L146)
  - `ndmin` — [`L147`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L147)
  - `op_name` — [`L149`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L149)
  - `trans1d` — [`L148`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L148)
- protocol/private: `__getitem__`[`L151`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L151), `__len__`[`L212`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L212)
- uses (calls/refs, reference-scoped): [`Array`](../basearray.md#Array), [`ndim`](../basearray.md#Array.ndim), [`array`](array_constructors.md#array), [`concatenate`](lax_numpy.md#concatenate), [`expand_dims`](lax_numpy.md#expand_dims), [`transpose`](lax_numpy.md#transpose), [`_make_1d_grid_from_slice`](index_tricks.md#_make_1d_grid_from_slice), [`CClass`](index_tricks.md#CClass), [`RClass`](index_tricks.md#RClass), [`_IndexType`](index_tricks.md#_IndexType)
- used by: [`CClass`](index_tricks.md#CClass), [`RClass`](index_tricks.md#RClass)

### `_Mgrid`
- def: [`jax/_src/numpy/index_tricks.py:55`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L55)
- doc: Return dense multi-dimensional "meshgrid".
- signature: `class _Mgrid:`
- protocol/private: `__getitem__`[`L84`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L84)
- uses (calls/refs, reference-scoped): [`Array`](../basearray.md#Array), [`arange`](lax_numpy.md#arange), [`promote_dtypes`](util.md#promote_dtypes), [`stack`](lax_numpy.md#stack), [`meshgrid`](lax_numpy.md#meshgrid), [`numpy_dtype_promotion`](../config.md#numpy_dtype_promotion), [`_make_1d_grid_from_slice`](index_tricks.md#_make_1d_grid_from_slice)
- used by: [`mgrid`](index_tricks.md#mgrid), [`mgrid`](../../numpy/__init__.pyi.md#mgrid.mgrid)

### `_Ogrid`
- def: [`jax/_src/numpy/index_tricks.py:99`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L99)
- doc: Return open multi-dimensional "meshgrid".
- signature: `class _Ogrid:`
- protocol/private: `__getitem__`[`L127`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L127)
- uses (calls/refs, reference-scoped): [`Array`](../basearray.md#Array), [`promote_dtypes`](util.md#promote_dtypes), [`meshgrid`](lax_numpy.md#meshgrid), [`numpy_dtype_promotion`](../config.md#numpy_dtype_promotion), [`_make_1d_grid_from_slice`](index_tricks.md#_make_1d_grid_from_slice)
- used by: [`ogrid`](index_tricks.md#ogrid), [`ogrid`](../../numpy/__init__.pyi.md#ogrid.ogrid)

## Functions
- `_make_1d_grid_from_slice(s: slice, op_name: str)` — [`L40`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L40)

## Module values
- `_IndexType` — [`L141`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L141)
- `__all__` — [`L37`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L37)
- `c_` — [`L334`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L334)
- `export` — [`L34`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L34)
- `index_exp` — [`L338`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L338)
- `mgrid` — [`L96`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L96)
- `ogrid` — [`L138`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L138)
- `r_` — [`L286`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L286)
- `s_` — [`L336`](../../../../../../../raw/code/jax/jax/_src/numpy/index_tricks.py#L336)

