---
title: 'Module: jax/_src/numpy/scalar_types.py'
type: catalog
provenance: extracted
module: jax/_src/numpy/scalar_types.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.numpy.scalar_types`/
symbols:
  _make_scalar_type: _make_scalar_type().
  int32: int32.
  int64: int64.
  bfloat16: bfloat16.
  bool_: bool_.
  uint64: uint64.
  float32: float32.
  float64: float64.
  _ScalarMeta.__call__: _ScalarMeta#__call__().
  int4: int4.
  float8_e4m3b11fnuz: float8_e4m3b11fnuz.
  uint2: uint2.
  uint4: uint4.
  int2: int2.
  float4_e2m1fn: float4_e2m1fn.
  float6_e2m3fn: float6_e2m3fn.
  float6_e3m2fn: float6_e3m2fn.
  float8_e3m4: float8_e3m4.
  float8_e4m3: float8_e4m3.
  float8_e8m0fnu: float8_e8m0fnu.
  float8_e4m3fn: float8_e4m3fn.
  float8_e4m3fnuz: float8_e4m3fnuz.
  float8_e5m2: float8_e5m2.
  float8_e5m2fnuz: float8_e5m2fnuz.
  uint32: uint32.
  int8: int8.
  int16: int16.
  _ScalarMeta.dtype: _ScalarMeta#dtype.
  uint1: uint1.
  int1: int1.
  uint8: uint8.
  uint16: uint16.
  float16: float16.
  single: single.
  double: double.
  csingle: csingle.
  cdouble: cdouble.
  int_: int_.
  uint: uint.
  float_: float_.
  complex_: complex_.
  _ScalarMeta: _ScalarMeta#
  complex128: complex128.
  _ScalarMeta.__numpy_dtype__: _ScalarMeta#__numpy_dtype__().
  _ScalarMeta.__hash__: _ScalarMeta#__hash__().
  _ScalarMeta.__eq__: _ScalarMeta#__eq__().
  _ScalarMeta.__instancecheck__: _ScalarMeta#__instancecheck__().
  complex64: complex64.
  _PUBLIC_MODULE_NAME: _PUBLIC_MODULE_NAME.
  _abstractify_scalar_meta: _abstractify_scalar_meta().
  _ScalarMeta.__ne__: _ScalarMeta#__ne__().
---
# Module: [`jax/_src/numpy/scalar_types.py`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py)

## Classes
### `_ScalarMeta`  ·  implements/extends type
- def: [`jax/_src/numpy/scalar_types.py:37`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L37)
- signature: `class _ScalarMeta(type):`
- members:
  - `dtype` — [`L38`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L38)
- protocol/private: `__call__`[`L54`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L54), `__eq__`[`L48`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L48), `__hash__`[`L45`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L45), `__instancecheck__`[`L57`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L57), `__ne__`[`L51`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L51), `__numpy_dtype__`[`L41`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L41)
- uses (calls/refs, reference-scoped): [`Array`](../basearray.md#Array), [`asarray`](array_constructors.md#asarray)
- used by: [`_make_scalar_type`](scalar_types.md#_make_scalar_type)

## Functions
- `_abstractify_scalar_meta(x)` — [`L60`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L60)
- `_make_scalar_type(np_scalar_type: type)` — [`L64`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L64)

## Module values
- `_PUBLIC_MODULE_NAME` — [`L34`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L34)
- `bfloat16` — [`L104`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L104)
- `bool_` — [`L76`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L76)
- `cdouble` — [`L109`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L109)
- `complex128` — [`L109`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L109)
- `complex64` — [`L108`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L108)
- `complex_` — [`L114`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L114)
- `csingle` — [`L108`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L108)
- `double` — [`L107`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L107)
- `float16` — [`L105`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L105)
- `float32` — [`L106`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L106)
- `float4_e2m1fn` — [`L93`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L93)
- `float64` — [`L107`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L107)
- `float6_e2m3fn` — [`L94`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L94)
- `float6_e3m2fn` — [`L95`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L95)
- `float8_e3m4` — [`L96`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L96)
- `float8_e4m3` — [`L97`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L97)
- `float8_e4m3b11fnuz` — [`L103`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L103)
- `float8_e4m3fn` — [`L99`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L99)
- `float8_e4m3fnuz` — [`L100`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L100)
- `float8_e5m2` — [`L101`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L101)
- `float8_e5m2fnuz` — [`L102`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L102)
- `float8_e8m0fnu` — [`L98`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L98)
- `float_` — [`L113`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L113)
- `int1` — [`L86`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L86)
- `int16` — [`L90`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L90)
- `int2` — [`L87`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L87)
- `int32` — [`L91`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L91)
- `int4` — [`L88`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L88)
- `int64` — [`L92`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L92)
- `int8` — [`L89`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L89)
- `int_` — [`L111`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L111)
- `single` — [`L106`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L106)
- `uint` — [`L112`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L112)
- `uint1` — [`L78`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L78)
- `uint16` — [`L82`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L82)
- `uint2` — [`L79`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L79)
- `uint32` — [`L83`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L83)
- `uint4` — [`L80`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L80)
- `uint64` — [`L84`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L84)
- `uint8` — [`L81`](../../../../../../../raw/code/jax/jax/_src/numpy/scalar_types.py#L81)

