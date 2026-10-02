---
title: 'Module: jax/_src/typing.py'
type: catalog
provenance: extracted
module: jax/_src/typing.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.typing`/
symbols:
  DTypeLike: DTypeLike.
  DimSize: DimSize.
  DType: DType.
  Shape: Shape.
  DLDeviceType: DLDeviceType#
  Index: Index.
  DuckTypedArray: DuckTypedArray#
  SupportsShape: SupportsShape#
  DeprecatedArg: DeprecatedArg#
  SupportsSize: SupportsSize#
  SupportsNdim: SupportsNdim#
  StaticIndex: StaticIndex.
  DLDeviceType.kDLCUDAHost: DLDeviceType#kDLCUDAHost.
  DLDeviceType.kDLTPUHost: DLDeviceType#kDLTPUHost.
  SupportsDType.dtype: SupportsDType#dtype().
  DuckTypedArray.dtype: DuckTypedArray#dtype().
  DuckTypedArray.shape: DuckTypedArray#shape().
  ExtendedDType: ExtendedDType.
  DLDeviceType.kDLCPU: DLDeviceType#kDLCPU.
  DLDeviceType.kDLCUDA: DLDeviceType#kDLCUDA.
  DLDeviceType.kDLROCM: DLDeviceType#kDLROCM.
  AnyInt: AnyInt.
  SupportsDType: SupportsDType#
  SupportsShape.shape: SupportsShape#shape().
  SupportsSize.size: SupportsSize#size().
  SupportsNdim.ndim: SupportsNdim#ndim().
  DeprecatedArg.__repr__: DeprecatedArg#__repr__().
---
# Module: [`jax/_src/typing.py`](../../../../../../raw/code/jax/jax/_src/typing.py)

## Classes
### `DLDeviceType`  ·  implements/extends IntEnum
- def: [`jax/_src/typing.py:106`](../../../../../../raw/code/jax/jax/_src/typing.py#L106)
- signature: `class DLDeviceType(enum.IntEnum):`
- members:
  - `kDLCPU` — [`L107`](../../../../../../raw/code/jax/jax/_src/typing.py#L107)
  - `kDLCUDA` — [`L108`](../../../../../../raw/code/jax/jax/_src/typing.py#L108)
  - `kDLCUDAHost` — [`L109`](../../../../../../raw/code/jax/jax/_src/typing.py#L109)
  - `kDLROCM` — [`L110`](../../../../../../raw/code/jax/jax/_src/typing.py#L110)
  - `kDLTPUHost` — [`L111`](../../../../../../raw/code/jax/jax/_src/typing.py#L111)
- used by: [`from_dlpack`](dlpack.md#from_dlpack), [`__dlpack_device__`](array.md#ArrayImpl.__dlpack_device__), [`to_dlpack`](dlpack.md#to_dlpack), [`_DL_DEVICE_TO_PLATFORM`](dlpack.md#_DL_DEVICE_TO_PLATFORM), [`__dlpack__`](array.md#ArrayImpl.__dlpack__)

### `DeprecatedArg`
- def: [`jax/_src/typing.py:101`](../../../../../../raw/code/jax/jax/_src/typing.py#L101)
- signature: `class DeprecatedArg:`
- protocol/private: `__repr__`[`L102`](../../../../../../raw/code/jax/jax/_src/typing.py#L102)
- used by: [`percentile`](../numpy/__init__.pyi.md#percentile), [`clip`](../numpy/__init__.pyi.md#clip), [`nanpercentile`](../numpy/__init__.pyi.md#nanpercentile), [`nanquantile`](../numpy/__init__.pyi.md#nanquantile), [`quantile`](../numpy/__init__.pyi.md#quantile)

### `DuckTypedArray`  ·  implements/extends Protocol
- def: [`jax/_src/typing.py:84`](../../../../../../raw/code/jax/jax/_src/typing.py#L84)
- signature: `class DuckTypedArray(Protocol):`
- members:
  - `dtype(self)` — [`L86`](../../../../../../raw/code/jax/jax/_src/typing.py#L86)
  - `shape(self)` — [`L88`](../../../../../../raw/code/jax/jax/_src/typing.py#L88)
- uses (calls/refs, reference-scoped): [`DType`](typing.md#DType), [`Shape`](typing.md#Shape)
- used by: [`full_like`](lax/lax.md#full_like), [`empty_like`](lax/lax.md#empty_like), [`full_like`](numpy/array_creation.md#full_like), [`zeros_like`](../numpy/__init__.pyi.md#zeros_like), [`zeros_like`](numpy/array_creation.md#zeros_like), [`ones_like`](numpy/array_creation.md#ones_like), [`empty_like`](numpy/array_creation.md#empty_like), [`full_like`](../numpy/__init__.pyi.md#full_like), [`ones_like`](../numpy/__init__.pyi.md#ones_like), [`empty_like`](../numpy/__init__.pyi.md#empty_like)

### `SupportsDType`  ·  implements/extends Protocol
- def: [`jax/_src/typing.py:49`](../../../../../../raw/code/jax/jax/_src/typing.py#L49)
- signature: `class SupportsDType(Protocol):`
- members:
  - `dtype(self,)` — [`L51`](../../../../../../raw/code/jax/jax/_src/typing.py#L51)
- uses (calls/refs, reference-scoped): [`DType`](typing.md#DType)
- used by: [`DTypeLike`](typing.md#DTypeLike)

### `SupportsNdim`  ·  implements/extends Protocol
- def: [`jax/_src/typing.py:61`](../../../../../../raw/code/jax/jax/_src/typing.py#L61)
- signature: `class SupportsNdim(Protocol):`
- members:
  - `ndim(self,)` — [`L63`](../../../../../../raw/code/jax/jax/_src/typing.py#L63)
- used by: [`ndim`](numpy/util.md#ndim), [`ndim`](../numpy/__init__.pyi.md#ndim)

### `SupportsShape`  ·  implements/extends Protocol
- def: [`jax/_src/typing.py:53`](../../../../../../raw/code/jax/jax/_src/typing.py#L53)
- signature: `class SupportsShape(Protocol):`
- members:
  - `shape(self,)` — [`L55`](../../../../../../raw/code/jax/jax/_src/typing.py#L55)
- used by: [`size`](numpy/util.md#size), [`tril_indices_from`](numpy/lax_numpy.md#tril_indices_from), [`triu_indices_from`](numpy/lax_numpy.md#triu_indices_from), [`shape`](numpy/util.md#shape), [`get_fusion_values`](pallas/fuser/block_spec.md#get_fusion_values), [`tril_indices_from`](../numpy/__init__.pyi.md#tril_indices_from), [`triu_indices_from`](../numpy/__init__.pyi.md#triu_indices_from), [`shape`](../numpy/__init__.pyi.md#shape)

### `SupportsSize`  ·  implements/extends Protocol
- def: [`jax/_src/typing.py:57`](../../../../../../raw/code/jax/jax/_src/typing.py#L57)
- signature: `class SupportsSize(Protocol):`
- members:
  - `size(self,)` — [`L59`](../../../../../../raw/code/jax/jax/_src/typing.py#L59)
- used by: [`size`](numpy/util.md#size), [`size`](../numpy/__init__.pyi.md#size)

## Module values
- `AnyInt` — [`L113`](../../../../../../raw/code/jax/jax/_src/typing.py#L113)
- `DType` — [`L42`](../../../../../../raw/code/jax/jax/_src/typing.py#L42)
- `DTypeLike` — [`L71`](../../../../../../raw/code/jax/jax/_src/typing.py#L71)
- `DimSize` — [`L81`](../../../../../../raw/code/jax/jax/_src/typing.py#L81)
- `ExtendedDType` — [`L45`](../../../../../../raw/code/jax/jax/_src/typing.py#L45)
- `Index` — [`L115`](../../../../../../raw/code/jax/jax/_src/typing.py#L115)
- `Shape` — [`L82`](../../../../../../raw/code/jax/jax/_src/typing.py#L82)
- `StaticIndex` — [`L114`](../../../../../../raw/code/jax/jax/_src/typing.py#L114)

