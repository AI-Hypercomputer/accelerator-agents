---
title: 'Module: jax/_src/numpy/array_api_metadata.py'
type: catalog
provenance: extracted
module: jax/_src/numpy/array_api_metadata.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.numpy.array_api_metadata`/
symbols:
  ArrayNamespaceInfo.default_dtypes: ArrayNamespaceInfo#default_dtypes().
  ArrayNamespaceInfo._build_dtype_dict: ArrayNamespaceInfo#_build_dtype_dict().
  ArrayNamespaceInfo.devices: ArrayNamespaceInfo#devices().
  ArrayNamespaceInfo.dtypes: ArrayNamespaceInfo#dtypes().
  ArrayNamespaceInfo: ArrayNamespaceInfo#
  __array_namespace__: __array_namespace__().
  __array_namespace_info__: __array_namespace_info__().
  ArrayNamespaceInfo.capabilities: ArrayNamespaceInfo#capabilities().
  __array_api_version__: __array_api_version__.
  ArrayNamespaceInfo._capabilities: ArrayNamespaceInfo#_capabilities.
  ArrayNamespaceInfo.default_device: ArrayNamespaceInfo#default_device().
---
# Module: [`jax/_src/numpy/array_api_metadata.py`](../../../../../../../raw/code/jax/jax/_src/numpy/array_api_metadata.py)

## Classes
### `ArrayNamespaceInfo`
- def: [`jax/_src/numpy/array_api_metadata.py:50`](../../../../../../../raw/code/jax/jax/_src/numpy/array_api_metadata.py#L50)
- members:
  - `capabilities(self)` — [`L83`](../../../../../../../raw/code/jax/jax/_src/numpy/array_api_metadata.py#L83)
  - `default_device(self)` — [`L72`](../../../../../../../raw/code/jax/jax/_src/numpy/array_api_metadata.py#L72)
  - `default_dtypes(self, *, device: xc.Device | Sharding | None = None)` — [`L86`](../../../../../../../raw/code/jax/jax/_src/numpy/array_api_metadata.py#L86)
  - `devices(self)` — [`L77`](../../../../../../../raw/code/jax/jax/_src/numpy/array_api_metadata.py#L77)
  - `dtypes(self, *, device: xc.Device | Sharding | None = None, kind: str | tuple[str, ...] | None = None)` — [`L96`](../../../../../../../raw/code/jax/jax/_src/numpy/array_api_metadata.py#L96)
- protocol/private: `_build_dtype_dict`[`L61`](../../../../../../../raw/code/jax/jax/_src/numpy/array_api_metadata.py#L61), `_capabilities`[`L55`](../../../../../../../raw/code/jax/jax/_src/numpy/array_api_metadata.py#L55)
- uses (calls/refs, reference-scoped): [`Sharding`](../sharding.md#Sharding), [`backends`](../xla_bridge.md#backends), [`enable_x64`](../config.md#enable_x64), [`default_int_dtype`](../dtypes.md#default_int_dtype), [`devices`](../xla_bridge.md#devices), [`_dtype_kinds`](../dtypes.md#_dtype_kinds._dtype_kinds), [`default_float_dtype`](../dtypes.md#default_float_dtype), [`default_complex_dtype`](../dtypes.md#default_complex_dtype)
- used by: [`__array_namespace_info__`](array_api_metadata.md#__array_namespace_info__), [`__array_namespace_info__`](../../numpy/__init__.pyi.md#__array_namespace_info__)

## Functions
- `__array_namespace__(self, *, api_version: None | str = None)` — [`L34`](../../../../../../../raw/code/jax/jax/_src/numpy/array_api_metadata.py#L34) — Return the `Python array API`_ namespace for JAX.
- `__array_namespace_info__()` — [`L46`](../../../../../../../raw/code/jax/jax/_src/numpy/array_api_metadata.py#L46)

## Module values
- `__array_api_version__` — [`L31`](../../../../../../../raw/code/jax/jax/_src/numpy/array_api_metadata.py#L31)

