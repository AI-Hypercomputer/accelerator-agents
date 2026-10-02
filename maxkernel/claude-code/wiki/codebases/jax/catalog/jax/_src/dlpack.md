---
title: 'Module: jax/_src/dlpack.py'
type: catalog
provenance: extracted
module: jax/_src/dlpack.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.dlpack`/
symbols:
  SUPPORTED_DTYPES.SUPPORTED_DTYPES: SUPPORTED_DTYPES.SUPPORTED_DTYPES.
  from_dlpack: from_dlpack().
  to_dlpack: to_dlpack().
  _DL_DEVICE_TO_PLATFORM: _DL_DEVICE_TO_PLATFORM.
  _to_dlpack: _to_dlpack().
  is_supported_dtype: is_supported_dtype().
  _place_array: _place_array().
  SUPPORTED_DTYPES_SET.SUPPORTED_DTYPES_SET: SUPPORTED_DTYPES_SET.SUPPORTED_DTYPES_SET.
  MIN_DLPACK_VERSION: MIN_DLPACK_VERSION.
  DLPACK_VERSION: DLPACK_VERSION.
  _check_device: _check_device().
  _is_tensorflow_tensor: _is_tensorflow_tensor().
---
# Module: [`jax/_src/dlpack.py`](../../../../../../raw/code/jax/jax/_src/dlpack.py)

## Functions
- `_check_device(device, dlpack_device, copy)` — [`L175`](../../../../../../raw/code/jax/jax/_src/dlpack.py#L175)
- `_is_tensorflow_tensor(external_array)` — [`L191`](../../../../../../raw/code/jax/jax/_src/dlpack.py#L191)
- `_place_array(_arr, device, dlpack_device, copy)` — [`L184`](../../../../../../raw/code/jax/jax/_src/dlpack.py#L184)
- `_to_dlpack(x: Array, stream: int | Any | None, src_device: _jax.Device | None = None, device: _jax.Device | None = None, copy: bool | None = None)` — [`L61`](../../../../../../raw/code/jax/jax/_src/dlpack.py#L61)
- `from_dlpack(external_array, device: _jax.Device | Sharding | None = None, copy: bool | None = None)` — [`L198`](../../../../../../raw/code/jax/jax/_src/dlpack.py#L198) — Returns a :class:`~jax.Array` representation of a DLPack tensor.
- `is_supported_dtype(dtype: DTypeLike)` — [`L53`](../../../../../../raw/code/jax/jax/_src/dlpack.py#L53) — Check if dtype is supported by jax.dlpack.
- `to_dlpack(x: Array, stream: int | Any | None = None, src_device: _jax.Device | None = None, dl_device: tuple[DLDeviceType, int] | None = None, max_version: tuple[int, int] | None = None, copy: bool | None = None)` — [`L93`](../../../../../../raw/code/jax/jax/_src/dlpack.py#L93) — Returns a DLPack tensor that encapsulates a :class:`~jax.Array` ``x``.

## Module values
- `DLPACK_VERSION` — [`L33`](../../../../../../raw/code/jax/jax/_src/dlpack.py#L33)
- `MIN_DLPACK_VERSION` — [`L34`](../../../../../../raw/code/jax/jax/_src/dlpack.py#L34)
- `SUPPORTED_DTYPES` — [`L44`](../../../../../../raw/code/jax/jax/_src/dlpack.py#L44)
- `SUPPORTED_DTYPES_SET` — [`L50`](../../../../../../raw/code/jax/jax/_src/dlpack.py#L50)
- `_DL_DEVICE_TO_PLATFORM` — [`L84`](../../../../../../raw/code/jax/jax/_src/dlpack.py#L84)

