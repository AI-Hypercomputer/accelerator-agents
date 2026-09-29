---
title: 'Module: jax/_src/numpy/array_constructors.py'
type: catalog
provenance: extracted
module: jax/_src/numpy/array_constructors.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.numpy.array_constructors`/
symbols:
  array: array().
  asarray: asarray().
  _get_platform: _get_platform().
  _make_string_array: _make_string_array().
  module_name: module_name.
  name: name.
  cuda_plugin_extension: cuda_plugin_extension.
  _convert_to_array_if_dtype_fails: _convert_to_array_if_dtype_fails().
  e: e.
  export: export.
  rocm_plugin_extension: rocm_plugin_extension.
  _supports_buffer_protocol: _supports_buffer_protocol().
  logger: logger.
  pkg_name: pkg_name.
  dist: dist.
---
# Module: [`jax/_src/numpy/array_constructors.py`](../../../../../../../raw/code/jax/jax/_src/numpy/array_constructors.py)

## Functions
- `_convert_to_array_if_dtype_fails(x: ArrayLike)` — [`L365`](../../../../../../../raw/code/jax/jax/_src/numpy/array_constructors.py#L365)
- `_get_platform(device_or_sharding: xc.Device | Sharding | None | str)` — [`L345`](../../../../../../../raw/code/jax/jax/_src/numpy/array_constructors.py#L345) — Get device_or_sharding platform or look up config.default_device.value.
- `_make_string_array(object: np.ndarray, dtype: DTypeLike | None = None, ndmin: int = 0, device: xc.Device | Sharding | None = None)` — [`L79`](../../../../../../../raw/code/jax/jax/_src/numpy/array_constructors.py#L79)
- `_supports_buffer_protocol(obj)` — [`L70`](../../../../../../../raw/code/jax/jax/_src/numpy/array_constructors.py#L70)
- `array(object: Any, dtype: DTypeLike | None = None, *args, copy: bool = True, order: str | None = "K", ndmin: int = 0, device: xc.Device | Sharding | None = None, out_sharding: NamedSharding | P | None = None)` — [`L108`](../../../../../../../raw/code/jax/jax/_src/numpy/array_constructors.py#L108) — Convert an object to a JAX array.
- `asarray(a: Any, dtype: DTypeLike | None = None, order: str | None = None, *, copy: bool | None = None, device: xc.Device | Sharding | None = None, out_sharding: NamedSharding | P | None = None)` — [`L375`](../../../../../../../raw/code/jax/jax/_src/numpy/array_constructors.py#L375) — Convert an object to a JAX array.

## Module values
- `cuda_plugin_extension` — [`L43`](../../../../../../../raw/code/jax/jax/_src/numpy/array_constructors.py#L43)
- `dist` — [`L55`](../../../../../../../raw/code/jax/jax/_src/numpy/array_constructors.py#L55)
- `e` — [`L66`](../../../../../../../raw/code/jax/jax/_src/numpy/array_constructors.py#L66)
- `export` — [`L39`](../../../../../../../raw/code/jax/jax/_src/numpy/array_constructors.py#L39)
- `logger` — [`L37`](../../../../../../../raw/code/jax/jax/_src/numpy/array_constructors.py#L37)
- `module_name` — [`L58`](../../../../../../../raw/code/jax/jax/_src/numpy/array_constructors.py#L58)
- `name` — [`L56`](../../../../../../../raw/code/jax/jax/_src/numpy/array_constructors.py#L56)
- `pkg_name` — [`L41`](../../../../../../../raw/code/jax/jax/_src/numpy/array_constructors.py#L41)
- `rocm_plugin_extension` — [`L52`](../../../../../../../raw/code/jax/jax/_src/numpy/array_constructors.py#L52)

