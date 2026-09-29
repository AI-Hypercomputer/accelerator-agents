---
title: 'Module: python/sgl_jax/srt/configs/dtype_config.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/configs/dtype_config.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.configs.dtype_config`/
symbols:
  DtypeConfig.config_dict: DtypeConfig#config_dict.
  DtypeConfig.default_dtype: DtypeConfig#default_dtype.
  DtypeConfig: DtypeConfig#
  DtypeConfig.get_config: DtypeConfig#get_config().
  STR_DTYPE_TO_JAX_DTYPE: STR_DTYPE_TO_JAX_DTYPE.
  DtypeConfig._parse_dict: DtypeConfig#_parse_dict().
  DtypeConfig.__repr__: DtypeConfig#__repr__().
  DtypeConfig.get_dtype: DtypeConfig#get_dtype().
  DtypeConfig.get_optional_dtype: DtypeConfig#get_optional_dtype().
  DtypeConfig.__init__: DtypeConfig#__init__().
---
# Module: [`python/sgl_jax/srt/configs/dtype_config.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/dtype_config.py)

## Classes
### `DtypeConfig`
- def: [`python/sgl_jax/srt/configs/dtype_config.py:15`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/dtype_config.py#L15)
- signature: `class DtypeConfig:`
- members:
  - `__repr__(self)` — [`L31`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/dtype_config.py#L31) — Return a concise representation including config_dict and default_dtype.
  - `_parse_dict(self, d: dict[str, Any])` — [`L38`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/dtype_config.py#L38) — Recursively parses a dictionary, converting string dtypes to jnp.dtype.
  - `get_config(self, key: str)` — [`L54`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/dtype_config.py#L54) — Returns a child config covering the sub-dictionary, preserving the default.
  - `get_dtype(self, key: str)` — [`L60`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/dtype_config.py#L60) — Returns the specific dtype, or falls back to the default.
  - `get_optional_dtype(self, key: str)` — [`L73`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/dtype_config.py#L73) — Returns the specific dtype, or None if the key is not explicitly set.
  - `config_dict` — [`L23`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/dtype_config.py#L23)
  - `default_dtype` — [`L27`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/dtype_config.py#L27)
- protocol/private: `__init__`[`L16`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/dtype_config.py#L16)
- uses (calls/refs, reference-scoped): [`STR_DTYPE_TO_JAX_DTYPE`](dtype_config.md#STR_DTYPE_TO_JAX_DTYPE)
- used by: [`dtype_config`](model_config.md#ModelConfig.dtype_config), [`dtype`](model_config.md#ModelConfig.dtype), [`__init__`](model_config.md#ModelConfig.__init__), [`_has_softmax_dtype`](../model_executor/model_runner.md#ModelRunner._has_softmax_dtype)

## Module values
- `STR_DTYPE_TO_JAX_DTYPE` — [`L5`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/dtype_config.py#L5)

