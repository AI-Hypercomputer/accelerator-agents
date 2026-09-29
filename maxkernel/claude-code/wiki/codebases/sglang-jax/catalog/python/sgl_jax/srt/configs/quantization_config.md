---
title: 'Module: python/sgl_jax/srt/configs/quantization_config.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/configs/quantization_config.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.configs.quantization_config`/
symbols:
  QuantizationConfig.to_dict: QuantizationConfig#to_dict().
  QuantizationConfig.from_yaml: QuantizationConfig#from_yaml().
  QuantizationConfig: QuantizationConfig#
  QuantizationConfig.has_moe_quantization: QuantizationConfig#has_moe_quantization().
  QuantizationConfig.ignored_layers: QuantizationConfig#ignored_layers.
  DTYPE_MAP: DTYPE_MAP.
  QuantizationConfig.moe_weight_dtype: QuantizationConfig#moe_weight_dtype.
  QuantizationConfig.moe_activation_dtype: QuantizationConfig#moe_activation_dtype.
  QuantizationConfig.is_static_checkpoint: QuantizationConfig#is_static_checkpoint.
  QuantizationConfig.from_path: QuantizationConfig#from_path().
  QuantizationConfig.get_linear_rules: QuantizationConfig#get_linear_rules().
  QuantizationConfig.linear_rules: QuantizationConfig#linear_rules.
  QuantizationConfig.weight_block_size: QuantizationConfig#weight_block_size.
  _str_to_dtype: _str_to_dtype().
  _resolve_config_path: _resolve_config_path().
  normalize_weight_block_size: normalize_weight_block_size().
  QuantizationConfig.get_moe_weight_dtype: QuantizationConfig#get_moe_weight_dtype().
  QuantizationConfig.get_moe_activation_dtype: QuantizationConfig#get_moe_activation_dtype().
  QuantizationConfig.has_linear_quantization: QuantizationConfig#has_linear_quantization().
  BUILTIN_CONFIG_PATH: BUILTIN_CONFIG_PATH.
  QuantizationConfig.allow_narrow_n_blockwise: QuantizationConfig#allow_narrow_n_blockwise.
---
# Module: [`python/sgl_jax/srt/configs/quantization_config.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/quantization_config.py)

## Classes
### `QuantizationConfig`
- def: [`python/sgl_jax/srt/configs/quantization_config.py:90`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/quantization_config.py#L90)
- doc: Quantization configuration with explicit settings (no fallbacks).
- signature: `class QuantizationConfig:`
- members:
  - `from_path(cls, config_path: str | None)` — [`L196`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/quantization_config.py#L196) — Load quantization config from a path.
  - `from_yaml(cls, yaml_path: str)` — [`L131`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/quantization_config.py#L131) — Load quantization config from a YAML file.
  - `get_linear_rules(self)` — [`L217`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/quantization_config.py#L217) — Get the quantization rules for linear layer quantization.
  - `get_moe_activation_dtype(self)` — [`L213`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/quantization_config.py#L213) — Get the dtype for MoE activation quantization.
  - `get_moe_weight_dtype(self)` — [`L209`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/quantization_config.py#L209) — Get the dtype for MoE weight quantization.
  - `has_linear_quantization(self)` — [`L225`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/quantization_config.py#L225) — Check if linear layer quantization is configured.
  - `has_moe_quantization(self)` — [`L221`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/quantization_config.py#L221) — Check if MoE quantization is configured.
  - `to_dict(self)` — [`L110`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/quantization_config.py#L110)
  - `allow_narrow_n_blockwise` — [`L108`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/quantization_config.py#L108)
  - `ignored_layers` — [`L106`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/quantization_config.py#L106)
  - `is_static_checkpoint` — [`L105`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/quantization_config.py#L105)
  - `linear_rules` — [`L102`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/quantization_config.py#L102)
  - `moe_activation_dtype` — [`L104`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/quantization_config.py#L104)
  - `moe_weight_dtype` — [`L103`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/quantization_config.py#L103)
  - `weight_block_size` — [`L107`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/quantization_config.py#L107)
- uses (calls/refs, reference-scoped): [`_str_to_dtype`](quantization_config.md#_str_to_dtype), [`_resolve_config_path`](quantization_config.md#_resolve_config_path), [`normalize_weight_block_size`](quantization_config.md#normalize_weight_block_size)
- used by: [`model`](../model_executor/model_runner.md#ModelRunner.model), [`_get_model`](../model_loader/loader.md#JAXModelLoader._get_model), [`_resolve_quantization_config`](model_config.md#ModelConfig._resolve_quantization_config), [`apply_linear_quantization`](../utils/quantization/quantization_utils.md#apply_linear_quantization), [`quantization_config`](model_config.md#ModelConfig.quantization_config), [`apply_moe_quantization`](../utils/quantization/quantization_utils.md#apply_moe_quantization), [`sliding_window`](model_config.md#ModelConfig.sliding_window), [`_get_hf_quant_config`](model_config.md#ModelConfig._get_hf_quant_config)

## Functions
- `_resolve_config_path(config_path: str)` — [`L40`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/quantization_config.py#L40) — Resolve a config path, checking both absolute and built-in locations.
- `_str_to_dtype(dtype_str: str | None)` — [`L31`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/quantization_config.py#L31) — Convert a string dtype name to a JAX numpy dtype.
- `normalize_weight_block_size(weight_block_size: list[int] | tuple[int, int] | None)` — [`L59`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/quantization_config.py#L59) — Validate and canonicalize ``weight_block_size`` to ``(block_n, block_k)``.

## Module values
- `BUILTIN_CONFIG_PATH` — [`L26`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/quantization_config.py#L26)
- `DTYPE_MAP` — [`L16`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/quantization_config.py#L16)

