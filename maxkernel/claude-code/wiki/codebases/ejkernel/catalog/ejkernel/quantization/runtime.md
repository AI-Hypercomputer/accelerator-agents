---
title: 'Module: ejkernel/quantization/runtime.py'
type: catalog
provenance: extracted
module: ejkernel/quantization/runtime.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.quantization.runtime`/
symbols:
  QuantRuntimeConfig: QuantRuntimeConfig#
  resolve_runtime_config: resolve_runtime_config().
  QuantRuntimeConfig.enable_u4_u8_fastpath: QuantRuntimeConfig#enable_u4_u8_fastpath.
  QuantRuntimeConfig.fastest_for_backend: QuantRuntimeConfig#fastest_for_backend().
  QuantRuntimeConfig.enable_threshold_codebook: QuantRuntimeConfig#enable_threshold_codebook.
  QuantRuntimeConfig.enable_parity_fallback: QuantRuntimeConfig#enable_parity_fallback.
  QuantRuntimeConfig.strict_shape_alignment: QuantRuntimeConfig#strict_shape_alignment.
  QuantRuntimeConfig.prefer_compute_dtype: QuantRuntimeConfig#prefer_compute_dtype.
  QuantRuntimeConfig.affine_metadata_dtype: QuantRuntimeConfig#affine_metadata_dtype.
  QuantRuntimeConfig.dequant_output_dtype: QuantRuntimeConfig#dequant_output_dtype.
  QuantRuntimeConfig.dequant_unpack_policy: QuantRuntimeConfig#dequant_unpack_policy.
  QuantRuntimeConfig.minifloat_decode_policy: QuantRuntimeConfig#minifloat_decode_policy.
  __all__: __all__.
---
# Module: [`ejkernel/quantization/runtime.py`](../../../../../../raw/code/ejkernel/ejkernel/quantization/runtime.py)

## Classes
### `QuantRuntimeConfig`
- def: [`ejkernel/quantization/runtime.py:34`](../../../../../../raw/code/ejkernel/ejkernel/quantization/runtime.py#L34)
- doc: Optional runtime policy for quantize/dequantize internals.
- signature: `class QuantRuntimeConfig:`
- members:
  - `fastest_for_backend(cls, *, backend: str | None = None, keep_fp32_output: bool = False)` — [`L95`](../../../../../../raw/code/ejkernel/ejkernel/quantization/runtime.py#L95) — Return an aggressive throughput profile tuned for the current backend.
  - `affine_metadata_dtype` — [`L89`](../../../../../../raw/code/ejkernel/ejkernel/quantization/runtime.py#L89)
  - `dequant_output_dtype` — [`L90`](../../../../../../raw/code/ejkernel/ejkernel/quantization/runtime.py#L90)
  - `dequant_unpack_policy` — [`L91`](../../../../../../raw/code/ejkernel/ejkernel/quantization/runtime.py#L91)
  - `enable_parity_fallback` — [`L86`](../../../../../../raw/code/ejkernel/ejkernel/quantization/runtime.py#L86)
  - `enable_threshold_codebook` — [`L85`](../../../../../../raw/code/ejkernel/ejkernel/quantization/runtime.py#L85)
  - `enable_u4_u8_fastpath` — [`L84`](../../../../../../raw/code/ejkernel/ejkernel/quantization/runtime.py#L84)
  - `minifloat_decode_policy` — [`L92`](../../../../../../raw/code/ejkernel/ejkernel/quantization/runtime.py#L92)
  - `prefer_compute_dtype` — [`L88`](../../../../../../raw/code/ejkernel/ejkernel/quantization/runtime.py#L88)
  - `strict_shape_alignment` — [`L87`](../../../../../../raw/code/ejkernel/ejkernel/quantization/runtime.py#L87)
- used by: [`dequantize`](_quants/quantizations.md#dequantize), [`prepack_quantized_array`](quantized_array.md#prepack_quantized_array), [`quantize_array`](quantized_array.md#quantize_array), [`quantize`](_quants/quantizations.md#quantize), [`_maybe_autotune_dequantize_runtime_cfg`](_quants/quantizations.md#_maybe_autotune_dequantize_runtime_cfg), [`_maybe_autotune_quantize_runtime_cfg`](_quants/quantizations.md#_maybe_autotune_quantize_runtime_cfg), [`_quantize_nvfp`](_quants/quantizations.md#_quantize_nvfp), [`_quantize_mxfp`](_quants/quantizations.md#_quantize_mxfp), [`dequantize`](quantized_array.md#QuantizedArray.dequantize), [`_dequantize_mxfp_bits`](_quants/quantizations.md#_dequantize_mxfp_bits), [`_dequantize_nvfp_bits`](_quants/quantizations.md#_dequantize_nvfp_bits), [`_dedupe_cfgs`](_quants/quantizations.md#_dedupe_cfgs), [`_quantize_nf4`](_quants/quantizations.md#_quantize_nf4), [`prepack_quantized_weights`](_quants/quantizations.md#prepack_quantized_weights), [`from_quantized`](quantized_array.md#QuantizedArray.from_quantized), [`_pack_group_codes`](_quants/quantizations.md#_pack_group_codes), [`_unpack_groups`](_quants/quantizations.md#_unpack_groups), [`_dequantize_nf4`](_quants/quantizations.md#_dequantize_nf4), [`_use_argmin_codebook`](_quants/quantizations.md#_use_argmin_codebook), [`_dequantize_affine_bits`](_quants/quantizations.md#_dequantize_affine_bits), [`_prefer_fast_unpack`](_quants/quantizations.md#_prefer_fast_unpack), [`_resolve_compute_dtype`](_quants/quantizations.md#_resolve_compute_dtype), [`_resolve_dequant_output_dtype`](_quants/quantizations.md#_resolve_dequant_output_dtype), [`_quantize_affine`](_quants/quantizations.md#_quantize_affine), [`resolve_runtime_config`](runtime.md#resolve_runtime_config), [`_AUTOTUNE_DEQUANT_CFG_CACHE`](_quants/quantizations.md#_AUTOTUNE_DEQUANT_CFG_CACHE._AUTOTUNE_DEQUANT_CFG_CACHE), [`_AUTOTUNE_QUANT_CFG_CACHE`](_quants/quantizations.md#_AUTOTUNE_QUANT_CFG_CACHE._AUTOTUNE_QUANT_CFG_CACHE), [`runtime_config`](quantized_array.md#QuantizedArray.runtime_config), [`_prefer_arith_minifloat_decode`](_quants/quantizations.md#_prefer_arith_minifloat_decode), [`_dequantize_candidate_cfgs`](_quants/quantizations.md#_dequantize_candidate_cfgs), [`_quantize_candidate_cfgs`](_quants/quantizations.md#_quantize_candidate_cfgs), [`_resolve_affine_metadata_dtype`](_quants/quantizations.md#_resolve_affine_metadata_dtype)

## Functions
- `resolve_runtime_config(config: QuantRuntimeConfig | None)` — [`L149`](../../../../../../raw/code/ejkernel/ejkernel/quantization/runtime.py#L149) — Return *config* unchanged, or a backend-tuned fast config when ``None``.

## Module values
- `__all__` — [`L164`](../../../../../../raw/code/ejkernel/ejkernel/quantization/runtime.py#L164)

