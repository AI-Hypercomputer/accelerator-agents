---
title: 'Module: ejkernel/quantization/_quants/quantizations.py'
type: catalog
provenance: extracted
module: ejkernel/quantization/_quants/quantizations.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.quantization._quants.quantizations`/
symbols:
  dequantize: dequantize().
  quantize: quantize().
  _maybe_autotune_quantize_runtime_cfg: _maybe_autotune_quantize_runtime_cfg().
  _maybe_autotune_dequantize_runtime_cfg: _maybe_autotune_dequantize_runtime_cfg().
  _quantize_nvfp: _quantize_nvfp().
  quantized_matmul: quantized_matmul().
  _quantize_mxfp: _quantize_mxfp().
  _dequantize_mxfp_bits: _dequantize_mxfp_bits().
  _dequantize_nvfp_bits: _dequantize_nvfp_bits().
  _dedupe_cfgs: _dedupe_cfgs().
  _quantize_nf4: _quantize_nf4().
  prepack_quantized_weights: prepack_quantized_weights().
  _pack_group_codes: _pack_group_codes().
  _unpack_groups: _unpack_groups().
  _dequantize_nf4: _dequantize_nf4().
  _use_argmin_codebook: _use_argmin_codebook().
  _prefer_fast_unpack: _prefer_fast_unpack().
  _dequantize_affine_bits: _dequantize_affine_bits().
  _resolve_compute_dtype: _resolve_compute_dtype().
  _resolve_dequant_output_dtype: _resolve_dequant_output_dtype().
  clear_runtime_autotune_cache: clear_runtime_autotune_cache().
  runtime_autotune_cache_sizes: runtime_autotune_cache_sizes().
  _quantize_affine: _quantize_affine().
  _AUTOTUNE_QUANT_CFG_CACHE._AUTOTUNE_QUANT_CFG_CACHE: _AUTOTUNE_QUANT_CFG_CACHE._AUTOTUNE_QUANT_CFG_CACHE.
  _AUTOTUNE_DEQUANT_CFG_CACHE._AUTOTUNE_DEQUANT_CFG_CACHE: _AUTOTUNE_DEQUANT_CFG_CACHE._AUTOTUNE_DEQUANT_CFG_CACHE.
  _prefer_arith_minifloat_decode: _prefer_arith_minifloat_decode().
  _bench_ms: _bench_ms().
  _AUTOTUNE_CACHE_LOCK: _AUTOTUNE_CACHE_LOCK.
  _quant_autotune_key: _quant_autotune_key().
  _dequant_autotune_key: _dequant_autotune_key().
  _quantize_candidate_cfgs: _quantize_candidate_cfgs().
  _dequantize_candidate_cfgs: _dequantize_candidate_cfgs().
  _resolve_affine_metadata_dtype: _resolve_affine_metadata_dtype().
  _to_quant_layout: _to_quant_layout().
  _block_tree: _block_tree().
  _pad_axis_to_multiple: _pad_axis_to_multiple().
  _autotune_enabled: _autotune_enabled().
  _is_tracing_tree: _is_tracing_tree().
  _autotune_min_gain: _autotune_min_gain().
  _resolve_matmul_align_multiple: _resolve_matmul_align_multiple().
---
# Module: [`ejkernel/quantization/_quants/quantizations.py`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py)

## Functions
- `_autotune_enabled()` — [`L84`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L84) — Check whether runtime autotuning is enabled. — documented in [ejkernel-quantization-_quants-quantizations](../../../../concepts/ejkernel-quantization-_quants-quantizations.md)
- `_autotune_min_gain()` — [`L133`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L133) — Return minimum relative win required to replace the baseline config. — documented in [ejkernel-quantization-_quants-quantizations](../../../../concepts/ejkernel-quantization-_quants-quantizations.md)
- `_bench_ms(fn, args: tuple, *, warmup: int | None = None, iters: int | None = None)` — [`L105`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L105) — Measure median runtime in milliseconds for a jitted callable. — documented in [ejkernel-quantization-_quants-quantizations](../../../../concepts/ejkernel-quantization-_quants-quantizations.md)
- `_block_tree(tree)` — [`L98`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L98) — Block on all array leaves in *tree*.
- `_dedupe_cfgs(candidates: list[QuantRuntimeConfig])` — [`L138`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L138) — Deduplicate candidate runtime configs while preserving order. — documented in [ejkernel-quantization-_quants-quantizations](../../../../concepts/ejkernel-quantization-_quants-quantizations.md)
- `_dequant_autotune_key(w_q: jax.Array, scales: jax.Array, zeros: jax.Array | None, *, mode: QuantizationMode, group_size: int, bits: int, axis: QuantizationAxis)` — [`L181`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L181) — Build cache key for dequantize autotuning. — documented in [ejkernel-quantization-_quants-quantizations](../../../../concepts/ejkernel-quantization-_quants-quantizations.md)
- `_dequantize_affine_bits(w_q: jax.Array, scales: jax.Array, zeros: jax.Array, *, group_size: int, bits: int, runtime_cfg: QuantRuntimeConfig)` — [`L865`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L865) — Affine dequantization specialized by static bit-width. — documented in [ejkernel-quantization-_quants-quantizations](../../../../concepts/ejkernel-quantization-_quants-quantizations.md)
- `_dequantize_candidate_cfgs(base: QuantRuntimeConfig)` — [`L217`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L217) — Build dequantize autotune candidates around *base*. — documented in [ejkernel-quantization-_quants-quantizations](../../../../concepts/ejkernel-quantization-_quants-quantizations.md)
- `_dequantize_mxfp_bits(w_q: jax.Array, scales: jax.Array, *, group_size: int, bits: int, runtime_cfg: QuantRuntimeConfig)` — [`L935`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L935) — MXFP dequantization specialized for 4-bit (E2M1) or 8-bit (E4M3) payloads.
- `_dequantize_nf4(w_q: jax.Array, scales: jax.Array, *, group_size: int, runtime_cfg: QuantRuntimeConfig)` — [`L902`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L902) — NF4 dequantization: table lookup then scale by per-group absmax.
- `_dequantize_nvfp_bits(w_q: jax.Array, scales: jax.Array, *, group_size: int, bits: int, runtime_cfg: QuantRuntimeConfig)` — [`L990`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L990) — NVFP dequantization specialized for 4-bit (E2M1) or 8-bit (E4M3) payloads.
- `_is_tracing_tree(tree)` — [`L90`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L90) — Return True if any leaf in *tree* is a JAX tracer.
- `_maybe_autotune_dequantize_runtime_cfg(w_q: jax.Array, scales: jax.Array, zeros: jax.Array | None, *, mode: QuantizationMode, group_size: int, bits: int, axis: QuantizationAxis, base_cfg: QuantRuntimeConfig)` — [`L295`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L295) — Pick and cache a fast dequantize runtime config for this shape/mode cell. — documented in [ejkernel-quantization-_quants-quantizations](../../../../concepts/ejkernel-quantization-_quants-quantizations.md)
- `_maybe_autotune_quantize_runtime_cfg(w: jax.Array, *, mode: QuantizationMode, group_size: int, bits: int, axis: QuantizationAxis, base_cfg: QuantRuntimeConfig)` — [`L249`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L249) — Pick and cache a fast quantize runtime config for this shape/mode cell. — documented in [ejkernel-quantization-_quants-quantizations](../../../../concepts/ejkernel-quantization-_quants-quantizations.md)
- `_pack_group_codes(w_layout: jax.Array, q: jax.Array, bits: int, runtime_cfg: QuantRuntimeConfig)` — [`L441`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L441) — Pack grouped quantization codes into uint32 storage layout.
- `_pad_axis_to_multiple(x: jax.Array, axis: int, multiple: int)` — [`L1253`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L1253) — Pad *x* along *axis* to the nearest multiple, returning ``(padded, original_size)``.
- `_prefer_arith_minifloat_decode(runtime_cfg: QuantRuntimeConfig)` — [`L790`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L790) — Return ``True`` to use arithmetic decode instead of table lookup.
- `_prefer_fast_unpack(*, runtime_cfg: QuantRuntimeConfig, mode: QuantizationMode, bits: int, batch_size_hint: int)` — [`L747`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L747) — Return ``True`` to select the fast grouped unpack path.
- `_quant_autotune_key(w: jax.Array, *, mode: QuantizationMode, group_size: int, bits: int, axis: QuantizationAxis)` — [`L160`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L160) — Build cache key for quantize autotuning. — documented in [ejkernel-quantization-_quants-quantizations](../../../../concepts/ejkernel-quantization-_quants-quantizations.md)
- `_quantize_affine(w_layout: jax.Array, w_groups: jax.Array, *, bits: int, runtime_cfg: QuantRuntimeConfig)` — [`L482`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L482) — Quantize groups with affine (scale + zero-point) metadata.
- `_quantize_candidate_cfgs(base: QuantRuntimeConfig, *, mode: QuantizationMode)` — [`L207`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L207) — Build quantize autotune candidates around *base*.
- `_quantize_mxfp(w_layout: jax.Array, w_groups: jax.Array, *, bits: int, runtime_cfg: QuantRuntimeConfig)` — [`L571`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L571) — Quantize groups using MXFP shared-exponent scaling.
- `_quantize_nf4(w_layout: jax.Array, w_groups: jax.Array, *, bits: int, runtime_cfg: QuantRuntimeConfig)` — [`L526`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L526) — Quantize groups into NF4 codebook indices and per-group absmax scales.
- `_quantize_nvfp(w_layout: jax.Array, w_groups: jax.Array, *, bits: int, runtime_cfg: QuantRuntimeConfig)` — [`L627`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L627) — Quantize groups using NVFP value codes and E4M3 quantized scales.
- `_resolve_affine_metadata_dtype(w_layout: jax.Array, runtime_cfg: QuantRuntimeConfig)` — [`L413`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L413) — Resolve storage dtype for affine scales/zeros metadata.
- `_resolve_compute_dtype(runtime_cfg: QuantRuntimeConfig)` — [`L697`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L697) — Resolve arithmetic dtype for dequantization math.
- `_resolve_dequant_output_dtype(runtime_cfg: QuantRuntimeConfig, compute_dtype: jnp.dtype)` — [`L720`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L720) — Resolve output dtype for dequantized tensors.
- `_resolve_matmul_align_multiple(align_multiple: int | None)` — [`L1228`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L1228) — Resolve K/N padding multiple for the dense dequantize-then-matmul path.
- `_to_quant_layout(w: jax.Array, axis: QuantizationAxis)` — [`L388`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L388) — Map logical weight layout to quantization/runtime layout.
- `_unpack_groups(w_q: jax.Array, scales: jax.Array, *, mode: QuantizationMode, group_size: int, bits: int, runtime_cfg: QuantRuntimeConfig)` — [`L818`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L818) — Unpack bitpacked codes and reshape to ``(..., n_groups, group_size)``.
- `_use_argmin_codebook(runtime_cfg: QuantRuntimeConfig)` — [`L472`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L472) — Return ``True`` when the slow argmin codebook path should be used.
- `clear_runtime_autotune_cache()` — [`L71`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L71) — Clear cached runtime autotune decisions for quantize/dequantize.
- `dequantize(w_q: jax.Array, scales: jax.Array, zeros: jax.Array | None = None, *, group_size: int | None = None, bits: int | None = None, mode: QuantizationMode = "affine", axis: QuantizationAxis = "row", runtime_config: QuantRuntimeConfig | None = None)` — [`L1116`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L1116) — Dequantize packed uint32 weights back to floating-point values. — documented in [ejkernel-quantization-_quants-quantizations](../../../../concepts/ejkernel-quantization-_quants-quantizations.md)
- `prepack_quantized_weights(w: jax.Array,, group_size: int | None = None, bits: int | None = None, mode: QuantizationMode = "affine", transpose: bool = True, axis: QuantizationAxis | None = None, runtime_config: QuantRuntimeConfig | None = None)` — [`L1361`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L1361) — Prepack logical ``(out_features, in_features)`` weights for quantized matmul.
- `quantize(w: jax.Array,, group_size: int | None = None, bits: int | None = None, mode: QuantizationMode = "affine", axis: QuantizationAxis = "row", runtime_config: QuantRuntimeConfig | None = None)` — [`L1046`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L1046) — Quantize weights into packed uint32 codes with per-group scaling. — documented in [ejkernel-quantization-_quants-quantizations](../../../../concepts/ejkernel-quantization-_quants-quantizations.md)
- `quantized_matmul(x: jax.Array, w: jax.Array,, scales: jax.Array, zeros: jax.Array | None = None, transpose: bool = False, group_size: int | None = None, bits: int | None = None, mode: QuantizationMode = "affine", axis: QuantizationAxis | None = None, align_multiple: int | None = None)` — [`L1282`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L1282) — Dense reference quantized matrix multiplication via dequantize-then-matmul.
- `runtime_autotune_cache_sizes()` — [`L78`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L78) — Return ``(quantize_cells, dequantize_cells)`` cached by runtime autotune.

## Module values
- `_AUTOTUNE_CACHE_LOCK` — [`L66`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L66) — documented in [ejkernel-quantization-_quants-quantizations](../../../../concepts/ejkernel-quantization-_quants-quantizations.md)
- `_AUTOTUNE_DEQUANT_CFG_CACHE` — [`L68`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L68) — documented in [ejkernel-quantization-_quants-quantizations](../../../../concepts/ejkernel-quantization-_quants-quantizations.md)
- `_AUTOTUNE_QUANT_CFG_CACHE` — [`L67`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_quants/quantizations.py#L67) — documented in [ejkernel-quantization-_quants-quantizations](../../../../concepts/ejkernel-quantization-_quants-quantizations.md)

