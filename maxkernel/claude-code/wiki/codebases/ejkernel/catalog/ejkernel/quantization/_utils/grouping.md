---
title: 'Module: ejkernel/quantization/_utils/grouping.py'
type: catalog
provenance: extracted
module: ejkernel/quantization/_utils/grouping.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.quantization._utils.grouping`/_
symbols:
  _quantize_to_codebook: quantize_to_codebook().
  _quantize_to_codebook_argmin: quantize_to_codebook_argmin().
  _quantize_to_codebook_threshold: quantize_to_codebook_threshold().
  _quantize_to_codebook_threshold_from_map: quantize_to_codebook_threshold_from_map().
  _reshape_groups: reshape_groups().
  _require_bits: require_bits().
  _quantize_to_codebook_argmin._body: quantize_to_codebook_argmin()._body().
---
# Module: [`ejkernel/quantization/_utils/grouping.py`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/grouping.py)

## Functions
- `_body(i: int, state: tuple[jax.Array, jax.Array])` — [`L44`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/grouping.py#L44)
- `_quantize_to_codebook(values: jax.Array, codebook: jax.Array, *, use_argmin_fallback: bool = False, sorted_idx: jax.Array | None = None, boundaries: jax.Array | None = None)` — [`L128`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/grouping.py#L128) — Quantize values to nearest codebook entries.
- `_quantize_to_codebook_argmin(values: jax.Array, codebook: jax.Array)` — [`L27`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/grouping.py#L27) — Reference argmin-based codebook quantization.
- `_quantize_to_codebook_threshold(values: jax.Array, codebook: jax.Array)` — [`L56`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/grouping.py#L56) — Threshold bucketization codebook quantization.
- `_quantize_to_codebook_threshold_from_map(values: jax.Array, *, sorted_idx: jax.Array, boundaries: jax.Array)` — [`L83`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/grouping.py#L83) — Threshold quantization using precomputed sorted-index and boundary tensors.
- `_require_bits(bits: int, allowed: set[int])` — [`L200`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/grouping.py#L200) — Validate and return the bit-width parameter.
- `_reshape_groups(w: jax.Array, group_size: int)` — [`L166`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/grouping.py#L166) — Reshape tensor for group-wise quantization.

