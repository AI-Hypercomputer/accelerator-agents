---
title: 'Module: python/sgl_jax/srt/layers/binary_search.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/layers/binary_search.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.layers.binary_search`/
symbols:
  float32_bsearch: float32_bsearch().
  topk_mask: topk_mask().
  topp_mask: topp_mask().
  int32_bsearch: int32_bsearch().
  _monotonic_int32_to_float32: _monotonic_int32_to_float32().
  float32_bsearch.int32_predicate: float32_bsearch().int32_predicate().
  _monotonic_int32_to_float32_bit_pattern: _monotonic_int32_to_float32_bit_pattern().
  int32_bsearch.loop_body: int32_bsearch().loop_body().
  topk_mask.predicate: topk_mask().predicate().
  topp_mask.predicate: topp_mask().predicate().
---
# Module: [`python/sgl_jax/srt/layers/binary_search.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/binary_search.py)

## Functions
- `_monotonic_int32_to_float32(x: int)` — [`L100`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/binary_search.py#L100) — Converts an int32 to a float32 with consistent ordering.
- `_monotonic_int32_to_float32_bit_pattern(x: int)` — [`L70`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/binary_search.py#L70) — Converts an int32 to a float32 bit pattern with consistent ordering.
- `float32_bsearch(batch_shape, predicate)` — [`L117`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/binary_search.py#L117) — Binary search on finite float32 numbers.
- `int32_bsearch(batch_shape: Sequence[int], predicate: Callable[[jax.Array], jax.Array])` — [`L30`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/binary_search.py#L30) — Batched binary search over int32 values.
- `int32_predicate(x)` — [`L136`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/binary_search.py#L136)
- `loop_body(i, current_bits)` — [`L58`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/binary_search.py#L58)
- `predicate(threshold)` — [`L202`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/binary_search.py#L202)
- `predicate(threshold)` — [`L274`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/binary_search.py#L274)
- `topk_mask(x: jax.Array, k: jax.Array, replace_val: float)` — [`L163`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/binary_search.py#L163) — Sets everything to replace_val, except the top k values per batch element.
- `topp_mask(logits: jax.Array, p: jax.Array, replace_val: float)` — [`L226`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/binary_search.py#L226) — Applies top-p masking to logits.

