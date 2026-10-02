---
title: 'Module: tpu_inference/layers/common/binary_search.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/common/binary_search.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.common.binary_search`/
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
# Module: [`tpu_inference/layers/common/binary_search.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/binary_search.py)

## Functions
- `_monotonic_int32_to_float32(x: int)` — [`L100`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/binary_search.py#L100) — Converts an int32 to a float32 with consistent ordering.
- `_monotonic_int32_to_float32_bit_pattern(x: int)` — [`L70`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/binary_search.py#L70) — Converts an int32 to a float32 bit pattern with consistent ordering.
- `float32_bsearch(batch_shape, predicate)` — [`L117`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/binary_search.py#L117) — Binary search on finite float32 numbers.
- `int32_bsearch(batch_shape: Sequence[int], predicate: Callable[[jnp.ndarray], jnp.ndarray])` — [`L27`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/binary_search.py#L27) — Batched binary search over int32 values.
- `int32_predicate(x)` — [`L136`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/binary_search.py#L136)
- `loop_body(i, current_bits)` — [`L57`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/binary_search.py#L57)
- `predicate(threshold)` — [`L203`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/binary_search.py#L203)
- `predicate(threshold)` — [`L276`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/binary_search.py#L276)
- `topk_mask(x: jnp.ndarray, k: int, replace_val: jnp.ndarray)` — [`L164`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/binary_search.py#L164) — Sets everything to replace_val, except the top k values per batch element.
- `topp_mask(logits: jnp.ndarray, p: float, replace_val: jnp.ndarray)` — [`L227`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/binary_search.py#L227) — Applies top-p masking to logits.

