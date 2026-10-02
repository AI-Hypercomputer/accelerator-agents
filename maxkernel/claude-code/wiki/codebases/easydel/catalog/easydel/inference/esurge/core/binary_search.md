---
title: 'Module: easydel/inference/esurge/core/binary_search.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/core/binary_search.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.core.binary_search`/
symbols:
  apply_float32_bsearch: apply_float32_bsearch().
  apply_topk_mask: apply_topk_mask().
  apply_topp_mask: apply_topp_mask().
  apply_min_p_mask: apply_min_p_mask().
  apply_penalties: apply_penalties().
  apply_topk_mask_bf16: apply_topk_mask_bf16().
  apply_topp_mask_bf16: apply_topp_mask_bf16().
  int32_bsearch: int32_bsearch().
  _monotonic_int32_to_float32: _monotonic_int32_to_float32().
  apply_float32_bsearch.int32_predicate: apply_float32_bsearch().int32_predicate().
  _monotonic_int32_to_float32_bit_pattern: _monotonic_int32_to_float32_bit_pattern().
  int32_bsearch.loop_body: int32_bsearch().loop_body().
  apply_topk_mask_bf16.predicate: apply_topk_mask_bf16().predicate().
  apply_topk_mask.predicate: apply_topk_mask().predicate().
  apply_topp_mask_bf16.predicate: apply_topp_mask_bf16().predicate().
  apply_topp_mask.predicate: apply_topp_mask().predicate().
---
# Module: [`easydel/inference/esurge/core/binary_search.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/binary_search.py)

## Functions
- `_monotonic_int32_to_float32(x: int)` — [`L102`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/binary_search.py#L102) — Convert int32 to float32 preserving total order.
- `_monotonic_int32_to_float32_bit_pattern(x: int)` — [`L81`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/binary_search.py#L81) — Convert int32 to float32 bit pattern preserving IEEE 754 total order.
- `apply_float32_bsearch(batch_shape, predicate)` — [`L115`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/binary_search.py#L115) — Perform batched binary search over finite float32 values.
- `apply_min_p_mask(logits: jax.Array, sampling_metadata: SamplingMetadata)` — [`L351`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/binary_search.py#L351) — Apply min-p masking to logits.
- `apply_penalties(logits: jax.Array, sampling_metadata: SamplingMetadata)` — [`L376`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/binary_search.py#L376) — Apply linear penalties to logits.
- `apply_topk_mask(x: jnp.ndarray, k: jax.Array, replace_val: float)` — [`L204`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/binary_search.py#L204) — Apply top-k masking using binary search.
- `apply_topk_mask_bf16(x: jnp.ndarray, k: jax.Array, replace_val: float)` — [`L152`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/binary_search.py#L152) — Apply top-k masking with bfloat16 optimization.
- `apply_topp_mask(logits: jnp.ndarray, p: jax.Array, replace_val: float)` — [`L302`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/binary_search.py#L302) — Apply top-p (nucleus) sampling mask using binary search.
- `apply_topp_mask_bf16(logits: jnp.ndarray, p: jax.Array, replace_val: float)` — [`L247`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/binary_search.py#L247) — Apply top-p (nucleus) masking with bfloat16 optimization.
- `int32_bsearch(batch_shape: Sequence[int], predicate: Callable[[jnp.ndarray], jnp.ndarray])` — [`L39`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/binary_search.py#L39) — Perform batched binary search over int32 bit patterns.
- `int32_predicate(x)` — [`L138`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/binary_search.py#L138) — Convert int32 bit pattern to float32 and evaluate the user predicate.
- `loop_body(i, current_bits)` — [`L68`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/binary_search.py#L68) — Test bit `i` and set it in the result based on the predicate.
- `predicate(threshold)` — [`L186`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/binary_search.py#L186) — Return True if at least k values exceed the negated threshold.
- `predicate(threshold)` — [`L234`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/binary_search.py#L234) — Return True if at least k values exceed the negated threshold (int32 variant).
- `predicate(threshold)` — [`L282`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/binary_search.py#L282) — Return True if cumulative probability above the threshold is less than p.
- `predicate(threshold)` — [`L337`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/binary_search.py#L337) — Return True if cumulative probability above the threshold is less than p (logits variant).

