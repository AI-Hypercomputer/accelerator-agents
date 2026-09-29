---
title: 'Module: python/sgl_jax/srt/constrained/bitmask_ops.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/constrained/bitmask_ops.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.constrained.bitmask_ops`/
symbols:
  apply_token_bitmask: apply_token_bitmask().
  allocate_token_bitmask: allocate_token_bitmask().
  unpack_bitmask: unpack_bitmask().
  fill_token_bitmask: fill_token_bitmask().
  unpack_bitmask.unpack_batch_item: unpack_bitmask().unpack_batch_item().
---
# Module: [`python/sgl_jax/srt/constrained/bitmask_ops.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/bitmask_ops.py)

## Functions
- `allocate_token_bitmask(batch_size: int, vocab_size: int)` — [`L9`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/bitmask_ops.py#L9) — Allocate a token bitmask array.
- `apply_token_bitmask(logits: jax.Array, vocab_mask: jax.Array)` — [`L45`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/bitmask_ops.py#L45) — Apply token bitmask to logits.
- `fill_token_bitmask(matcher: LLInterpreter, vocab_mask: np.ndarray, batch_idx: int)` — [`L23`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/bitmask_ops.py#L23) — Fill the bitmask for a specific batch index using llguidance matcher.
- `unpack_batch_item(mask_row)` — [`L99`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/bitmask_ops.py#L99)
- `unpack_bitmask(vocab_mask: jax.Array)` — [`L87`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/bitmask_ops.py#L87) — Unpack int32 bitmask to boolean array (no dynamic slicing).

