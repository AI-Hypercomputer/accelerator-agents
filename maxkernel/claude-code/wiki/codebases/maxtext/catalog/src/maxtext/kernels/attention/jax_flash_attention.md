---
title: 'Module: src/maxtext/kernels/attention/jax_flash_attention.py'
type: catalog
provenance: extracted
module: src/maxtext/kernels/attention/jax_flash_attention.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.kernels.attention.jax_flash_attention`/
symbols:
  flash_attention_block_masked: flash_attention_block_masked().
  flash_attention_block_masked.outer_loop_body.inner_loop_body: flash_attention_block_masked().outer_loop_body().inner_loop_body().
  flash_attention_block_masked.outer_loop_body: flash_attention_block_masked().outer_loop_body().
  SegmentIds: SegmentIds.
  flash_attention_block_masked.outer_loop_body.inner_loop_body.compute_attention_block: flash_attention_block_masked().outer_loop_body().inner_loop_body().compute_attention_block().
  flash_attention_block_masked.outer_loop_body.inner_loop_body.identity: flash_attention_block_masked().outer_loop_body().inner_loop_body().identity().
  mask_blocker: mask_blocker().
---
# Module: [`src/maxtext/kernels/attention/jax_flash_attention.py`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/jax_flash_attention.py)

## Functions
- `compute_attention_block(output, l, m)` — [`L142`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/jax_flash_attention.py#L142)
- `flash_attention_block_masked(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, segment_ids: SegmentIds | None, block_kv: int, block_q: int, mask: jnp.ndarray, mask_value: float, cap: Optional[float] = None, save_residuals: bool = False)` — [`L31`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/jax_flash_attention.py#L31) — Computes masked flash attention using block-sparse masking.
- `identity(output, l, m)` — [`L191`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/jax_flash_attention.py#L191) — A no-op identity function.
- `inner_loop_body(i, carried_inner)` — [`L134`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/jax_flash_attention.py#L134)
- `mask_blocker(mask: jnp.ndarray, block_q: int, block_kv: int)` — [`L234`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/jax_flash_attention.py#L234) — Creates a blocked mask from a full mask.
- `outer_loop_body(j, carried)` — [`L128`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/jax_flash_attention.py#L128)

## Module values
- `SegmentIds` — [`L22`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/jax_flash_attention.py#L22)

