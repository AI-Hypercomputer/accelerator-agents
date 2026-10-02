---
title: 'Module: axlearn/common/flash_attention/splash_attention_mask.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/splash_attention_mask.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.splash_attention_mask`/
symbols:
  ComputableMask.mask_fn: ComputableMask#mask_fn.
  ComputableMask.__init__: ComputableMask#__init__().
  classify_blocks: classify_blocks().
  ComputableMask.mask_function: ComputableMask#mask_function().
  classify_blocks.compute_block_mask: classify_blocks().compute_block_mask().
  classify_blocks.compute_block_mask.classify_q_row: classify_blocks().compute_block_mask().classify_q_row().
  classify_blocks.compute_block_mask.classify_q_row.classify_kv_col: classify_blocks().compute_block_mask().classify_q_row().classify_kv_col().
  ComputableMask: ComputableMask#
  ComputableMask.__eq__: ComputableMask#__eq__().
  ComputableMask.__hash__: ComputableMask#__hash__().
---
# Module: [`axlearn/common/flash_attention/splash_attention_mask.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/splash_attention_mask.py)

## Classes
### `ComputableMask`  ·  implements/extends _ComputableMask
- def: [`axlearn/common/flash_attention/splash_attention_mask.py:19`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/splash_attention_mask.py#L19)
- doc: Computable mask for splash attention that supports custom mask functions.
- signature: `class ComputableMask(_ComputableMask):`
- members:
  - `__init__(self, shape: tuple[int, int], mask_fn: Callable[[Tensor, Tensor], Tensor], shard_count: int = 1)` — [`L32`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/splash_attention_mask.py#L32) — Initialize ComputableMask.
  - `mask_function(q_ids, kv_ids)` — [`L48`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/splash_attention_mask.py#L48) — Computes the attention mask using the provided mask_fn.
  - `mask_fn` — [`L30`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/splash_attention_mask.py#L30)
- protocol/private: `__eq__`[`L60`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/splash_attention_mask.py#L60), `__hash__`[`L70`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/splash_attention_mask.py#L70)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor)
- used by: [`_to_splash_mask`](tpu_attention.md#_to_splash_mask)

## Functions
- `classify_blocks(mask: MaskFnAttentionBias, q_positions: np.ndarray | jax.Array, block_shape: tuple[int, int], *, kv_seq_len: int, downcast_smem_data: bool = True, head_shards: int = 1)` — [`L81`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/splash_attention_mask.py#L81) — Computes block-level sparsity via tiled mask function evaluation. Classify Q/KV block pairs
- `classify_kv_col(kv_block_idx)` — [`L122`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/splash_attention_mask.py#L122)
- `classify_q_row(q_block_idx)` — [`L119`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/splash_attention_mask.py#L119)
- `compute_block_mask(q_seq)` — [`L118`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/splash_attention_mask.py#L118)

