---
title: 'Module: axlearn/common/flash_attention/attention_types.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/attention_types.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.attention_types`/FlashAttentionWithShardMapSpecs#
symbols:
  FlashAttentionWithShardMapSpecs: ''
  FlashAttentionWithShardMapSpecs.fn: fn.
  FlashAttentionWithShardMapSpecs.additional_in_specs: additional_in_specs.
  FlashAttentionWithShardMapSpecs.additional_kwargs: additional_kwargs.
---
# Module: [`axlearn/common/flash_attention/attention_types.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/attention_types.py)

## Classes
### `FlashAttentionWithShardMapSpecs`
- def: [`axlearn/common/flash_attention/attention_types.py:14`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/attention_types.py#L14)
- doc: Wraps a function and its additional arguments for use in shard_map.
- signature: `class FlashAttentionWithShardMapSpecs:`
- members:
  - `additional_in_specs` — [`L41`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/attention_types.py#L41)
  - `additional_kwargs` — [`L42`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/attention_types.py#L42)
  - `fn` — [`L40`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/attention_types.py#L40)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor)
- used by: [`_compute_attention`](layer.md#FlashAttention._compute_attention), [`build`](tpu_attention.md#TPUSplashAttentionWithAllGather.build), [`flash_attention_implementation`](utils.md#flash_attention_implementation), [`build`](common.md#BaseFlashAttention.build)

