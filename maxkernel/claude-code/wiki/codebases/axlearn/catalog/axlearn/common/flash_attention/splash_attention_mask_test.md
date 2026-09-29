---
title: 'Module: axlearn/common/flash_attention/splash_attention_mask_test.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/splash_attention_mask_test.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.splash_attention_mask_test`/SplashAttentionMaskTest#
symbols:
  SplashAttentionMaskTest: ''
  SplashAttentionMaskTest.test_causal_mask_fn: test_causal_mask_fn().
  SplashAttentionMaskTest.test_sliding_window_causal_mask_fn: test_sliding_window_causal_mask_fn().
  SplashAttentionMaskTest.test_classify_blocks: test_classify_blocks().
  SplashAttentionMaskTest._compare_masks: _compare_masks().
---
# Module: [`axlearn/common/flash_attention/splash_attention_mask_test.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/splash_attention_mask_test.py)

## Classes
### `SplashAttentionMaskTest`
- def: [`axlearn/common/flash_attention/splash_attention_mask_test.py:25`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/splash_attention_mask_test.py#L25)
- signature: `class SplashAttentionMaskTest(TestCase):`
- members:
  - `test_causal_mask_fn(self, block_size, shape)` — [`L39`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/splash_attention_mask_test.py#L39) — Test ComputableMask with causal_mask function from attention_bias.py.
  - `test_classify_blocks(self, mask_fn_factory, block_shape, shape, q_offsets, downcast_smem_data)` — [`L89`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/splash_attention_mask_test.py#L89) — Test classify_blocks block_mask against process_mask as reference.
  - `test_sliding_window_causal_mask_fn(self, block_size, shape, window_size)` — [`L65`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/splash_attention_mask_test.py#L65) — Test ComputableMask with sliding_window_causal_mask from attention_bias.py.
- protocol/private: `_compare_masks`[`L153`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/splash_attention_mask_test.py#L153)

