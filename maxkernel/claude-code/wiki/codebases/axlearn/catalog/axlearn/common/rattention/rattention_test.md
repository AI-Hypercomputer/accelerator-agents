---
title: 'Module: axlearn/common/rattention/rattention_test.py'
type: catalog
provenance: extracted
module: axlearn/common/rattention/rattention_test.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.rattention.rattention_test`/R
symbols:
  ResidualLinearAttentionTest: esidualLinearAttentionTest#
  ResidualLinearAttentionTest.setUp: esidualLinearAttentionTest#setUp().
  ResidualLinearAttentionTest.tearDown: esidualLinearAttentionTest#tearDown().
  ResidualLinearAttentionTest.test_basic: esidualLinearAttentionTest#test_basic().
  ResidualLinearAttentionTest.test_extend: esidualLinearAttentionTest#test_extend().
  ResidualLinearAttentionTest.test_prefill: esidualLinearAttentionTest#test_prefill().
  RAttentionTest: AttentionTest#
  RAttentionTest.setUp: AttentionTest#setUp().
  RAttentionTest.tearDown: AttentionTest#tearDown().
  RAttentionTest.test_basic: AttentionTest#test_basic().
  RAttentionTest.test_against_attention: AttentionTest#test_against_attention().
  RAttentionTest.test_extend: AttentionTest#test_extend().
  RAttentionTest.test_prefill: AttentionTest#test_prefill().
  RAttentionTest.test_external_state: AttentionTest#test_external_state().
---
# Module: [`axlearn/common/rattention/rattention_test.py`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention_test.py)

## Classes
### `RAttentionTest`
- def: [`axlearn/common/rattention/rattention_test.py:279`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention_test.py#L279)
- signature: `class RAttentionTest(TestCase):`
- members:
  - `setUp(self)` — [`L280`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention_test.py#L280)
  - `tearDown(self)` — [`L289`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention_test.py#L289)
  - `test_against_attention(self, num_heads: int, num_kv_heads: int, sliding_window_size: int)` — [`L356`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention_test.py#L356) — Disabling residual linear attention to see if it matches FlashAttention.
  - `test_basic(self, num_heads: int, feat_fn: str, sliding_window_size: int, use_learned_init: bool, use_qk_scale: bool)` — [`L302`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention_test.py#L302)
  - `test_extend(self, num_heads: int, feat_fn: str, sliding_window_size: int, use_learned_init: bool, use_qk_scale: bool)` — [`L575`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention_test.py#L575)
  - `test_external_state(self, num_heads: int, feat_fn: str, sliding_window_size: int, use_learned_init: bool, use_qk_scale: bool)` — [`L770`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention_test.py#L770)
  - `test_prefill(self, num_heads: int, feat_fn: str, sliding_window_size: int, use_learned_init: bool, use_qk_scale: bool)` — [`L657`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention_test.py#L657)

### `ResidualLinearAttentionTest`
- def: [`axlearn/common/rattention/rattention_test.py:33`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention_test.py#L33)
- signature: `class ResidualLinearAttentionTest(TestCase):`
- members:
  - `setUp(self)` — [`L34`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention_test.py#L34)
  - `tearDown(self)` — [`L43`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention_test.py#L43)
  - `test_basic(self, num_heads: int, feat_fn: str, use_learned_init: bool, use_qk_scale: bool)` — [`L53`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention_test.py#L53)
  - `test_extend(self, num_heads: int, feat_fn: str, use_learned_init: bool, use_qk_scale: bool)` — [`L97`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention_test.py#L97)
  - `test_prefill(self, num_heads: int, feat_fn: str, use_learned_init: bool, use_qk_scale: bool)` — [`L173`](../../../../../../../raw/code/axlearn/axlearn/common/rattention/rattention_test.py#L173)

