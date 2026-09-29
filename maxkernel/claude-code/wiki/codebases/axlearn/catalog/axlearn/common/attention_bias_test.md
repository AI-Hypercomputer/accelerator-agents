---
title: 'Module: axlearn/common/attention_bias_test.py'
type: catalog
provenance: extracted
module: axlearn/common/attention_bias_test.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.attention_bias_test`/
symbols:
  MaskTest: MaskTest#
  MaskTest.test_sliding_window_mask: MaskTest#test_sliding_window_mask().
  MaskTest.test_sliding_window_mask_with_time_step: MaskTest#test_sliding_window_mask_with_time_step().
  MaskTest.test_left_right_window_mask: MaskTest#test_left_right_window_mask().
  MaskTest.test_key_limit_mask: MaskTest#test_key_limit_mask().
  AttentionBiasTest: AttentionBiasTest#
  AttentionBiasTest.test_has_bias: AttentionBiasTest#test_has_bias().
  AttentionBiasTest.test_causal_attention_bias: AttentionBiasTest#test_causal_attention_bias().
  AttentionBiasTest.test_sliding_window_attention_bias: AttentionBiasTest#test_sliding_window_attention_bias().
  AttentionBiasTest.test_left_right_window_attention_bias: AttentionBiasTest#test_left_right_window_attention_bias().
  AttentionBiasTest.test_zero_attention_bias: AttentionBiasTest#test_zero_attention_bias().
  AttentionBiasTest.test_base_attention_bias_value: AttentionBiasTest#test_base_attention_bias_value().
  AttentionBiasTest.test_base_attention_bias_and_residual: AttentionBiasTest#test_base_attention_bias_and_residual().
  AttentionBiasTest.test_composite_attention_has_bias: AttentionBiasTest#test_composite_attention_has_bias().
  AttentionBiasTest.test_bias_and_residual_has_bias: AttentionBiasTest#test_bias_and_residual_has_bias().
  AttentionBiasTest.test_composite_attention_bias_zero: AttentionBiasTest#test_composite_attention_bias_zero().
  AttentionBiasTest.test_composite_attention_bias: AttentionBiasTest#test_composite_attention_bias().
  AttentionBiasTest.test_bias_and_residual_repeated_call: AttentionBiasTest#test_bias_and_residual_repeated_call().
  AttentionBiasTest.test_split: AttentionBiasTest#test_split().
  AttentionBiasTest.test_split_subsets: AttentionBiasTest#test_split_subsets().
  AttentionBiasTest.test_tensor_attention_bias: AttentionBiasTest#test_tensor_attention_bias().
  AttentionBiasTest.test_segment_id_attention_bias: AttentionBiasTest#test_segment_id_attention_bias().
  AttentionBiasTest.test_mask_fn_attention_bias_from_sequence: AttentionBiasTest#test_mask_fn_attention_bias_from_sequence().
  AttentionBiasTest.test_mask_fn_attention_bias: AttentionBiasTest#test_mask_fn_attention_bias().
  AttentionBiasTest.test_mask_fn_attention_bias_target_positions_ndim: AttentionBiasTest#test_mask_fn_attention_bias_target_positions_ndim().
  AttentionBiasTest.test_mask_fn_attention_bias_with_target_positions: AttentionBiasTest#test_mask_fn_attention_bias_with_target_positions().
  AttentionBiasTest.test_bool_tensor_attention_bias: AttentionBiasTest#test_bool_tensor_attention_bias().
  AttentionBiasTest.test_astype: AttentionBiasTest#test_astype().
---
# Module: [`axlearn/common/attention_bias_test.py`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py)

## Classes
### `AttentionBiasTest`
- def: [`axlearn/common/attention_bias_test.py:90`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L90)
- signature: `class AttentionBiasTest(test_utils.TestCase):`
- members:
  - `test_astype(self)` — [`L574`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L574)
  - `test_base_attention_bias_and_residual(self)` — [`L206`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L206) — Tests `BaseAttentionBias.bias_and_residual()`.
  - `test_base_attention_bias_value(self)` — [`L184`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L184) — Tests `BaseAttentionBias.value()`.
  - `test_bias_and_residual_has_bias(self)` — [`L267`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L267)
  - `test_bias_and_residual_repeated_call(self)` — [`L342`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L342) — Test repeated calls to `bias_and_residual()`.
  - `test_bool_tensor_attention_bias(self)` — [`L568`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L568)
  - `test_causal_attention_bias(self)` — [`L121`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L121)
  - `test_composite_attention_bias(self)` — [`L297`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L297)
  - `test_composite_attention_bias_zero(self)` — [`L287`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L287)
  - `test_composite_attention_has_bias(self, bias, expected)` — [`L264`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L264)
  - `test_has_bias(self, bias, expected)` — [`L118`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L118)
  - `test_left_right_window_attention_bias(self)` — [`L155`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L155)
  - `test_mask_fn_attention_bias(self)` — [`L488`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L488)
  - `test_mask_fn_attention_bias_from_sequence(self)` — [`L464`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L464) — Tests `MaskFnAttentionBias.from_sequence()`.
  - `test_mask_fn_attention_bias_target_positions_ndim(self)` — [`L525`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L525) — Tests mask_fn_attention_bias` when `target_positions.ndim == 2.
  - `test_mask_fn_attention_bias_with_target_positions(self)` — [`L545`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L545)
  - `test_segment_id_attention_bias(self)` — [`L441`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L441)
  - `test_sliding_window_attention_bias(self)` — [`L135`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L135)
  - `test_split(self)` — [`L367`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L367)
  - `test_split_subsets(self, causal: Optional[CausalAttentionBias], segment_ids: Optional[SegmentIdAttentionBias], mask: Optional[MaskFnAttentionBias])` — [`L414`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L414) — Tests split() where the input CompositeBias contains any possible subsets of a
  - `test_tensor_attention_bias(self)` — [`L437`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L437)
  - `test_zero_attention_bias(self)` — [`L168`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L168)

### `MaskTest`
- def: [`axlearn/common/attention_bias_test.py:26`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L26)
- signature: `class MaskTest(test_utils.TestCase):`
- members:
  - `test_key_limit_mask(self, valid_k_len, expected)` — [`L79`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L79) — Test make_key_limit_mask masks keys beyond valid_k_len.
  - `test_left_right_window_mask(self, left_context, right_context, expected)` — [`L66`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L66)
  - `test_sliding_window_mask(self, left_context, expected)` — [`L32`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L32)
  - `test_sliding_window_mask_with_time_step(self, time_step, expected)` — [`L46`](../../../../../../raw/code/axlearn/axlearn/common/attention_bias_test.py#L46)

