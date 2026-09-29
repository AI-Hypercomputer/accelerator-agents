---
title: 'Module: axlearn/common/flash_attention/tpu_attention_test.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/tpu_attention_test.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.tpu_attention_test`/
symbols:
  skipIfGPU: skipIfGPU.
  jax_fn_mask: jax_fn_mask().
  _singleton_mask_fn: _singleton_mask_fn.
  TestFlashAttention: TestFlashAttention#
  TestFlashAttention.test_sliding_window_mask_equivalence: TestFlashAttention#test_sliding_window_mask_equivalence().
  TestFlashAttention.test_to_splash_mask: TestFlashAttention#test_to_splash_mask().
  TestFlashAttention.test_computable_mask: TestFlashAttention#test_computable_mask().
  TestFlashAttention.test_forward_len_dim: TestFlashAttention#test_forward_len_dim().
  TestFlashAttention.test_forward_mask: TestFlashAttention#test_forward_mask().
  TestFlashAttention.test_forward_dtype: TestFlashAttention#test_forward_dtype().
  TestFlashAttention._test_forward: TestFlashAttention#_test_forward().
  TestFlashAttention._test_gradient: TestFlashAttention#_test_gradient().
  TestFlashAttention.test_gradient_masks: TestFlashAttention#test_gradient_masks().
  TestFlashAttention.test_gradient_bias_types: TestFlashAttention#test_gradient_bias_types().
  TestFlashAttention.test_gradient_cross_attn: TestFlashAttention#test_gradient_cross_attn().
  TestFlashAttention.test_gradient_segment_ids: TestFlashAttention#test_gradient_segment_ids().
  TestFlashAttention.test_gradient_per_head_dim_64: TestFlashAttention#test_gradient_per_head_dim_64().
  TestFlashAttention.test_gradient_dtypes: TestFlashAttention#test_gradient_dtypes().
  TestFlashAttention.test_gradient_precision: TestFlashAttention#test_gradient_precision().
  TestFlashAttention.test_gradient_dropout: TestFlashAttention#test_gradient_dropout().
  TestFlashAttention.test_gradient_gqa: TestFlashAttention#test_gradient_gqa().
  TestFlashAttention.test_gradient_long_seq: TestFlashAttention#test_gradient_long_seq().
  TestFlashAttention._test_logit_sink: TestFlashAttention#_test_logit_sink().
  TestFlashAttention.test_logit_sink_values: TestFlashAttention#test_logit_sink_values().
  TestFlashAttention.test_logit_sink_two_blocks: TestFlashAttention#test_logit_sink_two_blocks().
  TestFlashAttention.test_logit_sink_single_head: TestFlashAttention#test_logit_sink_single_head().
  TestFlashAttention.test_logit_sink_float32: TestFlashAttention#test_logit_sink_float32().
  TestFlashAttention.test_logit_sink_gradient: TestFlashAttention#test_logit_sink_gradient().
  TestFlashAttention.test_logit_sink_shape_validation: TestFlashAttention#test_logit_sink_shape_validation().
  TestFlashAttention.test_splash_fused_bwd_heuristic: TestFlashAttention#test_splash_fused_bwd_heuristic().
  TestFlashAttention.test_all_gather_forward_mask: TestFlashAttention#test_all_gather_forward_mask().
  TestFlashAttention.test_all_gather_attention_gradient: TestFlashAttention#test_all_gather_attention_gradient().
  TestFlashAttention.test_incremental_prefill: TestFlashAttention#test_incremental_prefill().
---
# Module: [`axlearn/common/flash_attention/tpu_attention_test.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py)

## Classes
### `TestFlashAttention`
- def: [`axlearn/common/flash_attention/tpu_attention_test.py:67`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L67)
- doc: Tests FlashAttention layer.
- signature: `class TestFlashAttention(TestCase):`
- members:
  - `_test_gradient(self, *, batch_size=2, kv_len=256, num_heads=4, per_head_dim=128, query_length_multiplier=1, mask=None, attention_bias_type=None, with_segment_ids=False, q_dtype=jnp.bfloat16, kv_dtype=jnp.bfloat16, matmul_precision=None, dropout_rate=0, head_group_size=1)` — [`L302`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L302) — Shared gradient test: compares TPU flash attention gradients against reference.
  - `_test_logit_sink(self, *, batch_size=2, seq_len=128, num_heads=4, per_head_dim=128, logit_sink_values=1.0, q_dtype=jnp.bfloat16, check_gradient=False)` — [`L442`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L442) — Shared logit sink test: compares TPU splash attention against reference.
  - `test_all_gather_attention_gradient(self, batch_size, seq_len, num_heads, per_head_dim, mesh, sliding_window_sz, with_segment_ids)` — [`L747`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L747) — Test TPUSplashAttentionWithAllGather gradient computation with sequence parallelism.
  - `test_all_gather_forward_mask(self, per_head_dim, mask, with_segment_ids)` — [`L627`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L627)
  - `test_computable_mask(self, sliding_window_size, seq_len)` — [`L122`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L122) — Test that ComputableMask with mask_fn produces same results as equivalent splash mask.
  - `test_forward_dtype(self, kv_len, mask, with_segment_ids, per_head_dim, q_dtype, kv_dtype)` — [`L226`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L226)
  - `test_forward_len_dim(self, kv_len, mask, with_segment_ids, per_head_dim, q_dtype, kv_dtype)` — [`L204`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L204)
  - `test_forward_mask(self, kv_len, mask, with_segment_ids, per_head_dim, q_dtype, kv_dtype)` — [`L215`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L215)
  - `test_gradient_bias_types(self, attention_bias_type)` — [`L399`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L399) — Tests gradient with each attention bias type.
  - `test_gradient_cross_attn(self, query_length_multiplier)` — [`L404`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L404) — Tests gradient with different Q/KV lengths (cross-attention).
  - `test_gradient_dropout(self)` — [`L428`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L428) — Tests gradient with dropout.
  - `test_gradient_dtypes(self, q_dtype, kv_dtype)` — [`L420`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L420) — Tests gradient with different dtype combinations.
  - `test_gradient_gqa(self)` — [`L432`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L432) — Tests gradient with grouped query attention.
  - `test_gradient_long_seq(self)` — [`L436`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L436) — Tests gradient with longer sequence.
  - `test_gradient_masks(self, mask)` — [`L394`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L394) — Tests gradient with each mask type.
  - `test_gradient_per_head_dim_64(self)` — [`L412`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L412) — Tests gradient with per_head_dim=64.
  - `test_gradient_precision(self)` — [`L424`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L424) — Tests gradient with highest matmul precision.
  - `test_gradient_segment_ids(self)` — [`L408`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L408) — Tests gradient with segment IDs.
  - `test_incremental_prefill(self, prefix_len, suffix_len, mask, per_head_dim)` — [`L883`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L883) — Tests that TPUIncrementalSplashAttention produces correct results for asymmetric Q/K, as
  - `test_logit_sink_float32(self)` — [`L524`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L524) — Tests logit sink with float32 dtype.
  - `test_logit_sink_gradient(self)` — [`L528`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L528) — Tests logit sink gradient computation (representative case).
  - `test_logit_sink_shape_validation(self)` — [`L532`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L532) — Test that logit sink shape validation works correctly.
  - `test_logit_sink_single_head(self)` — [`L520`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L520) — Tests logit sink with num_heads=1.
  - `test_logit_sink_two_blocks(self)` — [`L516`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L516) — Tests logit sink with seq_len=256 (2 TPU blocks).
  - `test_logit_sink_values(self, logit_sink_values)` — [`L512`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L512) — Tests logit sink with different sink values.
  - `test_sliding_window_mask_equivalence(self, seq_len, sliding_window_size)` — [`L71`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L71)
  - `test_splash_fused_bwd_heuristic(self, kv_len, sliding_window_sz, use_fused_override)` — [`L580`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L580)
  - `test_to_splash_mask(self, mask, expected)` — [`L111`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L111)
- protocol/private: `_test_forward`[`L232`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L232)

## Functions
- `jax_fn_mask(sliding_window_size: int)` — [`L45`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L45) — A MaskFn that calls jax.

## Module values
- `_singleton_mask_fn` — [`L63`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L63)
- `skipIfGPU` — [`L40`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_test.py#L40)

