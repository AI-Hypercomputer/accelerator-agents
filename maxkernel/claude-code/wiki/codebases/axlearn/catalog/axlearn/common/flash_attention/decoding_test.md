---
title: 'Module: axlearn/common/flash_attention/decoding_test.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/decoding_test.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.decoding_test`/DecodingTest#
symbols:
  DecodingTest: ''
  DecodingTest.tolerance_map: tolerance_map.
  DecodingTest.test_paged_attention_against_ref: test_paged_attention_against_ref().
  DecodingTest.test_decode_against_ref: test_decode_against_ref().
---
# Module: [`axlearn/common/flash_attention/decoding_test.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/decoding_test.py)

## Classes
### `DecodingTest`
- def: [`axlearn/common/flash_attention/decoding_test.py:50`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/decoding_test.py#L50)
- doc: Tests GPU and TPU decoding.
- signature: `class DecodingTest(TestCase):`
- members:
  - `test_decode_against_ref(self, batch_size: int, seq_len: int, num_heads: int, per_head_dim: int, attention_bias_type: Literal['2d', '4d', None], input_dtype: jnp.dtype, padding: int, kv_head_factor: int, window_len: int, decoding_fn: BaseFlashAttention, use_logit_sink: bool)` — [`L187`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/decoding_test.py#L187)
  - `test_paged_attention_against_ref(self, batch_size: int, seq_len: int, num_heads: int, per_head_dim: int, attention_bias_type: Literal['2d', '4d', None], input_dtype: jnp.dtype, padding: int, kv_head_factor: int, window_len: int, page_size: int, decoding_fn: BasePagedAttention)` — [`L95`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/decoding_test.py#L95)
  - `tolerance_map` — [`L53`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/decoding_test.py#L53)

