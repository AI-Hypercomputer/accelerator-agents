---
title: 'Module: axlearn/common/rattention/kernels/linear_attention_kernels_test.py'
type: catalog
provenance: extracted
module: axlearn/common/rattention/kernels/linear_attention_kernels_test.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.rattention.kernels.linear_attention_kernels_test`/
symbols:
  _generate_test_inputs: _generate_test_inputs().
  LinearAttentionPallasKernelTest: LinearAttentionPallasKernelTest#
  LinearAttentionPallasKernelTest.test_rla_forward_and_backward: LinearAttentionPallasKernelTest#test_rla_forward_and_backward().
  LinearAttentionPallasKernelTest.test_rla_prefill: LinearAttentionPallasKernelTest#test_rla_prefill().
---
# Module: [`axlearn/common/rattention/kernels/linear_attention_kernels_test.py`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels_test.py)

## Classes
### `LinearAttentionPallasKernelTest`
- def: [`axlearn/common/rattention/kernels/linear_attention_kernels_test.py:48`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels_test.py#L48)
- signature: `class LinearAttentionPallasKernelTest(TestCase):`
- members:
  - `test_rla_forward_and_backward(self, batch_size, num_heads, num_kv_heads, seq_len, dtype, window_size, feat_map, chunk_size, seed)` — [`L62`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels_test.py#L62)
  - `test_rla_prefill(self, dtype)` — [`L122`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels_test.py#L122)

## Functions
- `_generate_test_inputs(shape: tuple, dtype: jnp.dtype, seed: int)` — [`L23`](../../../../../../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels_test.py#L23) — Args:

