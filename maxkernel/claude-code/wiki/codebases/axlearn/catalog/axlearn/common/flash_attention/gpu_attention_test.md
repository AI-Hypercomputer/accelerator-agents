---
title: 'Module: axlearn/common/flash_attention/gpu_attention_test.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/gpu_attention_test.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.gpu_attention_test`/
symbols:
  _default_tol_fn: _default_tol_fn().
  TestFn: TestFn.
  TolFn: TolFn.
  _test_forward_and_backward: _test_forward_and_backward().
  _cudnn_xla_forward_tol_fn: _cudnn_xla_forward_tol_fn().
  TritonFwdOnlyTest: TritonFwdOnlyTest#
  TritonFwdOnlyTest.test_triton_fwd_only_against_ref: TritonFwdOnlyTest#test_triton_fwd_only_against_ref().
  TritonAgainstXlaRefTest: TritonAgainstXlaRefTest#
  TritonAgainstXlaRefTest.test_triton_against_xla_ref: TritonAgainstXlaRefTest#test_triton_against_xla_ref().
  SlidingWindowMaskTest: SlidingWindowMaskTest#
  SlidingWindowMaskTest.test_sliding_window_mask: SlidingWindowMaskTest#test_sliding_window_mask().
  CuDNNTest: CuDNNTest#
  CuDNNTest.test_cudnn_against_triton_ref: CuDNNTest#test_cudnn_against_triton_ref().
  CuDNNTest.test_cudnn_dropout_against_xla_dropout: CuDNNTest#test_cudnn_dropout_against_xla_dropout().
  CuDNNTest.test_cudnn_seqlen_head_support: CuDNNTest#test_cudnn_seqlen_head_support().
  CuDNNTest.test_cudnn_dropout_determinism: CuDNNTest#test_cudnn_dropout_determinism().
---
# Module: [`axlearn/common/flash_attention/gpu_attention_test.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_test.py)

## Classes
### `CuDNNTest`
- def: [`axlearn/common/flash_attention/gpu_attention_test.py:313`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_test.py#L313)
- signature: `class CuDNNTest(parameterized.TestCase):`
- members:
  - `test_cudnn_against_triton_ref(self, batch_size: int, num_heads: int, seq_len: int, per_head_dim: int, causal: bool, dtype: jnp.dtype)` — [`L324`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_test.py#L324)
  - `test_cudnn_dropout_against_xla_dropout(self, batch_size: int, num_heads: int, seq_len: int, per_head_dim: int, causal: bool, dtype: jnp.dtype, dropout_rate: float)` — [`L378`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_test.py#L378) — Tests that cudnn dropout works as expected.
  - `test_cudnn_dropout_determinism(self)` — [`L490`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_test.py#L490) — Tests that cuDNN dropout produces identical outputs across runs.
  - `test_cudnn_seqlen_head_support(self, batch_size: int, num_heads: int, seq_len: int, kv_seq_len: int, per_head_dim: int, causal: bool, dtype: jnp.dtype)` — [`L453`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_test.py#L453) — Tests that cudnn supports any even sequence length and head dim % 8 == 0.

### `SlidingWindowMaskTest`
- def: [`axlearn/common/flash_attention/gpu_attention_test.py:258`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_test.py#L258)
- signature: `class SlidingWindowMaskTest(parameterized.TestCase):`
- members:
  - `test_sliding_window_mask(self, batch_size, seq_len, num_heads, per_head_dim, sliding_window_size, use_segment_ids, test_cls)` — [`L268`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_test.py#L268)

### `TritonAgainstXlaRefTest`
- def: [`axlearn/common/flash_attention/gpu_attention_test.py:181`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_test.py#L181)
- signature: `class TritonAgainstXlaRefTest(parameterized.TestCase):`
- members:
  - `test_triton_against_xla_ref(self, batch_size: int, num_heads: int, query_len: int, per_head_dim: int, kv_len: int, attention_bias_type: Literal['2d', '4d', None], with_segment_ids: bool, dropout_rate: float, block_size: int, mask_fn: Optional[MaskFn], dtype: jnp.dtype)` — [`L200`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_test.py#L200)

### `TritonFwdOnlyTest`
- def: [`axlearn/common/flash_attention/gpu_attention_test.py:105`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_test.py#L105)
- signature: `class TritonFwdOnlyTest(parameterized.TestCase):`
- members:
  - `test_triton_fwd_only_against_ref(self, batch_size: int, query_len: int, num_heads: int, per_head_dim: int, kv_len: int, dropout_rate: float, block_size: int, mask_fn: Optional[MaskFn], attention_bias_type: Literal['2d', '4d', None], with_segment_ids: bool, dtype: jnp.dtype)` — [`L125`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_test.py#L125)

## Functions
- `_cudnn_xla_forward_tol_fn(backend, dtype)` — [`L95`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_test.py#L95)
- `_default_tol_fn(backend, dtype)` — [`L44`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_test.py#L44)
- `_test_forward_and_backward(q: Tensor, k: Tensor, v: Tensor, bias, *, ref_fn: TestFn, test_fn: TestFn, forward_tol_fn: Callable=_default_tol_fn, backward_tol_fn: Callable=_default_tol_fn)` — [`L59`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_test.py#L59)

## Module values
- `TestFn` — [`L55`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_test.py#L55)
- `TolFn` — [`L56`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_test.py#L56)

