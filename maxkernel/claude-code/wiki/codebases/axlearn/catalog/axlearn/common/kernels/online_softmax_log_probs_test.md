---
title: 'Module: axlearn/common/kernels/online_softmax_log_probs_test.py'
type: catalog
provenance: extracted
module: axlearn/common/kernels/online_softmax_log_probs_test.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.kernels.online_softmax_log_probs_test`/TestOnlineSoftmaxLogProbsPallas#
symbols:
  TestOnlineSoftmaxLogProbsPallas: ''
  TestOnlineSoftmaxLogProbsPallas.test_matches_naive: test_matches_naive().
  TestOnlineSoftmaxLogProbsPallas.test_top_k_matches_naive: test_top_k_matches_naive().
  TestOnlineSoftmaxLogProbsPallas.test_matches_naive_with_different_tile_v: test_matches_naive_with_different_tile_v().
  TestOnlineSoftmaxLogProbsPallas.test_seq_padding: test_seq_padding().
---
# Module: [`axlearn/common/kernels/online_softmax_log_probs_test.py`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs_test.py)

## Classes
### `TestOnlineSoftmaxLogProbsPallas`
- def: [`axlearn/common/kernels/online_softmax_log_probs_test.py:13`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs_test.py#L13)
- doc: Tests the Pallas kernel via online_softmax_log_probs_pallas(interpret=True).
- signature: `class TestOnlineSoftmaxLogProbsPallas(parameterized.TestCase):`
- members:
  - `test_matches_naive(self, vocab_size: int, tile_v: int)` — [`L22`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs_test.py#L22) — Pallas kernel log-normalizer matches full matmul + log_softmax.
  - `test_matches_naive_with_different_tile_v(self)` — [`L89`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs_test.py#L89) — Two different tile_v values produce the same results.
  - `test_seq_padding(self)` — [`L122`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs_test.py#L122) — Handles S not divisible by tile_s.
  - `test_top_k_matches_naive(self, top_k: int, tile_v: int)` — [`L62`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs_test.py#L62) — Pallas top-k matches full log_softmax + jax.lax.top_k.

