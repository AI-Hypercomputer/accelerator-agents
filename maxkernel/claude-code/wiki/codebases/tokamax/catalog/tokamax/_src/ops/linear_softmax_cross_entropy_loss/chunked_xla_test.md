---
title: 'Module: tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.linear_softmax_cross_entropy_loss.chunked_xla_test`/
symbols:
  ChunkedXlaTest.test_fwd_matches_reference: ChunkedXlaTest#test_fwd_matches_reference().
  ChunkedXlaTest.test_bwd_matches_reference: ChunkedXlaTest#test_bwd_matches_reference().
  _chunked_xla_test_cases: _chunked_xla_test_cases().
  ChunkedXlaTest: ChunkedXlaTest#
---
# Module: [`tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla_test.py)

## Classes
### `ChunkedXlaTest`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla_test.py:50`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla_test.py#L50)
- signature: `class ChunkedXlaTest(parameterized.TestCase):`
- members:
  - `test_bwd_matches_reference(self, reduction, dtype, preferred_element_type, b_dim=256, h_dim=1024, v_dim=2048, b_block_sz=128, v_block_sz=256)` — [`L109`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla_test.py#L109)
  - `test_fwd_matches_reference(self, reduction, dtype, preferred_element_type, b_dim=256, h_dim=1024, v_dim=2048, b_block_sz=128, v_block_sz=256)` — [`L53`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla_test.py#L53)
- uses (calls/refs, reference-scoped): [`linear_softmax_cross_entropy_loss_fwd_reference`](reference.md#linear_softmax_cross_entropy_loss_fwd_reference), [`linear_softmax_cross_entropy_loss_bwd_reference`](reference.md#linear_softmax_cross_entropy_loss_bwd_reference), [`linear_softmax_cross_entropy_loss_bwd_chunked_xla`](chunked_xla.md#linear_softmax_cross_entropy_loss_bwd_chunked_xla), [`linear_softmax_cross_entropy_loss_fwd_chunked_xla`](chunked_xla.md#linear_softmax_cross_entropy_loss_fwd_chunked_xla), [`_chunked_xla_test_cases`](chunked_xla_test.md#_chunked_xla_test_cases)  (1 test-only)

## Functions
- `_chunked_xla_test_cases()` — [`L28`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla_test.py#L28)

