---
title: 'Module: tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.linear_softmax_cross_entropy_loss.reference_test`/FlashLceReferenceTest#
symbols:
  FlashLceReferenceTest.test_reference_bwd_matches_jax_grad: test_reference_bwd_matches_jax_grad().
  FlashLceReferenceTest.test_reference_fwd_running_correctly: test_reference_fwd_running_correctly().
  FlashLceReferenceTest: ''
---
# Module: [`tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference_test.py)

## Classes
### `FlashLceReferenceTest`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference_test.py:24`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference_test.py#L24)
- signature: `class FlashLceReferenceTest(parameterized.TestCase):`
- members:
  - `test_reference_bwd_matches_jax_grad(self, b_dim, h_dim, v_dim, reduction)` — [`L99`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference_test.py#L99)
  - `test_reference_fwd_running_correctly(self, b_dim, h_dim, v_dim, reduction)` — [`L56`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference_test.py#L56)
- uses (calls/refs, reference-scoped): [`linear_softmax_cross_entropy_loss_fwd_reference`](reference.md#linear_softmax_cross_entropy_loss_fwd_reference), [`linear_softmax_cross_entropy_loss_bwd_reference`](reference.md#linear_softmax_cross_entropy_loss_bwd_reference)  (1 test-only)

