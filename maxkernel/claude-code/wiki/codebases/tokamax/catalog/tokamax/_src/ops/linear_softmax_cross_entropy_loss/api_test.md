---
title: 'Module: tokamax/_src/ops/linear_softmax_cross_entropy_loss/api_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/linear_softmax_cross_entropy_loss/api_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.linear_softmax_cross_entropy_loss.api_test`/
symbols:
  ApiTest.test_correct_implementation_used: ApiTest#test_correct_implementation_used().
  ApiTest.test_api_fwd_bwd_matches_reference: ApiTest#test_api_fwd_bwd_matches_reference().
  _api_fwd_bwd_matches_reference_test_cases: _api_fwd_bwd_matches_reference_test_cases().
  ApiTest: ApiTest#
---
# Module: [`tokamax/_src/ops/linear_softmax_cross_entropy_loss/api_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/api_test.py)

## Classes
### `ApiTest`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/linear_softmax_cross_entropy_loss/api_test.py:51`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/api_test.py#L51)
- signature: `class ApiTest(parameterized.TestCase):`
- members:
  - `test_api_fwd_bwd_matches_reference(self, b_dim, h_dim, v_dim, reduction, test_impl, reference_impl)` — [`L54`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/api_test.py#L54)
  - `test_correct_implementation_used(self, b_dim, h_dim, v_dim, reduction)` — [`L98`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/api_test.py#L98)
- uses (calls/refs, reference-scoped): [`op`](../op.md#BoundArguments.op), [`get_opspecs`](../../hlo_utils.md#get_opspecs), [`IMPLEMENTATIONS`](api.md#IMPLEMENTATIONS), [`linear_softmax_cross_entropy_loss`](api.md#linear_softmax_cross_entropy_loss), [`_api_fwd_bwd_matches_reference_test_cases`](api_test.md#_api_fwd_bwd_matches_reference_test_cases)  (1 test-only)

## Functions
- `_api_fwd_bwd_matches_reference_test_cases()` — [`L27`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/api_test.py#L27)

