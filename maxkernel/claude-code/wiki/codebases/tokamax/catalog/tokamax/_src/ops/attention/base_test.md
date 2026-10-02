---
title: 'Module: tokamax/_src/ops/attention/base_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/base_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.base_test`/
symbols:
  MaskTest.test_intersection: MaskTest#test_intersection().
  MaskTest.test_k_range: MaskTest#test_k_range().
  MaskTest.test_take: MaskTest#test_take().
  MaskTest.test_q_range: MaskTest#test_q_range().
  MaskTest.test_is_causal: MaskTest#test_is_causal().
  MaskTest.f: MaskTest#f().
  DotProductAttentionTest: DotProductAttentionTest#
  DotProductAttentionTest.__init__: DotProductAttentionTest#__init__().
  MaskTest.test_as_array: MaskTest#test_as_array().
  DotProductAttentionWithExplicitVjpTest.__init__: DotProductAttentionWithExplicitVjpTest#__init__().
  MaskTest.test_as_array_symbolic_args_specs: MaskTest#test_as_array_symbolic_args_specs().
  DotProductAttentionTest._run_test: DotProductAttentionTest#_run_test().
  DotProductAttentionWithExplicitVjpTest: DotProductAttentionWithExplicitVjpTest#
  MaskTest: MaskTest#
---
# Module: [`tokamax/_src/ops/attention/base_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base_test.py)

## Classes
### `DotProductAttentionTest`  ·  implements/extends AttentionTestBase
- def: [`tokamax/_src/ops/attention/base_test.py:119`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base_test.py#L119)
- signature: `class DotProductAttentionTest(test_base.AttentionTestBase):`
- protocol/private: `__init__`[`L121`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base_test.py#L121), `_run_test`[`L124`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base_test.py#L124)
- uses (calls/refs, reference-scoped): [`DotProductAttention`](base.md#DotProductAttention), [`DotProductAttentionWithExplicitVjpTest`](base_test.md#DotProductAttentionWithExplicitVjpTest)  (3 test-only)
- used by: [`__init__`](base_test.md#DotProductAttentionWithExplicitVjpTest.__init__), [`DotProductAttentionWithExplicitVjpTest`](base_test.md#DotProductAttentionWithExplicitVjpTest)  (2 test-only)

### `DotProductAttentionWithExplicitVjpTest`  ·  implements/extends DotProductAttentionTest
- def: [`tokamax/_src/ops/attention/base_test.py:136`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base_test.py#L136)
- signature: `class DotProductAttentionWithExplicitVjpTest(DotProductAttentionTest):`
- protocol/private: `__init__`[`L138`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base_test.py#L138)
- uses (calls/refs, reference-scoped): [`DotProductAttentionVjp`](base.md#DotProductAttentionVjp), [`DotProductAttentionTest`](base_test.md#DotProductAttentionTest), [`__init__`](base_test.md#DotProductAttentionTest.__init__)
- used by: [`DotProductAttentionTest`](base_test.md#DotProductAttentionTest)

### `MaskTest`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/attention/base_test.py:25`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base_test.py#L25)
- signature: `class MaskTest(absltest.TestCase):`
- members:
  - `f(mask)` — [`L37`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base_test.py#L37)
  - `test_as_array(self)` — [`L27`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base_test.py#L27)
  - `test_as_array_symbolic_args_specs(self)` — [`L31`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base_test.py#L31)
  - `test_intersection(self)` — [`L78`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base_test.py#L78)
  - `test_is_causal(self)` — [`L71`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base_test.py#L71)
  - `test_k_range(self)` — [`L54`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base_test.py#L54)
  - `test_q_range(self)` — [`L63`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base_test.py#L63)
  - `test_take(self)` — [`L42`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base_test.py#L42)
- uses (calls/refs, reference-scoped): [`Mask`](base.md#Mask), [`as_array`](base.md#Mask.as_array), [`is_causal`](base.md#Mask.is_causal), [`q_end`](base.md#Mask.q_end), [`k_end`](base.md#Mask.k_end), [`k_start`](base.md#Mask.k_start), [`q_start`](base.md#Mask.q_start), [`take`](base.md#Mask.take)

