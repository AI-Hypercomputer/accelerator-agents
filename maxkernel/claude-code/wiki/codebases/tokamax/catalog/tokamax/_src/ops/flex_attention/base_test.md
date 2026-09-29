---
title: 'Module: tokamax/_src/ops/flex_attention/base_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/flex_attention/base_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.flex_attention.base_test`/
symbols:
  FlexAttentionTest.__init__: FlexAttentionTest#__init__().
  WrappedFlexAttentionTest.__init__: WrappedFlexAttentionTest#__init__().
  WrappedFlexAttentionTest.test_normalize_output: WrappedFlexAttentionTest#test_normalize_output().
  FlexAttentionTest: FlexAttentionTest#
  WrappedFlexAttentionTest: WrappedFlexAttentionTest#
---
# Module: [`tokamax/_src/ops/flex_attention/base_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/base_test.py)

## Classes
### `FlexAttentionTest`  ·  implements/extends FlexAttentionTestBase
- def: [`tokamax/_src/ops/flex_attention/base_test.py:22`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/base_test.py#L22)
- signature: `class FlexAttentionTest(test_base.FlexAttentionTestBase):`
- protocol/private: `__init__`[`L24`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/base_test.py#L24)
- uses (calls/refs, reference-scoped): [`FlexAttention`](base.md#FlexAttention)  (2 test-only)
- used by: (1 test-only callers)

### `WrappedFlexAttentionTest`  ·  implements/extends WrappedFlexAttentionTestBase
- def: [`tokamax/_src/ops/flex_attention/base_test.py:28`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/base_test.py#L28)
- signature: `class WrappedFlexAttentionTest(wrapper_test_base.WrappedFlexAttentionTestBase):`
- members:
  - `test_normalize_output(self)` — [`L34`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/base_test.py#L34)
- protocol/private: `__init__`[`L30`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/base_test.py#L30)
- uses (calls/refs, reference-scoped): [`FlexAttention`](base.md#FlexAttention), [`__init__`](wrapper_test_base.md#WrappedFlexAttentionTestBase.__init__), [`WrappedFlexAttentionTestBase`](wrapper_test_base.md#WrappedFlexAttentionTestBase)  (2 test-only)
- used by: [`WrappedFlexAttentionTestBase`](wrapper_test_base.md#WrappedFlexAttentionTestBase)  (1 test-only)

