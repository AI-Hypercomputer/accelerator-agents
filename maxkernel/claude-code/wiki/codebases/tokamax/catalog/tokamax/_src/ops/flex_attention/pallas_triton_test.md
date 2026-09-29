---
title: 'Module: tokamax/_src/ops/flex_attention/pallas_triton_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/flex_attention/pallas_triton_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.flex_attention.pallas_triton_test`/
symbols:
  PallasTritonFlexAttentionTest.__init__: PallasTritonFlexAttentionTest#__init__().
  WrappedPallasTritonFlexAttentionTest.__init__: WrappedPallasTritonFlexAttentionTest#__init__().
  PallasTritonFlexAttentionTest: PallasTritonFlexAttentionTest#
  WrappedPallasTritonFlexAttentionTest: WrappedPallasTritonFlexAttentionTest#
  WrappedPallasTritonFlexAttentionTest.setUp: WrappedPallasTritonFlexAttentionTest#setUp().
  PallasTritonFlexAttentionTest.setUp: PallasTritonFlexAttentionTest#setUp().
  WrappedPallasTritonFlexAttentionTest._test_vmap: WrappedPallasTritonFlexAttentionTest#_test_vmap().
---
# Module: [`tokamax/_src/ops/flex_attention/pallas_triton_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton_test.py)

## Classes
### `PallasTritonFlexAttentionTest`  ·  implements/extends FlexAttentionTestBase
- def: [`tokamax/_src/ops/flex_attention/pallas_triton_test.py:25`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton_test.py#L25)
- signature: `class PallasTritonFlexAttentionTest(test_base.FlexAttentionTestBase):`
- members:
  - `setUp(self)` — [`L30`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton_test.py#L30)
- protocol/private: `__init__`[`L27`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton_test.py#L27)
- uses (calls/refs, reference-scoped): [`PallasTritonFlexAttention`](pallas_triton.md#PallasTritonFlexAttention)  (2 test-only)
- used by: (1 test-only callers)

### `WrappedPallasTritonFlexAttentionTest`  ·  implements/extends WrappedFlexAttentionTestBase
- def: [`tokamax/_src/ops/flex_attention/pallas_triton_test.py:36`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton_test.py#L36)
- signature: `class WrappedPallasTritonFlexAttentionTest(wrapper_test_base.WrappedFlexAttentionTestBase):`
- members:
  - `setUp(self)` — [`L47`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton_test.py#L47)
- protocol/private: `__init__`[`L40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton_test.py#L40), `_test_vmap`[`L54`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton_test.py#L54)
- uses (calls/refs, reference-scoped): [`__init__`](wrapper_test_base.md#WrappedFlexAttentionTestBase.__init__), [`WrappedFlexAttentionTestBase`](wrapper_test_base.md#WrappedFlexAttentionTestBase), [`PallasTritonFlexAttention`](pallas_triton.md#PallasTritonFlexAttention), [`setUp`](wrapper_test_base.md#WrappedFlexAttentionTestBase.setUp)
- used by: [`WrappedFlexAttentionTestBase`](wrapper_test_base.md#WrappedFlexAttentionTestBase), [`setUp`](wrapper_test_base.md#WrappedFlexAttentionTestBase.setUp)  (1 test-only)

