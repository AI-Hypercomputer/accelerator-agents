---
title: 'Module: tokamax/_src/ops/ragged_dot/pallas_triton_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/pallas_triton_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.pallas_triton_test`/PallasTritonRaggedDotTest#
symbols:
  PallasTritonRaggedDotTest.test_split_k: test_split_k().
  PallasTritonRaggedDotTest.test_split_k_quantized: test_split_k_quantized().
  PallasTritonRaggedDotTest.__init__: __init__().
  PallasTritonRaggedDotTest._test_simple: _test_simple().
  PallasTritonRaggedDotTest: ''
  PallasTritonRaggedDotTest._test_bench: _test_bench().
  PallasTritonRaggedDotTest.setUp: setUp().
---
# Module: [`tokamax/_src/ops/ragged_dot/pallas_triton_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton_test.py)

## Classes
### `PallasTritonRaggedDotTest`  ·  implements/extends RaggedDotTestBase
- def: [`tokamax/_src/ops/ragged_dot/pallas_triton_test.py:27`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton_test.py#L27)
- signature: `class PallasTritonRaggedDotTest(test_base.RaggedDotTestBase):`
- members:
  - `setUp(self)` — [`L32`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton_test.py#L32)
  - `test_split_k(self, split_k)` — [`L38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton_test.py#L38)
  - `test_split_k_quantized(self)` — [`L53`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton_test.py#L53)
- protocol/private: `__init__`[`L29`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton_test.py#L29), `_test_bench`[`L75`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton_test.py#L75), `_test_simple`[`L70`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton_test.py#L70)
- uses (calls/refs, reference-scoped): [`config`](../op.md#Op.config), [`PallasTritonRaggedDot`](pallas_triton.md#PallasTritonRaggedDot), [`Config`](pallas_triton.md#Config), [`block_k`](pallas_triton.md#Config.block_k), [`block_m`](pallas_triton.md#Config.block_m), [`block_n`](pallas_triton.md#Config.block_n), [`num_stages`](pallas_triton.md#Config.num_stages), [`num_warps`](pallas_triton.md#Config.num_warps), [`split_k`](pallas_triton.md#Config.split_k), [`split_k_intermediate_dtype`](pallas_triton.md#PallasTritonRaggedDot.split_k_intermediate_dtype)  (5 test-only)
- used by: (3 test-only callers)

