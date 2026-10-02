---
title: 'Module: tokamax/_src/ops/attention/api_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/api_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.api_test`/
symbols:
  DotProductAttentionTest.IMPL: DotProductAttentionTest#IMPL.
  DotProductAttentionTest: DotProductAttentionTest#
  DotProductAttentionTest.test_dot_product_attention: DotProductAttentionTest#test_dot_product_attention().
  DotProductAttentionTest.test_dot_product_attention_bias_gradient: DotProductAttentionTest#test_dot_product_attention_bias_gradient().
  DotProductAttentionTest.setUp: DotProductAttentionTest#setUp().
  DotProductAttentionTest.test_symbolic_export: DotProductAttentionTest#test_symbolic_export().
  DotProductAttentionTest.testDotProductAttentionMask: DotProductAttentionTest#testDotProductAttentionMask().
  DotProductAttentionCudnnTest.test_impl_in_hlo: DotProductAttentionCudnnTest#test_impl_in_hlo().
  DotProductAttentionTest.test_memory_scaling: DotProductAttentionTest#test_memory_scaling().
  DotProductAttentionTest.f: DotProductAttentionTest#f().
  DotProductAttentionXlaTest.f: DotProductAttentionXlaTest#f().
  DotProductAttentionMosaicTest: DotProductAttentionMosaicTest#
  DotProductAttentionMosaicTest.setUp: DotProductAttentionMosaicTest#setUp().
  DotProductAttentionTritonTest: DotProductAttentionTritonTest#
  DotProductAttentionTritonTest.setUp: DotProductAttentionTritonTest#setUp().
  DotProductAttentionCudnnTest: DotProductAttentionCudnnTest#
  DotProductAttentionCudnnTest.setUp: DotProductAttentionCudnnTest#setUp().
  DotProductAttentionXlaTest: DotProductAttentionXlaTest#
  DotProductAttentionXlaTest.test_precision: DotProductAttentionXlaTest#test_precision().
  DotProductAttentionXlaChunkedTest: DotProductAttentionXlaChunkedTest#
  DotProductAttentionTest.attention: DotProductAttentionTest#attention().
  _CUDNN_CUSTOM_CALL_TARGET: _CUDNN_CUSTOM_CALL_TARGET.
  DotProductAttentionTest.fn_ans: DotProductAttentionTest#fn_ans().
  DotProductAttentionTest.bwd_ref: DotProductAttentionTest#bwd_ref().
  DotProductAttentionTest.bwd_ans: DotProductAttentionTest#bwd_ans().
  DotProductAttentionCudnnTest.IMPL: DotProductAttentionCudnnTest#IMPL.
  DotProductAttentionMosaicTest.IMPL: DotProductAttentionMosaicTest#IMPL.
  DotProductAttentionTritonTest.IMPL: DotProductAttentionTritonTest#IMPL.
  DotProductAttentionXlaTest.IMPL: DotProductAttentionXlaTest#IMPL.
  DotProductAttentionXlaChunkedTest.IMPL: DotProductAttentionXlaChunkedTest#IMPL.
---
# Module: [`tokamax/_src/ops/attention/api_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py)

## Classes
### `DotProductAttentionCudnnTest`  ·  implements/extends DotProductAttentionTest
- def: [`tokamax/_src/ops/attention/api_test.py:402`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L402)
- signature: `class DotProductAttentionCudnnTest(DotProductAttentionTest):`
- members:
  - `setUp(self)` — [`L405`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L405)
  - `test_impl_in_hlo(self)` — [`L418`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L418)
  - `IMPL` — [`L403`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L403)
- uses (calls/refs, reference-scoped): [`dot_product_attention`](api.md#dot_product_attention), [`DotProductAttentionTest`](api_test.md#DotProductAttentionTest), [`setUp`](api_test.md#DotProductAttentionTest.setUp), [`_CUDNN_CUSTOM_CALL_TARGET`](api_test.md#_CUDNN_CUSTOM_CALL_TARGET)
- used by: [`DotProductAttentionTest`](api_test.md#DotProductAttentionTest), [`setUp`](api_test.md#DotProductAttentionTest.setUp)

### `DotProductAttentionMosaicTest`  ·  implements/extends DotProductAttentionTest
- def: [`tokamax/_src/ops/attention/api_test.py:370`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L370)
- signature: `class DotProductAttentionMosaicTest(DotProductAttentionTest):`
- members:
  - `setUp(self)` — [`L373`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L373)
  - `IMPL` — [`L371`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L371)
- uses (calls/refs, reference-scoped): [`DotProductAttentionTest`](api_test.md#DotProductAttentionTest), [`setUp`](api_test.md#DotProductAttentionTest.setUp)
- used by: [`DotProductAttentionTest`](api_test.md#DotProductAttentionTest), [`setUp`](api_test.md#DotProductAttentionTest.setUp)

### `DotProductAttentionTest`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/attention/api_test.py:35`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L35)
- signature: `class DotProductAttentionTest(parameterized.TestCase):`
- members:
  - `attention(impl, x, bias, mask)` — [`L278`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L278)
  - `bwd_ans(x, bias, mask)` — [`L315`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L315)
  - `bwd_ref(x, bias, mask)` — [`L310`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L310)
  - `f(x, bias)` — [`L126`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L126)
  - `fn_ans(q, k, v, b, m, qs, kvs)` — [`L232`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L232)
  - `setUp(self)` — [`L38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L38)
  - `testDotProductAttentionMask(self, mask_mode)` — [`L163`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L163)
  - `test_dot_product_attention(self, dtype, group_num, use_vmap)` — [`L51`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L51)
  - `test_dot_product_attention_bias_gradient(self, batch_size, use_vmap)` — [`L261`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L261)
  - `test_memory_scaling(self, mode)` — [`L324`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L324)
  - `test_symbolic_export(self)` — [`L116`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L116)
  - `IMPL` — [`L36`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L36)
- uses (calls/refs, reference-scoped): [`op`](../op.md#BoundArguments.op), [`dot_product_attention`](api.md#dot_product_attention), [`arguments`](../op.md#BoundArguments.arguments), [`get_bound_args`](../../autotuning/api.md#get_bound_args), [`BatchedShapeDtype`](../../batching.md#BatchedShapeDtype), [`IMPLEMENTATIONS`](api.md#IMPLEMENTATIONS.IMPLEMENTATIONS), [`disable_jaxtyping`](../../jaxtyping.md#disable_jaxtyping), [`DotProductAttentionCudnnTest`](api_test.md#DotProductAttentionCudnnTest), [`DotProductAttentionMosaicTest`](api_test.md#DotProductAttentionMosaicTest), [`DotProductAttentionTritonTest`](api_test.md#DotProductAttentionTritonTest), [`DotProductAttentionXlaChunkedTest`](api_test.md#DotProductAttentionXlaChunkedTest), [`DotProductAttentionXlaTest`](api_test.md#DotProductAttentionXlaTest), [`setUp`](api_test.md#DotProductAttentionCudnnTest.setUp), [`setUp`](api_test.md#DotProductAttentionMosaicTest.setUp), [`setUp`](api_test.md#DotProductAttentionTritonTest.setUp)
- used by: [`DotProductAttentionCudnnTest`](api_test.md#DotProductAttentionCudnnTest), [`DotProductAttentionMosaicTest`](api_test.md#DotProductAttentionMosaicTest), [`DotProductAttentionTritonTest`](api_test.md#DotProductAttentionTritonTest), [`DotProductAttentionXlaChunkedTest`](api_test.md#DotProductAttentionXlaChunkedTest), [`DotProductAttentionXlaTest`](api_test.md#DotProductAttentionXlaTest), [`setUp`](api_test.md#DotProductAttentionCudnnTest.setUp), [`setUp`](api_test.md#DotProductAttentionMosaicTest.setUp), [`setUp`](api_test.md#DotProductAttentionTritonTest.setUp)

### `DotProductAttentionTritonTest`  ·  implements/extends DotProductAttentionTest
- def: [`tokamax/_src/ops/attention/api_test.py:393`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L393)
- signature: `class DotProductAttentionTritonTest(DotProductAttentionTest):`
- members:
  - `setUp(self)` — [`L396`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L396)
  - `IMPL` — [`L394`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L394)
- uses (calls/refs, reference-scoped): [`DotProductAttentionTest`](api_test.md#DotProductAttentionTest), [`setUp`](api_test.md#DotProductAttentionTest.setUp)
- used by: [`DotProductAttentionTest`](api_test.md#DotProductAttentionTest), [`setUp`](api_test.md#DotProductAttentionTest.setUp)

### `DotProductAttentionXlaChunkedTest`  ·  implements/extends DotProductAttentionTest
- def: [`tokamax/_src/ops/attention/api_test.py:448`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L448)
- signature: `class DotProductAttentionXlaChunkedTest(DotProductAttentionTest):`
- members:
  - `IMPL` — [`L449`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L449)
- uses (calls/refs, reference-scoped): [`DotProductAttentionTest`](api_test.md#DotProductAttentionTest)
- used by: [`DotProductAttentionTest`](api_test.md#DotProductAttentionTest)

### `DotProductAttentionXlaTest`  ·  implements/extends DotProductAttentionTest
- def: [`tokamax/_src/ops/attention/api_test.py:426`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L426)
- signature: `class DotProductAttentionXlaTest(DotProductAttentionTest):`
- members:
  - `f(x, precision)` — [`L437`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L437)
  - `test_precision(self)` — [`L429`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L429)
  - `IMPL` — [`L427`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L427)
- uses (calls/refs, reference-scoped): [`dot_product_attention`](api.md#dot_product_attention), [`DotProductAttentionTest`](api_test.md#DotProductAttentionTest)
- used by: [`DotProductAttentionTest`](api_test.md#DotProductAttentionTest)

## Module values
- `_CUDNN_CUSTOM_CALL_TARGET` — [`L32`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api_test.py#L32)

