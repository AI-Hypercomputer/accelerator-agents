---
title: 'Module: tokamax/_src/ops/attention/jax_nn_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/jax_nn_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.jax_nn_test`/
symbols:
  JaxNnDotProductAttentionTest: JaxNnDotProductAttentionTest#
  JaxNnDotProductAttentionTest.__init__: JaxNnDotProductAttentionTest#__init__().
  JaxNnDotProductAttentionCudnnTest._run_test_with_inputs: JaxNnDotProductAttentionCudnnTest#_run_test_with_inputs().
  JaxNnDotProductAttentionTest._run_test_with_inputs: JaxNnDotProductAttentionTest#_run_test_with_inputs().
  JaxNnDotProductAttentionCudnnTest.test_impl_in_hlo: JaxNnDotProductAttentionCudnnTest#test_impl_in_hlo().
  JaxNnDotProductAttentionTest.setUp: JaxNnDotProductAttentionTest#setUp().
  JaxNnDotProductAttentionXlaTest: JaxNnDotProductAttentionXlaTest#
  JaxNnDotProductAttentionXlaTest.__init__: JaxNnDotProductAttentionXlaTest#__init__().
  JaxNnDotProductAttentionCudnnTest: JaxNnDotProductAttentionCudnnTest#
  JaxNnDotProductAttentionCudnnTest.__init__: JaxNnDotProductAttentionCudnnTest#__init__().
  JaxNnDotProductAttentionCudnnTest.setUp: JaxNnDotProductAttentionCudnnTest#setUp().
  JaxNnDotProductAttentionCudnnTest._test_vmap: JaxNnDotProductAttentionCudnnTest#_test_vmap().
  _CUDNN_CUSTOM_CALL_TARGET: _CUDNN_CUSTOM_CALL_TARGET.
  JaxNnDotProductAttentionCudnnTest.impl: JaxNnDotProductAttentionCudnnTest#impl().
  JaxNnDotProductAttentionCudnnTest.test_padding_mask_with_nans: JaxNnDotProductAttentionCudnnTest#test_padding_mask_with_nans().
  JaxNnDotProductAttentionCudnnTest.test_bench_veo3_veo3: JaxNnDotProductAttentionCudnnTest#test_bench_veo3_veo3().
---
# Module: [`tokamax/_src/ops/attention/jax_nn_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn_test.py)

## Classes
### `JaxNnDotProductAttentionCudnnTest`  ·  implements/extends JaxNnDotProductAttentionTest
- def: [`tokamax/_src/ops/attention/jax_nn_test.py:65`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn_test.py#L65)
- signature: `class JaxNnDotProductAttentionCudnnTest(JaxNnDotProductAttentionTest):`
- members:
  - `impl(q, k, v, *, bias, **kwargs)` — [`L80`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn_test.py#L80)
  - `setUp(self)` — [`L71`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn_test.py#L71)
  - `test_bench_veo3_veo3(self)` — [`L93`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn_test.py#L93)
  - `test_impl_in_hlo(self)` — [`L104`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn_test.py#L104)
  - `test_padding_mask_with_nans(self)` — [`L90`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn_test.py#L90)
- protocol/private: `__init__`[`L67`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn_test.py#L67), `_run_test_with_inputs`[`L76`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn_test.py#L76), `_test_vmap`[`L97`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn_test.py#L97)
- uses (calls/refs, reference-scoped): [`JaxNnDotProductAttentionTest`](jax_nn_test.md#JaxNnDotProductAttentionTest), [`__init__`](jax_nn_test.md#JaxNnDotProductAttentionTest.__init__), [`_run_test_with_inputs`](jax_nn_test.md#JaxNnDotProductAttentionTest._run_test_with_inputs), [`setUp`](jax_nn_test.md#JaxNnDotProductAttentionTest.setUp), [`_CUDNN_CUSTOM_CALL_TARGET`](jax_nn_test.md#_CUDNN_CUSTOM_CALL_TARGET)  (2 test-only)
- used by: [`JaxNnDotProductAttentionTest`](jax_nn_test.md#JaxNnDotProductAttentionTest), [`_run_test_with_inputs`](jax_nn_test.md#JaxNnDotProductAttentionTest._run_test_with_inputs), [`setUp`](jax_nn_test.md#JaxNnDotProductAttentionTest.setUp)  (2 test-only)

### `JaxNnDotProductAttentionTest`  ·  implements/extends AttentionTestBase
- def: [`tokamax/_src/ops/attention/jax_nn_test.py:27`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn_test.py#L27)
- signature: `class JaxNnDotProductAttentionTest(test_base.AttentionTestBase):`
- members:
  - `setUp(self)` — [`L39`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn_test.py#L39)
- protocol/private: `__init__`[`L29`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn_test.py#L29), `_run_test_with_inputs`[`L44`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn_test.py#L44)
- uses (calls/refs, reference-scoped): [`implementation`](jax_nn.md#JaxNnDotProductAttention.implementation), [`JaxNnDotProductAttention`](jax_nn.md#JaxNnDotProductAttention), [`_run_test_with_inputs`](jax_nn_test.md#JaxNnDotProductAttentionCudnnTest._run_test_with_inputs), [`JaxNnDotProductAttentionCudnnTest`](jax_nn_test.md#JaxNnDotProductAttentionCudnnTest), [`JaxNnDotProductAttentionXlaTest`](jax_nn_test.md#JaxNnDotProductAttentionXlaTest), [`setUp`](jax_nn_test.md#JaxNnDotProductAttentionCudnnTest.setUp)  (3 test-only)
- used by: [`_run_test_with_inputs`](jax_nn_test.md#JaxNnDotProductAttentionCudnnTest._run_test_with_inputs), [`JaxNnDotProductAttentionCudnnTest`](jax_nn_test.md#JaxNnDotProductAttentionCudnnTest), [`JaxNnDotProductAttentionXlaTest`](jax_nn_test.md#JaxNnDotProductAttentionXlaTest), [`__init__`](jax_nn_test.md#JaxNnDotProductAttentionCudnnTest.__init__), [`__init__`](jax_nn_test.md#JaxNnDotProductAttentionXlaTest.__init__), [`setUp`](jax_nn_test.md#JaxNnDotProductAttentionCudnnTest.setUp)  (2 test-only)

### `JaxNnDotProductAttentionXlaTest`  ·  implements/extends JaxNnDotProductAttentionTest
- def: [`tokamax/_src/ops/attention/jax_nn_test.py:58`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn_test.py#L58)
- signature: `class JaxNnDotProductAttentionXlaTest(JaxNnDotProductAttentionTest):`
- protocol/private: `__init__`[`L60`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn_test.py#L60)
- uses (calls/refs, reference-scoped): [`JaxNnDotProductAttentionTest`](jax_nn_test.md#JaxNnDotProductAttentionTest), [`__init__`](jax_nn_test.md#JaxNnDotProductAttentionTest.__init__)
- used by: [`JaxNnDotProductAttentionTest`](jax_nn_test.md#JaxNnDotProductAttentionTest)

## Module values
- `_CUDNN_CUSTOM_CALL_TARGET` — [`L24`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn_test.py#L24)

