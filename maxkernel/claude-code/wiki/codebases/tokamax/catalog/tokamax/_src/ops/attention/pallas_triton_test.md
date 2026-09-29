---
title: 'Module: tokamax/_src/ops/attention/pallas_triton_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/pallas_triton_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.pallas_triton_test`/
symbols:
  PallasTritonFlashAttentionTest.test_block_d: PallasTritonFlashAttentionTest#test_block_d().
  PallasTritonFlashAttentionTest.test_small_block_q: PallasTritonFlashAttentionTest#test_small_block_q().
  PallasTritonFlashAttentionTest.impl: PallasTritonFlashAttentionTest#impl().
  PallasTritonFlashAttentionTest.test_split_k: PallasTritonFlashAttentionTest#test_split_k().
  PallasTritonFlashAttentionWithPallasTritonVjpTest.__init__: PallasTritonFlashAttentionWithPallasTritonVjpTest#__init__().
  PallasTritonFlashAttentionTest.__init__: PallasTritonFlashAttentionTest#__init__().
  PallasTritonFlashAttentionTest._run_test: PallasTritonFlashAttentionTest#_run_test().
  _F32PrecisionXlaAttentionVjp: _F32PrecisionXlaAttentionVjp#
  _F32PrecisionXlaAttentionVjp.__call__: _F32PrecisionXlaAttentionVjp#__call__().
  PallasTritonFlashAttentionTest: PallasTritonFlashAttentionTest#
  PallasTritonFlashAttentionWithPallasTritonVjpTest: PallasTritonFlashAttentionWithPallasTritonVjpTest#
  PallasTritonFlashAttentionWithPallasTritonVjpTest._run_test_with_inputs: PallasTritonFlashAttentionWithPallasTritonVjpTest#_run_test_with_inputs().
  PallasTritonFlashAttentionTest.ref_impl: PallasTritonFlashAttentionTest#ref_impl().
  PallasTritonFlashAttentionTest.setUp: PallasTritonFlashAttentionTest#setUp().
  PallasTritonFlashAttentionWithPallasTritonVjpTest.test_normalize_output: PallasTritonFlashAttentionWithPallasTritonVjpTest#test_normalize_output().
  _is_precision_supported: _is_precision_supported().
---
# Module: [`tokamax/_src/ops/attention/pallas_triton_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_test.py)

## Classes
### `PallasTritonFlashAttentionTest`  ·  implements/extends AttentionTestBase
- def: [`tokamax/_src/ops/attention/pallas_triton_test.py:45`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_test.py#L45)
- signature: `class PallasTritonFlashAttentionTest(test_base.AttentionTestBase):`
- members:
  - `impl(q, k, v, **kwargs)` — [`L75`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_test.py#L75)
  - `ref_impl(q, k, v, **kwargs)` — [`L79`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_test.py#L79)
  - `setUp(self)` — [`L51`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_test.py#L51)
  - `test_block_d(self)` — [`L62`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_test.py#L62) — Tests `block_d != None` with quantization and different head_dim_out.
  - `test_small_block_q(self, block_q: int)` — [`L89`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_test.py#L89)
  - `test_split_k(self, split_k)` — [`L97`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_test.py#L97)
- protocol/private: `__init__`[`L47`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_test.py#L47), `_run_test`[`L56`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_test.py#L56)
- uses (calls/refs, reference-scoped): [`vjp`](../op.md#Op.vjp), [`replace`](../op.md#Op.replace), [`PallasTritonFlashAttention`](pallas_triton.md#PallasTritonFlashAttention), [`Config`](pallas_triton.md#Config), [`block_q`](pallas_triton.md#Config.block_q), [`block_k`](pallas_triton.md#Config.block_k), [`num_stages`](pallas_triton.md#Config.num_stages), [`num_warps`](pallas_triton.md#Config.num_warps), [`block_d`](pallas_triton.md#Config.block_d), [`block_d_out`](pallas_triton.md#Config.block_d_out), [`_F32PrecisionXlaAttentionVjp`](pallas_triton_test.md#_F32PrecisionXlaAttentionVjp), [`split_k`](pallas_triton.md#Config.split_k)  (5 test-only)
- used by: (2 test-only callers)

### `PallasTritonFlashAttentionWithPallasTritonVjpTest`  ·  implements/extends AttentionTestBase
- def: [`tokamax/_src/ops/attention/pallas_triton_test.py:126`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_test.py#L126)
- signature: `class PallasTritonFlashAttentionWithPallasTritonVjpTest(test_base.AttentionTestBase):`
- members:
  - `test_normalize_output(self)` — [`L148`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_test.py#L148)
- protocol/private: `__init__`[`L130`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_test.py#L130), `_run_test_with_inputs`[`L136`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_test.py#L136)
- uses (calls/refs, reference-scoped): [`vjp`](../op.md#Op.vjp), [`PallasTritonFlashAttention`](pallas_triton.md#PallasTritonFlashAttention), [`PallasTritonFlashAttentionVjp`](pallas_triton_vjp.md#PallasTritonFlashAttentionVjp), [`dbias_intermediate_dtype`](pallas_triton_vjp.md#PallasTritonFlashAttentionVjp.dbias_intermediate_dtype)  (3 test-only)
- used by: (3 test-only callers)

### `_F32PrecisionXlaAttentionVjp`  ·  implements/extends DotProductAttentionVjp
- def: [`tokamax/_src/ops/attention/pallas_triton_test.py:34`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_test.py#L34)
- signature: `class _F32PrecisionXlaAttentionVjp(base.DotProductAttentionVjp):`
- protocol/private: `__call__`[`L36`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_test.py#L36)
- uses (calls/refs, reference-scoped): [`DotProductAttentionVjp`](base.md#DotProductAttentionVjp), [`__call__`](../op.md#Op.__call__)
- used by: [`DotProductAttentionVjp`](base.md#DotProductAttentionVjp), [`__init__`](pallas_triton_test.md#PallasTritonFlashAttentionTest.__init__)

## Functions
- `_is_precision_supported(precision: jax.lax.DotAlgorithmPreset)` — [`L152`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_test.py#L152)

