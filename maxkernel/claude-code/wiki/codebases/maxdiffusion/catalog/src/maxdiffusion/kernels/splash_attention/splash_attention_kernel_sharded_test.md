---
title: 'Module: src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_sharded_test.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_sharded_test.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.kernels.splash_attention.splash_attention_kernel_sharded_test`/
symbols:
  SplashAttentionShardingTest.test_manual_partitioning_mha_bwd: SplashAttentionShardingTest#test_manual_partitioning_mha_bwd().
  SplashAttentionShardingTest.test_manual_partitioning_mha_fwd: SplashAttentionShardingTest#test_manual_partitioning_mha_fwd().
  PartitionSpec: PartitionSpec.
  PallasBaseTest.setUp: PallasBaseTest#setUp().
  PallasBaseTest: PallasBaseTest#
  SplashAttentionShardingTest.f: SplashAttentionShardingTest#f().
  partial: partial.
  SplashAttentionShardingTest: SplashAttentionShardingTest#
  SplashAttentionShardingTest.setUp: SplashAttentionShardingTest#setUp().
  P: P.
  PallasBaseTest.INTERPRET: PallasBaseTest#INTERPRET.
---
# Module: [`src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_sharded_test.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_sharded_test.py)

## Classes
### `PallasBaseTest`  ·  implements/extends SplashAttentionTestCase
- def: [`src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_sharded_test.py:40`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_sharded_test.py#L40)
- signature: `class PallasBaseTest(test_utils.SplashAttentionTestCase):`
- members:
  - `setUp(self)` — [`L43`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_sharded_test.py#L43)
  - `INTERPRET` — [`L41`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_sharded_test.py#L41)
- uses (calls/refs, reference-scoped): [`setUp`](splash_attention_test_utils.md#SplashAttentionTestCase.setUp), [`SplashAttentionTestCase`](splash_attention_test_utils.md#SplashAttentionTestCase), [`SplashAttentionShardingTest`](splash_attention_kernel_sharded_test.md#SplashAttentionShardingTest), [`setUp`](splash_attention_kernel_sharded_test.md#SplashAttentionShardingTest.setUp), [`test_device_matches`](splash_attention_test_utils.md#test_device_matches)
- used by: [`setUp`](splash_attention_test_utils.md#SplashAttentionTestCase.setUp), [`SplashAttentionTestCase`](splash_attention_test_utils.md#SplashAttentionTestCase), [`SplashAttentionShardingTest`](splash_attention_kernel_sharded_test.md#SplashAttentionShardingTest), [`setUp`](splash_attention_kernel_sharded_test.md#SplashAttentionShardingTest.setUp)

### `SplashAttentionShardingTest`  ·  implements/extends PallasBaseTest
- def: [`src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_sharded_test.py:52`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_sharded_test.py#L52)
- signature: `class SplashAttentionShardingTest(PallasBaseTest):`
- members:
  - `f(kernel, q, k, v, segment_ids)` — [`L131`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_sharded_test.py#L131)
  - `setUp(self)` — [`L54`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_sharded_test.py#L54)
  - `test_manual_partitioning_mha_bwd(self, topology, num_heads, dtype, is_segmented, is_dynamic_mask)` — [`L145`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_sharded_test.py#L145)
  - `test_manual_partitioning_mha_fwd(self, topology, num_heads, dtype, is_segmented, is_dynamic_mask)` — [`L67`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_sharded_test.py#L67)
- uses (calls/refs, reference-scoped): [`manual_sharding_spec`](splash_attention_kernel.md#SplashAttentionKernel.manual_sharding_spec), [`_assert_allclose`](splash_attention_test_utils.md#SplashAttentionTestCase._assert_allclose), [`SegmentIds`](base.md#SegmentIds), [`q`](base.md#SegmentIds.q), [`attention_reference`](base.md#attention_reference), [`kv`](base.md#SegmentIds.kv), [`create_segment_ids`](splash_attention_test_utils.md#create_segment_ids), [`PartitionSpec`](splash_attention_kernel_sharded_test.md#PartitionSpec), [`make_dynamic_splash_mha`](splash_attention_kernel.md#make_dynamic_splash_mha), [`make_causal_mask`](splash_attention_mask.md#make_causal_mask), [`setUp`](splash_attention_kernel_sharded_test.md#PallasBaseTest.setUp), [`make_splash_mha`](splash_attention_kernel.md#make_splash_mha), [`PallasBaseTest`](splash_attention_kernel_sharded_test.md#PallasBaseTest), [`partial`](splash_attention_kernel_sharded_test.md#partial)
- used by: [`setUp`](splash_attention_kernel_sharded_test.md#PallasBaseTest.setUp), [`PallasBaseTest`](splash_attention_kernel_sharded_test.md#PallasBaseTest)

## Module values
- `P` — [`L34`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_sharded_test.py#L34)
- `PartitionSpec` — [`L33`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_sharded_test.py#L33)
- `partial` — [`L35`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_sharded_test.py#L35)

