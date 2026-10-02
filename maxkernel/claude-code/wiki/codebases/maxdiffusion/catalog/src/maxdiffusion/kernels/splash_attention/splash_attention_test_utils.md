---
title: 'Module: src/maxdiffusion/kernels/splash_attention/splash_attention_test_utils.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/kernels/splash_attention/splash_attention_test_utils.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.kernels.splash_attention.splash_attention_test_utils`/
symbols:
  SplashAttentionTestCase._assert_array_equal: SplashAttentionTestCase#_assert_array_equal().
  SplashAttentionTestCase.setUp: SplashAttentionTestCase#setUp().
  SplashAttentionTestCase._assert_allclose: SplashAttentionTestCase#_assert_allclose().
  SplashAttentionTestCase: SplashAttentionTestCase#
  create_segment_ids: create_segment_ids().
  SplashAttentionTestCase.INTERPRET: SplashAttentionTestCase#INTERPRET.
  thread_unsafe_test_class: thread_unsafe_test_class().
  test_device_matches: test_device_matches().
  thread_unsafe_test_class.f: thread_unsafe_test_class().f().
---
# Module: [`src/maxdiffusion/kernels/splash_attention/splash_attention_test_utils.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_test_utils.py)

## Classes
### `SplashAttentionTestCase`  ·  implements/extends TestCase
- def: [`src/maxdiffusion/kernels/splash_attention/splash_attention_test_utils.py:41`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_test_utils.py#L41)
- doc: Base class for SplashAttention tests.
- signature: `class SplashAttentionTestCase(parameterized.TestCase):`
- members:
  - `setUp(self)` — [`L46`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_test_utils.py#L46)
  - `INTERPRET` — [`L44`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_test_utils.py#L44)
- protocol/private: `_assert_allclose`[`L70`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_test_utils.py#L70), `_assert_array_equal`[`L52`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_test_utils.py#L52)
- uses (calls/refs, reference-scoped): [`setUp`](splash_attention_kernel_sharded_test.md#PallasBaseTest.setUp), [`PallasBaseTest`](splash_attention_kernel_sharded_test.md#PallasBaseTest), [`RingAttentionTest`](ring_attention_kernel_test.md#RingAttentionTest), [`SplashAttentionMaskInfoTest`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest), [`SplashAttentionMaskTest`](splash_attention_mask_test.md#SplashAttentionMaskTest), [`setUp`](ring_attention_kernel_test.md#RingAttentionTest.setUp), [`setUp`](splash_attention_kernel_test.md#SplashAttentionTest.setUp), [`setUp`](splash_attention_mask_test.md#SplashAttentionMaskTest.setUp), [`test_device_matches`](splash_attention_test_utils.md#test_device_matches)
- used by: [`_assert_mask_info_match`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest._assert_mask_info_match), [`test_ring_attention`](ring_attention_kernel_test.md#RingAttentionTest.test_ring_attention), [`test_splash_attention`](splash_attention_kernel_test.md#SplashAttentionTest.test_splash_attention), [`test_splash_attention_bwd`](splash_attention_kernel_test.md#SplashAttentionTest.test_splash_attention_bwd), [`test_splash_attention_fwd`](splash_attention_kernel_test.md#SplashAttentionTest.test_splash_attention_fwd), [`test_manual_partitioning_mha_bwd`](splash_attention_kernel_sharded_test.md#SplashAttentionShardingTest.test_manual_partitioning_mha_bwd), [`test_manual_partitioning_mha_fwd`](splash_attention_kernel_sharded_test.md#SplashAttentionShardingTest.test_manual_partitioning_mha_fwd), [`_compare_masks`](splash_attention_mask_test.md#SplashAttentionMaskTest._compare_masks), [`setUp`](splash_attention_kernel_sharded_test.md#PallasBaseTest.setUp), [`test_causal_mask`](splash_attention_mask_test.md#SplashAttentionMaskTest.test_causal_mask), [`test_chunked_causal_mask`](splash_attention_mask_test.md#SplashAttentionMaskTest.test_chunked_causal_mask), [`test_local_attention_mask`](splash_attention_mask_test.md#SplashAttentionMaskTest.test_local_attention_mask), [`test_local_attention_mask_tall_rectangle`](splash_attention_mask_test.md#SplashAttentionMaskTest.test_local_attention_mask_tall_rectangle), [`test_local_attention_mask_wide_rectangle`](splash_attention_mask_test.md#SplashAttentionMaskTest.test_local_attention_mask_wide_rectangle), [`PallasBaseTest`](splash_attention_kernel_sharded_test.md#PallasBaseTest), [`SplashAttentionTest`](splash_attention_kernel_test.md#SplashAttentionTest), [`RingAttentionTest`](ring_attention_kernel_test.md#RingAttentionTest), [`SplashAttentionMaskInfoTest`](splash_attention_mask_test.md#SplashAttentionMaskInfoTest), [`SplashAttentionMaskTest`](splash_attention_mask_test.md#SplashAttentionMaskTest), [`setUp`](ring_attention_kernel_test.md#RingAttentionTest.setUp), [`setUp`](splash_attention_kernel_test.md#SplashAttentionTest.setUp), [`setUp`](splash_attention_mask_test.md#SplashAttentionMaskTest.setUp)

## Functions
- `create_segment_ids(seq_len: int, num_breaks: int = 2)` — [`L80`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_test_utils.py#L80)
- `f(klass)` — [`L33`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_test_utils.py#L33)
- `test_device_matches(devices: list[str])` — [`L25`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_test_utils.py#L25) — Returns True if the test device matches any of the given devices.
- `thread_unsafe_test_class()` — [`L30`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_test_utils.py#L30) — Decorator that marks a TestCase class as thread-hostile.

