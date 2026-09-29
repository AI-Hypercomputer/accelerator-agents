---
title: 'Module: src/maxdiffusion/kernels/splash_attention/ring_attention_kernel_test.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/kernels/splash_attention/ring_attention_kernel_test.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.kernels.splash_attention.ring_attention_kernel_test`/
symbols:
  RingAttentionTest.test_ring_attention: RingAttentionTest#test_ring_attention().
  RingAttentionTest.ring_attn: RingAttentionTest#ring_attn().
  P: P.
  RingAttentionTest: RingAttentionTest#
  RingAttentionTest.setUp: RingAttentionTest#setUp().
  partial: partial.
---
# Module: [`src/maxdiffusion/kernels/splash_attention/ring_attention_kernel_test.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel_test.py)

## Classes
### `RingAttentionTest`  ·  implements/extends SplashAttentionTestCase
- def: [`src/maxdiffusion/kernels/splash_attention/ring_attention_kernel_test.py:38`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel_test.py#L38)
- signature: `class RingAttentionTest(test_utils.SplashAttentionTestCase):`
- members:
  - `ring_attn(ring_kernel, q, k, v, segment_ids)` — [`L139`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel_test.py#L139)
  - `setUp(self)` — [`L40`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel_test.py#L40)
  - `test_ring_attention(self, ring_size, num_heads, head_dim, dtype, is_mqa, is_segmented, mask_type)` — [`L59`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel_test.py#L59)
- uses (calls/refs, reference-scoped): [`make_ring_attention`](ring_attention_kernel.md#make_ring_attention), [`manual_sharding_spec`](ring_attention_kernel.md#RingSplashAttentionKernel.manual_sharding_spec), [`get_default`](splash_attention_kernel.md#SplashConfig.get_default), [`SplashConfig`](splash_attention_kernel.md#SplashConfig), [`_assert_allclose`](splash_attention_test_utils.md#SplashAttentionTestCase._assert_allclose), [`setUp`](splash_attention_test_utils.md#SplashAttentionTestCase.setUp), [`SegmentIds`](base.md#SegmentIds), [`SplashAttentionTestCase`](splash_attention_test_utils.md#SplashAttentionTestCase), [`q`](base.md#SegmentIds.q), [`attention_reference`](base.md#attention_reference), [`kv`](base.md#SegmentIds.kv), [`create_segment_ids`](splash_attention_test_utils.md#create_segment_ids), [`make_causal_mask`](splash_attention_mask.md#make_causal_mask), [`FullMask`](splash_attention_mask.md#FullMask), [`P`](ring_attention_kernel_test.md#P), [`_shape`](splash_attention_mask.md#FullMask._shape), [`partial`](ring_attention_kernel_test.md#partial)
- used by: [`setUp`](splash_attention_test_utils.md#SplashAttentionTestCase.setUp), [`SplashAttentionTestCase`](splash_attention_test_utils.md#SplashAttentionTestCase)

## Module values
- `P` — [`L32`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel_test.py#L32)
- `partial` — [`L33`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel_test.py#L33)

