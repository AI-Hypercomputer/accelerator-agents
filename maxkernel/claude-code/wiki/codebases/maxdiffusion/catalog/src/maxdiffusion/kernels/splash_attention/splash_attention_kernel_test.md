---
title: 'Module: src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.kernels.splash_attention.splash_attention_kernel_test`/
symbols:
  SplashAttentionTest.test_splash_attention: SplashAttentionTest#test_splash_attention().
  SplashAttentionTest.test_splash_attention_bwd: SplashAttentionTest#test_splash_attention_bwd().
  SplashAttentionTest.test_splash_attention_fwd: SplashAttentionTest#test_splash_attention_fwd().
  _generate_inputs: _generate_inputs().
  block_sizes_strategy: block_sizes_strategy().
  Mask: Mask#
  mask_strategy: mask_strategy().
  Mask.get_mask: Mask#get_mask().
  LocalAttentionMask.get_mask: LocalAttentionMask#get_mask().
  RandomMask.get_mask: RandomMask#get_mask().
  compose_mask_strategy: compose_mask_strategy().
  check_mask_no_empty_rows: check_mask_no_empty_rows().
  ComposeMask.get_mask: ComposeMask#get_mask().
  local_attention_mask_strategy: local_attention_mask_strategy().
  SplitMask.get_mask: SplitMask#get_mask().
  FullMask.get_mask: FullMask#get_mask().
  CausalMask.get_mask: CausalMask#get_mask().
  random_mask_strategy: random_mask_strategy().
  model_config_strategy: model_config_strategy().
  Draw: Draw.
  full_mask_strategy: full_mask_strategy().
  split_mask_strategy: split_mask_strategy().
  causal_mask_strategy: causal_mask_strategy().
  ModelConfig.kv_seq_len: ModelConfig#kv_seq_len.
  ModelConfig.dtype: ModelConfig#dtype.
  SplashAttentionTest: SplashAttentionTest#
  RandomMask: RandomMask#
  ModelConfig.q_seq_len: ModelConfig#q_seq_len.
  segment_ids_strategy: segment_ids_strategy().
  SplitMask: SplitMask#
  LocalAttentionMask: LocalAttentionMask#
  ComposeMask: ComposeMask#
  ComposeMask.left: ComposeMask#left.
  ComposeMask.right: ComposeMask#right.
  ComposeMask.op: ComposeMask#op.
  FullMask: FullMask#
  CausalMask: CausalMask#
  partial: partial.
  ModelConfig: ModelConfig#
  ModelConfig.num_q_heads: ModelConfig#num_q_heads.
  ModelConfig.head_dim_qk: ModelConfig#head_dim_qk.
  ModelConfig.head_dim_v: ModelConfig#head_dim_v.
  LocalAttentionMask.seq_len: LocalAttentionMask#seq_len.
  attn_logits_soft_cap_strategy: attn_logits_soft_cap_strategy().
  SplashAttentionTest.setUp: SplashAttentionTest#setUp().
  ModelConfig.num_kv_heads: ModelConfig#num_kv_heads.
  LocalAttentionMask.offset: LocalAttentionMask#offset.
  seed_strategy: seed_strategy().
  SplitMask.q_seq_len: SplitMask#q_seq_len.
  SplitMask.kv_seq_len: SplitMask#kv_seq_len.
  FullMask.q_seq_len: FullMask#q_seq_len.
  FullMask.kv_seq_len: FullMask#kv_seq_len.
  CausalMask.q_seq_len: CausalMask#q_seq_len.
  CausalMask.kv_seq_len: CausalMask#kv_seq_len.
  LocalAttentionMask.left: LocalAttentionMask#left.
  LocalAttentionMask.right: LocalAttentionMask#right.
  RandomMask.q_seq_len: RandomMask#q_seq_len.
  RandomMask.kv_seq_len: RandomMask#kv_seq_len.
  RandomMask.sparsity: RandomMask#sparsity.
  RandomMask.seed: RandomMask#seed.
---
# Module: [`src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py)

## Classes
### `CausalMask`
- def: [`src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py:120`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L120)
- members:
  - `get_mask(self)` — [`L124`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L124)
  - `kv_seq_len` — [`L122`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L122)
  - `q_seq_len` — [`L121`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L121)
- uses (calls/refs, reference-scoped): [`Mask`](splash_attention_mask.md#Mask), [`Mask`](splash_attention_kernel_test.md#Mask), [`CausalMask`](splash_attention_mask.md#CausalMask)
- used by: [`get_mask`](splash_attention_kernel_test.md#Mask.get_mask), [`causal_mask_strategy`](splash_attention_kernel_test.md#causal_mask_strategy)

### `ComposeMask`  ·  implements/extends Mask
- def: [`src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py:180`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L180)
- signature: `class ComposeMask(Mask):`
- members:
  - `get_mask(self)` — [`L185`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L185)
  - `left` — [`L181`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L181)
  - `op` — [`L183`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L183)
  - `right` — [`L182`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L182)
- uses (calls/refs, reference-scoped): [`Mask`](splash_attention_mask.md#Mask), [`Mask`](splash_attention_kernel_test.md#Mask), [`get_mask`](splash_attention_kernel_test.md#Mask.get_mask)
- used by: [`Mask`](splash_attention_kernel_test.md#Mask), [`get_mask`](splash_attention_kernel_test.md#Mask.get_mask), [`compose_mask_strategy`](splash_attention_kernel_test.md#compose_mask_strategy)

### `FullMask`
- def: [`src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py:107`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L107)
- members:
  - `get_mask(self)` — [`L111`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L111)
  - `kv_seq_len` — [`L109`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L109)
  - `q_seq_len` — [`L108`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L108)
- uses (calls/refs, reference-scoped): [`Mask`](splash_attention_mask.md#Mask), [`Mask`](splash_attention_kernel_test.md#Mask), [`FullMask`](splash_attention_mask.md#FullMask)
- used by: [`get_mask`](splash_attention_kernel_test.md#Mask.get_mask), [`full_mask_strategy`](splash_attention_kernel_test.md#full_mask_strategy)

### `LocalAttentionMask`  ·  implements/extends Mask
- def: [`src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py:129`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L129)
- signature: `class LocalAttentionMask(Mask):`
- members:
  - `get_mask(self)` — [`L135`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L135)
  - `left` — [`L131`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L131)
  - `offset` — [`L133`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L133)
  - `right` — [`L132`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L132)
  - `seq_len` — [`L130`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L130)
- uses (calls/refs, reference-scoped): [`Mask`](splash_attention_mask.md#Mask), [`Mask`](splash_attention_kernel_test.md#Mask), [`NumpyMask`](splash_attention_mask.md#NumpyMask), [`LocalMask`](splash_attention_mask.md#LocalMask)
- used by: [`Mask`](splash_attention_kernel_test.md#Mask), [`get_mask`](splash_attention_kernel_test.md#Mask.get_mask), [`local_attention_mask_strategy`](splash_attention_kernel_test.md#local_attention_mask_strategy)

### `Mask`
- def: [`src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py:81`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L81)
- signature: `class Mask:`
- members:
  - `get_mask(self)` — [`L83`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L83)
- uses (calls/refs, reference-scoped): [`Mask`](splash_attention_mask.md#Mask), [`get_mask`](splash_attention_kernel_test.md#LocalAttentionMask.get_mask), [`get_mask`](splash_attention_kernel_test.md#RandomMask.get_mask), [`get_mask`](splash_attention_kernel_test.md#ComposeMask.get_mask), [`get_mask`](splash_attention_kernel_test.md#CausalMask.get_mask), [`get_mask`](splash_attention_kernel_test.md#FullMask.get_mask), [`get_mask`](splash_attention_kernel_test.md#SplitMask.get_mask), [`RandomMask`](splash_attention_kernel_test.md#RandomMask), [`ComposeMask`](splash_attention_kernel_test.md#ComposeMask), [`LocalAttentionMask`](splash_attention_kernel_test.md#LocalAttentionMask), [`SplitMask`](splash_attention_kernel_test.md#SplitMask)
- used by: [`mask_strategy`](splash_attention_kernel_test.md#mask_strategy), [`compose_mask_strategy`](splash_attention_kernel_test.md#compose_mask_strategy), [`get_mask`](splash_attention_kernel_test.md#ComposeMask.get_mask), [`local_attention_mask_strategy`](splash_attention_kernel_test.md#local_attention_mask_strategy), [`random_mask_strategy`](splash_attention_kernel_test.md#random_mask_strategy), [`causal_mask_strategy`](splash_attention_kernel_test.md#causal_mask_strategy), [`full_mask_strategy`](splash_attention_kernel_test.md#full_mask_strategy), [`split_mask_strategy`](splash_attention_kernel_test.md#split_mask_strategy), [`RandomMask`](splash_attention_kernel_test.md#RandomMask), [`CausalMask`](splash_attention_kernel_test.md#CausalMask), [`ComposeMask`](splash_attention_kernel_test.md#ComposeMask), [`FullMask`](splash_attention_kernel_test.md#FullMask), [`LocalAttentionMask`](splash_attention_kernel_test.md#LocalAttentionMask), [`SplitMask`](splash_attention_kernel_test.md#SplitMask), [`left`](splash_attention_kernel_test.md#ComposeMask.left), [`right`](splash_attention_kernel_test.md#ComposeMask.right)

### `ModelConfig`
- def: [`src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py:54`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L54)
- signature: `class ModelConfig:`
- members:
  - `dtype` — [`L61`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L61)
  - `head_dim_qk` — [`L59`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L59)
  - `head_dim_v` — [`L60`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L60)
  - `kv_seq_len` — [`L56`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L56)
  - `num_kv_heads` — [`L58`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L58)
  - `num_q_heads` — [`L57`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L57)
  - `q_seq_len` — [`L55`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L55)
- used by: [`_generate_inputs`](splash_attention_kernel_test.md#_generate_inputs), [`model_config_strategy`](splash_attention_kernel_test.md#model_config_strategy)

### `RandomMask`  ·  implements/extends Mask
- def: [`src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py:156`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L156)
- signature: `class RandomMask(Mask):`
- members:
  - `get_mask(self)` — [`L162`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L162)
  - `kv_seq_len` — [`L158`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L158)
  - `q_seq_len` — [`L157`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L157)
  - `seed` — [`L160`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L160)
  - `sparsity` — [`L159`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L159)
- uses (calls/refs, reference-scoped): [`Mask`](splash_attention_mask.md#Mask), [`Mask`](splash_attention_kernel_test.md#Mask), [`NumpyMask`](splash_attention_mask.md#NumpyMask), [`make_random_mask`](splash_attention_mask.md#make_random_mask)
- used by: [`test_splash_attention`](splash_attention_kernel_test.md#SplashAttentionTest.test_splash_attention), [`Mask`](splash_attention_kernel_test.md#Mask), [`get_mask`](splash_attention_kernel_test.md#Mask.get_mask), [`random_mask_strategy`](splash_attention_kernel_test.md#random_mask_strategy)

### `SplashAttentionTest`
- def: [`src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py:329`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L329)
- signature: `class SplashAttentionTest(test_utils.SplashAttentionTestCase):`
- members:
  - `setUp(self)` — [`L331`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L331)
  - `test_splash_attention(self, is_mqa, is_segmented, is_dynamic_mask, data)` — [`L342`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L342) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel_test](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel_test.md)
  - `test_splash_attention_bwd(self, is_mqa, is_segmented, is_dynamic_mask, use_max_logit_estimate, dq_reduction_steps, use_sinks, data)` — [`L478`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L478) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel_test](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel_test.md)
  - `test_splash_attention_fwd(self, is_mqa, is_segmented, is_dynamic_mask, use_base2_exp, use_max_logit_estimate, fuse_reciprocal, use_sinks, data)` — [`L396`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L396) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel_test](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel_test.md)
- uses (calls/refs, reference-scoped): [`_generate_inputs`](splash_attention_kernel_test.md#_generate_inputs), [`block_sizes_strategy`](splash_attention_kernel_test.md#block_sizes_strategy), [`mask_strategy`](splash_attention_kernel_test.md#mask_strategy), [`_assert_allclose`](splash_attention_test_utils.md#SplashAttentionTestCase._assert_allclose), [`setUp`](splash_attention_test_utils.md#SplashAttentionTestCase.setUp), [`SplashAttentionTestCase`](splash_attention_test_utils.md#SplashAttentionTestCase), [`attention_reference`](base.md#attention_reference), [`check_mask_no_empty_rows`](splash_attention_kernel_test.md#check_mask_no_empty_rows), [`make_dynamic_splash_mha`](splash_attention_kernel.md#make_dynamic_splash_mha), [`make_dynamic_splash_mqa`](splash_attention_kernel.md#make_dynamic_splash_mqa), [`make_splash_mha_single_device`](splash_attention_kernel.md#make_splash_mha_single_device), [`make_splash_mqa_single_device`](splash_attention_kernel.md#make_splash_mqa_single_device), [`model_config_strategy`](splash_attention_kernel_test.md#model_config_strategy), [`RandomMask`](splash_attention_kernel_test.md#RandomMask), [`INTERPRET`](splash_attention_test_utils.md#SplashAttentionTestCase.INTERPRET), [`attention_reference_vjp`](base.md#attention_reference_vjp), [`thread_unsafe_test_class`](splash_attention_test_utils.md#thread_unsafe_test_class), [`attn_logits_soft_cap_strategy`](splash_attention_kernel_test.md#attn_logits_soft_cap_strategy), [`partial`](splash_attention_kernel_test.md#partial)
- used by: [`setUp`](splash_attention_test_utils.md#SplashAttentionTestCase.setUp)

### `SplitMask`  ·  implements/extends Mask
- def: [`src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py:92`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L92)
- signature: `class SplitMask(Mask):`
- members:
  - `get_mask(self)` — [`L96`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L96)
  - `kv_seq_len` — [`L94`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L94)
  - `q_seq_len` — [`L93`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L93)
- uses (calls/refs, reference-scoped): [`Mask`](splash_attention_mask.md#Mask), [`Mask`](splash_attention_kernel_test.md#Mask), [`NumpyMask`](splash_attention_mask.md#NumpyMask)
- used by: [`Mask`](splash_attention_kernel_test.md#Mask), [`get_mask`](splash_attention_kernel_test.md#Mask.get_mask), [`split_mask_strategy`](splash_attention_kernel_test.md#split_mask_strategy)

## Functions
- `_generate_inputs(data, config: ModelConfig, is_mqa: bool, is_segmented: bool, use_sinks: bool = False)` — [`L287`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L287) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel_test](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel_test.md)
- `attn_logits_soft_cap_strategy()` — [`L324`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L324)
- `block_sizes_strategy(draw: Draw, q_seq_len: int, kv_seq_len: int, include_bwd_blocks: bool = False)` — [`L248`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L248)
- `causal_mask_strategy(q_seq_len: int, kv_seq_len: int)` — [`L115`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L115)
- `check_mask_no_empty_rows(mask: mask_lib.Mask, segment_ids: splash.SegmentIds | None)` — [`L237`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L237) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel_test](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel_test.md)
- `compose_mask_strategy(draw: Draw, q_seq_len: int, kv_seq_len: int)` — [`L190`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L190)
- `full_mask_strategy(q_seq_len: int, kv_seq_len: int)` — [`L87`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L87)
- `local_attention_mask_strategy(draw: Draw, seq_len: int)` — [`L148`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L148)
- `mask_strategy(draw: Draw, q_seq_len: int, kv_seq_len: int)` — [`L198`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L198)
- `model_config_strategy(draw: Draw)` — [`L216`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L216)
- `random_mask_strategy(draw: Draw, q_seq_len: int, kv_seq_len: int)` — [`L172`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L172)
- `seed_strategy()` — [`L77`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L77)
- `segment_ids_strategy(draw, seq_len: int)` — [`L65`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L65)
- `split_mask_strategy(q_seq_len: int, kv_seq_len: int)` — [`L102`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L102)

## Module values
- `Draw` — [`L50`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L50)
- `partial` — [`L49`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel_test.py#L49)

