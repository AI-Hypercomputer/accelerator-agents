---
title: 'Module: tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_kernel_test`/
symbols:
  SplashAttentionTest.test_splash_attention_bwd: SplashAttentionTest#test_splash_attention_bwd().
  SplashAttentionTest.test_splash_attention: SplashAttentionTest#test_splash_attention().
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
  partial: partial.
  ModelConfig.q_seq_len: ModelConfig#q_seq_len.
  segment_ids_strategy: segment_ids_strategy().
  SplitMask: SplitMask#
  LocalAttentionMask: LocalAttentionMask#
  RandomMask: RandomMask#
  ComposeMask: ComposeMask#
  ComposeMask.left: ComposeMask#left.
  ComposeMask.right: ComposeMask#right.
  ComposeMask.op: ComposeMask#op.
  FullMask: FullMask#
  CausalMask: CausalMask#
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
# Module: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py)

## Classes
### `CausalMask`
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py:126`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L126)
- members:
  - `get_mask(self)` — [`L130`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L130)
  - `kv_seq_len` — [`L128`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L128)
  - `q_seq_len` — [`L127`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L127)
- uses (calls/refs, reference-scoped): [`Mask`](splash_attention_mask.md#Mask), [`Mask`](splash_attention_kernel_test.md#Mask), [`CausalMask`](splash_attention_mask.md#CausalMask)
- used by: [`get_mask`](splash_attention_kernel_test.md#Mask.get_mask), [`causal_mask_strategy`](splash_attention_kernel_test.md#causal_mask_strategy)

### `ComposeMask`  ·  implements/extends Mask
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py:192`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L192)
- signature: `class ComposeMask(Mask):`
- members:
  - `get_mask(self)` — [`L197`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L197)
  - `left` — [`L193`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L193)
  - `op` — [`L195`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L195)
  - `right` — [`L194`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L194)
- uses (calls/refs, reference-scoped): [`Mask`](splash_attention_mask.md#Mask), [`Mask`](splash_attention_kernel_test.md#Mask), [`get_mask`](splash_attention_kernel_test.md#Mask.get_mask)
- used by: [`Mask`](splash_attention_kernel_test.md#Mask), [`get_mask`](splash_attention_kernel_test.md#Mask.get_mask), [`compose_mask_strategy`](splash_attention_kernel_test.md#compose_mask_strategy)

### `FullMask`
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py:111`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L111)
- members:
  - `get_mask(self)` — [`L115`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L115)
  - `kv_seq_len` — [`L113`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L113)
  - `q_seq_len` — [`L112`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L112)
- uses (calls/refs, reference-scoped): [`Mask`](splash_attention_mask.md#Mask), [`Mask`](splash_attention_kernel_test.md#Mask), [`FullMask`](splash_attention_mask.md#FullMask)
- used by: [`get_mask`](splash_attention_kernel_test.md#Mask.get_mask), [`full_mask_strategy`](splash_attention_kernel_test.md#full_mask_strategy)

### `LocalAttentionMask`  ·  implements/extends Mask
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py:135`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L135)
- signature: `class LocalAttentionMask(Mask):`
- members:
  - `get_mask(self)` — [`L141`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L141)
  - `left` — [`L137`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L137)
  - `offset` — [`L139`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L139)
  - `right` — [`L138`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L138)
  - `seq_len` — [`L136`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L136)
- uses (calls/refs, reference-scoped): [`Mask`](splash_attention_mask.md#Mask), [`Mask`](splash_attention_kernel_test.md#Mask), [`NumpyMask`](splash_attention_mask.md#NumpyMask), [`LocalMask`](splash_attention_mask.md#LocalMask)
- used by: [`Mask`](splash_attention_kernel_test.md#Mask), [`get_mask`](splash_attention_kernel_test.md#Mask.get_mask), [`local_attention_mask_strategy`](splash_attention_kernel_test.md#local_attention_mask_strategy)

### `Mask`
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py:81`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L81)
- signature: `class Mask:`
- members:
  - `get_mask(self)` — [`L83`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L83)
- uses (calls/refs, reference-scoped): [`Mask`](splash_attention_mask.md#Mask), [`get_mask`](splash_attention_kernel_test.md#LocalAttentionMask.get_mask), [`get_mask`](splash_attention_kernel_test.md#RandomMask.get_mask), [`get_mask`](splash_attention_kernel_test.md#ComposeMask.get_mask), [`get_mask`](splash_attention_kernel_test.md#CausalMask.get_mask), [`get_mask`](splash_attention_kernel_test.md#FullMask.get_mask), [`get_mask`](splash_attention_kernel_test.md#SplitMask.get_mask), [`ComposeMask`](splash_attention_kernel_test.md#ComposeMask), [`LocalAttentionMask`](splash_attention_kernel_test.md#LocalAttentionMask), [`RandomMask`](splash_attention_kernel_test.md#RandomMask), [`SplitMask`](splash_attention_kernel_test.md#SplitMask)
- used by: [`mask_strategy`](splash_attention_kernel_test.md#mask_strategy), [`compose_mask_strategy`](splash_attention_kernel_test.md#compose_mask_strategy), [`get_mask`](splash_attention_kernel_test.md#ComposeMask.get_mask), [`local_attention_mask_strategy`](splash_attention_kernel_test.md#local_attention_mask_strategy), [`random_mask_strategy`](splash_attention_kernel_test.md#random_mask_strategy), [`causal_mask_strategy`](splash_attention_kernel_test.md#causal_mask_strategy), [`full_mask_strategy`](splash_attention_kernel_test.md#full_mask_strategy), [`split_mask_strategy`](splash_attention_kernel_test.md#split_mask_strategy), [`CausalMask`](splash_attention_kernel_test.md#CausalMask), [`ComposeMask`](splash_attention_kernel_test.md#ComposeMask), [`FullMask`](splash_attention_kernel_test.md#FullMask), [`LocalAttentionMask`](splash_attention_kernel_test.md#LocalAttentionMask), [`RandomMask`](splash_attention_kernel_test.md#RandomMask), [`SplitMask`](splash_attention_kernel_test.md#SplitMask), [`left`](splash_attention_kernel_test.md#ComposeMask.left), [`right`](splash_attention_kernel_test.md#ComposeMask.right)

### `ModelConfig`
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py:54`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L54)
- signature: `class ModelConfig:`
- members:
  - `dtype` — [`L61`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L61) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel_test](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel_test.md)
  - `head_dim_qk` — [`L59`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L59)
  - `head_dim_v` — [`L60`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L60)
  - `kv_seq_len` — [`L56`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L56) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel_test](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel_test.md)
  - `num_kv_heads` — [`L58`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L58)
  - `num_q_heads` — [`L57`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L57)
  - `q_seq_len` — [`L55`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L55) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel_test](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel_test.md)
- used by: [`_generate_inputs`](splash_attention_kernel_test.md#_generate_inputs), [`model_config_strategy`](splash_attention_kernel_test.md#model_config_strategy)

### `RandomMask`  ·  implements/extends Mask
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py:166`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L166)
- signature: `class RandomMask(Mask):`
- members:
  - `get_mask(self)` — [`L172`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L172)
  - `kv_seq_len` — [`L168`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L168)
  - `q_seq_len` — [`L167`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L167)
  - `seed` — [`L170`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L170)
  - `sparsity` — [`L169`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L169)
- uses (calls/refs, reference-scoped): [`Mask`](splash_attention_mask.md#Mask), [`Mask`](splash_attention_kernel_test.md#Mask), [`NumpyMask`](splash_attention_mask.md#NumpyMask), [`make_random_mask`](splash_attention_mask.md#make_random_mask)
- used by: [`Mask`](splash_attention_kernel_test.md#Mask), [`get_mask`](splash_attention_kernel_test.md#Mask.get_mask), [`random_mask_strategy`](splash_attention_kernel_test.md#random_mask_strategy)

### `SplashAttentionTest`
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py:368`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L368)
- signature: `class SplashAttentionTest(test_utils.SplashAttentionTestCase):`
- members:
  - `setUp(self)` — [`L370`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L370)
  - `test_splash_attention(self, is_mqa, is_segmented, is_dynamic_mask, data)` — [`L381`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L381) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel_test](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel_test.md)
  - `test_splash_attention_bwd(self, is_mqa, is_segmented, is_dynamic_mask, use_max_logit_estimate, dq_reduction_steps, use_sinks, save_residuals, data)` — [`L537`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L537) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel_test](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel_test.md)
  - `test_splash_attention_fwd(self, is_mqa, is_segmented, is_dynamic_mask, use_base2_exp, use_max_logit_estimate, fuse_reciprocal, use_sinks, data)` — [`L434`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L434) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel_test](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel_test.md)
- uses (calls/refs, reference-scoped): [`_generate_inputs`](splash_attention_kernel_test.md#_generate_inputs), [`block_sizes_strategy`](splash_attention_kernel_test.md#block_sizes_strategy), [`mask_strategy`](splash_attention_kernel_test.md#mask_strategy), [`_assert_allclose`](splash_attention_test_utils.md#SplashAttentionTestCase._assert_allclose), [`setUp`](splash_attention_test_utils.md#SplashAttentionTestCase.setUp), [`SplashAttentionTestCase`](splash_attention_test_utils.md#SplashAttentionTestCase), [`attention_reference`](base.md#attention_reference), [`check_mask_no_empty_rows`](splash_attention_kernel_test.md#check_mask_no_empty_rows), [`make_dynamic_splash_mha`](splash_attention_kernel.md#make_dynamic_splash_mha), [`make_dynamic_splash_mqa`](splash_attention_kernel.md#make_dynamic_splash_mqa), [`make_splash_mha_single_device`](splash_attention_kernel.md#make_splash_mha_single_device), [`make_splash_mqa_single_device`](splash_attention_kernel.md#make_splash_mqa_single_device), [`model_config_strategy`](splash_attention_kernel_test.md#model_config_strategy), [`INTERPRET`](splash_attention_test_utils.md#SplashAttentionTestCase.INTERPRET), [`partial`](splash_attention_kernel_test.md#partial), [`attention_reference_vjp`](base.md#attention_reference_vjp), [`thread_unsafe_test_class`](splash_attention_test_utils.md#thread_unsafe_test_class), [`attn_logits_soft_cap_strategy`](splash_attention_kernel_test.md#attn_logits_soft_cap_strategy)
- used by: [`setUp`](splash_attention_test_utils.md#SplashAttentionTestCase.setUp)

### `SplitMask`  ·  implements/extends Mask
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py:94`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L94)
- signature: `class SplitMask(Mask):`
- members:
  - `get_mask(self)` — [`L98`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L98)
  - `kv_seq_len` — [`L96`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L96)
  - `q_seq_len` — [`L95`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L95)
- uses (calls/refs, reference-scoped): [`Mask`](splash_attention_mask.md#Mask), [`Mask`](splash_attention_kernel_test.md#Mask), [`NumpyMask`](splash_attention_mask.md#NumpyMask)
- used by: [`Mask`](splash_attention_kernel_test.md#Mask), [`get_mask`](splash_attention_kernel_test.md#Mask.get_mask), [`split_mask_strategy`](splash_attention_kernel_test.md#split_mask_strategy)

## Functions
- `_generate_inputs(data, config: ModelConfig, is_mqa: bool, is_segmented: bool, use_sinks: bool = False)` — [`L319`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L319) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel_test](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel_test.md)
- `attn_logits_soft_cap_strategy()` — [`L363`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L363)
- `block_sizes_strategy(draw: Draw, q_seq_len: int, kv_seq_len: int, include_bwd_blocks: bool = False)` — [`L272`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L272)
- `causal_mask_strategy(q_seq_len: int, kv_seq_len: int)` — [`L119`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L119)
- `check_mask_no_empty_rows(mask: mask_lib.Mask, segment_ids: splash.SegmentIds | None)` — [`L259`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L259)
- `compose_mask_strategy(draw: Draw, q_seq_len: int, kv_seq_len: int)` — [`L202`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L202)
- `full_mask_strategy(q_seq_len: int, kv_seq_len: int)` — [`L87`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L87)
- `local_attention_mask_strategy(draw: Draw, seq_len: int)` — [`L154`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L154)
- `mask_strategy(draw: Draw, q_seq_len: int, kv_seq_len: int)` — [`L212`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L212)
- `model_config_strategy(draw: Draw)` — [`L230`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L230)
- `random_mask_strategy(draw: Draw, q_seq_len: int, kv_seq_len: int)` — [`L184`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L184)
- `seed_strategy()` — [`L77`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L77)
- `segment_ids_strategy(draw, seq_len: int)` — [`L65`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L65)
- `split_mask_strategy(q_seq_len: int, kv_seq_len: int)` — [`L104`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L104)

## Module values
- `Draw` — [`L50`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L50)
- `partial` — [`L49`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py#L49)

