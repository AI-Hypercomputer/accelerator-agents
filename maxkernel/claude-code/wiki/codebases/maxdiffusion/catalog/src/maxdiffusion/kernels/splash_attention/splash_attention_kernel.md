---
title: 'Module: src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.kernels.splash_attention.splash_attention_kernel`/
symbols:
  _splash_attention_forward: _splash_attention_forward().
  _splash_attention_forward_ring_raw: _splash_attention_forward_ring_raw().
  _splash_attention_bwd_dkv: _splash_attention_bwd_dkv().
  NUM_LANES: NUM_LANES.
  _make_splash_attention: _make_splash_attention().
  _make_dynamic_splash_attention: _make_dynamic_splash_attention().
  SplashAttentionKernel.manual_sharding_spec: SplashAttentionKernel#manual_sharding_spec().
  flash_attention_kernel: flash_attention_kernel().
  _splash_attention_bwd: _splash_attention_bwd().
  SplashConfig.get_default: SplashConfig#get_default().
  _flash_attention_dkv_kernel: _flash_attention_dkv_kernel().
  flash_attention_kernel.body: flash_attention_kernel().body().
  _splash_attention_custom: _splash_attention_custom().
  _flash_attention_dkv_kernel.body: _flash_attention_dkv_kernel().body().
  QKVLayout: QKVLayout#
  _flash_attention_dq_kernel.body: _flash_attention_dq_kernel().body().
  _splash_attention_fwd: _splash_attention_fwd().
  NUM_SUBLANES: NUM_SUBLANES.
  from_head_minor: from_head_minor().
  SplashConfig: SplashConfig#
  _flash_attention_dq_kernel: _flash_attention_dq_kernel().
  MaskInfo: MaskInfo.
  _splash_attention: _splash_attention().
  SplashConfig.k_layout: SplashConfig#k_layout.
  SplashConfig.use_base2_exp: SplashConfig#use_base2_exp.
  _make_dynamic_splash_attention.process_mask_shard: _make_dynamic_splash_attention().process_mask_shard().
  QKVLayout.HEAD_DIM_MINOR: QKVLayout#HEAD_DIM_MINOR.
  SplashConfig.q_layout: SplashConfig#q_layout.
  SplashConfig.v_layout: SplashConfig#v_layout.
  SplashConfig.__post_init__: SplashConfig#__post_init__().
  partial: partial.
  _splash_attention_bwd_dkv.unravel: _splash_attention_bwd_dkv().unravel().
  SplashConfig.has_backward_blocks: SplashConfig#has_backward_blocks().
  SplashAttentionKernel.__call__: SplashAttentionKernel#__call__().
  LOG2E: LOG2E.
  MaskFunctionType: MaskFunctionType.
  SplashAttentionKernel.fwd_mask_info: SplashAttentionKernel#fwd_mask_info.
  _apply_mask_and_soft_cap: _apply_mask_and_soft_cap().
  to_i32: to_i32().
  make_dynamic_splash_mha: make_dynamic_splash_mha.
  _splash_attention_forward.unravel: _splash_attention_forward().unravel().
  _splash_attention_forward_ring_raw.unravel: _splash_attention_forward_ring_raw().unravel().
  SplashConfig.block_kv: SplashConfig#block_kv.
  NN_DIM_NUMBERS: NN_DIM_NUMBERS.
  NT_DIM_NUMBERS: NT_DIM_NUMBERS.
  SplashConfig.block_q: SplashConfig#block_q.
  SplashConfig.block_kv_dkv: SplashConfig#block_kv_dkv.
  make_splash_mha: make_splash_mha.
  make_splash_mha_single_device: make_splash_mha_single_device.
  make_splash_mqa_single_device: make_splash_mqa_single_device.
  make_dynamic_splash_mqa: make_dynamic_splash_mqa.
  SplashAttentionKernel.tree_flatten: SplashAttentionKernel#tree_flatten().
  SplashConfig.block_q_dkv: SplashConfig#block_q_dkv.
  SplashConfig.block_kv_compute: SplashConfig#block_kv_compute.
  SplashConfig.dq_reduction_steps: SplashConfig#dq_reduction_steps.
  _bytes: _bytes().
  _splash_attention_forward.create_kv_index_map.index_map: _splash_attention_forward().create_kv_index_map().index_map().
  _splash_attention_forward_ring_raw.create_kv_index_map.index_map: _splash_attention_forward_ring_raw().create_kv_index_map().index_map().
  _splash_attention_bwd_dkv.create_kv_index_map.index_map: _splash_attention_bwd_dkv().create_kv_index_map().index_map().
  _splash_attention_bwd_dkv._bwd_cost_estimate: _splash_attention_bwd_dkv()._bwd_cost_estimate().
  SplashAttentionKernel: SplashAttentionKernel#
  SplashAttentionKernel._resolve_spec: SplashAttentionKernel#_resolve_spec().
  _make_dynamic_splash_attention.process_all_shards: _make_dynamic_splash_attention().process_all_shards().
  make_splash_mqa: make_splash_mqa.
  SplashConfig.block_kv_dkv_compute: SplashConfig#block_kv_dkv_compute.
  SplashConfig.max_logit_const: SplashConfig#max_logit_const.
  flash_attention_kernel.end: flash_attention_kernel().end().
  SplashAttentionKernel.tree_unflatten: SplashAttentionKernel#tree_unflatten().
  _splash_attention_forward.create_kv_index_map: _splash_attention_forward().create_kv_index_map().
  _splash_attention_forward_ring_raw.create_kv_index_map: _splash_attention_forward_ring_raw().create_kv_index_map().
  _splash_attention_bwd_dkv.mask_index_map: _splash_attention_bwd_dkv().mask_index_map().
  _splash_attention_bwd_dkv.create_kv_index_map: _splash_attention_bwd_dkv().create_kv_index_map().
  SplashConfig.attn_logits_soft_cap: SplashConfig#attn_logits_soft_cap.
  _div: _div().
  _splash_attention_forward.unravel.index_map: _splash_attention_forward().unravel().index_map().
  _splash_attention_forward.mask_index_map: _splash_attention_forward().mask_index_map().
  _splash_attention_forward._fwd_cost_estimate: _splash_attention_forward()._fwd_cost_estimate().
  _splash_attention_forward_ring_raw.unravel.index_map: _splash_attention_forward_ring_raw().unravel().index_map().
  _splash_attention_forward_ring_raw.mask_index_map: _splash_attention_forward_ring_raw().mask_index_map().
  _splash_attention_forward_ring_raw._fwd_cost_estimate: _splash_attention_forward_ring_raw()._fwd_cost_estimate().
  _splash_attention_bwd_dkv.unravel.index_map: _splash_attention_bwd_dkv().unravel().index_map().
  _splash_attention_bwd_dkv.create_dkv_index_map: _splash_attention_bwd_dkv().create_dkv_index_map().
  get_kernel_name: get_kernel_name().
  SplashConfig.residual_checkpoint_name: SplashConfig#residual_checkpoint_name.
  SplashConfig.interpret: SplashConfig#interpret.
  SplashConfig.use_experimental_scheduler: SplashConfig#use_experimental_scheduler.
  _splash_attention_forward.init_if_empty: _splash_attention_forward().init_if_empty().
  _splash_attention_forward_ring_raw.init_if_empty: _splash_attention_forward_ring_raw().init_if_empty().
  SplashAttentionKernel.kwargs: SplashAttentionKernel#kwargs.
  SplashAttentionKernel.dkv_mask_info: SplashAttentionKernel#dkv_mask_info.
  LOG2E_INV: LOG2E_INV.
  flash_attention_kernel.init: flash_attention_kernel().init().
  SplashAttentionKernel.__init__: SplashAttentionKernel#__init__().
  _not: _not().
  SegmentIds: SegmentIds#
  SplashConfig.fwd_cost_estimate: SplashConfig#fwd_cost_estimate.
  SplashConfig.fuse_reciprocal: SplashConfig#fuse_reciprocal.
  _apply_mask_and_soft_cap.cap_logits: _apply_mask_and_soft_cap().cap_logits().
  _flash_attention_dkv_kernel.body._load_kv: _flash_attention_dkv_kernel().body()._load_kv().
  SegmentIds.q: SegmentIds#q.
  SegmentIds.kv: SegmentIds#kv.
  SplashConfig.block_q_dq: SplashConfig#block_q_dq.
  SplashConfig.block_kv_dq: SplashConfig#block_kv_dq.
  SplashConfig.use_fused_bwd_kernel: SplashConfig#use_fused_bwd_kernel.
  SplashConfig.bwd_cost_estimate: SplashConfig#bwd_cost_estimate.
  P: P.
  QKVLayout.SEQ_MINOR: QKVLayout#SEQ_MINOR.
  _flash_attention_dq_kernel.init: _flash_attention_dq_kernel().init().
  _flash_attention_dq_kernel.end: _flash_attention_dq_kernel().end().
  _flash_attention_dkv_kernel.init: _flash_attention_dkv_kernel().init().
---
# Module: [`src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py)

## Classes
### `QKVLayout`  ·  implements/extends IntEnum
- def: [`src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py:102`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L102) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
- signature: `class QKVLayout(enum.IntEnum):`
- members:
  - `HEAD_DIM_MINOR` — [`L103`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L103)
  - `SEQ_MINOR` — [`L104`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L104)
- used by: [`_splash_attention_forward`](splash_attention_kernel.md#_splash_attention_forward), [`_splash_attention_forward_ring_raw`](splash_attention_kernel.md#_splash_attention_forward_ring_raw), [`_splash_attention_bwd_dkv`](splash_attention_kernel.md#_splash_attention_bwd_dkv), [`flash_attention_kernel`](splash_attention_kernel.md#flash_attention_kernel), [`_flash_attention_dkv_kernel`](splash_attention_kernel.md#_flash_attention_dkv_kernel), [`block_sizes_strategy`](splash_attention_kernel_test.md#block_sizes_strategy), [`_flash_attention_dq_kernel`](splash_attention_kernel.md#_flash_attention_dq_kernel), [`from_head_minor`](splash_attention_kernel.md#from_head_minor), [`k_layout`](splash_attention_kernel.md#SplashConfig.k_layout), [`q_layout`](splash_attention_kernel.md#SplashConfig.q_layout), [`v_layout`](splash_attention_kernel.md#SplashConfig.v_layout)

### `SegmentIds`  ·  implements/extends NamedTuple
- def: [`src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py:60`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L60)
- doc: SegmentIds for Q and KV sequences.
- signature: `class SegmentIds(NamedTuple):`
- members:
  - `kv` — [`L81`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L81)
  - `q` — [`L80`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L80)
- used by: [`_generate_inputs`](splash_attention_kernel_test.md#_generate_inputs), [`check_mask_no_empty_rows`](splash_attention_kernel_test.md#check_mask_no_empty_rows)

### `SplashAttentionKernel`
- def: [`src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py:2120`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L2120)
- signature: `class SplashAttentionKernel:`
- members:
  - `manual_sharding_spec(self, sharding: jax.sharding.NamedSharding)` — [`L2140`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L2140) — Returns a value that can be used as a shard_map partition spec for the kernel. — documented in [maxdiffusion-kernels-splash_attention-splash_attention_mask_info](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_mask_info.md)
  - `tree_flatten(self)` — [`L2172`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L2172)
  - `tree_unflatten(cls, kwargs, values)` — [`L2176`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L2176)
  - `dkv_mask_info` — [`L2130`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L2130)
  - `fwd_mask_info` — [`L2129`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L2129)
  - `kwargs` — [`L2128`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L2128)
- protocol/private: `__call__`[`L2132`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L2132), `__init__`[`L2122`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L2122), `_resolve_spec`[`L2152`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L2152)
- uses (calls/refs, reference-scoped): [`partial_mask_blocks`](splash_attention_mask_info.md#MaskInfo.partial_mask_blocks), [`block_mask`](splash_attention_mask_info.md#MaskInfo.block_mask), [`q_sequence`](splash_attention_mask_info.md#MaskInfo.q_sequence), [`active_rows`](splash_attention_mask_info.md#MaskInfo.active_rows), [`mask_next`](splash_attention_mask_info.md#MaskInfo.mask_next), [`active_cols`](splash_attention_mask_info.md#MaskInfo.active_cols), [`num_active_blocks`](splash_attention_mask_info.md#MaskInfo.num_active_blocks), [`MaskInfo`](splash_attention_kernel.md#MaskInfo), [`_splash_attention`](splash_attention_kernel.md#_splash_attention), [`SplashCustomReturnType`](base.md#SplashCustomReturnType.SplashCustomReturnType)
- used by: [`_make_dynamic_splash_attention`](splash_attention_kernel.md#_make_dynamic_splash_attention), [`_make_splash_attention`](splash_attention_kernel.md#_make_splash_attention), [`test_manual_partitioning_mha_bwd`](splash_attention_kernel_sharded_test.md#SplashAttentionShardingTest.test_manual_partitioning_mha_bwd), [`test_manual_partitioning_mha_fwd`](splash_attention_kernel_sharded_test.md#SplashAttentionShardingTest.test_manual_partitioning_mha_fwd)

### `SplashConfig`
- def: [`src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py:114`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L114) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
- doc: Tile sizes parameterizing SplashAttention kernels.
- signature: `class SplashConfig:`
- members:
  - `get_default(cls)` — [`L180`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L180)
  - `has_backward_blocks(self)` — [`L171`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L171)
  - `attn_logits_soft_cap` — [`L146`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L146)
  - `block_kv` — [`L126`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L126)
  - `block_kv_compute` — [`L127`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L127)
  - `block_kv_dkv` — [`L130`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L130)
  - `block_kv_dkv_compute` — [`L131`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L131)
  - `block_kv_dq` — [`L135`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L135)
  - `block_q` — [`L125`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L125)
  - `block_q_dkv` — [`L129`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L129)
  - `block_q_dq` — [`L134`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L134)
  - `bwd_cost_estimate` — [`L143`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L143)
  - `dq_reduction_steps` — [`L155`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L155)
  - `fuse_reciprocal` — [`L147`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L147)
  - `fwd_cost_estimate` — [`L142`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L142)
  - `interpret` — [`L150`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L150)
  - `k_layout` — [`L139`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L139) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
  - `max_logit_const` — [`L149`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L149) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
  - `q_layout` — [`L138`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L138) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
  - `residual_checkpoint_name` — [`L145`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L145)
  - `use_base2_exp` — [`L148`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L148) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
  - `use_experimental_scheduler` — [`L157`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L157)
  - `use_fused_bwd_kernel` — [`L136`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L136)
  - `v_layout` — [`L140`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L140) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
- protocol/private: `__post_init__`[`L159`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L159)
- uses (calls/refs, reference-scoped): [`QKVLayout`](splash_attention_kernel.md#QKVLayout), [`HEAD_DIM_MINOR`](splash_attention_kernel.md#QKVLayout.HEAD_DIM_MINOR)
- used by: [`_splash_attention_forward`](splash_attention_kernel.md#_splash_attention_forward), [`_splash_attention_forward_ring_raw`](splash_attention_kernel.md#_splash_attention_forward_ring_raw), [`_splash_attention_bwd_dkv`](splash_attention_kernel.md#_splash_attention_bwd_dkv), [`make_ring_attention`](ring_attention_kernel.md#make_ring_attention), [`test_ring_attention`](ring_attention_kernel_test.md#RingAttentionTest.test_ring_attention), [`_make_dynamic_splash_attention`](splash_attention_kernel.md#_make_dynamic_splash_attention), [`_make_splash_attention`](splash_attention_kernel.md#_make_splash_attention), [`_splash_attention_bwd`](splash_attention_kernel.md#_splash_attention_bwd), [`flash_attention_kernel`](splash_attention_kernel.md#flash_attention_kernel), [`_flash_attention_dkv_kernel`](splash_attention_kernel.md#_flash_attention_dkv_kernel), [`block_sizes_strategy`](splash_attention_kernel_test.md#block_sizes_strategy), [`_splash_attention_custom`](splash_attention_kernel.md#_splash_attention_custom), [`body`](splash_attention_kernel.md#_flash_attention_dkv_kernel.body), [`body`](splash_attention_kernel.md#flash_attention_kernel.body), [`_ring_attention_forward`](ring_attention_kernel.md#_ring_attention_forward), [`body`](splash_attention_kernel.md#_flash_attention_dq_kernel.body), [`_splash_attention_fwd`](splash_attention_kernel.md#_splash_attention_fwd), [`_flash_attention_dq_kernel`](splash_attention_kernel.md#_flash_attention_dq_kernel), [`_splash_attention`](splash_attention_kernel.md#_splash_attention), [`process_mask_shard`](splash_attention_kernel.md#_make_dynamic_splash_attention.process_mask_shard), [`SplashConfig`](ring_attention_kernel.md#SplashConfig), [`_bwd_cost_estimate`](splash_attention_kernel.md#_splash_attention_bwd_dkv._bwd_cost_estimate), [`end`](splash_attention_kernel.md#flash_attention_kernel.end), [`init`](splash_attention_kernel.md#flash_attention_kernel.init)

## Functions
- `_apply_mask_and_soft_cap(qk: jax.Array, mask_value: float, mask_ref, q_sequence_ref, q_segment_ids_ref, kv_segment_ids_ref, *, attn_logits_soft_cap: float | None, k_slice: pl.Slice, k_offset: int | jax.Array, bq: int, k_in_lanes=True, mask_function=None, has_partial_mask: bool = False)` — [`L199`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L199)
- `_bwd_cost_estimate(q: jax.Array, k: jax.Array, v: jax.Array, q_segment_ids: jax.Array | None, kv_segment_ids: jax.Array | None, logsumexp: jax.Array, do: jax.Array, di: jax.Array, partial_mask_blocks: jax.Array | None, q_sequence: jax.Array | None, out_shapes: list[jax.ShapeDtypeStruct], mask_sparsity_factor: float)` — [`L1896`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L1896)
- `_bytes(x: jax.Array | jax.ShapeDtypeStruct | None)` — [`L492`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L492)
- `_div(dividend: int, divisor: int)` — [`L485`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L485)
- `_flash_attention_dkv_kernel(active_rows_ref, active_cols_ref, mask_next_ref, bounds_start_ref, bounds_end_ref, block_mask_ref, q_ref, k_ref, v_ref, q_segment_ids_ref, kv_segment_ids_ref, logsumexp_ref, do_ref, di_ref, mask_ref, q_sequence_ref, dq_alias, dk_alias, dv_alias, dq_ref, dk_ref, dv_ref, dq_scratch_ref, dk_scratch_ref, dv_scratch_ref, *, mask_value: float, q_steps: int, bq: int, bkv_compute: int, bkv: int, mask_function: MaskFunctionType | None, q_heads_per_kv_head: int, config: SplashConfig)` — [`L1394`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L1394) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
- `_flash_attention_dq_kernel(active_rows_ref, active_cols_ref, mask_next_ref, bounds_start_ref, bounds_end_ref, block_mask_ref, q_ref, k_ref, v_ref, q_segment_ids_ref, kv_segment_ids_ref, logsumexp_ref, do_ref, di_ref, mask_ref, q_sequence_ref, dq_scratch_ref, dq_ref, *, mask_value: float, kv_steps: int, bq: int, bkv: int, mask_function: MaskFunctionType | None, config: SplashConfig)` — [`L1277`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L1277)
- `_fwd_cost_estimate(q: jax.Array, k: jax.Array, v: jax.Array, q_segment_ids: jax.Array | None, kv_segment_ids: jax.Array | None, partial_mask_blocks: jax.Array | None, out_shapes: list[jax.ShapeDtypeStruct], mask_sparsity: float)` — [`L717`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L717)
- `_fwd_cost_estimate(q: jax.Array, k: jax.Array, v: jax.Array, q_segment_ids: jax.Array | None, kv_segment_ids: jax.Array | None, partial_mask_blocks: jax.Array | None, out_shapes: list[jax.ShapeDtypeStruct | None], mask_sparsity: float)` — [`L1062`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L1062)
- `_load_kv(ref, layout)` — [`L1492`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L1492)
- `_make_dynamic_splash_attention(mask: jax.Array, *, mesh: jax.sharding.Mesh | None = None, mask_spec: jax.sharding.PartitionSpec | None = None, config: SplashConfig | None = None, is_mqa: bool, save_residuals: bool = False, mask_value: float = base.DEFAULT_MASK_VALUE, downcast_smem_data: bool = True, partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8)` — [`L2247`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L2247)
- `_make_splash_attention(mask: np.ndarray | mask_lib.Mask, *, config: SplashConfig | None = None, is_mqa: bool, save_residuals: bool = False, mask_value: float = base.DEFAULT_MASK_VALUE, downcast_smem_data: bool = True, partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8, q_seq_shards: int)` — [`L2183`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L2183) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_mask_info](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_mask_info.md)
- `_not(x: jax.Array | bool)` — [`L54`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L54)
- `_splash_attention(fwd_mask_info: MaskInfo, dkv_mask_info: MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: base.SegmentIds | None = None, sinks: jax.Array | None = None, *, is_mqa: bool, config: SplashConfig | None, save_residuals: bool, mask_value: float, max_logit_value: jax.Array | None = None, mask_function: MaskFunctionType | None, fwd_mask_sparsity: float, dkv_mask_sparsity: float)` — [`L2082`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L2082)
- `_splash_attention_bwd(save_residuals: bool, mask_value: float, is_mqa: bool, config: SplashConfig, mask_function: MaskFunctionType | None, fwd_mask_sparsity: float, dkv_mask_sparsity: float, res: base.SplashResidualsType, do: jax.Array)` — [`L1998`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L1998) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
- `_splash_attention_bwd_dkv(q, k, v, segment_ids, logsumexp, do, di, *, bq: int, bkv: int, bkv_compute: int, is_mqa: bool, mask_info: MaskInfo, mask_value: float, mask_function: MaskFunctionType | None, config: SplashConfig, dkv_mask_sparsity: float)` — [`L1605`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L1605) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
- `_splash_attention_custom(fwd_mask_info: MaskInfo, dkv_mask_info: MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: base.SegmentIds | None, sinks: jax.Array | None, save_residuals: bool, mask_value: float, is_mqa: bool, config: SplashConfig, mask_function: MaskFunctionType | None, fwd_mask_sparsity: float, dkv_mask_sparsity: float, max_logit_value: jax.Array | None = None)` — [`L1177`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L1177) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
- `_splash_attention_forward(mask_info: MaskInfo, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: base.SegmentIds | None, sinks: jax.Array | None, mask_value: float, is_mqa: bool, config: SplashConfig, save_residuals: bool, mask_function: MaskFunctionType | None, fwd_mask_sparsity: float, max_logit_value: jax.Array | None = None)` — [`L505`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L505) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
- `_splash_attention_forward_ring_raw(mask_info: MaskInfo, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: base.SegmentIds | None, sinks: jax.Array | None, mask_value: float, is_mqa: bool, config: SplashConfig, mask_function: MaskFunctionType | None, fwd_mask_sparsity: float, max_logit_value: jax.Array | None = None)` — [`L862`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L862) — Ring-specific forward path that returns pre-reciprocal fp32 accumulators. — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
- `_splash_attention_fwd(fwd_mask_info: MaskInfo, dkv_mask_info: MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: base.SegmentIds | None, sinks: jax.Array | None, save_residuals: bool, mask_value: float, is_mqa: bool, config: SplashConfig, mask_function: MaskFunctionType | None, fwd_mask_sparsity: float, dkv_mask_sparsity: float, max_logit_value: jax.Array | None = None)` — [`L1230`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L1230) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
- `body(kv_compute_index, _, has_partial_mask=False)` — [`L371`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L371)
- `body(has_partial_mask: bool = False)` — [`L1330`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L1330)
- `body(i, _, has_partial_mask=False)` — [`L1484`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L1484)
- `cap_logits(logits)` — [`L266`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L266)
- `create_dkv_index_map(h, i, j, *_)` — [`L1716`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L1716)
- `create_kv_index_map(layout)` — [`L592`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L592) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
- `create_kv_index_map(layout)` — [`L956`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L956) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
- `create_kv_index_map(layout)` — [`L1686`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L1686)
- `end()` — [`L463`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L463)
- `end()` — [`L1390`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L1390)
- `flash_attention_kernel(active_rows_ref, active_cols_ref, mask_next_ref, bounds_start_ref, bounds_end_ref, block_mask_ref, q_ref, k_ref, v_ref, q_segment_ids_ref, kv_segment_ids_ref, sinks_ref, mask_ref, q_sequence_ref, max_logit_value_ref, o_ref, logsumexp_ref, l_linear_ref, max_logits_ref, m_scratch_ref, l_scratch_ref, o_scratch_ref, *, mask_value: float, kv_steps: int, bq: int, bkv: int, bkv_compute: int, head_dim_v: int, mask_function: MaskFunctionType | None, fuse_reciprocal: bool, config: SplashConfig)` — [`L282`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L282) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
- `from_head_minor(vals: tuple[Any, ...], layout: QKVLayout)` — [`L107`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L107) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
- `get_kernel_name(is_mqa: bool, save_residuals: bool, is_segmented: bool, phase: str)` — [`L87`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L87) — Returns a unique name for all SplashAttention kernel variants.
- `index_map(h, grid_idx, rows_ref, cols_ref, *_)` — [`L581`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L581)
- `index_map(h, i, j)` — [`L593`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L593)
- `index_map(h, grid_idx, rows_ref, cols_ref, *_)` — [`L945`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L945)
- `index_map(h, i, j)` — [`L957`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L957)
- `index_map(h, grid_idx, rows_ref, cols_ref, *_)` — [`L1655`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L1655)
- `index_map(h, i, j, *_)` — [`L1687`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L1687)
- `init()` — [`L350`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L350)
- `init()` — [`L1327`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L1327)
- `init()` — [`L1480`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L1480)
- `init_if_empty(x: jax.Array, value: float)` — [`L824`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L824)
- `init_if_empty(x: jax.Array, value: float)` — [`L1149`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L1149)
- `mask_index_map(h, grid_idx, rows_ref, cols_ref, mask_next_ref=None, *_)` — [`L605`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L605)
- `mask_index_map(h, grid_idx, rows_ref, cols_ref, mask_next_ref=None, *_)` — [`L969`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L969)
- `mask_index_map(h, grid_idx, rows_ref, cols_ref, mask_next_ref=None, *_)` — [`L1665`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L1665)
- `process_all_shards(mask)` — [`L2326`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L2326)
- `process_mask_shard(mask)` — [`L2274`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L2274)
- `to_i32(x)` — [`L195`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L195)
- `unravel(f)` — [`L580`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L580) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
- `unravel(f)` — [`L944`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L944) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
- `unravel(f)` — [`L1654`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L1654)

## Module values
- `LOG2E` — [`L48`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L48)
- `LOG2E_INV` — [`L49`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L49)
- `MaskFunctionType` — [`L84`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L84)
- `MaskInfo` — [`L40`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L40)
- `NN_DIM_NUMBERS` — [`L45`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L45)
- `NT_DIM_NUMBERS` — [`L46`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L46)
- `NUM_LANES` — [`L42`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L42) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
- `NUM_SUBLANES` — [`L43`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L43) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md)
- `P` — [`L39`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L39)
- `make_dynamic_splash_mha` — [`L2344`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L2344) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel_test](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel_test.md)
- `make_dynamic_splash_mqa` — [`L2343`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L2343) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel_test](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel_test.md)
- `make_splash_mha` — [`L2336`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L2336)
- `make_splash_mha_single_device` — [`L2339`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L2339) — documented in [maxdiffusion-kernels-splash_attention-splash_attention_kernel_test](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel_test.md)
- `make_splash_mqa` — [`L2337`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L2337)
- `make_splash_mqa_single_device` — [`L2341`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L2341)
- `partial` — [`L41`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/splash_attention_kernel.py#L41)

