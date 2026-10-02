---
title: 'Module: tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_kernel`/
symbols:
  _splash_attention_forward: _splash_attention_forward().
  _splash_attention_bwd_dkv: _splash_attention_bwd_dkv().
  _make_splash_attention: _make_splash_attention().
  _make_dynamic_splash_attention: _make_dynamic_splash_attention().
  _splash_attention_bwd: _splash_attention_bwd().
  NUM_LANES: NUM_LANES.
  SplashAttentionKernel.manual_sharding_spec: SplashAttentionKernel#manual_sharding_spec().
  SplashConfig.get_default: SplashConfig#get_default().
  flash_attention_kernel: flash_attention_kernel().
  _flash_attention_dkv_kernel: _flash_attention_dkv_kernel().
  QKVLayout: QKVLayout#
  flash_attention_kernel.body: flash_attention_kernel().body().
  _splash_attention_custom: _splash_attention_custom().
  _flash_attention_dkv_kernel.body: _flash_attention_dkv_kernel().body().
  SplashConfig: SplashConfig#
  _flash_attention_dq_kernel.body: _flash_attention_dq_kernel().body().
  _splash_attention_fwd: _splash_attention_fwd().
  _flash_attention_dq_kernel: _flash_attention_dq_kernel().
  MaskInfo: MaskInfo.
  _splash_attention: _splash_attention().
  QKVLayout.HEAD_DIM_MINOR: QKVLayout#HEAD_DIM_MINOR.
  SplashConfig.use_base2_exp: SplashConfig#use_base2_exp.
  _make_dynamic_splash_attention.process_mask_shard: _make_dynamic_splash_attention().process_mask_shard().
  SplashConfig.k_layout: SplashConfig#k_layout.
  LOG2E: LOG2E.
  from_head_minor: from_head_minor().
  SplashConfig.__post_init__: SplashConfig#__post_init__().
  NUM_SUBLANES: NUM_SUBLANES.
  SplashConfig.q_layout: SplashConfig#q_layout.
  _splash_attention_bwd_dkv.unravel: _splash_attention_bwd_dkv().unravel().
  partial: partial.
  SplashConfig.v_layout: SplashConfig#v_layout.
  SplashConfig.has_backward_blocks: SplashConfig#has_backward_blocks().
  SplashAttentionKernel.__call__: SplashAttentionKernel#__call__().
  SplashAttentionKernel.fwd_mask_info: SplashAttentionKernel#fwd_mask_info.
  make_dynamic_splash_mha: make_dynamic_splash_mha.
  _apply_mask_and_soft_cap: _apply_mask_and_soft_cap().
  MaskFunctionType: MaskFunctionType.
  _splash_attention_forward.unravel: _splash_attention_forward().unravel().
  make_splash_mha: make_splash_mha.
  make_dynamic_splash_mqa: make_dynamic_splash_mqa.
  NN_DIM_NUMBERS: NN_DIM_NUMBERS.
  NT_DIM_NUMBERS: NT_DIM_NUMBERS.
  SplashConfig.block_kv: SplashConfig#block_kv.
  SplashConfig.block_kv_dkv: SplashConfig#block_kv_dkv.
  make_splash_mha_single_device: make_splash_mha_single_device.
  make_splash_mqa_single_device: make_splash_mqa_single_device.
  SplashAttentionKernel.tree_flatten: SplashAttentionKernel#tree_flatten().
  SplashConfig.block_q: SplashConfig#block_q.
  SplashConfig.block_q_dkv: SplashConfig#block_q_dkv.
  to_i32: to_i32.
  make_splash_mqa: make_splash_mqa.
  SplashConfig.dq_reduction_steps: SplashConfig#dq_reduction_steps.
  _splash_attention_forward.create_kv_index_map.index_map: _splash_attention_forward().create_kv_index_map().index_map().
  _splash_attention_bwd_dkv.create_kv_index_map.index_map: _splash_attention_bwd_dkv().create_kv_index_map().index_map().
  _splash_attention_bwd_dkv._bwd_cost_estimate: _splash_attention_bwd_dkv()._bwd_cost_estimate().
  SplashAttentionKernel: SplashAttentionKernel#
  _make_dynamic_splash_attention.process_all_shards: _make_dynamic_splash_attention().process_all_shards().
  SplashConfig.block_kv_compute: SplashConfig#block_kv_compute.
  SplashConfig.block_kv_dkv_compute: SplashConfig#block_kv_dkv_compute.
  flash_attention_kernel.init: flash_attention_kernel().init().
  flash_attention_kernel.end: flash_attention_kernel().end().
  SplashAttentionKernel.tree_unflatten: SplashAttentionKernel#tree_unflatten().
  _splash_attention_forward.create_kv_index_map: _splash_attention_forward().create_kv_index_map().
  _splash_attention_bwd_dkv.mask_index_map: _splash_attention_bwd_dkv().mask_index_map().
  _splash_attention_bwd_dkv.create_kv_index_map: _splash_attention_bwd_dkv().create_kv_index_map().
  SplashConfig.attn_logits_soft_cap: SplashConfig#attn_logits_soft_cap.
  _bytes: _bytes().
  _splash_attention_forward.unravel.index_map: _splash_attention_forward().unravel().index_map().
  _splash_attention_forward.mask_index_map: _splash_attention_forward().mask_index_map().
  _splash_attention_forward._fwd_cost_estimate: _splash_attention_forward()._fwd_cost_estimate().
  _splash_attention_bwd_dkv.unravel.index_map: _splash_attention_bwd_dkv().unravel().index_map().
  _splash_attention_bwd_dkv.create_dkv_index_map: _splash_attention_bwd_dkv().create_dkv_index_map().
  SplashConfig.residual_checkpoint_name: SplashConfig#residual_checkpoint_name.
  SplashConfig.max_logit_const: SplashConfig#max_logit_const.
  _div: _div().
  _splash_attention_forward.init_if_empty: _splash_attention_forward().init_if_empty().
  SplashAttentionKernel.kwargs: SplashAttentionKernel#kwargs.
  SplashAttentionKernel.dkv_mask_info: SplashAttentionKernel#dkv_mask_info.
  LOG2E_INV: LOG2E_INV.
  SplashAttentionKernel.__init__: SplashAttentionKernel#__init__().
  _not: _not().
  SegmentIds: SegmentIds#
  get_kernel_name: get_kernel_name().
  SplashConfig.fuse_reciprocal: SplashConfig#fuse_reciprocal.
  SplashConfig.interpret: SplashConfig#interpret.
  SplashConfig.use_experimental_scheduler: SplashConfig#use_experimental_scheduler.
  _apply_mask_and_soft_cap.cap_logits: _apply_mask_and_soft_cap().cap_logits().
  _flash_attention_dkv_kernel.body._load_kv: _flash_attention_dkv_kernel().body()._load_kv().
  SegmentIds.q: SegmentIds#q.
  SegmentIds.kv: SegmentIds#kv.
  QKVLayout.SEQ_MINOR: QKVLayout#SEQ_MINOR.
  SplashConfig.block_q_dq: SplashConfig#block_q_dq.
  SplashConfig.block_kv_dq: SplashConfig#block_kv_dq.
  SplashConfig.use_fused_bwd_kernel: SplashConfig#use_fused_bwd_kernel.
  SplashConfig.num_stacked_q_heads: SplashConfig#num_stacked_q_heads.
  SplashConfig.fwd_cost_estimate: SplashConfig#fwd_cost_estimate.
  SplashConfig.bwd_cost_estimate: SplashConfig#bwd_cost_estimate.
  P: P.
  _flash_attention_dq_kernel.init: _flash_attention_dq_kernel().init().
  _flash_attention_dq_kernel.end: _flash_attention_dq_kernel().end().
  _flash_attention_dkv_kernel.init: _flash_attention_dkv_kernel().init().
---
# Module: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py)

## Classes
### `QKVLayout`  ·  implements/extends IntEnum
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py:103`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L103) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel.md)
- signature: `class QKVLayout(enum.IntEnum):`
- members:
  - `HEAD_DIM_MINOR` — [`L104`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L104) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel.md)
  - `SEQ_MINOR` — [`L105`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L105)
- used by: [`_splash_attention_forward`](splash_attention_kernel.md#_splash_attention_forward), [`_splash_attention_bwd_dkv`](splash_attention_kernel.md#_splash_attention_bwd_dkv), [`_get_autotuning_configs`](../../../attention/pallas_mosaic_tpu.md#PallasMosaicTpuFlashAttention._get_autotuning_configs), [`flash_attention_kernel`](splash_attention_kernel.md#flash_attention_kernel), [`_get_heuristics_config`](../../../attention/pallas_mosaic_tpu.md#PallasMosaicTpuFlashAttention._get_heuristics_config), [`_flash_attention_dkv_kernel`](splash_attention_kernel.md#_flash_attention_dkv_kernel), [`block_sizes_strategy`](splash_attention_kernel_test.md#block_sizes_strategy), [`_flash_attention_dq_kernel`](splash_attention_kernel.md#_flash_attention_dq_kernel), [`k_layout`](splash_attention_kernel.md#SplashConfig.k_layout), [`from_head_minor`](splash_attention_kernel.md#from_head_minor), [`q_layout`](splash_attention_kernel.md#SplashConfig.q_layout), [`v_layout`](splash_attention_kernel.md#SplashConfig.v_layout), [`k_layout`](../../../attention/pallas_mosaic_tpu.md#Config.k_layout), [`q_layout`](../../../attention/pallas_mosaic_tpu.md#Config.q_layout), [`v_layout`](../../../attention/pallas_mosaic_tpu.md#Config.v_layout)

### `SegmentIds`  ·  implements/extends NamedTuple
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py:60`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L60)
- doc: SegmentIds for Q and KV sequences.
- signature: `class SegmentIds(NamedTuple):`
- members:
  - `kv` — [`L81`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L81)
  - `q` — [`L80`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L80)
- used by: [`_generate_inputs`](splash_attention_kernel_test.md#_generate_inputs), [`check_mask_no_empty_rows`](splash_attention_kernel_test.md#check_mask_no_empty_rows)

### `SplashAttentionKernel`
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py:2003`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L2003) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel.md)
- signature: `class SplashAttentionKernel:`
- members:
  - `manual_sharding_spec(self, sharding: jax.sharding.NamedSharding)` — [`L2023`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L2023) — Returns a value that can be used as a shard_map partition spec for the kernel.
  - `tree_flatten(self)` — [`L2055`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L2055)
  - `tree_unflatten(cls, kwargs, values)` — [`L2059`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L2059)
  - `dkv_mask_info` — [`L2013`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L2013)
  - `fwd_mask_info` — [`L2012`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L2012)
  - `kwargs` — [`L2011`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L2011)
- protocol/private: `__call__`[`L2015`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L2015), `__init__`[`L2005`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L2005)
- uses (calls/refs, reference-scoped): [`block_mask`](splash_attention_mask_info.md#MaskInfo.block_mask), [`partial_mask_blocks`](splash_attention_mask_info.md#MaskInfo.partial_mask_blocks), [`q_sequence`](splash_attention_mask_info.md#MaskInfo.q_sequence), [`active_rows`](splash_attention_mask_info.md#MaskInfo.active_rows), [`mask_next`](splash_attention_mask_info.md#MaskInfo.mask_next), [`active_cols`](splash_attention_mask_info.md#MaskInfo.active_cols), [`num_active_blocks`](splash_attention_mask_info.md#MaskInfo.num_active_blocks), [`MaskInfo`](splash_attention_kernel.md#MaskInfo), [`_splash_attention`](splash_attention_kernel.md#_splash_attention), [`SplashCustomReturnType`](base.md#SplashCustomReturnType.SplashCustomReturnType)
- used by: [`_make_dynamic_splash_attention`](splash_attention_kernel.md#_make_dynamic_splash_attention), [`_make_splash_attention`](splash_attention_kernel.md#_make_splash_attention), [`test_manual_partitioning_mha_bwd`](splash_attention_kernel_sharded_test.md#SplashAttentionShardingTest.test_manual_partitioning_mha_bwd), [`test_manual_partitioning_mha_fwd`](splash_attention_kernel_sharded_test.md#SplashAttentionShardingTest.test_manual_partitioning_mha_fwd)

### `SplashConfig`
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py:115`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L115) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel.md)
- doc: Tile sizes parameterizing SplashAttention kernels.
- signature: `class SplashConfig:`
- members:
  - `get_default(cls)` — [`L184`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L184)
  - `has_backward_blocks(self)` — [`L175`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L175)
  - `attn_logits_soft_cap` — [`L147`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L147)
  - `block_kv` — [`L127`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L127) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel.md)
  - `block_kv_compute` — [`L128`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L128)
  - `block_kv_dkv` — [`L131`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L131)
  - `block_kv_dkv_compute` — [`L132`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L132)
  - `block_kv_dq` — [`L136`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L136)
  - `block_q` — [`L126`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L126)
  - `block_q_dkv` — [`L130`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L130)
  - `block_q_dq` — [`L135`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L135)
  - `bwd_cost_estimate` — [`L144`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L144)
  - `dq_reduction_steps` — [`L156`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L156)
  - `fuse_reciprocal` — [`L148`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L148)
  - `fwd_cost_estimate` — [`L143`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L143)
  - `interpret` — [`L151`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L151)
  - `k_layout` — [`L140`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L140)
  - `max_logit_const` — [`L150`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L150)
  - `num_stacked_q_heads` — [`L138`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L138)
  - `q_layout` — [`L139`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L139)
  - `residual_checkpoint_name` — [`L146`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L146)
  - `use_base2_exp` — [`L149`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L149)
  - `use_experimental_scheduler` — [`L158`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L158)
  - `use_fused_bwd_kernel` — [`L137`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L137)
  - `v_layout` — [`L141`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L141)
- protocol/private: `__post_init__`[`L160`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L160)
- uses (calls/refs, reference-scoped): [`QKVLayout`](splash_attention_kernel.md#QKVLayout), [`HEAD_DIM_MINOR`](splash_attention_kernel.md#QKVLayout.HEAD_DIM_MINOR)
- used by: [`_splash_attention_forward`](splash_attention_kernel.md#_splash_attention_forward), [`_splash_attention_bwd_dkv`](splash_attention_kernel.md#_splash_attention_bwd_dkv), [`make_ring_attention`](ring_attention_kernel.md#make_ring_attention), [`_fwd`](../../../attention/pallas_mosaic_tpu_vjp.md#PallasMosaicTpuFlashAttentionVjp._fwd), [`test_ring_attention`](ring_attention_kernel_test.md#RingAttentionTest.test_ring_attention), [`_make_dynamic_splash_attention`](splash_attention_kernel.md#_make_dynamic_splash_attention), [`_make_splash_attention`](splash_attention_kernel.md#_make_splash_attention), [`_splash_attention_bwd`](splash_attention_kernel.md#_splash_attention_bwd), [`flash_attention_kernel`](splash_attention_kernel.md#flash_attention_kernel), [`_fwd`](../../../attention/pallas_mosaic_tpu.md#PallasMosaicTpuFlashAttention._fwd), [`build_splash_kernel`](../../../attention/pallas_mosaic_tpu_common.md#build_splash_kernel), [`_flash_attention_dkv_kernel`](splash_attention_kernel.md#_flash_attention_dkv_kernel), [`block_sizes_strategy`](splash_attention_kernel_test.md#block_sizes_strategy), [`_splash_attention_custom`](splash_attention_kernel.md#_splash_attention_custom), [`body`](splash_attention_kernel.md#_flash_attention_dkv_kernel.body), [`body`](splash_attention_kernel.md#flash_attention_kernel.body), [`body`](splash_attention_kernel.md#_flash_attention_dq_kernel.body), [`_splash_attention_fwd`](splash_attention_kernel.md#_splash_attention_fwd), [`_flash_attention_dq_kernel`](splash_attention_kernel.md#_flash_attention_dq_kernel), [`_splash_attention`](splash_attention_kernel.md#_splash_attention), [`process_mask_shard`](splash_attention_kernel.md#_make_dynamic_splash_attention.process_mask_shard), [`SplashConfig`](ring_attention_kernel.md#SplashConfig), [`_bwd_cost_estimate`](splash_attention_kernel.md#_splash_attention_bwd_dkv._bwd_cost_estimate), [`end`](splash_attention_kernel.md#flash_attention_kernel.end), [`init`](splash_attention_kernel.md#flash_attention_kernel.init)

## Functions
- `_apply_mask_and_soft_cap(qk: jax.Array, mask_value: float, mask_ref, q_sequence_ref, q_segment_ids_ref, kv_segment_ids_ref, *, attn_logits_soft_cap: float | None, k_slice: pl.Slice, k_offset: int | jax.Array, bq: int, k_in_lanes=True, mask_function=None, has_partial_mask: bool = False)` — [`L202`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L202)
- `_bwd_cost_estimate(q: jax.Array, k: jax.Array, v: jax.Array, q_segment_ids: jax.Array | None, kv_segment_ids: jax.Array | None, logsumexp: jax.Array, do: jax.Array, di: jax.Array, partial_mask_blocks: jax.Array | None, q_sequence: jax.Array | None, out_shapes: list[jax.ShapeDtypeStruct], mask_sparsity_factor: float)` — [`L1763`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L1763)
- `_bytes(x: jax.Array | jax.ShapeDtypeStruct | None)` — [`L537`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L537)
- `_div(dividend: int, divisor: int)` — [`L530`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L530)
- `_flash_attention_dkv_kernel(active_rows_ref, active_cols_ref, mask_next_ref, bounds_start_ref, bounds_end_ref, block_mask_ref, q_ref, k_ref, v_ref, q_segment_ids_ref, kv_segment_ids_ref, logsumexp_ref, do_ref, di_ref, mask_ref, q_sequence_ref, dq_alias, dk_alias, dv_alias, dq_ref, dk_ref, dv_ref, dq_scratch_ref, dk_scratch_ref, dv_scratch_ref, *, mask_value: float, q_steps: int, bq: int, bkv_compute: int, bkv: int, mask_function: MaskFunctionType | None, q_heads_per_kv_head: int, config: SplashConfig)` — [`L1234`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L1234)
- `_flash_attention_dq_kernel(active_rows_ref, active_cols_ref, mask_next_ref, bounds_start_ref, bounds_end_ref, block_mask_ref, q_ref, k_ref, v_ref, q_segment_ids_ref, kv_segment_ids_ref, logsumexp_ref, do_ref, di_ref, mask_ref, q_sequence_ref, dq_scratch_ref, dq_ref, *, mask_value: float, kv_steps: int, bq: int, bkv: int, mask_function: MaskFunctionType | None, config: SplashConfig)` — [`L1111`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L1111)
- `_fwd_cost_estimate(q: jax.Array, k: jax.Array, v: jax.Array, q_segment_ids: jax.Array | None, kv_segment_ids: jax.Array | None, partial_mask_blocks: jax.Array | None, out_shapes: list[jax.ShapeDtypeStruct], mask_sparsity: float)` — [`L832`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L832)
- `_load_kv(ref, layout)` — [`L1337`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L1337)
- `_make_dynamic_splash_attention(mask: jax.Array, *, mesh: jax.sharding.Mesh | None = None, mask_spec: jax.sharding.PartitionSpec | None = None, config: SplashConfig | None = None, is_mqa: bool, save_residuals: bool = False, mask_value: float = base.DEFAULT_MASK_VALUE, downcast_smem_data: bool = True, partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8)` — [`L2134`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L2134) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel.md)
- `_make_splash_attention(mask: np.ndarray | mask_lib.Mask, *, config: SplashConfig | None = None, is_mqa: bool, save_residuals: bool = False, mask_value: float = base.DEFAULT_MASK_VALUE, downcast_smem_data: bool = True, partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8, q_seq_shards: int)` — [`L2070`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L2070) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel.md)
- `_not(x: jax.Array | bool)` — [`L54`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L54)
- `_splash_attention(fwd_mask_info: MaskInfo, dkv_mask_info: MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: base.SegmentIds | None = None, sinks: jax.Array | None = None, *, is_mqa: bool, config: SplashConfig | None, save_residuals: bool, mask_value: float, max_logit_value: jax.Array | None = None, mask_function: MaskFunctionType | None, fwd_mask_sparsity: float, dkv_mask_sparsity: float)` — [`L1965`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L1965)
- `_splash_attention_bwd(save_residuals: bool, mask_value: float, is_mqa: bool, config: SplashConfig, mask_function: MaskFunctionType | None, fwd_mask_sparsity: float, dkv_mask_sparsity: float, res: base.SplashResidualsType, grads: jax.Array | tuple[jax.Array, dict[str, jax.Array]])` — [`L1871`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L1871)
- `_splash_attention_bwd_dkv(q, k, v, segment_ids, logsumexp, do, di, *, bq: int, bkv: int, bkv_compute: int, is_mqa: bool, mask_info: MaskInfo, mask_value: float, mask_function: MaskFunctionType | None, config: SplashConfig, dkv_mask_sparsity: float)` — [`L1462`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L1462) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel.md)
- `_splash_attention_custom(fwd_mask_info: MaskInfo, dkv_mask_info: MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: base.SegmentIds | None, sinks: jax.Array | None, save_residuals: bool, mask_value: float, is_mqa: bool, config: SplashConfig, mask_function: MaskFunctionType | None, fwd_mask_sparsity: float, dkv_mask_sparsity: float, max_logit_value: jax.Array | None = None)` — [`L1010`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L1010)
- `_splash_attention_forward(mask_info: MaskInfo, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: base.SegmentIds | None, sinks: jax.Array | None, mask_value: float, is_mqa: bool, config: SplashConfig, save_residuals: bool, mask_function: MaskFunctionType | None, fwd_mask_sparsity: float, max_logit_value: jax.Array | None = None)` — [`L550`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L550) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel.md)
- `_splash_attention_fwd(fwd_mask_info: MaskInfo, dkv_mask_info: MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: base.SegmentIds | None, sinks: jax.Array | None, save_residuals: bool, mask_value: float, is_mqa: bool, config: SplashConfig, mask_function: MaskFunctionType | None, fwd_mask_sparsity: float, dkv_mask_sparsity: float, max_logit_value: jax.Array | None = None)` — [`L1063`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L1063)
- `body(kv_compute_index, _, has_partial_mask=False)` — [`L397`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L397)
- `body(has_partial_mask: bool = False)` — [`L1164`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L1164)
- `body(i, _, has_partial_mask=False)` — [`L1328`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L1328)
- `cap_logits(logits)` — [`L280`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L280)
- `create_dkv_index_map(h, i, j, *_)` — [`L1577`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L1577)
- `create_kv_index_map(layout)` — [`L675`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L675)
- `create_kv_index_map(layout)` — [`L1545`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L1545)
- `end()` — [`L508`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L508)
- `end()` — [`L1230`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L1230)
- `flash_attention_kernel(active_rows_ref, active_cols_ref, mask_next_ref, bounds_start_ref, bounds_end_ref, block_mask_ref, q_ref, k_ref, v_ref, q_segment_ids_ref, kv_segment_ids_ref, sinks_ref, mask_ref, q_sequence_ref, max_logit_value_ref, o_ref, logsumexp_ref, l_linear_ref, max_logits_ref, m_scratch_ref, l_scratch_ref, o_scratch_ref, *, mask_value: float, kv_steps: int, bq: int, bkv: int, bkv_compute: int, head_dim_v: int, num_stacked_q_heads: int, mask_function: MaskFunctionType | None, fuse_reciprocal: bool, config: SplashConfig)` — [`L299`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L299)
- `from_head_minor(vals: tuple[Any, ...], layout: QKVLayout)` — [`L108`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L108)
- `get_kernel_name(is_mqa: bool, save_residuals: bool, is_segmented: bool, phase: str)` — [`L86`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L86) — Returns a unique name for all SplashAttention kernel variants.
- `index_map(h_block, grid_idx, rows_ref, cols_ref, *_)` — [`L664`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L664)
- `index_map(h_block, i, j)` — [`L676`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L676)
- `index_map(h, grid_idx, rows_ref, cols_ref, *_)` — [`L1515`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L1515)
- `index_map(h, i, j, *_)` — [`L1546`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L1546)
- `init()` — [`L372`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L372)
- `init()` — [`L1161`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L1161)
- `init()` — [`L1324`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L1324)
- `init_if_empty(x: jax.Array, value: float)` — [`L956`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L956)
- `mask_index_map(h_block, grid_idx, rows_ref, cols_ref, mask_next_ref=None, *_)` — [`L691`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L691)
- `mask_index_map(h, grid_idx, rows_ref, cols_ref, mask_next_ref=None, *_)` — [`L1525`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L1525)
- `process_all_shards(mask)` — [`L2219`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L2219)
- `process_mask_shard(mask)` — [`L2163`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L2163)
- `unravel(f)` — [`L663`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L663)
- `unravel(f)` — [`L1514`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L1514)

## Module values
- `LOG2E` — [`L48`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L48) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel.md)
- `LOG2E_INV` — [`L49`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L49)
- `MaskFunctionType` — [`L83`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L83)
- `MaskInfo` — [`L40`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L40)
- `NN_DIM_NUMBERS` — [`L45`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L45)
- `NT_DIM_NUMBERS` — [`L46`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L46)
- `NUM_LANES` — [`L42`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L42)
- `NUM_SUBLANES` — [`L43`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L43)
- `P` — [`L39`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L39)
- `make_dynamic_splash_mha` — [`L2237`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L2237)
- `make_dynamic_splash_mqa` — [`L2236`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L2236)
- `make_splash_mha` — [`L2229`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L2229)
- `make_splash_mha_single_device` — [`L2232`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L2232)
- `make_splash_mqa` — [`L2230`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L2230)
- `make_splash_mqa_single_device` — [`L2234`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L2234)
- `partial` — [`L41`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L41)
- `to_i32` — [`L199`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py#L199)

