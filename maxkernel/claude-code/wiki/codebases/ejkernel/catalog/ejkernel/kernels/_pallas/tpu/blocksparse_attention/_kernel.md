---
title: 'Module: ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.blocksparse_attention._kernel`/
symbols:
  blocksparse_attention: blocksparse_attention().
  _splash_attention_bwd_dkv: _splash_attention_bwd_dkv().
  _splash_attention_bwd_dq: _splash_attention_bwd_dq().
  _make_splash_attention: _make_splash_attention().
  _splash_attention_bwd: _splash_attention_bwd().
  QKVLayout: QKVLayout#
  NUM_LANES: NUM_LANES.
  SplashAttentionKernel.manual_sharding_spec: SplashAttentionKernel#manual_sharding_spec().
  BlockSizes.get_default: BlockSizes#get_default().
  from_head_minor: from_head_minor().
  _splash_attention: _splash_attention().
  BlockSizes.has_backward_blocks: BlockSizes#has_backward_blocks().
  _splash_attention_custom: _splash_attention_custom().
  SegmentIds: SegmentIds#
  make_attention_reference._wrapped: make_attention_reference()._wrapped().
  _next_nonzero: _next_nonzero().
  BlockSizes.__post_init__: BlockSizes#__post_init__().
  QKVLayout.HEAD_DIM_MINOR: QKVLayout#HEAD_DIM_MINOR.
  _splash_attention_fwd: _splash_attention_fwd().
  _splash_attention_forward: _splash_attention_forward().
  SplashAttentionKernel.__call__: SplashAttentionKernel#__call__().
  flash_attention_kernel: flash_attention_kernel().
  partial: partial.
  MaskFunctionType: MaskFunctionType.
  BlockSizes: BlockSizes#
  NUM_SUBLANES: NUM_SUBLANES.
  _attention_reference_custom: _attention_reference_custom.
  attention_reference: attention_reference().
  flash_attention_kernel.body: flash_attention_kernel().body().
  _flash_attention_dq_kernel: _flash_attention_dq_kernel().
  _flash_attention_dkv_kernel: _flash_attention_dkv_kernel().
  _flash_attention_dkv_kernel.body: _flash_attention_dkv_kernel().body().
  _apply_mask_and_soft_cap: _apply_mask_and_soft_cap().
  SplashAttentionKernel.fwd_mask_info: SplashAttentionKernel#fwd_mask_info.
  SplashAttentionKernel.tree_flatten: SplashAttentionKernel#tree_flatten().
  attention_reference_custom: attention_reference_custom().
  BlockSizes.block_kv_dkv: BlockSizes#block_kv_dkv.
  make_splash_mha: make_splash_mha.
  _attention_reference_default: _attention_reference_default().
  _splash_attention_forward.k_index_map: _splash_attention_forward().k_index_map().
  _splash_attention_forward.v_index_map: _splash_attention_forward().v_index_map().
  _splash_attention_bwd_dq.k_index_map: _splash_attention_bwd_dq().k_index_map().
  _splash_attention_bwd_dq.v_index_map: _splash_attention_bwd_dq().v_index_map().
  make_splash_mqa_single_device: make_splash_mqa_single_device.
  NN_DIM_NUMBERS: NN_DIM_NUMBERS.
  NT_DIM_NUMBERS: NT_DIM_NUMBERS.
  make_attention_reference: make_attention_reference().
  BlockSizes.block_kv: BlockSizes#block_kv.
  BlockSizes.block_q_dkv: BlockSizes#block_q_dkv.
  BlockSizes.q_layout: BlockSizes#q_layout.
  BlockSizes.k_layout: BlockSizes#k_layout.
  BlockSizes.v_layout: BlockSizes#v_layout.
  _splash_attention._collapse_partial_mask_blocks: _splash_attention()._collapse_partial_mask_blocks().
  _attention_reference: _attention_reference().
  _flash_attention_dq_kernel.run: _flash_attention_dq_kernel().run().
  SplashResidualsType: SplashResidualsType.
  BlockSizes.block_q: BlockSizes#block_q.
  BlockSizes.block_q_dq: BlockSizes#block_q_dq.
  BlockSizes.block_kv_dq: BlockSizes#block_kv_dq.
  _div: _div().
  SegmentIds.q: SegmentIds#q.
  SegmentIds.kv: SegmentIds#kv.
  SplashCustomReturnType: SplashCustomReturnType.
  _attention_reference_custom_fwd: _attention_reference_custom_fwd().
  make_masked_mha_reference: make_masked_mha_reference.
  make_masked_mqa_reference: make_masked_mqa_reference.
  BlockSizes.block_kv_dkv_compute: BlockSizes#block_kv_dkv_compute.
  BlockSizes.use_fused_bwd_kernel: BlockSizes#use_fused_bwd_kernel.
  _splash_attention_bwd_dkv.q_index_map: _splash_attention_bwd_dkv().q_index_map().
  _splash_attention_bwd_dkv.k_index_map: _splash_attention_bwd_dkv().k_index_map().
  _splash_attention_bwd_dkv.v_index_map: _splash_attention_bwd_dkv().v_index_map().
  make_splash_mqa: make_splash_mqa.
  make_splash_mha_single_device: make_splash_mha_single_device.
  blocksparse_attention.attn_static_fn: blocksparse_attention().attn_static_fn().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  BlockSizes.block_kv_compute: BlockSizes#block_kv_compute.
  SplashAttentionKernel.tree_unflatten: SplashAttentionKernel#tree_unflatten().
  _splash_attention_bwd_dkv.dkv_index_map: _splash_attention_bwd_dkv().dkv_index_map().
  _splash_attention_bwd_dkv.q_segment_ids_index_map: _splash_attention_bwd_dkv().q_segment_ids_index_map().
  _splash_attention_bwd_dkv.logsumexp_index_map: _splash_attention_bwd_dkv().logsumexp_index_map().
  _next_nonzero.to_i32: _next_nonzero().to_i32().
  _splash_attention_forward.q_index_map: _splash_attention_forward().q_index_map().
  _splash_attention_forward.mask_index_map: _splash_attention_forward().mask_index_map().
  _splash_attention_forward.kv_segment_ids_index_map: _splash_attention_forward().kv_segment_ids_index_map().
  _splash_attention_bwd_dq.q_index_map: _splash_attention_bwd_dq().q_index_map().
  _splash_attention_bwd_dq.mask_index_map: _splash_attention_bwd_dq().mask_index_map().
  _splash_attention_bwd_dq.kv_segment_ids_index_map: _splash_attention_bwd_dq().kv_segment_ids_index_map().
  _splash_attention_bwd_dkv.o_index_map: _splash_attention_bwd_dkv().o_index_map().
  _splash_attention_bwd_dkv.mask_index_map: _splash_attention_bwd_dkv().mask_index_map().
  get_kernel_name: get_kernel_name().
  SplashAttentionKernel: SplashAttentionKernel#
  SplashAttentionKernel.kwargs: SplashAttentionKernel#kwargs.
  SplashAttentionKernel.dq_mask_info: SplashAttentionKernel#dq_mask_info.
  SplashAttentionKernel.dkv_mask_info: SplashAttentionKernel#dkv_mask_info.
  flash_attention_kernel.run: flash_attention_kernel().run().
  flash_attention_kernel.end: flash_attention_kernel().end().
  _flash_attention_dkv_kernel.run: _flash_attention_dkv_kernel().run().
  SplashAttentionKernel.__init__: SplashAttentionKernel#__init__().
  make_attention_reference._wrapped.reshape_activations: make_attention_reference()._wrapped().reshape_activations().
  _apply_mask_and_soft_cap.cap_logits: _apply_mask_and_soft_cap().cap_logits().
  _splash_attention_forward.q_segment_ids_index_map: _splash_attention_forward().q_segment_ids_index_map().
  _splash_attention_bwd_dq.q_segment_ids_index_map: _splash_attention_bwd_dq().q_segment_ids_index_map().
  _splash_attention_bwd_dq.logsumexp_index_map: _splash_attention_bwd_dq().logsumexp_index_map().
  _flash_attention_dkv_kernel.body._load_kv: _flash_attention_dkv_kernel().body()._load_kv().
  _attention_reference_custom_bwd: _attention_reference_custom_bwd().
  make_attention_reference._wrapped.reshape_residuals: make_attention_reference()._wrapped().reshape_residuals().
  _splash_attention_forward.out_index_map: _splash_attention_forward().out_index_map().
  _splash_attention_forward.logsumexp_index_map: _splash_attention_forward().logsumexp_index_map().
  _splash_attention_bwd_dq.o_index_map: _splash_attention_bwd_dq().o_index_map().
  _splash_attention_bwd_dkv.dq_index_map: _splash_attention_bwd_dkv().dq_index_map().
  _splash_attention_bwd_dkv.kv_segment_ids_index_map: _splash_attention_bwd_dkv().kv_segment_ids_index_map().
  QKVLayout.SEQ_MINOR: QKVLayout#SEQ_MINOR.
  flash_attention_kernel.init: flash_attention_kernel().init().
  _flash_attention_dq_kernel.init: _flash_attention_dq_kernel().init().
  _flash_attention_dq_kernel.end: _flash_attention_dq_kernel().end().
  _flash_attention_dkv_kernel.init: _flash_attention_dkv_kernel().init().
  _flash_attention_dkv_kernel.end: _flash_attention_dkv_kernel().end().
---
# Module: [`ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py)

## Classes
### `BlockSizes`
- def: [`ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py:657`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L657) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
- doc: Tile sizes parameterizing SplashAttention kernels.
- signature: `class BlockSizes:`
- members:
  - `get_default(cls)` — [`L706`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L706) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
  - `has_backward_blocks(self)` — [`L695`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L695) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
  - `block_kv` — [`L669`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L669) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
  - `block_kv_compute` — [`L670`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L670)
  - `block_kv_dkv` — [`L673`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L673) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
  - `block_kv_dkv_compute` — [`L674`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L674) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
  - `block_kv_dq` — [`L677`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L677) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
  - `block_q` — [`L668`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L668) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
  - `block_q_dkv` — [`L672`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L672) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
  - `block_q_dq` — [`L676`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L676) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
  - `k_layout` — [`L682`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L682)
  - `q_layout` — [`L681`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L681)
  - `use_fused_bwd_kernel` — [`L679`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L679) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
  - `v_layout` — [`L683`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L683)
- protocol/private: `__post_init__`[`L685`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L685)
- uses (calls/refs, reference-scoped): [`QKVLayout`](_kernel.md#QKVLayout), [`HEAD_DIM_MINOR`](_kernel.md#QKVLayout.HEAD_DIM_MINOR)
- used by: [`blocksparse_attention`](_kernel.md#blocksparse_attention), [`_make_splash_attention`](_kernel.md#_make_splash_attention), [`_splash_attention_bwd`](_kernel.md#_splash_attention_bwd), [`_make_block_sizes`](../ring_attention/_pallas_impl_fwd.md#_make_block_sizes), [`make_ring_attention`](../ring_attention/_pallas_impl_bwd.md#make_ring_attention), [`ring_splash_attention`](../ring_attention/_pallas_impl_bwd.md#ring_splash_attention), [`_splash_attention`](_kernel.md#_splash_attention), [`_splash_attention_custom`](_kernel.md#_splash_attention_custom), [`single_batch_attention`](../ring_attention/_pallas_impl_fwd.md#ring_attention.single_batch_attention), [`BlockSizes`](../ring_attention/_pallas_impl_bwd.md#BlockSizes), [`_splash_attention_fwd`](_kernel.md#_splash_attention_fwd), [`_splash_attention_forward`](_kernel.md#_splash_attention_forward)

### `QKVLayout`  ·  implements/extends IntEnum
- def: [`ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py:622`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L622) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
- doc: Physical memory layout for Q, K, V tensors in Splash Attention.
- signature: `class QKVLayout(enum.IntEnum):`
- members:
  - `HEAD_DIM_MINOR` — [`L634`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L634) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
  - `SEQ_MINOR` — [`L635`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L635)
- used by: [`_splash_attention_bwd_dkv`](_kernel.md#_splash_attention_bwd_dkv), [`_splash_attention_bwd_dq`](_kernel.md#_splash_attention_bwd_dq), [`from_head_minor`](_kernel.md#from_head_minor), [`flash_attention_kernel`](_kernel.md#flash_attention_kernel), [`_flash_attention_dkv_kernel`](_kernel.md#_flash_attention_dkv_kernel), [`_flash_attention_dq_kernel`](_kernel.md#_flash_attention_dq_kernel), [`k_layout`](_kernel.md#BlockSizes.k_layout), [`q_layout`](_kernel.md#BlockSizes.q_layout), [`v_layout`](_kernel.md#BlockSizes.v_layout)

### `SegmentIds`  ·  implements/extends NamedTuple
- def: [`ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py:99`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L99) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
- doc: SegmentIds for Q and KV sequences.
- signature: `class SegmentIds(NamedTuple):`
- members:
  - `kv` — [`L120`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L120)
  - `q` — [`L119`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L119)
- used by: [`_splash_attention_bwd`](_kernel.md#_splash_attention_bwd), [`_ring_attention_forward`](../ring_attention/_pallas_impl_bwd.md#_ring_attention_forward), [`_splash_attention`](_kernel.md#_splash_attention), [`_splash_attention_custom`](_kernel.md#_splash_attention_custom), [`_ring_attention_backward`](../ring_attention/_pallas_impl_bwd.md#_ring_attention_backward), [`_wrapped`](_kernel.md#make_attention_reference._wrapped), [`_splash_attention_fwd`](_kernel.md#_splash_attention_fwd), [`_splash_attention_forward`](_kernel.md#_splash_attention_forward), [`attention_reference`](_kernel.md#attention_reference), [`attention_reference_custom`](_kernel.md#attention_reference_custom), [`_attention_reference_default`](_kernel.md#_attention_reference_default), [`_attention_reference`](_kernel.md#_attention_reference), [`SplashResidualsType`](_kernel.md#SplashResidualsType), [`_attention_reference_custom_fwd`](_kernel.md#_attention_reference_custom_fwd), [`attn_static_fn`](_kernel.md#blocksparse_attention.attn_static_fn)

### `SplashAttentionKernel`
- def: [`ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py:2460`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2460)
- signature: `class SplashAttentionKernel:`
- members:
  - `__call__(self, *args, **kwargs)` — [`L2484`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2484) — Execute the Splash Attention kernel.
  - `__init__(self, fwd_mask_info: mask_info_lib.MaskInfo, dq_mask_info: mask_info_lib.MaskInfo | None, dkv_mask_info: mask_info_lib.MaskInfo | None, **kwargs)` — [`L2461`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2461) — Initialize the SplashAttentionKernel with mask metadata.
  - `manual_sharding_spec(self, sharding: jax.sharding.NamedSharding)` — [`L2507`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2507) — Returns a value that can be used as a shard_map partition spec for the kernel.
  - `tree_flatten(self)` — [`L2537`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2537)
  - `tree_unflatten(cls, kwargs, values)` — [`L2544`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2544)
  - `dkv_mask_info` — [`L2482`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2482)
  - `dq_mask_info` — [`L2481`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2481)
  - `fwd_mask_info` — [`L2480`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2480)
  - `kwargs` — [`L2479`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2479)
- uses (calls/refs, reference-scoped): [`MaskInfo`](_info.md#MaskInfo), [`_splash_attention`](_kernel.md#_splash_attention), [`partial_mask_blocks`](_info.md#MaskInfo.partial_mask_blocks), [`data_next`](_info.md#MaskInfo.data_next), [`q_sequence`](_info.md#MaskInfo.q_sequence), [`block_mask`](_info.md#MaskInfo.block_mask), [`mask_next`](_info.md#MaskInfo.mask_next), [`SplashCustomReturnType`](_kernel.md#SplashCustomReturnType), [`is_dynamic_mask`](_info.md#MaskInfo.is_dynamic_mask)
- used by: [`_make_splash_attention`](_kernel.md#_make_splash_attention)

## Functions
- `_apply_mask_and_soft_cap(qk: jax.Array, mask_value: float, should_not_mask, mask_ref, q_sequence_ref, q_segment_ids_ref, kv_segment_ids_ref, *, logits_soft_cap: float, k_slice: pl.Slice, k_offset: int | jax.Array, bq: int, k_in_lanes=True, mask_function=None)` — [`L782`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L782) — Apply attention masking and optional logits soft capping to QK scores.
- `_attention_reference(mask: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None, save_residuals: Literal[False], mask_value: float, custom_type: str, logits_soft_cap: float | None)` — [`L164`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L164)
- `_attention_reference_custom_bwd(mask_value: float, save_residuals: bool, custom_type: str, logits_soft_cap: float | None, res, do: jax.Array)` — [`L380`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L380) — Backward pass for custom VJP attention reference.
- `_attention_reference_custom_fwd(mask: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None, mask_value: float, save_residuals: bool, custom_type: str, logits_soft_cap: float | None)` — [`L326`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L326) — Forward pass for custom VJP attention reference.
- `_attention_reference_default(mask: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None, mask_value: float, save_residuals: bool, custom_type: str, logits_soft_cap: float | None)` — [`L219`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L219) — Compute reference attention output using standard softmax formulation.
- `_collapse_partial_mask_blocks(mask_info: mask_info_lib.MaskInfo | None)` — [`L2428`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2428)
- `_div(dividend: int, divisor: int)` — [`L1062`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1062)
- `_flash_attention_dkv_kernel(data_next_ref, block_mask_ref, mask_next_ref, q_ref, k_ref, v_ref, q_segment_ids_ref, kv_segment_ids_ref, sinks_ref, logsumexp_ref, do_ref, di_ref, mask_ref, q_sequence_ref, dq_scratch_ref, dk_scratch_ref, dv_scratch_ref, dq_ref, dk_ref, dv_ref, *, num_q_heads: int, num_kv_heads: int, mask_value: float, grid_width: int, bq: int, bkv_compute: int, is_mqa: bool, logits_soft_cap: float | None, q_layout: QKVLayout, k_layout: QKVLayout, v_layout: QKVLayout, bkv: int, mask_function: MaskFunctionType | None)` — [`L1750`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1750)
- `_flash_attention_dq_kernel(data_next_ref, block_mask_ref, mask_next_ref, q_ref, k_ref, v_ref, q_segment_ids_ref, kv_segment_ids_ref, sinks_ref, logsumexp_ref, do_ref, di_ref, mask_ref, q_sequence_ref, dq_scratch_ref, dq_ref, *, mask_value: float, grid_width: int, bq: int, bkv: int, logits_soft_cap: float | None = None, q_layout: QKVLayout, k_layout: QKVLayout, v_layout: QKVLayout, mask_function: MaskFunctionType | None)` — [`L1426`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1426)
- `_load_kv(ref, layout)` — [`L1824`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1824)
- `_make_splash_attention(mask: np.ndarray | jax.Array | mask_lib.MultiHeadMask, *, block_sizes: BlockSizes | None = None, is_mqa: bool, save_residuals: bool = False, mask_value: float = DEFAULT_MASK_VALUE, logits_soft_cap: float | None = None, downcast_smem_data: bool = True, head_shards: int, q_seq_shards: int, residual_checkpoint_name: str | None = None, interpret: bool = False)` — [`L2557`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2557) — Create a SplashAttentionKernel from a dense or lazy mask. — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
- `_next_nonzero(h, i, j, data_next_ref, block_mask_ref, m_next_ref, next_i=False)` — [`L719`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L719) — Look up the next nonzero block index from the sparse mask metadata. — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
- `_splash_attention(fwd_mask_info: mask_info_lib.MaskInfo, dq_mask_info: mask_info_lib.MaskInfo | None, dkv_mask_info: mask_info_lib.MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None = None, sinks: jax.Array | None = None, *, is_mqa: bool, block_sizes: BlockSizes | None, save_residuals: bool, mask_value: float, logits_soft_cap: float | None, residual_checkpoint_name: str | None, mask_function: MaskFunctionType | None, interpret: bool)` — [`L2401`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2401) — For dynamic masks, `partial_mask_blocks` has shape (head_count, q_blocks, kv_blocks, block_q, block_kv).
- `_splash_attention_bwd(save_residuals: bool, mask_value: float, is_mqa: bool, block_sizes: BlockSizes, residual_checkpoint_name: str | None, mask_function: MaskFunctionType | None, logits_soft_cap: float | None, interpret: bool, res: SplashResidualsType, do: jax.Array)` — [`L2277`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2277)
- `_splash_attention_bwd_dkv(q, k, v, segment_ids, sinks, lse, do, di, *, bq: int, bkv: int, bkv_compute: int, is_mqa: bool, mask_info: mask_info_lib.MaskInfo, mask_value: float, logits_soft_cap: float | None, use_fused_bwd_kernel: bool, q_layout: QKVLayout, k_layout: QKVLayout, v_layout: QKVLayout, mask_function: MaskFunctionType | None, interpret: bool)` — [`L1918`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1918) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
- `_splash_attention_bwd_dq(q, k, v, segment_ids, sinks, lse, do, di, *, bq: int, bkv: int, is_mqa: bool, mask_info: mask_info_lib.MaskInfo, mask_value: float, logits_soft_cap: float | None, q_layout: QKVLayout, k_layout: QKVLayout, v_layout: QKVLayout, mask_function: MaskFunctionType | None, interpret: bool)` — [`L1524`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1524) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
- `_splash_attention_custom(fwd_mask_info: mask_info_lib.MaskInfo, dq_mask_info: mask_info_lib.MaskInfo | None, dkv_mask_info: mask_info_lib.MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None, save_residuals: bool, mask_value: float, is_mqa: bool, block_sizes: BlockSizes, residual_checkpoint_name: str | None, mask_function: MaskFunctionType | None, logits_soft_cap: float | None = None, interpret: bool = False)` — [`L1335`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1335)
- `_splash_attention_forward(fwd_mask_info: mask_info_lib.MaskInfo, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, mask_value: float, is_mqa: bool, block_sizes: BlockSizes, residual_checkpoint_name: str | None, mask_function: MaskFunctionType | None, save_residuals: Literal[False] = False, logits_soft_cap: float | None = None)` — [`L1028`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1028)
- `_splash_attention_fwd(fwd_mask_info: mask_info_lib.MaskInfo, dq_mask_info: mask_info_lib.MaskInfo | None, dkv_mask_info: mask_info_lib.MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None, save_residuals: bool, mask_value: float, is_mqa: bool, block_sizes: BlockSizes, residual_checkpoint_name: str | None, mask_function: MaskFunctionType | None, logits_soft_cap: float | None = None, interpret: bool = False)` — [`L1373`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1373)
- `_wrapped(mask: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None = None, sinks: jax.Array | None = None, *, mask_value: float = DEFAULT_MASK_VALUE, save_residuals: bool = False, logits_soft_cap: float | None = None)` — [`L536`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L536)
- `attention_reference(mask: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None = None, *, mask_value: float = DEFAULT_MASK_VALUE, save_residuals: bool = False, custom_type: str = "flash", logits_soft_cap: float | None = None)` — [`L279`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L279) — Compute reference attention with default (non-custom-VJP) backward pass.
- `attention_reference_custom(mask: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None = None, *, mask_value: float = DEFAULT_MASK_VALUE, save_residuals: bool = False, custom_type: str = "flash", logits_soft_cap: float | None = None)` — [`L453`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L453) — Compute reference attention with custom VJP backward pass.
- `attn_static_fn(q, k, v, q_segment_ids, kv_segment_ids, softmax_aux)` — [`L2823`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2823)
- `blocksparse_attention(query: Float[Array, "batch num_heads seq_len head_dim"], key: Float[Array, "batch kv_num_heads kv_len head_dim"], value: Float[Array, "batch kv_num_heads kv_len vhead_dim"], q_segment_ids: Int[Array, "batch seq_len"] | None = None, kv_segment_ids: Int[Array, "batch kv_len"] | None = None, q_positions: Int[Array, "batch seq_len"] | None = None, kv_positions: Int[Array, "batch kv_len"] | None = None, softmax_aux: Float[Array, num_sinks] | None = None, bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None, attention_mask: (Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | Int[Array, "batch num_heads_or_1 seq_len kv_len"] | None) = None, sequence_parallelism_mesh_axis_name: str | None = None, logits_soft_cap: float | None = None, qkv_layouts: tuple[SparseMask] | None = None, softmax_scale: float | None = None, fwd_params: FwdParams | None = None, bwd_params: BwdParams | None = None, mask_builder: Callable[[int, int, int, int, int], Mask] | Callable[[], SparseMask] | None = None, sliding_window: int | tuple[int, int] | None = None, chunk_size: int | None = None, causal: bool = True, fused_backward: bool = False)` — [`L2685`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2685) — Pallas TPU block-sparse attention kernel implementation. — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
- `body(kv_compute_index, _)` — [`L944`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L944)
- `body(i, _)` — [`L1820`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1820)
- `cap_logits(logits)` — [`L871`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L871)
- `dkv_index_map(kv_index, head_index, *_)` — [`L2061`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2061)
- `dq_index_map(kv_index, head_index, q_index, *_)` — [`L2048`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2048)
- `end()` — [`L1014`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1014)
- `end()` — [`L1519`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1519)
- `end()` — [`L1908`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1908)
- `flash_attention_kernel(data_next_ref, block_mask_ref, mask_next_ref, q_ref, k_ref, v_ref, q_segment_ids_ref, kv_segment_ids_ref, sinks_ref, mask_ref, q_sequence_ref, m_scratch_ref, l_scratch_ref, o_scratch_ref, o_ref, logsumexp_ref=None, *, mask_value: float, grid_width: int, bq: int, bkv: int, bkv_compute: int, head_dim_v: int, q_layout: QKVLayout, k_layout: QKVLayout, v_layout: QKVLayout, logits_soft_cap: float | None, mask_function: MaskFunctionType | None)` — [`L887`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L887)
- `from_head_minor(vals: tuple[Any, ...], layout: QKVLayout)` — [`L638`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L638) — Convert index tuple from HEAD_DIM_MINOR layout to the target layout.
- `get_kernel_name(block_metadata: Mapping[str, Any], is_mqa: bool, save_residuals: bool, is_segmented: bool, phase: str)` — [`L140`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L140) — Returns a unique name for all SplashAttention kernel variants.
- `init()` — [`L924`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L924)
- `init()` — [`L1461`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1461)
- `init()` — [`L1806`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1806)
- `k_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L1149`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1149)
- `k_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_)` — [`L1591`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1591)
- `k_index_map(kv_index, head_index, *_)` — [`L2022`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2022) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
- `kv_segment_ids_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L1169`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1169)
- `kv_segment_ids_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_)` — [`L1623`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1623)
- `kv_segment_ids_index_map(kv_index, *_)` — [`L2117`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2117)
- `logsumexp_index_map(h, i, *_)` — [`L1235`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1235)
- `logsumexp_index_map(h, i, *_)` — [`L1644`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1644)
- `logsumexp_index_map(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L2137`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2137)
- `make_attention_reference(mask: mask_lib.Mask | np.ndarray, is_mqa: bool, backward_impl: str = "vanilla", **params: Any)` — [`L501`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L501) — Create a JIT-compiled reference attention function with the given mask.
- `mask_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L1161`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1161)
- `mask_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_)` — [`L1611`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1611)
- `mask_index_map(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref)` — [`L2075`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2075)
- `o_index_map(h, i, *_)` — [`L1581`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1581)
- `o_index_map(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L1980`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1980)
- `out_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L1143`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1143)
- `q_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L1139`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1139)
- `q_index_map(h, i, *_)` — [`L1586`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1586)
- `q_index_map(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L2001`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2001) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
- `q_segment_ids_index_map(h, i, j, *_)` — [`L1165`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1165)
- `q_segment_ids_index_map(h, i, j, *_)` — [`L1617`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1617)
- `q_segment_ids_index_map(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L2096`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2096)
- `reshape_activations(activations)` — [`L593`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L593)
- `reshape_residuals(residuals)` — [`L599`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L599)
- `run()` — [`L1008`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1008)
- `run()` — [`L1469`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1469)
- `run()` — [`L1893`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1893)
- `to_i32(x)` — [`L768`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L768)
- `v_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L1156`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1156)
- `v_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_)` — [`L1601`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1601)
- `v_index_map(kv_index, head_index, *_)` — [`L2034`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2034)

## Module values
- `DEFAULT_MASK_VALUE` — [`L89`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L89)
- `MaskFunctionType` — [`L137`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L137) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
- `NN_DIM_NUMBERS` — [`L93`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L93)
- `NT_DIM_NUMBERS` — [`L94`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L94)
- `NUM_LANES` — [`L90`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L90) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
- `NUM_SUBLANES` — [`L91`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L91) — documented in [ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md)
- `SplashCustomReturnType` — [`L123`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L123)
- `SplashResidualsType` — [`L125`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L125)
- `_attention_reference_custom` — [`L447`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L447)
- `make_masked_mha_reference` — [`L618`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L618)
- `make_masked_mqa_reference` — [`L619`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L619)
- `make_splash_mha` — [`L2665`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2665)
- `make_splash_mha_single_device` — [`L2668`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2668)
- `make_splash_mqa` — [`L2666`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2666)
- `make_splash_mqa_single_device` — [`L2669`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L2669)
- `partial` — [`L88`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L88)

