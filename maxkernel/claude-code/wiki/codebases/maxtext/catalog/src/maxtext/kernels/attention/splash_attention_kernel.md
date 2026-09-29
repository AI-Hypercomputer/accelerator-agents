---
title: 'Module: src/maxtext/kernels/attention/splash_attention_kernel.py'
type: catalog
provenance: extracted
module: src/maxtext/kernels/attention/splash_attention_kernel.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.kernels.attention.splash_attention_kernel`/
symbols:
  _splash_attention_bwd_dkv: _splash_attention_bwd_dkv().
  _splash_attention_bwd: _splash_attention_bwd().
  _splash_attention_bwd_dq: _splash_attention_bwd_dq().
  NUM_LANES: NUM_LANES.
  QKVLayout: QKVLayout#
  _make_splash_attention: _make_splash_attention().
  BlockSizes.get_default: BlockSizes#get_default().
  from_head_minor: from_head_minor().
  make_attention_reference._wrapped: make_attention_reference()._wrapped().
  _splash_attention: _splash_attention().
  _splash_attention_manual_fwd: _splash_attention_manual_fwd().
  SegmentIds: SegmentIds#
  BlockSizes.__post_init__: BlockSizes#__post_init__().
  BlockSizes.has_backward_blocks: BlockSizes#has_backward_blocks().
  _splash_attention_custom: _splash_attention_custom().
  QKVLayout.HEAD_DIM_MINOR: QKVLayout#HEAD_DIM_MINOR.
  _next_nonzero: _next_nonzero().
  SplashAttentionKernel.__call__: SplashAttentionKernel#__call__().
  SplashAttentionKernel.manual_fwd: SplashAttentionKernel#manual_fwd().
  partial: partial.
  MaskFunctionType: MaskFunctionType.
  flash_attention_kernel: flash_attention_kernel().
  _splash_attention_fwd: _splash_attention_fwd().
  _splash_attention_forward: _splash_attention_forward().
  SplashAttentionKernel.manual_bwd: SplashAttentionKernel#manual_bwd().
  SplashAttentionKernel.manual_sharding_spec: SplashAttentionKernel#manual_sharding_spec().
  NUM_SUBLANES: NUM_SUBLANES.
  _attention_reference_custom: _attention_reference_custom.
  BlockSizes: BlockSizes#
  SplashAttentionKernel.fwd_mask_info: SplashAttentionKernel#fwd_mask_info.
  attention_reference: attention_reference().
  flash_attention_kernel.body: flash_attention_kernel().body().
  _flash_attention_dq_kernel: _flash_attention_dq_kernel().
  _flash_attention_dkv_kernel: _flash_attention_dkv_kernel().
  _flash_attention_dkv_kernel.body: _flash_attention_dkv_kernel().body().
  _splash_attention_manual_bwd: _splash_attention_manual_bwd().
  _apply_mask_and_soft_cap: _apply_mask_and_soft_cap().
  SplashAttentionKernel.tree_flatten: SplashAttentionKernel#tree_flatten().
  attention_reference_custom: attention_reference_custom().
  _attention_reference_default: _attention_reference_default().
  _splash_attention_forward.k_index_map: _splash_attention_forward().k_index_map().
  _splash_attention_forward.v_index_map: _splash_attention_forward().v_index_map().
  _splash_attention_bwd_dq.k_index_map: _splash_attention_bwd_dq().k_index_map().
  _splash_attention_bwd_dq.v_index_map: _splash_attention_bwd_dq().v_index_map().
  NN_DIM_NUMBERS: NN_DIM_NUMBERS.
  NT_DIM_NUMBERS: NT_DIM_NUMBERS.
  SplashCustomReturnType: SplashCustomReturnType.
  BlockSizes.q_layout: BlockSizes#q_layout.
  BlockSizes.k_layout: BlockSizes#k_layout.
  BlockSizes.v_layout: BlockSizes#v_layout.
  _attention_reference: _attention_reference().
  _flash_attention_dq_kernel.run: _flash_attention_dq_kernel().run().
  _div: _div().
  make_splash_mha: make_splash_mha.
  _attention_reference_custom_fwd: _attention_reference_custom_fwd().
  _splash_attention_bwd_dkv.q_index_map: _splash_attention_bwd_dkv().q_index_map().
  _splash_attention_bwd_dkv.k_index_map: _splash_attention_bwd_dkv().k_index_map().
  _splash_attention_bwd_dkv.v_index_map: _splash_attention_bwd_dkv().v_index_map().
  make_masked_mha_reference: make_masked_mha_reference.
  make_masked_mqa_reference: make_masked_mqa_reference.
  BlockSizes.block_kv_dkv: BlockSizes#block_kv_dkv.
  BlockSizes.block_q_dq: BlockSizes#block_q_dq.
  BlockSizes.block_kv_dq: BlockSizes#block_kv_dq.
  BlockSizes.use_fused_bwd_kernel: BlockSizes#use_fused_bwd_kernel.
  SplashAttentionKernel.kwargs: SplashAttentionKernel#kwargs.
  SplashAttentionKernel.dq_mask_info: SplashAttentionKernel#dq_mask_info.
  SplashAttentionKernel.dkv_mask_info: SplashAttentionKernel#dkv_mask_info.
  make_splash_mqa: make_splash_mqa.
  make_splash_mha_single_device: make_splash_mha_single_device.
  make_splash_mqa_single_device: make_splash_mqa_single_device.
  SplashResidualsType: SplashResidualsType.
  make_attention_reference: make_attention_reference().
  _splash_attention_bwd_dkv.dkv_index_map: _splash_attention_bwd_dkv().dkv_index_map().
  _splash_attention_bwd_dkv.q_segment_ids_index_map: _splash_attention_bwd_dkv().q_segment_ids_index_map().
  _splash_attention_bwd_dkv.logsumexp_index_map: _splash_attention_bwd_dkv().logsumexp_index_map().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  SegmentIds.q: SegmentIds#q.
  SegmentIds.kv: SegmentIds#kv.
  BlockSizes.block_kv: BlockSizes#block_kv.
  BlockSizes.block_q_dkv: BlockSizes#block_q_dkv.
  BlockSizes.block_kv_dkv_compute: BlockSizes#block_kv_dkv_compute.
  _splash_attention_forward.q_index_map: _splash_attention_forward().q_index_map().
  _splash_attention_forward.mask_index_map: _splash_attention_forward().mask_index_map().
  _splash_attention_forward.kv_segment_ids_index_map: _splash_attention_forward().kv_segment_ids_index_map().
  _splash_attention_bwd_dq.q_index_map: _splash_attention_bwd_dq().q_index_map().
  _splash_attention_bwd_dq.mask_index_map: _splash_attention_bwd_dq().mask_index_map().
  _splash_attention_bwd_dq.kv_segment_ids_index_map: _splash_attention_bwd_dq().kv_segment_ids_index_map().
  _splash_attention_bwd_dkv.o_index_map: _splash_attention_bwd_dkv().o_index_map().
  _splash_attention_bwd_dkv.mask_index_map: _splash_attention_bwd_dkv().mask_index_map().
  get_kernel_name: get_kernel_name().
  BlockSizes.block_q: BlockSizes#block_q.
  BlockSizes.block_kv_compute: BlockSizes#block_kv_compute.
  _splash_attention._collapse_partial_mask_blocks: _splash_attention()._collapse_partial_mask_blocks().
  _splash_attention_manual_fwd._collapse_partial_mask_blocks: _splash_attention_manual_fwd()._collapse_partial_mask_blocks().
  SplashAttentionKernel: SplashAttentionKernel#
  flash_attention_kernel.run: flash_attention_kernel().run().
  flash_attention_kernel.end: flash_attention_kernel().end().
  _flash_attention_dkv_kernel.run: _flash_attention_dkv_kernel().run().
  SplashAttentionKernel.tree_unflatten: SplashAttentionKernel#tree_unflatten().
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
  SplashAttentionKernel.__init__: SplashAttentionKernel#__init__().
---
# Module: [`src/maxtext/kernels/attention/splash_attention_kernel.py`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py)

## Classes
### `BlockSizes`
- def: [`src/maxtext/kernels/attention/splash_attention_kernel.py:476`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L476) — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
- doc: Tile sizes parameterizing SplashAttention kernels.
- signature: `class BlockSizes:`
- members:
  - `get_default(cls)` — [`L525`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L525)
  - `has_backward_blocks(self)` — [`L514`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L514) — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
  - `block_kv` — [`L488`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L488)
  - `block_kv_compute` — [`L489`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L489)
  - `block_kv_dkv` — [`L492`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L492) — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
  - `block_kv_dkv_compute` — [`L493`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L493) — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
  - `block_kv_dq` — [`L496`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L496) — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
  - `block_q` — [`L487`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L487)
  - `block_q_dkv` — [`L491`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L491) — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
  - `block_q_dq` — [`L495`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L495) — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
  - `k_layout` — [`L501`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L501) — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
  - `q_layout` — [`L500`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L500) — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
  - `use_fused_bwd_kernel` — [`L498`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L498) — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
  - `v_layout` — [`L502`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L502) — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
- protocol/private: `__post_init__`[`L504`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L504)
- uses (calls/refs, reference-scoped): [`QKVLayout`](splash_attention_kernel.md#QKVLayout), [`HEAD_DIM_MINOR`](splash_attention_kernel.md#QKVLayout.HEAD_DIM_MINOR)
- used by: [`_splash_attention_bwd`](splash_attention_kernel.md#_splash_attention_bwd), [`_make_splash_attention`](splash_attention_kernel.md#_make_splash_attention), [`_splash_attention`](splash_attention_kernel.md#_splash_attention), [`_splash_attention_manual_fwd`](splash_attention_kernel.md#_splash_attention_manual_fwd), [`_splash_attention_custom`](splash_attention_kernel.md#_splash_attention_custom), [`_splash_attention_fwd`](splash_attention_kernel.md#_splash_attention_fwd), [`_splash_attention_forward`](splash_attention_kernel.md#_splash_attention_forward), [`_splash_attention_manual_bwd`](splash_attention_kernel.md#_splash_attention_manual_bwd)

### `QKVLayout`  ·  implements/extends IntEnum
- def: [`src/maxtext/kernels/attention/splash_attention_kernel.py:464`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L464) — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
- signature: `class QKVLayout(enum.IntEnum):`
- members:
  - `HEAD_DIM_MINOR` — [`L465`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L465) — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
  - `SEQ_MINOR` — [`L466`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L466)
- used by: [`_splash_attention_bwd_dkv`](splash_attention_kernel.md#_splash_attention_bwd_dkv), [`_splash_attention_bwd_dq`](splash_attention_kernel.md#_splash_attention_bwd_dq), [`from_head_minor`](splash_attention_kernel.md#from_head_minor), [`flash_attention_kernel`](splash_attention_kernel.md#flash_attention_kernel), [`_flash_attention_dkv_kernel`](splash_attention_kernel.md#_flash_attention_dkv_kernel), [`_flash_attention_dq_kernel`](splash_attention_kernel.md#_flash_attention_dq_kernel), [`k_layout`](splash_attention_kernel.md#BlockSizes.k_layout), [`q_layout`](splash_attention_kernel.md#BlockSizes.q_layout), [`v_layout`](splash_attention_kernel.md#BlockSizes.v_layout)

### `SegmentIds`  ·  implements/extends NamedTuple
- def: [`src/maxtext/kernels/attention/splash_attention_kernel.py:50`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L50) — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
- doc: SegmentIds for Q and KV sequences.
- signature: `class SegmentIds(NamedTuple):`
- members:
  - `kv` — [`L68`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L68)
  - `q` — [`L67`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L67)
- used by: [`_splash_attention_bwd`](splash_attention_kernel.md#_splash_attention_bwd), [`_splash_attention`](splash_attention_kernel.md#_splash_attention), [`_splash_attention_manual_fwd`](splash_attention_kernel.md#_splash_attention_manual_fwd), [`_wrapped`](splash_attention_kernel.md#make_attention_reference._wrapped), [`_splash_attention_custom`](splash_attention_kernel.md#_splash_attention_custom), [`_splash_attention_fwd`](splash_attention_kernel.md#_splash_attention_fwd), [`_splash_attention_forward`](splash_attention_kernel.md#_splash_attention_forward), [`_splash_attention_manual_bwd`](splash_attention_kernel.md#_splash_attention_manual_bwd), [`attention_reference`](splash_attention_kernel.md#attention_reference), [`attention_reference_custom`](splash_attention_kernel.md#attention_reference_custom), [`_attention_reference_default`](splash_attention_kernel.md#_attention_reference_default), [`_attention_reference`](splash_attention_kernel.md#_attention_reference), [`_attention_reference_custom_fwd`](splash_attention_kernel.md#_attention_reference_custom_fwd), [`SplashResidualsType`](splash_attention_kernel.md#SplashResidualsType)

### `SplashAttentionKernel`
- def: [`src/maxtext/kernels/attention/splash_attention_kernel.py:2363`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L2363)
- doc: Defines a SplashAttention kernel object.
- signature: `class SplashAttentionKernel:`
- members:
  - `manual_bwd(self, *args, **kwargs)` — [`L2398`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L2398)
  - `manual_fwd(self, *args, **kwargs)` — [`L2388`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L2388)
  - `manual_sharding_spec(self, sharding: jax.sharding.NamedSharding)` — [`L2408`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L2408) — Returns a value that can be used as a shard_map partition spec for the kernel.
  - `tree_flatten(self)` — [`L2438`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L2438)
  - `tree_unflatten(cls, kwargs, values)` — [`L2445`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L2445)
  - `dkv_mask_info` — [`L2376`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L2376)
  - `dq_mask_info` — [`L2375`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L2375)
  - `fwd_mask_info` — [`L2374`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L2374)
  - `kwargs` — [`L2373`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L2373)
- protocol/private: `__call__`[`L2378`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L2378), `__init__`[`L2366`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L2366)
- uses (calls/refs, reference-scoped): [`_splash_attention`](splash_attention_kernel.md#_splash_attention), [`_splash_attention_manual_fwd`](splash_attention_kernel.md#_splash_attention_manual_fwd), [`_splash_attention_manual_bwd`](splash_attention_kernel.md#_splash_attention_manual_bwd), [`SplashCustomReturnType`](splash_attention_kernel.md#SplashCustomReturnType)
- used by: [`_make_splash_attention`](splash_attention_kernel.md#_make_splash_attention)

## Functions
- `_apply_mask_and_soft_cap(qk: jax.Array, mask_value: float, should_not_mask, mask_ref, q_sequence_ref, q_segment_ids_ref, kv_segment_ids_ref, *, attn_logits_soft_cap: float, k_slice: pl.Slice, k_offset: int | jax.Array, bq: int, k_in_lanes=True, mask_function=None)` — [`L586`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L586) — Applies the mask and soft cap to the logits.
- `_attention_reference(mask: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, save_residuals: Literal[False], mask_value: float, custom_type: str, attn_logits_soft_cap: float | None)` — [`L117`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L117) — Reference attention implementation.
- `_attention_reference_custom_bwd(mask_value: float, save_residuals: bool, custom_type: str, attn_logits_soft_cap: float | None, res, do: jax.Array)` — [`L263`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L263) — Reference attention custom backward implementation.
- `_attention_reference_custom_fwd(mask: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, mask_value: float, save_residuals: bool, custom_type: str, attn_logits_soft_cap: float | None)` — [`L234`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L234) — Reference attention custom forward implementation.
- `_attention_reference_default(mask: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, mask_value: float, save_residuals: bool, custom_type: str, attn_logits_soft_cap: float | None)` — [`L172`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L172) — Reference attention default implementation.
- `_collapse_partial_mask_blocks(mask_info: mask_info_lib.MaskInfo | None)` — [`L2212`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L2212)
- `_collapse_partial_mask_blocks(mask_info: mask_info_lib.MaskInfo | None)` — [`L2279`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L2279)
- `_div(dividend: int, divisor: int)` — [`L853`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L853) — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
- `_flash_attention_dkv_kernel(data_next_ref, block_mask_ref, mask_next_ref, q_ref, k_ref, v_ref, q_segment_ids_ref, kv_segment_ids_ref, logsumexp_ref, do_ref, di_ref, mask_ref, q_sequence_ref, dq_scratch_ref, dk_scratch_ref, dv_scratch_ref, dq_ref, dk_ref, dv_ref, *, num_q_heads: int, num_kv_heads: int, mask_value: float, grid_width: int, bq: int, bkv_compute: int, is_mqa: bool, attn_logits_soft_cap: float | None, q_layout: QKVLayout, k_layout: QKVLayout, v_layout: QKVLayout, bkv: int, mask_function: MaskFunctionType | None)` — [`L1528`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1528) — Backward pass for the DKV part of splash attention. — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
- `_flash_attention_dq_kernel(data_next_ref, block_mask_ref, mask_next_ref, q_ref, k_ref, v_ref, q_segment_ids_ref, kv_segment_ids_ref, logsumexp_ref, do_ref, di_ref, mask_ref, q_sequence_ref, dq_scratch_ref, dq_ref, *, mask_value: float, grid_width: int, bq: int, bkv: int, attn_logits_soft_cap: float | None = None, q_layout: QKVLayout, k_layout: QKVLayout, v_layout: QKVLayout, mask_function: MaskFunctionType | None)` — [`L1206`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1206) — Backprop kernel for the DQ part of flash attention. — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
- `_load_kv(ref, layout)` — [`L1614`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1614)
- `_make_splash_attention(mask: np.ndarray | jax.Array | mask_lib.MultiHeadMask, *, block_sizes: BlockSizes | None = None, is_mqa: bool, save_residuals: bool = False, mask_value: float = DEFAULT_MASK_VALUE, attn_logits_soft_cap: float | None = None, downcast_smem_data: bool = True, head_shards: int, q_seq_shards: int, residual_checkpoint_name: str | None = None, interpret: bool = False)` — [`L2458`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L2458) — Creates a SplashAttentionKernel.
- `_next_nonzero(h, i, j, data_next_ref, block_mask_ref, m_next_ref, next_i=False)` — [`L539`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L539) — Returns the next nonzero index and the mask for the current index. — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
- `_splash_attention(fwd_mask_info: mask_info_lib.MaskInfo, dq_mask_info: mask_info_lib.MaskInfo | None, dkv_mask_info: mask_info_lib.MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None = None, *, is_mqa: bool, block_sizes: BlockSizes | None, save_residuals: bool, mask_value: float, attn_logits_soft_cap: float | None, residual_checkpoint_name: str | None, mask_function: MaskFunctionType | None, interpret: bool)` — [`L2185`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L2185) — For dynamic masks, `partial_mask_blocks` has shape (head_count, q_blocks, kv_blocks, block_q, block_kv).
- `_splash_attention_bwd(save_residuals: bool, mask_value: float, is_mqa: bool, block_sizes: BlockSizes, residual_checkpoint_name: str | None, mask_function: MaskFunctionType | None, attn_logits_soft_cap: float | None, interpret: bool, res: SplashResidualsType, do: jax.Array)` — [`L2068`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L2068) — Backward pass for splash attention. — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
- `_splash_attention_bwd_dkv(q, k, v, segment_ids, logsumexp, do, di, *, bq: int, bkv: int, bkv_compute: int, is_mqa: bool, mask_info: mask_info_lib.MaskInfo, mask_value: float, attn_logits_soft_cap: float | None, use_fused_bwd_kernel: bool, q_layout: QKVLayout, k_layout: QKVLayout, v_layout: QKVLayout, mask_function: MaskFunctionType | None, interpret: bool)` — [`L1710`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1710) — Backward pass for the DKV part of splash attention. — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
- `_splash_attention_bwd_dq(q, k, v, segment_ids, logsumexp, do, di, *, bq: int, bkv: int, is_mqa: bool, mask_info: mask_info_lib.MaskInfo, mask_value: float, attn_logits_soft_cap: float | None, q_layout: QKVLayout, k_layout: QKVLayout, v_layout: QKVLayout, mask_function: MaskFunctionType | None, interpret: bool)` — [`L1308`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1308) — Backward pass for the DQ part of splash attention. — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
- `_splash_attention_custom(fwd_mask_info: mask_info_lib.MaskInfo, dq_mask_info: mask_info_lib.MaskInfo | None, dkv_mask_info: mask_info_lib.MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, save_residuals: bool, mask_value: float, is_mqa: bool, block_sizes: BlockSizes, residual_checkpoint_name: str | None, mask_function: MaskFunctionType | None, attn_logits_soft_cap: float | None = None, interpret: bool = False)` — [`L1109`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1109) — Custom splash attention kernel.
- `_splash_attention_forward(fwd_mask_info: mask_info_lib.MaskInfo, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, mask_value: float, is_mqa: bool, block_sizes: BlockSizes, residual_checkpoint_name: str | None, mask_function: MaskFunctionType | None, save_residuals: Literal[False] = False, attn_logits_soft_cap: float | None = None)` — [`L818`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L818)
- `_splash_attention_fwd(fwd_mask_info: mask_info_lib.MaskInfo, dq_mask_info: mask_info_lib.MaskInfo | None, dkv_mask_info: mask_info_lib.MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, save_residuals: bool, mask_value: float, is_mqa: bool, block_sizes: BlockSizes, residual_checkpoint_name: str | None, mask_function: MaskFunctionType | None, attn_logits_soft_cap: float | None = None, interpret: bool = False)` — [`L1155`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1155) — Forward pass for splash attention.
- `_splash_attention_manual_bwd(fwd_mask_info: mask_info_lib.MaskInfo, dq_mask_info: mask_info_lib.MaskInfo | None, dkv_mask_info: mask_info_lib.MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, out: jax.Array, logsumexp: jax.Array, do: jax.Array, segment_ids: SegmentIds | None = None, sinks: jax.Array | None = None, *, is_mqa: bool, block_sizes: BlockSizes | None, save_residuals: bool, mask_value: float, attn_logits_soft_cap: float | None, residual_checkpoint_name: str | None, mask_function: MaskFunctionType | None, interpret: bool)` — [`L2313`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L2313) — Transpose of _splash_attention_manual_fwd that uses attention output and logsumexp. — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
- `_splash_attention_manual_fwd(fwd_mask_info: mask_info_lib.MaskInfo, dq_mask_info: mask_info_lib.MaskInfo | None, dkv_mask_info: mask_info_lib.MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None = None, sinks: jax.Array | None = None, *, is_mqa: bool, block_sizes: BlockSizes | None, save_residuals: bool, mask_value: float, attn_logits_soft_cap: float | None, residual_checkpoint_name: str | None, mask_function: MaskFunctionType | None, interpret: bool)` — [`L2255`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L2255) — Returns both the attention output and logsumexp.
- `_wrapped(mask: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None = None, *, mask_value: float = DEFAULT_MASK_VALUE, save_residuals: bool = False, attn_logits_soft_cap: float | None = None)` — [`L357`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L357)
- `attention_reference(mask: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, *, mask_value: float = DEFAULT_MASK_VALUE, save_residuals: bool = False, custom_type: str = "flash", attn_logits_soft_cap: float | None = None)` — [`L208`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L208) — Reference attention implementation.
- `attention_reference_custom(mask: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, *, mask_value: float = DEFAULT_MASK_VALUE, save_residuals: bool = False, custom_type: str = "flash", attn_logits_soft_cap: float | None = None)` — [`L315`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L315) — Reference attention custom implementation.
- `body(kv_compute_index, _)` — [`L730`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L730)
- `body(i, _)` — [`L1610`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1610)
- `cap_logits(logits)` — [`L657`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L657)
- `dkv_index_map(kv_index, head_index, *_)` — [`L1853`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1853)
- `dq_index_map(kv_index, head_index, q_index, *_)` — [`L1840`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1840)
- `end()` — [`L804`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L804)
- `end()` — [`L1303`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1303)
- `end()` — [`L1700`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1700)
- `flash_attention_kernel(data_next_ref, block_mask_ref, mask_next_ref, q_ref, k_ref, v_ref, q_segment_ids_ref, kv_segment_ids_ref, mask_ref, q_sequence_ref, m_scratch_ref, l_scratch_ref, o_scratch_ref, o_ref, logsumexp_ref=None, *, mask_value: float, grid_width: int, bq: int, bkv: int, bkv_compute: int, head_dim_v: int, q_layout: QKVLayout, k_layout: QKVLayout, v_layout: QKVLayout, attn_logits_soft_cap: float | None, mask_function: MaskFunctionType | None)` — [`L673`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L673) — Flash attention kernel. — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
- `from_head_minor(vals: tuple[Any, ...], layout: QKVLayout)` — [`L469`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L469) — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
- `get_kernel_name(block_metadata: Mapping[str, Any], is_mqa: bool, save_residuals: bool, is_segmented: bool, phase: str)` — [`L90`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L90) — Returns a unique name for all SplashAttention kernel variants.
- `init()` — [`L716`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L716)
- `init()` — [`L1243`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1243)
- `init()` — [`L1596`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1596)
- `k_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L938`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L938)
- `k_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_)` — [`L1378`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1378)
- `k_index_map(kv_index, head_index, *_)` — [`L1814`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1814)
- `kv_segment_ids_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L958`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L958)
- `kv_segment_ids_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_)` — [`L1410`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1410)
- `kv_segment_ids_index_map(kv_index, *_)` — [`L1909`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1909)
- `logsumexp_index_map(h, i, *_)` — [`L1022`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1022)
- `logsumexp_index_map(h, i, *_)` — [`L1424`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1424)
- `logsumexp_index_map(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L1922`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1922)
- `make_attention_reference(mask: mask_lib.Mask | np.ndarray, is_mqa: bool, backward_impl: str = "vanilla", **params: Any)` — [`L341`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L341) — Returns a function that computes reference attention.
- `mask_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L950`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L950)
- `mask_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_)` — [`L1398`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1398)
- `mask_index_map(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref)` — [`L1867`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1867)
- `o_index_map(h, i, *_)` — [`L1368`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1368)
- `o_index_map(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L1772`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1772)
- `out_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L932`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L932)
- `q_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L928`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L928)
- `q_index_map(h, i, *_)` — [`L1373`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1373)
- `q_index_map(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L1793`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1793)
- `q_segment_ids_index_map(h, i, j, *_)` — [`L954`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L954)
- `q_segment_ids_index_map(h, i, j, *_)` — [`L1404`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1404)
- `q_segment_ids_index_map(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L1888`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1888)
- `reshape_activations(activations)` — [`L427`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L427)
- `reshape_residuals(residuals)` — [`L437`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L437)
- `run()` — [`L798`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L798)
- `run()` — [`L1249`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1249)
- `run()` — [`L1685`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1685)
- `v_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L945`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L945)
- `v_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_)` — [`L1388`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1388)
- `v_index_map(kv_index, head_index, *_)` — [`L1826`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1826)

## Module values
- `DEFAULT_MASK_VALUE` — [`L40`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L40)
- `MaskFunctionType` — [`L87`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L87)
- `NN_DIM_NUMBERS` — [`L44`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L44)
- `NT_DIM_NUMBERS` — [`L45`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L45)
- `NUM_LANES` — [`L41`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L41) — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
- `NUM_SUBLANES` — [`L42`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L42) — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
- `SplashCustomReturnType` — [`L74`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L74)
- `SplashResidualsType` — [`L76`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L76) — documented in [maxtext-kernels-attention-splash_attention_kernel](../../../../../concepts/maxtext-kernels-attention-splash_attention_kernel.md)
- `_attention_reference_custom` — [`L311`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L311)
- `make_masked_mha_reference` — [`L456`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L456)
- `make_masked_mqa_reference` — [`L457`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L457)
- `make_splash_mha` — [`L2541`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L2541)
- `make_splash_mha_single_device` — [`L2544`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L2544)
- `make_splash_mqa` — [`L2542`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L2542)
- `make_splash_mqa_single_device` — [`L2546`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L2546)
- `partial` — [`L39`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L39)

