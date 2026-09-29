---
title: 'Module: jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel`/
symbols:
  _splash_attention_bwd_dkv: _splash_attention_bwd_dkv().
  _splash_attention_bwd_dq: _splash_attention_bwd_dq().
  _splash_attention_bwd: _splash_attention_bwd().
  _make_splash_attention: _make_splash_attention().
  _apply_mask_and_soft_cap: _apply_mask_and_soft_cap().
  _attention_reference_default: _attention_reference_default().
  SplashAttentionKernel.manual_sharding_spec: SplashAttentionKernel#manual_sharding_spec().
  QKVLayout: QKVLayout#
  flash_attention_kernel.body: flash_attention_kernel().body().
  NUM_LANES: NUM_LANES.
  make_attention_reference._wrapped: make_attention_reference()._wrapped().
  _flash_attention_dkv_kernel.body: _flash_attention_dkv_kernel().body().
  _flash_attention_dq_kernel.run: _flash_attention_dq_kernel().run().
  _splash_attention: _splash_attention().
  _splash_attention_custom: _splash_attention_custom().
  _attention_reference_custom_bwd: _attention_reference_custom_bwd().
  BlockSizes.get_default: BlockSizes#get_default().
  from_head_minor: from_head_minor().
  flash_attention_kernel: flash_attention_kernel().
  _flash_attention_dkv_kernel: _flash_attention_dkv_kernel().
  SplashAttentionKernel.__call__: SplashAttentionKernel#__call__().
  _next_nonzero: _next_nonzero().
  _splash_attention_fwd: _splash_attention_fwd().
  NUM_SUBLANES: NUM_SUBLANES.
  BlockSizes.__post_init__: BlockSizes#__post_init__().
  SegmentIds: SegmentIds#
  _attention_reference_custom: _attention_reference_custom.
  BlockSizes.has_backward_blocks: BlockSizes#has_backward_blocks().
  _splash_attention_forward: _splash_attention_forward().
  QKVLayout.HEAD_DIM_MINOR: QKVLayout#HEAD_DIM_MINOR.
  _flash_attention_dq_kernel: _flash_attention_dq_kernel().
  MaskFunctionType: MaskFunctionType.
  attention_reference: attention_reference().
  partial: partial.
  flash_attention_kernel.end: flash_attention_kernel().end().
  attention_reference_custom: attention_reference_custom().
  BlockSizes: BlockSizes#
  SplashAttentionKernel.fwd_mask_info: SplashAttentionKernel#fwd_mask_info.
  _attention_reference: _attention_reference().
  flash_attention_kernel.init: flash_attention_kernel().init().
  SplashAttentionKernel.tree_flatten: SplashAttentionKernel#tree_flatten().
  SplashResidualsType: SplashResidualsType.
  make_attention_reference: make_attention_reference().
  _div: _div().
  SplashCustomReturnType: SplashCustomReturnType.
  _attention_reference_custom_fwd: _attention_reference_custom_fwd().
  _splash_attention_forward.k_index_map: _splash_attention_forward().k_index_map().
  _splash_attention_forward.v_index_map: _splash_attention_forward().v_index_map().
  _splash_attention_bwd_dq.k_index_map: _splash_attention_bwd_dq().k_index_map().
  _splash_attention_bwd_dq.v_index_map: _splash_attention_bwd_dq().v_index_map().
  NN_DIM_NUMBERS: NN_DIM_NUMBERS.
  NT_DIM_NUMBERS: NT_DIM_NUMBERS.
  BlockSizes.q_layout: BlockSizes#q_layout.
  BlockSizes.k_layout: BlockSizes#k_layout.
  BlockSizes.v_layout: BlockSizes#v_layout.
  _splash_attention._collapse_partial_mask_blocks: _splash_attention()._collapse_partial_mask_blocks().
  make_splash_mha: make_splash_mha.
  flash_attention_kernel.run: flash_attention_kernel().run().
  _flash_attention_dkv_kernel.run: _flash_attention_dkv_kernel().run().
  SegmentIds.q: SegmentIds#q.
  SegmentIds.kv: SegmentIds#kv.
  make_masked_mha_reference: make_masked_mha_reference.
  make_masked_mqa_reference: make_masked_mqa_reference.
  _splash_attention_bwd_dkv.q_index_map: _splash_attention_bwd_dkv().q_index_map().
  _splash_attention_bwd_dkv.k_index_map: _splash_attention_bwd_dkv().k_index_map().
  _splash_attention_bwd_dkv.v_index_map: _splash_attention_bwd_dkv().v_index_map().
  make_splash_mqa: make_splash_mqa.
  make_splash_mha_single_device: make_splash_mha_single_device.
  make_splash_mqa_single_device: make_splash_mqa_single_device.
  SplashAttentionKernel: SplashAttentionKernel#
  BlockSizes.block_kv_dkv: BlockSizes#block_kv_dkv.
  BlockSizes.block_q_dq: BlockSizes#block_q_dq.
  BlockSizes.block_kv_dq: BlockSizes#block_kv_dq.
  BlockSizes.use_fused_bwd_kernel: BlockSizes#use_fused_bwd_kernel.
  _flash_attention_dq_kernel.init: _flash_attention_dq_kernel().init().
  _flash_attention_dq_kernel.end: _flash_attention_dq_kernel().end().
  _flash_attention_dkv_kernel.init: _flash_attention_dkv_kernel().init().
  _flash_attention_dkv_kernel.end: _flash_attention_dkv_kernel().end().
  SplashAttentionKernel.tree_unflatten: SplashAttentionKernel#tree_unflatten().
  _apply_mask_and_soft_cap.cap_logits: _apply_mask_and_soft_cap().cap_logits().
  _splash_attention_bwd_dkv.dkv_index_map: _splash_attention_bwd_dkv().dkv_index_map().
  _splash_attention_bwd_dkv.q_segment_ids_index_map: _splash_attention_bwd_dkv().q_segment_ids_index_map().
  _splash_attention_bwd_dkv.logsumexp_index_map: _splash_attention_bwd_dkv().logsumexp_index_map().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
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
  SplashAttentionKernel.kwargs: SplashAttentionKernel#kwargs.
  SplashAttentionKernel.dq_mask_info: SplashAttentionKernel#dq_mask_info.
  SplashAttentionKernel.dkv_mask_info: SplashAttentionKernel#dkv_mask_info.
  SplashAttentionKernel.__init__: SplashAttentionKernel#__init__().
  make_attention_reference._wrapped.reshape_activations: make_attention_reference()._wrapped().reshape_activations().
  _splash_attention_forward.q_segment_ids_index_map: _splash_attention_forward().q_segment_ids_index_map().
  _splash_attention_bwd_dq.q_segment_ids_index_map: _splash_attention_bwd_dq().q_segment_ids_index_map().
  _splash_attention_bwd_dq.logsumexp_index_map: _splash_attention_bwd_dq().logsumexp_index_map().
  _flash_attention_dkv_kernel.body._load_kv: _flash_attention_dkv_kernel().body()._load_kv().
  make_attention_reference._wrapped.reshape_residuals: make_attention_reference()._wrapped().reshape_residuals().
  _splash_attention_forward.out_index_map: _splash_attention_forward().out_index_map().
  _splash_attention_forward.logsumexp_index_map: _splash_attention_forward().logsumexp_index_map().
  _splash_attention_bwd_dq.o_index_map: _splash_attention_bwd_dq().o_index_map().
  _splash_attention_bwd_dkv.dq_index_map: _splash_attention_bwd_dkv().dq_index_map().
  _splash_attention_bwd_dkv.kv_segment_ids_index_map: _splash_attention_bwd_dkv().kv_segment_ids_index_map().
  QKVLayout.SEQ_MINOR: QKVLayout#SEQ_MINOR.
---
# Module: [`jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py)

## Classes
### `BlockSizes`
- def: [`jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py:488`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L488)
- doc: Tile sizes parameterizing SplashAttention kernels.
- signature: `class BlockSizes:`
- members:
  - `get_default(cls)` — [`L536`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L536)
  - `has_backward_blocks(self)` — [`L527`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L527)
  - `block_kv` — [`L499`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L499)
  - `block_kv_compute` — [`L500`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L500)
  - `block_kv_dkv` — [`L503`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L503)
  - `block_kv_dkv_compute` — [`L504`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L504)
  - `block_kv_dq` — [`L507`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L507)
  - `block_q` — [`L498`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L498)
  - `block_q_dkv` — [`L502`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L502)
  - `block_q_dq` — [`L506`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L506)
  - `k_layout` — [`L512`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L512)
  - `q_layout` — [`L511`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L511)
  - `use_fused_bwd_kernel` — [`L509`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L509)
  - `v_layout` — [`L513`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L513)
- protocol/private: `__post_init__`[`L515`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L515)
- uses (calls/refs, reference-scoped): [`QKVLayout`](splash_attention_kernel.md#QKVLayout), [`HEAD_DIM_MINOR`](splash_attention_kernel.md#QKVLayout.HEAD_DIM_MINOR)
- used by: [`_splash_attention_bwd`](splash_attention_kernel.md#_splash_attention_bwd), [`_make_splash_attention`](splash_attention_kernel.md#_make_splash_attention), [`_splash_attention`](splash_attention_kernel.md#_splash_attention), [`_splash_attention_custom`](splash_attention_kernel.md#_splash_attention_custom), [`_splash_attention_fwd`](splash_attention_kernel.md#_splash_attention_fwd), [`_splash_attention_forward`](splash_attention_kernel.md#_splash_attention_forward)

### `QKVLayout`  ·  implements/extends IntEnum
- def: [`jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py:476`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L476)
- signature: `class QKVLayout(enum.IntEnum):`
- members:
  - `HEAD_DIM_MINOR` — [`L477`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L477)
  - `SEQ_MINOR` — [`L478`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L478)
- used by: [`_splash_attention_bwd_dkv`](splash_attention_kernel.md#_splash_attention_bwd_dkv), [`_splash_attention_bwd_dq`](splash_attention_kernel.md#_splash_attention_bwd_dq), [`from_head_minor`](splash_attention_kernel.md#from_head_minor), [`_flash_attention_dkv_kernel`](splash_attention_kernel.md#_flash_attention_dkv_kernel), [`flash_attention_kernel`](splash_attention_kernel.md#flash_attention_kernel), [`_flash_attention_dq_kernel`](splash_attention_kernel.md#_flash_attention_dq_kernel), [`k_layout`](splash_attention_kernel.md#BlockSizes.k_layout), [`q_layout`](splash_attention_kernel.md#BlockSizes.q_layout), [`v_layout`](splash_attention_kernel.md#BlockSizes.v_layout)

### `SegmentIds`  ·  implements/extends NamedTuple
- def: [`jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py:46`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L46)
- doc: SegmentIds for Q and KV sequences.
- signature: `class SegmentIds(NamedTuple):`
- members:
  - `kv` — [`L67`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L67)
  - `q` — [`L66`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L66)
- uses (calls/refs, reference-scoped): [`Array`](../../../../../_src/basearray.md#Array)
- used by: [`_splash_attention_bwd`](splash_attention_kernel.md#_splash_attention_bwd), [`_attention_reference_default`](splash_attention_kernel.md#_attention_reference_default), [`_wrapped`](splash_attention_kernel.md#make_attention_reference._wrapped), [`_splash_attention`](splash_attention_kernel.md#_splash_attention), [`_splash_attention_custom`](splash_attention_kernel.md#_splash_attention_custom), [`_splash_attention_fwd`](splash_attention_kernel.md#_splash_attention_fwd), [`_splash_attention_forward`](splash_attention_kernel.md#_splash_attention_forward), [`attention_reference`](splash_attention_kernel.md#attention_reference), [`attention_reference_custom`](splash_attention_kernel.md#attention_reference_custom), [`_attention_reference`](splash_attention_kernel.md#_attention_reference), [`SplashResidualsType`](splash_attention_kernel.md#SplashResidualsType), [`_attention_reference_custom_fwd`](splash_attention_kernel.md#_attention_reference_custom_fwd)

### `SplashAttentionKernel`
- def: [`jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py:2448`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2448)
- signature: `class SplashAttentionKernel:`
- members:
  - `manual_sharding_spec(self, sharding: jax.sharding.NamedSharding)` — [`L2478`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2478) — Returns a value that can be used as a shard_map partition spec for the kernel.
  - `tree_flatten(self)` — [`L2516`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2516)
  - `tree_unflatten(cls, kwargs, values)` — [`L2523`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2523)
  - `dkv_mask_info` — [`L2460`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2460)
  - `dq_mask_info` — [`L2459`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2459)
  - `fwd_mask_info` — [`L2458`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2458)
  - `kwargs` — [`L2457`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2457)
- protocol/private: `__call__`[`L2462`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2462), `__init__`[`L2450`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2450)
- uses (calls/refs, reference-scoped): [`NamedSharding`](../../../../../_src/named_sharding.md#NamedSharding), [`PartitionSpec`](../../../../../_src/partition_spec.md#PartitionSpec), [`spec`](../../../../../_src/named_sharding.md#NamedSharding.spec), [`named_scope`](../../../../../_src/api.md#named_scope), [`MaskInfo`](splash_attention_mask_info.md#MaskInfo), [`_splash_attention`](splash_attention_kernel.md#_splash_attention), [`partial_mask_blocks`](splash_attention_mask_info.md#MaskInfo.partial_mask_blocks), [`data_next`](splash_attention_mask_info.md#MaskInfo.data_next), [`register_pytree_node_class`](../../../../../_src/tree_util.md#register_pytree_node_class), [`q_sequence`](splash_attention_mask_info.md#MaskInfo.q_sequence), [`shard_shape`](../../../../../_src/sharding.md#Sharding.shard_shape), [`block_mask`](splash_attention_mask_info.md#MaskInfo.block_mask), [`mask_next`](splash_attention_mask_info.md#MaskInfo.mask_next), [`SplashCustomReturnType`](splash_attention_kernel.md#SplashCustomReturnType), [`is_dynamic_mask`](splash_attention_mask_info.md#MaskInfo.is_dynamic_mask)
- used by: [`_make_splash_attention`](splash_attention_kernel.md#_make_splash_attention)

## Functions
- `_apply_mask_and_soft_cap(qk: jax.Array, mask_value: float, should_not_mask, mask_ref, q_sequence_ref, q_segment_ids_ref, kv_segment_ids_ref, *, attn_logits_soft_cap: float | None, k_slice: pl.Slice, k_offset: int | jax.Array, bq: int, k_in_lanes=True, mask_function=None)` — [`L596`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L596)
- `_attention_reference(mask: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None, mask_value: float, save_residuals: Literal[False], custom_type: str, attn_logits_soft_cap: float | None)` — [`L113`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L113)
- `_attention_reference_custom_bwd(mask_value: float, save_residuals: bool, custom_type: str, attn_logits_soft_cap: float | None, res, do: jax.Array)` — [`L267`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L267)
- `_attention_reference_custom_fwd(mask: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None, mask_value: float, save_residuals: bool, custom_type: str, attn_logits_soft_cap: float | None)` — [`L237`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L237)
- `_attention_reference_default(mask: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None, mask_value: float, save_residuals: bool, custom_type: str, attn_logits_soft_cap: float | None)` — [`L170`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L170)
- `_collapse_partial_mask_blocks(mask_info: mask_info_lib.MaskInfo | None)` — [`L2414`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2414)
- `_div(dividend: int, divisor: int)` — [`L893`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L893)
- `_flash_attention_dkv_kernel(data_next_ref, block_mask_ref, mask_next_ref, q_ref, k_ref, v_ref, q_segment_ids_ref, kv_segment_ids_ref, sinks_ref, logsumexp_ref, do_ref, di_ref, mask_ref, q_sequence_ref, dq_scratch_ref, dk_scratch_ref, dv_scratch_ref, dq_ref, dk_ref, dv_ref, *, num_q_heads: int, num_kv_heads: int, mask_value: float, grid_width: int, bq: int, bkv_compute: int, is_mqa: bool, attn_logits_soft_cap: float | None, q_layout: QKVLayout, k_layout: QKVLayout, v_layout: QKVLayout, bkv: int, mask_function: MaskFunctionType | None)` — [`L1679`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1679)
- `_flash_attention_dq_kernel(data_next_ref, block_mask_ref, mask_next_ref, q_ref, k_ref, v_ref, q_segment_ids_ref, kv_segment_ids_ref, sinks_ref, logsumexp_ref, do_ref, di_ref, mask_ref, q_sequence_ref, dq_scratch_ref, dq_ref, *, mask_value: float, grid_width: int, bq: int, bkv: int, attn_logits_soft_cap: float | None = None, q_layout: QKVLayout, k_layout: QKVLayout, v_layout: QKVLayout, mask_function: MaskFunctionType | None)` — [`L1313`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1313)
- `_load_kv(ref, layout)` — [`L1767`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1767)
- `_make_splash_attention(mask: np.ndarray | jax.Array | mask_lib.MultiHeadMask, *, block_sizes: BlockSizes | None = None, is_mqa: bool, save_residuals: bool = False, mask_value: float = DEFAULT_MASK_VALUE, attn_logits_soft_cap: float | None = None, downcast_smem_data: bool = True, head_shards: int, q_seq_shards: int, residual_checkpoint_name: str | None = None, interpret: bool = False)` — [`L2544`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2544)
- `_next_nonzero(h, i, j, data_next_ref, block_mask_ref, m_next_ref, next_i=False)` — [`L550`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L550)
- `_splash_attention(fwd_mask_info: mask_info_lib.MaskInfo, dq_mask_info: mask_info_lib.MaskInfo | None, dkv_mask_info: mask_info_lib.MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None = None, sinks: jax.Array | None = None, *, is_mqa: bool, block_sizes: BlockSizes | None, save_residuals: bool, mask_value: float, attn_logits_soft_cap: float | None, residual_checkpoint_name: str | None, mask_function: MaskFunctionType | None, interpret: bool)` — [`L2388`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2388) — For dynamic masks, `partial_mask_blocks` has shape (head_count, q_blocks, kv_blocks, block_q, block_kv).
- `_splash_attention_bwd(save_residuals: bool, mask_value: float, is_mqa: bool, block_sizes: BlockSizes, residual_checkpoint_name: str | None, mask_function: MaskFunctionType | None, attn_logits_soft_cap: float | None, interpret: bool, res: SplashResidualsType, do: jax.Array)` — [`L2255`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2255)
- `_splash_attention_bwd_dkv(q, k, v, segment_ids, sinks, logsumexp, do, di, *, bq: int, bkv: int, bkv_compute: int, is_mqa: bool, mask_info: mask_info_lib.MaskInfo, mask_value: float, attn_logits_soft_cap: float | None, use_fused_bwd_kernel: bool, q_layout: QKVLayout, k_layout: QKVLayout, v_layout: QKVLayout, mask_function: MaskFunctionType | None, interpret: bool)` — [`L1869`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1869)
- `_splash_attention_bwd_dq(q, k, v, segment_ids, sinks, logsumexp, do, di, *, bq: int, bkv: int, is_mqa: bool, mask_info: mask_info_lib.MaskInfo, mask_value: float, attn_logits_soft_cap: float | None, q_layout: QKVLayout, k_layout: QKVLayout, v_layout: QKVLayout, mask_function: MaskFunctionType | None, interpret: bool)` — [`L1413`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1413)
- `_splash_attention_custom(fwd_mask_info: mask_info_lib.MaskInfo, dq_mask_info: mask_info_lib.MaskInfo | None, dkv_mask_info: mask_info_lib.MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None, save_residuals: bool, mask_value: float, is_mqa: bool, block_sizes: BlockSizes, residual_checkpoint_name: str | None, mask_function: MaskFunctionType | None, attn_logits_soft_cap: float | None = None, interpret: bool = False)` — [`L1213`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1213)
- `_splash_attention_forward(fwd_mask_info: mask_info_lib.MaskInfo, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None, mask_value: float, is_mqa: bool, block_sizes: BlockSizes, residual_checkpoint_name: str | None, save_residuals: Literal[False], mask_function: MaskFunctionType | None, attn_logits_soft_cap: float | None = None, interpret: bool = False)` — [`L854`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L854)
- `_splash_attention_fwd(fwd_mask_info: mask_info_lib.MaskInfo, dq_mask_info: mask_info_lib.MaskInfo | None, dkv_mask_info: mask_info_lib.MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None, save_residuals: bool, mask_value: float, is_mqa: bool, block_sizes: BlockSizes, residual_checkpoint_name: str | None, mask_function: MaskFunctionType | None, attn_logits_soft_cap: float | None = None, interpret: bool = False)` — [`L1260`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1260)
- `_wrapped(mask: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None = None, sinks: jax.Array | None = None, *, mask_value: float = DEFAULT_MASK_VALUE, save_residuals: bool = False, attn_logits_soft_cap: float | None = None)` — [`L371`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L371)
- `attention_reference(mask: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None = None, *, mask_value: float = DEFAULT_MASK_VALUE, save_residuals: bool = False, custom_type: str = "flash", attn_logits_soft_cap: float | None = None)` — [`L210`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L210)
- `attention_reference_custom(mask: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None = None, *, mask_value: float = DEFAULT_MASK_VALUE, save_residuals: bool = False, custom_type: str = "flash", attn_logits_soft_cap: float | None = None)` — [`L330`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L330)
- `body(kv_compute_index, _)` — [`L756`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L756)
- `body(i, _)` — [`L1763`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1763)
- `cap_logits(logits)` — [`L680`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L680)
- `dkv_index_map(kv_index, head_index, *_)` — [`L2019`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2019)
- `dq_index_map(kv_index, head_index, q_index, *_)` — [`L2007`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2007)
- `end()` — [`L837`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L837)
- `end()` — [`L1408`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1408)
- `end()` — [`L1859`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1859)
- `flash_attention_kernel(data_next_ref, block_mask_ref, mask_next_ref, q_ref, k_ref, v_ref, q_segment_ids_ref, kv_segment_ids_ref, sinks_ref, mask_ref, q_sequence_ref, m_scratch_ref, l_scratch_ref, o_scratch_ref, o_ref, logsumexp_ref=None, *, mask_value: float, grid_width: int, bq: int, bkv: int, bkv_compute: int, head_dim_v: int, q_layout: QKVLayout, k_layout: QKVLayout, v_layout: QKVLayout, attn_logits_soft_cap: float | None, mask_function: MaskFunctionType | None)` — [`L696`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L696)
- `from_head_minor(vals: tuple[Any, ...], layout: QKVLayout)` — [`L481`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L481)
- `get_kernel_name(is_mqa: bool, save_residuals: bool, is_segmented: bool, phase: str)` — [`L93`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L93) — Returns a unique name for all SplashAttention kernel variants.
- `init()` — [`L736`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L736)
- `init()` — [`L1350`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1350)
- `init()` — [`L1749`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1749)
- `k_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L997`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L997)
- `k_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_)` — [`L1490`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1490)
- `k_index_map(kv_index, head_index, *_)` — [`L1982`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1982)
- `kv_segment_ids_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L1023`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1023)
- `kv_segment_ids_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_)` — [`L1532`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1532)
- `kv_segment_ids_index_map(kv_index, *_)` — [`L2073`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2073)
- `logsumexp_index_map(h, i, *_)` — [`L1118`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1118)
- `logsumexp_index_map(h, i, *_)` — [`L1568`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1568)
- `logsumexp_index_map(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L2102`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2102)
- `make_attention_reference(mask: mask_lib.Mask | np.ndarray, is_mqa: bool, backward_impl: str = "vanilla", **params: Any)` — [`L357`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L357)
- `mask_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L1012`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1012)
- `mask_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_)` — [`L1518`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1518)
- `mask_index_map(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref)` — [`L2032`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2032)
- `o_index_map(h, i, *_)` — [`L1478`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1478)
- `o_index_map(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L1938`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1938)
- `out_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L992`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L992)
- `q_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L989`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L989)
- `q_index_map(h, i, *_)` — [`L1483`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1483)
- `q_index_map(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L1959`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1959)
- `q_segment_ids_index_map(h, i, j, *_)` — [`L1019`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1019)
- `q_segment_ids_index_map(h, i, j, *_)` — [`L1526`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1526)
- `q_segment_ids_index_map(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L2053`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2053)
- `reshape_activations(activations)` — [`L442`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L442)
- `reshape_residuals(residuals)` — [`L450`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L450)
- `run()` — [`L829`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L829)
- `run()` — [`L1357`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1357)
- `run()` — [`L1838`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1838)
- `v_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None)` — [`L1005`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1005)
- `v_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_)` — [`L1504`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1504)
- `v_index_map(kv_index, head_index, *_)` — [`L1994`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1994)

## Module values
- `DEFAULT_MASK_VALUE` — [`L38`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L38)
- `MaskFunctionType` — [`L90`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L90)
- `NN_DIM_NUMBERS` — [`L42`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L42)
- `NT_DIM_NUMBERS` — [`L43`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L43)
- `NUM_LANES` — [`L39`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L39)
- `NUM_SUBLANES` — [`L40`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L40)
- `SplashCustomReturnType` — [`L71`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L71)
- `SplashResidualsType` — [`L78`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L78)
- `_attention_reference_custom` — [`L322`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L322)
- `make_masked_mha_reference` — [`L469`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L469)
- `make_masked_mqa_reference` — [`L470`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L470)
- `make_splash_mha` — [`L2634`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2634)
- `make_splash_mha_single_device` — [`L2637`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2637)
- `make_splash_mqa` — [`L2635`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2635)
- `make_splash_mqa_single_device` — [`L2641`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L2641)
- `partial` — [`L37`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L37)

