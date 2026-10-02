---
title: 'Module: src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.kernels.splash_attention.ring_attention_kernel`/
symbols:
  make_ring_attention: make_ring_attention().
  RingSplashAttentionKernel.manual_sharding_spec: RingSplashAttentionKernel#manual_sharding_spec().
  _dynamic_slice_mask_info: _dynamic_slice_mask_info().
  _ring_attention_forward: _ring_attention_forward().
  _ring_attention: _ring_attention().
  _ring_attention_custom: _ring_attention_custom().
  _ring_attention_fwd: _ring_attention_fwd().
  MaskInfo: MaskInfo.
  RingSplashAttentionKernel.__call__: RingSplashAttentionKernel#__call__().
  RingSplashAttentionKernel.tree_flatten: RingSplashAttentionKernel#tree_flatten().
  _ring_attention_bwd: _ring_attention_bwd().
  _custom_bidirectional_ring_forward: _custom_bidirectional_ring_forward().
  _custom_ring_attention_forward: _custom_ring_attention_forward().
  SegmentIds: SegmentIds.
  SplashConfig: SplashConfig.
  partial: partial.
  _ring_attention_forward.body: _ring_attention_forward().body().
  _ring_attention_bwd.body: _ring_attention_bwd().body().
  RingSplashAttentionKernel.fwd_mask_info: RingSplashAttentionKernel#fwd_mask_info.
  MaskFunctionType: MaskFunctionType.
  make_custom_ring_attention: make_custom_ring_attention().
  RingSplashAttentionKernel._resolve_spec: RingSplashAttentionKernel#_resolve_spec().
  _custom_bidirectional_ring_forward.body: _custom_bidirectional_ring_forward().body().
  _custom_bidirectional_ring_forward._attn: _custom_bidirectional_ring_forward()._attn().
  _dynamic_slice_mask_info.slice_if_exists: _dynamic_slice_mask_info().slice_if_exists().
  SplashCustomReturnType: SplashCustomReturnType.
  RingSplashAttentionKernel.ring_axis: RingSplashAttentionKernel#ring_axis.
  SplashResidualsType: SplashResidualsType.
  _splash_attention_forward_ring_raw: _splash_attention_forward_ring_raw.
  _splash_attention_bwd: _splash_attention_bwd.
  _custom_ring_attention_forward.body: _custom_ring_attention_forward().body().
  make_custom_ring_attention._ring: make_custom_ring_attention()._ring().
  RingSplashAttentionKernel.dkv_mask_info: RingSplashAttentionKernel#dkv_mask_info.
  RingSplashAttentionKernel.rotate_segment_ids: RingSplashAttentionKernel#rotate_segment_ids.
  RingSplashAttentionKernel.kwargs: RingSplashAttentionKernel#kwargs.
  _splash_attention_forward: _splash_attention_forward.
  RingSplashAttentionKernel.__init__: RingSplashAttentionKernel#__init__().
  RingSplashAttentionKernel.tree_unflatten: RingSplashAttentionKernel#tree_unflatten().
  RingSplashAttentionKernel: RingSplashAttentionKernel#
  _custom_bidirectional_ring_forward._merge: _custom_bidirectional_ring_forward()._merge().
  _has_axis: _has_axis().
  P: P.
---
# Module: [`src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py)

## Classes
### `RingSplashAttentionKernel`
- def: [`src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py:535`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L535)
- doc: Implements Ring Attention using SplashAttention for sequence parallelism.
- signature: `class RingSplashAttentionKernel:`
- members:
  - `manual_sharding_spec(self)` — [`L576`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L576) — Ring attention expects MaskInfo to be sharded by `q_seq_shards`. — documented in [maxdiffusion-kernels-splash_attention-splash_attention_mask_info](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_mask_info.md)
  - `tree_flatten(self)` — [`L606`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L606)
  - `tree_unflatten(cls, aux_data, children)` — [`L614`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L614)
  - `dkv_mask_info` — [`L560`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L560)
  - `fwd_mask_info` — [`L559`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L559)
  - `kwargs` — [`L563`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L563)
  - `ring_axis` — [`L561`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L561)
  - `rotate_segment_ids` — [`L562`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L562)
- protocol/private: `__call__`[`L565`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L565), `__init__`[`L551`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L551), `_resolve_spec`[`L586`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L586)
- uses (calls/refs, reference-scoped): [`MaskInfo`](splash_attention_mask_info.md#MaskInfo), [`partial_mask_blocks`](splash_attention_mask_info.md#MaskInfo.partial_mask_blocks), [`block_mask`](splash_attention_mask_info.md#MaskInfo.block_mask), [`q_sequence`](splash_attention_mask_info.md#MaskInfo.q_sequence), [`active_rows`](splash_attention_mask_info.md#MaskInfo.active_rows), [`mask_next`](splash_attention_mask_info.md#MaskInfo.mask_next), [`active_cols`](splash_attention_mask_info.md#MaskInfo.active_cols), [`_ring_attention`](ring_attention_kernel.md#_ring_attention), [`num_active_blocks`](splash_attention_mask_info.md#MaskInfo.num_active_blocks), [`MaskInfo`](ring_attention_kernel.md#MaskInfo)
- used by: [`make_ring_attention`](ring_attention_kernel.md#make_ring_attention), [`test_ring_attention`](ring_attention_kernel_test.md#RingAttentionTest.test_ring_attention)

## Functions
- `_attn(kc, vc)` — [`L765`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L765)
- `_custom_bidirectional_ring_forward(q: jax.Array, k: jax.Array, v: jax.Array, *, block_sizes: custom_splash._BlockSizes, bkv_compute_in: int, orig_q_seq_len: int, orig_kv_seq_len: int, use_base2_exp: bool, use_experimental_scheduler: bool, vmem_limit_bytes: int | None, mask_value: float, ring_axis: str)` — [`L729`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L729) — Wrap-free (bidirectional) ring attention for a NON-wrapping ring axis.
- `_custom_ring_attention_forward(q: jax.Array, k: jax.Array, v: jax.Array, *, block_sizes: custom_splash._BlockSizes, bkv_compute_in: int, orig_q_seq_len: int, orig_kv_seq_len: int, use_base2_exp: bool, use_experimental_scheduler: bool, vmem_limit_bytes: int | None, mask_value: float, ring_axis: str, ring_size: int | None = None, perm: list[tuple[int, int]] | None = None, bidirectional: bool = False)` — [`L832`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L832) — Forward-only ring attention using the custom dense splash kernel.
- `_dynamic_slice_mask_info(mask_info: MaskInfo, kv_shard_idx: jax.Array, ring_size: int)` — [`L46`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L46) — Slices MaskInfo for the current ring step.
- `_has_axis(axis_name: str)` — [`L438`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L438)
- `_merge(m, l, o, mc, lc, oc, valid)` — [`L780`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L780)
- `_ring(q, k, v)` — [`L992`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L992)
- `_ring_attention(fwd_mask_info: MaskInfo, dkv_mask_info: MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None = None, sinks: jax.Array | None = None, *, is_mqa: bool, config: SplashConfig | None, mask_value: float, mask_function: MaskFunctionType | None, fwd_mask_sparsity: float, dkv_mask_sparsity: float, save_residuals: bool = False, ring_axis: str, rotate_segment_ids: bool = True)` — [`L463`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L463) — Performs ring attention using SplashAttention kernels.
- `_ring_attention_bwd(mask_value: float, is_mqa: bool, config: SplashConfig | None, mask_function: MaskFunctionType | None, fwd_mask_sparsity: float, dkv_mask_sparsity: float, save_residuals: bool, ring_axis: str, rotate_segment_ids: bool, res: Any, do: jax.Array)` — [`L179`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L179)
- `_ring_attention_custom(fwd_mask_info: MaskInfo, dkv_mask_info: MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None, mask_value: float, is_mqa: bool, config: SplashConfig | None, mask_function: MaskFunctionType | None, fwd_mask_sparsity: float, dkv_mask_sparsity: float, save_residuals: bool, ring_axis: str, rotate_segment_ids: bool)` — [`L373`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L373) — Performs ring attention with a custom VJP.
- `_ring_attention_forward(fwd_mask_info: MaskInfo, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, mask_value: float, is_mqa: bool, config: SplashConfig | None, mask_function: MaskFunctionType | None, fwd_mask_sparsity: float, *, sinks: jax.Array | None = None, ring_axis: str, rotate_segment_ids: bool = True)` — [`L68`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L68)
- `_ring_attention_fwd(fwd_mask_info: MaskInfo, dkv_mask_info: MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None, mask_value: float, is_mqa: bool, config: SplashConfig | None, mask_function: MaskFunctionType | None, fwd_mask_sparsity: float, dkv_mask_sparsity: float, save_residuals: bool, ring_axis: str, rotate_segment_ids: bool)` — [`L290`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L290) — Forward pass for the custom VJP of ring attention.
- `body(carry, i: int)` — [`L126`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L126)
- `body(carry, i: int)` — [`L210`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L210)
- `body(carry, t)` — [`L802`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L802)
- `body(carry, i)` — [`L918`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L918)
- `make_custom_ring_attention(*, block_sizes: custom_splash._BlockSizes, bkv_compute_in: int, orig_q_seq_len: int, orig_kv_seq_len: int, use_base2_exp: bool = True, use_experimental_scheduler: bool = False, vmem_limit_bytes: int | None = None, mask_value: float = base.DEFAULT_MASK_VALUE, ring_axis: str = "context", ring_size: int | None = None, perm: list[tuple[int, int]] | None = None, bidirectional: bool = False)` — [`L961`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L961) — Builds a forward-only ring-attention callable around the custom kernel.
- `make_ring_attention(mask: np.ndarray | mask_lib.Mask, *, config: SplashConfig | None = None, is_mqa: bool, save_residuals: bool = False, mask_value: float = base.DEFAULT_MASK_VALUE, downcast_smem_data: bool = True, partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8, ring_axis: str, q_seq_shards: int = 1, kv_seq_shards: int = 1, rotate_segment_ids: bool = True)` — [`L624`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L624) — Creates a RingSplashAttentionKernel. — documented in [maxdiffusion-kernels-splash_attention-splash_attention_mask_info](../../../../../concepts/maxdiffusion-kernels-splash_attention-splash_attention_mask_info.md)
- `slice_if_exists(arr: jax.Array | None)` — [`L49`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L49)

## Module values
- `MaskFunctionType` — [`L40`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L40)
- `MaskInfo` — [`L33`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L33)
- `P` — [`L32`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L32)
- `SegmentIds` — [`L36`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L36)
- `SplashConfig` — [`L37`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L37)
- `SplashCustomReturnType` — [`L39`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L39)
- `SplashResidualsType` — [`L38`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L38)
- `_splash_attention_bwd` — [`L43`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L43)
- `_splash_attention_forward` — [`L41`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L41)
- `_splash_attention_forward_ring_raw` — [`L42`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L42)
- `partial` — [`L34`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py#L34)

