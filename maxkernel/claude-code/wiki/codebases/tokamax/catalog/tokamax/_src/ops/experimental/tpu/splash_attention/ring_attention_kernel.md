---
title: 'Module: tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.experimental.tpu.splash_attention.ring_attention_kernel`/
symbols:
  make_ring_attention: make_ring_attention().
  RingSplashAttentionKernel.manual_sharding_spec: RingSplashAttentionKernel#manual_sharding_spec().
  _dynamic_slice_mask_info: _dynamic_slice_mask_info().
  _ring_attention: _ring_attention().
  _ring_attention_forward: _ring_attention_forward().
  _ring_attention_custom: _ring_attention_custom().
  _ring_attention_fwd: _ring_attention_fwd().
  MaskInfo: MaskInfo.
  RingSplashAttentionKernel.__call__: RingSplashAttentionKernel#__call__().
  _ring_attention_bwd: _ring_attention_bwd().
  RingSplashAttentionKernel.tree_flatten: RingSplashAttentionKernel#tree_flatten().
  SplashConfig: SplashConfig.
  SegmentIds: SegmentIds.
  _ring_attention_forward.body: _ring_attention_forward().body().
  _ring_attention_bwd.body: _ring_attention_bwd().body().
  RingSplashAttentionKernel.fwd_mask_info: RingSplashAttentionKernel#fwd_mask_info.
  MaskFunctionType: MaskFunctionType.
  partial: partial.
  _dynamic_slice_mask_info.slice_if_exists: _dynamic_slice_mask_info().slice_if_exists().
  SplashCustomReturnType: SplashCustomReturnType.
  RingSplashAttentionKernel.ring_axis: RingSplashAttentionKernel#ring_axis.
  SplashResidualsType: SplashResidualsType.
  _splash_attention_forward: _splash_attention_forward.
  _splash_attention_bwd: _splash_attention_bwd.
  RingSplashAttentionKernel.dkv_mask_info: RingSplashAttentionKernel#dkv_mask_info.
  RingSplashAttentionKernel.kwargs: RingSplashAttentionKernel#kwargs.
  RingSplashAttentionKernel.__init__: RingSplashAttentionKernel#__init__().
  RingSplashAttentionKernel.tree_unflatten: RingSplashAttentionKernel#tree_unflatten().
  RingSplashAttentionKernel: RingSplashAttentionKernel#
  _has_axis: _has_axis().
  P: P.
---
# Module: [`tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py)

## Classes
### `RingSplashAttentionKernel`
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py:531`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L531)
- doc: Implements Ring Attention using SplashAttention for sequence parallelism.
- signature: `class RingSplashAttentionKernel:`
- members:
  - `manual_sharding_spec(self)` — [`L569`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L569) — Ring attention expects MaskInfo to be sharded by `q_seq_shards`.
  - `tree_flatten(self)` — [`L596`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L596)
  - `tree_unflatten(cls, aux_data, children)` — [`L603`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L603)
  - `dkv_mask_info` — [`L555`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L555)
  - `fwd_mask_info` — [`L554`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L554)
  - `kwargs` — [`L557`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L557)
  - `ring_axis` — [`L556`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L556)
- protocol/private: `__call__`[`L559`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L559), `__init__`[`L547`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L547)
- uses (calls/refs, reference-scoped): [`MaskInfo`](splash_attention_mask_info.md#MaskInfo), [`block_mask`](splash_attention_mask_info.md#MaskInfo.block_mask), [`partial_mask_blocks`](splash_attention_mask_info.md#MaskInfo.partial_mask_blocks), [`q_sequence`](splash_attention_mask_info.md#MaskInfo.q_sequence), [`active_rows`](splash_attention_mask_info.md#MaskInfo.active_rows), [`mask_next`](splash_attention_mask_info.md#MaskInfo.mask_next), [`_ring_attention`](ring_attention_kernel.md#_ring_attention), [`active_cols`](splash_attention_mask_info.md#MaskInfo.active_cols), [`num_active_blocks`](splash_attention_mask_info.md#MaskInfo.num_active_blocks), [`MaskInfo`](ring_attention_kernel.md#MaskInfo)
- used by: [`make_ring_attention`](ring_attention_kernel.md#make_ring_attention), [`test_ring_attention`](ring_attention_kernel_test.md#RingAttentionTest.test_ring_attention)

## Functions
- `_dynamic_slice_mask_info(mask_info: MaskInfo, kv_shard_idx: jax.Array, ring_size: int)` — [`L44`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L44) — Slices MaskInfo for the current ring step. — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-ring_attention_kernel](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-ring_attention_kernel.md)
- `_has_axis(axis_name: str)` — [`L438`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L438)
- `_ring_attention(fwd_mask_info: MaskInfo, dkv_mask_info: MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None = None, sinks: jax.Array | None = None, *, is_mqa: bool, config: SplashConfig | None, mask_value: float, mask_function: MaskFunctionType | None, fwd_mask_sparsity: float, dkv_mask_sparsity: float, save_residuals: bool = False, ring_axis: str)` — [`L462`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L462) — Performs ring attention using SplashAttention kernels. — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-ring_attention_kernel](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-ring_attention_kernel.md)
- `_ring_attention_bwd(mask_value: float, is_mqa: bool, config: SplashConfig | None, mask_function: MaskFunctionType | None, fwd_mask_sparsity: float, dkv_mask_sparsity: float, save_residuals: bool, ring_axis: str, res: Any, do: jax.Array)` — [`L182`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L182)
- `_ring_attention_custom(fwd_mask_info: MaskInfo, dkv_mask_info: MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None, mask_value: float, is_mqa: bool, config: SplashConfig | None, mask_function: MaskFunctionType | None, fwd_mask_sparsity: float, dkv_mask_sparsity: float, save_residuals: bool, ring_axis: str)` — [`L376`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L376) — Performs ring attention with a custom VJP.
- `_ring_attention_forward(fwd_mask_info: MaskInfo, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, mask_value: float, is_mqa: bool, config: SplashConfig | None, mask_function: MaskFunctionType | None, fwd_mask_sparsity: float, *, sinks: jax.Array | None = None, ring_axis: str)` — [`L68`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L68)
- `_ring_attention_fwd(fwd_mask_info: MaskInfo, dkv_mask_info: MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None, mask_value: float, is_mqa: bool, config: SplashConfig | None, mask_function: MaskFunctionType | None, fwd_mask_sparsity: float, dkv_mask_sparsity: float, save_residuals: bool, ring_axis: str)` — [`L296`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L296) — Forward pass for the custom VJP of ring attention.
- `body(carry, i: int)` — [`L126`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L126)
- `body(carry, i: int)` — [`L212`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L212)
- `make_ring_attention(mask: np.ndarray | mask_lib.Mask, *, config: SplashConfig | None = None, is_mqa: bool, save_residuals: bool = False, mask_value: float = base.DEFAULT_MASK_VALUE, downcast_smem_data: bool = True, partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8, ring_axis: str, q_seq_shards: int = 1, kv_seq_shards: int = 1)` — [`L617`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L617) — Creates a RingSplashAttentionKernel. — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-ring_attention_kernel](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-ring_attention_kernel.md)
- `slice_if_exists(arr: jax.Array | None)` — [`L49`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L49)

## Module values
- `MaskFunctionType` — [`L39`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L39)
- `MaskInfo` — [`L32`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L32)
- `P` — [`L31`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L31)
- `SegmentIds` — [`L35`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L35)
- `SplashConfig` — [`L36`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L36)
- `SplashCustomReturnType` — [`L38`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L38)
- `SplashResidualsType` — [`L37`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L37)
- `_splash_attention_bwd` — [`L41`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L41)
- `_splash_attention_forward` — [`L40`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L40)
- `partial` — [`L33`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py#L33)

