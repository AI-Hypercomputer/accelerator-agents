---
title: 'Module: ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.ring_attention._pallas_impl_bwd`/
symbols:
  make_ring_attention: make_ring_attention().
  _ring_attention_forward: _ring_attention_forward().
  ring_splash_attention: ring_splash_attention().
  _ring_attention_forward.body: _ring_attention_forward().body().
  MaskInfo: MaskInfo.
  _ring_attention_custom: _ring_attention_custom().
  _ring_attention_backward: _ring_attention_backward().
  BlockSizes: BlockSizes.
  _ring_attention_fwd_rule: _ring_attention_fwd_rule().
  _ring_attention_custom_fwd: _ring_attention_custom_fwd().
  RingSplashAttentionKernel.__call__: RingSplashAttentionKernel#__call__().
  MaskFunctionType: MaskFunctionType.
  RING_AXIS: RING_AXIS.
  RingSplashAttentionKernel.tree_flatten: RingSplashAttentionKernel#tree_flatten().
  SegmentIds: SegmentIds#
  _ring_attention_bwd_rule: _ring_attention_bwd_rule().
  _ring_attention_custom_bwd: _ring_attention_custom_bwd().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  RingSplashAttentionKernel.__init__: RingSplashAttentionKernel#__init__().
  partial: partial.
  _ring_attention_backward.body: _ring_attention_backward().body().
  RingSplashAttentionKernel: RingSplashAttentionKernel#
  RingSplashAttentionKernel.tree_unflatten: RingSplashAttentionKernel#tree_unflatten().
  SegmentIds.q: SegmentIds#q.
  SegmentIds.kv: SegmentIds#kv.
  RingSplashAttentionKernel.fwd_mask_info: RingSplashAttentionKernel#fwd_mask_info.
  RingSplashAttentionKernel.dkv_mask_info: RingSplashAttentionKernel#dkv_mask_info.
  RingSplashAttentionKernel.ring_axis: RingSplashAttentionKernel#ring_axis.
  RingSplashAttentionKernel.kwargs: RingSplashAttentionKernel#kwargs.
  _update_out_and_lse: _update_out_and_lse().
  _ring_attention_forward.ring_mask_fn: _ring_attention_forward().ring_mask_fn().
  _has_axis: _has_axis().
---
# Module: [`ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py)

## Classes
### `RingSplashAttentionKernel`
- def: [`ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py:750`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L750) — documented in [ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd.md)
- doc: Callable ring attention kernel with pre-computed mask information.
- signature: `class RingSplashAttentionKernel:`
- members:
  - `__call__(self, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None = None, sinks: jax.Array | None = None)` — [`L786`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L786) — Compute ring attention with the pre-configured mask and parameters. — documented in [ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd.md)
  - `__init__(self, fwd_mask_info: MaskInfo, dkv_mask_info: MaskInfo | None, ring_axis: str = RING_AXIS, **kwargs)` — [`L765`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L765) — Initialize the ring splash attention kernel.
  - `tree_flatten(self)` — [`L818`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L818) — Flatten for JAX pytree serialization.
  - `tree_unflatten(cls, aux_data, children)` — [`L830`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L830) — Reconstruct from JAX pytree serialization.
  - `dkv_mask_info` — [`L782`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L782)
  - `fwd_mask_info` — [`L781`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L781)
  - `kwargs` — [`L784`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L784)
  - `ring_axis` — [`L783`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L783)
- uses (calls/refs, reference-scoped): [`ring_splash_attention`](_pallas_impl_bwd.md#ring_splash_attention), [`MaskInfo`](_pallas_impl_bwd.md#MaskInfo), [`RING_AXIS`](_pallas_impl_bwd.md#RING_AXIS), [`SegmentIds`](_pallas_impl_bwd.md#SegmentIds)
- used by: [`make_ring_attention`](_pallas_impl_bwd.md#make_ring_attention)

### `SegmentIds`  ·  implements/extends NamedTuple
- def: [`ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py:76`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L76) — documented in [ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd.md)
- doc: Segment IDs for query and KV sequences in packed-sequence attention.
- signature: `class SegmentIds(NamedTuple):`
- members:
  - `kv` — [`L91`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L91)
  - `q` — [`L90`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L90)
- used by: [`ring_attention`](_pallas_impl_fwd.md#ring_attention), [`_ring_attention_forward`](_pallas_impl_bwd.md#_ring_attention_forward), [`ring_splash_attention`](_pallas_impl_bwd.md#ring_splash_attention), [`_ring_attention_custom`](_pallas_impl_bwd.md#_ring_attention_custom), [`_ring_attention_custom_fwd`](_pallas_impl_bwd.md#_ring_attention_custom_fwd), [`_ring_attention_fwd_rule`](_pallas_impl_bwd.md#_ring_attention_fwd_rule), [`__call__`](_pallas_impl_bwd.md#RingSplashAttentionKernel.__call__)

## Functions
- `_has_axis(axis_name: str)` — [`L646`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L646) — Check whether a named axis exists in the current JAX context.
- `_ring_attention_backward(res: tuple, do: jax.Array, *, mask_value: float, is_mqa: bool, block_sizes: BlockSizes, mask_function: MaskFunctionType | None, logits_soft_cap: float | None, ring_axis: str)` — [`L272`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L272) — Backward pass for ring attention with distributed gradient accumulation.
- `_ring_attention_bwd_rule(mask_value: float, is_mqa: bool, block_sizes: BlockSizes, mask_function: MaskFunctionType | None, logits_soft_cap: float | None, ring_axis: str, res: tuple, do: jax.Array)` — [`L466`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L466) — Custom VJP backward rule for ring attention.
- `_ring_attention_custom(fwd_mask_info: MaskInfo, dq_mask_info: MaskInfo | None, dkv_mask_info: MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None, mask_value: float, is_mqa: bool, block_sizes: BlockSizes, mask_function: MaskFunctionType | None, logits_soft_cap: float | None, ring_axis: str = RING_AXIS, causal: bool = False, sliding_window: int | tuple[int, int] | None = None)` — [`L512`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L512) — Ring attention with custom VJP for efficient gradient computation. — documented in [ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd.md)
- `_ring_attention_custom_bwd(mask_value: float, is_mqa: bool, block_sizes: BlockSizes, mask_function: MaskFunctionType | None, logits_soft_cap: float | None, ring_axis: str, causal: bool, sliding_window: int | tuple[int, int] | None, res: tuple, do: jax.Array)` — [`L614`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L614) — VJP backward implementation for _ring_attention_custom.
- `_ring_attention_custom_fwd(fwd_mask_info: MaskInfo, dq_mask_info: MaskInfo | None, dkv_mask_info: MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None, mask_value: float, is_mqa: bool, block_sizes: BlockSizes, mask_function: MaskFunctionType | None, logits_soft_cap: float | None, ring_axis: str = RING_AXIS, causal: bool = False, sliding_window: int | tuple[int, int] | None = None)` — [`L575`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L575) — VJP forward implementation for _ring_attention_custom.
- `_ring_attention_forward(fwd_mask_info: MaskInfo, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None, mask_value: float, is_mqa: bool, block_sizes: BlockSizes, mask_function: MaskFunctionType | None, logits_soft_cap: float | None, ring_axis: str = RING_AXIS, causal: bool = False, sliding_window: int | tuple[int, int] | None = None)` — [`L130`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L130) — Forward pass for ring attention over distributed devices. — documented in [ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd.md)
- `_ring_attention_fwd_rule(fwd_mask_info: MaskInfo, dq_mask_info: MaskInfo | None, dkv_mask_info: MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None, *, mask_value: float, is_mqa: bool, block_sizes: BlockSizes, mask_function: MaskFunctionType | None, logits_soft_cap: float | None, ring_axis: str = RING_AXIS, causal: bool = False, sliding_window: int | tuple[int, int] | None = None)` — [`L391`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L391) — Custom VJP forward rule for ring attention.
- `_update_out_and_lse(out: jax.Array, lse: jax.Array, block_out: jax.Array, block_lse: jax.Array)` — [`L94`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L94) — Numerically stable update of attention output and log-sum-exp.
- `body(carry, iteration)` — [`L210`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L210) — documented in [ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd.md)
- `body(carry, _: int)` — [`L326`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L326)
- `make_ring_attention(mask: np.ndarray | jax.Array | mask_lib.Mask, *, block_sizes: BlockSizes | None = None, is_mqa: bool = False, mask_value: float = DEFAULT_MASK_VALUE, logits_soft_cap: float | None = None, ring_axis: str = RING_AXIS, q_seq_shards: int = 1)` — [`L848`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L848) — Create a ring attention kernel from an attention mask. — documented in [ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd.md)
- `ring_mask_fn(q_ids, kv_ids)` — [`L201`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L201) — Dynamic mask function for ring rotation combining causal + sliding window.
- `ring_splash_attention(fwd_mask_info: MaskInfo, dkv_mask_info: MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None = None, sinks: jax.Array | None = None, *, is_mqa: bool, block_sizes: BlockSizes, mask_value: float = DEFAULT_MASK_VALUE, mask_function: MaskFunctionType | None = None, logits_soft_cap: float | None = None, ring_axis: str = RING_AXIS, causal: bool = False, sliding_window: int | tuple[int, int] | None = None)` — [`L679`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L679) — Compute ring attention using Splash Attention kernels.

## Module values
- `BlockSizes` — [`L71`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L71) — documented in [ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd.md)
- `DEFAULT_MASK_VALUE` — [`L73`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L73)
- `MaskFunctionType` — [`L72`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L72) — documented in [ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd.md)
- `MaskInfo` — [`L70`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L70) — documented in [ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd.md)
- `RING_AXIS` — [`L68`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L68) — documented in [ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd.md)
- `partial` — [`L66`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py#L66)

