---
title: 'Module: tokamax/_src/ops/experimental/tpu/splash_attention/base.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/experimental/tpu/splash_attention/base.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.experimental.tpu.splash_attention.base`/
symbols:
  SegmentIds: SegmentIds#
  attention_reference: attention_reference().
  _attention_reference_impl: _attention_reference_impl().
  SegmentIds.q: SegmentIds#q.
  SegmentIds.kv: SegmentIds#kv.
  SplashResidualsType: SplashResidualsType.
  SplashCustomReturnType.SplashCustomReturnType: SplashCustomReturnType.SplashCustomReturnType.
  DEFAULT_MASK_VALUE.DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.DEFAULT_MASK_VALUE.
  MaskInfo: MaskInfo.
  _attention_reference_custom_bwd: _attention_reference_custom_bwd().
  attention_reference_vjp: attention_reference_vjp().
---
# Module: [`tokamax/_src/ops/experimental/tpu/splash_attention/base.py`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/base.py)

## Classes
### `SegmentIds`  ·  implements/extends NamedTuple
- def: [`tokamax/_src/ops/experimental/tpu/splash_attention/base.py:33`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/base.py#L33) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel.md)
- doc: SegmentIds for Q and KV sequences.
- signature: `class SegmentIds(NamedTuple):`
- members:
  - `kv` — [`L54`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/base.py#L54) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel.md)
  - `q` — [`L53`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/base.py#L53) — documented in [tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel](../../../../../../../concepts/tokamax-_src-ops-experimental-tpu-splash_attention-splash_attention_kernel.md)
- used by: [`_splash_attention_forward`](splash_attention_kernel.md#_splash_attention_forward), [`test_ring_attention`](ring_attention_kernel_test.md#RingAttentionTest.test_ring_attention), [`_splash_attention_bwd`](splash_attention_kernel.md#_splash_attention_bwd), [`test_manual_partitioning_mha_bwd`](splash_attention_kernel_sharded_test.md#SplashAttentionShardingTest.test_manual_partitioning_mha_bwd), [`test_manual_partitioning_mha_fwd`](splash_attention_kernel_sharded_test.md#SplashAttentionShardingTest.test_manual_partitioning_mha_fwd), [`_splash_attention_custom`](splash_attention_kernel.md#_splash_attention_custom), [`_splash_attention_fwd`](splash_attention_kernel.md#_splash_attention_fwd), [`_splash_attention`](splash_attention_kernel.md#_splash_attention), [`attention_reference`](base.md#attention_reference), [`_attention_reference_impl`](base.md#_attention_reference_impl), [`create_segment_ids`](splash_attention_test_utils.md#create_segment_ids), [`SegmentIds`](ring_attention_kernel.md#SegmentIds), [`body`](ring_attention_kernel.md#_ring_attention_forward.body), [`SplashResidualsType`](base.md#SplashResidualsType), [`segment_ids_strategy`](splash_attention_kernel_test.md#segment_ids_strategy)

## Functions
- `_attention_reference_custom_bwd(do, q, k, v, mask, segment_ids, sinks, o, logsumexp, mask_value: float = DEFAULT_MASK_VALUE, backward_impl: str = "vanilla", attn_logits_soft_cap: float | None = None)` — [`L115`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/base.py#L115)
- `_attention_reference_impl(q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array, segment_ids: SegmentIds | None, sinks: jax.Array | None, mask_value: float, save_residuals: bool, attn_logits_soft_cap: float | None)` — [`L74`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/base.py#L74)
- `attention_reference(q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array, segment_ids: SegmentIds | None = None, sinks: jax.Array | None = None, *, is_mqa: bool, mask_value: float = DEFAULT_MASK_VALUE, save_residuals: bool = False, attn_logits_soft_cap: float | None = None)` — [`L185`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/base.py#L185) — A JIT-compiled reference implementation of attention, handles MQA and MHA.
- `attention_reference_vjp(do, q, k, v, mask, segment_ids, sinks, o, logsumexp, *, is_mqa: bool, backward_impl: str = "vanilla", attn_logits_soft_cap: float | None = None)` — [`L235`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/base.py#L235) — Wrapper for backward reference that handles GQA/MQA broadcasting and reduction.

## Module values
- `DEFAULT_MASK_VALUE` — [`L28`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/base.py#L28)
- `MaskInfo` — [`L25`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/base.py#L25)
- `SplashCustomReturnType` — [`L58`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/base.py#L58)
- `SplashResidualsType` — [`L62`](../../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/base.py#L62)

