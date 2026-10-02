---
title: 'Module: easydel/operations/kernels/ring_attention.py'
type: catalog
provenance: extracted
module: easydel/operations/kernels/ring_attention.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.operations.kernels.ring_attention`/
symbols:
  RingAttn.forward_native: RingAttn#forward_native().
  RingAttn.get_requirements: RingAttn#get_requirements().
  vout: vout.
  out: out.
  q: q.
  k: k.
  v: v.
  mask_info: mask_info.
  RingAttn: RingAttn#
  ring: ring.
  RingAttn.forward_cuda: RingAttn#forward_cuda().
  RingAttn.forward_gpu: RingAttn#forward_gpu().
  RingAttn.forward_tpu: RingAttn#forward_tpu().
  RingAttn.forward_cpu: RingAttn#forward_cpu().
  RingAttn.forward_rocm: RingAttn#forward_rocm().
  RingAttn.__call__: RingAttn#__call__().
  b: b.
  ks: ks.
  qs: qs.
  kh: kh.
  d: d.
  qh: qh.
  vd: vd.
  RingAttn.get_impl_name: RingAttn#get_impl_name().
---
# Module: [`easydel/operations/kernels/ring_attention.py`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py)

## Classes
### `RingAttn`  ·  implements/extends OperationImpl
- def: [`easydel/operations/kernels/ring_attention.py:84`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L84)
- doc: Ring attention implementation for distributed and memory-efficient processing.
- signature: `class RingAttn(OperationImpl):`
- members:
  - `__call__(self, query: Float[Array, "batch seq_len_q num_heads head_dim"], key: Float[Array, "batch seq_len_k num_kv_heads head_dim"], value: Float[Array, "batch seq_len_k num_kv_heads head_dim"], softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, num_sinks] | None = None, mask_info: MaskInfo | None = None, logits_soft_cap: float | None = None, softmax_scale: float | None = None, sliding_window: int | tuple[int, int] | None = None, causal: bool = True, fused_backward: bool = False, **ignore)` — [`L319`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L319) — Executes the Ring Attention computation.
  - `forward_cpu(self, *args, **kwargs)` — [`L279`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L279) — CPU forward pass. Delegates to `forward_native` (scan-based).
  - `forward_cuda(self, *args, **kwargs)` — [`L291`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L291) — CUDA GPU forward pass. Currently delegates to `forward_native` (scan-based).
  - `forward_gpu(self, *args, **kwargs)` — [`L262`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L262) — GPU forward pass. Currently delegates to `forward_native` (scan-based).
  - `forward_native(self, query: Float[Array, "batch seq_len_q num_heads head_dim"], key: Float[Array, "batch seq_len_k num_kv_heads head_dim"], value: Float[Array, "batch seq_len_k num_kv_heads head_dim"], softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, num_sinks] | None = None, mask_info: MaskInfo | None = None, logits_soft_cap: float | None = None, softmax_scale: float | None = None, sliding_window: int | tuple[int, int] | None = None, causal: bool = True, fused_backward: bool = False, **ignore)` — [`L132`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L132) — Computes attention using the scan-based `blockwise_attn` function.
  - `forward_rocm(self, *args, **kwargs)` — [`L305`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L305) — ROCm GPU forward pass. Currently delegates to `forward_native` (scan-based).
  - `forward_tpu(self, *args, **kwargs)` — [`L276`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L276)
  - `get_impl_name(cls)` — [`L106`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L106) — Get the registered name for this attention implementation.
  - `get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED)` — [`L115`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L115) — Returns requirements for RingAttn.
- uses (calls/refs, reference-scoped): [`TransformerCacheView`](../../caching/transformer/cache.md#TransformerCacheView), [`AttentionOutput`](../_attention_outputs.md#AttentionOutput), [`MetadataField`](../requirements/types.md#MetadataField), [`CacheType`](../requirements/types.md#CacheType), [`metadata`](../_base_operation.md#BaseOperation.metadata), [`ExecutionMode`](../requirements/types.md#ExecutionMode), [`OperationRequirements`](../requirements/requirements.md#OperationRequirements), [`OperationImpl`](../_operation_impl.md#OperationImpl), [`get_shardings`](../_operation_meta.md#OperationMetadata.get_shardings), [`create_stable_sharding`](../_operation_impl.md#OperationImpl.create_stable_sharding), [`mesh`](../_operation_meta.md#OperationMetadata.mesh), [`create`](../requirements/requirements.md#OperationRequirements.create), [`register`](../_base_operation.md#OperationRegistry.register), [`MIXED`](../requirements/types.md#ExecutionMode.MIXED), [`VanillaAttn`](vanilla_attention.md#VanillaAttn), [`basic`](../requirements/types.md#MetadataField.basic), [`query`](../_operation_meta.md#AttnShardingRules.query), [`TRANSFORMER`](../requirements/types.md#CacheType.TRANSFORMER), [`get_operation_config`](../_operation_meta.md#OperationMetadata.get_operation_config), [`get_mode`](../_operation_impl.md#OperationImpl.get_mode), [`HYBRID`](../requirements/types.md#CacheType.HYBRID), [`runtime_dtype`](../_operation_meta.md#OperationMetadata.runtime_dtype), [`output`](../_operation_meta.md#AttnShardingRules.output), [`key`](../_operation_meta.md#AttnShardingRules.key), [`value`](../_operation_meta.md#AttnShardingRules.value), [`softmax_aux`](../_operation_meta.md#AttnShardingRules.softmax_aux), [`sequence_axis_name`](../_operation_meta.md#OperationMetadata.sequence_axis_name)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`OperationImpl`](../_operation_impl.md#OperationImpl), [`get_requirements`](../_base_operation.md#BaseOperation.get_requirements), [`forward_native`](../_base_operation.md#BaseOperation.forward_native), [`get_impl_name`](../_base_operation.md#BaseOperation.get_impl_name), [`forward_gpu`](../_base_operation.md#BaseOperation.forward_gpu), [`forward_tpu`](../_base_operation.md#BaseOperation.forward_tpu), [`forward_cpu`](../_base_operation.md#BaseOperation.forward_cpu), [`forward_cuda`](../_base_operation.md#BaseOperation.forward_cuda), [`forward_rocm`](../_base_operation.md#BaseOperation.forward_rocm), [`ring`](ring_attention.md#ring)

## Module values
- `b` — [`L371`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L371)
- `d` — [`L371`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L371)
- `k` — [`L373`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L373)
- `kh` — [`L371`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L371)
- `ks` — [`L371`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L371)
- `mask_info` — [`L375`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L375)
- `out` — [`L384`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L384)
- `q` — [`L372`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L372)
- `qh` — [`L371`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L371)
- `qs` — [`L371`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L371)
- `ring` — [`L376`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L376)
- `v` — [`L374`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L374)
- `vd` — [`L371`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L371)
- `vout` — [`L385`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/ring_attention.py#L385)

