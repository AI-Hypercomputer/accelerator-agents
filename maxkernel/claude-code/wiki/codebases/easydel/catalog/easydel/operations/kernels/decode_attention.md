---
title: 'Module: easydel/operations/kernels/decode_attention.py'
type: catalog
provenance: extracted
module: easydel/operations/kernels/decode_attention.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.operations.kernels.decode_attention`/
symbols:
  AutoRegressiveDecodeAttn.forward_native: AutoRegressiveDecodeAttn#forward_native().
  AutoRegressiveDecodeAttn.get_requirements: AutoRegressiveDecodeAttn#get_requirements().
  AutoRegressiveDecodeAttn: AutoRegressiveDecodeAttn#
  AutoRegressiveDecodeAttn.__call__: AutoRegressiveDecodeAttn#__call__().
  AutoRegressiveDecodeAttn.forward_gpu: AutoRegressiveDecodeAttn#forward_gpu().
  _slice_decode_window_for_vanilla_fallback: _slice_decode_window_for_vanilla_fallback().
  AutoRegressiveDecodeAttn.forward_tpu: AutoRegressiveDecodeAttn#forward_tpu().
  AutoRegressiveDecodeAttn.forward_cpu: AutoRegressiveDecodeAttn#forward_cpu().
  AutoRegressiveDecodeAttn.forward_cuda: AutoRegressiveDecodeAttn#forward_cuda().
  AutoRegressiveDecodeAttn.forward_rocm: AutoRegressiveDecodeAttn#forward_rocm().
  AutoRegressiveDecodeAttn.get_impl_name: AutoRegressiveDecodeAttn#get_impl_name().
---
# Module: [`easydel/operations/kernels/decode_attention.py`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/decode_attention.py)

## Classes
### `AutoRegressiveDecodeAttn`  ·  implements/extends OperationImpl
- def: [`easydel/operations/kernels/decode_attention.py:176`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/decode_attention.py#L176)
- doc: Attention implementation tailored for the autoregressive decoding step.
- signature: `class AutoRegressiveDecodeAttn(OperationImpl):`
- members:
  - `__call__(self, query: Float[Array, "batch 1 num_q_heads head_dim"], key: Float[Array, "batch kv_seq_len num_kv_heads head_dim"], value: Float[Array, "batch kv_seq_len num_kv_heads head_dim"], cache_metadata: TransformerMetadata, softmax_scale: float | None = None, sliding_window: tuple[int, int] | None = None, logits_soft_cap: float | None = None, softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, num_sinks] | None = None, **ignores)` — [`L455`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/decode_attention.py#L455) — Executes autoregressive decode attention by dispatching to the appropriate backend.
  - `forward_cpu(self, *args, **kwargs)` — [`L408`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/decode_attention.py#L408) — CPU forward pass for autoregressive decoding attention.
  - `forward_cuda(self, *args, **kwargs)` — [`L423`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/decode_attention.py#L423) — CUDA GPU forward pass for autoregressive decoding attention.
  - `forward_gpu(self, *args, **kwargs)` — [`L378`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/decode_attention.py#L378) — CPU forward pass for autoregressive decoding attention.
  - `forward_native(self, query: Float[Array, "batch 1 num_q_heads head_dim"], key: Float[Array, "batch kv_seq_len num_kv_heads head_dim"], value: Float[Array, "batch kv_seq_len num_kv_heads head_dim"], cache_metadata: TransformerMetadata, softmax_scale: float | None = None, sliding_window: tuple[int, int] | None = None, logits_soft_cap: float | None = None, softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, num_sinks] | None = None, **ignores)` — [`L219`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/decode_attention.py#L219) — Performs the native JAX/XLA forward pass for autoregressive decoding attention.
  - `forward_rocm(self, *args, **kwargs)` — [`L439`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/decode_attention.py#L439) — ROCm GPU forward pass for autoregressive decoding attention.
  - `forward_tpu(self, *args, **kwargs)` — [`L393`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/decode_attention.py#L393) — CPU forward pass for autoregressive decoding attention.
  - `get_impl_name(cls)` — [`L191`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/decode_attention.py#L191) — Returns the registered name of this attention implementation.
  - `get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED)` — [`L201`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/decode_attention.py#L201) — Returns requirements for AutoRegressiveDecodeAttn.
- uses (calls/refs, reference-scoped): [`TransformerMetadata`](../../caching/transformer/cache.md#TransformerMetadata), [`TransformerCacheView`](../../caching/transformer/cache.md#TransformerCacheView), [`AttentionOutput`](../_attention_outputs.md#AttentionOutput), [`MetadataField`](../requirements/types.md#MetadataField), [`CacheType`](../requirements/types.md#CacheType), [`metadata`](../_base_operation.md#BaseOperation.metadata), [`ExecutionMode`](../requirements/types.md#ExecutionMode), [`OperationRequirements`](../requirements/requirements.md#OperationRequirements), [`OperationImpl`](../_operation_impl.md#OperationImpl), [`get_shardings`](../_operation_meta.md#OperationMetadata.get_shardings), [`create_stable_sharding`](../_operation_impl.md#OperationImpl.create_stable_sharding), [`__call__`](../_base_operation.md#BaseOperation.__call__), [`mesh`](../_operation_meta.md#OperationMetadata.mesh), [`create`](../requirements/requirements.md#OperationRequirements.create), [`register`](../_base_operation.md#OperationRegistry.register), [`MIXED`](../requirements/types.md#ExecutionMode.MIXED), [`VanillaAttn`](vanilla_attention.md#VanillaAttn), [`basic`](../requirements/types.md#MetadataField.basic), [`query`](../_operation_meta.md#AttnShardingRules.query), [`TRANSFORMER`](../requirements/types.md#CacheType.TRANSFORMER), [`get_mode`](../_operation_impl.md#OperationImpl.get_mode), [`HYBRID`](../requirements/types.md#CacheType.HYBRID), [`key`](../_operation_meta.md#AttnShardingRules.key), [`value`](../_operation_meta.md#AttnShardingRules.value), [`CONTEXT_LENS`](../requirements/types.md#MetadataField.CONTEXT_LENS), [`_slice_decode_window_for_vanilla_fallback`](decode_attention.md#_slice_decode_window_for_vanilla_fallback), [`softmax_aux`](../_operation_meta.md#AttnShardingRules.softmax_aux), [`indexs`](../../caching/transformer/cache.md#TransformerMetadata.indexs), [`query3d`](../_operation_meta.md#AttnShardingRules.query3d), [`starts`](../../caching/transformer/cache.md#TransformerMetadata.starts)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`OperationImpl`](../_operation_impl.md#OperationImpl), [`get_requirements`](../_base_operation.md#BaseOperation.get_requirements), [`forward_native`](../_base_operation.md#BaseOperation.forward_native), [`get_impl_name`](../_base_operation.md#BaseOperation.get_impl_name), [`forward_gpu`](../_base_operation.md#BaseOperation.forward_gpu), [`forward_tpu`](../_base_operation.md#BaseOperation.forward_tpu), [`forward_cpu`](../_base_operation.md#BaseOperation.forward_cpu), [`forward_cuda`](../_base_operation.md#BaseOperation.forward_cuda), [`forward_rocm`](../_base_operation.md#BaseOperation.forward_rocm)

## Functions
- `_slice_decode_window_for_vanilla_fallback(key: Float[Array, "batch kv_seq_len num_kv_heads head_dim"], value: Float[Array, "batch kv_seq_len num_kv_heads head_dim"], mask_info: MaskInfo | None, cache_metadata: TransformerMetadata, sliding_window: int | tuple[int, int] | None)` — [`L82`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/decode_attention.py#L82) — Bake the active decode KV window into TPU/CPU vanilla-attention fallback inputs.

