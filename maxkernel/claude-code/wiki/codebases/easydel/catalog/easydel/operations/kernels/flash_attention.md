---
title: 'Module: easydel/operations/kernels/flash_attention.py'
type: catalog
provenance: extracted
module: easydel/operations/kernels/flash_attention.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.operations.kernels.flash_attention`/
symbols:
  FlashAttn.forward_native: FlashAttn#forward_native().
  FlashAttn.get_requirements: FlashAttn#get_requirements().
  vout: vout.
  fout: fout.
  FlashAttn._fallback_attention: FlashAttn#_fallback_attention().
  query: query.
  key: key.
  value: value.
  a: a.
  FlashAttn.forward_gpu: FlashAttn#forward_gpu().
  metadata: metadata.
  FlashAttn: FlashAttn#
  attn: attn.
  vanilla: vanilla.
  FlashAttn.forward_cuda: FlashAttn#forward_cuda().
  FlashAttn.forward_tpu: FlashAttn#forward_tpu().
  FlashAttn.forward_cpu: FlashAttn#forward_cpu().
  FlashAttn.forward_rocm: FlashAttn#forward_rocm().
  FlashAttn.__call__: FlashAttn#__call__().
  b: b.
  logger: logger.
  ks: ks.
  qs: qs.
  kh: kh.
  d: d.
  qh: qh.
  vd: vd.
  FlashAttn.get_impl_name: FlashAttn#get_impl_name().
---
# Module: [`easydel/operations/kernels/flash_attention.py`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py)

## Classes
### `FlashAttn`  ·  implements/extends OperationImpl
- def: [`easydel/operations/kernels/flash_attention.py:83`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L83)
- doc: An implementation of Flash Attention V2 using specialized JAX primitives.
- signature: `class FlashAttn(OperationImpl):`
- members:
  - `__call__(self, query: Float[Array, "batch seq_len_q num_heads head_dim"], key: Float[Array, "batch seq_len_k num_kv_heads head_dim"], value: Float[Array, "batch seq_len_k num_kv_heads head_dim"], mask_info: MaskInfo | None = None, bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None, softmax_scale: float | None = None, dropout_prob: float = 0, causal: bool = False, dropout_seed: int | None = None, cum_seqlens_q: Int[Array, batch_plus_one] | None = None, cum_seqlens_k: Int[Array, batch_plus_one] | None = None, sliding_window: int | tuple[int, int] | None = None, logits_soft_cap: float | None = None, softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, num_sinks] | None = None, normalize_output: bool = True, precision: lax.PrecisionLike = lax.Precision.DEFAULT, **ignore)` — [`L393`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L393) — Executes Flash Attention V2 by dispatching to the appropriate backend implementation.
  - `forward_cpu(self, *args, **kwargs)` — [`L353`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L353) — CPU forward pass. Delegates to `forward_native`, which raises an error.
  - `forward_cuda(self, *args, **kwargs)` — [`L326`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L326) — GPU forward pass. Delegates to the CUDA-specific implementation.
  - `forward_gpu(self, *args, **kwargs)` — [`L363`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L363) — GPU forward pass. Delegates to the CUDA-specific implementation.
  - `forward_native(self, query: Float[Array, "batch seq_len_q num_heads head_dim"], key: Float[Array, "batch seq_len_k num_kv_heads head_dim"], value: Float[Array, "batch seq_len_k num_kv_heads head_dim"], mask_info: MaskInfo | None = None, bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None, softmax_scale: float | None = None, dropout_prob: float = 0, causal: bool = False, dropout_seed: int | None = None, cum_seqlens_q: Int[Array, batch_plus_one] | None = None, cum_seqlens_k: Int[Array, batch_plus_one] | None = None, sliding_window: int | tuple[int, int] | None = None, logits_soft_cap: float | None = None, softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, num_sinks] | None = None, normalize_output: bool = True, precision: lax.PrecisionLike = lax.Precision.DEFAULT, **ignore)` — [`L119`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L119) — Performs Flash Attention V2 using optimized kernels (TPU Pallas or GPU Triton).
  - `forward_rocm(self, *args, **kwargs)` — [`L376`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L376) — ROCm GPU forward pass.
  - `forward_tpu(self, *args, **kwargs)` — [`L340`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L340) — GPU forward pass. Delegates to the CUDA-specific implementation.
  - `get_impl_name(cls)` — [`L93`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L93) — Returns the registered name of this attention implementation.
  - `get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED)` — [`L103`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L103) — Returns requirements for FlashAttn.
- protocol/private: `_fallback_attention`[`L181`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L181)
- uses (calls/refs, reference-scoped): [`TransformerCacheView`](../../caching/transformer/cache.md#TransformerCacheView), [`AttentionOutput`](../_attention_outputs.md#AttentionOutput), [`MetadataField`](../requirements/types.md#MetadataField), [`CacheType`](../requirements/types.md#CacheType), [`metadata`](../_base_operation.md#BaseOperation.metadata), [`ExecutionMode`](../requirements/types.md#ExecutionMode), [`OperationRequirements`](../requirements/requirements.md#OperationRequirements), [`OperationImpl`](../_operation_impl.md#OperationImpl), [`get_shardings`](../_operation_meta.md#OperationMetadata.get_shardings), [`create_stable_sharding`](../_operation_impl.md#OperationImpl.create_stable_sharding), [`__call__`](../_base_operation.md#BaseOperation.__call__), [`mesh`](../_operation_meta.md#OperationMetadata.mesh), [`create`](../requirements/requirements.md#OperationRequirements.create), [`MIXED`](../requirements/types.md#ExecutionMode.MIXED), [`VanillaAttn`](vanilla_attention.md#VanillaAttn), [`basic`](../requirements/types.md#MetadataField.basic), [`query`](../_operation_meta.md#AttnShardingRules.query), [`TRANSFORMER`](../requirements/types.md#CacheType.TRANSFORMER), [`ScaledDotProductAttn`](scaled_dot_product_attention.md#ScaledDotProductAttn), [`get_operation_config`](../_operation_meta.md#OperationMetadata.get_operation_config), [`get_mode`](../_operation_impl.md#OperationImpl.get_mode), [`HYBRID`](../requirements/types.md#CacheType.HYBRID), [`runtime_dtype`](../_operation_meta.md#OperationMetadata.runtime_dtype), [`output`](../_operation_meta.md#AttnShardingRules.output), [`key`](../_operation_meta.md#AttnShardingRules.key), [`value`](../_operation_meta.md#AttnShardingRules.value), [`softmax_aux`](../_operation_meta.md#AttnShardingRules.softmax_aux), [`bias`](../_operation_meta.md#AttnShardingRules.bias), [`logger`](flash_attention.md#logger), [`get_unsupported_fallback_features`](scaled_dot_product_attention.md#ScaledDotProductAttn.get_unsupported_fallback_features)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`OperationImpl`](../_operation_impl.md#OperationImpl), [`get_requirements`](../_base_operation.md#BaseOperation.get_requirements), [`forward_native`](../_base_operation.md#BaseOperation.forward_native), [`get_impl_name`](../_base_operation.md#BaseOperation.get_impl_name), [`forward_gpu`](../_base_operation.md#BaseOperation.forward_gpu), [`forward_tpu`](../_base_operation.md#BaseOperation.forward_tpu), [`forward_cpu`](../_base_operation.md#BaseOperation.forward_cpu), [`forward_cuda`](../_base_operation.md#BaseOperation.forward_cuda), [`forward_rocm`](../_base_operation.md#BaseOperation.forward_rocm), [`attn`](flash_attention.md#attn)

## Module values
- `a` — [`L471`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L471)
- `attn` — [`L476`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L476)
- `b` — [`L467`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L467)
- `d` — [`L467`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L467)
- `fout` — [`L478`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L478)
- `key` — [`L469`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L469)
- `kh` — [`L467`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L467)
- `ks` — [`L467`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L467)
- `logger` — [`L79`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L79)
- `metadata` — [`L472`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L472)
- `qh` — [`L467`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L467)
- `qs` — [`L467`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L467)
- `query` — [`L468`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L468)
- `value` — [`L470`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L470)
- `vanilla` — [`L477`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L477)
- `vd` — [`L467`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L467)
- `vout` — [`L479`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/flash_attention.py#L479)

