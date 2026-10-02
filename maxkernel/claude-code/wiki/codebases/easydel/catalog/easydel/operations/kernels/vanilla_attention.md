---
title: 'Module: easydel/operations/kernels/vanilla_attention.py'
type: catalog
provenance: extracted
module: easydel/operations/kernels/vanilla_attention.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.operations.kernels.vanilla_attention`/
symbols:
  VanillaAttn.forward_native: VanillaAttn#forward_native().
  VanillaAttn.get_requirements: VanillaAttn#get_requirements().
  VanillaAttn: VanillaAttn#
  out: out.
  query: query.
  key: key.
  value: value.
  mask_info: mask_info.
  VanillaAttn.forward_cuda: VanillaAttn#forward_cuda().
  metadata: metadata.
  VanillaAttn.forward_gpu: VanillaAttn#forward_gpu().
  VanillaAttn.forward_tpu: VanillaAttn#forward_tpu().
  VanillaAttn.forward_cpu: VanillaAttn#forward_cpu().
  VanillaAttn.forward_rocm: VanillaAttn#forward_rocm().
  VanillaAttn.__call__: VanillaAttn#__call__().
  b: b.
  ks: ks.
  qs: qs.
  kh: kh.
  d: d.
  qh: qh.
  vd: vd.
  VanillaAttn.get_impl_name: VanillaAttn#get_impl_name().
---
# Module: [`easydel/operations/kernels/vanilla_attention.py`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py)

## Classes
### `VanillaAttn`  ·  implements/extends OperationImpl
- def: [`easydel/operations/kernels/vanilla_attention.py:78`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py#L78)
- doc: A standard, non-optimized implementation of multi-head attention.
- signature: `class VanillaAttn(OperationImpl):`
- members:
  - `__call__(self, query: Float[Array, "batch seq_len num_q_heads head_dim"], key: Float[Array, "batch kv_len num_kv_heads head_dim"], value: Float[Array, "batch kv_len num_kv_heads head_dim"], mask_info: MaskInfo | None = None, bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None, init_bias: tp.Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None, deterministic: bool = True, dropout_rng: PRNGKeyArray | None = None, softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, num_sinks] | None = None, softmax_scale: float | None = None, logits_soft_cap: float | None = None, dropout_prob: float = 0, causal: bool = False, sliding_window: int | tuple[int, int] | None = None, **ignore)` — [`L285`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py#L285) — Executes the vanilla attention computation.
  - `forward_cpu(self, *args, **kwargs)` — [`L249`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py#L249) — CPU forward pass. Delegates to `forward_native`.
  - `forward_cuda(self, *args, **kwargs)` — [`L261`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py#L261) — CUDA GPU forward pass. Delegates to `forward_native`.
  - `forward_gpu(self, *args, **kwargs)` — [`L225`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py#L225) — GPU forward pass. Delegates to `forward_native`.
  - `forward_native(self, query: Float[Array, "batch seq_len num_q_heads head_dim"], key: Float[Array, "batch kv_len num_kv_heads head_dim"], value: Float[Array, "batch kv_len num_kv_heads head_dim"], mask_info: MaskInfo | None = None, bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None, init_bias: tp.Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None, deterministic: bool = True, dropout_rng: PRNGKeyArray | None = None, softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, num_sinks] | None = None, softmax_scale: float | None = None, logits_soft_cap: float | None = None, dropout_prob: float = 0, causal: bool = False, sliding_window: int | tuple[int, int] | None = None, **ignore)` — [`L119`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py#L119) — Standard multi-head attention implementation using basic JAX operations.
  - `forward_rocm(self, *args, **kwargs)` — [`L273`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py#L273) — ROCm GPU forward pass. Delegates to `forward_native`.
  - `forward_tpu(self, *args, **kwargs)` — [`L237`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py#L237) — TPU forward pass. Delegates to `forward_native`.
  - `get_impl_name(cls)` — [`L92`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py#L92) — Returns the registered name of this attention implementation.
  - `get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED)` — [`L102`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py#L102) — Returns requirements for VanillaAttn.
- uses (calls/refs, reference-scoped): [`TransformerCacheView`](../../caching/transformer/cache.md#TransformerCacheView), [`AttentionOutput`](../_attention_outputs.md#AttentionOutput), [`MetadataField`](../requirements/types.md#MetadataField), [`CacheType`](../requirements/types.md#CacheType), [`metadata`](../_base_operation.md#BaseOperation.metadata), [`ExecutionMode`](../requirements/types.md#ExecutionMode), [`OperationRequirements`](../requirements/requirements.md#OperationRequirements), [`OperationImpl`](../_operation_impl.md#OperationImpl), [`get_shardings`](../_operation_meta.md#OperationMetadata.get_shardings), [`__call__`](../_base_operation.md#BaseOperation.__call__), [`mesh`](../_operation_meta.md#OperationMetadata.mesh), [`create`](../requirements/requirements.md#OperationRequirements.create), [`register`](../_base_operation.md#OperationRegistry.register), [`MIXED`](../requirements/types.md#ExecutionMode.MIXED), [`basic`](../requirements/types.md#MetadataField.basic), [`query`](../_operation_meta.md#AttnShardingRules.query), [`TRANSFORMER`](../requirements/types.md#CacheType.TRANSFORMER), [`get_mode`](../_operation_impl.md#OperationImpl.get_mode), [`HYBRID`](../requirements/types.md#CacheType.HYBRID), [`runtime_dtype`](../_operation_meta.md#OperationMetadata.runtime_dtype), [`output`](../_operation_meta.md#AttnShardingRules.output), [`key`](../_operation_meta.md#AttnShardingRules.key), [`value`](../_operation_meta.md#AttnShardingRules.value), [`bias`](../_operation_meta.md#AttnShardingRules.bias), [`runtime_softmax_dtype`](../_operation_meta.md#OperationMetadata.runtime_softmax_dtype)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`OperationImpl`](../_operation_impl.md#OperationImpl), [`get_requirements`](../_base_operation.md#BaseOperation.get_requirements), [`forward_native`](../_base_operation.md#BaseOperation.forward_native), [`get_impl_name`](../_base_operation.md#BaseOperation.get_impl_name), [`forward_gpu`](../_base_operation.md#BaseOperation.forward_gpu), [`forward_tpu`](../_base_operation.md#BaseOperation.forward_tpu), [`forward_cpu`](../_base_operation.md#BaseOperation.forward_cpu), [`forward_native`](decode_attention.md#AutoRegressiveDecodeAttn.forward_native), [`forward_native`](ring_attention.md#RingAttn.forward_native), [`forward_cuda`](../_base_operation.md#BaseOperation.forward_cuda), [`forward_rocm`](../_base_operation.md#BaseOperation.forward_rocm), [`out`](vanilla_attention.md#out), [`_fallback_attention`](flash_attention.md#FlashAttn._fallback_attention), [`_run_vanilla_fallback`](blocksparse_attention.md#BlockSparseAttn._run_vanilla_fallback), [`vanilla`](flash_attention.md#vanilla), [`vanilla_attn`](blocksparse_attention.md#vanilla_attn)

## Module values
- `b` — [`L347`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py#L347)
- `d` — [`L347`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py#L347)
- `key` — [`L349`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py#L349)
- `kh` — [`L347`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py#L347)
- `ks` — [`L347`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py#L347)
- `mask_info` — [`L351`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py#L351)
- `metadata` — [`L353`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py#L353)
- `out` — [`L359`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py#L359)
- `qh` — [`L347`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py#L347)
- `qs` — [`L347`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py#L347)
- `query` — [`L348`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py#L348)
- `value` — [`L350`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py#L350)
- `vd` — [`L347`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/vanilla_attention.py#L347)

