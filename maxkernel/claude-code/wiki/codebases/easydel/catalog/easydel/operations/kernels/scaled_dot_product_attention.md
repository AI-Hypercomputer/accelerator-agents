---
title: 'Module: easydel/operations/kernels/scaled_dot_product_attention.py'
type: catalog
provenance: extracted
module: easydel/operations/kernels/scaled_dot_product_attention.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.operations.kernels.scaled_dot_product_attention`/
symbols:
  ScaledDotProductAttn.forward_native: ScaledDotProductAttn#forward_native().
  ScaledDotProductAttn.get_requirements: ScaledDotProductAttn#get_requirements().
  tout: tout.
  ScaledDotProductAttn: ScaledDotProductAttn#
  gout: gout.
  cout: cout.
  query: query.
  key: key.
  value: value.
  mask_info: mask_info.
  gpu_attn: gpu_attn.
  cpu_attn: cpu_attn.
  tpu_attn: tpu_attn.
  ScaledDotProductAttn.forward_cuda: ScaledDotProductAttn#forward_cuda().
  ScaledDotProductAttn.forward_gpu: ScaledDotProductAttn#forward_gpu().
  ScaledDotProductAttn.forward_tpu: ScaledDotProductAttn#forward_tpu().
  ScaledDotProductAttn.forward_cpu: ScaledDotProductAttn#forward_cpu().
  ScaledDotProductAttn.forward_rocm: ScaledDotProductAttn#forward_rocm().
  ScaledDotProductAttn.__call__: ScaledDotProductAttn#__call__().
  b: b.
  ks: ks.
  ScaledDotProductAttn.get_unsupported_fallback_features: ScaledDotProductAttn#get_unsupported_fallback_features().
  qs: qs.
  kh: kh.
  d: d.
  qh: qh.
  vd: vd.
  ScaledDotProductAttn.get_impl_name: ScaledDotProductAttn#get_impl_name().
---
# Module: [`easydel/operations/kernels/scaled_dot_product_attention.py`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py)

## Classes
### `ScaledDotProductAttn`  ·  implements/extends OperationImpl
- def: [`easydel/operations/kernels/scaled_dot_product_attention.py:92`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L92)
- doc: An attention implementation that leverages `jax.nn.dot_product_attention`.
- signature: `class ScaledDotProductAttn(OperationImpl):`
- members:
  - `__call__(self, query: Float[Array, "batch seq_len num_q_heads head_dim"], key: Float[Array, "batch kv_len num_kv_heads head_dim"], value: Float[Array, "batch kv_len num_kv_heads head_dim"], mask_info: MaskInfo | None = None, bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None, init_bias: tp.Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None, softmax_scale: float | None = None, causal: bool = False, sliding_window: int | tuple[int, int] | None = None, cum_seqlens_q: Int[Array, batch] | None = None, cum_seqlens_k: Int[Array, batch] | None = None, **ignore)` — [`L318`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L318) — Executes the Scaled Dot Product Attention computation using the appropriate backend.
  - `forward_cpu(self, *args, **kwargs)` — [`L280`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L280) — CPU forward pass. Delegates to `forward_native` (XLA implementation).
  - `forward_cuda(self, *args, **kwargs)` — [`L292`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L292) — CPU forward pass. Delegates to `forward_native` (AUTO-DETECT implementation).
  - `forward_gpu(self, *args, **kwargs)` — [`L256`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L256) — GPU forward pass. Delegates to the CUDA-specific implementation.
  - `forward_native(self, query: Float[Array, "batch seq_len num_q_heads head_dim"], key: Float[Array, "batch kv_len num_kv_heads head_dim"], value: Float[Array, "batch kv_len num_kv_heads head_dim"], mask_info: MaskInfo | None = None, bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None, init_bias: tp.Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None, softmax_scale: float | None = None, causal: bool = False, sliding_window: int | tuple[int, int] | None = None, cum_seqlens_q: Int[Array, batch] | None = None, cum_seqlens_k: Int[Array, batch] | None = None, **ignore)` — [`L151`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L151) — Computes attention using `jax.nn.dot_product_attention` with the "xla" implementation.
  - `forward_rocm(self, *args, **kwargs)` — [`L304`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L304) — ROCm GPU forward pass. Currently delegates to `forward_native`.
  - `forward_tpu(self, *args, **kwargs)` — [`L268`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L268) — TPU forward pass. Delegates to `forward_native` (XLA implementation).
  - `get_impl_name(cls)` — [`L107`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L107) — Returns the registered name(s) for this implementation.
  - `get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED)` — [`L117`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L117) — Returns requirements for ScaledDotProductAttn.
  - `get_unsupported_fallback_features(*, softmax_aux: tp.Any = None, logits_soft_cap: float | None = None, dropout_prob: float = 0, normalize_output: bool = True)` — [`L131`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L131) — Returns attention features that the SDPA fallback cannot preserve.
- uses (calls/refs, reference-scoped): [`TransformerCacheView`](../../caching/transformer/cache.md#TransformerCacheView), [`AttentionOutput`](../_attention_outputs.md#AttentionOutput), [`MetadataField`](../requirements/types.md#MetadataField), [`CacheType`](../requirements/types.md#CacheType), [`metadata`](../_base_operation.md#BaseOperation.metadata), [`ExecutionMode`](../requirements/types.md#ExecutionMode), [`OperationRequirements`](../requirements/requirements.md#OperationRequirements), [`OperationImpl`](../_operation_impl.md#OperationImpl), [`get_shardings`](../_operation_meta.md#OperationMetadata.get_shardings), [`create_stable_sharding`](../_operation_impl.md#OperationImpl.create_stable_sharding), [`__call__`](../_base_operation.md#BaseOperation.__call__), [`mesh`](../_operation_meta.md#OperationMetadata.mesh), [`create`](../requirements/requirements.md#OperationRequirements.create), [`register`](../_base_operation.md#OperationRegistry.register), [`MIXED`](../requirements/types.md#ExecutionMode.MIXED), [`basic`](../requirements/types.md#MetadataField.basic), [`query`](../_operation_meta.md#AttnShardingRules.query), [`TRANSFORMER`](../requirements/types.md#CacheType.TRANSFORMER), [`get_mode`](../_operation_impl.md#OperationImpl.get_mode), [`HYBRID`](../requirements/types.md#CacheType.HYBRID), [`runtime_dtype`](../_operation_meta.md#OperationMetadata.runtime_dtype), [`output`](../_operation_meta.md#AttnShardingRules.output), [`key`](../_operation_meta.md#AttnShardingRules.key), [`value`](../_operation_meta.md#AttnShardingRules.value), [`bias`](../_operation_meta.md#AttnShardingRules.bias)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`OperationImpl`](../_operation_impl.md#OperationImpl), [`get_requirements`](../_base_operation.md#BaseOperation.get_requirements), [`forward_native`](../_base_operation.md#BaseOperation.forward_native), [`get_impl_name`](../_base_operation.md#BaseOperation.get_impl_name), [`forward_gpu`](../_base_operation.md#BaseOperation.forward_gpu), [`forward_tpu`](../_base_operation.md#BaseOperation.forward_tpu), [`forward_cpu`](../_base_operation.md#BaseOperation.forward_cpu), [`forward_cuda`](../_base_operation.md#BaseOperation.forward_cuda), [`forward_rocm`](../_base_operation.md#BaseOperation.forward_rocm), [`_maybe_route_varlen_multihost_tpu_attention`](../../layers/attention/_flexible.md#FlexibleAttentionModule._maybe_route_varlen_multihost_tpu_attention), [`_fallback_attention`](flash_attention.md#FlashAttn._fallback_attention), [`cpu_attn`](scaled_dot_product_attention.md#cpu_attn), [`gpu_attn`](scaled_dot_product_attention.md#gpu_attn), [`tpu_attn`](scaled_dot_product_attention.md#tpu_attn)

## Module values
- `b` — [`L372`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L372)
- `cout` — [`L388`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L388)
- `cpu_attn` — [`L381`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L381)
- `d` — [`L372`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L372)
- `gout` — [`L389`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L389)
- `gpu_attn` — [`L378`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L378)
- `key` — [`L374`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L374)
- `kh` — [`L372`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L372)
- `ks` — [`L372`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L372)
- `mask_info` — [`L376`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L376)
- `qh` — [`L372`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L372)
- `qs` — [`L372`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L372)
- `query` — [`L373`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L373)
- `tout` — [`L390`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L390)
- `tpu_attn` — [`L384`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L384)
- `value` — [`L375`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L375)
- `vd` — [`L372`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/scaled_dot_product_attention.py#L372)

