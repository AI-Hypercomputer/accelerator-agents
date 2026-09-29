---
title: 'Module: easydel/operations/kernels/blocksparse_attention.py'
type: catalog
provenance: extracted
module: easydel/operations/kernels/blocksparse_attention.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.operations.kernels.blocksparse_attention`/
symbols:
  BlockSparseAttn.forward_native: BlockSparseAttn#forward_native().
  BlockSparseAttn.get_requirements: BlockSparseAttn#get_requirements().
  vd: vd.
  splash_out: splash_out.
  vanilla_out: vanilla_out.
  query: query.
  key: key.
  value: value.
  max_diff: max_diff.
  BlockSparseAttn._run_vanilla_fallback: BlockSparseAttn#_run_vanilla_fallback().
  mask_info: mask_info.
  BlockSparseAttn.__call__: BlockSparseAttn#__call__().
  metadata: metadata.
  BlockSparseAttn: BlockSparseAttn#
  splash_attn: splash_attn.
  vanilla_attn: vanilla_attn.
  is_close: is_close.
  kh: kh.
  BlockSparseAttn.forward_gpu: BlockSparseAttn#forward_gpu().
  BlockSparseAttn.forward_tpu: BlockSparseAttn#forward_tpu().
  BlockSparseAttn.forward_cpu: BlockSparseAttn#forward_cpu().
  BlockSparseAttn.forward_cuda: BlockSparseAttn#forward_cuda().
  BlockSparseAttn.forward_rocm: BlockSparseAttn#forward_rocm().
  b: b.
  logger: logger.
  BlockSparseAttn._extract_block_size: BlockSparseAttn#_extract_block_size().
  ks: ks.
  qs: qs.
  d: d.
  test_cases: test_cases.
  idx: idx.
  qh: qh.
  key_q: key_q.
  key_k: key_k.
  key_v: key_v.
  BlockSparseAttn.get_impl_name: BlockSparseAttn#get_impl_name().
---
# Module: [`easydel/operations/kernels/blocksparse_attention.py`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py)

## Classes
### `BlockSparseAttn`  ·  implements/extends OperationImpl
- def: [`easydel/operations/kernels/blocksparse_attention.py:96`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L96)
- doc: An attention implementation using the Pallas Splash Attention kernel for TPUs.
- signature: `class BlockSparseAttn(OperationImpl):`
- members:
  - `__call__(self, query: Float[Array, "batch num_heads seq_len head_dim"], key: Float[Array, "batch kv_num_heads kv_len head_dim"], value: Float[Array, "batch kv_num_heads kv_len vhead_dim"], softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, num_sinks] | None = None, mask_info: MaskInfo | None = None, logits_soft_cap: float | None = None, softmax_scale: float | None = None, sliding_window: int | tuple[int, int] | None = None, causal: bool = True, fused_backward: bool = False, cache_metadata: TransformerMetadata | None = None, **ignore)` — [`L440`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L440) — Executes the Splash Attention computation or falls back to Vanilla Attention.
  - `forward_cpu(self, *args, **kwargs)` — [`L398`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L398) — GPU forward pass. Not implemented for Splash Attention.
  - `forward_cuda(self, *args, **kwargs)` — [`L412`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L412) — GPU forward pass. Not implemented for Splash Attention.
  - `forward_gpu(self, *args, **kwargs)` — [`L370`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L370) — GPU forward pass. Not implemented for Splash Attention.
  - `forward_native(self, query: Float[Array, "batch num_heads seq_len head_dim"], key: Float[Array, "batch kv_num_heads kv_len head_dim"], value: Float[Array, "batch kv_num_heads kv_len vhead_dim"], softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, num_sinks] | None = None, mask_info: MaskInfo | None = None, logits_soft_cap: float | None = None, softmax_scale: float | None = None, sliding_window: int | tuple[int, int] | None = None, causal: bool = True, fused_backward: bool = False, cache_metadata: TransformerMetadata | None = None, **ignore)` — [`L143`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L143) — Performs Splash Attention on TPU/GPU using the Pallas/Triton kernel.
  - `forward_rocm(self, *args, **kwargs)` — [`L426`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L426) — GPU forward pass. Not implemented for Splash Attention.
  - `forward_tpu(self, *args, **kwargs)` — [`L384`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L384) — GPU forward pass. Not implemented for Splash Attention.
  - `get_impl_name(cls)` — [`L117`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L117) — Returns the registered name of this attention implementation.
  - `get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED)` — [`L127`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L127) — Returns requirements for BlockSparseAttn (Splash Attention).
- protocol/private: `_extract_block_size`[`L196`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L196), `_run_vanilla_fallback`[`L180`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L180)
- uses (calls/refs, reference-scoped): [`TransformerMetadata`](../../caching/transformer/cache.md#TransformerMetadata), [`TransformerCacheView`](../../caching/transformer/cache.md#TransformerCacheView), [`AttentionOutput`](../_attention_outputs.md#AttentionOutput), [`MetadataField`](../requirements/types.md#MetadataField), [`CacheType`](../requirements/types.md#CacheType), [`metadata`](../_base_operation.md#BaseOperation.metadata), [`ExecutionMode`](../requirements/types.md#ExecutionMode), [`OperationRequirements`](../requirements/requirements.md#OperationRequirements), [`OperationImpl`](../_operation_impl.md#OperationImpl), [`get_shardings`](../_operation_meta.md#OperationMetadata.get_shardings), [`create_stable_sharding`](../_operation_impl.md#OperationImpl.create_stable_sharding), [`__call__`](../_base_operation.md#BaseOperation.__call__), [`mesh`](../_operation_meta.md#OperationMetadata.mesh), [`create`](../requirements/requirements.md#OperationRequirements.create), [`MIXED`](../requirements/types.md#ExecutionMode.MIXED), [`VanillaAttn`](vanilla_attention.md#VanillaAttn), [`basic`](../requirements/types.md#MetadataField.basic), [`query`](../_operation_meta.md#AttnShardingRules.query), [`TRANSFORMER`](../requirements/types.md#CacheType.TRANSFORMER), [`get_operation_config`](../_operation_meta.md#OperationMetadata.get_operation_config), [`get_mode`](../_operation_impl.md#OperationImpl.get_mode), [`HYBRID`](../requirements/types.md#CacheType.HYBRID), [`runtime_dtype`](../_operation_meta.md#OperationMetadata.runtime_dtype), [`output`](../_operation_meta.md#AttnShardingRules.output), [`key`](../_operation_meta.md#AttnShardingRules.key), [`value`](../_operation_meta.md#AttnShardingRules.value), [`base_config`](../_operation_meta.md#OperationMetadata.base_config), [`softmax_aux`](../_operation_meta.md#AttnShardingRules.softmax_aux), [`logger`](blocksparse_attention.md#logger)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`OperationImpl`](../_operation_impl.md#OperationImpl), [`get_requirements`](../_base_operation.md#BaseOperation.get_requirements), [`forward_native`](../_base_operation.md#BaseOperation.forward_native), [`get_impl_name`](../_base_operation.md#BaseOperation.get_impl_name), [`forward_gpu`](../_base_operation.md#BaseOperation.forward_gpu), [`forward_tpu`](../_base_operation.md#BaseOperation.forward_tpu), [`forward_cpu`](../_base_operation.md#BaseOperation.forward_cpu), [`forward_cuda`](../_base_operation.md#BaseOperation.forward_cuda), [`forward_rocm`](../_base_operation.md#BaseOperation.forward_rocm), [`splash_attn`](blocksparse_attention.md#splash_attn)

## Module values
- `b` — [`L510`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L510)
- `d` — [`L511`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L511)
- `idx` — [`L510`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L510)
- `is_close` — [`L537`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L537)
- `key` — [`L518`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L518)
- `key_k` — [`L515`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L515)
- `key_q` — [`L515`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L515)
- `key_v` — [`L515`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L515)
- `kh` — [`L510`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L510)
- `ks` — [`L510`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L510)
- `logger` — [`L92`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L92)
- `mask_info` — [`L520`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L520)
- `max_diff` — [`L538`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L538)
- `metadata` — [`L502`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L502)
- `qh` — [`L510`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L510)
- `qs` — [`L510`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L510)
- `query` — [`L517`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L517)
- `splash_attn` — [`L507`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L507)
- `splash_out` — [`L521`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L521)
- `test_cases` — [`L494`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L494)
- `value` — [`L519`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L519)
- `vanilla_attn` — [`L508`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L508)
- `vanilla_out` — [`L529`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L529)
- `vd` — [`L511`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/blocksparse_attention.py#L511)

