---
title: 'Module: ejkernel/modules/operations/prefill_page_attention.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/prefill_page_attention.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.prefill_page_attention`/
symbols:
  _prefill_page_attention_executor._prefill_page_attention_executor: _prefill_page_attention_executor._prefill_page_attention_executor.
  PrefillPageAttention.candidate_cfgs_gpu: PrefillPageAttention#candidate_cfgs_gpu().
  PrefillPageAttention.heuristic_cfg: PrefillPageAttention#heuristic_cfg().
  PrefillPageAttention.run: PrefillPageAttention#run().
  PrefillPageAttention.candidate_cfgs: PrefillPageAttention#candidate_cfgs().
  PrefillPageAttention.candidate_cfgs_tpu: PrefillPageAttention#candidate_cfgs_tpu().
  PrefillPageAttention.get_impl: PrefillPageAttention#get_impl().
  prefill_page_attention: prefill_page_attention().
  PrefillPageAttention: PrefillPageAttention#
  PrefillPageAttention._wrapper: PrefillPageAttention#_wrapper().
  PrefillPageAttention.create_shard_map_wrapper: PrefillPageAttention#create_shard_map_wrapper().
  PrefillPageAttention.__init__: PrefillPageAttention#__init__().
---
# Module: [`ejkernel/modules/operations/prefill_page_attention.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/prefill_page_attention.py)

## Classes
### `PrefillPageAttention`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/prefill_page_attention.py:71`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/prefill_page_attention.py#L71)
- doc: Prefill Page Attention with custom optimization logic.
- signature: `class PrefillPageAttention(Kernel[PrefillPageAttentionConfig, Array]):`
- members:
  - `__init__(self)` — [`L86`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/prefill_page_attention.py#L86) — Initialize Prefill Page Attention module.
  - `_wrapper(query, key_cache, value_cache, context_len, page_indices)` — [`L319`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/prefill_page_attention.py#L319) — Shard-map compatible wrapper that delegates to self.run with captured params.
  - `candidate_cfgs(self, inv: Invocation[PrefillPageAttentionConfig, Array])` — [`L190`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/prefill_page_attention.py#L190) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[PrefillPageAttentionConfig, Array])` — [`L217`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/prefill_page_attention.py#L217) — Generate GPU candidates for TileLang prefill paged attention.
  - `candidate_cfgs_tpu(self, inv: Invocation[PrefillPageAttentionConfig, Array])` — [`L254`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/prefill_page_attention.py#L254) — Generate TPU candidates for Pallas and XLA prefill paged attention.
  - `create_shard_map_wrapper(self, query: Float[Array, "chunk_size num_heads head_dim"], key_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"], value_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"], context_len: Int[Array, 1], page_indices: Int[Array, num_pages], platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, cfg: PrefillPageAttentionConfig | None = None, softmax_scale: float | None = None, mask_value: float = -2.381976426469702e+38, attn_logits_soft_cap: float | None = None, sliding_window: int | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec, ...] | None = None, out_specs: PartitionSpec | None = None, check_vma: bool = False)` — [`L274`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/prefill_page_attention.py#L274) — Create a shard_map wrapper for distributed execution.
  - `get_impl(self, cfg: PrefillPageAttentionConfig)` — [`L90`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/prefill_page_attention.py#L90) — Get kernel implementation from registry.
  - `heuristic_cfg(self, inv: Invocation[PrefillPageAttentionConfig, Array])` — [`L171`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/prefill_page_attention.py#L171) — Provide default configuration optimized for prefill page attention.
  - `run(self, query: Float[Array, "chunk_size num_heads head_dim"], key_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"], value_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"], context_len: Int[Array, 1], page_indices: Int[Array, num_pages], platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, cfg: PrefillPageAttentionConfig, softmax_scale: float | None = None, mask_value: float = -2.381976426469702e+38, attn_logits_soft_cap: float | None = None, sliding_window: int | None = None)` — [`L102`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/prefill_page_attention.py#L102) — Execute prefill page attention over paged KV cache.
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`PrefillPageAttentionConfig`](configs.md#PrefillPageAttentionConfig), [`block_k`](configs.md#PrefillPageAttentionConfig.block_k), [`num_stages`](configs.md#PrefillPageAttentionConfig.num_stages), [`num_warps`](configs.md#PrefillPageAttentionConfig.num_warps)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`create_shard_map_wrapper`](../../ops/core/kernel.md#Kernel.create_shard_map_wrapper), [`prefill_page_attention`](prefill_page_attention.md#prefill_page_attention)

## Functions
- `prefill_page_attention(query: Float[Array, "chunk_size num_heads head_dim"], key_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"], value_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"], context_len: Int[Array, 1], page_indices: Int[Array, num_pages],, *, softmax_scale: float | None = None, mask_value: float = -2.381976426469702e+38, attn_logits_soft_cap: float | None = None, sliding_window: int | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: PrefillPageAttentionConfig | None = None)` — [`L374`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/prefill_page_attention.py#L374) — Execute prefill page attention with automatic optimization.

## Module values
- `_prefill_page_attention_executor` — [`L360`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/prefill_page_attention.py#L360)

