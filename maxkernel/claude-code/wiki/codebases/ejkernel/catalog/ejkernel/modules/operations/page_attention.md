---
title: 'Module: ejkernel/modules/operations/page_attention.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/page_attention.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.page_attention`/
symbols:
  PageAttention.run: PageAttention#run().
  _page_attention_executor._page_attention_executor: _page_attention_executor._page_attention_executor.
  PageAttention.candidate_cfgs_gpu: PageAttention#candidate_cfgs_gpu().
  PageAttention.heuristic_cfg: PageAttention#heuristic_cfg().
  PageAttention.candidate_cfgs: PageAttention#candidate_cfgs().
  PageAttention.candidate_cfgs_tpu: PageAttention#candidate_cfgs_tpu().
  PageAttention.get_impl: PageAttention#get_impl().
  page_attention: page_attention().
  PageAttention: PageAttention#
  PageAttention._wrapper: PageAttention#_wrapper().
  PageAttention.create_shard_map_wrapper: PageAttention#create_shard_map_wrapper().
  PageAttention.__init__: PageAttention#__init__().
---
# Module: [`ejkernel/modules/operations/page_attention.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/page_attention.py)

## Classes
### `PageAttention`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/page_attention.py:82`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/page_attention.py#L82)
- doc: Page Attention with custom optimization logic.
- signature: `class PageAttention(Kernel[PageAttentionConfig, Array]):`
- members:
  - `__init__(self)` — [`L105`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/page_attention.py#L105) — Initialize Page Attention module.
  - `_wrapper(query, key_cache, value_cache, context_lens, block_tables)` — [`L400`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/page_attention.py#L400) — Shard-map compatible wrapper that delegates to self.run with captured params.
  - `candidate_cfgs(self, inv: Invocation[PageAttentionConfig, Array])` — [`L245`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/page_attention.py#L245) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[PageAttentionConfig, Array])` — [`L274`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/page_attention.py#L274) — Generate GPU candidates for paged attention across Triton, TileLang and XLA.
  - `candidate_cfgs_tpu(self, inv: Invocation[PageAttentionConfig, Array])` — [`L327`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/page_attention.py#L327) — Generate TPU candidates for Pallas and XLA page attention.
  - `create_shard_map_wrapper(self, query: Float[Array, "num_seqs num_heads head_dim"], key_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"], value_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"], context_lens: Int[Array, num_seqs], block_tables: Int[Array, "num_seqs max_blocks"], attn_scale: float | None = None, max_context_len: int | None = None, num_splits: int = 0, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, cfg: PageAttentionConfig | None = None, mask_value: float = -2.381976426469702e+38, attn_logits_soft_cap: float | None = None, pages_per_compute_block: int | None = None, megacore_mode: str | None = None, inline_seq_dim: bool = True, sliding_window: int | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec, ...] | None = None, out_specs: PartitionSpec | None = None, check_vma: bool = False)` — [`L342`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/page_attention.py#L342) — Create a shard_map wrapper for distributed execution.
  - `get_impl(self, cfg: PageAttentionConfig)` — [`L113`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/page_attention.py#L113) — Get kernel implementation from registry.
  - `heuristic_cfg(self, inv: Invocation[PageAttentionConfig, Array])` — [`L226`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/page_attention.py#L226) — Provide default configuration optimized for paged attention.
  - `run(self, query: Float[Array, "num_seqs num_heads head_dim"], key_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"], value_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"], context_lens: Int[Array, num_seqs], block_tables: Int[Array, "num_seqs max_blocks"], attn_scale: float | None = None, max_context_len: int | None = None, num_splits: int = 0, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, cfg: PageAttentionConfig, mask_value: float = -2.381976426469702e+38, attn_logits_soft_cap: float | None = None, pages_per_compute_block: int | None = None, megacore_mode: str | None = None, inline_seq_dim: bool = True, sliding_window: int | None = None)` — [`L128`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/page_attention.py#L128) — Execute page attention over paged KV cache.
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`Platform`](../../kernels/_registry.md#Platform), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`PageAttentionConfig`](configs.md#PageAttentionConfig), [`TRITON`](../../kernels/_registry.md#Platform.TRITON), [`num_splits`](configs.md#PageAttentionConfig.num_splits), [`num_stages`](configs.md#PageAttentionConfig.num_stages), [`num_warps`](configs.md#PageAttentionConfig.num_warps), [`pages_per_compute_block`](configs.md#PageAttentionConfig.pages_per_compute_block)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`create_shard_map_wrapper`](../../ops/core/kernel.md#Kernel.create_shard_map_wrapper), [`page_attention`](page_attention.md#page_attention)

## Functions
- `page_attention(query: Float[Array, "num_seqs num_heads head_dim"], key_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"], value_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"], context_lens: Int[Array, num_seqs], block_tables: Int[Array, "num_seqs max_blocks"],, *, attn_scale: float | None = None, max_context_len: int | None = None, num_splits: int = 0, mask_value: float = -2.381976426469702e+38, attn_logits_soft_cap: float | None = None, pages_per_compute_block: int | None = None, megacore_mode: str | None = None, inline_seq_dim: bool = True, sliding_window: int | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: PageAttentionConfig | None = None)` — [`L460`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/page_attention.py#L460) — Execute page attention with automatic optimization.

## Module values
- `_page_attention_executor` — [`L446`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/page_attention.py#L446)

