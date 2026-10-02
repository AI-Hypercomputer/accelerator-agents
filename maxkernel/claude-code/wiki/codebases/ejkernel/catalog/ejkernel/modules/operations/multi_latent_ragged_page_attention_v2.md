---
title: 'Module: ejkernel/modules/operations/multi_latent_ragged_page_attention_v2.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/multi_latent_ragged_page_attention_v2.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.multi_latent_ragged_page_attention_v2`/
symbols:
  MultiLatentRaggedPageAttentionV2.candidate_cfgs_gpu: MultiLatentRaggedPageAttentionV2#candidate_cfgs_gpu().
  MultiLatentRaggedPageAttentionV2.candidate_cfgs: MultiLatentRaggedPageAttentionV2#candidate_cfgs().
  MultiLatentRaggedPageAttentionV2.run: MultiLatentRaggedPageAttentionV2#run().
  MultiLatentRaggedPageAttentionV2.heuristic_cfg: MultiLatentRaggedPageAttentionV2#heuristic_cfg().
  MultiLatentRaggedPageAttentionV2.candidate_cfgs_tpu: MultiLatentRaggedPageAttentionV2#candidate_cfgs_tpu().
  _mlrpa_v2_executor._mlrpa_v2_executor: _mlrpa_v2_executor._mlrpa_v2_executor.
  MultiLatentRaggedPageAttentionV2.get_impl: MultiLatentRaggedPageAttentionV2#get_impl().
  MultiLatentRaggedPageAttentionV2._estimate_kv_pages: MultiLatentRaggedPageAttentionV2#_estimate_kv_pages().
  multi_latent_ragged_page_attention_v2: multi_latent_ragged_page_attention_v2().
  _repeat_case: _repeat_case().
  MultiLatentRaggedPageAttentionV2: MultiLatentRaggedPageAttentionV2#
  MultiLatentRaggedPageAttentionV2.__init__: MultiLatentRaggedPageAttentionV2#__init__().
  MultiLatentRaggedPageAttentionV2.version: MultiLatentRaggedPageAttentionV2#version.
  __all__: __all__.
---
# Module: [`ejkernel/modules/operations/multi_latent_ragged_page_attention_v2.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention_v2.py)

## Classes
### `MultiLatentRaggedPageAttentionV2`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/multi_latent_ragged_page_attention_v2.py:84`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention_v2.py#L84)
- doc: Multi-Latent Ragged Page Attention v2 with per-case block-size tuning.
- signature: `class MultiLatentRaggedPageAttentionV2(Kernel[MultiLatentRaggedPageAttentionV2Config, tuple[Array, Array]]):`
- members:
  - `__init__(self)` — [`L97`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention_v2.py#L97) — Initialize MLRPA-v2 with its registry operation identifier.
  - `_estimate_kv_pages(self, inv: Invocation[MultiLatentRaggedPageAttentionV2Config, tuple[Array, Array]])` — [`L214`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention_v2.py#L214) — Estimate a reasonable KV-pages-per-block from invocation shapes.
  - `candidate_cfgs(self, inv: Invocation[MultiLatentRaggedPageAttentionV2Config, tuple[Array, Array]])` — [`L273`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention_v2.py#L273) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[MultiLatentRaggedPageAttentionV2Config, tuple[Array, Array]])` — [`L304`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention_v2.py#L304) — Generate GPU candidates for TileLang and XLA MLRPA-v2.
  - `candidate_cfgs_tpu(self, inv: Invocation[MultiLatentRaggedPageAttentionV2Config, tuple[Array, Array]])` — [`L338`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention_v2.py#L338) — Generate TPU candidates for Pallas and XLA MLRPA-v2.
  - `get_impl(self, cfg: MultiLatentRaggedPageAttentionV2Config)` — [`L101`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention_v2.py#L101) — Get kernel implementation from registry based on configuration.
  - `heuristic_cfg(self, inv: Invocation[MultiLatentRaggedPageAttentionV2Config, tuple[Array, Array]])` — [`L245`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention_v2.py#L245) — Generate default configuration based on input characteristics.
  - `run(self, queries_nope: Float[Array, "total_tokens num_q_heads kv_latent_dim"], queries_pe: Float[Array, "total_tokens num_q_heads qk_pe_dim"], keys_values: Float[Array, "total_tokens kv_latent_dim"], keys_pe: Float[Array, "total_tokens qk_pe_dim"], kv_cache: Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"], kv_lens: Int32[Array, max_num_seqs], block_tables: Int32[Array, max_num_seqs_times_pages_per_seq], query_start_loc: Int32[Array, max_num_seqs_plus_1], distribution: Int32[Array, 3], *, softmax_scale: float | None = None, sliding_window: int | None = None, logits_soft_cap: float | None = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: MultiLatentRaggedPageAttentionV2Config)` — [`L124`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention_v2.py#L124) — Execute MLA ragged paged attention v2 and update the paged KV cache.
  - `version` — [`L95`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention_v2.py#L95)
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`MultiLatentRaggedPageAttentionV2Config`](configs.md#MultiLatentRaggedPageAttentionV2Config), [`num_kv_pages_per_block`](configs.md#MultiLatentRaggedPageAttentionV2Config.num_kv_pages_per_block), [`num_queries_per_block`](configs.md#MultiLatentRaggedPageAttentionV2Config.num_queries_per_block), [`_repeat_case`](multi_latent_ragged_page_attention_v2.md#_repeat_case), [`chunk_prefill_size`](configs.md#MultiLatentRaggedPageAttentionV2Config.chunk_prefill_size), [`vmem_limit_bytes`](configs.md#MultiLatentRaggedPageAttentionV2Config.vmem_limit_bytes), [`num_stages`](configs.md#MultiLatentRaggedPageAttentionV2Config.num_stages), [`num_warps`](configs.md#MultiLatentRaggedPageAttentionV2Config.num_warps)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`multi_latent_ragged_page_attention_v2`](multi_latent_ragged_page_attention_v2.md#multi_latent_ragged_page_attention_v2)

## Functions
- `_repeat_case(value: int)` — [`L69`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention_v2.py#L69) — Broadcast a scalar block size into a ``(decode, prefill, mixed)`` triple.
- `multi_latent_ragged_page_attention_v2(queries_nope: Float[Array, "total_tokens num_q_heads kv_latent_dim"], queries_pe: Float[Array, "total_tokens num_q_heads qk_pe_dim"], keys_values: Float[Array, "total_tokens kv_latent_dim"], keys_pe: Float[Array, "total_tokens qk_pe_dim"], kv_cache: Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"], kv_lens: Int32[Array, max_num_seqs], block_tables: Int32[Array, max_num_seqs_times_pages_per_seq], query_start_loc: Int32[Array, max_num_seqs_plus_1], distribution: Int32[Array, 3],, *, softmax_scale: float | None = None, sliding_window: int | None = None, logits_soft_cap: float | None = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: MultiLatentRaggedPageAttentionV2Config | None = None)` — [`L374`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention_v2.py#L374) — Execute Multi-Latent Ragged Page Attention v2 with automatic optimization.

## Module values
- `__all__` — [`L457`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention_v2.py#L457)
- `_mlrpa_v2_executor` — [`L360`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention_v2.py#L360)

