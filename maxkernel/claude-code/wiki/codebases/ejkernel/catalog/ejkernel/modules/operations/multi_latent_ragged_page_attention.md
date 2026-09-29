---
title: 'Module: ejkernel/modules/operations/multi_latent_ragged_page_attention.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/multi_latent_ragged_page_attention.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.multi_latent_ragged_page_attention`/
symbols:
  MultiLatentRaggedPageAttention.candidate_cfgs_gpu: MultiLatentRaggedPageAttention#candidate_cfgs_gpu().
  MultiLatentRaggedPageAttention.run: MultiLatentRaggedPageAttention#run().
  MultiLatentRaggedPageAttention.candidate_cfgs: MultiLatentRaggedPageAttention#candidate_cfgs().
  MultiLatentRaggedPageAttention.heuristic_cfg: MultiLatentRaggedPageAttention#heuristic_cfg().
  MultiLatentRaggedPageAttention.candidate_cfgs_tpu: MultiLatentRaggedPageAttention#candidate_cfgs_tpu().
  _mlrpa_executor._mlrpa_executor: _mlrpa_executor._mlrpa_executor.
  MultiLatentRaggedPageAttention.get_impl: MultiLatentRaggedPageAttention#get_impl().
  MultiLatentRaggedPageAttention._estimate_kv_pages: MultiLatentRaggedPageAttention#_estimate_kv_pages().
  multi_latent_ragged_page_attention: multi_latent_ragged_page_attention().
  MultiLatentRaggedPageAttention: MultiLatentRaggedPageAttention#
  MultiLatentRaggedPageAttention.__init__: MultiLatentRaggedPageAttention#__init__().
  MultiLatentRaggedPageAttention.version: MultiLatentRaggedPageAttention#version.
  __all__: __all__.
---
# Module: [`ejkernel/modules/operations/multi_latent_ragged_page_attention.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention.py)

## Classes
### `MultiLatentRaggedPageAttention`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/multi_latent_ragged_page_attention.py:82`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention.py#L82)
- doc: Multi-Latent Ragged Page Attention (MLRPA) operation.
- signature: `class MultiLatentRaggedPageAttention(Kernel[MultiLatentRaggedPageAttentionConfig, tuple[Array, Array]]):`
- members:
  - `__init__(self)` — [`L102`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention.py#L102) — Initialize MLRPA with its registry operation identifier.
  - `_estimate_kv_pages(self, inv: Invocation[MultiLatentRaggedPageAttentionConfig, tuple[Array, Array]])` — [`L248`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention.py#L248) — Estimate ``num_kv_pages_per_block`` from input shapes.
  - `candidate_cfgs(self, inv: Invocation[MultiLatentRaggedPageAttentionConfig, tuple[Array, Array]])` — [`L307`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention.py#L307) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[MultiLatentRaggedPageAttentionConfig, tuple[Array, Array]])` — [`L338`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention.py#L338) — Generate GPU candidates for TileLang and XLA MLRPA.
  - `candidate_cfgs_tpu(self, inv: Invocation[MultiLatentRaggedPageAttentionConfig, tuple[Array, Array]])` — [`L375`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention.py#L375) — Generate TPU candidates for Pallas and XLA MLRPA.
  - `get_impl(self, cfg: MultiLatentRaggedPageAttentionConfig)` — [`L106`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention.py#L106) — Resolve the concrete kernel implementation from the registry.
  - `heuristic_cfg(self, inv: Invocation[MultiLatentRaggedPageAttentionConfig, tuple[Array, Array]])` — [`L281`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention.py#L281) — Provide a default configuration based on input shapes.
  - `run(self, queries_nope: Float[Array, "total_tokens num_q_heads kv_latent_dim"], queries_pe: Float[Array, "total_tokens num_q_heads qk_pe_dim"], keys_values: Float[Array, "total_tokens kv_latent_dim"], keys_pe: Float[Array, "total_tokens qk_pe_dim"], kv_cache: Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"], kv_lens: Int32[Array, max_num_seqs], block_tables: Int32[Array, max_num_seqs_times_pages_per_seq], query_start_loc: Int32[Array, max_num_seqs_plus_1], distribution: Int32[Array, 3], *, softmax_scale: float | None = None, sliding_window: int | None = None, logits_soft_cap: float | None = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: MultiLatentRaggedPageAttentionConfig)` — [`L131`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention.py#L131) — Execute multi-latent ragged paged attention.
  - `version` — [`L100`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention.py#L100)
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`MultiLatentRaggedPageAttentionConfig`](configs.md#MultiLatentRaggedPageAttentionConfig), [`chunk_prefill_size`](configs.md#MultiLatentRaggedPageAttentionConfig.chunk_prefill_size), [`num_kv_pages_per_block`](configs.md#MultiLatentRaggedPageAttentionConfig.num_kv_pages_per_block), [`num_queries_per_block`](configs.md#MultiLatentRaggedPageAttentionConfig.num_queries_per_block), [`vmem_limit_bytes`](configs.md#MultiLatentRaggedPageAttentionConfig.vmem_limit_bytes), [`num_stages`](configs.md#MultiLatentRaggedPageAttentionConfig.num_stages), [`num_warps`](configs.md#MultiLatentRaggedPageAttentionConfig.num_warps)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`multi_latent_ragged_page_attention`](multi_latent_ragged_page_attention.md#multi_latent_ragged_page_attention)

## Functions
- `multi_latent_ragged_page_attention(queries_nope: Float[Array, "total_tokens num_q_heads kv_latent_dim"], queries_pe: Float[Array, "total_tokens num_q_heads qk_pe_dim"], keys_values: Float[Array, "total_tokens kv_latent_dim"], keys_pe: Float[Array, "total_tokens qk_pe_dim"], kv_cache: Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"], kv_lens: Int32[Array, max_num_seqs], block_tables: Int32[Array, max_num_seqs_times_pages_per_seq], query_start_loc: Int32[Array, max_num_seqs_plus_1], distribution: Int32[Array, 3],, *, softmax_scale: float | None = None, sliding_window: int | None = None, logits_soft_cap: float | None = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: MultiLatentRaggedPageAttentionConfig | None = None)` — [`L411`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention.py#L411) — Execute Multi-Latent Ragged Page Attention with automatic optimization.

## Module values
- `__all__` — [`L503`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention.py#L503)
- `_mlrpa_executor` — [`L397`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_latent_ragged_page_attention.py#L397)

