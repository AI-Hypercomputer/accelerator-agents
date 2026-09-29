---
title: 'Module: ejkernel/modules/operations/unified_attention.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/unified_attention.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.unified_attention`/
symbols:
  UnifiedAttention.heuristic_cfg: UnifiedAttention#heuristic_cfg().
  UnifiedAttention.run: UnifiedAttention#run().
  UnifiedAttention.candidate_cfgs_gpu: UnifiedAttention#candidate_cfgs_gpu().
  _unified_attention_executor._unified_attention_executor: _unified_attention_executor._unified_attention_executor.
  UnifiedAttention.candidate_cfgs_tpu: UnifiedAttention#candidate_cfgs_tpu().
  UnifiedAttention.get_impl: UnifiedAttention#get_impl().
  _resolve_inv_arg: _resolve_inv_arg().
  unified_attention: unified_attention().
  UnifiedAttention: UnifiedAttention#
  UnifiedAttention.candidate_cfgs: UnifiedAttention#candidate_cfgs().
  UnifiedAttention.create_shard_map_wrapper: UnifiedAttention#create_shard_map_wrapper().
  _ARGUMENT_INDEX: _ARGUMENT_INDEX.
  UnifiedAttention._wrapped_unified_attention: UnifiedAttention#_wrapped_unified_attention().
  UnifiedAttention.__init__: UnifiedAttention#__init__().
  MIN_LAUNCH_GRID_SIZE_2D: MIN_LAUNCH_GRID_SIZE_2D.
  NUM_PAR_SOFTMAX_SEGMENTS: NUM_PAR_SOFTMAX_SEGMENTS.
  _ARGUMENT_ORDER: _ARGUMENT_ORDER.
---
# Module: [`ejkernel/modules/operations/unified_attention.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/unified_attention.py)

## Classes
### `UnifiedAttention`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/unified_attention.py:141`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/unified_attention.py#L141)
- doc: vLLM-style unified attention over a paged KV cache (inference-only).
- signature: `class UnifiedAttention(Kernel[UnifiedAttentionConfig, Array]):`
- members:
  - `__init__(self)` — [`L177`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/unified_attention.py#L177) — Initialize the UnifiedAttention kernel.
  - `_wrapped_unified_attention(queries: Float[Array, "total_tokens num_q_heads head_dim"], key_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"], value_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"], kv_lens: Int32[Array, num_seqs], block_tables: Int32[Array, "num_seqs max_blocks_per_seq"], query_start_loc: Int32[Array, num_seqs_plus_1], alibi_slopes: Float[Array, num_q_heads] | None, qq_bias: Float[Array, "num_query_tokens num_query_tokens"] | None, softmax_aux: Float[Array, num_q_heads] | None)` — [`L231`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/unified_attention.py#L231) — Shard-map compatible wrapper that delegates to self.run with captured params.
  - `candidate_cfgs(self, inv: Invocation[UnifiedAttentionConfig, Array])` — [`L417`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/unified_attention.py#L417) — Return candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[UnifiedAttentionConfig, Array])` — [`L431`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/unified_attention.py#L431) — Return GPU candidates for every registered unified-attention backend.
  - `candidate_cfgs_tpu(self, inv: Invocation[UnifiedAttentionConfig, Array])` — [`L519`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/unified_attention.py#L519) — Return the XLA TPU candidate for unified attention.
  - `create_shard_map_wrapper(self, queries: Float[Array, "total_tokens num_q_heads head_dim"], key_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"], value_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"], kv_lens: Int32[Array, num_seqs], block_tables: Int32[Array, "num_seqs max_blocks_per_seq"], query_start_loc: Int32[Array, num_seqs_plus_1], alibi_slopes: Float[Array, num_q_heads] | None = None, qq_bias: Float[Array, "num_query_tokens num_query_tokens"] | None = None, softmax_aux: Float[Array, num_q_heads] | None = None, softmax_scale: float | None = None, causal: bool = True, sliding_window: int | None = None, logits_soft_cap: float | None = None, seq_threshold_3d: int | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: UnifiedAttentionConfig | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec, ...] | None = None, out_specs: PartitionSpec | None = None, check_vma: bool = False)` — [`L181`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/unified_attention.py#L181) — Create a shard_map wrapper for distributed unified attention.
  - `get_impl(self, cfg: UnifiedAttentionConfig)` — [`L284`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/unified_attention.py#L284) — Get the platform-specific implementation.
  - `heuristic_cfg(self, inv: Invocation[UnifiedAttentionConfig, Array])` — [`L389`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/unified_attention.py#L389) — Generate default configuration based on input characteristics.
  - `run(self, queries: Float[Array, "total_tokens num_q_heads head_dim"], key_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"], value_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"], kv_lens: Int32[Array, num_seqs], block_tables: Int32[Array, "num_seqs max_blocks_per_seq"], query_start_loc: Int32[Array, num_seqs_plus_1], alibi_slopes: Float[Array, num_q_heads] | None = None, qq_bias: Float[Array, "num_query_tokens num_query_tokens"] | None = None, softmax_aux: Float[Array, num_q_heads] | None = None, *, softmax_scale: float | None = None, causal: bool = True, sliding_window: int | None = None, logits_soft_cap: float | None = None, seq_threshold_3d: int | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: UnifiedAttentionConfig)` — [`L296`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/unified_attention.py#L296) — Execute unified paged attention.
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`UnifiedAttentionConfig`](configs.md#UnifiedAttentionConfig), [`num_par_softmax_segments`](configs.md#UnifiedAttentionConfig.num_par_softmax_segments), [`seq_threshold_3d`](configs.md#UnifiedAttentionConfig.seq_threshold_3d), [`block_dim`](configs.md#UnifiedAttentionConfig.block_dim), [`_resolve_inv_arg`](unified_attention.md#_resolve_inv_arg), [`num_stages`](configs.md#UnifiedAttentionConfig.num_stages), [`num_warps`](configs.md#UnifiedAttentionConfig.num_warps), [`MIN_LAUNCH_GRID_SIZE_2D`](unified_attention.md#MIN_LAUNCH_GRID_SIZE_2D), [`NUM_PAR_SOFTMAX_SEGMENTS`](unified_attention.md#NUM_PAR_SOFTMAX_SEGMENTS)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`create_shard_map_wrapper`](../../ops/core/kernel.md#Kernel.create_shard_map_wrapper), [`unified_attention`](unified_attention.md#unified_attention)

## Functions
- `_resolve_inv_arg(inv: Invocation, name: str)` — [`L116`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/unified_attention.py#L116) — Resolve an argument from an Invocation by name.
- `unified_attention(queries: Float[Array, "total_tokens num_q_heads head_dim"], key_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"], value_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"], kv_lens: Int32[Array, num_seqs], block_tables: Int32[Array, "num_seqs max_blocks_per_seq"], query_start_loc: Int32[Array, num_seqs_plus_1], alibi_slopes: Float[Array, num_q_heads] | None = None, qq_bias: Float[Array, "num_query_tokens num_query_tokens"] | None = None, softmax_aux: Float[Array, num_q_heads] | None = None,, *, softmax_scale: float | None = None, causal: bool = True, sliding_window: int | None = None, logits_soft_cap: float | None = None, seq_threshold_3d: int | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec | None, ...] | None = None, out_specs: PartitionSpec | None = None, cfg: UnifiedAttentionConfig | None = None)` — [`L545`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/unified_attention.py#L545) — Execute unified paged attention with automatic platform selection.

## Module values
- `MIN_LAUNCH_GRID_SIZE_2D` — [`L102`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/unified_attention.py#L102)
- `NUM_PAR_SOFTMAX_SEGMENTS` — [`L103`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/unified_attention.py#L103)
- `_ARGUMENT_INDEX` — [`L113`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/unified_attention.py#L113)
- `_ARGUMENT_ORDER` — [`L105`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/unified_attention.py#L105)
- `_unified_attention_executor` — [`L535`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/unified_attention.py#L535)

