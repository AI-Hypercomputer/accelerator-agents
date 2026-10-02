---
title: 'Module: ejkernel/modules/operations/ragged_page_attention_v2.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/ragged_page_attention_v2.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.ragged_page_attention_v2`/
symbols:
  RaggedPageAttentionv2.candidate_cfgs_gpu: RaggedPageAttentionv2#candidate_cfgs_gpu().
  RaggedPageAttentionv2.run: RaggedPageAttentionv2#run().
  RaggedPageAttentionv2.candidate_cfgs_tpu: RaggedPageAttentionv2#candidate_cfgs_tpu().
  _ragged_page_attention_executor._ragged_page_attention_executor: _ragged_page_attention_executor._ragged_page_attention_executor.
  _xla_block_candidates_v2: _xla_block_candidates_v2().
  RaggedPageAttentionv2.heuristic_cfg: RaggedPageAttentionv2#heuristic_cfg().
  RaggedPageAttentionv2.candidate_cfgs: RaggedPageAttentionv2#candidate_cfgs().
  RaggedPageAttentionv2.get_impl: RaggedPageAttentionv2#get_impl().
  ragged_page_attention_v2: ragged_page_attention_v2().
  RaggedPageAttentionv2: RaggedPageAttentionv2#
  RaggedPageAttentionv2.create_shard_map_wrapper: RaggedPageAttentionv2#create_shard_map_wrapper().
  RaggedPageAttentionv2._wrapped_ragged_page_attn: RaggedPageAttentionv2#_wrapped_ragged_page_attn().
  RaggedPageAttentionv2.__init__: RaggedPageAttentionv2#__init__().
  RaggedPageAttentionv2.candidate_cfgs_shard_map_tpu: RaggedPageAttentionv2#candidate_cfgs_shard_map_tpu.
  RaggedPageAttentionv2.candidate_cfgs_shard_map_gpu: RaggedPageAttentionv2#candidate_cfgs_shard_map_gpu.
  RaggedPageAttentionv2.pick_warps_stages: RaggedPageAttentionv2#pick_warps_stages().
---
# Module: [`ejkernel/modules/operations/ragged_page_attention_v2.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v2.py)

## Classes
### `RaggedPageAttentionv2`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/ragged_page_attention_v2.py:136`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v2.py#L136)
- doc: Ragged Page Attention with custom optimization logic.
- signature: `class RaggedPageAttentionv2(Kernel[RaggedPageAttentionv2Config, Array]):`
- members:
  - `__init__(self)` — [`L162`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v2.py#L162) — Initialize Ragged Page Attention module.
  - `_wrapped_ragged_page_attn(queries: Float[Array, "total_tokens num_q_heads head_dim"], kv_pages: Float[Array, "num_pages page_size num_combined_kv_heads head_dim"], context_lens: Int[Array, num_seqs], block_tables: Int[Array, "num_seqs pages_per_seq"], query_start_loc: Int[Array, num_seqs_plus_one], num_seqs: Array | int, softmax_aux: Float[Array, num_sinks] | None = None)` — [`L218`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v2.py#L218) — Shard-map compatible wrapper that delegates to self.run with captured params.
  - `candidate_cfgs(self, inv: Invocation[RaggedPageAttentionv2Config, Array])` — [`L396`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v2.py#L396) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[RaggedPageAttentionv2Config, Array])` — [`L435`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v2.py#L435) — Generate candidate configurations for autotuning on GPU (Triton).
  - `candidate_cfgs_tpu(self, inv: Invocation[RaggedPageAttentionv2Config, Array])` — [`L602`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v2.py#L602) — Generate candidate configurations for autotuning on TPU (Pallas backend).
  - `create_shard_map_wrapper(self, queries: Float[Array, "total_tokens num_q_heads head_dim"], kv_pages: Float[Array, "num_pages page_size num_combined_kv_heads head_dim"], context_lens: Int[Array, num_seqs], block_tables: Int[Array, "num_seqs pages_per_seq"], query_start_loc: Int[Array, num_seqs_plus_one], num_seqs: Array | int, softmax_scale: float | None = None, logits_soft_cap: float | None = None, compute_dtype: DTypeLike = jnp.bfloat16, optimized: bool = False, sliding_window: int | None = None, softmax_aux: Float[Array, num_sinks] | None = None, mask_value: float | None = None, vmem_limit_bytes: int | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RaggedPageAttentionv2Config | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec, ...] | None = None, out_specs: PartitionSpec | None = None, check_vma: bool = False)` — [`L170`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v2.py#L170) — Create a shard_map wrapper specifically for ragged page attention.
  - `get_impl(self, cfg: RaggedPageAttentionv2Config)` — [`L267`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v2.py#L267) — Get kernel implementation from registry.
  - `heuristic_cfg(self, inv: Invocation[RaggedPageAttentionv2Config, Array])` — [`L377`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v2.py#L377) — Provide default configuration optimized for ragged page attention.
  - `pick_warps_stages(block_m: int, npages: int)` — [`L501`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v2.py#L501) — Select warp count and pipeline stages based on block size and head dimension.
  - `run(self, queries: Float[Array, "total_tokens num_q_heads head_dim"], kv_pages: Float[Array, "num_pages page_size num_combined_kv_heads head_dim"], context_lens: Int[Array, num_seqs], block_tables: Int[Array, "num_seqs pages_per_seq"], query_start_loc: Int[Array, num_seqs_plus_one], num_seqs: Array | int, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, softmax_scale: float | None = None, logits_soft_cap: float | None = None, compute_dtype: DTypeLike = jnp.bfloat16, optimized: bool = False, sliding_window: int | None = None, softmax_aux: Float[Array, num_sinks] | None = None, mask_value: float | None = None, vmem_limit_bytes: int | None = None, *, cfg: RaggedPageAttentionv2Config)` — [`L282`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v2.py#L282) — Execute ragged page attention over variable-length sequences.
  - `candidate_cfgs_shard_map_gpu` — [`L665`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v2.py#L665)
  - `candidate_cfgs_shard_map_tpu` — [`L664`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v2.py#L664)
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`RaggedPageAttentionv2Config`](configs.md#RaggedPageAttentionv2Config), [`_xla_block_candidates_v2`](ragged_page_attention_v2.md#_xla_block_candidates_v2), [`num_kv_pages_per_block`](configs.md#RaggedPageAttentionv2Config.num_kv_pages_per_block), [`num_queries_per_block`](configs.md#RaggedPageAttentionv2Config.num_queries_per_block), [`num_stages`](configs.md#RaggedPageAttentionv2Config.num_stages), [`num_warps`](configs.md#RaggedPageAttentionv2Config.num_warps)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`create_shard_map_wrapper`](../../ops/core/kernel.md#Kernel.create_shard_map_wrapper), [`ragged_page_attention_v2`](ragged_page_attention_v2.md#ragged_page_attention_v2)

## Functions
- `_xla_block_candidates_v2(inv: Invocation[RaggedPageAttentionv2Config, Array])` — [`L80`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v2.py#L80) — Generate power-of-2 XLA configurations with larger block sizes.
- `ragged_page_attention_v2(queries: Float[Array, "total_tokens num_q_heads head_dim"], kv_pages: Float[Array, "num_pages page_size num_combined_kv_heads head_dim"], context_lens: Int[Array, num_seqs], block_tables: Int[Array, "num_seqs pages_per_seq"], query_start_loc: Int[Array, num_seqs_plus_one], num_seqs: Array | int, softmax_aux: Float[Array, num_sinks] | None = None,, *, softmax_scale: float | None = None, logits_soft_cap: float | None = None, compute_dtype: DTypeLike = jnp.bfloat16, optimized: bool = False, sliding_window: int | None = None, mask_value: float | None = None, vmem_limit_bytes: int | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RaggedPageAttentionv2Config | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec | None, ...] | None = None, out_specs: PartitionSpec | None = None)` — [`L682`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v2.py#L682) — Execute ragged page attention with automatic optimization.

## Module values
- `_ragged_page_attention_executor` — [`L668`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_page_attention_v2.py#L668)

