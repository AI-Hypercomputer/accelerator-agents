---
title: 'Module: ejkernel/modules/operations/native_sparse_attention.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/native_sparse_attention.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.native_sparse_attention`/
symbols:
  NativeSparseAttention.run: NativeSparseAttention#run().
  NativeSparseAttention.candidate_cfgs_gpu: NativeSparseAttention#candidate_cfgs_gpu().
  NativeSparseAttention.candidate_cfgs_xla: NativeSparseAttention#candidate_cfgs_xla().
  _sparse_executor._sparse_executor: _sparse_executor._sparse_executor.
  NativeSparseAttention.heuristic_cfg: NativeSparseAttention#heuristic_cfg().
  NativeSparseAttention.candidate_cfgs: NativeSparseAttention#candidate_cfgs().
  NativeSparseAttention.get_impl: NativeSparseAttention#get_impl().
  NativeSparseAttention.candidate_cfgs_tpu: NativeSparseAttention#candidate_cfgs_tpu().
  native_sparse_attention: native_sparse_attention().
  NativeSparseAttention: NativeSparseAttention#
  NativeSparseAttention.__init__: NativeSparseAttention#__init__().
  NativeSparseAttention.candidate_cfgs_shard_map_gpu: NativeSparseAttention#candidate_cfgs_shard_map_gpu.
  NativeSparseAttention.candidate_cfgs_shard_map_tpu: NativeSparseAttention#candidate_cfgs_shard_map_tpu.
  NativeSparseAttention.candidate_cfgs_shard_map_xla: NativeSparseAttention#candidate_cfgs_shard_map_xla.
---
# Module: [`ejkernel/modules/operations/native_sparse_attention.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/native_sparse_attention.py)

## Classes
### `NativeSparseAttention`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/native_sparse_attention.py:94`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/native_sparse_attention.py#L94)
- doc: Native Sparse Attention with custom optimization logic.
- signature: `class NativeSparseAttention(Kernel[NativeSparseAttentionConfig, Array]):`
- members:
  - `__init__(self)` — [`L114`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/native_sparse_attention.py#L114) — Initialize Native Sparse Attention module.
  - `candidate_cfgs(self, inv: Invocation[NativeSparseAttentionConfig, Array])` — [`L266`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/native_sparse_attention.py#L266) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[NativeSparseAttentionConfig, Array])` — [`L297`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/native_sparse_attention.py#L297) — Generate GPU-optimized candidate configurations for NSA.
  - `candidate_cfgs_tpu(self, inv: Invocation[NativeSparseAttentionConfig, Array])` — [`L357`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/native_sparse_attention.py#L357) — Generate TPU-optimized candidate configurations for autotuning.
  - `candidate_cfgs_xla(self, inv: Invocation[NativeSparseAttentionConfig, Array])` — [`L371`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/native_sparse_attention.py#L371) — Generate XLA-optimized candidate configurations for autotuning.
  - `get_impl(self, cfg: NativeSparseAttentionConfig)` — [`L122`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/native_sparse_attention.py#L122) — Get kernel implementation from registry.
  - `heuristic_cfg(self, inv: Invocation[NativeSparseAttentionConfig, Array])` — [`L243`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/native_sparse_attention.py#L243) — Provide default configuration with block sizes.
  - `run(self, query: Float[Array, "batch seq_len num_q_heads head_dim"], key: Float[Array, "batch seq_len num_kv_heads head_dim"], value: Float[Array, "batch seq_len num_kv_heads head_dim"], g_cmp: Float[Array, "batch seq_len num_q_heads"] | None = None, g_slc: Float[Array, "batch seq_len num_q_heads"] | None = None, block_indices: Int[Array, "batch seq_len num_kv_heads num_selected_blocks"] | None = None, block_counts: Int[Array, "batch seq_len num_kv_heads"] | int = 16, softmax_scale: float | None = None, cu_seqlens: Int[Array, num_seqs_plus_one] | None = None, platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, cfg: NativeSparseAttentionConfig)` — [`L137`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/native_sparse_attention.py#L137) — Execute native sparse attention with explicit block indices.
  - `candidate_cfgs_shard_map_gpu` — [`L400`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/native_sparse_attention.py#L400)
  - `candidate_cfgs_shard_map_tpu` — [`L401`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/native_sparse_attention.py#L401)
  - `candidate_cfgs_shard_map_xla` — [`L402`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/native_sparse_attention.py#L402)
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`Platform`](../../kernels/_registry.md#Platform), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`NativeSparseAttentionConfig`](configs.md#NativeSparseAttentionConfig), [`TRITON`](../../kernels/_registry.md#Platform.TRITON), [`block_d`](configs.md#NativeSparseAttentionConfig.block_d), [`block_k`](configs.md#NativeSparseAttentionConfig.block_k), [`block_size`](configs.md#NativeSparseAttentionConfig.block_size), [`num_stages`](configs.md#NativeSparseAttentionConfig.num_stages), [`num_warps`](configs.md#NativeSparseAttentionConfig.num_warps), [`block_q`](configs.md#NativeSparseAttentionConfig.block_q)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`native_sparse_attention`](native_sparse_attention.md#native_sparse_attention)

## Functions
- `native_sparse_attention(query: Float[Array, "batch seq_len num_q_heads head_dim"], key: Float[Array, "batch seq_len num_kv_heads head_dim"], value: Float[Array, "batch seq_len num_kv_heads head_dim"], g_cmp: Float[Array, "batch seq_len num_q_heads"] | None = None, g_slc: Float[Array, "batch seq_len num_q_heads"] | None = None, block_indices: Int[Array, "batch seq_len num_kv_heads num_selected_blocks"] | None = None, block_counts: Int[Array, "batch seq_len num_kv_heads"] | int = 16, cu_seqlens: Int[Array, num_seqs_plus_one] | None = None,, *, softmax_scale: float | None = None, platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: NativeSparseAttentionConfig | None = None)` — [`L419`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/native_sparse_attention.py#L419) — Execute native sparse attention with automatic optimization.

## Module values
- `_sparse_executor` — [`L405`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/native_sparse_attention.py#L405)

