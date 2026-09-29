---
title: 'Module: ejkernel/modules/operations/blocksparse_attention.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/blocksparse_attention.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.blocksparse_attention`/
symbols:
  BlockSparseAttention.candidate_cfgs_gpu: BlockSparseAttention#candidate_cfgs_gpu().
  BlockSparseAttention.candidate_cfgs_tpu: BlockSparseAttention#candidate_cfgs_tpu().
  BlockSparseAttention.candidate_cfgs_xla: BlockSparseAttention#candidate_cfgs_xla().
  BlockSparseAttention.heuristic_cfg: BlockSparseAttention#heuristic_cfg().
  BlockSparseAttention.heuristic_cfg_gpu: BlockSparseAttention#heuristic_cfg_gpu().
  BlockSparseAttention.heuristic_cfg_tpu: BlockSparseAttention#heuristic_cfg_tpu().
  BlockSparseAttention.candidate_cfgs: BlockSparseAttention#candidate_cfgs().
  blocksparse_attention: blocksparse_attention().
  BlockSparseAttention.run: BlockSparseAttention#run().
  _executor._executor: _executor._executor.
  BlockSparseAttention.get_impl: BlockSparseAttention#get_impl().
  BlockSparseAttention.create_shard_map_wrapper: BlockSparseAttention#create_shard_map_wrapper().
  BlockSparseAttention: BlockSparseAttention#
  BlockSparseAttention._wrapped_blocksparse_attn: BlockSparseAttention#_wrapped_blocksparse_attn().
  BlockSparseAttention.bwd_block: BlockSparseAttention#bwd_block().
  BlockSparseAttention.nearest_128_from_set: BlockSparseAttention#nearest_128_from_set().
  BlockSparseAttention.bwd_tile: BlockSparseAttention#bwd_tile().
  BlockSparseAttention.__init__: BlockSparseAttention#__init__().
  BlockSparseAttention.candidate_cfgs_shard_map_gpu: BlockSparseAttention#candidate_cfgs_shard_map_gpu.
  BlockSparseAttention.candidate_cfgs_shard_map_tpu: BlockSparseAttention#candidate_cfgs_shard_map_tpu.
  BlockSparseAttention.candidate_cfgs_shard_map_xla: BlockSparseAttention#candidate_cfgs_shard_map_xla.
  BlockSparseAttention.pick_warps_stages: BlockSparseAttention#pick_warps_stages().
  BlockSparseAttention.win_span: BlockSparseAttention#win_span().
  BlockSparseAttention.window_total: BlockSparseAttention#window_total().
  BlockSparseAttention.smem_est_bytes: BlockSparseAttention#smem_est_bytes().
---
# Module: [`ejkernel/modules/operations/blocksparse_attention.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py)

## Classes
### `BlockSparseAttention`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/blocksparse_attention.py:96`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L96)
- doc: Block-sparse attention kernel with custom optimization logic.
- signature: `class BlockSparseAttention(Kernel[BlockSparseAttentionConfig, Array]):`
- members:
  - `__init__(self)` — [`L139`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L139) — Initialize BlockSparseAttention module.
  - `_wrapped_blocksparse_attn(query: Float[Array, "batch num_heads seq_len head_dim"], key: Float[Array, "batch kv_num_heads kv_len head_dim"], value: Float[Array, "batch kv_num_heads kv_len vhead_dim"], softmax_aux: Float[Array, num_sinks] | None, bias: Float[Array, "batch num_heads seq_len kv_len"] | None, q_segment_ids: Int[Array, "batch seq_len"] | None, kv_segment_ids: Int[Array, "batch kv_len"] | None, q_positions: Int[Array, "batch seq_len"] | None, kv_positions: Int[Array, "batch kv_len"] | None)` — [`L188`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L188) — Shard-local blocksparse attention forwarding to self.run.
  - `bwd_block(x: int, cap: int = 128)` — [`L571`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L571) — Compute backward block size from a forward block size.
  - `bwd_tile(x: int)` — [`L693`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L693) — Compute backward tile size from a forward tile size.
  - `candidate_cfgs(self, inv: Invocation[BlockSparseAttentionConfig, Array])` — [`L440`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L440) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[BlockSparseAttentionConfig, Array])` — [`L482`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L482) — Generate GPU-optimized candidate configurations for autotuning (Triton).
  - `candidate_cfgs_tpu(self, inv: Invocation[BlockSparseAttentionConfig, Array])` — [`L656`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L656) — Generate TPU-optimized candidate configurations for autotuning (Pallas).
  - `candidate_cfgs_xla(self, inv: Invocation[BlockSparseAttentionConfig, Array])` — [`L749`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L749) — Generate XLA-optimized candidate configurations for autotuning.
  - `create_shard_map_wrapper(self, query: Float[Array, "batch num_heads seq_len head_dim"], key: Float[Array, "batch kv_num_heads kv_len head_dim"], value: Float[Array, "batch kv_num_heads kv_len vhead_dim"], softmax_aux: Float[Array, num_sinks] | None = None, bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None, q_segment_ids: Int[Array, "batch seq_len"] | None = None, kv_segment_ids: Int[Array, "batch kv_len"] | None = None, q_positions: Int[Array, "batch seq_len"] | None = None, kv_positions: Int[Array, "batch kv_len"] | None = None, sequence_parallelism_mesh_axis_name: str | None = None, logits_soft_cap: float | None = None, qkv_layouts: tuple[SparseMask] | None = None, softmax_scale: float | None = None, mask_builder: (typing.Callable[[int, int, int, int, int], Mask] | typing.Callable[[], SparseMask] | None) = None, sliding_window: int | tuple[int, int] | None = None, chunk_size: int | None = None, causal: bool = True, fused_backward: bool = False, platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: BlockSparseAttentionConfig | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec, ...] | None = None, out_specs: PartitionSpec | None = None, check_vma: bool = False)` — [`L143`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L143) — Create a shard_map wrapper specifically for blocksparse attention.
  - `get_impl(self, cfg: BlockSparseAttentionConfig)` — [`L247`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L247) — Get kernel implementation from registry based on configuration.
  - `heuristic_cfg(self, inv: Invocation[BlockSparseAttentionConfig, Array])` — [`L411`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L411) — Provide default configuration based on invocation context.
  - `heuristic_cfg_gpu(self, inv: Invocation[BlockSparseAttentionConfig, Array])` — [`L353`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L353) — Provide default configuration based on invocation context.
  - `heuristic_cfg_tpu(self, inv: Invocation[BlockSparseAttentionConfig, Array])` — [`L382`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L382) — Provide default configuration based on invocation context.
  - `nearest_128_from_set(x: int, allowed=(128, 256, 512, 1024))` — [`L676`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L676) — Return the allowed value nearest to x, breaking ties by smallest.
  - `pick_warps_stages(qb: int, kb: int, dh: int)` — [`L554`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L554) — Select num_warps and num_stages based on block sizes and head dim.
  - `run(self, query: Float[Array, "batch num_heads seq_len head_dim"], key: Float[Array, "batch kv_num_heads kv_len head_dim"], value: Float[Array, "batch kv_num_heads kv_len vhead_dim"], softmax_aux: Float[Array, num_sinks] | None = None, bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None, q_segment_ids: Int[Array, "batch seq_len"] | None = None, kv_segment_ids: Int[Array, "batch kv_len"] | None = None, q_positions: Int[Array, "batch seq_len"] | None = None, kv_positions: Int[Array, "batch kv_len"] | None = None, sequence_parallelism_mesh_axis_name: str | None = None, logits_soft_cap: float | None = None, qkv_layouts: tuple[SparseMask] | None = None, softmax_scale: float | None = None, mask_builder: (typing.Callable[[int, int, int, int, int], Mask] | typing.Callable[[], SparseMask] | None) = None, sliding_window: int | tuple[int, int] | None = None, chunk_size: int | None = None, causal: bool = True, fused_backward: bool = False, platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, attention_mask: (Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | Int[Array, "batch num_heads_or_1 seq_len kv_len"] | None) = None, cfg: BlockSparseAttentionConfig)` — [`L265`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L265) — Execute block-sparse attention with the given configuration.
  - `smem_est_bytes(qb: int, kb: int, num_stages: int)` — [`L521`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L521) — Estimate shared memory usage in bytes for given block sizes and stages.
  - `win_span(sw)` — [`L666`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L666) — Compute total window span from a sliding window specification.
  - `window_total(sw)` — [`L504`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L504) — Compute total window span from a sliding window specification.
  - `candidate_cfgs_shard_map_gpu` — [`L849`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L849)
  - `candidate_cfgs_shard_map_tpu` — [`L850`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L850)
  - `candidate_cfgs_shard_map_xla` — [`L851`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L851)
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`FwdParams`](../../ops/utils/datacarrier.md#FwdParams), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`BwdParams`](../../ops/utils/datacarrier.md#BwdParams), [`Mask`](../../kernels/_pallas/tpu/blocksparse_attention/_masks.md#Mask), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`kv_blocksize`](../../ops/utils/datacarrier.md#FwdParams.kv_blocksize), [`BlockSparseAttentionConfig`](configs.md#BlockSparseAttentionConfig), [`kv_blocksize`](../../ops/utils/datacarrier.md#BwdParams.kv_blocksize), [`q_blocksize`](../../ops/utils/datacarrier.md#BwdParams.q_blocksize), [`q_blocksize`](../../ops/utils/datacarrier.md#FwdParams.q_blocksize), [`num_stages`](../../ops/utils/datacarrier.md#FwdParams.num_stages), [`num_warps`](../../ops/utils/datacarrier.md#FwdParams.num_warps), [`num_stages`](../../ops/utils/datacarrier.md#BwdParams.num_stages), [`num_warps`](../../ops/utils/datacarrier.md#BwdParams.num_warps), [`bwd_params`](configs.md#BlockSparseAttentionConfig.bwd_params), [`fwd_params`](configs.md#BlockSparseAttentionConfig.fwd_params), [`SparseMask`](../../kernels/_triton/blocksparse_attention/_mask.md#SparseMask)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`create_shard_map_wrapper`](../../ops/core/kernel.md#Kernel.create_shard_map_wrapper), [`blocksparse_attention`](blocksparse_attention.md#blocksparse_attention)

## Functions
- `blocksparse_attention(query: Float[Array, "batch num_heads seq_len head_dim"], key: Float[Array, "batch kv_num_heads kv_len head_dim"], value: Float[Array, "batch kv_num_heads kv_len vhead_dim"], softmax_aux: Float[Array, num_sinks] | None = None, bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None, *, mask_info: MaskInfo | None = None, attention_mask: (Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | Int[Array, "batch num_heads_or_1 seq_len kv_len"] | None) = None, sequence_parallelism_mesh_axis_name: str | None = None, logits_soft_cap: float | None = None, qkv_layouts: tuple[SparseMask] | None = None, softmax_scale: float | None = None, mask_builder: typing.Callable[[int, int, int, int, int], Mask] | typing.Callable[[], SparseMask] | None = None, sliding_window: int | tuple[int, int] | None = None, chunk_size: int | None = None, causal: bool = True, fused_backward: bool = False, purify: bool = False, platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: BlockSparseAttentionConfig | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec | None, ...] | None = None, out_specs: PartitionSpec | None = None)` — [`L868`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L868) — Execute block-sparse attention with automatic optimization.

## Module values
- `_executor` — [`L854`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/blocksparse_attention.py#L854)

