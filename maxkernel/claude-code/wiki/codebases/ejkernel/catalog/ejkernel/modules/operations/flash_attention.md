---
title: 'Module: ejkernel/modules/operations/flash_attention.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/flash_attention.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.flash_attention`/
symbols:
  FlashAttention.candidate_cfgs_tpu: FlashAttention#candidate_cfgs_tpu().
  FlashAttention.candidate_cfgs_xla: FlashAttention#candidate_cfgs_xla().
  FlashAttention.candidate_cfgs_gpu: FlashAttention#candidate_cfgs_gpu().
  FlashAttention.heuristic_cfg_gpu: FlashAttention#heuristic_cfg_gpu().
  flash_attention: flash_attention().
  FlashAttention.heuristic_cfg_tpu: FlashAttention#heuristic_cfg_tpu().
  FlashAttention.heuristic_cfg: FlashAttention#heuristic_cfg().
  FlashAttention.candidate_cfgs: FlashAttention#candidate_cfgs().
  FlashAttention.run: FlashAttention#run().
  _flash_executor._flash_executor: _flash_executor._flash_executor.
  FlashAttention.get_impl: FlashAttention#get_impl().
  FlashAttention.create_shard_map_wrapper: FlashAttention#create_shard_map_wrapper().
  FlashAttention: FlashAttention#
  PagedKV: PagedKV.
  DenseKV: DenseKV.
  FlashAttention._wraped_flash_attn: FlashAttention#_wraped_flash_attn().
  BlockTables: BlockTables.
  FlashAttention.bwd_block: FlashAttention#bwd_block().
  FlashAttention.round128: FlashAttention#round128().
  FlashAttention.bwd_tile: FlashAttention#bwd_tile().
  FlashAttention.__init__: FlashAttention#__init__().
  FlashAttention.candidate_cfgs_shard_map_gpu: FlashAttention#candidate_cfgs_shard_map_gpu.
  FlashAttention.candidate_cfgs_shard_map_tpu: FlashAttention#candidate_cfgs_shard_map_tpu.
  FlashAttention.candidate_cfgs_shard_map_xla: FlashAttention#candidate_cfgs_shard_map_xla.
  FlashAttention.pick_warps_stages: FlashAttention#pick_warps_stages().
  FlashAttention.win_span: FlashAttention#win_span().
  FlashAttention.window_total: FlashAttention#window_total().
  FlashAttention.next_pow2_ge: FlashAttention#next_pow2_ge().
  FlashAttention.smem_est_bytes: FlashAttention#smem_est_bytes().
  FlashAttention.version: FlashAttention#version.
---
# Module: [`ejkernel/modules/operations/flash_attention.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py)

## Classes
### `FlashAttention`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/flash_attention.py:85`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L85)
- doc: Flash Attention with custom optimization logic.
- signature: `class FlashAttention(Kernel[FlashAttentionConfig, Array]):`
- members:
  - `__init__(self)` — [`L123`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L123) — Initialize Flash Attention module.
  - `_wraped_flash_attn(query: Float[Array, "batch num_heads seq_len head_dim"], key: Float[Array, "batch kv_num_heads kv_len head_dim"], value: Float[Array, "batch kv_num_heads kv_len vhead_dim"], bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None, softmax_aux: Float[Array, num_sinks] | None = None, cum_seqlens_q: Int[Array, batch_plus_one] | None = None, cum_seqlens_k: Int[Array, batch_plus_one] | None = None, attention_mask: Int[Array, "batch num_heads seq_len kv_len"] | None = None, block_tables: BlockTables | None = None, q_segment_ids: Int[Array, "batch seq_len_q"] | None = None, kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None)` — [`L176`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L176) — Shard-local flash attention forwarding to self.run.
  - `bwd_block(x: int, cap: int = 128)` — [`L568`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L568) — Compute backward block size from a forward block size.
  - `bwd_tile(_x: int)` — [`L681`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L681) — Return a fixed backward tile size of 128.
  - `candidate_cfgs(self, inv: Invocation[FlashAttentionConfig, Array])` — [`L444`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L444) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[FlashAttentionConfig, Array])` — [`L481`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L481) — Generate GPU-optimized candidate configurations for autotuning (Triton).
  - `candidate_cfgs_tpu(self, inv: Invocation[FlashAttentionConfig, Array])` — [`L635`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L635) — Generate TPU-optimized candidate configurations for autotuning (Pallas).
  - `candidate_cfgs_xla(self, inv: Invocation[FlashAttentionConfig, Array])` — [`L723`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L723) — Generate XLA-optimized candidate configurations for autotuning.
  - `create_shard_map_wrapper(self, query: Float[Array, "batch seq_len_q num_heads head_dim"], key: DenseKV | PagedKV, value: DenseKV | PagedKV, attention_mask: (Bool[Array, "batch num_heads_or_1 seq_len_q seq_len_k"] | Int[Array, "batch num_heads_or_1 seq_len_q seq_len_k"] | None) = None, bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None, softmax_scale: float | None = None, dropout_prob: float = 0, causal: bool = False, dropout_seed: int | None = None, cum_seqlens_q: Int[Array, batch_plus_one] | None = None, cum_seqlens_k: Int[Array, batch_plus_one] | None = None, block_tables: BlockTables | None = None, sliding_window: int | tuple[int, int] | None = None, logits_soft_cap: float | None = None, softmax_aux: Float[Array, num_sinks] | None = None, normalize_output: bool = True, precision: lax.PrecisionLike = lax.Precision.DEFAULT, logits_dtype: DTypeLike = jnp.float32, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, q_segment_ids: Int[Array, "batch seq_len_q"] | None = None, kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None, cfg: FlashAttentionConfig | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec, ...] | None = None, out_specs: PartitionSpec | None = None, check_vma: bool = False)` — [`L127`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L127) — Create a shard_map wrapper specifically for flash attention.
  - `get_impl(self, cfg: FlashAttentionConfig)` — [`L239`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L239) — Get kernel implementation from registry based on configuration.
  - `heuristic_cfg(self, inv: Invocation[FlashAttentionConfig, Array])` — [`L415`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L415) — Provide default configuration based on invocation context.
  - `heuristic_cfg_gpu(self, inv: Invocation[FlashAttentionConfig, Array])` — [`L356`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L356) — Provide default configuration based on invocation context.
  - `heuristic_cfg_tpu(self, inv: Invocation[FlashAttentionConfig, Array])` — [`L386`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L386) — Provide default configuration based on invocation context.
  - `next_pow2_ge(x: int, min_val: int = 16)` — [`L519`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L519) — Return the smallest power of two >= x, with a minimum of min_val.
  - `pick_warps_stages(qb: int, kb: int, dh: int)` — [`L556`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L556) — Select num_warps and num_stages based on block sizes and head dim.
  - `round128(x: int | float)` — [`L663`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L663) — Round x up to the nearest multiple of 128.
  - `run(self, query: Float[Array, "batch seq_len_q num_heads head_dim"], key: DenseKV | PagedKV, value: DenseKV | PagedKV, attention_mask: (Bool[Array, "batch num_heads_or_1 seq_len_q seq_len_k"] | Int[Array, "batch num_heads_or_1 seq_len_q seq_len_k"] | None) = None, bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None, softmax_scale: float | None = None, dropout_prob: float = 0, causal: bool = False, dropout_seed: int | None = None, cum_seqlens_q: Int[Array, batch_plus_one] | None = None, cum_seqlens_k: Int[Array, batch_plus_one] | None = None, block_tables: BlockTables | None = None, sliding_window: int | tuple[int, int] | None = None, logits_soft_cap: float | None = None, softmax_aux: Float[Array, num_sinks] | None = None, normalize_output: bool = True, precision: lax.PrecisionLike = lax.Precision.DEFAULT, logits_dtype: DTypeLike = jnp.float32, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, q_segment_ids: Int[Array, "batch seq_len_q"] | None = None, kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None, cfg: FlashAttentionConfig)` — [`L261`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L261) — Execute flash attention with the given configuration.
  - `smem_est_bytes(qb: int, kb: int, num_stages: int)` — [`L526`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L526) — Estimate shared memory usage in bytes for given block sizes and stages.
  - `win_span(sw)` — [`L653`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L653) — Compute total window span from a sliding window specification.
  - `window_total(sw)` — [`L505`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L505) — Compute total window span from a sliding window specification.
  - `candidate_cfgs_shard_map_gpu` — [`L810`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L810)
  - `candidate_cfgs_shard_map_tpu` — [`L811`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L811)
  - `candidate_cfgs_shard_map_xla` — [`L812`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L812)
  - `version` — [`L121`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L121)
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`FwdParams`](../../ops/utils/datacarrier.md#FwdParams), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`BwdParams`](../../ops/utils/datacarrier.md#BwdParams), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`kv_blocksize`](../../ops/utils/datacarrier.md#FwdParams.kv_blocksize), [`FlashAttentionConfig`](configs.md#FlashAttentionConfig), [`kv_blocksize`](../../ops/utils/datacarrier.md#BwdParams.kv_blocksize), [`q_blocksize`](../../ops/utils/datacarrier.md#BwdParams.q_blocksize), [`q_blocksize`](../../ops/utils/datacarrier.md#FwdParams.q_blocksize), [`num_stages`](../../ops/utils/datacarrier.md#FwdParams.num_stages), [`num_warps`](../../ops/utils/datacarrier.md#FwdParams.num_warps), [`num_stages`](../../ops/utils/datacarrier.md#BwdParams.num_stages), [`num_warps`](../../ops/utils/datacarrier.md#BwdParams.num_warps), [`bwd_params`](configs.md#FlashAttentionConfig.bwd_params), [`fwd_params`](configs.md#FlashAttentionConfig.fwd_params), [`DenseKV`](flash_attention.md#DenseKV), [`PagedKV`](flash_attention.md#PagedKV), [`BlockTables`](flash_attention.md#BlockTables)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`create_shard_map_wrapper`](../../ops/core/kernel.md#Kernel.create_shard_map_wrapper), [`flash_attention`](flash_attention.md#flash_attention)

## Functions
- `flash_attention(query: Float[Array, "batch seq_len_q num_heads head_dim"], key: DenseKV | PagedKV, value: DenseKV | PagedKV, bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None, cum_seqlens_q: Int[Array, batch_plus_one] | None = None, cum_seqlens_k: Int[Array, batch_plus_one] | None = None, softmax_aux: Float[Array, num_sinks] | None = None, block_tables: BlockTables | None = None,, *, mask_info: MaskInfo | None = None, softmax_scale: float | None = None, dropout_prob: float = 0, causal: bool = False, dropout_seed: int | None = None, sliding_window: int | tuple[int, int] | None = None, logits_soft_cap: float | None = None, normalize_output: bool = True, precision: lax.PrecisionLike = lax.Precision.DEFAULT, logits_dtype: DTypeLike = jnp.float32, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: FlashAttentionConfig | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec | None, ...] | None = None, out_specs: PartitionSpec | None = None)` — [`L829`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L829) — Execute flash attention with automatic optimization.

## Module values
- `BlockTables` — [`L82`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L82)
- `DenseKV` — [`L81`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L81)
- `PagedKV` — [`L80`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L80)
- `_flash_executor` — [`L815`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/flash_attention.py#L815)

