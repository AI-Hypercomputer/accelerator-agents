---
title: 'Module: ejkernel/modules/operations/ragged_decode_attention.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/ragged_decode_attention.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.ragged_decode_attention`/
symbols:
  RaggedDecodeAttention.try_add: RaggedDecodeAttention#try_add().
  RaggedDecodeAttention.candidate_cfgs: RaggedDecodeAttention#candidate_cfgs().
  _ragged_decode_attention_executor._ragged_decode_attention_executor: _ragged_decode_attention_executor._ragged_decode_attention_executor.
  RaggedDecodeAttention.run: RaggedDecodeAttention#run().
  RaggedDecodeAttention._fwd_params_for_seq_len: RaggedDecodeAttention#_fwd_params_for_seq_len().
  RaggedDecodeAttention._default_fwd_params: RaggedDecodeAttention#_default_fwd_params().
  RaggedDecodeAttention.get_impl: RaggedDecodeAttention#get_impl().
  RaggedDecodeAttention.heuristic_cfg: RaggedDecodeAttention#heuristic_cfg().
  RaggedDecodeAttention.candidate_cfgs_gpu: RaggedDecodeAttention#candidate_cfgs_gpu().
  RaggedDecodeAttention.candidate_cfgs_tpu: RaggedDecodeAttention#candidate_cfgs_tpu().
  ragged_decode_attention: ragged_decode_attention().
  RaggedDecodeAttention: RaggedDecodeAttention#
  RaggedDecodeAttention._wrapper: RaggedDecodeAttention#_wrapper().
  RaggedDecodeAttention.create_shard_map_wrapper: RaggedDecodeAttention#create_shard_map_wrapper().
  RaggedDecodeAttention.best_splits: RaggedDecodeAttention#best_splits().
  RaggedDecodeAttention.__init__: RaggedDecodeAttention#__init__().
  RaggedDecodeAttention.score: RaggedDecodeAttention#score().
  RaggedDecodeAttention.next_pow2_ge: RaggedDecodeAttention#next_pow2_ge().
  RaggedDecodeAttention.smem_est_bytes: RaggedDecodeAttention#smem_est_bytes().
  RaggedDecodeAttention.warp_options: RaggedDecodeAttention#warp_options().
  RaggedDecodeAttention.stage_options: RaggedDecodeAttention#stage_options().
---
# Module: [`ejkernel/modules/operations/ragged_decode_attention.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_decode_attention.py)

## Classes
### `RaggedDecodeAttention`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/ragged_decode_attention.py:75`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_decode_attention.py#L75)
- doc: Ragged Decode Attention with custom optimization logic.
- signature: `class RaggedDecodeAttention(Kernel[RaggedDecodeAttentionConfig, Array]):`
- members:
  - `__init__(self)` — [`L95`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_decode_attention.py#L95) — Initialize Ragged Decode Attention module.
  - `_default_fwd_params(self, inv: Invocation[RaggedDecodeAttentionConfig, Array])` — [`L141`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_decode_attention.py#L141) — Build a valid deterministic launch shape for the current sequence length.
  - `_fwd_params_for_seq_len(self, seq_len: int)` — [`L118`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_decode_attention.py#L118) — Build a valid deterministic launch shape for a sequence length.
  - `_wrapper(query, key, value, sequence_start, sequence_end, softmax_aux)` — [`L519`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_decode_attention.py#L519) — Shard-map compatible wrapper that delegates to self.run with captured params.
  - `best_splits(n: int, targets=preferred_split_lens, min_len=32, max_len=8192)` — [`L294`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_decode_attention.py#L294) — Find divisor-based split candidates ranked by proximity to preferred lengths.
  - `candidate_cfgs(self, inv: Invocation[RaggedDecodeAttentionConfig, Array])` — [`L416`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_decode_attention.py#L416) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[RaggedDecodeAttentionConfig, Array])` — [`L240`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_decode_attention.py#L240) — Generate candidate configurations for autotuning on GPU (Pallas backend).
  - `candidate_cfgs_tpu(self, inv: Invocation[RaggedDecodeAttentionConfig, Array])` — [`L458`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_decode_attention.py#L458) — Generate TPU candidates for Pallas and XLA ragged decode attention.
  - `create_shard_map_wrapper(self, query: Float[Array, "batch num_q_heads head_dim"], key: Float[Array, "batch seq_len num_kv_heads head_dim"], value: Float[Array, "batch seq_len num_kv_heads head_dim"], sequence_start: Int[Array, batch], sequence_end: Int[Array, batch], softmax_scale: float | None = None, sliding_window: tuple[int, int] | None = None, logits_soft_cap: float | None = None, softmax_aux: Float[Array, num_sinks] | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, cfg: RaggedDecodeAttentionConfig | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec, ...] | None = None, out_specs: PartitionSpec | None = None, check_vma: bool = False)` — [`L470`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_decode_attention.py#L470) — Create a shard_map wrapper for distributed execution.
  - `get_impl(self, cfg: RaggedDecodeAttentionConfig)` — [`L103`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_decode_attention.py#L103) — Get kernel implementation from registry.
  - `heuristic_cfg(self, inv: Invocation[RaggedDecodeAttentionConfig, Array])` — [`L228`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_decode_attention.py#L228) — Provide default configuration optimized for decode attention.
  - `next_pow2_ge(x, min_val=16)` — [`L324`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_decode_attention.py#L324) — Return the smallest power of 2 >= x, clamped to min_val.
  - `run(self, query: Float[Array, "batch num_q_heads head_dim"], key: Float[Array, "batch seq_len num_kv_heads head_dim"], value: Float[Array, "batch seq_len num_kv_heads head_dim"], sequence_start: Int[Array, batch], sequence_end: Int[Array, batch], softmax_scale: float | None = None, sliding_window: tuple[int, int] | None = None, logits_soft_cap: float | None = None, softmax_aux: Float[Array, num_sinks] | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, cfg: RaggedDecodeAttentionConfig)` — [`L147`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_decode_attention.py#L147) — Execute ragged decode attention with variable-length sequences.
  - `score(sl)` — [`L308`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_decode_attention.py#L308) — Score a split length by minimum distance to any preferred target.
  - `smem_est_bytes(block_heads: int, block_k: int, num_stages: int)` — [`L330`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_decode_attention.py#L330) — Estimate shared memory usage in bytes for a given tile configuration.
  - `stage_options(block_k: int)` — [`L345`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_decode_attention.py#L345) — Return valid pipeline stage counts based on block_k size.
  - `try_add(H, K, s, sl)` — [`L360`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_decode_attention.py#L360) — Try adding configs for the given block_heads, block_k, splits, and split_len.
  - `warp_options(block_heads: int, block_k: int)` — [`L338`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_decode_attention.py#L338) — Return valid warp counts based on block sizes and head dimension.
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`FwdParams`](../../ops/utils/datacarrier.md#FwdParams), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`kv_blocksize`](../../ops/utils/datacarrier.md#FwdParams.kv_blocksize), [`args`](../../ops/core/kernel.md#Invocation.args), [`num_stages`](../../ops/utils/datacarrier.md#FwdParams.num_stages), [`num_warps`](../../ops/utils/datacarrier.md#FwdParams.num_warps), [`RaggedDecodeAttentionConfig`](configs.md#RaggedDecodeAttentionConfig), [`fwd_params`](configs.md#RaggedDecodeAttentionConfig.fwd_params), [`blocksize_heads`](../../ops/utils/datacarrier.md#FwdParams.blocksize_heads), [`num_key_splits`](../../ops/utils/datacarrier.md#FwdParams.num_key_splits)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`create_shard_map_wrapper`](../../ops/core/kernel.md#Kernel.create_shard_map_wrapper), [`ragged_decode_attention`](ragged_decode_attention.md#ragged_decode_attention)

## Functions
- `ragged_decode_attention(query: Float[Array, "batch num_q_heads head_dim"] | Float[Array, "batch 1 num_q_heads head_dim"], key: Float[Array, "batch seq_len num_kv_heads head_dim"], value: Float[Array, "batch seq_len num_kv_heads head_dim"], sequence_start: Int[Array, batch], sequence_end: Int[Array, batch], softmax_aux: Float[Array, num_sinks] | None = None,, *, softmax_scale: float | None = None, sliding_window: tuple[int, int] | None = None, logits_soft_cap: float | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RaggedDecodeAttentionConfig | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec | None, ...] | None = None, out_specs: PartitionSpec | None = None)` — [`L576`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_decode_attention.py#L576) — Execute ragged decode attention with automatic optimization.

## Module values
- `_ragged_decode_attention_executor` — [`L562`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ragged_decode_attention.py#L562)

