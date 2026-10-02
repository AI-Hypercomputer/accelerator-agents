---
title: 'Module: ejkernel/modules/operations/ring_attention.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/ring_attention.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.ring_attention`/
symbols:
  RingAttention.heuristic_cfg: RingAttention#heuristic_cfg().
  RingAttention.candidate_cfgs_gpu: RingAttention#candidate_cfgs_gpu().
  RingAttention.candidate_cfgs: RingAttention#candidate_cfgs().
  RingAttention.candidate_cfgs_tpu: RingAttention#candidate_cfgs_tpu().
  ring_attention: ring_attention().
  _ring_executor._ring_executor: _ring_executor._ring_executor.
  RingAttention.run: RingAttention#run().
  RingAttention.get_impl: RingAttention#get_impl().
  RingAttention: RingAttention#
  RingAttention.create_shard_map_wrapper: RingAttention#create_shard_map_wrapper().
  RingAttention._wrapped_ring_attn: RingAttention#_wrapped_ring_attn().
  RingAttention.__init__: RingAttention#__init__().
  RingAttention.candidate_cfgs_shard_map_tpu: RingAttention#candidate_cfgs_shard_map_tpu.
  RingAttention._largest_pow2_divisor: RingAttention#_largest_pow2_divisor().
  RingAttention.version: RingAttention#version.
---
# Module: [`ejkernel/modules/operations/ring_attention.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ring_attention.py)

## Classes
### `RingAttention`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/ring_attention.py:96`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ring_attention.py#L96)
- doc: Ring Attention with custom optimization logic.
- signature: `class RingAttention(Kernel[RingAttentionConfig, Array]):`
- members:
  - `__init__(self)` — [`L125`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ring_attention.py#L125) — Initialize Ring Attention module.
  - `_largest_pow2_divisor(n: int, *, max_block: int = 512)` — [`L376`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ring_attention.py#L376) — Return the largest power-of-2 divisor of n that is <= max_block.
  - `_wrapped_ring_attn(query: Float[Array, "batch seq_len_q num_heads head_dim"], key: Float[Array, "batch seq_len_k num_kv_heads head_dim"], value: Float[Array, "batch seq_len_k num_kv_heads head_dim"], softmax_aux: Float[Array, num_sinks] | None = None, bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None, q_segment_ids: Int[Array, "batch seq_len_q"] | None = None, kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None, q_position_ids: Int[Array, "batch seq_len_q"] | None = None, kv_position_ids: Int[Array, "batch seq_len_k"] | None = None)` — [`L194`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ring_attention.py#L194) — Shard-map compatible wrapper that delegates to self.run with captured params.
  - `candidate_cfgs(self, inv: Invocation[RingAttentionConfig, Array])` — [`L393`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ring_attention.py#L393) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[RingAttentionConfig, Array])` — [`L422`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ring_attention.py#L422) — Generate GPU candidates for Triton, TileLang, and XLA ring attention.
  - `candidate_cfgs_tpu(self, inv: Invocation[RingAttentionConfig, Array])` — [`L467`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ring_attention.py#L467) — Generate TPU-optimized candidate configurations for autotuning.
  - `create_shard_map_wrapper(self, query: Float[Array, "batch seq_len_q num_heads head_dim"], key: Float[Array, "batch seq_len_k num_kv_heads head_dim"], value: Float[Array, "batch seq_len_k num_kv_heads head_dim"], q_segment_ids: Int[Array, "batch seq_len_q"] | None = None, kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None, q_position_ids: Int[Array, "batch seq_len_q"] | None = None, kv_position_ids: Int[Array, "batch seq_len_k"] | None = None, softmax_aux: Float[Array, num_sinks] | None = None, bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None, mask_builder: Callable[[int, int, int, int, int], Mask] | None = None, sliding_window: int | tuple[int, int] | None = None, chunk_size: int | None = None, causal: bool = False, logits_soft_cap: float | None = None, softmax_scale: float | None = None, axis_name: str | None = None, fused_backward: bool = False, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RingAttentionConfig | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec, ...] | None = None, out_specs: PartitionSpec | None = None, check_vma: bool = False)` — [`L133`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ring_attention.py#L133) — Create a shard_map wrapper specifically for ring attention.
  - `get_impl(self, cfg: RingAttentionConfig)` — [`L250`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ring_attention.py#L250) — Get kernel implementation from registry.
  - `heuristic_cfg(self, inv: Invocation[RingAttentionConfig, Array])` — [`L359`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ring_attention.py#L359) — Provide default configuration optimized for ring attention.
  - `run(self, query: Float[Array, "batch seq_len_q num_heads head_dim"], key: Float[Array, "batch seq_len_k num_kv_heads head_dim"], value: Float[Array, "batch seq_len_k num_kv_heads head_dim"], q_segment_ids: Int[Array, "batch seq_len_q"] | None = None, kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None, q_position_ids: Int[Array, "batch seq_len_q"] | None = None, kv_position_ids: Int[Array, "batch seq_len_k"] | None = None, softmax_aux: Float[Array, num_sinks] | None = None, bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None, mask_builder: Callable[[int, int, int, int, int], Mask] | None = None, sliding_window: int | tuple[int, int] | None = None, chunk_size: int | None = None, causal: bool = False, logits_soft_cap: float | None = None, softmax_scale: float | None = None, axis_name: str | None = None, fused_backward: bool = False, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, cfg: RingAttentionConfig)` — [`L265`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ring_attention.py#L265) — Execute ring attention with distributed KV processing.
  - `candidate_cfgs_shard_map_tpu` — [`L500`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ring_attention.py#L500)
  - `version` — [`L123`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ring_attention.py#L123)
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`FwdParams`](../../ops/utils/datacarrier.md#FwdParams), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`BwdParams`](../../ops/utils/datacarrier.md#BwdParams), [`Mask`](../../kernels/_pallas/tpu/blocksparse_attention/_masks.md#Mask), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`kv_blocksize`](../../ops/utils/datacarrier.md#FwdParams.kv_blocksize), [`kv_blocksize`](../../ops/utils/datacarrier.md#BwdParams.kv_blocksize), [`q_blocksize`](../../ops/utils/datacarrier.md#BwdParams.q_blocksize), [`q_blocksize`](../../ops/utils/datacarrier.md#FwdParams.q_blocksize), [`num_stages`](../../ops/utils/datacarrier.md#FwdParams.num_stages), [`num_warps`](../../ops/utils/datacarrier.md#FwdParams.num_warps), [`RingAttentionConfig`](configs.md#RingAttentionConfig), [`num_stages`](../../ops/utils/datacarrier.md#BwdParams.num_stages), [`num_warps`](../../ops/utils/datacarrier.md#BwdParams.num_warps), [`bwd_params`](configs.md#RingAttentionConfig.bwd_params), [`fwd_params`](configs.md#RingAttentionConfig.fwd_params)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`create_shard_map_wrapper`](../../ops/core/kernel.md#Kernel.create_shard_map_wrapper), [`ring_attention`](ring_attention.md#ring_attention)

## Functions
- `ring_attention(query: Float[Array, "batch seq_len_q num_heads head_dim"], key: Float[Array, "batch seq_len_k num_kv_heads head_dim"], value: Float[Array, "batch seq_len_k num_kv_heads head_dim"], softmax_aux: Float[Array, num_sinks] | None = None, bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,, *, mask_info: MaskInfo | None = None, mask_builder: Callable[[int, int, int, int, int], Mask] | None = None, sliding_window: int | tuple[int, int] | None = None, chunk_size: int | None = None, causal: bool = False, logits_soft_cap: float | None = None, softmax_scale: float | None = None, axis_name: str | None = None, fused_backward: bool = False, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RingAttentionConfig | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec | None, ...] | None = None, out_specs: PartitionSpec | None = None)` — [`L517`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ring_attention.py#L517) — Execute ring attention with automatic optimization.

## Module values
- `_ring_executor` — [`L503`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/ring_attention.py#L503)

