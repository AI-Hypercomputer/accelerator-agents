---
title: 'Module: ejkernel/modules/operations/decode_attention.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/decode_attention.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.decode_attention`/
symbols:
  _decode_attention_executor._decode_attention_executor: _decode_attention_executor._decode_attention_executor.
  DecodeAttention.candidate_cfgs_gpu: DecodeAttention#candidate_cfgs_gpu().
  DecodeAttention.heuristic_cfg: DecodeAttention#heuristic_cfg().
  DecodeAttention.run: DecodeAttention#run().
  DecodeAttention.candidate_cfgs: DecodeAttention#candidate_cfgs().
  DecodeAttention.candidate_cfgs_tpu: DecodeAttention#candidate_cfgs_tpu().
  DecodeAttention.get_impl: DecodeAttention#get_impl().
  decode_attention: decode_attention().
  DecodeAttention: DecodeAttention#
  DecodeAttention._wrapped_decode_attention: DecodeAttention#_wrapped_decode_attention().
  DecodeAttention.create_shard_map_wrapper: DecodeAttention#create_shard_map_wrapper().
  DecodeAttention.__init__: DecodeAttention#__init__().
---
# Module: [`ejkernel/modules/operations/decode_attention.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/decode_attention.py)

## Classes
### `DecodeAttention`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/decode_attention.py:86`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/decode_attention.py#L86)
- doc: vLLM-style paged decode attention returning both output and LSE.
- signature: `class DecodeAttention(Kernel[DecodeAttentionConfig, tuple[Array, Array]]):`
- members:
  - `__init__(self)` — [`L95`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/decode_attention.py#L95) — Initialize the decode attention kernel.
  - `_wrapped_decode_attention(query: Float[Array, "batch num_q_heads head_dim"], key_buffer: Float[Array, "total_tokens num_kv_heads head_dim"], value_buffer: Float[Array, "total_tokens num_kv_heads head_dim"], req_to_tokens: Int32[Array, "batch max_pages"], seq_lens: Int32[Array, batch])` — [`L148`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/decode_attention.py#L148) — Shard-local decode attention forwarding to self.run.
  - `candidate_cfgs(self, inv: Invocation[DecodeAttentionConfig, tuple[Array, Array]])` — [`L291`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/decode_attention.py#L291) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[DecodeAttentionConfig, tuple[Array, Array]])` — [`L318`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/decode_attention.py#L318) — Generate GPU decode-attention candidates.
  - `candidate_cfgs_tpu(self, inv: Invocation[DecodeAttentionConfig, tuple[Array, Array]])` — [`L374`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/decode_attention.py#L374) — Return TPU candidates for the XLA fallback path.
  - `create_shard_map_wrapper(self, query: Float[Array, "batch num_q_heads head_dim"], key_buffer: Float[Array, "total_tokens num_kv_heads head_dim"], value_buffer: Float[Array, "total_tokens num_kv_heads head_dim"], req_to_tokens: Int32[Array, "batch max_pages"], seq_lens: Int32[Array, batch], *, softmax_scale: float | None = None, num_kv_splits: int | None = None, page_size: int = 1, logits_soft_cap: float | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: DecodeAttentionConfig | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec, ...] | None = None, out_specs: tuple[PartitionSpec, PartitionSpec] | None = None, check_vma: bool = False)` — [`L99`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/decode_attention.py#L99) — Create a shard_map wrapper for distributed execution.
  - `get_impl(self, cfg: DecodeAttentionConfig)` — [`L184`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/decode_attention.py#L184) — Get the platform-specific kernel implementation.
  - `heuristic_cfg(self, inv: Invocation[DecodeAttentionConfig, tuple[Array, Array]])` — [`L261`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/decode_attention.py#L261) — Generate default heuristic configuration.
  - `run(self, query: Float[Array, "batch num_q_heads head_dim"], key_buffer: Float[Array, "total_tokens num_kv_heads head_dim"], value_buffer: Float[Array, "total_tokens num_kv_heads head_dim"], req_to_tokens: Int32[Array, "batch max_pages"], seq_lens: Int32[Array, batch], *, softmax_scale: float | None = None, num_kv_splits: int | None = None, page_size: int = 1, logits_soft_cap: float | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: DecodeAttentionConfig)` — [`L196`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/decode_attention.py#L196) — Execute paged decode attention.
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`DecodeAttentionConfig`](configs.md#DecodeAttentionConfig), [`num_kv_splits`](configs.md#DecodeAttentionConfig.num_kv_splits), [`num_stages`](configs.md#DecodeAttentionConfig.num_stages), [`num_warps`](configs.md#DecodeAttentionConfig.num_warps)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`create_shard_map_wrapper`](../../ops/core/kernel.md#Kernel.create_shard_map_wrapper), [`decode_attention`](decode_attention.md#decode_attention)

## Functions
- `decode_attention(query: Float[Array, "batch num_q_heads head_dim"], key_buffer: Float[Array, "total_tokens num_kv_heads head_dim"], value_buffer: Float[Array, "total_tokens num_kv_heads head_dim"], req_to_tokens: Int32[Array, "batch max_pages"], seq_lens: Int32[Array, batch],, *, softmax_scale: float | None = None, num_kv_splits: int | None = None, page_size: int = 1, logits_soft_cap: float | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, mesh: Mesh | None = None, in_specs: tuple[PartitionSpec | None, ...] | None = None, out_specs: tuple[PartitionSpec, PartitionSpec] | None = None, cfg: DecodeAttentionConfig | None = None)` — [`L402`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/decode_attention.py#L402) — Execute vLLM-style paged decode attention.

## Module values
- `_decode_attention_executor` — [`L388`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/decode_attention.py#L388)

