---
title: 'Module: ejkernel/modules/operations/recurrent.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/recurrent.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.recurrent`/
symbols:
  RecurrentAttention.run: RecurrentAttention#run().
  RecurrentAttention.candidate_cfgs_gpu: RecurrentAttention#candidate_cfgs_gpu().
  _recurrent_executor._recurrent_executor: _recurrent_executor._recurrent_executor.
  RecurrentAttention.candidate_cfgs: RecurrentAttention#candidate_cfgs().
  RecurrentAttention.heuristic_cfg: RecurrentAttention#heuristic_cfg().
  RecurrentAttention.candidate_cfgs_tpu: RecurrentAttention#candidate_cfgs_tpu().
  RecurrentAttention.get_impl: RecurrentAttention#get_impl().
  recurrent_attention: recurrent_attention().
  RecurrentAttention: RecurrentAttention#
  RecurrentAttention.__init__: RecurrentAttention#__init__().
---
# Module: [`ejkernel/modules/operations/recurrent.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/recurrent.py)

## Classes
### `RecurrentAttention`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/recurrent.py:94`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/recurrent.py#L94)
- doc: Recurrent Attention with custom optimization logic.
- signature: `class RecurrentAttention(Kernel[RecurrentAttentionConfig, Array]):`
- members:
  - `__init__(self)` — [`L137`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/recurrent.py#L137) — Initialize Recurrent Attention module.
  - `candidate_cfgs(self, inv: Invocation[RecurrentAttentionConfig, Array])` — [`L295`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/recurrent.py#L295) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[RecurrentAttentionConfig, Array])` — [`L330`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/recurrent.py#L330) — Generate GPU candidates for recurrent attention across Triton, TileLang and XLA.
  - `candidate_cfgs_tpu(self, inv: Invocation[RecurrentAttentionConfig, Array])` — [`L397`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/recurrent.py#L397) — Generate TPU candidates for the XLA recurrent-attention path.
  - `get_impl(self, cfg: RecurrentAttentionConfig)` — [`L145`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/recurrent.py#L145) — Get kernel implementation from registry.
  - `heuristic_cfg(self, inv: Invocation[RecurrentAttentionConfig, Array])` — [`L275`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/recurrent.py#L275) — Provide default configuration with block sizes.
  - `run(self, query: Float[Array, "batch seq_len num_heads qk_head_dim"], key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"], value: Float[Array, "batch seq_len num_kv_heads v_head_dim"], g: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None, g_gamma: Float[Array, "... num_heads"] | None = None, gk: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None, gv: Float[Array, "batch seq_len num_heads v_head_dim"] | None = None, softmax_scale: float | None = None, initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None, reverse: bool = False, cu_seqlens: Int[Array, num_seqs_plus_one] | None = None, return_state: bool = False, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, cfg: RecurrentAttentionConfig)` — [`L160`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/recurrent.py#L160) — Execute recurrent attention with stateful computation.
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`Platform`](../../kernels/_registry.md#Platform), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`RecurrentAttentionConfig`](configs.md#RecurrentAttentionConfig), [`TRITON`](../../kernels/_registry.md#Platform.TRITON), [`block_d`](configs.md#RecurrentAttentionConfig.block_d), [`block_k`](configs.md#RecurrentAttentionConfig.block_k), [`num_stages`](configs.md#RecurrentAttentionConfig.num_stages), [`num_warps`](configs.md#RecurrentAttentionConfig.num_warps), [`block_q`](configs.md#RecurrentAttentionConfig.block_q)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`recurrent_attention`](recurrent.md#recurrent_attention)

## Functions
- `recurrent_attention(query: Float[Array, "batch seq_len num_heads qk_head_dim"], key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"], value: Float[Array, "batch seq_len num_kv_heads v_head_dim"], g: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None, g_gamma: Float[Array, "... num_heads"] | None = None, gk: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None, gv: Float[Array, "batch seq_len num_heads v_head_dim"] | None = None, initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None, cu_seqlens: Int[Array, num_seqs_plus_one] | None = None,, *, softmax_scale: float | None = None, reverse: bool = False, return_state: bool = False, platform: typing.Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: RecurrentAttentionConfig | None = None)` — [`L426`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/recurrent.py#L426) — Execute recurrent attention with automatic optimization.

## Module values
- `_recurrent_executor` — [`L412`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/recurrent.py#L412)

