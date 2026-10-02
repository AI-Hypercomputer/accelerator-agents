---
title: 'Module: ejkernel/modules/operations/gated_linear_attention.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/gated_linear_attention.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.gated_linear_attention`/
symbols:
  GLAttention.run: GLAttention#run().
  GLAttention.candidate_cfgs_gpu: GLAttention#candidate_cfgs_gpu().
  _gla_executor._gla_executor: _gla_executor._gla_executor.
  GLAttention.candidate_cfgs: GLAttention#candidate_cfgs().
  GLAttention.heuristic_cfg: GLAttention#heuristic_cfg().
  GLAttention.candidate_cfgs_tpu: GLAttention#candidate_cfgs_tpu().
  GLAttention.get_impl: GLAttention#get_impl().
  gla_attention: gla_attention().
  GLAttention: GLAttention#
  GLAttention.__init__: GLAttention#__init__().
---
# Module: [`ejkernel/modules/operations/gated_linear_attention.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_linear_attention.py)

## Classes
### `GLAttention`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/gated_linear_attention.py:90`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_linear_attention.py#L90)
- doc: Gated Linear Attention with custom optimization logic.
- signature: `class GLAttention(Kernel[GLAttentionConfig, Array]):`
- members:
  - `__init__(self)` — [`L130`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_linear_attention.py#L130) — Initialize GLA module.
  - `candidate_cfgs(self, inv: Invocation[GLAttentionConfig, Array])` — [`L262`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_linear_attention.py#L262) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[GLAttentionConfig, Array])` — [`L293`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_linear_attention.py#L293) — Generate GPU candidates for GLA.
  - `candidate_cfgs_tpu(self, inv: Invocation[GLAttentionConfig, Array])` — [`L356`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_linear_attention.py#L356) — Generate TPU candidates for the XLA GLA path.
  - `get_impl(self, cfg: GLAttentionConfig)` — [`L138`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_linear_attention.py#L138) — Get kernel implementation from registry.
  - `heuristic_cfg(self, inv: Invocation[GLAttentionConfig, Array])` — [`L242`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_linear_attention.py#L242) — Provide default configuration with block sizes.
  - `run(self, query: Float[Array, "batch seq_len num_heads qk_head_dim"], key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"], value: Float[Array, "batch seq_len num_kv_heads v_head_dim"], g: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None, g_gamma: Float[Array, "... num_heads"] | None = None, softmax_scale: float | None = None, initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None, reverse: bool = False, cu_seqlens: Int[Array, num_seqs_plus_one] | None = None, return_state: bool = False, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, cfg: GLAttentionConfig)` — [`L153`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_linear_attention.py#L153) — Execute gated linear attention computation.
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`Platform`](../../kernels/_registry.md#Platform), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`GLAttentionConfig`](configs.md#GLAttentionConfig), [`TRITON`](../../kernels/_registry.md#Platform.TRITON), [`block_d`](configs.md#GLAttentionConfig.block_d), [`block_k`](configs.md#GLAttentionConfig.block_k), [`num_stages`](configs.md#GLAttentionConfig.num_stages), [`num_warps`](configs.md#GLAttentionConfig.num_warps), [`block_q`](configs.md#GLAttentionConfig.block_q)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`gla_attention`](gated_linear_attention.md#gla_attention)

## Functions
- `gla_attention(query: Float[Array, "batch seq_len num_heads qk_head_dim"], key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"], value: Float[Array, "batch seq_len num_kv_heads v_head_dim"], g: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None, g_gamma: Float[Array, "... num_heads"] | None = None, initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None, cu_seqlens: Int[Array, num_seqs_plus_one] | None = None,, *, softmax_scale: float | None = None, reverse: bool = False, return_state: bool = False, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: GLAttentionConfig | None = None)` — [`L385`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_linear_attention.py#L385) — Execute gated linear attention with automatic optimization.

## Module values
- `_gla_executor` — [`L371`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/gated_linear_attention.py#L371)

