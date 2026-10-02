---
title: 'Module: ejkernel/modules/operations/lightning_attention.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/lightning_attention.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.lightning_attention`/
symbols:
  LightningAttention.run: LightningAttention#run().
  LightningAttention.candidate_cfgs_gpu: LightningAttention#candidate_cfgs_gpu().
  _lightning_executor._lightning_executor: _lightning_executor._lightning_executor.
  LightningAttention.candidate_cfgs: LightningAttention#candidate_cfgs().
  LightningAttention.heuristic_cfg: LightningAttention#heuristic_cfg().
  LightningAttention.candidate_cfgs_tpu: LightningAttention#candidate_cfgs_tpu().
  LightningAttention.get_impl: LightningAttention#get_impl().
  lightning_attention: lightning_attention().
  LightningAttention: LightningAttention#
  LightningAttention.__init__: LightningAttention#__init__().
---
# Module: [`ejkernel/modules/operations/lightning_attention.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/lightning_attention.py)

## Classes
### `LightningAttention`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/lightning_attention.py:93`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/lightning_attention.py#L93)
- doc: Lightning Attention with custom optimization logic.
- signature: `class LightningAttention(Kernel[LightningAttentionConfig, Array]):`
- members:
  - `__init__(self)` — [`L141`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/lightning_attention.py#L141) — Initialize Lightning Attention module.
  - `candidate_cfgs(self, inv: Invocation[LightningAttentionConfig, Array])` — [`L272`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/lightning_attention.py#L272) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[LightningAttentionConfig, Array])` — [`L307`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/lightning_attention.py#L307) — Generate GPU candidates for Lightning Attention.
  - `candidate_cfgs_tpu(self, inv: Invocation[LightningAttentionConfig, Array])` — [`L373`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/lightning_attention.py#L373) — Generate TPU candidates for the XLA Lightning Attention path.
  - `get_impl(self, cfg: LightningAttentionConfig)` — [`L149`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/lightning_attention.py#L149) — Get kernel implementation from registry.
  - `heuristic_cfg(self, inv: Invocation[LightningAttentionConfig, Array])` — [`L252`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/lightning_attention.py#L252) — Provide default configuration with block sizes.
  - `run(self, query: Float[Array, "batch seq_len num_heads qk_head_dim"], key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"], value: Float[Array, "batch seq_len num_kv_heads v_head_dim"], layer_idx: int, num_layers: int, softmax_scale: float | None = None, initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None, reverse: bool = False, cu_seqlens: Int[Array, num_seqs_plus_one] | None = None, return_state: bool = False, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, cfg: LightningAttentionConfig)` — [`L164`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/lightning_attention.py#L164) — Execute lightning attention with layer-specific optimization.
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`Platform`](../../kernels/_registry.md#Platform), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`LightningAttentionConfig`](configs.md#LightningAttentionConfig), [`TRITON`](../../kernels/_registry.md#Platform.TRITON), [`block_d`](configs.md#LightningAttentionConfig.block_d), [`block_k`](configs.md#LightningAttentionConfig.block_k), [`num_stages`](configs.md#LightningAttentionConfig.num_stages), [`num_warps`](configs.md#LightningAttentionConfig.num_warps), [`block_q`](configs.md#LightningAttentionConfig.block_q)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`lightning_attention`](lightning_attention.md#lightning_attention)

## Functions
- `lightning_attention(query: Float[Array, "batch seq_len num_heads qk_head_dim"], key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"], value: Float[Array, "batch seq_len num_kv_heads v_head_dim"], initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None, cu_seqlens: Int[Array, num_seqs_plus_one] | None = None,, *, layer_idx: int, num_layers: int, softmax_scale: float | None = None, reverse: bool = False, return_state: bool = False, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: LightningAttentionConfig | None = None)` — [`L402`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/lightning_attention.py#L402) — Execute lightning attention with automatic optimization.

## Module values
- `_lightning_executor` — [`L388`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/lightning_attention.py#L388)

