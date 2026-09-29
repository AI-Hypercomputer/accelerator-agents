---
title: 'Module: ejkernel/modules/operations/multi_head_latent_attention.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/multi_head_latent_attention.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.multi_head_latent_attention`/
symbols:
  _mla_executor._mla_executor: _mla_executor._mla_executor.
  FlashMLA.run: FlashMLA#run().
  FlashMLA.candidate_cfgs_gpu: FlashMLA#candidate_cfgs_gpu().
  FlashMLA.heuristic_cfg: FlashMLA#heuristic_cfg().
  FlashMLA.candidate_cfgs_tpu: FlashMLA#candidate_cfgs_tpu().
  FlashMLA.get_impl: FlashMLA#get_impl().
  flash_mla: flash_mla().
  FlashMLA.candidate_cfgs: FlashMLA#candidate_cfgs().
  FlashMLA: FlashMLA#
  FlashMLA.__init__: FlashMLA#__init__().
---
# Module: [`ejkernel/modules/operations/multi_head_latent_attention.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_head_latent_attention.py)

## Classes
### `FlashMLA`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/multi_head_latent_attention.py:93`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_head_latent_attention.py#L93)
- doc: Flash Multi-head Latent Attention with custom optimization logic.
- signature: `class FlashMLA(Kernel[FlashMLAConfig, Array]):`
- members:
  - `__init__(self)` — [`L113`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_head_latent_attention.py#L113) — Initialize Flash MLA module.
  - `candidate_cfgs(self, inv: Invocation[FlashMLAConfig, Array])` — [`L238`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_head_latent_attention.py#L238) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[FlashMLAConfig, Array])` — [`L253`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_head_latent_attention.py#L253) — Generate GPU candidates for TileLang and XLA FlashMLA.
  - `candidate_cfgs_tpu(self, inv: Invocation[FlashMLAConfig, Array])` — [`L294`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_head_latent_attention.py#L294) — Generate TPU candidates for Pallas and XLA FlashMLA.
  - `get_impl(self, cfg: FlashMLAConfig)` — [`L121`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_head_latent_attention.py#L121) — Get kernel implementation from registry.
  - `heuristic_cfg(self, inv: Invocation[FlashMLAConfig, Array])` — [`L219`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_head_latent_attention.py#L219) — Provide default configuration with block sizes.
  - `run(self, query: Float[Array, "batch seq_len q_heads q_head_dim"], key_value: Float[Array, "batch seq_len kv_lora_rank"], w_kc: Float[Array, "kv_lora_rank kv_heads qk_nope_head_dim"], w_vc: Float[Array, "kv_lora_rank kv_heads v_head_dim"], b_q: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None, b_k: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None, softmax_scale: float | None = None, causal: bool = False, cu_seqlens: Int[Array, num_seqs_plus_one] | None = None, attention_mask: Bool[Array, "batch heads_or_1 seq_len kv_len"] | None = None, bias: Float[Array, "batch heads_or_1 seq_len kv_len"] | None = None, softmax_aux: Float[Array, ...] | None = None, logits_soft_cap: float | None = None, deterministic: bool = True, dropout_rng: PRNGKeyArray | None = None, dropout_prob: float = 0, sliding_window: int | tuple[int, int] | None = None, softmax_dtype: DTypeLike | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, cfg: FlashMLAConfig)` — [`L136`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_head_latent_attention.py#L136) — Execute flash multi-head latent attention.
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`FlashMLAConfig`](configs.md#FlashMLAConfig), [`block_k`](configs.md#FlashMLAConfig.block_k), [`block_q`](configs.md#FlashMLAConfig.block_q), [`num_stages`](configs.md#FlashMLAConfig.num_stages), [`num_warps`](configs.md#FlashMLAConfig.num_warps)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`flash_mla`](multi_head_latent_attention.md#flash_mla)

## Functions
- `flash_mla(query: Float[Array, "batch seq_len q_heads q_head_dim"], key_value: Float[Array, "batch seq_len kv_lora_rank"], w_kc: Float[Array, "kv_lora_rank kv_heads qk_nope_head_dim"], w_vc: Float[Array, "kv_lora_rank kv_heads v_head_dim"], b_q: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None, b_k: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None, cu_seqlens: Int[Array, num_seqs_plus_one] | None = None,, *, softmax_scale: float | None = None, causal: bool = False, attention_mask: Bool[Array, "batch heads_or_1 seq_len kv_len"] | None = None, bias: Float[Array, "batch heads_or_1 seq_len kv_len"] | None = None, softmax_aux: Float[Array, ...] | None = None, logits_soft_cap: float | None = None, deterministic: bool = True, dropout_rng: PRNGKeyArray | None = None, dropout_prob: float = 0, sliding_window: int | tuple[int, int] | None = None, softmax_dtype: DTypeLike | None = None, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: FlashMLAConfig | None = None)` — [`L324`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_head_latent_attention.py#L324) — Execute flash multi-head latent attention with automatic optimization.

## Module values
- `_mla_executor` — [`L310`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/multi_head_latent_attention.py#L310)

