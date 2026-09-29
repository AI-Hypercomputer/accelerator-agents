---
title: 'Module: ejkernel/modules/operations/deepseek_attn.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/deepseek_attn.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.deepseek_attn`/
symbols:
  DeepSeekAttention.run: DeepSeekAttention#run().
  DeepSeekAttention.heuristic_cfg: DeepSeekAttention#heuristic_cfg().
  DeepSeekAttention.candidate_cfgs_gpu: DeepSeekAttention#candidate_cfgs_gpu().
  DeepSeekAttention.candidate_cfgs_tpu: DeepSeekAttention#candidate_cfgs_tpu().
  _dsa_executor._dsa_executor: _dsa_executor._dsa_executor.
  DeepSeekAttention.get_impl: DeepSeekAttention#get_impl().
  DeepSeekAttention._seq_len_from_inv: DeepSeekAttention#_seq_len_from_inv().
  deepseek_attn: deepseek_attn().
  DeepSeekAttention: DeepSeekAttention#
  DeepSeekAttention.candidate_cfgs: DeepSeekAttention#candidate_cfgs().
  DeepSeekAttention.__init__: DeepSeekAttention#__init__().
  DeepSeekAttention._heuristic_gemm_block: DeepSeekAttention#_heuristic_gemm_block().
---
# Module: [`ejkernel/modules/operations/deepseek_attn.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/deepseek_attn.py)

## Classes
### `DeepSeekAttention`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/deepseek_attn.py:61`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/deepseek_attn.py#L61)
- doc: DeepSeek Sparse Attention with MLA + Lightning Indexer.
- signature: `class DeepSeekAttention(Kernel[DeepSeekAttentionConfig, Array]):`
- members:
  - `_heuristic_gemm_block(seq_len: int)` — [`L179`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/deepseek_attn.py#L179) — Operation-side tile heuristic — single source of truth.
  - `_seq_len_from_inv(inv: Invocation[DeepSeekAttentionConfig, Array])` — [`L168`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/deepseek_attn.py#L168) — Pull ``seq_len`` from the invocation's ``query`` tensor.
  - `candidate_cfgs(self, inv: Invocation[DeepSeekAttentionConfig, Array])` — [`L202`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/deepseek_attn.py#L202) — Generate candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[DeepSeekAttentionConfig, Array])` — [`L210`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/deepseek_attn.py#L210) — Generate GPU candidates for TileLang and XLA DeepSeek attention.
  - `candidate_cfgs_tpu(self, inv: Invocation[DeepSeekAttentionConfig, Array])` — [`L264`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/deepseek_attn.py#L264) — Generate TPU candidates for Pallas and XLA DeepSeek attention.
  - `get_impl(self, cfg: DeepSeekAttentionConfig)` — [`L72`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/deepseek_attn.py#L72) — Get kernel implementation from registry based on configuration.
  - `heuristic_cfg(self, inv: Invocation[DeepSeekAttentionConfig, Array])` — [`L188`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/deepseek_attn.py#L188) — Cold-start configuration with shape-aware ``gemm_block``.
  - `run(self, query: Float[Array, "batch seq_len q_heads q_head_dim"], key_value: Float[Array, "batch seq_len kv_lora_rank"], w_kc: Float[Array, "kv_lora_rank kv_heads qk_nope_head_dim"], w_vc: Float[Array, "kv_lora_rank kv_heads v_head_dim"], query_index: Float[Array, "batch seq_len index_heads index_head_dim"], key_index: Float[Array, "batch seq_len index_head_dim"], index_weights: Float[Array, "batch seq_len index_heads"], index_topk: int = 2048, softmax_scale: float | None = None, index_softmax_scale: float | None = None, b_q: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None, b_k: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None, causal: bool = True, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, *, cfg: DeepSeekAttentionConfig)` — [`L84`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/deepseek_attn.py#L84) — Execute DeepSeek Sparse Attention.
- protocol/private: `__init__`[`L69`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/deepseek_attn.py#L69)
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`Backend`](../../kernels/_registry.md#Backend), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`Platform`](../../kernels/_registry.md#Platform), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`ANY`](../../kernels/_registry.md#Backend.ANY), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`args`](../../ops/core/kernel.md#Invocation.args), [`DeepSeekAttentionConfig`](configs.md#DeepSeekAttentionConfig), [`TILELANG`](../../kernels/_registry.md#Platform.TILELANG), [`gemm_block`](configs.md#DeepSeekAttentionConfig.gemm_block), [`index_topk`](configs.md#DeepSeekAttentionConfig.index_topk), [`block_k`](configs.md#DeepSeekAttentionConfig.block_k), [`block_q`](configs.md#DeepSeekAttentionConfig.block_q), [`num_stages`](configs.md#DeepSeekAttentionConfig.num_stages), [`num_warps`](configs.md#DeepSeekAttentionConfig.num_warps)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`deepseek_attn`](deepseek_attn.md#deepseek_attn)

## Functions
- `deepseek_attn(query: Float[Array, "batch seq_len q_heads q_head_dim"], key_value: Float[Array, "batch seq_len kv_lora_rank"], w_kc: Float[Array, "kv_lora_rank kv_heads qk_nope_head_dim"], w_vc: Float[Array, "kv_lora_rank kv_heads v_head_dim"], query_index: Float[Array, "batch seq_len index_heads index_head_dim"], key_index: Float[Array, "batch seq_len index_head_dim"], index_weights: Float[Array, "batch seq_len index_heads"], b_q: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None, b_k: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,, *, index_topk: int = 2048, softmax_scale: float | None = None, index_softmax_scale: float | None = None, causal: bool = True, platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None, cfg: DeepSeekAttentionConfig | None = None)` — [`L296`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/deepseek_attn.py#L296) — Execute DeepSeek Sparse Attention with automatic optimization.

## Module values
- `_dsa_executor` — [`L282`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/deepseek_attn.py#L282)

