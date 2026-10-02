---
title: 'Module: ejkernel/kernels/_xla/deepseek_attn/_xla_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_xla/deepseek_attn/_xla_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._xla.deepseek_attn._xla_impl_fwd`/_
symbols:
  _lightning_indexer: lightning_indexer().
  _deepseek_attention_fwd: deepseek_attention_fwd().
---
# Module: [`ejkernel/kernels/_xla/deepseek_attn/_xla_impl_fwd.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/deepseek_attn/_xla_impl_fwd.py)

## Functions
- `_deepseek_attention_fwd(query: Float[Array, "batch seq_len q_heads q_head_dim"], key_value: Float[Array, "batch seq_len kv_lora_rank"], w_kc: Float[Array, "kv_lora_rank kv_heads qk_nope_head_dim"], w_vc: Float[Array, "kv_lora_rank kv_heads v_head_dim"], query_index: Float[Array, "batch seq_len index_heads index_head_dim"], key_index: Float[Array, "batch seq_len index_head_dim"], index_weights: Float[Array, "batch seq_len index_heads"], index_topk: int, softmax_scale: float, index_softmax_scale: float | None = None, b_q: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None, b_k: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None, causal: bool = True)` — [`L85`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/deepseek_attn/_xla_impl_fwd.py#L85) — DeepSeek Sparse Attention forward pass with MLA-style inputs.
- `_lightning_indexer(query_index: Float[Array, "batch seq_len index_heads index_head_dim"], key_index: Float[Array, "batch seq_len index_head_dim"], index_weights: Float[Array, "batch seq_len index_heads"], index_topk: int, softmax_scale: float, causal: bool = True)` — [`L41`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/deepseek_attn/_xla_impl_fwd.py#L41) — Compute Lightning Indexer scores and select top-k tokens.

