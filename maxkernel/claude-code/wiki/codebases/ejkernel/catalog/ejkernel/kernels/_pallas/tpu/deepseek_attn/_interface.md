---
title: 'Module: ejkernel/kernels/_pallas/tpu/deepseek_attn/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/deepseek_attn/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.deepseek_attn._interface`/deepseek_attn().
symbols:
  deepseek_attn: ''
---
# Module: [`ejkernel/kernels/_pallas/tpu/deepseek_attn/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/deepseek_attn/_interface.py)

## Functions
- `deepseek_attn(query: Float[Array, "batch seq_len q_heads q_head_dim"], key_value: Float[Array, "batch seq_len kv_lora_rank"], w_kc: Float[Array, "kv_lora_rank kv_heads qk_nope_head_dim"], w_vc: Float[Array, "kv_lora_rank kv_heads v_head_dim"], query_index: Float[Array, "batch seq_len index_heads index_head_dim"], key_index: Float[Array, "batch seq_len index_head_dim"], index_weights: Float[Array, "batch seq_len index_heads"], index_topk: int = 2048, softmax_scale: float | None = None, index_softmax_scale: float | None = None, b_q: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None, b_k: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None, causal: bool = True)` — [`L47`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/deepseek_attn/_interface.py#L47) — DeepSeek Sparse Attention on TPU using Pallas.

