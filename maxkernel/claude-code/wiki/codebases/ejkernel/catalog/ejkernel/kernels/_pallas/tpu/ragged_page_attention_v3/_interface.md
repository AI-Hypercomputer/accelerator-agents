---
title: 'Module: ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._interface`/ragged_page_attention_v3().
symbols:
  ragged_page_attention_v3: ''
---
# Module: [`ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_interface.py)

## Functions
- `ragged_page_attention_v3(queries: Float[Array, "total_tokens num_q_heads head_dim"], keys: Float[Array, "total_tokens num_kv_heads head_dim"], values: Float[Array, "total_tokens num_kv_heads head_dim"], kv_cache: Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"], kv_lens: Int32[Array, max_num_seqs], block_tables: Int32[Array, max_num_seqs_times_pages_per_seq], query_start_loc: Int32[Array, max_num_seqs_plus_1], distribution: Int32[Array, 3], softmax_aux: Float[Array, num_q_heads] | None = None, *, softmax_scale: float = 1, sliding_window: int | None = None, logits_soft_cap: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None)` — [`L90`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_interface.py#L90) — Ragged paged attention with KV cache write for mixed prefill and decode.

