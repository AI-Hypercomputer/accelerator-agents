---
title: 'Module: ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.ragged_page_attention_v2._interface`/
symbols:
  _ragged_page_attention: _ragged_page_attention().
  ragged_page_attention_v2: ragged_page_attention_v2().
  _HBM_ANY: _HBM_ANY.
  _ragged_page_attention.q_index_map: _ragged_page_attention().q_index_map().
---
# Module: [`ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_interface.py)

## Functions
- `_ragged_page_attention(q: Float[Array, "max_num_batched_tokens num_q_heads head_dim"], kv_pages: Float[Array, "total_num_pages page_size num_combined_kv_heads head_dim"], context_lens: Int[Array, max_num_seqs], block_tables: Int[Array, "max_num_seqs pages_per_seq"], query_start_loc: Int[Array, max_num_seqs_plus_one], num_seqs: jax.Array, *, softmax_scale: float = 1, sliding_window: int | None = None, logits_soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None, softmax_aux: jax.Array | None = None)` — [`L111`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_interface.py#L111) — Ragged paged attention that supports mixed prefill and decode.
- `q_index_map(heads_blk_idx, q_blk_idx, *_)` — [`L202`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_interface.py#L202)
- `ragged_page_attention_v2(queries: Float[Array, "total_tokens num_q_heads head_dim"], kv_pages: Float[Array, "num_pages page_size num_combined_kv_heads head_dim"], context_lens: Int[Array, num_seqs], block_tables: Int[Array, "num_seqs pages_per_seq"], query_start_loc: Int[Array, num_seqs_plus_one], num_seqs: Array | int, *, softmax_scale: float | None = None, logits_soft_cap: float | None = None, compute_dtype: DTypeLike = jnp.bfloat16, optimized: bool = False, sliding_window: int | None = None, softmax_aux: Float[Array, num_q_heads] | None = None, mask_value: float | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None, num_warps: int | None = None, num_stages: int | None = None)` — [`L249`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_interface.py#L249) — Ragged paged attention that supports mixed prefill and decode.

## Module values
- `_HBM_ANY` — [`L95`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_interface.py#L95)

