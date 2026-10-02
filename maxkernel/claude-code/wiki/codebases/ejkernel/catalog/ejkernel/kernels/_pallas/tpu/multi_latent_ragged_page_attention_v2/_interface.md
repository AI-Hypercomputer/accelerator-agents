---
title: 'Module: ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention_v2/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention_v2/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.multi_latent_ragged_page_attention_v2._interface`/
symbols:
  multi_latent_ragged_page_attention_v2: multi_latent_ragged_page_attention_v2().
  _default_num_kv_pages_per_block: _default_num_kv_pages_per_block().
  _default_num_queries_per_block: _default_num_queries_per_block().
  __all__: __all__.
---
# Module: [`ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention_v2/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention_v2/_interface.py)

## Functions
- `_default_num_kv_pages_per_block(block_tables: Int32[Array, max_num_seqs_times_pages_per_seq], kv_lens: Int32[Array, max_num_seqs])` — [`L27`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention_v2/_interface.py#L27) — Heuristic KV-pages-per-block based on average pages per sequence.
- `_default_num_queries_per_block(queries_nope: Float[Array, "total_tokens num_q_heads kv_latent_dim"])` — [`L52`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention_v2/_interface.py#L52) — Heuristic queries-per-block based on total token count.
- `multi_latent_ragged_page_attention_v2(queries_nope: Float[Array, "total_tokens num_q_heads kv_latent_dim"], queries_pe: Float[Array, "total_tokens num_q_heads qk_pe_dim"], keys_values: Float[Array, "total_tokens kv_latent_dim"], keys_pe: Float[Array, "total_tokens qk_pe_dim"], kv_cache: Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"], kv_lens: Int32[Array, max_num_seqs], block_tables: Int32[Array, max_num_seqs_times_pages_per_seq], query_start_loc: Int32[Array, max_num_seqs_plus_1], distribution: Int32[Array, 3], *, softmax_scale: float | None = None, sliding_window: int | None = None, logits_soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_block: tuple[int, int, int] | int | None = None, num_queries_per_block: tuple[int, int, int] | int | None = None, vmem_limit_bytes: int | None = None, debug_mode: bool = False)` — [`L74`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention_v2/_interface.py#L74) — Compute TPU MLA ragged paged attention v2 and update paged KV cache.

## Module values
- `__all__` — [`L177`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention_v2/_interface.py#L177)

