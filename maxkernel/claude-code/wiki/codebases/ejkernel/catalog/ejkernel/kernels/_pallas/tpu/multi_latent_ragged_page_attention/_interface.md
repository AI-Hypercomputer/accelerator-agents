---
title: 'Module: ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.multi_latent_ragged_page_attention._interface`/
symbols:
  multi_latent_ragged_page_attention: multi_latent_ragged_page_attention().
  __all__: __all__.
---
# Module: [`ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_interface.py)

## Functions
- `multi_latent_ragged_page_attention(queries_nope: Float[Array, "total_tokens num_q_heads kv_latent_dim"], queries_pe: Float[Array, "total_tokens num_q_heads qk_pe_dim"], keys_values: Float[Array, "total_tokens kv_latent_dim"], keys_pe: Float[Array, "total_tokens qk_pe_dim"], kv_cache: Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"], kv_lens: Int32[Array, max_num_seqs], block_tables: Int32[Array, max_num_seqs_times_pages_per_seq], query_start_loc: Int32[Array, max_num_seqs_plus_1], distribution: Int32[Array, 3], *, softmax_scale: float | None = None, sliding_window: int | None = None, logits_soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None, debug_mode: bool = False)` — [`L34`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_interface.py#L34) — Compute TPU MLA ragged paged attention and update paged KV cache.

## Module values
- `__all__` — [`L143`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_interface.py#L143)

