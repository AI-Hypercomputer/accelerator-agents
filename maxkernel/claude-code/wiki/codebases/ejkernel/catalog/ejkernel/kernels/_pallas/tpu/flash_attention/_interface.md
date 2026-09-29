---
title: 'Module: ejkernel/kernels/_pallas/tpu/flash_attention/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/flash_attention/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.flash_attention._interface`/
symbols:
  flash_attention: flash_attention().
  BlockTables: BlockTables.
  _flash_attention: _flash_attention().
  PagedKV: PagedKV.
  DenseKV: DenseKV.
---
# Module: [`ejkernel/kernels/_pallas/tpu/flash_attention/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_interface.py)

## Functions
- `_flash_attention(query, key, value, ab, segment_ids, save_residuals, causal, softmax_scale, block_sizes, sliding_window, logits_soft_cap)` — [`L327`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_interface.py#L327) — Internal flash attention with custom VJP for gradient computation.
- `flash_attention(query: Float[Array, "batch seq_len_q num_heads head_dim"], key: DenseKV | PagedKV, value: DenseKV | PagedKV, attention_mask: (Bool[Array, "batch num_heads_or_1 seq_len_q seq_len_k"] | Int[Array, "batch num_heads_or_1 seq_len_q seq_len_k"] | None) = None, bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None, softmax_scale: float | None = None, dropout_prob: float = 0, causal: bool = False, dropout_seed: int | None = None, cum_seqlens_q: Int[Array, batch_plus_one] | None = None, cum_seqlens_k: Int[Array, batch_plus_one] | None = None, sliding_window: int | tuple[int, int] | None = None, fwd_params: FwdParams | None = None, bwd_params: BwdParams | None = None, logits_soft_cap: float | None = None, softmax_aux: Float[Array, num_sinks] | None = None, normalize_output: bool = True, precision: lax.PrecisionLike = jax.lax.Precision.DEFAULT, logits_dtype: DTypeLike = jnp.float32, *, q_segment_ids: Int[Array, "batch seq_len_q"] | None = None, kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None, block_tables: BlockTables | None = None)` — [`L116`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_interface.py#L116) — Compute flash attention on TPU using Pallas kernels.

## Module values
- `BlockTables` — [`L97`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_interface.py#L97)
- `DenseKV` — [`L96`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_interface.py#L96)
- `PagedKV` — [`L95`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_interface.py#L95)

