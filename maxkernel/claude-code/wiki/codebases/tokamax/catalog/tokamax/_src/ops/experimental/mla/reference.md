---
title: 'Module: tokamax/_src/ops/experimental/mla/reference.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/experimental/mla/reference.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.experimental.mla.reference`/
symbols:
  mla_attention: mla_attention().
  update_kv_cache: update_kv_cache().
  align_to: align_to().
  update_kv_cache.per_batch_loop: update_kv_cache().per_batch_loop().
  update_kv_cache.per_batch_loop.per_token_loop: update_kv_cache().per_batch_loop().per_token_loop().
  mla_attention._run_per_batch: mla_attention()._run_per_batch().
  unsigned_cdiv: unsigned_cdiv().
---
# Module: [`tokamax/_src/ops/experimental/mla/reference.py`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/reference.py)

## Functions
- `_run_per_batch(start, end)` — [`L219`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/reference.py#L219)
- `align_to(a, b)` — [`L25`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/reference.py#L25)
- `mla_attention(ql_nope: jax.Array, q_pe: jax.Array, new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, s_dtype: jnp.dtype = jnp.bfloat16)` — [`L125`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/reference.py#L125) — Performs Multi-Head Latent Attention.
- `per_batch_loop(i, cache_kv)` — [`L89`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/reference.py#L89)
- `per_token_loop(j, cache_kv_)` — [`L94`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/reference.py#L94)
- `unsigned_cdiv(a, b)` — [`L29`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/reference.py#L29)
- `update_kv_cache(new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, batch_size: int)` — [`L38`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/reference.py#L38) — Updates the paged KV cache with new key and value per batch.

