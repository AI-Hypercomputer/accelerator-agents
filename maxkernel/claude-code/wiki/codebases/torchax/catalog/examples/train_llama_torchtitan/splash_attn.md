---
title: 'Module: examples/train_llama_torchtitan/splash_attn.py'
type: catalog
provenance: extracted
module: examples/train_llama_torchtitan/splash_attn.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `examples.train_llama_torchtitan.splash_attn`/tpu_splash_attention().
symbols:
  tpu_splash_attention: ''
  tpu_splash_attention.wrap_flash_attention: wrap_flash_attention().
---
# Module: [`examples/train_llama_torchtitan/splash_attn.py`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/splash_attn.py)

## Functions
- `tpu_splash_attention(mesh, q_sharding, apply_shard_map, query: jax.Array, key: jax.Array, value: jax.Array, decoder_segment_ids: jax.Array | None, attn_logits_soft_cap: float | None = None)` — [`L25`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/splash_attn.py#L25) — TPU Flash Attention.
- `wrap_flash_attention(query, key, value, decoder_segment_ids)` — [`L55`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/splash_attn.py#L55)

