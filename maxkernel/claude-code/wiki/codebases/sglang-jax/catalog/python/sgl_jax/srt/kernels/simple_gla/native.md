---
title: 'Module: python/sgl_jax/srt/kernels/simple_gla/native.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/simple_gla/native.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.simple_gla.native`/
symbols:
  naive_gla_prefill: naive_gla_prefill().
  naive_gla_prefill.scan_fn: naive_gla_prefill().scan_fn().
  naive_gla_decode: naive_gla_decode().
  __all__: __all__.
---
# Module: [`python/sgl_jax/srt/kernels/simple_gla/native.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/native.py)

## Functions
- `naive_gla_decode(q: jax.Array, k: jax.Array, v: jax.Array, g_gamma: jax.Array, h0: jax.Array, scale: float | None = None)` — [`L12`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/native.py#L12) — Naive GLA decode using jnp.einsum.
- `naive_gla_prefill(q: jax.Array, k: jax.Array, v: jax.Array, g_gamma: jax.Array, h0: jax.Array, cu_seqlens: jax.Array, scale: float | None = None)` — [`L57`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/native.py#L57) — Naive GLA prefill using per-request scan + jnp.einsum.
- `scan_fn(carry, inputs)` — [`L100`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/native.py#L100)

## Module values
- `__all__` — [`L125`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/native.py#L125)

