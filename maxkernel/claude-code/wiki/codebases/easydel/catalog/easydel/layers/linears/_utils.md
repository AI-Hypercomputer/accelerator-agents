---
title: 'Module: easydel/layers/linears/_utils.py'
type: catalog
provenance: extracted
module: easydel/layers/linears/_utils.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.layers.linears._utils`/
symbols:
  nf4_qmm_jax: nf4_qmm_jax().
  nf4xf32_to_f32: nf4xf32_to_f32().
  i4tou4: i4tou4().
---
# Module: [`easydel/layers/linears/_utils.py`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_utils.py)

## Functions
- `i4tou4(x)` — [`L72`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_utils.py#L72) — Convert signed int4 values to unsigned int4 values.
- `nf4_qmm_jax(x: jax.Array, wq: jax.Array, wscale: jax.Array, BK: int = 2048, BM: int = 2048, BNQL: int = 2048, compute_dtype: jnp.dtype = jnp.bfloat16)` — [`L94`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_utils.py#L94) — Perform quantized matrix multiplication with NF4 weights.
- `nf4xf32_to_f32(x)` — [`L37`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_utils.py#L37) — Convert NF4 quantized values to float32 using polynomial approximation.

