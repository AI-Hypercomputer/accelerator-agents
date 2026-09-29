---
title: 'Module: python/sgl_jax/srt/kernels/kda/naive.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/kda/naive.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.kda.naive`/
symbols:
  naive_recurrent_kda: naive_recurrent_kda().
  acc_dtype: acc_dtype().
---
# Module: [`python/sgl_jax/srt/kernels/kda/naive.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/naive.py)

## Functions
- `acc_dtype(input_dtype)` — [`L10`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/naive.py#L10) — Accumulator dtype: fp64 for fp64 inputs, fp32 otherwise.
- `naive_recurrent_kda(q: jax.Array, k: jax.Array, v: jax.Array, g: jax.Array, beta: jax.Array, scale: float | None = None, initial_state: jax.Array | None = None, output_final_state: bool = False)` — [`L15`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/naive.py#L15) — Core recurrence (per timestep):

