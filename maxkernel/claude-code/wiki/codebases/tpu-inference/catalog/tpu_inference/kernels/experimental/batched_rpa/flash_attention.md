---
title: 'Module: tpu_inference/kernels/experimental/batched_rpa/flash_attention.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/experimental/batched_rpa/flash_attention.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.experimental.batched_rpa.flash_attention`/flash_attention_
symbols:
  flash_attention_qk_softmax: qk_softmax().
  flash_attention_pv: pv().
---
# Module: [`tpu_inference/kernels/experimental/batched_rpa/flash_attention.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/flash_attention.py)

## Functions
- `flash_attention_pv(p: jax.Array, v: jax.Array, alpha: jax.Array, o_prev: jax.Array, cfgs: configs.RpaConfigs)` — [`L98`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/flash_attention.py#L98) — Flash attention kernel.
- `flash_attention_qk_softmax(q: jax.Array, k: jax.Array, m_prev: jax.Array, l_prev: jax.Array, *, processed_q_len: list[jax.Array], processed_kv_len: list[jax.Array], effective_kv_len: list[jax.Array], cfgs: configs.RpaConfigs, bq_start: int)` — [`L23`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/flash_attention.py#L23) — Flash attention kernel.

