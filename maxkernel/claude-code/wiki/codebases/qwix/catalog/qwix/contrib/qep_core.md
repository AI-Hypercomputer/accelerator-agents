---
title: 'Module: qwix/contrib/qep_core.py'
type: catalog
provenance: extracted
module: qwix/contrib/qep_core.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix.contrib.qep_core`/
symbols:
  weight_correct: weight_correct().
  compute_qep_stats: compute_qep_stats().
---
# Module: [`qwix/contrib/qep_core.py`](../../../../../../raw/code/qwix/qwix/contrib/qep_core.py)

## Functions
- `compute_qep_stats(X_q: jax.Array, X_float: jax.Array)` — [`L31`](../../../../../../raw/code/qwix/qwix/contrib/qep_core.py#L31) — Computes QEP (Quantization Error Propagation) statistics.
- `weight_correct(W: jax.Array, H: jax.Array, H_delta: jax.Array, *, correction_factor: float = 0.5, damping_factor: float = 0.01)` — [`L57`](../../../../../../raw/code/qwix/qwix/contrib/qep_core.py#L57) — Applies QEP weight correction to compensate for input quantization noise.

