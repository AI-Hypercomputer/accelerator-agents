---
title: 'Module: qwix/contrib/awq_core.py'
type: catalog
provenance: extracted
module: qwix/contrib/awq_core.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix.contrib.awq_core`/
symbols:
  quantize_weight: quantize_weight().
  search_optimal_scales: search_optimal_scales().
  search_optimal_scales.compute_loss_and_scales: search_optimal_scales().compute_loss_and_scales().
  compute_act_scale: compute_act_scale().
---
# Module: [`qwix/contrib/awq_core.py`](../../../../../../raw/code/qwix/qwix/contrib/awq_core.py)

## Functions
- `compute_act_scale(x: jax.Array, axis: int = 0)` — [`L34`](../../../../../../raw/code/qwix/qwix/contrib/awq_core.py#L34) — Computes per-channel activation magnitudes.
- `compute_loss_and_scales(ratio: jax.Array)` — [`L92`](../../../../../../raw/code/qwix/qwix/contrib/awq_core.py#L92)
- `quantize_weight(w: jax.Array, activation_scales: jax.Array, how: qarray.HowToQuantize, n_grid: int = 20)` — [`L124`](../../../../../../raw/code/qwix/qwix/contrib/awq_core.py#L124) — Quantizes a weight matrix using AWQ.
- `search_optimal_scales(w: jax.Array, activation_scales: jax.Array, how: qarray.HowToQuantize, n_grid: int = 20, min_scale: float = 0.0001)` — [`L51`](../../../../../../raw/code/qwix/qwix/contrib/awq_core.py#L51) — Searches for optimal per-channel scaling factors using grid search.

