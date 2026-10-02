---
title: 'Module: src/maxtext/kernels/sort_activations.py'
type: catalog
provenance: extracted
module: src/maxtext/kernels/sort_activations.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.kernels.sort_activations`/
symbols:
  _route_fwd: _route_fwd().
  _unroute_fwd: _unroute_fwd().
  _route_impl: _route_impl().
  _unroute_impl: _unroute_impl().
  route: route().
  _route_bwd: _route_bwd().
  unroute: unroute().
  _unroute_bwd: _unroute_bwd().
  _sort_impl: _sort_impl().
---
# Module: [`src/maxtext/kernels/sort_activations.py`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/sort_activations.py)

## Functions
- `_route_bwd(use_gather_mosaic_kernel: bool, residuals: jax.Array, grads: jax.Array)` — [`L45`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/sort_activations.py#L45)
- `_route_fwd(tokens: jax.Array, selected_experts: jax.Array, use_gather_mosaic_kernel: bool)` — [`L34`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/sort_activations.py#L34)
- `_route_impl(tokens: jax.Array, selected_experts: jax.Array, use_gather_mosaic_kernel: bool)` — [`L85`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/sort_activations.py#L85) — Gather `tokens` according to `selected_experts`.
- `_sort_impl(tokens: jax.Array, inds: jax.Array, use_gather_mosaic_kernel: bool)` — [`L124`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/sort_activations.py#L124)
- `_unroute_bwd(use_gather_mosaic_kernel: bool, residuals: jax.Array, grads: jax.Array)` — [`L77`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/sort_activations.py#L77)
- `_unroute_fwd(tokens: jax.Array, selected_experts: jax.Array, use_gather_mosaic_kernel: bool)` — [`L66`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/sort_activations.py#L66)
- `_unroute_impl(tokens: jax.Array, selected_experts: jax.Array, use_gather_mosaic_kernel: bool)` — [`L98`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/sort_activations.py#L98) — Reverse the routing operation, restoring tokens to their original order.
- `route(tokens: jax.Array, selected_experts: jax.Array, use_gather_mosaic_kernel: bool)` — [`L25`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/sort_activations.py#L25) — Route tokens to selected experts.
- `unroute(tokens: jax.Array, selected_experts: jax.Array, use_gather_mosaic_kernel: bool)` — [`L58`](../../../../../../../raw/code/maxtext/src/maxtext/kernels/sort_activations.py#L58)

