---
title: 'Module: axlearn/common/ops/_optimization_barrier.py'
type: catalog
provenance: extracted
module: axlearn/common/ops/_optimization_barrier.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.ops._optimization_barrier`/forward_optimization_barrier
symbols:
  forward_optimization_barrier: ().
  forward_optimization_barrier_jvp: _jvp().
  forward_optimization_barrier_vmap: _vmap().
---
# Module: [`axlearn/common/ops/_optimization_barrier.py`](../../../../../../../raw/code/axlearn/axlearn/common/ops/_optimization_barrier.py)

## Functions
- `forward_optimization_barrier(pytree: Any)` — [`L13`](../../../../../../../raw/code/axlearn/axlearn/common/ops/_optimization_barrier.py#L13) — Returns `pytree` after transparently wrapping the computation in an XLA optimization barrier.
- `forward_optimization_barrier_jvp(primals: tuple, tangents: tuple)` — [`L100`](../../../../../../../raw/code/axlearn/axlearn/common/ops/_optimization_barrier.py#L100) — The JVP for `optimization_barrier`.
- `forward_optimization_barrier_vmap(batch_axis_size: int, in_batched: tuple, pytree: Any)` — [`L122`](../../../../../../../raw/code/axlearn/axlearn/common/ops/_optimization_barrier.py#L122) — VMAP rule for`optimization_barrier`.

