---
title: 'Module: ejkernel/ops/execution/batch.py'
type: catalog
provenance: extracted
module: ejkernel/ops/execution/batch.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.ops.execution.batch`/
symbols:
  vmap_with_config.wrapped: vmap_with_config().wrapped().
  vmap_with_config: vmap_with_config().
  pmap_with_config: pmap_with_config().
  pmap_with_config.wrapped: pmap_with_config().wrapped().
  vmap_with_config.wrapped.slice0: vmap_with_config().wrapped().slice0().
  vmap_with_config.wrapped.fn: vmap_with_config().wrapped().fn().
  pmap_with_config.wrapped.fn: pmap_with_config().wrapped().fn().
---
# Module: [`ejkernel/ops/execution/batch.py`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/batch.py)

## Functions
- `fn(*a, **k)` — [`L122`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/batch.py#L122) — Execute ``kernel`` via ``executor`` — vectorized by ``jax.vmap``.
- `fn(*a, **k)` — [`L185`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/batch.py#L185) — Execute ``kernel`` via ``executor`` — parallelised by ``jax.pmap``.
- `pmap_with_config(executor, kernel, in_axes=0, axis_name="devices")` — [`L131`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/batch.py#L131) — Parallel execution across devices with shared configuration selection.
- `slice0(x, axis)` — [`L110`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/batch.py#L110) — Return the first element of ``x`` along ``axis``, or ``x`` unchanged.
- `vmap_with_config(executor, kernel, in_axes=0)` — [`L66`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/batch.py#L66) — Vectorized execution with shared configuration selection.
- `wrapped(*args, **kwargs)` — [`L107`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/batch.py#L107) — Select a configuration using a representative sample, then vmap over the batch.
- `wrapped(*args, **kwargs)` — [`L178`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/batch.py#L178) — Select a configuration using device-0 data, then pmap over all devices.

