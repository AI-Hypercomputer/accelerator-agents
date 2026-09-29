---
title: 'Module: jax/_src/pallas/mosaic/helpers.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic/helpers.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic.helpers`/
symbols:
  sync_copy: sync_copy().
  core_barrier: core_barrier().
  sync_copy._._copy_start_or_wait: sync_copy()._()._copy_start_or_wait().
  run_on_first_core: run_on_first_core().
  run_on_first_core.wrapped: run_on_first_core().wrapped().
  core_barrier._.signal_core: core_barrier()._().signal_core().
---
# Module: [`jax/_src/pallas/mosaic/helpers.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/helpers.py)

## Functions
- `_copy_start_or_wait(action, src_ref, dst_ref)` — [`L35`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/helpers.py#L35)
- `core_barrier(sem, *, core_axis_name: str)` — [`L73`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/helpers.py#L73) — Synchronizes all cores in a given axis.
- `run_on_first_core(core_axis_name: str)` — [`L56`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/helpers.py#L56) — Runs a function on the first core in a given axis.
- `signal_core(i)` — [`L82`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/helpers.py#L82)
- `sync_copy(src_ref, dst_ref, *, add: bool = False)` — [`L25`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/helpers.py#L25) — Synchronously copies a PyTree of refs to another PyTree of refs.
- `wrapped(f)` — [`L62`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/helpers.py#L62)

