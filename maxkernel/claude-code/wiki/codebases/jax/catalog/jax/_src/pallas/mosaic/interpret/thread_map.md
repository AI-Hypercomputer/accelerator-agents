---
title: 'Module: jax/_src/pallas/mosaic/interpret/thread_map.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic/interpret/thread_map.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic.interpret.thread_map`/
symbols:
  thread_map: thread_map().
  _call_threadmap_callback: _call_threadmap_callback().
  _run_jaxpr: _run_jaxpr().
  _thread_map_callback: _thread_map_callback().
  thread_map._f: thread_map()._f().
  _run_jaxpr._run: _run_jaxpr()._run().
  NESTED_TOKEN_VALUE: NESTED_TOKEN_VALUE.
---
# Module: [`jax/_src/pallas/mosaic/interpret/thread_map.py`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/thread_map.py)

## Functions
- `_call_threadmap_callback(token, device_id, jaxpr, num_threads, consts, invals, use_ordered_callback, on_exception)` — [`L75`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/thread_map.py#L75)
- `_f(core_or_thread_index, *args)` — [`L112`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/thread_map.py#L112)
- `_run(jaxpr, consts, *args)` — [`L30`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/thread_map.py#L30)
- `_run_jaxpr(jaxpr, consts, *args)` — [`L29`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/thread_map.py#L29)
- `_thread_map_callback(jaxpr, token, device_id, num_threads, consts, invals, *, on_exception)` — [`L38`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/thread_map.py#L38)
- `thread_map(f, num_threads, token, *args, use_ordered_callback=False, device_id=None, on_exception=lambda *args, **kwargs: None)` — [`L97`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/thread_map.py#L97) — Executes `f(thread_id, token, *args)` for `num_threads` threads.

## Module values
- `NESTED_TOKEN_VALUE` — [`L26`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/thread_map.py#L26)

