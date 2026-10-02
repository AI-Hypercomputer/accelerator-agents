---
title: 'Module: python/sgl_jax/srt/disaggregation/runtime.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/disaggregation/runtime.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.disaggregation.runtime`/
symbols:
  install_disaggregation_wiring: install_disaggregation_wiring().
  _make_disagg_shutdown._shutdown: _make_disagg_shutdown()._shutdown().
  _make_disagg_shutdown: _make_disagg_shutdown().
  logger: logger.
  install_disaggregation_wiring._handler: install_disaggregation_wiring()._handler().
---
# Module: [`python/sgl_jax/srt/disaggregation/runtime.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/runtime.py)

## Functions
- `_handler(_signum, _frame, _prev=previous)` — [`L189`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/runtime.py#L189)
- `_make_disagg_shutdown(scheduler: Scheduler, mode: str)` — [`L207`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/runtime.py#L207) — Create an idempotent graceful-shutdown closure.
- `_shutdown()` — [`L212`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/runtime.py#L212)
- `install_disaggregation_wiring(scheduler: Scheduler, server_args: ServerArgs)` — [`L18`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/runtime.py#L18) — Wire up PD runtime attributes when disaggregation mode is enabled.

## Module values
- `logger` — [`L15`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/runtime.py#L15)

