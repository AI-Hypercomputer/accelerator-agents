---
title: 'Module: ejkernel/ops/execution/offline.py'
type: catalog
provenance: extracted
module: ejkernel/ops/execution/offline.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.ops.execution.offline`/
symbols:
  autotune_lowered: autotune_lowered().
  _labels_to_invocations: _labels_to_invocations().
  autotune_lowered.mk: autotune_lowered().mk().
  autotune_lowered.mk.f: autotune_lowered().mk().f().
---
# Module: [`ejkernel/ops/execution/offline.py`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/offline.py)

## Functions
- `_labels_to_invocations(lowered)` — [`L59`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/offline.py#L59) — Parse ejkernel labels from a lowered computation into (op_id_v, call_key) pairs.
- `autotune_lowered(selector: ConfigSelectorChain, lowered)` — [`L89`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/offline.py#L89) — Autotune all ejkernel operations found in a lowered JAX computation.
- `f(*a, **k)` — [`L150`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/offline.py#L150) — Call the resolved run method with config ``c`` and closed-over static kwargs.
- `mk(c, _run=run_method, _static=static_fun_kwargs)` — [`L147`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/offline.py#L147) — Return a callable that runs ``kernel.run`` with configuration ``c``.

