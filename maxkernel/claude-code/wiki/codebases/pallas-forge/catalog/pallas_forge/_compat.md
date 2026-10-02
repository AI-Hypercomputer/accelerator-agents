---
title: 'Module: pallas_forge/_compat.py'
type: catalog
provenance: extracted
module: pallas_forge/_compat.py
status: fresh
symbol_base: scip-python python pallas-forge 0.0.0 `pallas_forge._compat`/
symbols:
  pallas_call_compat: pallas_call_compat().
  get_default_interpret_mode: get_default_interpret_mode().
  is_tpu_available: is_tpu_available().
  _pallas_call_accepts_num_stages: _pallas_call_accepts_num_stages().
---
# Module: [`pallas_forge/_compat.py`](../../../../../raw/code/pallas-forge/pallas_forge/_compat.py)

## Functions
- `_pallas_call_accepts_num_stages()` — [`L36`](../../../../../raw/code/pallas-forge/pallas_forge/_compat.py#L36) — Detect whether the installed JAX accepts num_stages directly.
- `get_default_interpret_mode()` — [`L30`](../../../../../raw/code/pallas-forge/pallas_forge/_compat.py#L30) — Returns True if no TPU is available (use interpret mode for CPU).
- `is_tpu_available()` — [`L22`](../../../../../raw/code/pallas-forge/pallas_forge/_compat.py#L22) — Check if TPU hardware is available.
- `pallas_call_compat(kernel_fn, *, grid: tuple[int, ...] | int, in_specs: Sequence[pl.BlockSpec], out_specs: pl.BlockSpec | Sequence[pl.BlockSpec], out_shape: Any, interpret: bool | None = None, **kwargs)` — [`L53`](../../../../../raw/code/pallas-forge/pallas_forge/_compat.py#L53) — Wrapper around pallas_call that auto-sets interpret=True on CPU.

