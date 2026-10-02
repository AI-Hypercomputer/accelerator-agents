---
title: 'Module: src/maxtext/common/managed_mldiagnostics.py'
type: catalog
provenance: extracted
module: src/maxtext/common/managed_mldiagnostics.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.common.managed_mldiagnostics`/
symbols:
  ManagedMLDiagnostics.__new__: ManagedMLDiagnostics#__new__().
  ManagedMLDiagnostics.__init__: ManagedMLDiagnostics#__init__().
  ManagedMLDiagnostics._instance: ManagedMLDiagnostics#_instance.
  ManagedMLDiagnostics: ManagedMLDiagnostics#
  ManagedMLDiagnostics.should_log_key: ManagedMLDiagnostics#should_log_key().
  mldiag: mldiag.
  ManagedMLDiagnostics._initialized: ManagedMLDiagnostics#_initialized.
---
# Module: [`src/maxtext/common/managed_mldiagnostics.py`](../../../../../../../raw/code/maxtext/src/maxtext/common/managed_mldiagnostics.py)

## Classes
### `ManagedMLDiagnostics`
- def: [`src/maxtext/common/managed_mldiagnostics.py:27`](../../../../../../../raw/code/maxtext/src/maxtext/common/managed_mldiagnostics.py#L27)
- doc: ML Diagnostics Run, implemented with the Singleton pattern.
- signature: `class ManagedMLDiagnostics:`
- members:
  - `__init__(self, config)` — [`L45`](../../../../../../../raw/code/maxtext/src/maxtext/common/managed_mldiagnostics.py#L45) — Initializes the ManagedMLDiagnostics, ensuring this method runs only once.
  - `__new__(cls, *args: Any, **kwargs: Any)` — [`L35`](../../../../../../../raw/code/maxtext/src/maxtext/common/managed_mldiagnostics.py#L35) — Overrides the instance creation method.
  - `should_log_key(key, value)` — [`L56`](../../../../../../../raw/code/maxtext/src/maxtext/common/managed_mldiagnostics.py#L56)
- protocol/private: `_initialized`[`L51`](../../../../../../../raw/code/maxtext/src/maxtext/common/managed_mldiagnostics.py#L51), `_instance`[`L33`](../../../../../../../raw/code/maxtext/src/maxtext/common/managed_mldiagnostics.py#L33)
- uses (calls/refs, reference-scoped): [`mldiag`](managed_mldiagnostics.md#mldiag)

## Module values
- `mldiag` — [`L22`](../../../../../../../raw/code/maxtext/src/maxtext/common/managed_mldiagnostics.py#L22)

