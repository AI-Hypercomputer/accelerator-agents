---
title: 'Module: axlearn/common/compiler_options.py'
type: catalog
provenance: extracted
module: axlearn/common/compiler_options.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.compiler_options`/
symbols:
  infer_tpu_version: infer_tpu_version().
  default_xla_options: default_xla_options().
  infer_tpu_type: infer_tpu_type().
  infer_xla_performance_flags: infer_xla_performance_flags().
  infer_xsc_compiler_options: infer_xsc_compiler_options().
  _TPU_VERSIONS: _TPU_VERSIONS.
  _apply_overrides_from_env: _apply_overrides_from_env().
  NotTpuError: NotTpuError#
  _TPU_VERSION_ALIASES: _TPU_VERSION_ALIASES.
  xla_flags_from_options: xla_flags_from_options().
---
# Module: [`axlearn/common/compiler_options.py`](../../../../../../raw/code/axlearn/axlearn/common/compiler_options.py)

## Classes
### `NotTpuError`  ·  implements/extends ValueError
- def: [`axlearn/common/compiler_options.py:311`](../../../../../../raw/code/axlearn/axlearn/common/compiler_options.py#L311)
- signature: `class NotTpuError(ValueError):`
- used by: [`infer_tpu_type`](compiler_options.md#infer_tpu_type)

## Functions
- `_apply_overrides_from_env(options: dict[str, Union[str, bool, int]])` — [`L235`](../../../../../../raw/code/axlearn/axlearn/common/compiler_options.py#L235) — Apply environment variable overrides to XLA options.
- `default_xla_options(*, instance_type: str, num_slices: int, backend: str)` — [`L18`](../../../../../../raw/code/axlearn/axlearn/common/compiler_options.py#L18) — Return the default flags for the given instance type and backend.
- `infer_tpu_type(instance_type: str)` — [`L316`](../../../../../../raw/code/axlearn/axlearn/common/compiler_options.py#L316) — Infers tpu type (e.g. v4-8 or v6e-8-1) from instance type
- `infer_tpu_version(tpu_type: str)` — [`L325`](../../../../../../raw/code/axlearn/axlearn/common/compiler_options.py#L325) — Infer TPU version from the TPU type.
- `infer_xla_performance_flags(*, mesh_shape: MeshShape, mesh_axis_names: Sequence[str], device_kind: str)` — [`L414`](../../../../../../raw/code/axlearn/axlearn/common/compiler_options.py#L414) — Performs automatic XLA flag tuning based on mesh shape and device kind.
- `infer_xsc_compiler_options(*, halt_on_detection: bool = True, repeat_count: int = 1, device_kind: str)` — [`L349`](../../../../../../raw/code/axlearn/axlearn/common/compiler_options.py#L349) — Infers compiler options for running compiled function with XLA SDC check enabled.
- `xla_flags_from_options(xla_options: dict[str, Union[str, bool, int]])` — [`L298`](../../../../../../raw/code/axlearn/axlearn/common/compiler_options.py#L298) — Convert an XLA options dict suitable for

## Module values
- `_TPU_VERSIONS` — [`L411`](../../../../../../raw/code/axlearn/axlearn/common/compiler_options.py#L411)
- `_TPU_VERSION_ALIASES` — [`L410`](../../../../../../raw/code/axlearn/axlearn/common/compiler_options.py#L410)

