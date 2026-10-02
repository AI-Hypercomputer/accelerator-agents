---
title: 'Module: qwix/_src/core/pallas.py'
type: catalog
provenance: extracted
module: qwix/_src/core/pallas.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.core.pallas`/
symbols:
  transform_block_specs_for_tpu: transform_block_specs_for_tpu().
  update_block_specs_for_qarray._update_block_spec: update_block_specs_for_qarray()._update_block_spec().
  pallas_call.wrapper: pallas_call().wrapper().
  pallas_call: pallas_call().
  update_block_specs_for_qarray: update_block_specs_for_qarray().
  transform_block_specs_for_tpu.restore: transform_block_specs_for_tpu().restore().
  _is_optimal_for_tpu: _is_optimal_for_tpu().
  _reorder: _reorder().
  _can_fit_tpu_requirements: _can_fit_tpu_requirements().
---
# Module: [`qwix/_src/core/pallas.py`](../../../../../../../raw/code/qwix/qwix/_src/core/pallas.py)

## Functions
- `_can_fit_tpu_requirements(block_shape: tuple[int | None, ...], arg_shape: tuple[int, ...])` — [`L217`](../../../../../../../raw/code/qwix/qwix/_src/core/pallas.py#L217) — Check if the block shape can fit the TPU requirements.
- `_is_optimal_for_tpu(block_shape: tuple[int | None, ...], arg_shape: tuple[int, ...])` — [`L227`](../../../../../../../raw/code/qwix/qwix/_src/core/pallas.py#L227) — Check if the block shape is already optimal for TPU.
- `_reorder(sequence: tuple[Any, ...], order: tuple[int, ...])` — [`L210`](../../../../../../../raw/code/qwix/qwix/_src/core/pallas.py#L210) — Reorder/transpose a sequence of elements.
- `_update_block_spec(spec: pl.BlockSpec, arg)` — [`L93`](../../../../../../../raw/code/qwix/qwix/_src/core/pallas.py#L93)
- `pallas_call(kernel: Callable[..., None], out_shape: Any, *, grid_spec: pl.GridSpec | None = None, grid=(), in_specs=pl.no_block_spec, out_specs=pl.no_block_spec, scratch_shapes=(), **kwargs)` — [`L27`](../../../../../../../raw/code/qwix/qwix/_src/core/pallas.py#L27) — A lifted version of jax.pallas_call that takes QArray as arguments.
- `restore(kernel_args)` — [`L191`](../../../../../../../raw/code/qwix/qwix/_src/core/pallas.py#L191)
- `transform_block_specs_for_tpu(block_specs: Any, args: Any)` — [`L122`](../../../../../../../raw/code/qwix/qwix/_src/core/pallas.py#L122) — Transform block specs and arguments so that they can be used on TPU.
- `update_block_specs_for_qarray(block_specs: Any, args: Any)` — [`L90`](../../../../../../../raw/code/qwix/qwix/_src/core/pallas.py#L90) — Update block specs for QArray arguments.
- `wrapper(*args)` — [`L61`](../../../../../../../raw/code/qwix/qwix/_src/core/pallas.py#L61)

