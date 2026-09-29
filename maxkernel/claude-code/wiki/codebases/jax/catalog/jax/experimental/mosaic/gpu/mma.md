---
title: 'Module: jax/experimental/mosaic/gpu/mma.py'
type: catalog
provenance: extracted
module: jax/experimental/mosaic/gpu/mma.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.mosaic.gpu.mma`/
symbols:
  mma: mma().
  _mma_single_tile: _mma_single_tile().
  MMALayouts.rhs: MMALayouts#rhs.
  MMALayouts.acc: MMALayouts#acc.
  MMALayouts.lhs: MMALayouts#lhs.
  MMALayouts: MMALayouts#
  MMALayouts.__init__: MMALayouts#__init__().
  _ptx_dtype_str: _ptx_dtype_str().
  SUPPORTED_F8_TYPES: SUPPORTED_F8_TYPES.
---
# Module: [`jax/experimental/mosaic/gpu/mma.py`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/mma.py)

## Classes
### `MMALayouts`
- def: [`jax/experimental/mosaic/gpu/mma.py:28`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/mma.py#L28)
- doc: Container for MMA layouts, providing a convenient way to create
- signature: `class MMALayouts:`
- members:
  - `acc` — [`L49`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/mma.py#L49)
  - `lhs` — [`L37`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/mma.py#L37)
  - `rhs` — [`L43`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/mma.py#L43)
- protocol/private: `__init__`[`L33`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/mma.py#L33)
- uses (calls/refs, reference-scoped): [`bitwidth`](utils.md#bitwidth), [`TiledLayout`](fragmented_array.md#TiledLayout), [`Replicated`](fragmented_array.md#Replicated), [`lane_dims`](fragmented_array.md#TiledLayout.lane_dims), [`warp_dims`](fragmented_array.md#TiledLayout.warp_dims), [`vector_dim`](fragmented_array.md#TiledLayout.vector_dim), [`Tiling`](fragmented_array.md#Tiling)
- used by: [`mma`](mma.md#mma), [`_mma_constraint_system`](layout_inference.md#_mma_constraint_system), [`_mgpu_mma_op_lowering_rule`](dialect_lowering.md#_mgpu_mma_op_lowering_rule)

## Functions
- `_mma_single_tile(acc: fa.FragmentedArray, a: fa.FragmentedArray, b: fa.FragmentedArray)` — [`L70`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/mma.py#L70) — Performs `acc + a @ b` using warp level MMA instructions.
- `_ptx_dtype_str(dtype: ir.Type, *, is_signed: bool | None = None)` — [`L57`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/mma.py#L57)
- `mma(acc: fa.FragmentedArray, a: fa.FragmentedArray, b: fa.FragmentedArray)` — [`L155`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/mma.py#L155) — Computes `acc + a @ b` using synchronous warp-level MMA instructions.

## Module values
- `SUPPORTED_F8_TYPES` — [`L25`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/mma.py#L25)

