---
title: 'Module: jax/experimental/mosaic/gpu/wgmma.py'
type: catalog
provenance: extracted
module: jax/experimental/mosaic/gpu/wgmma.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.mosaic.gpu.wgmma`/
symbols:
  wgmma_m64: wgmma_m64().
  wgmma: wgmma().
  WGMMAAccumulator._value: WGMMAAccumulator#_value.
  WGMMAAccumulator: WGMMAAccumulator#
  WGMMAAccumulator.__init__: WGMMAAccumulator#__init__().
  WGMMAAccumulator.value: WGMMAAccumulator#value().
  WGMMAAccumulator.zero: WGMMAAccumulator#zero().
  WGMMAAccumulator.from_registers: WGMMAAccumulator#from_registers().
  bytewidth: bytewidth.
  wgmma_fence: wgmma_fence().
  WGMMAAccumulator._original_layout: WGMMAAccumulator#_original_layout.
  wgmma_m64.take_regs: wgmma_m64().take_regs().
  WGMMAAccumulator.tree_flatten: WGMMAAccumulator#tree_flatten().
  c: c.
  _as_i32_reg: _as_i32_reg().
  wgmma_m64.lc: wgmma_m64().lc().
  _llvm_add: _llvm_add().
  _supported_wgmma_types: _supported_wgmma_types().
  _lc: _lc().
  _unpack_i32: _unpack_i32().
  WGMMAAccumulator.tree_unflatten: WGMMAAccumulator#tree_unflatten().
---
# Module: [`jax/experimental/mosaic/gpu/wgmma.py`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/wgmma.py)

## Classes
### `WGMMAAccumulator`
- def: [`jax/experimental/mosaic/gpu/wgmma.py:39`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/wgmma.py#L39)
- doc: A FragmentedArray that has is synchronized with the async proxy.
- signature: `class WGMMAAccumulator:`
- members:
  - `from_registers(cls, registers, sync=True)` — [`L86`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/wgmma.py#L86)
  - `tree_flatten(self)` — [`L94`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/wgmma.py#L94)
  - `tree_unflatten(cls, aux, value)` — [`L98`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/wgmma.py#L98)
  - `value(self)` — [`L62`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/wgmma.py#L62)
  - `zero(cls, m, n, dtype=None, *, is_signed: bool | None = None)` — [`L66`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/wgmma.py#L66)
- protocol/private: `__init__`[`L49`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/wgmma.py#L49), `_original_layout`[`L46`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/wgmma.py#L46), `_value`[`L47`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/wgmma.py#L47)
- uses (calls/refs, reference-scoped): [`FragmentedArray`](fragmented_array.md#FragmentedArray), [`bitwidth`](utils.md#bitwidth), [`to_layout`](fragmented_array.md#FragmentedArray.to_layout), [`WGMMA_LAYOUT`](fragmented_array.md#WGMMA_LAYOUT), [`splat`](fragmented_array.md#FragmentedArray.splat), [`FragmentedLayout`](fragmented_array.md#FragmentedLayout.FragmentedLayout), [`WGMMA_LAYOUT_ACC_32BIT`](fragmented_array.md#WGMMA_LAYOUT_ACC_32BIT), [`wgmma_fence`](wgmma.md#wgmma_fence)
- used by: [`_run_scoped_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_run_scoped_lowering_rule), [`_run_state_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_run_state_lowering_rule), [`_cond_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_cond_lowering_rule), [`_while_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_while_lowering_rule), [`_wgmma_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_wgmma_lowering), [`wgmma`](wgmma.md#wgmma), [`_lower_jaxpr_to_for_loop`](../../../_src/pallas/mosaic_gpu/lowering.md#_lower_jaxpr_to_for_loop), [`f_lowered`](../../../_src/pallas/mosaic_gpu/lowering.md#_lower_fun.f_lowered), [`as_values`](../../../_src/pallas/mosaic_gpu/lowering.md#_lower_jaxpr_to_for_loop.as_values), [`_wgmma_accumulator_store_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_wgmma_accumulator_store_lowering)

## Functions
- `_as_i32_reg(v)` — [`L503`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/wgmma.py#L503)
- `_lc(x)` — [`L510`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/wgmma.py#L510)
- `_llvm_add(x, y)` — [`L515`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/wgmma.py#L515)
- `_supported_wgmma_types(dtype, abtype)` — [`L102`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/wgmma.py#L102)
- `_unpack_i32(vec_ty, r)` — [`L519`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/wgmma.py#L519)
- `lc(x)` — [`L247`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/wgmma.py#L247)
- `take_regs(n)` — [`L211`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/wgmma.py#L211)
- `wgmma(acc: WGMMAAccumulator, a: fa.FragmentedArray | ir.Value, b: ir.Value, *, swizzle: int = 128)` — [`L302`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/wgmma.py#L302) — Perform acc += a @ b using the WGMMA instruction.
- `wgmma_fence(array: fa.FragmentedArray)` — [`L491`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/wgmma.py#L491) — Fences the array construction from WGMMA instructions.
- `wgmma_m64(acc: np.ndarray, a, b_descriptor: ir.Value, a_transpose: bool | None, b_transpose: bool, a_k_stride: int | None, b_k_stride: int, n: int, swizzle: int, element_type: ir.Type)` — [`L119`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/wgmma.py#L119)

## Module values
- `bytewidth` — [`L34`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/wgmma.py#L34)
- `c` — [`L33`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/wgmma.py#L33)

