---
title: 'Module: tokamax/_src/hlo_utils.py'
type: catalog
provenance: extracted
module: tokamax/_src/hlo_utils.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.hlo_utils`/
symbols:
  get_opspecs: get_opspecs().
  _get_kernel_info_stablehlo.handle_op: _get_kernel_info_stablehlo().handle_op().
  get_kernel_info: get_kernel_info().
  _get_op_name: _get_op_name().
  _get_common_kernel_info: _get_common_kernel_info().
  _get_source_file_line: _get_source_file_line().
  _kernel_info_getter: _kernel_info_getter().
  _get_kernel_info_stablehlo: _get_kernel_info_stablehlo().
  _KERNEL_GETTER._KERNEL_GETTER: _KERNEL_GETTER._KERNEL_GETTER.
  KernelInfoBase.KernelInfoBase: KernelInfoBase.KernelInfoBase.
  _get_shape_dtype: _get_shape_dtype().
  _get_tokamax_xla_kernel_info: _get_tokamax_xla_kernel_info.
  DISABLE_JAX_EXPORT_CHECKS.DISABLE_JAX_EXPORT_CHECKS: DISABLE_JAX_EXPORT_CHECKS.DISABLE_JAX_EXPORT_CHECKS.
  HloComputation.HloComputation: HloComputation.HloComputation.
  _ALL_DTYPES._ALL_DTYPES: _ALL_DTYPES._ALL_DTYPES.
  TritonKernelInfo.TritonKernelInfo: TritonKernelInfo.TritonKernelInfo.
---
# Module: [`tokamax/_src/hlo_utils.py`](../../../../../../raw/code/tokamax/tokamax/_src/hlo_utils.py)

## Functions
- `_get_common_kernel_info(op: ir.OpView, call_stack: tuple[str, ...])` — [`L162`](../../../../../../raw/code/tokamax/tokamax/_src/hlo_utils.py#L162) — Extracts common kernel information from a `stablehlo` op.
- `_get_kernel_info_stablehlo(x: jax.stages.Lowered | ir.Module, include_xla_kernels: bool = True)` — [`L212`](../../../../../../raw/code/tokamax/tokamax/_src/hlo_utils.py#L212) — Extracts accelerator kernel information from a lowered JAX function.
- `_get_op_name(loc: ir.Location)` — [`L91`](../../../../../../raw/code/tokamax/tokamax/_src/hlo_utils.py#L91)
- `_get_shape_dtype(ty: ir.Type)` — [`L137`](../../../../../../raw/code/tokamax/tokamax/_src/hlo_utils.py#L137)
- `_get_source_file_line(loc: ir.Location)` — [`L146`](../../../../../../raw/code/tokamax/tokamax/_src/hlo_utils.py#L146) — Returns the source file and line number of a location.
- `_kernel_info_getter(cls)` — [`L184`](../../../../../../raw/code/tokamax/tokamax/_src/hlo_utils.py#L184)
- `get_kernel_info(x: HloComputation, include_xla_kernels: bool = True)` — [`L49`](../../../../../../raw/code/tokamax/tokamax/_src/hlo_utils.py#L49) — Extracts accelerator kernel information from HLO.
- `get_opspecs(x: HloComputation | KernelInfoBase, include_xla_kernels: bool = True)` — [`L67`](../../../../../../raw/code/tokamax/tokamax/_src/hlo_utils.py#L67) — Returns `BoundArguments` for all Tokamax ops in a given computation.
- `handle_op(op: ir.Operation, call_stack: tuple[str, ...] = ())` — [`L231`](../../../../../../raw/code/tokamax/tokamax/_src/hlo_utils.py#L231)

## Module values
- `DISABLE_JAX_EXPORT_CHECKS` — [`L32`](../../../../../../raw/code/tokamax/tokamax/_src/hlo_utils.py#L32)
- `HloComputation` — [`L40`](../../../../../../raw/code/tokamax/tokamax/_src/hlo_utils.py#L40)
- `KernelInfoBase` — [`L46`](../../../../../../raw/code/tokamax/tokamax/_src/hlo_utils.py#L46)
- `TritonKernelInfo` — [`L45`](../../../../../../raw/code/tokamax/tokamax/_src/hlo_utils.py#L45)
- `_ALL_DTYPES` — [`L105`](../../../../../../raw/code/tokamax/tokamax/_src/hlo_utils.py#L105)
- `_KERNEL_GETTER` — [`L188`](../../../../../../raw/code/tokamax/tokamax/_src/hlo_utils.py#L188)
- `_get_tokamax_xla_kernel_info` — [`L207`](../../../../../../raw/code/tokamax/tokamax/_src/hlo_utils.py#L207)

