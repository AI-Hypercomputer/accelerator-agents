---
title: 'Module: jaxlib/mlir/_mlir_libs/_jax_mlir_ext.pyi'
type: catalog
provenance: extracted
module: jaxlib/mlir/_mlir_libs/_jax_mlir_ext.pyi
status: fresh
symbol_base: scip-python python jax 0.0.0 `jaxlib.mlir._mlir_libs._jax_mlir_ext`/
symbols:
  arith_constant: arith_constant().
  inlined_func_call: inlined_func_call().
  register_dialects: register_dialects().
  enter_multi_threaded_execution: enter_multi_threaded_execution().
  exit_multi_threaded_execution: exit_multi_threaded_execution().
  TracebackToLocationCache: TracebackToLocationCache#
  TracebackToLocationCache.__init__: TracebackToLocationCache#__init__().
  TracebackToLocationCache.get: TracebackToLocationCache#get().
---
# Module: [`jaxlib/mlir/_mlir_libs/_jax_mlir_ext.pyi`](../../../../../../../raw/code/jax/jaxlib/mlir/_mlir_libs/_jax_mlir_ext.pyi)

## Classes
### `TracebackToLocationCache`
- def: [`jaxlib/mlir/_mlir_libs/_jax_mlir_ext.pyi:38`](../../../../../../../raw/code/jax/jaxlib/mlir/_mlir_libs/_jax_mlir_ext.pyi#L38)
- signature: `class TracebackToLocationCache:`
- members:
  - `get(self, traceback: Traceback,)` — [`L45`](../../../../../../../raw/code/jax/jaxlib/mlir/_mlir_libs/_jax_mlir_ext.pyi#L45)
- protocol/private: `__init__`[`L39`](../../../../../../../raw/code/jax/jaxlib/mlir/_mlir_libs/_jax_mlir_ext.pyi#L39)

## Functions
- `arith_constant(value: int | float | bool, type: ir.Type,)` — [`L35`](../../../../../../../raw/code/jax/jaxlib/mlir/_mlir_libs/_jax_mlir_ext.pyi#L35) — Creates an arith.constant operation.
- `enter_multi_threaded_execution(arg: ir.Context,)` — [`L26`](../../../../../../../raw/code/jax/jaxlib/mlir/_mlir_libs/_jax_mlir_ext.pyi#L26)
- `exit_multi_threaded_execution(arg: ir.Context,)` — [`L27`](../../../../../../../raw/code/jax/jaxlib/mlir/_mlir_libs/_jax_mlir_ext.pyi#L27)
- `inlined_func_call(callee: ir.Operation, args: Sequence[ir.Value], loc: ir.Location | None = ...)` — [`L28`](../../../../../../../raw/code/jax/jaxlib/mlir/_mlir_libs/_jax_mlir_ext.pyi#L28) — Makes an inlined call to a function containing a single block with a single return op.
- `register_dialects(arg: ir.DialectRegistry,, register_pipelines: bool = True)` — [`L23`](../../../../../../../raw/code/jax/jaxlib/mlir/_mlir_libs/_jax_mlir_ext.pyi#L23)

