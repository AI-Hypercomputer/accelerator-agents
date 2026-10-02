---
title: 'Module: torchax/export.py'
type: catalog
provenance: extracted
module: torchax/export.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `torchax.export`/
symbols:
  exported_program_to_jax: exported_program_to_jax().
  exported_program_to_stablehlo: exported_program_to_stablehlo().
  extract_avals: extract_avals().
  JaxInterpreter.call_function: JaxInterpreter#call_function().
  exported_program_to_jax.func: exported_program_to_jax().func().
  extract_avals._to_aval: extract_avals()._to_aval().
  DEBUG: DEBUG.
  extract_avals._build_symbolic_shapes: extract_avals()._build_symbolic_shapes().
  extract_avals._build_symbolic_shapes._build_symbolic_shape: extract_avals()._build_symbolic_shapes()._build_symbolic_shape().
  JaxInterpreter.run_node: JaxInterpreter#run_node().
  JaxInterpreter: JaxInterpreter#
  _extract_states_from_exported_program: _extract_states_from_exported_program().
  exported_program_to_jax._extract_args: exported_program_to_jax()._extract_args().
  extract_avals._to_aval._get_dim: extract_avals()._to_aval()._get_dim().
  extract_avals._get_inputs: extract_avals()._get_inputs().
  extract_avals._build_symbolic_shapes._build_symbolic_constraints: extract_avals()._build_symbolic_shapes()._build_symbolic_constraints().
  JaxInterpreter.__init__: JaxInterpreter#__init__().
  _extra_decomp: _extra_decomp.
---
# Module: [`torchax/export.py`](../../../../../raw/code/torchax/torchax/export.py)

## Classes
### `JaxInterpreter`  ·  implements/extends Interpreter
- def: [`torchax/export.py:36`](../../../../../raw/code/torchax/torchax/export.py#L36) — documented in [torchax-export](../../concepts/torchax-export.md)
- doc: Experimental.
- signature: `class JaxInterpreter(torch.fx.Interpreter):`
- members:
  - `call_function(self, target, args: tuple, kwargs: dict)` — [`L42`](../../../../../raw/code/torchax/torchax/export.py#L42) — documented in [torchax-export](../../concepts/torchax-export.md)
  - `run_node(self, n)` — [`L61`](../../../../../raw/code/torchax/torchax/export.py#L61)
- protocol/private: `__init__`[`L39`](../../../../../raw/code/torchax/torchax/export.py#L39)
- uses (calls/refs, reference-scoped): [`all_aten_ops`](ops/ops_registry.md#all_aten_ops.all_aten_ops), [`func`](ops/ops_registry.md#Operator.func), [`is_jax_function`](ops/ops_registry.md#Operator.is_jax_function), [`DEBUG`](export.md#DEBUG)
- used by: [`func`](export.md#exported_program_to_jax.func)

## Functions
- `_build_symbolic_constraints(symbol_name, torch_constraint)` — [`L174`](../../../../../raw/code/torchax/torchax/export.py#L174) — Convert torch SymInt constraints to string for JAX symbolic_shape
- `_build_symbolic_shape(sym, constraint, free_symbols)` — [`L201`](../../../../../raw/code/torchax/torchax/export.py#L201) — Returns a JAX symbolic shape for a given symbol and constraint
- `_build_symbolic_shapes(range_constraints)` — [`L163`](../../../../../raw/code/torchax/torchax/export.py#L163) — Convert torch SymInt to JAX symbolic_shape and stores in a map using the
- `_extract_args(args, kwargs)` — [`L108`](../../../../../raw/code/torchax/torchax/export.py#L108)
- `_extract_states_from_exported_program(exported_model)` — [`L73`](../../../../../raw/code/torchax/torchax/export.py#L73) — documented in [torchax-export](../../concepts/torchax-export.md)
- `_get_dim(d)` — [`L139`](../../../../../raw/code/torchax/torchax/export.py#L139)
- `_get_inputs(exported)` — [`L153`](../../../../../raw/code/torchax/torchax/export.py#L153) — Return placeholders with input metadata
- `_to_aval(arg_meta, symbolic_shapes)` — [`L136`](../../../../../raw/code/torchax/torchax/export.py#L136) — Convet from torch type to jax abstract value for export tracing
- `exported_program_to_jax(exported_program, export_raw: bool = False)` — [`L90`](../../../../../raw/code/torchax/torchax/export.py#L90) — returns a pytree of jax arrays(state), and — documented in [torchax-export](../../concepts/torchax-export.md)
- `exported_program_to_stablehlo(exported_program)` — [`L247`](../../../../../raw/code/torchax/torchax/export.py#L247) — Replacement for torch_xla.stablehlo.exported_program_to_stablehlo — documented in [torchax-export](../../concepts/torchax-export.md)
- `extract_avals(exported)` — [`L131`](../../../../../raw/code/torchax/torchax/export.py#L131) — Return JAX Abstract Value shapes for all input parameters of the exported — documented in [torchax-export](../../concepts/torchax-export.md)
- `func(states, inputs)` — [`L114`](../../../../../raw/code/torchax/torchax/export.py#L114) — documented in [torchax-export](../../concepts/torchax-export.md)

## Module values
- `DEBUG` — [`L33`](../../../../../raw/code/torchax/torchax/export.py#L33)
- `_extra_decomp` — [`L70`](../../../../../raw/code/torchax/torchax/export.py#L70)

