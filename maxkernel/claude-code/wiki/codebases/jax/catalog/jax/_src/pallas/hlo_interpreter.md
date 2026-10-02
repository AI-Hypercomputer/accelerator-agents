---
title: 'Module: jax/_src/pallas/hlo_interpreter.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/hlo_interpreter.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.hlo_interpreter`/
symbols:
  pallas_call_hlo_interpret: pallas_call_hlo_interpret().
  eval_jaxpr_recursive: eval_jaxpr_recursive().
  pallas_call_hlo_interpret.body: pallas_call_hlo_interpret().body().
  kernel_to_hlo_jaxpr: kernel_to_hlo_jaxpr().
  resolve_physical_types: resolve_physical_types().
  pad_jaxpr_constvars: pad_jaxpr_constvars().
  _logical_aval_to_interpret_mode_aval: _logical_aval_to_interpret_mode_aval().
  make_hop_rule._resolve_jaxpr: make_hop_rule()._resolve_jaxpr().
  _initialize_output_vals: _initialize_output_vals().
  _dynamic_update_slice: _dynamic_update_slice().
  _dynamic_slice: _dynamic_slice().
  make_hop_rule.rule: make_hop_rule().rule().
  _run_scoped_physicalize_rule: _run_scoped_physicalize_rule().
  eval_jaxpr_recursive.read: eval_jaxpr_recursive().read().
  eval_jaxpr_recursive.write: eval_jaxpr_recursive().write().
  _pad_to_block_dimension: _pad_to_block_dimension().
  _eval_jaxpr_hop_rules: _eval_jaxpr_hop_rules.
  make_hop_rule: make_hop_rule().
  GridMapping: GridMapping.
  BlockMapping: BlockMapping.
  CostEstimate: CostEstimate.
  _get_next_indices: _get_next_indices().
  unsafe_map: unsafe_map.
  unsafe_zip: unsafe_zip.
  _logical_to_interpret_mode_dtype: _logical_to_interpret_mode_dtype().
  pallas_call_hlo_interpret.cond: pallas_call_hlo_interpret().cond().
---
# Module: [`jax/_src/pallas/hlo_interpreter.py`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py)

## Functions
- `_dynamic_slice(start_idx, block_shape: tuple[int, ...], value, is_squeeze)` — [`L93`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L93)
- `_dynamic_update_slice(start_idx, block_shape, value, update, is_squeeze)` — [`L103`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L103)
- `_get_next_indices(grid, indices)` — [`L113`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L113)
- `_initialize_output_vals(block_mappings_output: Iterable[BlockMapping], input_args, input_output_aliases)` — [`L146`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L146)
- `_logical_aval_to_interpret_mode_aval(aval)` — [`L83`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L83)
- `_logical_to_interpret_mode_dtype(dtype)` — [`L66`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L66) — Converts logical dtypes into JAX dtypes for interpret mode.
- `_pad_to_block_dimension(value, block_shape: tuple[int, ...])` — [`L123`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L123) — Pads values so the shape evenly divides into block dimensions.
- `_resolve_jaxpr(interpreter, value: jax_core.Jaxpr | jax_core.ClosedJaxpr, mapped_idx=None)` — [`L266`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L266)
- `_run_scoped_physicalize_rule(interpreter, *consts, jaxpr: jax_core.Jaxpr, collective_axes, **params)` — [`L317`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L317)
- `body(carry)` — [`L428`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L428)
- `cond(carry)` — [`L425`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L425)
- `eval_jaxpr_recursive(jaxpr: jax_core.Jaxpr, consts, *args, recurse_hop_rule: Callable[[jax_core.Jaxpr, Sequence[Any]], tuple[jax_core.Jaxpr, Sequence[Any]]], propagate_source_info=True)` — [`L178`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L178) — Evaluates a Jaxpr with recursion into higher-order primitives.
- `kernel_to_hlo_jaxpr(jaxpr: jax_core.Jaxpr, consts: Sequence[Any], grid_mapping: GridMapping)` — [`L161`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L161) — Converts a Pallas kernel jaxpr to a valid HLO jaxpr.
- `make_hop_rule(primitive, *keys)` — [`L254`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L254) — Makes a rule for higher-order ops by recursively applying the jaxpr pass.
- `pad_jaxpr_constvars(jaxpr: jax_core.Jaxpr, i: int, all_const_avals: Sequence[Any])` — [`L232`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L232) — Pads a Jaxpr with constvars from all branches.
- `pallas_call_hlo_interpret(*args, jaxpr: jax_core.Jaxpr, debug: bool, input_output_aliases: tuple[tuple[int, int], ...], grid_mapping: GridMapping, mesh: pallas_core.Mesh | None, compiler_params: Any, cost_estimate: CostEstimate, out_avals: tuple[jax_core.AbstractValue, ...], metadata: frozen_dict.FrozenDict[str, str] | None, name: str | None)` — [`L346`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L346)
- `read(v: jax_core.Atom)` — [`L199`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L199)
- `resolve_physical_types(jaxpr: jax_core.Jaxpr, consts: Sequence[Any])` — [`L332`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L332)
- `rule(interpreter, *args, **params)` — [`L292`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L292)
- `write(v: jax_core.Var, val: Any)` — [`L202`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L202)

## Module values
- `BlockMapping` — [`L61`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L61)
- `CostEstimate` — [`L63`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L63)
- `GridMapping` — [`L62`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L62)
- `_eval_jaxpr_hop_rules` — [`L230`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L230)
- `unsafe_map` — [`L58`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L58)
- `unsafe_zip` — [`L59`](../../../../../../../raw/code/jax/jax/_src/pallas/hlo_interpreter.py#L59)

