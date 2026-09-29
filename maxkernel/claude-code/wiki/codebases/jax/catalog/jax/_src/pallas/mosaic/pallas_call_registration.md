---
title: 'Module: jax/_src/pallas/mosaic/pallas_call_registration.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic/pallas_call_registration.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic.pallas_call_registration`/
symbols:
  _lower_to_custom_call: _lower_to_custom_call().
  _get_memory_space_from_aval: _get_memory_space_from_aval().
  mpmd_map_tpu_lowering_rule: mpmd_map_tpu_lowering_rule().
  pallas_call_tpu_lowering_rule: pallas_call_tpu_lowering_rule().
  _rewrite_jaxpr_for_lowering: _rewrite_jaxpr_for_lowering().
  _rewrite_jaxpr_for_lowering.new_body: _rewrite_jaxpr_for_lowering().new_body().
  _resolve_memory_spaces: _resolve_memory_spaces().
  _jaxpr_kernel_aval_to_mosaic: _jaxpr_kernel_aval_to_mosaic().
  _get_memory_spaces_from_avals: _get_memory_spaces_from_avals().
  _resolve_side_effect_type: _resolve_side_effect_type().
  _resolve_tiling: _resolve_tiling().
  _jax_value_to_mosaic_value: _jax_value_to_mosaic_value().
  _lower_to_custom_call._maybe_cast_inputs: _lower_to_custom_call()._maybe_cast_inputs().
  _lower_to_custom_call._maybe_cast_outputs: _lower_to_custom_call()._maybe_cast_outputs().
  mpmd_map_tpu_lowering_rule._maybe_expand_scalar_input: mpmd_map_tpu_lowering_rule()._maybe_expand_scalar_input().
---
# Module: [`jax/_src/pallas/mosaic/pallas_call_registration.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pallas_call_registration.py)

## Functions
- `_get_memory_space_from_aval(out_aval: jax_core.AbstractValue, kernel_type: tpu_core.CoreType | None)` — [`L49`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pallas_call_registration.py#L49)
- `_get_memory_spaces_from_avals(avals: Sequence[jax_core.AbstractValue], kernel_type: tpu_core.CoreType | None)` — [`L106`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pallas_call_registration.py#L106)
- `_jax_value_to_mosaic_value(x: jax.Array)` — [`L248`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pallas_call_registration.py#L248)
- `_jaxpr_kernel_aval_to_mosaic(aval: jax_core.AbstractValue)` — [`L232`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pallas_call_registration.py#L232)
- `_lower_to_custom_call(ctx: mlir.LoweringRuleContext, *in_nodes, mosaic_module: ir.Module, mosaic_params: tpu_core.CompilerParams, kernel_type: tpu_core.CoreType | None, num_dynamic_grid_bounds: int, input_output_aliases: tuple[tuple[int, int], ...], cost_estimate: pallas_core.CostEstimate | None, out_avals: tuple[jax_core.AbstractValue, ...], effects: jax_core.Effects, metadata: frozen_dict.FrozenDict[str, str] | None, name: str, jax_mesh)` — [`L254`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pallas_call_registration.py#L254)
- `_maybe_cast_inputs(*args)` — [`L281`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pallas_call_registration.py#L281)
- `_maybe_cast_outputs(*args)` — [`L366`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pallas_call_registration.py#L366)
- `_maybe_expand_scalar_input(is_scalar, in_node, aval)` — [`L674`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pallas_call_registration.py#L674)
- `_resolve_memory_spaces(in_avals: Sequence[jax_core.AbstractValue], out_avals: Sequence[jax_core.AbstractValue], *, input_output_aliases: tuple[tuple[int, int], ...], kernel_type: tpu_core.CoreType | None)` — [`L123`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pallas_call_registration.py#L123)
- `_resolve_side_effect_type(has_side_effects: bool | tpu_core.SideEffectType)` — [`L197`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pallas_call_registration.py#L197)
- `_resolve_tiling(mosaic_params: tpu_core.CompilerParams, kernel_type: tpu_core.CoreType | None)` — [`L213`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pallas_call_registration.py#L213)
- `_rewrite_jaxpr_for_lowering(jaxpr: jax_core.Jaxpr, mesh: pallas_core.Mesh, all_meshes: tuple[pallas_core.Mesh, ...])` — [`L446`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pallas_call_registration.py#L446)
- `mpmd_map_tpu_lowering_rule(ctx: mlir.LoweringRuleContext, *in_nodes, meshes, jaxprs, out_avals, input_output_aliases, compiler_params, interpret, debug, cost_estimate, metadata, name, external_meshes, num_scratch)` — [`L546`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pallas_call_registration.py#L546)
- `new_body(*args)` — [`L486`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pallas_call_registration.py#L486)
- `pallas_call_tpu_lowering_rule(ctx: mlir.LoweringRuleContext, *in_nodes, jaxpr: jax_core.Jaxpr, grid_mapping: pallas_core.GridMapping, mesh: pallas_core.Mesh | None, input_output_aliases: tuple[tuple[int, int], ...], debug: bool, interpret: bool, compiler_params: pallas_core.CompilerParams | None, cost_estimate: pallas_core.CostEstimate | None, out_avals: tuple[jax_core.AbstractValue, ...], metadata: frozen_dict.FrozenDict[str, str] | None, name: str | None)` — [`L374`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pallas_call_registration.py#L374) — Lowers a pallas_call to a Mosaic TPU custom call.

