---
title: 'Module: jax/_src/pallas/triton/pallas_call_registration.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/triton/pallas_call_registration.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.triton.pallas_call_registration`/
symbols:
  pallas_call_lowering: pallas_call_lowering().
  avals_to_layouts: avals_to_layouts().
  normalize_grid: normalize_grid().
  CUSTOM_CALL_TARGET_NAME.CUSTOM_CALL_TARGET_NAME: CUSTOM_CALL_TARGET_NAME.CUSTOM_CALL_TARGET_NAME.
  USE_NEW_CUSTOM_CALL: USE_NEW_CUSTOM_CALL.
---
# Module: [`jax/_src/pallas/triton/pallas_call_registration.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/pallas_call_registration.py)

## Functions
- `avals_to_layouts(avals)` — [`L53`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/pallas_call_registration.py#L53)
- `normalize_grid(grid: pallas_core.StaticGrid)` — [`L45`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/pallas_call_registration.py#L45)
- `pallas_call_lowering(ctx: mlir.LoweringRuleContext, *in_nodes, jaxpr: jax_core.Jaxpr, interpret: bool, debug: bool, input_output_aliases: tuple[tuple[int, int], ...], grid_mapping: pallas_core.GridMapping, mesh: pallas_core.Mesh | None, compiler_params: pallas_core.CompilerParams | None, cost_estimate: pallas_core.CostEstimate | None, out_avals: tuple[jax_core.AbstractValue, ...], metadata: frozen_dict.FrozenDict[str, str] | None, name: str | None)` — [`L57`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/pallas_call_registration.py#L57)

## Module values
- `CUSTOM_CALL_TARGET_NAME` — [`L38`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/pallas_call_registration.py#L38)
- `USE_NEW_CUSTOM_CALL` — [`L37`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/pallas_call_registration.py#L37)

