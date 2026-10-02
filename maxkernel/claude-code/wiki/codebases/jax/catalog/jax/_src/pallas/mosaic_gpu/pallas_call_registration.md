---
title: 'Module: jax/_src/pallas/mosaic_gpu/pallas_call_registration.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic_gpu/pallas_call_registration.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic_gpu.pallas_call_registration`/
symbols:
  pallas_call_lowering: pallas_call_lowering().
  _as_shaped_array: _as_shaped_array().
  pallas_call_lowering.zero_init_gmem_scratch: pallas_call_lowering().zero_init_gmem_scratch().
  pallas_call_lowering.dump_profile: pallas_call_lowering().dump_profile().
  pallas_call_lowering.do_callback: pallas_call_lowering().do_callback().
---
# Module: [`jax/_src/pallas/mosaic_gpu/pallas_call_registration.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pallas_call_registration.py)

## Functions
- `_as_shaped_array(t: jax.ShapeDtypeStruct)` — [`L160`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pallas_call_registration.py#L160)
- `do_callback(prof_buffer)` — [`L149`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pallas_call_registration.py#L149)
- `dump_profile(prof_buffer)` — [`L134`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pallas_call_registration.py#L134)
- `pallas_call_lowering(ctx: mlir.LoweringRuleContext, *args, jaxpr: jax_core.Jaxpr, interpret: bool, debug: bool, input_output_aliases: tuple[tuple[int, int], ...], grid_mapping: pallas_core.GridMapping, mesh: pallas_core.Mesh | None, compiler_params: pallas_core.CompilerParams | None, cost_estimate: pallas_core.CostEstimate | None, out_avals: tuple[jax_core.AbstractValue, ...], metadata: frozen_dict.FrozenDict[str, str] | None, name: str | None)` — [`L37`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pallas_call_registration.py#L37)
- `zero_init_gmem_scratch()` — [`L113`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pallas_call_registration.py#L113)

