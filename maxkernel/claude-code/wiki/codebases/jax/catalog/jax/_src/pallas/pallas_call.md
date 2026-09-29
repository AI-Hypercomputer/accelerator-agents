---
title: 'Module: jax/_src/pallas/pallas_call.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/pallas_call.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.pallas_call`/
symbols:
  _pallas_call_batching_rule: _pallas_call_batching_rule().
  _pallas_call_state_discharge_rule: _pallas_call_state_discharge_rule().
  _pallas_call_to_lojax: _pallas_call_to_lojax().
  _pallas_call_jvp_rule: _pallas_call_jvp_rule().
  pallas_call: pallas_call().
  _pallas_call.wrapped: _pallas_call().wrapped().
  _batch_with_explicit_loop: _batch_with_explicit_loop().
  _batch_block_mapping: _batch_block_mapping().
  _trace_kernel_to_jaxpr: _trace_kernel_to_jaxpr().
  _pallas_call_abstract_eval: _pallas_call_abstract_eval().
  pallas_call_p: pallas_call_p.
  _pallas_call: _pallas_call().
  _pallas_call_lowering: _pallas_call_lowering().
  _pallas_call_lowering.gpu_lowering: _pallas_call_lowering().gpu_lowering().
  _batch_block_mapping._block_map_function: _batch_block_mapping()._block_map_function().
  _batch_with_explicit_loop.body: _batch_with_explicit_loop().body().
  _pallas_call_state_discharge_rule._rewritten_body: _pallas_call_state_discharge_rule()._rewritten_body().
  _pallas_call_batching_rule.temp_f: _pallas_call_batching_rule().temp_f().
  CostEstimate: CostEstimate.
  GridMapping: GridMapping.
  _pallas_call_impl._jit_run: _pallas_call_impl()._jit_run().
  _pallas_call_lowering.tpu_lowering: _pallas_call_lowering().tpu_lowering().
  _pallas_call_batching_rule._maybe_squeeze_out_bdim: _pallas_call_batching_rule()._maybe_squeeze_out_bdim().
  CompilerParams: CompilerParams.
  _broadcast_input_output_aliases: _broadcast_input_output_aliases().
  no_block_spec: no_block_spec.
  _pallas_call_impl: _pallas_call_impl().
  _pallas_call_dce_rule: _pallas_call_dce_rule().
  _pallas_call_typecheck_rule: _pallas_call_typecheck_rule().
  BlockMapping: BlockMapping.
  _get_index_mapping: _get_index_mapping().
  _unsupported_lowering_error: _unsupported_lowering_error().
  unsafe_map: unsafe_map.
  unsafe_zip: unsafe_zip.
  _pallas_call_is_high: _pallas_call_is_high().
  _pallas_call_lowering.cpu_lowering: _pallas_call_lowering().cpu_lowering().
---
# Module: [`jax/_src/pallas/pallas_call.py`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py)

## Functions
- `_batch_block_mapping(grid_mapping: GridMapping, axis_size: int, aval: jax_core.ShapedArray, dim: int | batching.NotMapped, block_mapping: BlockMapping)` — [`L332`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L332)
- `_batch_with_explicit_loop(args: Sequence[jax_typing.Array], dims: Sequence[int | batching.NotMapped], *, jaxpr: jax_core.Jaxpr, grid_mapping: GridMapping, mesh: pallas_core.Mesh | None, input_output_aliases: tuple[tuple[int, int], ...], debug: bool, interpret: Any, compiler_params: Any, cost_estimate: CostEstimate | None, out_avals: tuple[jax_core.AbstractValue, ...], metadata: FrozenDict[str, str] | None, name: str | None)` — [`L411`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L411) — Batch the pallas_call by calling it in loop over the batch size.
- `_block_map_function(new_idx, *args)` — [`L339`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L339)
- `_broadcast_input_output_aliases(args: Sequence[jax_typing.Array], dims: Sequence[int | batching.NotMapped], *, input_output_aliases: tuple[tuple[int, int], ...], axis_size: int)` — [`L380`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L380) — Broadcast input/output operands.
- `_get_index_mapping(avals)` — [`L148`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L148)
- `_jit_run(*args)` — [`L79`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L79)
- `_maybe_squeeze_out_bdim(x: jax_typing.Array, bdim: int | batching.NotMapped)` — [`L551`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L551)
- `_pallas_call(kernel: Callable[..., None], out_shape: Any, *, grid_spec: pallas_core.GridSpec, scratch_tree: tree_util.PyTreeDef, mesh: pallas_core.Mesh | None = None, input_output_aliases: Mapping[int, int] = {}, debug: bool = False, interpret: Any = False, name: str | None = None, compiler_params: CompilerParams | None = None, cost_estimate: CostEstimate | None = None, metadata: dict[str, str] | None = None)` — [`L1231`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L1231)
- `_pallas_call_abstract_eval(*avals, out_avals: tuple[jax_core.AbstractValue, ...], interpret, compiler_params: CompilerParams | None, input_output_aliases, grid_mapping, **params)` — [`L88`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L88)
- `_pallas_call_batching_rule(axis_data, args, dims, *, jaxpr: jax_core.Jaxpr, grid_mapping: GridMapping, mesh: pallas_core.Mesh | None, input_output_aliases: tuple[tuple[int, int], ...], debug: bool, interpret: Any, compiler_params: CompilerParams | None, cost_estimate: CostEstimate | None, out_avals: tuple[jax_core.AbstractValue, ...], metadata: FrozenDict[str, str] | None = None, name: str | None = None)` — [`L512`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L512)
- `_pallas_call_dce_rule(used_outs: list[bool], eqn: pe.JaxprEqn)` — [`L775`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L775)
- `_pallas_call_impl(*args, **params)` — [`L75`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L75)
- `_pallas_call_is_high(*_, jaxpr, **params)` — [`L142`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L142)
- `_pallas_call_jvp_rule(primals, tangents, *, jaxpr: jax_core.Jaxpr, input_output_aliases: tuple[tuple[int, int], ...], grid_mapping: GridMapping, mesh: pallas_core.Mesh | None, debug: bool, interpret: Any, compiler_params: CompilerParams | None, cost_estimate: CostEstimate | None, out_avals: tuple[jax_core.AbstractValue, ...], metadata: FrozenDict[str, str] | None, name: str | None)` — [`L249`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L249)
- `_pallas_call_lowering(ctx: mlir.LoweringRuleContext, *in_nodes, interpret: Any, **params)` — [`L837`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L837)
- `_pallas_call_state_discharge_rule(avals_in, avals_out, *args, jaxpr: jax_core.Jaxpr, input_output_aliases: tuple[tuple[int, int], ...], grid_mapping: GridMapping, mesh: pallas_core.Mesh | None, debug: bool, interpret: Any, compiler_params: CompilerParams | None, cost_estimate: CostEstimate | None, out_avals: tuple[jax_core.AbstractValue, ...], metadata: FrozenDict[str, str] | None, name: str | None)` — [`L994`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L994)
- `_pallas_call_to_lojax(*hi_args, jaxpr: jax_core.Jaxpr, input_output_aliases: tuple[tuple[int, int], ...], grid_mapping: GridMapping, mesh: pallas_core.Mesh | None, debug: bool, interpret: Any, compiler_params: Any, cost_estimate: CostEstimate | None, out_avals: tuple[jax_core.AbstractValue, ...], metadata: FrozenDict[str, str] | None, name: str | None)` — [`L160`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L160)
- `_pallas_call_typecheck_rule(ctx_factory, *in_atoms, grid_mapping, **params)` — [`L984`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L984)
- `_rewritten_body(*args)` — [`L1064`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L1064)
- `_trace_kernel_to_jaxpr(fun: Callable, debug_info: jax_core.DebugInfo, grid_mapping: GridMapping, kernel_avals: ft.FlatTree, kernel_in_transforms: tuple[tuple[state.Transform, ...], ...], indexer: bool = False)` — [`L784`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L784)
- `_unsupported_lowering_error(platform: str)` — [`L829`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L829)
- `body(batch_index: jax_typing.Array, state: list[jax_typing.Array])` — [`L462`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L462)
- `cpu_lowering(ctx: mlir.LoweringRuleContext, *in_nodes: ir.Value | Sequence[ir.Value], **params)` — [`L873`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L873)
- `gpu_lowering(ctx: mlir.LoweringRuleContext, *in_nodes: ir.Value | Sequence[ir.Value], is_rocm: bool, compiler_params: pallas_core.CompilerParams | None, **params)` — [`L900`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L900) — Shared GPU lowering implementation for CUDA and ROCm.
- `pallas_call(kernel: Callable[..., None], out_shape: Any, *, grid_spec: pallas_core.GridSpec | None = None, grid: pallas_core.TupleGrid = (), in_specs: pallas_core.BlockSpecTree = no_block_spec, out_specs: pallas_core.BlockSpecTree = no_block_spec, scratch_shapes: pallas_core.ScratchShapeTree = (), input_output_aliases: Mapping[int, int] = {}, debug: bool = False, interpret: Any = False, name: str | None = None, compiler_params: pallas_core.CompilerParams | None = None, cost_estimate: CostEstimate | None = None, metadata: dict[str, str] | None = None)` — [`L1122`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L1122) — Entry point for creating a Pallas kernel.
- `temp_f(*args)` — [`L570`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L570)
- `tpu_lowering(ctx: mlir.LoweringRuleContext, *in_nodes: ir.Value | Sequence[ir.Value], **params)` — [`L880`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L880)
- `wrapped(*args)` — [`L1266`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L1266)

## Module values
- `BlockMapping` — [`L64`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L64)
- `CompilerParams` — [`L68`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L68)
- `CostEstimate` — [`L67`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L67)
- `GridMapping` — [`L65`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L65)
- `no_block_spec` — [`L66`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L66)
- `pallas_call_p` — [`L71`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L71)
- `unsafe_map` — [`L61`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L61)
- `unsafe_zip` — [`L62`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_call.py#L62)

