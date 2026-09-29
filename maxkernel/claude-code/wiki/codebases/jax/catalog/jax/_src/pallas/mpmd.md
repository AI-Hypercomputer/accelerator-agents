---
title: 'Module: jax/_src/pallas/mpmd.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mpmd.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mpmd`/
symbols:
  _mpmd_map.wrapper: _mpmd_map().wrapper().
  _dedup_consts_and_unify_jaxpr_signatures: _dedup_consts_and_unify_jaxpr_signatures().
  mpmd_map_p: mpmd_map_p.
  _mpmd_map_discharge_rule: _mpmd_map_discharge_rule().
  _mpmd_map_fallback_lowering: _mpmd_map_fallback_lowering().
  _mpmd_map: _mpmd_map().
  _mpmd_map_to_lojax: _mpmd_map_to_lojax().
  _mpmd_map_abstract_eval: _mpmd_map_abstract_eval().
  _mpmd_map_discharge_rule._rewrite_to_include_new_outputs: _mpmd_map_discharge_rule()._rewrite_to_include_new_outputs().
  _mpmd_map_batching_rule: _mpmd_map_batching_rule().
  mpmd_map_tracing_context: mpmd_map_tracing_context().
  mpmd_map: mpmd_map().
  _aval_to_ref_aval: _aval_to_ref_aval().
  _error_if_non_ref_consts: _error_if_non_ref_consts().
  _mpmd_map_impl: _mpmd_map_impl().
  _mpmd_map_lowering: _mpmd_map_lowering().
  _mpmd_map_discharge_rule._rewrite_to_include_new_outputs.new_body: _mpmd_map_discharge_rule()._rewrite_to_include_new_outputs().new_body().
  get_super_mesh_shape: get_super_mesh_shape().
  _mpmd_map_dce_rule: _mpmd_map_dce_rule().
  _mpmd_map_partial_eval_custom: _mpmd_map_partial_eval_custom().
  _mpmd_map_tpu_lowering: _mpmd_map_tpu_lowering().
  _dedup_consts_and_unify_jaxpr_signatures.make_rewritten_body._rewritten_body: _dedup_consts_and_unify_jaxpr_signatures().make_rewritten_body()._rewritten_body().
  _dedup_consts_and_unify_jaxpr_signatures.make_rewritten_body: _dedup_consts_and_unify_jaxpr_signatures().make_rewritten_body().
  _mpmd_map_typecheck_rule: _mpmd_map_typecheck_rule().
  _get_unique_consts: _get_unique_consts().
  _T: _T.
  _mpmd_map_is_high: _mpmd_map_is_high().
---
# Module: [`jax/_src/pallas/mpmd.py`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py)

## Functions
- `_aval_to_ref_aval(aval: Any, meshes: Sequence[pallas_core.Mesh])` — [`L641`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L641)
- `_dedup_consts_and_unify_jaxpr_signatures(jaxprs: Sequence[jax_core.Jaxpr], consts_per_fn: Sequence[Sequence[Any]], flat_args: Sequence[Any], unflat_in_avals: Sequence[jax_core.AbstractValue], unflat_out_avals: Sequence[jax_core.AbstractValue], flat_kernel_avals: Sequence[jax_core.AbstractValue], meshes: Sequence[pallas_core.Mesh], all_meshes: tuple[pallas_core.Mesh, ...])` — [`L704`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L704)
- `_error_if_non_ref_consts(consts, debug_info)` — [`L668`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L668)
- `_get_unique_consts(consts: Sequence[Sequence[Any]])` — [`L691`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L691)
- `_mpmd_map(meshes_and_fns: Sequence[tuple[pallas_core.Mesh, Callable[..., None]]],, out_types: tree_util.PyTree = (), *, input_output_aliases: Mapping[int, int] = {}, scratch_types: pallas_core.ScratchShapeTree = (), compiler_params: Any | None = None, interpret: bool | Any = False, debug: bool = False, cost_estimate: pallas_core.CostEstimate | None = None, name: str | None = None, metadata: dict[str, str] | None = None)` — [`L780`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L780) — Like ``pallas_call``, but MPMD and without pipelining.
- `_mpmd_map_abstract_eval(*in_avals, jaxprs, out_avals, input_output_aliases, interpret, compiler_params, meshes, **params)` — [`L89`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L89)
- `_mpmd_map_batching_rule(axis_data, args, dims, *, jaxprs, meshes, out_avals, input_output_aliases, **params)` — [`L298`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L298)
- `_mpmd_map_dce_rule(used_outs: list[bool], eqn: pe.JaxprEqn)` — [`L260`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L260)
- `_mpmd_map_discharge_rule(avals_in: Sequence[jax_core.AbstractValue], avals_out: Sequence[jax_core.AbstractValue], *args: Any, jaxprs, meshes, input_output_aliases, debug, interpret, compiler_params, cost_estimate, metadata, name, external_meshes, **_)` — [`L153`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L153)
- `_mpmd_map_fallback_lowering(ctx: mlir.LoweringRuleContext, *in_nodes, meshes, jaxprs, out_avals, input_output_aliases, compiler_params, interpret, debug, cost_estimate, metadata, name, external_meshes)` — [`L492`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L492)
- `_mpmd_map_impl(*args, **params)` — [`L82`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L82)
- `_mpmd_map_is_high(*args, jaxprs, **params)` — [`L378`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L378)
- `_mpmd_map_lowering(ctx: mlir.LoweringRuleContext, *in_nodes, **params)` — [`L595`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L595)
- `_mpmd_map_partial_eval_custom(saveable, unks_in, inst_in, eqn)` — [`L268`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L268)
- `_mpmd_map_to_lojax(*hi_args, meshes, jaxprs, external_meshes, out_avals, input_output_aliases, compiler_params, interpret, debug, cost_estimate, metadata, name, **params)` — [`L386`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L386)
- `_mpmd_map_tpu_lowering(ctx: mlir.LoweringRuleContext, *in_nodes, jaxprs, meshes, input_output_aliases, debug, interpret, compiler_params, cost_estimate, out_avals, metadata, name, external_meshes)` — [`L454`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L454)
- `_mpmd_map_typecheck_rule(ctx_factory, *in_atoms, **params)` — [`L141`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L141)
- `_rewrite_to_include_new_outputs(jaxpr: jax_core.Jaxpr)` — [`L190`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L190)
- `_rewritten_body(*args)` — [`L745`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L745)
- `get_super_mesh_shape(meshes: Iterable[pallas_core.Mesh])` — [`L49`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L49)
- `make_rewritten_body(original_jaxpr, original_consts)` — [`L744`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L744)
- `mpmd_map(meshes_and_fns: Sequence[tuple[pallas_core.Mesh, Callable[..., None]]],, out_types: tree_util.PyTree = (), *, scratch_types: pallas_core.ScratchShapeTree = (), compiler_params: Any | None = None, interpret: bool | Any = False, debug: bool = False, cost_estimate: pallas_core.CostEstimate | None = None, name: str | None = None, metadata: dict[str, str] | None = None)` — [`L611`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L611)
- `mpmd_map_tracing_context(mesh: pallas_core.Mesh, other_meshes: tuple[pallas_core.Mesh, ...])` — [`L64`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L64)
- `new_body(*args)` — [`L192`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L192)
- `wrapper(*args)` — [`L810`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L810)

## Module values
- `_T` — [`L46`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L46)
- `mpmd_map_p` — [`L77`](../../../../../../../raw/code/jax/jax/_src/pallas/mpmd.py#L77)

