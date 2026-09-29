---
title: 'Module: jax/_src/lax/control_flow/conditionals.py'
type: catalog
provenance: extracted
module: jax/_src/lax/control_flow/conditionals.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.lax.control_flow.conditionals`/
symbols:
  cond_p: cond_p.
  cond: cond().
  _switch_internal: _switch_internal().
  _cond_partial_eval: _cond_partial_eval().
  _cond_batching_rule: _cond_batching_rule().
  _cond_partial_eval_custom: _cond_partial_eval_custom().
  _cond_transpose_fancy: _cond_transpose_fancy().
  _cond_state_discharge_rule: _cond_state_discharge_rule().
  _check_branch_outputs: _check_branch_outputs().
  _cond_dce_rule: _cond_dce_rule().
  _cond_remat: _cond_remat().
  _cond_linearize: _cond_linearize().
  platform_dependent: platform_dependent().
  platform_index_p: platform_index_p.
  _cond_typecheck: _cond_typecheck().
  _cond_lowering: _cond_lowering().
  switch: switch().
  _cond_jvp: _cond_jvp().
  _join_cond_effects: _join_cond_effects().
  _join_cond_outputs.augment_jaxpr: _join_cond_outputs().augment_jaxpr().
  _merge_branch_residuals: _merge_branch_residuals().
  _join_cond_pe_staged_jaxpr_inputs: _join_cond_pe_staged_jaxpr_inputs().
  _cond_abstract_eval: _cond_abstract_eval().
  _join_cond_outputs.augment_jaxpr.f_aug: _join_cond_outputs().augment_jaxpr().f_aug().
  _join_cond_pe_staged_jaxpr_inputs.augment_jaxpr: _join_cond_pe_staged_jaxpr_inputs().augment_jaxpr().
  _join_cond_outputs: _join_cond_outputs().
  _cond_linearize.tangent_fun: _cond_linearize().tangent_fun().
  _cond_to_lojax: _cond_to_lojax().
  _platform_index_aval: _platform_index_aval().
  _transpose_jaxpr_fancy: _transpose_jaxpr_fancy().
  _no_operand_sentinel: _no_operand_sentinel.
  _bcast_select: _bcast_select().
  _bcast_select_n: _bcast_select_n().
  _cond_remat.rem: _cond_remat().rem().
  BranchesPlatforms: BranchesPlatforms.
  _platform_index_lowering: _platform_index_lowering().
  _transpose_jaxpr_fancy.transposed: _transpose_jaxpr_fancy().transposed().
  unsafe_map: unsafe_map.
  unsafe_zip: unsafe_zip.
  _T: _T.
  _capitalize: _capitalize().
  _merge_branch_residuals.enumerate_equal: _merge_branch_residuals().enumerate_equal().
  _ordered_unique: _ordered_unique().
  _cond_is_high: _cond_is_high().
  _platform_index_lowering.lower_constant: _platform_index_lowering().lower_constant().
---
# Module: [`jax/_src/lax/control_flow/conditionals.py`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py)

## Functions
- `_bcast_select(pred, on_true, on_false)` — [`L451`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L451)
- `_bcast_select_n(pred, *cases)` — [`L457`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L457)
- `_capitalize(s)` — [`L419`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L419)
- `_check_branch_outputs(api_name, name1, name2, f1, f2, out_avals1, out_avals2)` — [`L337`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L337)
- `_cond_abstract_eval(*avals: core.AbstractValue, branches: Sequence[core.ClosedJaxpr], **_)` — [`L433`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L433)
- `_cond_batching_rule(axis_data, args, dims, *, branches, **params)` — [`L463`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L463)
- `_cond_dce_rule(used_outputs: list[bool], eqn: core.JaxprEqn)` — [`L811`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L811)
- `_cond_is_high(*_, branches, **__)` — [`L982`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L982)
- `_cond_jvp(primals, tangents, *, branches, **params)` — [`L568`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L568)
- `_cond_linearize(is_vjp, nzs, *primals_in, branches, **params)` — [`L529`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L529)
- `_cond_lowering(ctx, index, *args, branches, **params)` — [`L1015`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L1015)
- `_cond_partial_eval(trace, *tracers, branches, **params)` — [`L594`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L594)
- `_cond_partial_eval_custom(saveable, unks_in, inst_in, eqn)` — [`L655`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L655)
- `_cond_remat(policy, *args, branches, **params)` — [`L938`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L938)
- `_cond_state_discharge_rule(should_discharge, in_avals, out_avals, index, *args, branches, **params)` — [`L1087`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L1087)
- `_cond_to_lojax(pred, *hi_args, branches, **kwds)` — [`L986`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L986)
- `_cond_transpose_fancy(cts_in, index, *args, branches, **params)` — [`L847`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L847)
- `_cond_typecheck(bind_time, *in_atoms, branches, **params)` — [`L883`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L883)
- `_join_cond_effects(branches: Sequence[core.ClosedJaxpr])` — [`L423`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L423)
- `_join_cond_outputs(jaxprs: Sequence[core.ClosedJaxpr], all_res_avals, res_aval_indices_per_jaxpr, num_non_res_outputs)` — [`L771`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L771)
- `_join_cond_pe_staged_jaxpr_inputs(jaxprs: Sequence[core.ClosedJaxpr], all_res_avals, res_aval_indices_per_jaxpr)` — [`L790`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L790)
- `_merge_branch_residuals(branch_res_avals)` — [`L756`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L756)
- `_ordered_unique(xs)` — [`L807`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L807)
- `_platform_index_aval(*_, **__)` — [`L1224`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L1224)
- `_platform_index_lowering(ctx: mlir.LoweringRuleContext, *, platforms: BranchesPlatforms)` — [`L1227`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L1227)
- `_switch_internal(index: ArrayLike, branches: Sequence[Callable], operands: Sequence[Any], *, branches_platforms: BranchesPlatforms | None)` — [`L136`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L136)
- `_transpose_jaxpr_fancy(jaxpr, in_tree, in_avals, specs, inst_out)` — [`L867`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L867)
- `augment_jaxpr(jaxpr: core.ClosedJaxpr, res_indices)` — [`L774`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L774)
- `augment_jaxpr(jaxpr: core.ClosedJaxpr, res_indices)` — [`L795`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L795)
- `cond(pred, true_fun: Callable, false_fun: Callable, *operands, operand=_no_operand_sentinel)` — [`L193`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L193) — Conditionally apply ``true_fun`` or ``false_fun``.
- `enumerate_equal(xs)` — [`L757`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L757)
- `f_aug(*args)` — [`L776`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L776)
- `lower_constant(ctx: mlir.LoweringRuleContext, *, i: int)` — [`L1230`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L1230)
- `platform_dependent(*args: Any, default: Callable[..., _T] | None = None, **per_platform: Callable[..., _T])` — [`L1134`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L1134) — Stages out platform-specific code.
- `rem(idx, *args)` — [`L953`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L953)
- `switch(index, branches: Sequence[Callable], *operands: Any, operand: Any = _no_operand_sentinel)` — [`L69`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L69) — Apply exactly one of the ``branches`` given by ``index``.
- `tangent_fun(res, *tangents_in)` — [`L554`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L554)
- `transposed(*in_flat)` — [`L869`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L869)

## Module values
- `BranchesPlatforms` — [`L958`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L958)
- `_T` — [`L1133`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L1133)
- `_no_operand_sentinel` — [`L66`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L66)
- `cond_p` — [`L967`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L967)
- `platform_index_p` — [`L1218`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L1218)
- `unsafe_map` — [`L59`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L59)
- `unsafe_zip` — [`L60`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/conditionals.py#L60)

