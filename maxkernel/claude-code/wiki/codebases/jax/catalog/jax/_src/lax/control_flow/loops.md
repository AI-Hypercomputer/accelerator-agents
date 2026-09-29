---
title: 'Module: jax/_src/lax/control_flow/loops.py'
type: catalog
provenance: extracted
module: jax/_src/lax/control_flow/loops.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.lax.control_flow.loops`/
symbols:
  scan_p: scan_p.
  scan: scan().
  while_p: while_p.
  while_loop: while_loop().
  fori_loop: fori_loop().
  _scan_partial_eval_custom: _scan_partial_eval_custom().
  zip: zip.
  _scan_partial_eval: _scan_partial_eval().
  _scan_impl: _scan_impl().
  _map: _map.
  _scan_dce_rule: _scan_dce_rule().
  scan_nocarry: scan_nocarry().
  _scan_transpose_fancy: _scan_transpose_fancy().
  _scan_to_lojax: _scan_to_lojax().
  _while_partial_discharge_rule: _while_partial_discharge_rule().
  _while_to_lojax: _while_to_lojax().
  _scan_linearize: _scan_linearize().
  _while_partial_eval_custom: _while_partial_eval_custom().
  _scan_state_partial_discharge_rule: _scan_state_partial_discharge_rule().
  _scan_typecheck: _scan_typecheck().
  _while_loop_jvp: _while_loop_jvp().
  Scan3.jvp: Scan3#jvp().
  _check_carry_type: _check_carry_type().
  _while_partial_eval: _while_partial_eval().
  _cumulative_reduction_primitive: _cumulative_reduction_primitive().
  _rearrange_mutable_binders: _rearrange_mutable_binders().
  associative_scan: associative_scan().
  scan3: scan3().
  _scan_leaf: _scan_leaf().
  _scan_jvp: _scan_jvp().
  _while_loop_batching_rule: _while_loop_batching_rule().
  _scan_impl.inner: _scan_impl().inner().
  _remainder_leaf: _remainder_leaf().
  associative_scan._scan: associative_scan()._scan().
  cumsum_p: cumsum_p.
  map: map().
  Scan3.body_fun: Scan3#body_fun().
  _scan_linearize.tangent_fun: _scan_linearize().tangent_fun().
  _scan_batching_rule: _scan_batching_rule().
  _scan_state_partial_discharge_rule.body: _scan_state_partial_discharge_rule().body().
  _while_lowering: _while_lowering().
  _empty_array: _empty_array().
  cumsum: cumsum().
  Scan3.__init__: Scan3#__init__().
  Scan3.expand: Scan3#expand().
  _infer_scan_length: _infer_scan_length().
  _scan_known_hoisting: _scan_known_hoisting().
  while_loop._create_jaxpr: while_loop()._create_jaxpr().
  _scan_impl.body_fun: _scan_impl().body_fun().
  _scan_partial_eval_custom.known: _scan_partial_eval_custom().known().
  _scan_remat: _scan_remat().
  _while_lowering.fun: _while_lowering().fun().
  _while_partial_discharge_rule.new_body: _while_partial_discharge_rule().new_body().
  _while_partial_discharge_rule.new_cond: _while_partial_discharge_rule().new_cond().
  _insert_binders: _insert_binders().
  _pred_bcast_select_hlo: _pred_bcast_select_hlo().
  _promote_weak_typed_input: _promote_weak_typed_input().
  cumprod: cumprod().
  cumlogsumexp_p: cumlogsumexp_p.
  cumprod_p: cumprod_p.
  cummax_p: cummax_p.
  cummin_p: cummin_p.
  Carry: Carry.
  _scan_abstract_eval: _scan_abstract_eval().
  _transpose_scan_jaxpr_fancy: _transpose_scan_jaxpr_fancy().
  _fori_body_fun: _fori_body_fun().
  _fori_scan_body_fun: _fori_scan_body_fun().
  _batch_and_remainder: _batch_and_remainder().
  _interleave: _interleave().
  _stack: _stack().
  _while_loop_abstract_eval: _while_loop_abstract_eval().
  _cumulative_jvp_rule: _cumulative_jvp_rule().
  _maybe_put: _maybe_put().
  _transpose_scan_jaxpr_fancy.transposed: _transpose_scan_jaxpr_fancy().transposed().
  _rng_bit_generator_batching_rule: _rng_bit_generator_batching_rule().
  _cumred_gpu_lowering: _cumred_gpu_lowering().
  cummax: cummax().
  cummin: cummin().
  cumlogsumexp: cumlogsumexp().
  _scan_state_partial_discharge_rule.rearrange: _scan_state_partial_discharge_rule().rearrange().
  _join_while_effects: _join_while_effects().
  associative_scan.combine: associative_scan().combine().
  _constrain_unreduced: _constrain_unreduced().
  _while_lowering.cond: _while_lowering().cond().
  _fori_body_fun.while_body_fun: _fori_body_fun().while_body_fun().
  X: X.
  Y: Y.
  scan3.body_no_carry: scan3().body_no_carry().
  _scan_remat.rem: _scan_remat().rem().
  _move_right: _move_right().
  _while_lowering.new_body: _while_lowering().new_body().
  T: T.
  Scan3: Scan3#
  Scan3.jaxpr: Scan3#jaxpr.
  keep: keep().
  _split_leading: _split_leading().
  _concat: _concat().
  scan3.write_carry: scan3().write_carry().
  _while_lowering.body: _while_lowering().body().
  _while_typecheck: _while_typecheck().
  _fori_cond_fun: _fori_cond_fun().
  _cumred_sharding_rule: _cumred_sharding_rule().
  _cumsum_transpose_rule: _cumsum_transpose_rule().
  _cumred_dtype_rule: _cumred_dtype_rule().
  Scan3.extensives: Scan3#extensives.
  cumred_reduce_window_impl: cumred_reduce_window_impl().
  Scan3.length: Scan3#length.
  scan3.read_carry: scan3().read_carry().
  scan._create_jaxpr: scan()._create_jaxpr().
  _capitalize: _capitalize().
  _fori_scan_body_fun.scanned_fun: _fori_scan_body_fun().scanned_fun().
  _scan_leaf.f: _scan_leaf().f().
  _remainder_leaf.f: _remainder_leaf().f().
  BooleanNumeric: BooleanNumeric.
  Scan3.reverse: Scan3#reverse.
  Scan3.unroll: Scan3#unroll.
  Scan3.slice_arg: Scan3#slice_arg().
  _scan_impl.cond_fun: _scan_impl().cond_fun().
  _scan_is_high: _scan_is_high().
  _while_transpose_error: _while_transpose_error().
  _while_lowering.new_cond: _while_lowering().new_cond().
  _while_is_high: _while_is_high().
  _cumred_chlo_lowering: _cumred_chlo_lowering().
  _is_supported_cumred: _is_supported_cumred().
  _cumred_shape_rule: _cumred_shape_rule().
  _cumred_batch_rule: _cumred_batch_rule().
  _cumulative_reduction_primitive.register_lowering: _cumulative_reduction_primitive().register_lowering().
---
# Module: [`jax/_src/lax/control_flow/loops.py`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py)

## Classes
### `Scan3`  ·  implements/extends VJPHiPrimitive
- def: [`jax/_src/lax/control_flow/loops.py:111`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L111)
- signature: `class Scan3(hijax.VJPHiPrimitive):`
- members:
  - `body_fun(i)` — [`L139`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L139)
  - `expand(self, args)` — [`L134`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L134)
  - `jvp(self, primals, tangents)` — [`L153`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L153)
  - `slice_arg(extensive, arg)` — [`L140`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L140)
  - `extensives` — [`L113`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L113)
  - `jaxpr` — [`L115`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L115)
  - `length` — [`L114`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L114)
  - `reverse` — [`L116`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L116)
  - `unroll` — [`L117`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L117)
- protocol/private: `__init__`[`L119`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L119)
- uses (calls/refs, reference-scoped): [`Zero`](../../ad_util.md#Zero), [`jaxpr`](../../core.md#ClosedJaxpr.jaxpr), [`ClosedJaxpr`](../../core.md#ClosedJaxpr), [`eval_jaxpr`](../../core.md#eval_jaxpr), [`while_loop`](loops.md#while_loop), [`zip`](loops.md#zip), [`zeros_like_aval`](../../ad_util.md#zeros_like_aval), [`VJPHiPrimitive`](../../hijax.md#VJPHiPrimitive), [`in_avals`](../../hijax.md#VJPHiPrimitive.in_avals), [`__init__`](../../hijax.md#VJPHiPrimitive.__init__), [`out_aval`](../../hijax.md#VJPHiPrimitive.out_aval), [`split_list_checked`](../../util.md#split_list_checked), [`params`](../../hijax.md#VJPHiPrimitive.params), [`new_ref`](../../core.md#new_ref), [`unmapped_leading_aval`](../../core.md#unmapped_leading_aval), [`a2tz`](../../ad_util.md#a2tz), [`keep`](loops.md#keep)
- used by: [`scan_nocarry`](loops.md#scan_nocarry), [`expand`](../../hijax.md#VJPHiPrimitive.expand), [`VJPHiPrimitive`](../../hijax.md#VJPHiPrimitive), [`jvp`](../../hijax.md#VJPHiPrimitive.jvp)

## Functions
- `_batch_and_remainder(x, batch_size: int)` — [`L2644`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2644)
- `_capitalize(s)` — [`L533`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L533)
- `_check_carry_type(name, body_fun, in_carry, out_carry)` — [`L537`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L537)
- `_concat(a, b)` — [`L716`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L716)
- `_constrain_unreduced(val)` — [`L708`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L708)
- `_create_jaxpr(carry_avals)` — [`L386`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L386)
- `_create_jaxpr(init_avals)` — [`L1613`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1613)
- `_cumred_batch_rule(prim, batched_args, batch_dims, *, axis: int, reverse: bool)` — [`L3029`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L3029)
- `_cumred_chlo_lowering(ctx, x, *, axis, reverse, reducer, identity)` — [`L2918`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2918)
- `_cumred_dtype_rule(name, operand, *args, **kw)` — [`L3036`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L3036)
- `_cumred_gpu_lowering(reduce_window_fn: Callable, reducer: Callable, identity: Callable, ctx, x, *, axis, reverse)` — [`L2957`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2957)
- `_cumred_shape_rule(x, *, axis: int, reverse: bool)` — [`L2997`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2997)
- `_cumred_sharding_rule(x, *, axis: int, reverse: bool)` — [`L3005`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L3005)
- `_cumsum_transpose_rule(t, operand, *, axis: int, reverse: bool)` — [`L3012`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L3012)
- `_cumulative_jvp_rule(primals, tangents, *, axis: int, reverse: bool, combine_fn: Callable)` — [`L3079`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L3079)
- `_cumulative_reduction_primitive(name, reduce_fn, reduce_window_fn)` — [`L3043`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L3043)
- `_empty_array(prefix, length_spec, aval)` — [`L718`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L718)
- `_fori_body_fun(body_fun: Callable, body_fun_dbg: core.DebugInfo)` — [`L2454`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2454)
- `_fori_cond_fun(loop_carry)` — [`L2449`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2449)
- `_fori_scan_body_fun(body_fun: Callable, body_fun_dbg: core.DebugInfo)` — [`L2474`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2474)
- `_infer_scan_length(xs_flat: list[Any], xs_avals: list[AbstractValue], length: Any | None)` — [`L483`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L483)
- `_insert_binders(jaxpr, n_after, vals)` — [`L2420`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2420)
- `_interleave(a, b, axis)` — [`L2905`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2905) — Given two Tensors of static shape, interleave them along the first axis.
- `_is_supported_cumred(inp, axis, reverse)` — [`L2945`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2945)
- `_join_while_effects(body_jaxpr, cond_jaxpr, body_nconsts, cond_nconsts)` — [`L1698`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1698)
- `_maybe_put(x)` — [`L1017`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1017)
- `_move_right(lst, to_move)` — [`L1540`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1540)
- `_pred_bcast_select_hlo(ctx, pred_aval: core.ShapedArray, pred: ir.Value, x: mlir.IrValues, y: mlir.IrValues, x_y_aval: core.AbstractValue)` — [`L2428`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2428)
- `_promote_weak_typed_input(in_val: Any, in_aval: AbstractValue, out_aval: AbstractValue)` — [`L96`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L96)
- `_rearrange_mutable_binders(jaxpr: ClosedJaxpr, num_prefix: int, num_binders: int)` — [`L1032`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1032)
- `_remainder_leaf(leaf, batch_elems)` — [`L2634`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2634)
- `_rng_bit_generator_batching_rule(batched_args, batch_dims, *, shape, dtype, algorithm, out_sharding)` — [`L2740`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2740)
- `_scan(elems)` — [`L2864`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2864) — Perform scan on `elems`.
- `_scan_abstract_eval(*args, reverse, length, num_consts, num_carry, jaxpr, unroll)` — [`L736`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L736)
- `_scan_batching_rule(axis_data, args, dims, reverse, length, jaxpr, num_consts, num_carry, unroll)` — [`L1141`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1141)
- `_scan_dce_rule(used_outputs: list[bool], eqn: core.JaxprEqn)` — [`L1186`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1186)
- `_scan_impl(*args, reverse, length, num_consts, num_carry, jaxpr, unroll)` — [`L624`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L624)
- `_scan_is_high(*_, jaxpr, **__)` — [`L1484`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1484)
- `_scan_jvp(primals, tangents, reverse, length, jaxpr, num_consts, num_carry, unroll)` — [`L753`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L753)
- `_scan_known_hoisting(jaxpr_known, known_consts, num_res)` — [`L886`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L886)
- `_scan_leaf(leaf, batch_elems, num_batches, batch_size)` — [`L2618`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2618)
- `_scan_linearize(is_vjp, nzs, *primals_in, reverse: bool, length: int, num_consts: int, num_carry: int, jaxpr: ClosedJaxpr, unroll: int)` — [`L803`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L803)
- `_scan_partial_eval(trace, *tracers, reverse: bool, length: int, num_consts: int, num_carry: int, jaxpr: ClosedJaxpr, unroll: int)` — [`L907`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L907)
- `_scan_partial_eval_custom(saveable, unks_in, inst_in, eqn: core.JaxprEqn)` — [`L1229`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1229)
- `_scan_remat(policy, *args, jaxpr, **params)` — [`L1456`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1456)
- `_scan_state_partial_discharge_rule(should_discharge, in_avals, out_avals, *args, jaxpr, num_consts, num_carry, unroll, reverse, length)` — [`L1384`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1384)
- `_scan_to_lojax(*hi_args, jaxpr, num_carry, num_consts, **params)` — [`L1488`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1488)
- `_scan_transpose_fancy(cts, *args, reverse, length, num_consts, num_carry, jaxpr, unroll)` — [`L1052`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1052)
- `_scan_typecheck(bind_time, *in_atoms, reverse, length, num_consts, num_carry, jaxpr, unroll)` — [`L1343`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1343)
- `_split_leading(sz, x)` — [`L712`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L712)
- `_stack(arrs: Sequence[Array], axis: int = 0)` — [`L93`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L93)
- `_transpose_scan_jaxpr_fancy(jaxpr, trans_tree, trans_avals, lin_refs, immut_xs_avals)` — [`L1121`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1121)
- `_while_is_high(*_, cond_jaxpr, body_jaxpr, **__)` — [`L2375`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2375)
- `_while_loop_abstract_eval(*avals, cond_jaxpr, body_jaxpr, body_nconsts, cond_nconsts)` — [`L1714`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1714)
- `_while_loop_batching_rule(axis_data, args, dims, cond_nconsts, cond_jaxpr, body_nconsts, body_jaxpr)` — [`L1748`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1748)
- `_while_loop_jvp(primals, tangents, cond_nconsts, cond_jaxpr, body_nconsts, body_jaxpr)` — [`L1825`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1825)
- `_while_lowering(ctx, *args, cond_jaxpr, body_jaxpr, cond_nconsts, body_nconsts)` — [`L2051`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2051)
- `_while_partial_discharge_rule(should_discharge, in_avals, out_avals, *args, cond_jaxpr, body_jaxpr, cond_nconsts, body_nconsts)` — [`L2193`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2193)
- `_while_partial_eval(trace: pe.JaxprTrace, *tracers: pe.Tracer, cond_nconsts: int, cond_jaxpr: pe.ClosedJaxpr, body_nconsts: int, body_jaxpr: pe.ClosedJaxpr)` — [`L1881`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1881)
- `_while_partial_eval_custom(saveable, unks_in, inst_in, eqn)` — [`L1945`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1945)
- `_while_to_lojax(*hi_args, cond_jaxpr, body_jaxpr, cond_nconsts, body_nconsts)` — [`L2379`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2379)
- `_while_transpose_error(*_, **kwargs)` — [`L2020`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2020)
- `_while_typecheck(_, *in_atoms, cond_jaxpr, body_jaxpr, cond_nconsts, body_nconsts)` — [`L2182`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2182)
- `associative_scan(fn: Callable, elems, reverse: bool = False, axis: int = 0)` — [`L2764`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2764) — Performs a scan with an associative binary operation, in parallel.
- `body(*consts_carry_xs)` — [`L1403`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1403)
- `body(args)` — [`L2063`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2063)
- `body_fun(while_carry)` — [`L667`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L667)
- `body_no_carry(x)` — [`L241`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L241)
- `combine(a_flat, b_flat)` — [`L2827`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2827)
- `cond(args)` — [`L2057`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2057)
- `cond_fun(while_carry)` — [`L686`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L686)
- `cumlogsumexp(operand: Array, axis: int = 0, reverse: bool = False)` — [`L2993`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2993) — Computes a cumulative logsumexp along `axis`.
- `cummax(operand: Array, axis: int = 0, reverse: bool = False)` — [`L2985`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2985) — Computes a cumulative maximum along `axis`.
- `cummin(operand: Array, axis: int = 0, reverse: bool = False)` — [`L2989`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2989) — Computes a cumulative minimum along `axis`.
- `cumprod(operand: Array, axis: int = 0, reverse: bool = False)` — [`L2981`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2981) — Computes a cumulative product along `axis`.
- `cumred_reduce_window_impl(window_reduce: Callable, x, *, axis: int, reverse: bool)` — [`L3016`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L3016)
- `cumsum(operand: Array, axis: int = 0, reverse: bool = False)` — [`L2977`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2977) — Computes a cumulative sum along `axis`.
- `f(l)` — [`L2619`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2619)
- `f(l)` — [`L2635`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2635)
- `fori_loop(lower, upper, body_fun, init_val, *, unroll: int | bool | None = None)` — [`L2486`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2486) — Loop from ``lower`` to ``upper`` by reduction to :func:`jax.lax.while_loop`.
- `fun(*args)` — [`L2073`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2073)
- `inner(n, carry, xs)` — [`L653`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L653)
- `keep(keeps, xs)` — [`L169`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L169)
- `known(*ins_known)` — [`L1303`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1303)
- `map(f, xs, *, batch_size: int | None = None)` — [`L2667`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2667) — Map a function over leading array axes.
- `new_body(pred_args)` — [`L2068`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2068)
- `new_body(*consts_refs_carry)` — [`L2269`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2269)
- `new_cond(pred_args)` — [`L2065`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2065)
- `new_cond(*consts_refs_carry)` — [`L2316`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2316)
- `read_carry()` — [`L232`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L232)
- `rearrange(lst)` — [`L1416`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1416)
- `register_lowering(fn, platform=None)` — [`L3051`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L3051)
- `rem(*args)` — [`L1461`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1461)
- `scan(f: Callable[[Carry, X], tuple[Carry, Y]], init: Carry, xs: X | None = None, length: int | None = None, reverse: bool = False, unroll: int | bool = 1, _split_transpose: bool = False)` — [`L253`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L253) — Scan a function over leading array axes while carrying along state.
- `scan3(f: Callable[[Carry, X], tuple[Carry, Y]], init: Carry, xs: X | None = None, length: int | None = None, reverse: bool = False, unroll: int | bool = 1, _split_transpose: bool = False)` — [`L221`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L221)
- `scan_nocarry(f: Callable[[Carry, X], tuple[Carry, Y]], xs: X | None = None, length: int | None = None, reverse: bool = False, unroll: int | bool = 1)` — [`L173`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L173)
- `scanned_fun(loop_carry, _)` — [`L2476`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2476)
- `tangent_fun(res, *tangents)` — [`L868`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L868)
- `transposed(*args)` — [`L1123`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1123)
- `while_body_fun(loop_carry)` — [`L2457`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2457)
- `while_loop(cond_fun: Callable[[T], BooleanNumeric], body_fun: Callable[[T], T], init_val: T)` — [`L1548`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1548) — Call ``body_fun`` repeatedly in a loop while ``cond_fun`` is True.
- `write_carry(val)` — [`L235`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L235)

## Module values
- `BooleanNumeric` — [`L89`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L89)
- `Carry` — [`L107`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L107)
- `T` — [`L88`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L88)
- `X` — [`L108`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L108)
- `Y` — [`L109`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L109)
- `_map` — [`L85`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L85)
- `cumlogsumexp_p` — [`L3069`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L3069)
- `cummax_p` — [`L3073`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L3073)
- `cummin_p` — [`L3075`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L3075)
- `cumprod_p` — [`L3071`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L3071)
- `cumsum_p` — [`L3065`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L3065)
- `scan_p` — [`L1465`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L1465)
- `while_p` — [`L2361`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L2361)
- `zip` — [`L86`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/loops.py#L86)

