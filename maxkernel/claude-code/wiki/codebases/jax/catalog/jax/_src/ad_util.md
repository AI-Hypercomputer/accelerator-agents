---
title: 'Module: jax/_src/ad_util.py'
type: catalog
provenance: extracted
module: jax/_src/ad_util.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.
symbols:
  Zero: ad_util`/Zero#
  p2tz: ad_util`/p2tz().
  zeros_like_aval: ad_util`/zeros_like_aval().
  add_jaxvals: ad_util`/add_jaxvals().
  instantiate: ad_util`/instantiate().
  empty_like_aval: ad_util`/empty_like_aval().
  stop_gradient_p: core`/stop_gradient_p#
  SymbolicZero: ad_util`/SymbolicZero#
  replace_rule_output_symbolic_zeros: ad_util`/replace_rule_output_symbolic_zeros().
  zero_from_primal.f: ad_util`/zero_from_primal().f().
  replace_internal_symbolic_zeros: ad_util`/replace_internal_symbolic_zeros().
  zeros_like_jaxval: ad_util`/zeros_like_jaxval().
  Zero.instantiate: ad_util`/Zero#instantiate().
  add_impl: ad_util`/add_impl().
  SymbolicZero.__getattr__: ad_util`/SymbolicZero#__getattr__().
  raw_jaxval_adders: ad_util`/raw_jaxval_adders.
  add_jaxvals_p: ad_util`/add_jaxvals_p.
  add_any_p: ad_util`/add_any_p.
  _stop_gradient_impl: ad_util`/_stop_gradient_impl().
  SymbolicZero.aval: ad_util`/SymbolicZero#aval.
  a2tz: ad_util`/a2tz().
  aval_zeros_likers.aval_zeros_likers: ad_util`/aval_zeros_likers.aval_zeros_likers.
  Zero.__init__: ad_util`/Zero#__init__().
  p2cz: ad_util`/p2cz().
  SymbolicZero.from_primal_value: ad_util`/SymbolicZero#from_primal_value().
  zero_from_primal: ad_util`/zero_from_primal().
  aval_empty_likers.aval_empty_likers: ad_util`/aval_empty_likers.aval_empty_likers.
  JaxTypeOrTracer: ad_util`/JaxTypeOrTracer.
  map: ad_util`/map.
  add_abstract: ad_util`/add_abstract().
  Zero.__repr__: ad_util`/Zero#__repr__().
  SymbolicZero.__init__: ad_util`/SymbolicZero#__init__().
  zeros_like_p.zeros_like_p: ad_util`/zeros_like_p.zeros_like_p.
  T: ad_util`/T.
  Zero.__slots__: ad_util`/Zero#__slots__.
  SymbolicZero.__repr__: ad_util`/SymbolicZero#__repr__().
---
# Module: [`jax/_src/ad_util.py`](../../../../../../raw/code/jax/jax/_src/ad_util.py)

## Classes
### `SymbolicZero`
- def: [`jax/_src/ad_util.py:111`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L111)
- signature: `class SymbolicZero:`
- members:
  - `from_primal_value(val: Any)` — [`L138`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L138)
  - `aval` — [`L113`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L113)
- protocol/private: `__getattr__`[`L120`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L120), `__init__`[`L112`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L112), `__repr__`[`L115`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L115)
- uses (calls/refs, reference-scoped): [`typeof`](core.md#typeof), [`AbstractValue`](core.md#AbstractValue), [`aval_method`](core.md#aval_method), [`aval_property`](core.md#aval_property)
- used by: [`_flatten_bwd`](custom_derivatives.md#_flatten_bwd), [`_vjp_bwd_aval_mismatch_err`](hijax.md#_vjp_bwd_aval_mismatch_err), [`_flatten_jvp`](custom_derivatives.md#_flatten_jvp), [`dce_bwd`](custom_derivatives.md#_custom_vjp_call_dce.dce_bwd), [`jvp`](custom_derivatives.md#lift_jvp.jvp), [`replace_rule_output_symbolic_zeros`](ad_util.md#replace_rule_output_symbolic_zeros), [`f`](ad_util.md#zero_from_primal.f), [`replace_internal_symbolic_zeros`](ad_util.md#replace_internal_symbolic_zeros)

### `Zero`
- def: [`jax/_src/ad_util.py:78`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L78)
- members:
  - `instantiate(self)` — [`L84`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L84)
- protocol/private: `__init__`[`L80`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L80), `__repr__`[`L82`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L82), `__slots__`[`L79`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L79)
- uses (calls/refs, reference-scoped): [`AbstractValue`](core.md#AbstractValue), [`zeros_like_aval`](ad_util.md#zeros_like_aval)
- used by: [`_flatten_bwd`](custom_derivatives.md#_flatten_bwd), [`_pallas_call_jvp_rule`](pallas/pallas_call.md#_pallas_call_jvp_rule), [`_scatter_jvp`](lax/slicing.md#_scatter_jvp), [`vjp_bwd_retval`](hijax.md#CustomVJPTraced.vjp_bwd_retval), [`_sort_jvp`](lax/lax.md#_sort_jvp), [`jvp`](hijax.md#CustomVJPTraced.jvp), [`_shard_map_linearize`](shard_map.md#_shard_map_linearize), [`p2tz`](ad_util.md#p2tz), [`_scatter_extremal_jvp`](lax/slicing.md#_scatter_extremal_jvp), [`_pjit_transpose_fancy`](pjit.md#_pjit_transpose_fancy), [`_ragged_all_to_all_transpose`](lax/parallel.md#_ragged_all_to_all_transpose), [`_run_state_jvp`](state/discharge.md#_run_state_jvp), [`_shard_map_transpose`](shard_map.md#_shard_map_transpose), [`_top_k_jvp`](lax/lax.md#_top_k_jvp), [`_vjp_bwd_aval_mismatch_err`](hijax.md#_vjp_bwd_aval_mismatch_err), [`_cond_transpose_fancy`](lax/control_flow/conditionals.md#_cond_transpose_fancy), [`_custom_linear_solve_jvp`](lax/control_flow/solves.md#_custom_linear_solve_jvp), [`_dynamic_slice_transpose_fancy`](lax/slicing.md#_dynamic_slice_transpose_fancy), [`_gather_transpose_rule`](lax/slicing.md#_gather_transpose_rule), [`_remat_opt_jvp`](custom_derivatives.md#_remat_opt_jvp), [`_scatter_transpose_rule`](lax/slicing.md#_scatter_transpose_rule), [`_while_loop_jvp`](lax/control_flow/loops.md#_while_loop_jvp), [`transposed`](ad_checkpoint.md#_transpose_jaxpr.transposed), [`jvp`](lax/control_flow/loops.md#Scan3.jvp), [`_scatter_addsub_transpose_rule`](lax/slicing.md#_scatter_addsub_transpose_rule), [`_scatter_mul_transpose_rule`](lax/slicing.md#_scatter_mul_transpose_rule), [`_linear_solve_transpose_rule`](lax/control_flow/solves.md#_linear_solve_transpose_rule), [`_maybe_perturbed`](custom_derivatives.md#_maybe_perturbed), [`instantiate`](ad_util.md#instantiate), [`_conv_general_dilated_transpose_rhs`](lax/convolution.md#_conv_general_dilated_transpose_rhs), [`_scan_jvp`](lax/control_flow/loops.md#_scan_jvp), [`_select_and_gather_add_transpose`](lax/windowed_reductions.md#_select_and_gather_add_transpose), [`remat_jvp`](ad_checkpoint.md#remat_jvp), [`remat_transpose`](ad_checkpoint.md#remat_transpose), [`_lift_linearized`](api.md#_lift_linearized), [`_linear_call_transpose_rule`](custom_derivatives.md#_linear_call_transpose_rule), [`_sub_jvp`](lax/lax.md#_sub_jvp), [`_vjp3_bwd`](api.md#_vjp3_bwd), [`reduce_window_jvp`](lax/windowed_reductions.md#reduce_window_jvp), [`_add_jvp`](lax/lax.md#_add_jvp)  (+65 more)

### `stop_gradient_p`
- def: [`jax/_src/ad_util.py:105`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L105)
- uses (calls/refs, reference-scoped): [`Primitive`](core.md#Primitive), [`def_abstract_eval`](core.md#Primitive.def_abstract_eval), [`def_impl`](core.md#Primitive.def_impl), [`_stop_gradient_impl`](ad_util.md#_stop_gradient_impl)
- used by: [`_stop_gradient_lowering_rule`](pallas/mosaic/lowering.md#_stop_gradient_lowering_rule)

## Functions
- `_stop_gradient_impl(x: T)` — [`L99`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L99)
- `a2tz(primal_aval)` — [`L95`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L95)
- `add_abstract(x, y)` — [`L51`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L51)
- `add_impl(x, y)` — [`L46`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L46)
- `add_jaxvals(x: ArrayLike, y: ArrayLike)` — [`L34`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L34)
- `empty_like_aval(aval)` — [`L65`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L65)
- `f(x)` — [`L142`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L142)
- `instantiate(z: Zero | Array)` — [`L72`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L72)
- `p2cz(primal_value)` — [`L92`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L92)
- `p2tz(primal_value)` — [`L89`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L89)
- `replace_internal_symbolic_zeros(x: JaxTypeOrTracer | Zero)` — [`L150`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L150)
- `replace_rule_output_symbolic_zeros(x: JaxTypeOrTracer | SymbolicZero)` — [`L154`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L154)
- `zero_from_primal(val, symbolic_zeros=False)` — [`L141`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L141)
- `zeros_like_aval(aval: core.AbstractValue)` — [`L55`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L55)
- `zeros_like_jaxval(val)` — [`L62`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L62)

## Module values
- `JaxTypeOrTracer` — [`L148`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L148)
- `T` — [`L30`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L30)
- `add_any_p` — [`L43`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L43)
- `add_jaxvals_p` — [`L42`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L42)
- `aval_empty_likers` — [`L70`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L70)
- `aval_zeros_likers` — [`L60`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L60)
- `map` — [`L32`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L32)
- `raw_jaxval_adders` — [`L48`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L48)
- `zeros_like_p` — [`L160`](../../../../../../raw/code/jax/jax/_src/ad_util.py#L160)

