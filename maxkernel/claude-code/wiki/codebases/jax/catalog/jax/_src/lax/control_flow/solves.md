---
title: 'Module: jax/_src/lax/control_flow/solves.py'
type: catalog
provenance: extracted
module: jax/_src/lax/control_flow/solves.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.lax.control_flow.solves`/
symbols:
  linear_solve_p: linear_solve_p.
  custom_linear_solve: custom_linear_solve().
  _custom_linear_solve_jvp: _custom_linear_solve_jvp().
  custom_root: custom_root().
  _linear_solve_transpose_rule: _linear_solve_transpose_rule().
  _linear_solve_batching_rule: _linear_solve_batching_rule().
  _root_jvp: _root_jvp().
  _map: _map.
  _split_linear_solve_args: _split_linear_solve_args().
  _LinearSolveTuple.transpose: _LinearSolveTuple#transpose().
  _tangent_linear_map: _tangent_linear_map().
  _custom_root: _custom_root().
  _custom_linear_solve_impl: _custom_linear_solve_impl().
  _linear_solve_abstract_eval: _linear_solve_abstract_eval().
  _split_root_args: _split_root_args().
  custom_linear_solve._shape_checked: custom_linear_solve()._shape_checked().
  _transpose_one_output: _transpose_one_output().
  _flatten: _flatten().
  custom_linear_solve._shape_checked.f: custom_linear_solve()._shape_checked().f().
  custom_linear_solve._shape_checked.f_aux: custom_linear_solve()._shape_checked().f_aux().
  _check_shapes: _check_shapes().
  _LinearSolveTuple: _LinearSolveTuple#
  _LinearSolveTuple.matvec: _LinearSolveTuple#matvec.
  custom_root.linearize_and_solve: custom_root().linearize_and_solve().
  _RootTuple: _RootTuple.
  _LinearSolveTuple.solve: _LinearSolveTuple#solve.
  _LinearSolveTuple.vecmat: _LinearSolveTuple#vecmat.
  _LinearSolveTuple.transpose_solve: _LinearSolveTuple#transpose_solve.
  _transpose_one_output.transposed_fun: _transpose_one_output().transposed_fun().
---
# Module: [`jax/_src/lax/control_flow/solves.py`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py)

## Classes
### `_LinearSolveTuple`  ·  implements/extends NamedTuple
- def: [`jax/_src/lax/control_flow/solves.py:174`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L174)
- signature: `class _LinearSolveTuple(NamedTuple):`
- members:
  - `transpose(self)` — [`L180`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L180)
  - `matvec` — [`L175`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L175)
  - `solve` — [`L177`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L177)
  - `transpose_solve` — [`L178`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L178)
  - `vecmat` — [`L176`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L176)
- used by: [`custom_linear_solve`](solves.md#custom_linear_solve), [`_custom_linear_solve_jvp`](solves.md#_custom_linear_solve_jvp), [`_linear_solve_transpose_rule`](solves.md#_linear_solve_transpose_rule), [`_linear_solve_batching_rule`](solves.md#_linear_solve_batching_rule), [`_split_linear_solve_args`](solves.md#_split_linear_solve_args), [`_custom_linear_solve_impl`](solves.md#_custom_linear_solve_impl)

## Functions
- `_check_shapes(func_name, expected_name, actual, expected)` — [`L201`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L201)
- `_custom_linear_solve_impl(*args, const_lengths, jaxprs)` — [`L342`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L342)
- `_custom_linear_solve_jvp(primals, tangents, const_lengths, jaxprs)` — [`L364`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L364)
- `_custom_root(const_lengths, jaxprs, *args)` — [`L134`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L134)
- `_flatten(args)` — [`L197`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L197)
- `_linear_solve_abstract_eval(*args, const_lengths, jaxprs)` — [`L326`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L326)
- `_linear_solve_batching_rule(axis_data, args, dims, const_lengths, jaxprs)` — [`L423`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L423)
- `_linear_solve_transpose_rule(cotangent, *primals, const_lengths, jaxprs)` — [`L402`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L402)
- `_root_jvp(const_lengths, jaxprs, primals, tangents)` — [`L141`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L141)
- `_shape_checked(fun, name, has_aux)` — [`L258`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L258)
- `_split_linear_solve_args(args, const_lengths)` — [`L184`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L184)
- `_split_root_args(args, const_lengths)` — [`L45`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L45)
- `_tangent_linear_map(func: Callable, params, params_dot, debug_info: core.DebugInfo, *x)` — [`L348`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L348) — Compute the tangent of a linear map.
- `_transpose_one_output(linear_fun, primals)` — [`L189`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L189)
- `custom_linear_solve(matvec: Callable, b: Any, solve: Callable[[Callable, Any], Any], transpose_solve: Callable[[Callable, Any], Any] | None = None, symmetric=False, has_aux=False)` — [`L211`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L211) — Perform a matrix-free linear solve with implicitly defined gradients.
- `custom_root(f: Callable, initial_guess: Any, solve: Callable[[Callable, Any], Any], tangent_solve: Callable[[Callable, Any], Any], has_aux=False)` — [`L51`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L51) — Differentiably solve for the roots of a function.
- `f(x)` — [`L259`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L259)
- `f_aux(x)` — [`L264`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L264)
- `linearize_and_solve(x, b)` — [`L110`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L110)
- `transposed_fun(x)` — [`L191`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L191)

## Module values
- `_RootTuple` — [`L42`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L42)
- `_map` — [`L40`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L40)
- `linear_solve_p` — [`L501`](../../../../../../../../raw/code/jax/jax/_src/lax/control_flow/solves.py#L501)

