---
title: 'Module: jax/_src/lax/eval_jaxpr.py'
type: catalog
provenance: extracted
module: jax/_src/lax/eval_jaxpr.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.lax.eval_jaxpr`/
symbols:
  eval_jaxpr_p: eval_jaxpr_p.
  _eval_jaxpr_jvp: _eval_jaxpr_jvp().
  _eval_jaxpr_linearize: _eval_jaxpr_linearize().
  _eval_jaxpr_batching_rule: _eval_jaxpr_batching_rule().
  _eval_jaxpr_linearize.tangent_fun: _eval_jaxpr_linearize().tangent_fun().
  zip: zip.
  _stage_jaxpr_abstract_eval: _stage_jaxpr_abstract_eval().
  _eval_jaxpr_impl: _eval_jaxpr_impl().
  _stage_jaxpr: _stage_jaxpr().
  _eval_jaxpr_transpose: _eval_jaxpr_transpose().
  _map: _map.
---
# Module: [`jax/_src/lax/eval_jaxpr.py`](../../../../../../../raw/code/jax/jax/_src/lax/eval_jaxpr.py)

## Functions
- `_eval_jaxpr_batching_rule(axis_data, args, dims, *, jaxpr)` — [`L58`](../../../../../../../raw/code/jax/jax/_src/lax/eval_jaxpr.py#L58)
- `_eval_jaxpr_impl(*args, jaxpr)` — [`L43`](../../../../../../../raw/code/jax/jax/_src/lax/eval_jaxpr.py#L43)
- `_eval_jaxpr_jvp(primals, tangents, *, jaxpr)` — [`L46`](../../../../../../../raw/code/jax/jax/_src/lax/eval_jaxpr.py#L46)
- `_eval_jaxpr_linearize(is_vjp, nzs, *primals_in, jaxpr)` — [`L69`](../../../../../../../raw/code/jax/jax/_src/lax/eval_jaxpr.py#L69)
- `_eval_jaxpr_transpose(ct, *args, jaxpr)` — [`L90`](../../../../../../../raw/code/jax/jax/_src/lax/eval_jaxpr.py#L90)
- `_stage_jaxpr(trace: pe.DynamicJaxprTrace, source_info, *tracers, jaxpr: ClosedJaxpr)` — [`L31`](../../../../../../../raw/code/jax/jax/_src/lax/eval_jaxpr.py#L31)
- `_stage_jaxpr_abstract_eval(*_, jaxpr)` — [`L39`](../../../../../../../raw/code/jax/jax/_src/lax/eval_jaxpr.py#L39)
- `tangent_fun(res, *tangents)` — [`L77`](../../../../../../../raw/code/jax/jax/_src/lax/eval_jaxpr.py#L77)

## Module values
- `_map` — [`L24`](../../../../../../../raw/code/jax/jax/_src/lax/eval_jaxpr.py#L24)
- `eval_jaxpr_p` — [`L28`](../../../../../../../raw/code/jax/jax/_src/lax/eval_jaxpr.py#L28)
- `zip` — [`L25`](../../../../../../../raw/code/jax/jax/_src/lax/eval_jaxpr.py#L25)

