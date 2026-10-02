---
title: 'Module: jax/_src/pallas/cost_estimate.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/cost_estimate.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.cost_estimate`/
symbols:
  cost_estimate_jaxpr: cost_estimate_jaxpr().
  TRANSCENDENTAL_OPS: TRANSCENDENTAL_OPS.
  BINARY_OPS: BINARY_OPS.
  UN_OPS: UN_OPS.
  _pjit_cost_rule: _pjit_cost_rule().
  _custom_vjp_rule: _custom_vjp_rule().
  _run_state_rule: _run_state_rule().
  estimate_cost: estimate_cost().
  dot_general_cost_rule: dot_general_cost_rule().
  binary_cost_rule: binary_cost_rule().
  unary_cost_rule.cost_rule: unary_cost_rule().cost_rule().
  _integer_pow_cost_rule: _integer_pow_cost_rule().
  CostEstimate: CostEstimate#
  op: op.
  register_cost_rule: register_cost_rule().
  CostEstimate.flops: CostEstimate#flops.
  CostEstimate.transcendentals: CostEstimate#transcendentals.
  CostEstimate.bytes_accessed: CostEstimate#bytes_accessed.
  CostEstimate.__add__: CostEstimate#__add__().
  Context: Context#
  unary_cost_rule: unary_cost_rule().
  Context.avals_in: Context#avals_in.
  unsafe_map: unsafe_map.
  unsafe_zip: unsafe_zip.
  _cost_rules: _cost_rules.
  Context.avals_out: Context#avals_out.
---
# Module: [`jax/_src/pallas/cost_estimate.py`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py)

## Classes
### `Context`
- def: [`jax/_src/pallas/cost_estimate.py:55`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L55)
- signature: `class Context:`
- members:
  - `avals_in` — [`L56`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L56)
  - `avals_out` — [`L57`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L57)
- used by: [`cost_estimate_jaxpr`](cost_estimate.md#cost_estimate_jaxpr), [`dot_general_cost_rule`](cost_estimate.md#dot_general_cost_rule), [`_integer_pow_cost_rule`](cost_estimate.md#_integer_pow_cost_rule), [`binary_cost_rule`](cost_estimate.md#binary_cost_rule), [`cost_rule`](cost_estimate.md#unary_cost_rule.cost_rule)

### `CostEstimate`
- def: [`jax/_src/pallas/cost_estimate.py:39`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L39)
- signature: `class CostEstimate:`
- members:
  - `bytes_accessed` — [`L42`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L42)
  - `flops` — [`L40`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L40)
  - `transcendentals` — [`L41`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L41)
- protocol/private: `__add__`[`L44`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L44)
- used by: [`cost_estimate_jaxpr`](cost_estimate.md#cost_estimate_jaxpr), [`_custom_vjp_rule`](cost_estimate.md#_custom_vjp_rule), [`_pjit_cost_rule`](cost_estimate.md#_pjit_cost_rule), [`_run_state_rule`](cost_estimate.md#_run_state_rule), [`dot_general_cost_rule`](cost_estimate.md#dot_general_cost_rule), [`_integer_pow_cost_rule`](cost_estimate.md#_integer_pow_cost_rule), [`binary_cost_rule`](cost_estimate.md#binary_cost_rule), [`cost_rule`](cost_estimate.md#unary_cost_rule.cost_rule)

## Functions
- `_custom_vjp_rule(ctx, *, call_jaxpr: jax_core.ClosedJaxpr, **_)` — [`L241`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L241)
- `_integer_pow_cost_rule(ctx: Context, *, y: int)` — [`L176`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L176)
- `_pjit_cost_rule(ctx, *, jaxpr: jax_core.ClosedJaxpr, **_)` — [`L231`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L231)
- `_run_state_rule(*_, jaxpr: jax_core.Jaxpr, **_2)` — [`L251`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L251)
- `binary_cost_rule(ctx: Context, **_)` — [`L109`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L109)
- `cost_estimate_jaxpr(jaxpr: jax_core.ClosedJaxpr)` — [`L59`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L59) — Returns the cost estimate for the given Jaxpr.
- `cost_rule(ctx: Context, **_)` — [`L133`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L133)
- `dot_general_cost_rule(ctx: Context, dimension_numbers: lax.DotDimensionNumbers, **_)` — [`L194`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L194)
- `estimate_cost(fun, *args, **kwargs)` — [`L79`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L79) — Computes a cost estimate for the given function.
- `register_cost_rule(primitive: jax_core.Primitive, rule)` — [`L51`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L51)
- `unary_cost_rule(transcendental: bool)` — [`L132`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L132)

## Module values
- `BINARY_OPS` — [`L117`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L117)
- `TRANSCENDENTAL_OPS` — [`L158`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L158)
- `UN_OPS` — [`L148`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L148)
- `_cost_rules` — [`L36`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L36)
- `op` — [`L128`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L128)
- `unsafe_map` — [`L33`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L33)
- `unsafe_zip` — [`L34`](../../../../../../../raw/code/jax/jax/_src/pallas/cost_estimate.py#L34)

