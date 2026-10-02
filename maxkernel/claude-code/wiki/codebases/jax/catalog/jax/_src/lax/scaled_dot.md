---
title: 'Module: jax/_src/lax/scaled_dot.py'
type: catalog
provenance: extracted
module: jax/_src/lax/scaled_dot.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.lax.scaled_dot`/
symbols:
  scaled_dot: scaled_dot().
  _scaled_dot_impl: _scaled_dot_impl().
  scaled_dot_p: scaled_dot_p.
  _scaled_dot_validate_inputs: _scaled_dot_validate_inputs().
  _scale_broadcast: _scale_broadcast().
  _scaled_dot_abstract_eval: _scaled_dot_abstract_eval().
  _create_dummy_scale: _create_dummy_scale().
  scaled_dot_lowering: scaled_dot_lowering.
  _scaled_dot_batching_rule: _scaled_dot_batching_rule().
  _validate_operand_scale: _validate_operand_scale().
---
# Module: [`jax/_src/lax/scaled_dot.py`](../../../../../../../raw/code/jax/jax/_src/lax/scaled_dot.py)

## Functions
- `_create_dummy_scale(operand, contracting_dims)` — [`L228`](../../../../../../../raw/code/jax/jax/_src/lax/scaled_dot.py#L228)
- `_scale_broadcast(scale: Array, operand_shape: tuple[int, ...], contracting_dims: Sequence[int])` — [`L166`](../../../../../../../raw/code/jax/jax/_src/lax/scaled_dot.py#L166)
- `_scaled_dot_abstract_eval(lhs, rhs, lhs_scale, rhs_scale, *, dimension_numbers: lax.DotDimensionNumbers, preferred_element_type: DTypeLike | None = None)` — [`L120`](../../../../../../../raw/code/jax/jax/_src/lax/scaled_dot.py#L120)
- `_scaled_dot_batching_rule(batched_args, batch_dims, *, dimension_numbers, preferred_element_type)` — [`L235`](../../../../../../../raw/code/jax/jax/_src/lax/scaled_dot.py#L235)
- `_scaled_dot_impl(lhs: Array, rhs: Array, lhs_scale: Array, rhs_scale: Array, *, dimension_numbers: lax.DotDimensionNumbers, preferred_element_type: DTypeLike | None = None)` — [`L187`](../../../../../../../raw/code/jax/jax/_src/lax/scaled_dot.py#L187) — Implementation of scaled_dot that could be replaced by XLA.
- `_scaled_dot_validate_inputs(lhs: Array, rhs: Array, lhs_scale: Array | None, rhs_scale: Array | None, *, dimension_numbers: lax.DotDimensionNumbers, preferred_element_type: DTypeLike | None)` — [`L51`](../../../../../../../raw/code/jax/jax/_src/lax/scaled_dot.py#L51) — Validates the inputs to scaled_dot.
- `_validate_operand_scale(side, operand, scale, contracting_dims: Sequence[int])` — [`L28`](../../../../../../../raw/code/jax/jax/_src/lax/scaled_dot.py#L28)
- `scaled_dot(lhs: Array, rhs: Array, *, lhs_scale: Array | None = None, rhs_scale: Array | None = None, dimension_numbers: lax.DotDimensionNumbers | None = None, preferred_element_type: DTypeLike | None = None)` — [`L299`](../../../../../../../raw/code/jax/jax/_src/lax/scaled_dot.py#L299) — Computes a scaled dot product.

## Module values
- `scaled_dot_lowering` — [`L223`](../../../../../../../raw/code/jax/jax/_src/lax/scaled_dot.py#L223)
- `scaled_dot_p` — [`L220`](../../../../../../../raw/code/jax/jax/_src/lax/scaled_dot.py#L220)

