---
title: 'Module: aqt/jax/v2/transpose.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/transpose.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.transpose`/
symbols:
  rhs_scale_transpose_for_lhs_input: rhs_scale_transpose_for_lhs_input().
  lhs_scale_transpose_for_rhs_input: lhs_scale_transpose_for_rhs_input().
  lhs_scale_transpose_to_output: lhs_scale_transpose_to_output().
  rhs_scale_transpose_to_output: rhs_scale_transpose_to_output().
  transpose: transpose().
  _scale_trans: _scale_trans().
  _scale_trans_back: _scale_trans_back().
  lhs_recover_scale_from_scale_t: lhs_recover_scale_from_scale_t().
  rhs_recover_scale_from_scale_t: rhs_recover_scale_from_scale_t().
  _scale_trans_for_other_input: _scale_trans_for_other_input().
---
# Module: [`aqt/jax/v2/transpose.py`](../../../../../../../raw/code/aqt/aqt/jax/v2/transpose.py)

## Functions
- `_scale_trans(x, ca, ba)` — [`L98`](../../../../../../../raw/code/aqt/aqt/jax/v2/transpose.py#L98) — Transposes x to output dimension order.
- `_scale_trans_back(scale_t: jax.Array, ca: Sequence[utils.AxisIdx], ba: Sequence[utils.AxisIdx])` — [`L151`](../../../../../../../raw/code/aqt/aqt/jax/v2/transpose.py#L151) — Transposes scale (transposed for output) back to its original dimension.
- `_scale_trans_for_other_input(x: jax.Array, my_ca: Sequence[utils.AxisIdx], my_ba: Sequence[utils.AxisIdx], other_ca: Sequence[utils.AxisIdx], other_ba: Sequence[utils.AxisIdx], other_rank: int)` — [`L218`](../../../../../../../raw/code/aqt/aqt/jax/v2/transpose.py#L218) — Transposes x to other inputs' dimension order.
- `lhs_recover_scale_from_scale_t(lhs_scale_t: jax.Array, dimension_numbers: jax.lax.DotDimensionNumbers, lhs_shape: Sequence[int], rhs_shape: Sequence[int])` — [`L183`](../../../../../../../raw/code/aqt/aqt/jax/v2/transpose.py#L183) — Recovers lhs_scale from lhs_scale_t.
- `lhs_scale_transpose_for_rhs_input(lhs_scale, dimension_numbers, rhs_shape)` — [`L266`](../../../../../../../raw/code/aqt/aqt/jax/v2/transpose.py#L266) — Transposes lhs_scale to rhs input dimension order. — documented in [aqt-jax-aqt_dot_general](../../../../concepts/aqt-jax-aqt_dot_general.md)
- `lhs_scale_transpose_to_output(lhs_scale, dimension_numbers, lhs_shape, rhs_shape)` — [`L114`](../../../../../../../raw/code/aqt/aqt/jax/v2/transpose.py#L114) — Transposes lhs_scale to output dimension order. — documented in [aqt-jax-aqt_dot_general](../../../../concepts/aqt-jax-aqt_dot_general.md)
- `rhs_recover_scale_from_scale_t(rhs_scale_t: jax.Array, dimension_numbers: jax.lax.DotDimensionNumbers, lhs_shape: Sequence[int], rhs_shape: Sequence[int])` — [`L200`](../../../../../../../raw/code/aqt/aqt/jax/v2/transpose.py#L200) — Recovers rhs_scale from rhs_scale_t.
- `rhs_scale_transpose_for_lhs_input(rhs_scale, dimension_numbers, lhs_shape)` — [`L277`](../../../../../../../raw/code/aqt/aqt/jax/v2/transpose.py#L277) — Transposes lhs_scale to rhs input dimension order. — documented in [aqt-jax-aqt_dot_general](../../../../concepts/aqt-jax-aqt_dot_general.md)
- `rhs_scale_transpose_to_output(rhs_scale, dimension_numbers, lhs_shape, rhs_shape)` — [`L135`](../../../../../../../raw/code/aqt/aqt/jax/v2/transpose.py#L135) — Transposes rhs_scale to output dimension order. — documented in [aqt-jax-aqt_dot_general](../../../../concepts/aqt-jax-aqt_dot_general.md)
- `transpose(t: jax.Array, axes: list[utils.AxisIdx])` — [`L29`](../../../../../../../raw/code/aqt/aqt/jax/v2/transpose.py#L29) — Transpose tensor through reshape if possible.

