---
title: 'Module: axlearn/common/ein_ops.py'
type: catalog
provenance: extracted
module: axlearn/common/ein_ops.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.ein_ops`/
symbols:
  rearrange: rearrange().
  _parse_axes: _parse_axes().
  repeat: repeat().
  _Axes: _Axes.
  _compute_rearrange_plan: _compute_rearrange_plan().
  _get_input_reshape: _get_input_reshape().
  _compute_repeat_plan: _compute_repeat_plan().
  _TOKEN_RE: _TOKEN_RE.
  _resolve_ellipsis: _resolve_ellipsis().
  _get_rearrange_shape: _get_rearrange_shape().
  _get_repeat_shape: _get_repeat_shape().
  _parse_pattern: _parse_pattern().
  _get_repeat_shape.update_and_shape: _get_repeat_shape().update_and_shape().
  _parse_axes._validate_token: _parse_axes()._validate_token().
  _parse_axes._validate_group_token: _parse_axes()._validate_group_token().
  _resolve_ellipsis.count_explicit_axes: _resolve_ellipsis().count_explicit_axes().
  _Plan.operand: _Plan#operand.
  _IDENTIFIER: _IDENTIFIER.
  _Plan: _Plan#
  _resolve_ellipsis.find_ellipsis_index: _resolve_ellipsis().find_ellipsis_index().
  _resolve_ellipsis.replace: _resolve_ellipsis().replace().
  _Plan.input_reshape: _Plan#input_reshape.
  _Plan.output_shape: _Plan#output_shape.
  _ELLIPSIS: _ELLIPSIS.
  _parse_axes._validate_duplicated: _parse_axes()._validate_duplicated().
  _get_input_reshape.update_dim_size_map: _get_input_reshape().update_dim_size_map().
  _to_input_reshape: _to_input_reshape().
  _to_input_axes_map: _to_input_axes_map().
  _RearrangeRHSPlan: _RearrangeRHSPlan#
  _get_rearrange_shape.update_and_shape: _get_rearrange_shape().update_and_shape().
  _Axis: _Axis.
  _COMPOSITE_DIM: _COMPOSITE_DIM.
  _ONE: _ONE.
  _resolve_ellipsis.count_explicit_axes.count: _resolve_ellipsis().count_explicit_axes().count().
  _get_input_reshape.infer_dims: _get_input_reshape().infer_dims().
  _get_repeat_shape.update_existing_axis: _get_repeat_shape().update_existing_axis().
  _get_repeat_shape.update_new_axis: _get_repeat_shape().update_new_axis().
  _RearrangeRHSPlan.perm: _RearrangeRHSPlan#perm.
  _RearrangeRHSPlan.output_shape: _RearrangeRHSPlan#output_shape.
---
# Module: [`axlearn/common/ein_ops.py`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py)

## Classes
### `_Plan`
- def: [`axlearn/common/ein_ops.py:191`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L191)
- members:
  - `input_reshape` — [`L200`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L200)
  - `operand` — [`L201`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L201)
  - `output_shape` — [`L202`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L202)
- used by: [`rearrange`](ein_ops.md#rearrange), [`repeat`](ein_ops.md#repeat), [`_compute_rearrange_plan`](ein_ops.md#_compute_rearrange_plan), [`_compute_repeat_plan`](ein_ops.md#_compute_repeat_plan), [`_get_repeat_shape`](ein_ops.md#_get_repeat_shape)

### `_RearrangeRHSPlan`  ·  implements/extends NamedTuple
- def: [`axlearn/common/ein_ops.py:578`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L578)
- doc: Plan for rearrange from RHS axes.
- signature: `class _RearrangeRHSPlan(NamedTuple):`
- members:
  - `output_shape` — [`L587`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L587)
  - `perm` — [`L586`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L586)
- used by: [`_get_rearrange_shape`](ein_ops.md#_get_rearrange_shape)

## Functions
- `_compute_rearrange_plan(in_shape: tuple[int, ...], pattern: str, **axes_lengths)` — [`L206`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L206) — Computes the reshape and transpose plan for `rearrange()`.
- `_compute_repeat_plan(in_shape: tuple[int, ...], pattern: str, **axes_lengths)` — [`L239`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L239) — Computes the reshape, tile, and output shape plan for `repeat()`.
- `_get_input_reshape(*, shape: tuple[int, ...], lhs_axes: _Axes, axes_lengths: dict[str, int])` — [`L471`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L471) — Computes the input shape to reshape into, based on left-side axes.
- `_get_rearrange_shape(*, rhs_axes: _Axes, dim_size_map: dict[str, int])` — [`L590`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L590) — Computes the final output shape and permutation order.
- `_get_repeat_shape(*, rhs_axes: _Axes, dim_size_map: dict[str, int], axes_lengths: dict[str, int])` — [`L641`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L641) — Computes shapes for applying `jnp.tile` and the final output shape in `repeat`.
- `_parse_axes(axes: str)` — [`L322`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L322) — Parses an axis pattern string into a hierarchical tuple structure.
- `_parse_pattern(*, pattern: str, in_shape: tuple[int, ...])` — [`L280`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L280) — Parses an einops-style pattern into left and right axes, expanding ellipsis if present.
- `_resolve_ellipsis(in_shape: tuple[int, ...], lhs: _Axes, rhs: _Axes)` — [`L383`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L383) — Resolves ellipsis ("...") in parsed lhs and rhs axes into named dummy axes.
- `_to_input_axes_map(dim_size_map: dict[str, int])` — [`L562`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L562) — Infer input_axes_map from dim_size_map.
- `_to_input_reshape(dim_size_map: dict[str, int])` — [`L546`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L546) — Infer input_reshape from dim_size_map.
- `_validate_duplicated(name: str)` — [`L350`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L350)
- `_validate_group_token(name: str)` — [`L363`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L363)
- `_validate_token(name: str)` — [`L355`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L355)
- `count(ax)` — [`L444`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L444)
- `count_explicit_axes(axes: _Axes)` — [`L443`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L443)
- `find_ellipsis_index(axes: _Axes)` — [`L403`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L403) — Finds the position and context of an ellipsis ("...") in a parsed einops axis pattern.
- `infer_dims(group: list[str], group_dim_size: int)` — [`L498`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L498)
- `rearrange(x: Tensor, pattern: str, **axes_lengths)` — [`L48`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L48) — JAX implementation of `einops.rearrange`.
- `repeat(x: Tensor, pattern: str, **axes_lengths)` — [`L120`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L120) — JAX implementation of `einops.repeat`.
- `replace(axes: _Axes, idx: int, is_group: bool)` — [`L457`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L457)
- `update_and_shape(axis_name)` — [`L612`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L612)
- `update_and_shape(axis_name)` — [`L689`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L689)
- `update_dim_size_map(axis_name, dim_size)` — [`L516`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L516)
- `update_existing_axis(axis_name: str)` — [`L673`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L673)
- `update_new_axis(axis_name: str)` — [`L683`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L683)

## Module values
- `_Axes` — [`L277`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L277)
- `_Axis` — [`L276`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L276)
- `_COMPOSITE_DIM` — [`L316`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L316)
- `_ELLIPSIS` — [`L318`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L318)
- `_IDENTIFIER` — [`L311`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L311)
- `_ONE` — [`L317`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L317)
- `_TOKEN_RE` — [`L319`](../../../../../../raw/code/axlearn/axlearn/common/ein_ops.py#L319)

