---
title: 'Module: qwix/_src/core/mxfp_dot.py'
type: catalog
provenance: extracted
module: qwix/_src/core/mxfp_dot.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.core.mxfp_dot`/
symbols:
  _flatten_to_3d: _flatten_to_3d().
  mxfp_dot_general: mxfp_dot_general().
  _unflatten_from_3d: _unflatten_from_3d().
  _gpu_mxfp_dot: _gpu_mxfp_dot().
  _is_mxfp: _is_mxfp().
  _get_primary_platform: _get_primary_platform().
---
# Module: [`qwix/_src/core/mxfp_dot.py`](../../../../../../../raw/code/qwix/qwix/_src/core/mxfp_dot.py)

## Functions
- `_flatten_to_3d(operand: qarray.QArray, ca: Sequence[int], ba: Sequence[int])` — [`L94`](../../../../../../../raw/code/qwix/qwix/_src/core/mxfp_dot.py#L94) — Flattens a QArray operand and its scale to 3D for scaled_matmul.
- `_get_primary_platform()` — [`L67`](../../../../../../../raw/code/qwix/qwix/_src/core/mxfp_dot.py#L67) — Returns the JAX platform name cached to avoid tracer round-trips.
- `_gpu_mxfp_dot(lhs, rhs, dimension_numbers, preferred_element_type)` — [`L72`](../../../../../../../raw/code/qwix/qwix/_src/core/mxfp_dot.py#L72) — GPU specific MXFP dot.
- `_is_mxfp(operand: Any)` — [`L59`](../../../../../../../raw/code/qwix/qwix/_src/core/mxfp_dot.py#L59) — Verifies whether the operand is an OCP/NVIDIA microscaled format.
- `_unflatten_from_3d(out_3d: jax.Array, lhs: qarray.QArray, rhs: qarray.QArray, dimension_numbers: jax.lax.DotDimensionNumbers)` — [`L129`](../../../../../../../raw/code/qwix/qwix/_src/core/mxfp_dot.py#L129) — Reshapes the 3D scaled_matmul output back to the expected target shape.
- `mxfp_dot_general(lhs: qarray.MaybeQArray, rhs: qarray.MaybeQArray, dimension_numbers: jax.lax.DotDimensionNumbers, preferred_element_type: jax.typing.DTypeLike | None = None)` — [`L25`](../../../../../../../raw/code/qwix/qwix/_src/core/mxfp_dot.py#L25) — Handles MXFP dot_general on ZFC and Blackwell GPUs. — documented in [qwix-_src-core-dot_general](../../../../concepts/qwix-_src-core-dot_general.md)

