---
title: 'Module: qwix/_src/core/ragged_dot_qt.py'
type: catalog
provenance: extracted
module: qwix/_src/core/ragged_dot_qt.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.core.ragged_dot_qt`/
symbols:
  ragged_dot_qt_bwd: ragged_dot_qt_bwd().
  ragged_dot_qt_fwd: ragged_dot_qt_fwd().
  ragged_dot_qt: ragged_dot_qt().
  RaggedDotQtConfig: RaggedDotQtConfig#
  RaggedDotQtConfig.drhs_grad_qtype: RaggedDotQtConfig#drhs_grad_qtype.
  RaggedDotQtConfig.lhs_qtype: RaggedDotQtConfig#lhs_qtype.
  RaggedDotQtConfig.rhs_qtype: RaggedDotQtConfig#rhs_qtype.
  RaggedDotQtConfig.dlhs_grad_qtype: RaggedDotQtConfig#dlhs_grad_qtype.
---
# Module: [`qwix/_src/core/ragged_dot_qt.py`](../../../../../../../raw/code/qwix/qwix/_src/core/ragged_dot_qt.py)

## Classes
### `RaggedDotQtConfig`
- def: [`qwix/_src/core/ragged_dot_qt.py:26`](../../../../../../../raw/code/qwix/qwix/_src/core/ragged_dot_qt.py#L26) — documented in [qwix-_src-providers-qt](../../../../concepts/qwix-_src-providers-qt.md)
- doc: Configuration for ragged_dot_qt.
- signature: `class RaggedDotQtConfig:`
- members:
  - `dlhs_grad_qtype` — [`L34`](../../../../../../../raw/code/qwix/qwix/_src/core/ragged_dot_qt.py#L34)
  - `drhs_grad_qtype` — [`L35`](../../../../../../../raw/code/qwix/qwix/_src/core/ragged_dot_qt.py#L35)
  - `lhs_qtype` — [`L30`](../../../../../../../raw/code/qwix/qwix/_src/core/ragged_dot_qt.py#L30)
  - `rhs_qtype` — [`L31`](../../../../../../../raw/code/qwix/qwix/_src/core/ragged_dot_qt.py#L31)
- uses (calls/refs, reference-scoped): [`disable_interceptions`](../interception.md#disable_interceptions)
- used by: [`ragged_dot_qt_bwd`](ragged_dot_qt.md#ragged_dot_qt_bwd), [`_create_ragged_dot_qt_config`](../providers/qt.md#QtProvider._create_ragged_dot_qt_config), [`ragged_dot_qt_fwd`](ragged_dot_qt.md#ragged_dot_qt_fwd), [`ragged_dot_qt`](ragged_dot_qt.md#ragged_dot_qt)  (3 test-only)

## Functions
- `ragged_dot_qt(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, config: RaggedDotQtConfig, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, group_offset: jax.Array | None = None)` — [`L132`](../../../../../../../raw/code/qwix/qwix/_src/core/ragged_dot_qt.py#L132) — Quantized ragged_dot with backpropagation support.
- `ragged_dot_qt_bwd(config: RaggedDotQtConfig, precision: jax.lax.PrecisionLike, preferred_element_type: jax.typing.DTypeLike | None, group_offset: jax.Array | None, residuals: tuple[qarray.MaybeQArray, qarray.MaybeQArray, jax.Array], g: jax.Array)` — [`L62`](../../../../../../../raw/code/qwix/qwix/_src/core/ragged_dot_qt.py#L62) — Backward pass for ragged_dot_qt custom VJP.
- `ragged_dot_qt_fwd(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, config: RaggedDotQtConfig, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, group_offset: jax.Array | None = None)` — [`L39`](../../../../../../../raw/code/qwix/qwix/_src/core/ragged_dot_qt.py#L39) — Forward pass for ragged_dot_qt custom VJP.

