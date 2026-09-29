---
title: 'Module: axlearn/common/quantized_dot_general/fp8_ops.py'
type: catalog
provenance: extracted
module: axlearn/common/quantized_dot_general/fp8_ops.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.quantized_dot_general.fp8_ops`/
symbols:
  q_dot_q: q_dot_q().
  _quantize: _quantize().
  _dequantize: _dequantize().
  _q_dot_dq_impl: _q_dot_dq_impl().
  _q_dot_dq_fwd: _q_dot_dq_fwd().
  _q_dot_dq_bwd: _q_dot_dq_bwd().
---
# Module: [`axlearn/common/quantized_dot_general/fp8_ops.py`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/fp8_ops.py)

## Functions
- `_dequantize(x: Tensor, scale: Tensor, *, dq_dtype: DTypeLike)` — [`L56`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/fp8_ops.py#L56)
- `_q_dot_dq_bwd(dimension_numbers: tuple, precision: jax.lax.PrecisionLike, preferred_element_type: DTypeLike, res: tuple[Tensor, ...], g: Tensor)` — [`L193`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/fp8_ops.py#L193)
- `_q_dot_dq_fwd(lhs: Tensor, rhs: Tensor, lhs_scale: Tensor, rhs_scale: Tensor, out_grad_scale: Tensor, lhs_amax_history: Optional[Tensor], rhs_amax_history: Optional[Tensor], out_grad_amax_history: Optional[Tensor], dimension_numbers: tuple, precision: jax.lax.PrecisionLike, preferred_element_type: DTypeLike)` — [`L170`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/fp8_ops.py#L170) — See `q_dot_dq_in_batch`.
- `_q_dot_dq_impl(lhs: Tensor, rhs: Tensor, lhs_scale: Tensor, rhs_scale: Tensor, out_grad_scale: Tensor, lhs_amax_history: Optional[Tensor], rhs_amax_history: Optional[Tensor], out_grad_amax_history: Optional[Tensor], dimension_numbers: tuple, precision: jax.lax.PrecisionLike, preferred_element_type: DTypeLike, is_training: bool)` — [`L64`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/fp8_ops.py#L64) — See `q_dot_dq_in_batch`.
- `_quantize(x: Tensor, scale: Tensor, amax_history: Optional[Tensor], *, dtype: DTypeLike, preferred_element_type: DTypeLike)` — [`L16`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/fp8_ops.py#L16)
- `q_dot_q(lhs: Tensor, rhs: Tensor, lhs_scale: Tensor, rhs_scale: Tensor, out_grad_scale: Tensor, lhs_amax_history: Optional[Tensor], rhs_amax_history: Optional[Tensor], out_grad_amax_history: Optional[Tensor], dimension_numbers: jax.lax.DotDimensionNumbers, precision: jax.lax.PrecisionLike, preferred_element_type: DTypeLike=None)` — [`L128`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/fp8_ops.py#L128) — Computes lhs @ rhs in FP8 using either in-batch scaling or delayed scaling.

