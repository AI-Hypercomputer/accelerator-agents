---
title: 'Module: ejkernel/kernels/_xla/fused_kl_divergence/_xla_impl_bwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_xla/fused_kl_divergence/_xla_impl_bwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._xla.fused_kl_divergence._xla_impl_bwd`/_
symbols:
  _kl_fwd: kl_fwd().
  _kl_bwd: kl_bwd().
  _kl_tp_fwd: kl_tp_fwd().
  _kl_tp_bwd: kl_tp_bwd().
  __all__: _all__.
---
# Module: [`ejkernel/kernels/_xla/fused_kl_divergence/_xla_impl_bwd.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_kl_divergence/_xla_impl_bwd.py)

## Functions
- `_kl_bwd(residual, dy)` — [`L55`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_kl_divergence/_xla_impl_bwd.py#L55) — Analytic backward: ``dstudent = weight * (p_s - p_t) * dy``; teacher → 0.
- `_kl_fwd(student, teacher, weights)` — [`L39`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_kl_divergence/_xla_impl_bwd.py#L39) — VJP primal for single-shard forward KL.
- `_kl_tp_bwd(vocab_axis, residual, dy)` — [`L102`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_kl_divergence/_xla_impl_bwd.py#L102) — Backward for vocab-parallel KL.
- `_kl_tp_fwd(student_local, teacher_local, weights, vocab_axis)` — [`L64`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_kl_divergence/_xla_impl_bwd.py#L64) — VJP primal for vocab-parallel forward KL.

## Module values
- `__all__` — [`L117`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_kl_divergence/_xla_impl_bwd.py#L117)

