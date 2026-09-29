---
title: 'Module: ejkernel/kernels/_xla/fused_kl_divergence/_xla_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_xla/fused_kl_divergence/_xla_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._xla.fused_kl_divergence._xla_impl_fwd`/
symbols:
  fused_kl_divergence: fused_kl_divergence().
  _kl_per_row_tp: _kl_per_row_tp().
  _fused_kl_core_tp: _fused_kl_core_tp().
  _fused_kl_core: _fused_kl_core().
  _kl_per_row_tp._log_softmax_tp: _kl_per_row_tp()._log_softmax_tp().
  _kl_per_row: _kl_per_row().
---
# Module: [`ejkernel/kernels/_xla/fused_kl_divergence/_xla_impl_fwd.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_kl_divergence/_xla_impl_fwd.py)

## Functions
- `_fused_kl_core(student, teacher, weights)` — [`L45`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_kl_divergence/_xla_impl_fwd.py#L45) — Per-row forward KL on rank-N ``(..., V)`` tensors.
- `_fused_kl_core_tp(student_local, teacher_local, weights, vocab_axis)` — [`L61`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_kl_divergence/_xla_impl_fwd.py#L61) — Vocab-parallel forward KL.
- `_kl_per_row(student_logits, teacher_logits, weights, direction, temperature, beta)` — [`L95`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_kl_divergence/_xla_impl_fwd.py#L95) — Pure-JAX per-row KL — supports forward / reverse / JSD + temperature.
- `_kl_per_row_tp(student_logits, teacher_logits, weights, direction, temperature, beta, vocab_axis)` — [`L141`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_kl_divergence/_xla_impl_fwd.py#L141) — Vocab-parallel per-row KL for *any* direction (forward / reverse / JSD) + temperature.
- `_log_softmax_tp(z)` — [`L156`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_kl_divergence/_xla_impl_fwd.py#L156)
- `fused_kl_divergence(student_logits, teacher_logits, weights=None, *, reduction: str = "mean", direction: str = "forward", temperature: float = 1, beta: float = 0.5, vocab_parallel_axis: str | None = None)` — [`L189`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_kl_divergence/_xla_impl_fwd.py#L189) — JAX/XLA reference for fused KL (forward / reverse / JSD, optional T).

