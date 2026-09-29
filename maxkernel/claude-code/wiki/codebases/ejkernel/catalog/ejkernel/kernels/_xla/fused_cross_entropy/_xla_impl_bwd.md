---
title: 'Module: ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_bwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_bwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._xla.fused_cross_entropy._xla_impl_bwd`/_
symbols:
  _ce_fwd: ce_fwd().
  _ce_bwd: ce_bwd().
  _ce_tp_fwd: ce_tp_fwd().
  _ce_tp_bwd: ce_tp_bwd().
  _soft_ce_tp_fwd: soft_ce_tp_fwd().
  _soft_ce_tp_bwd: soft_ce_tp_bwd().
  __all__: _all__.
---
# Module: [`ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_bwd.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_bwd.py)

## Functions
- `_ce_bwd(ignore_index, residual, dy)` — [`L57`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_bwd.py#L57) — Analytic backward: ``dlogits = weight * (softmax - onehot) * dy``.
- `_ce_fwd(logits, targets, weights, ignore_index)` — [`L38`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_bwd.py#L38) — VJP primal for the single-shard fused cross-entropy.
- `_ce_tp_bwd(ignore_index, vocab_axis, residual, dy)` — [`L105`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_bwd.py#L105) — Backward for vocab-parallel CE.
- `_ce_tp_fwd(logits_local, targets, weights, ignore_index, vocab_axis)` — [`L71`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_bwd.py#L71) — VJP primal for vocab-parallel cross-entropy.
- `_soft_ce_tp_bwd(vocab_axis, residual, dy)` — [`L146`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_bwd.py#L146) — Backward for vocab-parallel dense CE: ``dlogits = (softmax·S - soft) · dy`` (local, no collectives).
- `_soft_ce_tp_fwd(logits_local, soft_local, vocab_axis)` — [`L124`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_bwd.py#L124) — VJP primal for vocab-parallel *dense* (soft-target) cross-entropy.

## Module values
- `__all__` — [`L160`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_bwd.py#L160)

