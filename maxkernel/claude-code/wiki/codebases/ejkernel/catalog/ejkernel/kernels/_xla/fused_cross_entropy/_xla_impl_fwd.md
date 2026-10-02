---
title: 'Module: ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._xla.fused_cross_entropy._xla_impl_fwd`/
symbols:
  fused_cross_entropy: fused_cross_entropy().
  _fused_ce_core: _fused_ce_core().
  _fused_ce_core_tp: _fused_ce_core_tp().
  _label_smoothing_correction: _label_smoothing_correction().
  _fused_soft_ce_core_tp: _fused_soft_ce_core_tp().
  _per_token_weights: _per_token_weights().
---
# Module: [`ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_fwd.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_fwd.py)

## Functions
- `_fused_ce_core(logits, targets, weights, ignore_index)` — [`L52`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_fwd.py#L52) — Per-row sparse cross-entropy on a rank-N ``(..., V)`` tensor.
- `_fused_ce_core_tp(logits_local, targets, weights, ignore_index, vocab_axis)` — [`L74`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_fwd.py#L74) — Vocab-parallel CE. ``logits_local`` is the per-shard ``V_local`` slice.
- `_fused_soft_ce_core_tp(logits_local, soft_local, vocab_axis)` — [`L110`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_fwd.py#L110) — Vocab-parallel *dense* (soft-target) CE on the per-shard ``V_local`` slice.
- `_label_smoothing_correction(logits, lse, targets, weights, label_smoothing)` — [`L131`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_fwd.py#L131) — Compute the label-smoothing loss correction in pure JAX.
- `_per_token_weights(targets, weights, ignore_index)` — [`L44`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_fwd.py#L44) — Build per-token float32 weights consistent with the TileLang interface.
- `fused_cross_entropy(logits, targets=None, weights=None, *, ignore_index: int = -100, label_smoothing: float = 0, z_loss: float = 0, soft_targets=None, reduction: str = "mean", vocab_parallel_axis: str | None = None)` — [`L155`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_fwd.py#L155) — JAX/XLA reference for fused cross-entropy.

