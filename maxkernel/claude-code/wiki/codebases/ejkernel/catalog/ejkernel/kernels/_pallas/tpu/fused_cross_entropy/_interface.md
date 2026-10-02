---
title: 'Module: ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.fused_cross_entropy._interface`/
symbols:
  fused_cross_entropy: fused_cross_entropy().
  __all__: __all__.
---
# Module: [`ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_interface.py)

## Functions
- `fused_cross_entropy(logits: Float[Array, "... vocab_size"], targets: Int[Array, ...] | None = None, weights: Float[Array, ...] | None = None, *, ignore_index: int = -100, label_smoothing: float = 0, z_loss: float = 0, soft_targets: Float[Array, "... vocab_size"] | None = None, reduction: str = "mean", vocab_parallel_axis: str | None = None, block_v: int = 0, block_m: int = 0)` — [`L35`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_interface.py#L35) — Dispatch fused sparse cross-entropy to the TPU Pallas implementation.

## Module values
- `__all__` — [`L90`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_interface.py#L90)

