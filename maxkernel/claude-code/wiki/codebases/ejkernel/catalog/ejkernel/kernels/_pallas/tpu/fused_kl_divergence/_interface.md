---
title: 'Module: ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.fused_kl_divergence._interface`/
symbols:
  fused_kl_divergence: fused_kl_divergence().
  __all__: __all__.
---
# Module: [`ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_interface.py)

## Functions
- `fused_kl_divergence(student_logits: Float[Array, "... vocab_size"], teacher_logits: Float[Array, "... vocab_size"], weights: Float[Array, ...] | None = None, *, reduction: str = "mean", direction: str = "forward", temperature: float = 1, beta: float = 0.5, vocab_parallel_axis: str | None = None, block_v: int = 0, block_m: int = 0)` — [`L35`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_interface.py#L35) — Dispatch fused KL divergence to the TPU Pallas implementation.

## Module values
- `__all__` — [`L84`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_interface.py#L84)

