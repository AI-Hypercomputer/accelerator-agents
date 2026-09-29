---
title: 'Module: tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.linear_softmax_cross_entropy_loss.reference`/linear_softmax_cross_entropy_loss_
symbols:
  linear_softmax_cross_entropy_loss_fwd_reference: fwd_reference().
  linear_softmax_cross_entropy_loss_bwd_reference: bwd_reference().
---
# Module: [`tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference.py)

## Functions
- `linear_softmax_cross_entropy_loss_bwd_reference(dout: Real[Array, ""], lse: Real[Array, B], x: Real[Array, "B H"], labels: Integer[Array, B], w: Real[Array, "H V"], *, reduction: Literal["sum", "mean"] = "sum")` — [`L50`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference.py#L50) — The reference Jax implementation of the linear softmax cross-entropy loss backward kernel.
- `linear_softmax_cross_entropy_loss_fwd_reference(x: Real[Array, "B H"], labels: Integer[Array, B], w: Real[Array, "H V"], *, reduction: Literal["sum", "mean"] = "sum")` — [`L25`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference.py#L25) — The reference Jax implementation of the linear softmax cross-entropy loss.

