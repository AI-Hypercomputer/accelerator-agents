---
title: 'Module: jax/experimental/pallas/ops/gpu/softmax.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/gpu/softmax.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.gpu.softmax`/
symbols:
  softmax: softmax().
  _vmappable_softmax_kernel: _vmappable_softmax_kernel().
---
# Module: [`jax/experimental/pallas/ops/gpu/softmax.py`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/softmax.py)

## Functions
- `_vmappable_softmax_kernel(input_ref, probs_ref, *, block_row: int)` — [`L24`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/softmax.py#L24)
- `softmax(x: jax.Array, *, axis: int = -1, num_warps: int = 4, interpret: bool = False, debug: bool = False)` — [`L54`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/softmax.py#L54) — Computes the softmax of the input array along the specified axis.

