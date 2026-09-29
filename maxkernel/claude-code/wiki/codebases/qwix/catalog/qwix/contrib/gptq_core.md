---
title: 'Module: qwix/contrib/gptq_core.py'
type: catalog
provenance: extracted
module: qwix/contrib/gptq_core.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix.contrib.gptq_core`/
symbols:
  quantize_weight: quantize_weight().
  find_params: find_params().
  quantize: quantize().
  compute_hessian: compute_hessian().
  cholesky_inverse: cholesky_inverse().
---
# Module: [`qwix/contrib/gptq_core.py`](../../../../../../raw/code/qwix/qwix/contrib/gptq_core.py)

## Functions
- `cholesky_inverse(L: jax.Array)` — [`L31`](../../../../../../raw/code/qwix/qwix/contrib/gptq_core.py#L31)
- `compute_hessian(X: jax.Array)` — [`L189`](../../../../../../raw/code/qwix/qwix/contrib/gptq_core.py#L189) — Computes the Hessian of the GPTQ objective function.
- `find_params(w: jax.Array, how: qarray.HowToQuantize)` — [`L35`](../../../../../../raw/code/qwix/qwix/contrib/gptq_core.py#L35) — Finds the optimal quantization parameters for a given weight tensor.
- `quantize(w: jax.Array, qtype: jax.typing.DTypeLike, scale: jax.Array, zero_point: jax.Array | None)` — [`L53`](../../../../../../raw/code/qwix/qwix/contrib/gptq_core.py#L53) — Quantize w and return the raw quantized and dequantized w.
- `quantize_weight(W: jax.Array, H: jax.Array, how: qarray.HowToQuantize, blocksize: int = 128, percdamp: float = 0.01)` — [`L76`](../../../../../../raw/code/qwix/qwix/contrib/gptq_core.py#L76) — Quantize a weight matrix using GPTQ. — documented in [qwix-contrib-gptq](../../../concepts/qwix-contrib-gptq.md)

