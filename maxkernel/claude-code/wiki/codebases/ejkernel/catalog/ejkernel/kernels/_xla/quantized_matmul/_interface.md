---
title: 'Module: ejkernel/kernels/_xla/quantized_matmul/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_xla/quantized_matmul/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._xla.quantized_matmul._interface`/
symbols:
  quantized_matmul: quantized_matmul().
  __all__: __all__.
---
# Module: [`ejkernel/kernels/_xla/quantized_matmul/_interface.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_interface.py)

## Functions
- `quantized_matmul(x: Float[Array, "m k"], w: Array, scales: Array, zeros: Array | None = None, transpose: bool = False, group_size: int | None = None, bits: int | None = None, mode: str = "affine", axis: QuantizationAxis | None = None, gemv_mode: GemvMode = "auto", revsplit_k: RevSplitKMode = "auto", revsplit_k_parts: int | None = None, *, tpu_path: str | None = None, allow_dense_fallback: bool = True, block_m: int = 128, block_n: int = 128, block_k: int = 64, use_bf16: bool = True, num_warps: int | None = None, num_stages: int | None = None, split_k: int | None = None)` — [`L34`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_interface.py#L34) — Compute quantized matrix multiplication using XLA (registry entry point).

## Module values
- `__all__` — [`L139`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_interface.py#L139)

