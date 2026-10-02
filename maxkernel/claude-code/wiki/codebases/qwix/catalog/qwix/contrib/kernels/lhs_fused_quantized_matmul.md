---
title: 'Module: qwix/contrib/kernels/lhs_fused_quantized_matmul.py'
type: catalog
provenance: extracted
module: qwix/contrib/kernels/lhs_fused_quantized_matmul.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix.contrib.kernels.lhs_fused_quantized_matmul`/
symbols:
  lhs_fused_qmm_kernel: lhs_fused_qmm_kernel().
  lhs_fused_quantized_matmul: lhs_fused_quantized_matmul().
  lhs_fused_qmm_kernel.kernel_body: lhs_fused_qmm_kernel().kernel_body().
  _CORE_AXIS_NAME: _CORE_AXIS_NAME.
  quantize_a_tile: quantize_a_tile().
  lhs_fused_qmm_kernel.kernel_body._init: lhs_fused_qmm_kernel().kernel_body()._init().
  lhs_fused_qmm_kernel.kernel_body._write: lhs_fused_qmm_kernel().kernel_body()._write().
---
# Module: [`qwix/contrib/kernels/lhs_fused_quantized_matmul.py`](../../../../../../../raw/code/qwix/qwix/contrib/kernels/lhs_fused_quantized_matmul.py)

## Functions
- `_init()` — [`L111`](../../../../../../../raw/code/qwix/qwix/contrib/kernels/lhs_fused_quantized_matmul.py#L111)
- `_write()` — [`L153`](../../../../../../../raw/code/qwix/qwix/contrib/kernels/lhs_fused_quantized_matmul.py#L153)
- `kernel_body(x_vmem: jax.Ref, y_vmem: jax.Ref, sy_vmem: jax.Ref, o_vmem: jax.Ref)` — [`L105`](../../../../../../../raw/code/qwix/qwix/contrib/kernels/lhs_fused_quantized_matmul.py#L105)
- `lhs_fused_qmm_kernel(x_hbm: jax.Ref, y_hbm: jax.Ref, sy_hbm: jax.Ref, o_hbm: jax.Ref, accum_vmem: jax.Ref, *, bm: int, bk: int, bn: int, sm_global: int)` — [`L57`](../../../../../../../raw/code/qwix/qwix/contrib/kernels/lhs_fused_quantized_matmul.py#L57) — Fused Quantized Matmul kernel.
- `lhs_fused_quantized_matmul(x: jax.Array, y: jax.Array, sy: jax.Array, *, bm: int, bk: int, bn: int, sm: int, accum_dtype=jnp.float32)` — [`L169`](../../../../../../../raw/code/qwix/qwix/contrib/kernels/lhs_fused_quantized_matmul.py#L169) — LHS Fused Quantized Matmul.
- `quantize_a_tile(x: jax.Array, *, qtype=jnp.int8, method="absmax")` — [`L44`](../../../../../../../raw/code/qwix/qwix/contrib/kernels/lhs_fused_quantized_matmul.py#L44) — Quantizes a tile of LHS values.

## Module values
- `_CORE_AXIS_NAME` — [`L41`](../../../../../../../raw/code/qwix/qwix/contrib/kernels/lhs_fused_quantized_matmul.py#L41)

