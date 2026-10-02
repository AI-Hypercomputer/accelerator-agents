---
title: 'Module: qwix/contrib/kernels/quantized_matmul.py'
type: catalog
provenance: extracted
module: qwix/contrib/kernels/quantized_matmul.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix.contrib.kernels.quantized_matmul`/
symbols:
  can_use_qmm_in_dot_general: can_use_qmm_in_dot_general().
  quantized_matmul_kernel: quantized_matmul_kernel().
  quantized_matmul: quantized_matmul().
  can_use_qmm: can_use_qmm().
  _CORE_AXIS_NAME: _CORE_AXIS_NAME.
  QuantizedMatmulConfig: QuantizedMatmulConfig#
  QuantizedMatmulConfig.bm: QuantizedMatmulConfig#bm.
  QuantizedMatmulConfig.bk: QuantizedMatmulConfig#bk.
  QuantizedMatmulConfig.bn: QuantizedMatmulConfig#bn.
  quantized_matmul_kernel.quantized_matmul_body: quantized_matmul_kernel().quantized_matmul_body().
  QuantizedMatmulConfig.dtype: QuantizedMatmulConfig#dtype.
  quantized_matmul_kernel.quantized_matmul_body._init: quantized_matmul_kernel().quantized_matmul_body()._init().
  quantized_matmul_kernel.quantized_matmul_body._write: quantized_matmul_kernel().quantized_matmul_body()._write().
---
# Module: [`qwix/contrib/kernels/quantized_matmul.py`](../../../../../../../raw/code/qwix/qwix/contrib/kernels/quantized_matmul.py)

## Classes
### `QuantizedMatmulConfig`
- def: [`qwix/contrib/kernels/quantized_matmul.py:49`](../../../../../../../raw/code/qwix/qwix/contrib/kernels/quantized_matmul.py#L49)
- doc: Configuration for the quantized matmul kernel determined by TPUv5 sweep.
- signature: `class QuantizedMatmulConfig:`
- members:
  - `bk` — [`L53`](../../../../../../../raw/code/qwix/qwix/contrib/kernels/quantized_matmul.py#L53)
  - `bm` — [`L52`](../../../../../../../raw/code/qwix/qwix/contrib/kernels/quantized_matmul.py#L52)
  - `bn` — [`L54`](../../../../../../../raw/code/qwix/qwix/contrib/kernels/quantized_matmul.py#L54)
  - `dtype` — [`L55`](../../../../../../../raw/code/qwix/qwix/contrib/kernels/quantized_matmul.py#L55)
- used by: [`can_use_qmm_in_dot_general`](quantized_matmul.md#can_use_qmm_in_dot_general)

## Functions
- `_init()` — [`L252`](../../../../../../../raw/code/qwix/qwix/contrib/kernels/quantized_matmul.py#L252)
- `_write()` — [`L287`](../../../../../../../raw/code/qwix/qwix/contrib/kernels/quantized_matmul.py#L287)
- `can_use_qmm(x, sx, y, sy, *, bm, bk, bn)` — [`L58`](../../../../../../../raw/code/qwix/qwix/contrib/kernels/quantized_matmul.py#L58) — Returns whether the quantized matmul can be used.
- `can_use_qmm_in_dot_general(lhs: qarray.QArray | jax.Array, rhs: qarray.QArray | jax.Array, dimension_numbers: Any, *, config: QuantizedMatmulConfig)` — [`L96`](../../../../../../../raw/code/qwix/qwix/contrib/kernels/quantized_matmul.py#L96) — Returns whether the quantized matmul can be used in dot_general.
- `quantized_matmul(x: jax.Array, sx: jax.Array, y: jax.Array, sy: jax.Array, *, bm: int, bk: int, bn: int, max_sublock_size_m: int = 128, max_sublock_size_n: int = 128, max_sublock_size_k: int = 128, accum_dtype=jnp.float32, dtype)` — [`L304`](../../../../../../../raw/code/qwix/qwix/contrib/kernels/quantized_matmul.py#L304) — Computes a quantized matmul using Pallas.
- `quantized_matmul_body(x_vmem: jax.Ref, sx_vmem: jax.Ref, y_vmem: jax.Ref, sy_vmem: jax.Ref, o_vmem: jax.Ref)` — [`L241`](../../../../../../../raw/code/qwix/qwix/contrib/kernels/quantized_matmul.py#L241)
- `quantized_matmul_kernel(x_hbm: jax.Ref, sx_hbm: jax.Ref, y_hbm: jax.Ref, sy_hbm: jax.Ref, o_hbm: jax.Ref, accum_vmem: jax.Ref, *, bm: int, bk: int, bn: int, max_sublock_size_m: int = 128, max_sublock_size_n: int = 128, max_sublock_size_k: int = 128)` — [`L138`](../../../../../../../raw/code/qwix/qwix/contrib/kernels/quantized_matmul.py#L138) — Quantized matmul kernel.

## Module values
- `_CORE_AXIS_NAME` — [`L45`](../../../../../../../raw/code/qwix/qwix/contrib/kernels/quantized_matmul.py#L45)

