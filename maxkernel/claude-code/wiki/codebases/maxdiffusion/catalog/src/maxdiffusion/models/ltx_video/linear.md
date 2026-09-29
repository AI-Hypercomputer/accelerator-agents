---
title: 'Module: src/maxdiffusion/models/ltx_video/linear.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/linear.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.linear`/
symbols:
  DenseGeneral.__call__: DenseGeneral#__call__().
  InitializerAxis: InitializerAxis.
  KernelInitializer: KernelInitializer.
  NdInitializer: NdInitializer.
  Shape: Shape.
  Initializer: Initializer.
  DenseGeneral.kernel_init: DenseGeneral#kernel_init.
  DenseGeneral.bias_init: DenseGeneral#bias_init.
  DenseGeneral.compute_dot_general: DenseGeneral#compute_dot_general().
  DenseGeneral.dtype: DenseGeneral#dtype.
  _canonicalize_tuple: _canonicalize_tuple().
  DenseGeneral.weight_dtype: DenseGeneral#weight_dtype.
  DenseGeneral.kernel_axes: DenseGeneral#kernel_axes.
  _normalize_axes: _normalize_axes().
  DenseGeneral.features: DenseGeneral#features.
  DenseGeneral.axis: DenseGeneral#axis.
  DenseGeneral.use_bias: DenseGeneral#use_bias.
  DenseGeneral.matmul_precision: DenseGeneral#matmul_precision.
  DenseGeneral: DenseGeneral#
---
# Module: [`src/maxdiffusion/models/ltx_video/linear.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/linear.py)

## Classes
### `DenseGeneral`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx_video/linear.py:47`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/linear.py#L47)
- doc: A linear transformation with flexible axes.
- signature: `class DenseGeneral(nn.Module):`
- members:
  - `__call__(self, inputs: jax.Array)` — [`L75`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/linear.py#L75) — Applies a linear transformation to the inputs along multiple dimensions.
  - `compute_dot_general(inputs, kernel, axis, contract_ind)` — [`L85`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/linear.py#L85) — Computes a dot_general operation that may be quantized.
  - `axis` — [`L64`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/linear.py#L64)
  - `bias_init` — [`L72`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/linear.py#L72)
  - `dtype` — [`L66`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/linear.py#L66)
  - `features` — [`L63`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/linear.py#L63)
  - `kernel_axes` — [`L68`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/linear.py#L68)
  - `kernel_init` — [`L67`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/linear.py#L67)
  - `matmul_precision` — [`L70`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/linear.py#L70)
  - `use_bias` — [`L69`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/linear.py#L69)
  - `weight_dtype` — [`L65`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/linear.py#L65)
- uses (calls/refs, reference-scoped): [`KernelInitializer`](linear.md#KernelInitializer), [`Initializer`](linear.md#Initializer), [`_canonicalize_tuple`](linear.md#_canonicalize_tuple), [`_normalize_axes`](linear.md#_normalize_axes)

## Functions
- `_canonicalize_tuple(x)` — [`L36`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/linear.py#L36)
- `_normalize_axes(axes: Iterable[int], ndim: int)` — [`L31`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/linear.py#L31)

## Module values
- `Initializer` — [`L27`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/linear.py#L27)
- `InitializerAxis` — [`L28`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/linear.py#L28)
- `KernelInitializer` — [`L44`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/linear.py#L44)
- `NdInitializer` — [`L43`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/linear.py#L43)
- `Shape` — [`L26`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/linear.py#L26)

