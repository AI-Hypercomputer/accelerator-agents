---
title: 'Module: torchax/ops/jax_reimplement.py'
type: catalog
provenance: extracted
module: torchax/ops/jax_reimplement.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `torchax.ops.jax_reimplement`/
symbols:
  _scale_and_translate: _scale_and_translate().
  scale_and_translate: scale_and_translate().
  compute_weight_mat: compute_weight_mat().
---
# Module: [`torchax/ops/jax_reimplement.py`](../../../../../../raw/code/torchax/torchax/ops/jax_reimplement.py)

## Functions
- `_scale_and_translate(x, output_shape: core.Shape, spatial_dims: Sequence[int], scale, translation, kernel, antialias: bool, precision)` — [`L82`](../../../../../../raw/code/torchax/torchax/ops/jax_reimplement.py#L82)
- `compute_weight_mat(input_size: core.DimSize, output_size: core.DimSize, scale, translation, kernel: Callable, antialias: bool)` — [`L31`](../../../../../../raw/code/torchax/torchax/ops/jax_reimplement.py#L31)
- `scale_and_translate(image, shape: core.Shape, spatial_dims: Sequence[int], scale, translation, method: str, antialias: bool = True, precision=lax.Precision.HIGHEST)` — [`L120`](../../../../../../raw/code/torchax/torchax/ops/jax_reimplement.py#L120) — Apply a scale and translation to an image.

