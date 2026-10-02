---
title: 'Module: torchax/ops/jimage.py'
type: catalog
provenance: extracted
module: torchax/ops/jimage.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `torchax.ops.jimage`/
symbols:
  interpolate_bicubic_no_aa: interpolate_bicubic_no_aa().
  compute_contribs: compute_contribs().
  interpolate_along_axis_bchw: interpolate_along_axis_bchw().
  interpolate_along_axis_bchw.gather_and_weight: interpolate_along_axis_bchw().gather_and_weight().
  cubic_kernel: cubic_kernel().
  interpolate_along_axis_bchw.gather_and_weight.gather_one: interpolate_along_axis_bchw().gather_and_weight().gather_one().
  gather_weights: gather_weights().
---
# Module: [`torchax/ops/jimage.py`](../../../../../../raw/code/torchax/torchax/ops/jimage.py)

## Functions
- `compute_contribs(in_size, out_size, scale, support=2, align_corners=False, dtype=None)` — [`L31`](../../../../../../raw/code/torchax/torchax/ops/jimage.py#L31)
- `cubic_kernel(x, a=-0.75)` — [`L19`](../../../../../../raw/code/torchax/torchax/ops/jimage.py#L19) — Cubic kernel with a = -0.75 (PyTorch-like Keys kernel)
- `gather_and_weight(i)` — [`L73`](../../../../../../raw/code/torchax/torchax/ops/jimage.py#L73)
- `gather_one(offset)` — [`L77`](../../../../../../raw/code/torchax/torchax/ops/jimage.py#L77)
- `gather_weights(img, idxs, axis)` — [`L54`](../../../../../../raw/code/torchax/torchax/ops/jimage.py#L54) — Safely gather with boundary handling
- `interpolate_along_axis_bchw(img, idxs, weights, axis)` — [`L60`](../../../../../../raw/code/torchax/torchax/ops/jimage.py#L60) — Interpolate along H (axis=2) or W (axis=3) for tensor (B, C, H, W).
- `interpolate_bicubic_no_aa(img, out_h, out_w, align_corners=False)` — [`L93`](../../../../../../raw/code/torchax/torchax/ops/jimage.py#L93)

