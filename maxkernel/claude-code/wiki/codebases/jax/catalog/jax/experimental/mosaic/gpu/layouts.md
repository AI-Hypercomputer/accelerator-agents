---
title: 'Module: jax/experimental/mosaic/gpu/layouts.py'
type: catalog
provenance: extracted
module: jax/experimental/mosaic/gpu/layouts.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.mosaic.gpu.layouts`/
symbols:
  to_layout_attr: to_layout_attr().
  from_layout_attr: from_layout_attr().
  _to_tiled_layout_attr: _to_tiled_layout_attr().
  _from_tiled_layout_attr: _from_tiled_layout_attr().
  splat_is_compatible_with_tiled: splat_is_compatible_with_tiled().
  to_transform_attr: to_transform_attr().
  _to_strided_fragmented_layout_attr: _to_strided_fragmented_layout_attr().
  _from_strided_fragmented_layout_attr: _from_strided_fragmented_layout_attr().
  _to_tiled_layout_attr._int_or_replicated: _to_tiled_layout_attr()._int_or_replicated().
  _from_tiled_layout_attr._from_int_or_replicated_attr: _from_tiled_layout_attr()._from_int_or_replicated_attr().
  _to_splat_fragmented_layout_attr: _to_splat_fragmented_layout_attr().
  _from_splat_fragmented_layout_attr: _from_splat_fragmented_layout_attr().
  from_transform_attr: from_transform_attr().
  _to_tiled_layout_attr._tile_attr: _to_tiled_layout_attr()._tile_attr().
---
# Module: [`jax/experimental/mosaic/gpu/layouts.py`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layouts.py)

## Functions
- `_from_int_or_replicated_attr(d_attr: ir.Attribute)` — [`L91`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layouts.py#L91)
- `_from_splat_fragmented_layout_attr(attr: mgpu.WGSplatFragLayoutAttr)` — [`L34`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layouts.py#L34)
- `_from_strided_fragmented_layout_attr(attr: mgpu.WGStridedFragLayoutAttr)` — [`L48`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layouts.py#L48) — Constructs a WGStridedFragLayout from a #mosaic_gpu.WGStridedFragLayout attribute.
- `_from_tiled_layout_attr(attr: mgpu.TiledLayoutAttr)` — [`L86`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layouts.py#L86) — Constructs a TiledLayout from a #mosaic_gpu.TiledLayout attribute.
- `_int_or_replicated(d: int | fa.Replicated)` — [`L63`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layouts.py#L63)
- `_tile_attr(tile)` — [`L68`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layouts.py#L68)
- `_to_splat_fragmented_layout_attr(layout: fa.WGSplatFragLayout)` — [`L26`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layouts.py#L26) — Constructs a #mosaic_gpu.WGSplatFragLayout attribute from a WGSplatFragLayout.
- `_to_strided_fragmented_layout_attr(layout: fa.WGStridedFragLayout)` — [`L40`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layouts.py#L40) — Constructs a #mosaic_gpu.WGStridedFragLayout attribute from a WGStridedFragLayout.
- `_to_tiled_layout_attr(layout: fa.TiledLayout)` — [`L57`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layouts.py#L57) — Constructs a #mosaic_gpu.TiledLayout attribute from a TiledLayout.
- `from_layout_attr(attr: ir.Attribute)` — [`L124`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layouts.py#L124) — Constructs a layout from an MLIR attribute.
- `from_transform_attr(transform: ir.Attribute)` — [`L159`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layouts.py#L159)
- `splat_is_compatible_with_tiled(l1: fa.WGSplatFragLayout, l2: fa.TiledLayout)` — [`L138`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layouts.py#L138)
- `to_layout_attr(layout: fa.FragmentedLayout)` — [`L111`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layouts.py#L111) — Constructs an MLIR attribute that corresponds to the given layout.
- `to_transform_attr(transform: launch_context.MemRefTransform | mgpu.SwizzlingMode)` — [`L148`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layouts.py#L148)

