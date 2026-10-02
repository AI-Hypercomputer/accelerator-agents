---
title: 'Module: torchax/ops/jtorchvision_nms.py'
type: catalog
provenance: extracted
module: torchax/ops/jtorchvision_nms.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `torchax.ops.jtorchvision_nms`/
symbols:
  _suppression_loop_body: _suppression_loop_body().
  _NMS_TILE_SIZE: _NMS_TILE_SIZE.
  non_max_suppression_padded: non_max_suppression_padded().
  _cross_suppression: _cross_suppression().
  non_max_suppression_padded._loop_cond: non_max_suppression_padded()._loop_cond().
  nms: nms().
  _bbox_overlap: _bbox_overlap().
  _self_suppression: _self_suppression().
  _suppression_loop_body._loop_cond: _suppression_loop_body()._loop_cond().
  _suppression_loop_body._loop_cond2: _suppression_loop_body()._loop_cond2().
---
# Module: [`torchax/ops/jtorchvision_nms.py`](../../../../../../raw/code/torchax/torchax/ops/jtorchvision_nms.py)

## Functions
- `_bbox_overlap(boxes, gt_boxes)` — [`L28`](../../../../../../raw/code/torchax/torchax/ops/jtorchvision_nms.py#L28) — Find Bounding box overlap.
- `_cross_suppression(in_args)` — [`L81`](../../../../../../raw/code/torchax/torchax/ops/jtorchvision_nms.py#L81)
- `_loop_cond(in_args)` — [`L116`](../../../../../../raw/code/torchax/torchax/ops/jtorchvision_nms.py#L116)
- `_loop_cond(in_args)` — [`L215`](../../../../../../raw/code/torchax/torchax/ops/jtorchvision_nms.py#L215)
- `_loop_cond2(in_args)` — [`L133`](../../../../../../raw/code/torchax/torchax/ops/jtorchvision_nms.py#L133)
- `_self_suppression(in_args)` — [`L64`](../../../../../../raw/code/torchax/torchax/ops/jtorchvision_nms.py#L64)
- `_suppression_loop_body(in_args)` — [`L95`](../../../../../../raw/code/torchax/torchax/ops/jtorchvision_nms.py#L95) — Process boxes in the range [idx*_NMS_TILE_SIZE, (idx+1)*_NMS_TILE_SIZE).
- `nms(boxes, scores, iou_threshold)` — [`L255`](../../../../../../raw/code/torchax/torchax/ops/jtorchvision_nms.py#L255)
- `non_max_suppression_padded(scores, boxes, max_output_size, iou_threshold)` — [`L157`](../../../../../../raw/code/torchax/torchax/ops/jtorchvision_nms.py#L157) — A wrapper that handles non-maximum suppression.

## Module values
- `_NMS_TILE_SIZE` — [`L25`](../../../../../../raw/code/torchax/torchax/ops/jtorchvision_nms.py#L25)

