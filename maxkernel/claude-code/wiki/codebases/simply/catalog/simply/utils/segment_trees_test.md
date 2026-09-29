---
title: 'Module: simply/utils/segment_trees_test.py'
type: catalog
provenance: extracted
module: simply/utils/segment_trees_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.segment_trees_test`/SegmentTreesTest#
symbols:
  SegmentTreesTest.test_sum_segment_tree: test_sum_segment_tree().
  SegmentTreesTest.test_min_segment_tree: test_min_segment_tree().
  SegmentTreesTest._scan_upper_bound_ref: _scan_upper_bound_ref().
  SegmentTreesTest: ''
---
# Module: [`simply/utils/segment_trees_test.py`](../../../../../../raw/code/simply/simply/utils/segment_trees_test.py)

## Classes
### `SegmentTreesTest`  ·  implements/extends TestCase
- def: [`simply/utils/segment_trees_test.py:21`](../../../../../../raw/code/simply/simply/utils/segment_trees_test.py#L21)
- signature: `class SegmentTreesTest(absltest.TestCase):`
- members:
  - `test_min_segment_tree(self)` — [`L116`](../../../../../../raw/code/simply/simply/utils/segment_trees_test.py#L116)
  - `test_sum_segment_tree(self)` — [`L23`](../../../../../../raw/code/simply/simply/utils/segment_trees_test.py#L23)
- protocol/private: `_scan_upper_bound_ref`[`L85`](../../../../../../raw/code/simply/simply/utils/segment_trees_test.py#L85)
- uses (calls/refs, reference-scoped): [`scan_upper_bound`](segment_trees.md#SumSegmentTree.scan_upper_bound), [`sum`](segment_trees.md#SumSegmentTree.sum), [`capacity`](segment_trees.md#SegmentTree.capacity), [`min`](segment_trees.md#MinSegmentTree.min), [`ValueType`](segment_trees.md#ValueType), [`SumSegmentTree`](segment_trees.md#SumSegmentTree), [`MinSegmentTree`](segment_trees.md#MinSegmentTree), [`dtype`](segment_trees.md#SegmentTree.dtype), [`identity_element`](segment_trees.md#SegmentTree.identity_element)

