---
title: 'Module: simply/utils/segment_trees.py'
type: catalog
provenance: extracted
module: simply/utils/segment_trees.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.segment_trees`/
symbols:
  SumSegmentTree.scan_upper_bound: SumSegmentTree#scan_upper_bound().
  SumSegmentTree.sum: SumSegmentTree#sum().
  SegmentTree.reduce: SegmentTree#reduce().
  SegmentTree._data: SegmentTree#_data.
  SegmentTree.capacity: SegmentTree#capacity().
  SegmentTree.update: SegmentTree#update().
  MinSegmentTree.min: MinSegmentTree#min().
  ValueType: ValueType.
  SumSegmentTree._scan_upper_bound: SumSegmentTree#_scan_upper_bound().
  SumSegmentTree._vectorized_scan_upper_bound: SumSegmentTree#_vectorized_scan_upper_bound().
  SegmentTree.__getitem__: SegmentTree#__getitem__().
  SumSegmentTree: SumSegmentTree#
  SegmentTree.__setitem__: SegmentTree#__setitem__().
  SegmentTree.identity_element: SegmentTree#identity_element().
  SegmentTree.dtype: SegmentTree#dtype().
  MinSegmentTree: MinSegmentTree#
  SegmentTree: SegmentTree#
  SegmentTree.__init__: SegmentTree#__init__().
  IndexType: IndexType.
  SegmentTree._size: SegmentTree#_size.
  SegmentTree._capacity: SegmentTree#_capacity.
  SegmentTree.__len__: SegmentTree#__len__().
  SumSegmentTree.__init__: SumSegmentTree#__init__().
  MinSegmentTree.__init__: MinSegmentTree#__init__().
  SegmentTree._reduce_op: SegmentTree#_reduce_op.
  SegmentTree._identity_element: SegmentTree#_identity_element.
---
# Module: [`simply/utils/segment_trees.py`](../../../../../../raw/code/simply/simply/utils/segment_trees.py)

## Classes
### `MinSegmentTree`  ·  implements/extends SegmentTree
- def: [`simply/utils/segment_trees.py:186`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L186) — documented in [simply-utils-replay_buffers](../../../concepts/simply-utils-replay_buffers.md)
- doc: SegmentTree that maintains the min value.
- signature: `class MinSegmentTree(SegmentTree):`
- members:
  - `min(self, start: int = 0, end: int | None = None)` — [`L201`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L201)
- protocol/private: `__init__`[`L189`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L189)
- uses (calls/refs, reference-scoped): [`reduce`](segment_trees.md#SegmentTree.reduce), [`ValueType`](segment_trees.md#ValueType), [`SegmentTree`](segment_trees.md#SegmentTree), [`__init__`](segment_trees.md#SegmentTree.__init__)
- used by: [`sample`](replay_buffers.md#PrioritizedReplayBuffer.sample), [`test_min_segment_tree`](segment_trees_test.md#SegmentTreesTest.test_min_segment_tree), [`SegmentTree`](segment_trees.md#SegmentTree), [`_min_tree`](replay_buffers.md#PrioritizedReplayBuffer._min_tree)

### `SegmentTree`
- def: [`simply/utils/segment_trees.py:24`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L24)
- doc: SegmentTree that supports single element updates and range reductions.
- signature: `class SegmentTree:`
- members:
  - `capacity(self)` — [`L66`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L66)
  - `dtype(self)` — [`L74`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L74)
  - `identity_element(self)` — [`L70`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L70)
  - `reduce(self, start: int = 0, end: int | None = None)` — [`L86`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L86) — Reduces the range [start, end). The time complexity is O(logN).
  - `update(self, key: int, value: ValueType)` — [`L77`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L77) — Updates data at key to value. The time complexity is O(logN).
- protocol/private: `__getitem__`[`L48`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L48), `__init__`[`L30`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L30), `__len__`[`L45`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L45), `__setitem__`[`L51`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L51), `_capacity`[`L38`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L38), `_data`[`L43`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L43), `_identity_element`[`L42`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L42), `_reduce_op`[`L41`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L41), `_size`[`L37`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L37)
- uses (calls/refs, reference-scoped): [`ValueType`](segment_trees.md#ValueType), [`SumSegmentTree`](segment_trees.md#SumSegmentTree), [`MinSegmentTree`](segment_trees.md#MinSegmentTree), [`IndexType`](segment_trees.md#IndexType)
- used by: [`sum`](segment_trees.md#SumSegmentTree.sum), [`test_sum_segment_tree`](segment_trees_test.md#SegmentTreesTest.test_sum_segment_tree), [`_sample_indices`](replay_buffers.md#PrioritizedReplayBuffer._sample_indices), [`test_min_segment_tree`](segment_trees_test.md#SegmentTreesTest.test_min_segment_tree), [`min`](segment_trees.md#MinSegmentTree.min), [`_scan_upper_bound`](segment_trees.md#SumSegmentTree._scan_upper_bound), [`_vectorized_scan_upper_bound`](segment_trees.md#SumSegmentTree._vectorized_scan_upper_bound), [`SumSegmentTree`](segment_trees.md#SumSegmentTree), [`MinSegmentTree`](segment_trees.md#MinSegmentTree), [`__init__`](segment_trees.md#MinSegmentTree.__init__), [`__init__`](segment_trees.md#SumSegmentTree.__init__)

### `SumSegmentTree`  ·  implements/extends SegmentTree
- def: [`simply/utils/segment_trees.py:116`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L116) — documented in [simply-utils-replay_buffers](../../../concepts/simply-utils-replay_buffers.md)
- doc: SegmentTree that maintains the sum.
- signature: `class SumSegmentTree(SegmentTree):`
- members:
  - `_scan_upper_bound(self, value: float)` — [`L135`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L135) — Returns the smallest `i` that sum(arr[0:i]) > value.
  - `_vectorized_scan_upper_bound(self, value: np.ndarray)` — [`L160`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L160) — The vectorized version of _scan_upper_bound.
  - `scan_upper_bound(self, value: ValueType)` — [`L129`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L129)
  - `sum(self, start: int = 0, end: int | None = None)` — [`L126`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L126)
- protocol/private: `__init__`[`L119`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L119)
- uses (calls/refs, reference-scoped): [`reduce`](segment_trees.md#SegmentTree.reduce), [`_data`](segment_trees.md#SegmentTree._data), [`capacity`](segment_trees.md#SegmentTree.capacity), [`ValueType`](segment_trees.md#ValueType), [`SegmentTree`](segment_trees.md#SegmentTree), [`__init__`](segment_trees.md#SegmentTree.__init__), [`IndexType`](segment_trees.md#IndexType), [`_size`](segment_trees.md#SegmentTree._size)
- used by: [`test_sum_segment_tree`](segment_trees_test.md#SegmentTreesTest.test_sum_segment_tree), [`_sum_tree`](replay_buffers.md#PrioritizedReplayBuffer._sum_tree), [`_sample_indices`](replay_buffers.md#PrioritizedReplayBuffer._sample_indices), [`SegmentTree`](segment_trees.md#SegmentTree)

## Module values
- `IndexType` — [`L20`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L20)
- `ValueType` — [`L21`](../../../../../../raw/code/simply/simply/utils/segment_trees.py#L21)

