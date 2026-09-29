---
title: 'Module: simply/utils/replay_buffers.py'
type: catalog
provenance: extracted
module: simply/utils/replay_buffers.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.replay_buffers`/
symbols:
  PrioritizedReplayBuffer.extend: PrioritizedReplayBuffer#extend().
  PrioritizedReplayBuffer.append: PrioritizedReplayBuffer#append().
  ReplayBuffer.append: ReplayBuffer#append().
  PrioritizedReplayBuffer._sum_tree: PrioritizedReplayBuffer#_sum_tree.
  PrioritizedReplayBuffer.sample: PrioritizedReplayBuffer#sample().
  PrioritizedReplayBuffer._sample_indices: PrioritizedReplayBuffer#_sample_indices().
  ReplayBuffer: ReplayBuffer#
  ReplayBuffer.cursor: ReplayBuffer#cursor().
  ReplayBuffer.extend: ReplayBuffer#extend().
  PrioritizedReplayBuffer.update_priorities: PrioritizedReplayBuffer#update_priorities().
  PrioritizedReplayBuffer: PrioritizedReplayBuffer#
  PrioritizedReplayBuffer.max_priority: PrioritizedReplayBuffer#max_priority().
  PrioritizedReplayBuffer._max_priority: PrioritizedReplayBuffer#_max_priority.
  ReplayBuffer.sample: ReplayBuffer#sample().
  PrioritizedReplayBuffer.alpha: PrioritizedReplayBuffer#alpha().
  ReplayBuffer.iterator: ReplayBuffer#iterator().
  PrioritizedReplayBuffer._min_tree: PrioritizedReplayBuffer#_min_tree.
  ReplayBuffer._data: ReplayBuffer#_data.
  ReplayBuffer.__getitem__: ReplayBuffer#__getitem__().
  ReplayBuffer.__iter__: ReplayBuffer#__iter__().
  ReplayBuffer.capacity: ReplayBuffer#capacity().
  PrioritizedReplayBuffer.beta: PrioritizedReplayBuffer#beta().
  ReplayBuffer._cursor: ReplayBuffer#_cursor.
  ReplayBuffer._rng: ReplayBuffer#_rng.
  ReplayBuffer._capacity: ReplayBuffer#_capacity.
  ReplayBuffer.__len__: ReplayBuffer#__len__().
  PrioritizedReplayBuffer.__init__: PrioritizedReplayBuffer#__init__().
  ReplayBuffer.__init__: ReplayBuffer#__init__().
  PrioritizedReplayBuffer._alpha: PrioritizedReplayBuffer#_alpha.
  PrioritizedReplayBuffer._beta: PrioritizedReplayBuffer#_beta.
---
# Module: [`simply/utils/replay_buffers.py`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py)

## Classes
### `PrioritizedReplayBuffer`  ·  implements/extends ReplayBuffer
- def: [`simply/utils/replay_buffers.py:91`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L91)
- doc: The PrioritizedReplayBuffer for RL algorithms.
- signature: `class PrioritizedReplayBuffer(ReplayBuffer):`
- members:
  - `alpha(self)` — [`L109`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L109)
  - `append(self, x: common.PyTree, priority: float | None = None)` — [`L120`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L120) — documented in [simply-utils-replay_buffers](../../../concepts/simply-utils-replay_buffers.md)
  - `beta(self)` — [`L113`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L113)
  - `extend(self, batch: common.PyTree, priorities: np.ndarray | None = None)` — [`L132`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L132)
  - `max_priority(self)` — [`L117`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L117)
  - `sample(self, batch_size: int, replace: bool = False)` — [`L152`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L152)
  - `update_priorities(self, indices: np.ndarray, priorities: np.ndarray)` — [`L160`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L160) — documented in [simply-utils-replay_buffers](../../../concepts/simply-utils-replay_buffers.md)
- protocol/private: `__init__`[`L97`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L97), `_alpha`[`L102`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L102), `_beta`[`L103`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L103), `_max_priority`[`L106`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L106), `_min_tree`[`L105`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L105), `_sample_indices`[`L173`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L173), `_sum_tree`[`L104`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L104)
- uses (calls/refs, reference-scoped): [`PyTree`](common.md#PyTree.PyTree), [`scan_upper_bound`](segment_trees.md#SumSegmentTree.scan_upper_bound), [`append`](replay_buffers.md#ReplayBuffer.append), [`sum`](segment_trees.md#SumSegmentTree.sum), [`ReplayBuffer`](replay_buffers.md#ReplayBuffer), [`min`](segment_trees.md#MinSegmentTree.min), [`cursor`](replay_buffers.md#ReplayBuffer.cursor), [`SumSegmentTree`](segment_trees.md#SumSegmentTree), [`MinSegmentTree`](segment_trees.md#MinSegmentTree), [`dtype`](segment_trees.md#SegmentTree.dtype), [`_rng`](replay_buffers.md#ReplayBuffer._rng), [`__init__`](replay_buffers.md#ReplayBuffer.__init__)
- used by: [`test_prioritized_replay_buffer`](replay_buffers_test.md#ReplayBuffersTest.test_prioritized_replay_buffer), [`_test_sample`](replay_buffers_test.md#ReplayBuffersTest._test_sample), [`append`](replay_buffers.md#ReplayBuffer.append), [`ReplayBuffer`](replay_buffers.md#ReplayBuffer), [`extend`](replay_buffers.md#ReplayBuffer.extend), [`sample`](replay_buffers.md#ReplayBuffer.sample)

### `ReplayBuffer`  ·  implements/extends Sequence
- def: [`simply/utils/replay_buffers.py:26`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L26)
- doc: The ReplayBuffer for RL algorithms.
- signature: `class ReplayBuffer(Sequence[common.PyTree]):`
- members:
  - `append(self, x: common.PyTree)` — [`L58`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L58) — documented in [simply-utils-replay_buffers](../../../concepts/simply-utils-replay_buffers.md)
  - `capacity(self)` — [`L51`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L51)
  - `cursor(self)` — [`L55`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L55)
  - `extend(self, batch: common.PyTree)` — [`L65`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L65) — documented in [simply-utils-replay_buffers](../../../concepts/simply-utils-replay_buffers.md)
  - `iterator(self, batch_size: int = 1, shuffle: bool = False)` — [`L75`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L75)
  - `sample(self, batch_size: int, replace: bool = False)` — [`L70`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L70)
- protocol/private: `__getitem__`[`L35`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L35), `__init__`[`L29`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L29), `__iter__`[`L47`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L47), `__len__`[`L44`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L44), `_capacity`[`L30`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L30), `_cursor`[`L32`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L32), `_data`[`L31`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L31), `_rng`[`L33`](../../../../../../raw/code/simply/simply/utils/replay_buffers.py#L33)
- uses (calls/refs, reference-scoped): [`PyTree`](common.md#PyTree.PyTree), [`extend`](replay_buffers.md#PrioritizedReplayBuffer.extend), [`append`](replay_buffers.md#PrioritizedReplayBuffer.append), [`sample`](replay_buffers.md#PrioritizedReplayBuffer.sample), [`PrioritizedReplayBuffer`](replay_buffers.md#PrioritizedReplayBuffer)
- used by: [`run_experiment`](../rl_lib.md#run_experiment), [`test_prioritized_replay_buffer`](replay_buffers_test.md#ReplayBuffersTest.test_prioritized_replay_buffer), [`extend`](replay_buffers.md#PrioritizedReplayBuffer.extend), [`append`](replay_buffers.md#PrioritizedReplayBuffer.append), [`_test_sample`](replay_buffers_test.md#ReplayBuffersTest._test_sample), [`test_replay_buffer`](replay_buffers_test.md#ReplayBuffersTest.test_replay_buffer), [`_sample_indices`](replay_buffers.md#PrioritizedReplayBuffer._sample_indices), [`PrioritizedReplayBuffer`](replay_buffers.md#PrioritizedReplayBuffer), [`__init__`](replay_buffers.md#PrioritizedReplayBuffer.__init__)

