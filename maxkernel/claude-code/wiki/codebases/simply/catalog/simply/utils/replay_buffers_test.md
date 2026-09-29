---
title: 'Module: simply/utils/replay_buffers_test.py'
type: catalog
provenance: extracted
module: simply/utils/replay_buffers_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.replay_buffers_test`/ReplayBuffersTest#
symbols:
  ReplayBuffersTest.test_prioritized_replay_buffer: test_prioritized_replay_buffer().
  ReplayBuffersTest._test_sample: _test_sample().
  ReplayBuffersTest.test_replay_buffer: test_replay_buffer().
  ReplayBuffersTest: ''
---
# Module: [`simply/utils/replay_buffers_test.py`](../../../../../../raw/code/simply/simply/utils/replay_buffers_test.py)

## Classes
### `ReplayBuffersTest`  ·  implements/extends TestCase
- def: [`simply/utils/replay_buffers_test.py:24`](../../../../../../raw/code/simply/simply/utils/replay_buffers_test.py#L24)
- signature: `class ReplayBuffersTest(absltest.TestCase):`
- members:
  - `test_prioritized_replay_buffer(self)` — [`L102`](../../../../../../raw/code/simply/simply/utils/replay_buffers_test.py#L102)
  - `test_replay_buffer(self)` — [`L26`](../../../../../../raw/code/simply/simply/utils/replay_buffers_test.py#L26)
- protocol/private: `_test_sample`[`L65`](../../../../../../raw/code/simply/simply/utils/replay_buffers_test.py#L65)
- uses (calls/refs, reference-scoped): [`extend`](replay_buffers.md#PrioritizedReplayBuffer.extend), [`append`](replay_buffers.md#PrioritizedReplayBuffer.append), [`append`](replay_buffers.md#ReplayBuffer.append), [`sample`](replay_buffers.md#PrioritizedReplayBuffer.sample), [`ReplayBuffer`](replay_buffers.md#ReplayBuffer), [`cursor`](replay_buffers.md#ReplayBuffer.cursor), [`extend`](replay_buffers.md#ReplayBuffer.extend), [`update_priorities`](replay_buffers.md#PrioritizedReplayBuffer.update_priorities), [`PrioritizedReplayBuffer`](replay_buffers.md#PrioritizedReplayBuffer), [`max_priority`](replay_buffers.md#PrioritizedReplayBuffer.max_priority), [`alpha`](replay_buffers.md#PrioritizedReplayBuffer.alpha), [`sample`](replay_buffers.md#ReplayBuffer.sample), [`iterator`](replay_buffers.md#ReplayBuffer.iterator), [`beta`](replay_buffers.md#PrioritizedReplayBuffer.beta), [`capacity`](replay_buffers.md#ReplayBuffer.capacity)

