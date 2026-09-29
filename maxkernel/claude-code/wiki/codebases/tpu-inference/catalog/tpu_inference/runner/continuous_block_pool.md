---
title: 'Module: tpu_inference/runner/continuous_block_pool.py'
type: catalog
provenance: extracted
module: tpu_inference/runner/continuous_block_pool.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.runner.continuous_block_pool`/
symbols:
  logger: logger.
  ContinuousFreeQueue: ContinuousFreeQueue#
  ContinuousFreeQueue.__init__: ContinuousFreeQueue#__init__().
  ContinuousFreeQueue.num_free_blocks: ContinuousFreeQueue#num_free_blocks().
  ContinuousFreeQueue._add_to_intervals: ContinuousFreeQueue#_add_to_intervals().
  ContinuousFreeQueue._remove_from_intervals: ContinuousFreeQueue#_remove_from_intervals().
  ContinuousFreeQueue.append_n: ContinuousFreeQueue#append_n().
  ContinuousFreeQueue.remove: ContinuousFreeQueue#remove().
  ContinuousFreeQueue.popleft: ContinuousFreeQueue#popleft().
  ContinuousFreeQueue.popleft_n: ContinuousFreeQueue#popleft_n().
---
# Module: [`tpu_inference/runner/continuous_block_pool.py`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/continuous_block_pool.py)

## Classes
### `ContinuousFreeQueue`
- def: [`tpu_inference/runner/continuous_block_pool.py:23`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/continuous_block_pool.py#L23)
- doc: A custom FreeKVCacheBlockQueue that maintains free blocks as continuous intervals
- signature: `class ContinuousFreeQueue:`
- members:
  - `append_n(self, blocks: List)` — [`L87`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/continuous_block_pool.py#L87)
  - `num_free_blocks(self)` — [`L40`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/continuous_block_pool.py#L40)
  - `popleft(self)` — [`L98`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/continuous_block_pool.py#L98)
  - `popleft_n(self, num_blocks: int)` — [`L117`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/continuous_block_pool.py#L117)
  - `remove(self, block)` — [`L93`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/continuous_block_pool.py#L93)
- protocol/private: `__init__`[`L33`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/continuous_block_pool.py#L33), `_add_to_intervals`[`L43`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/continuous_block_pool.py#L43), `_remove_from_intervals`[`L73`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/continuous_block_pool.py#L73)

## Module values
- `logger` — [`L20`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/continuous_block_pool.py#L20)

