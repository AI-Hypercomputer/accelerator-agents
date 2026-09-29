---
title: 'Module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_common_test`/GroupInfoTest#
symbols:
  GroupInfoTest.test_create_aligned: test_create_aligned().
  GroupInfoTest.test_create_aligned_empty_groups: test_create_aligned_empty_groups().
  GroupInfoTest.test_create_aligned_multiple_blocks_per_group: test_create_aligned_multiple_blocks_per_group().
  GroupInfoTest.test_create_aligned_multiple_blocks_per_group_aligned: test_create_aligned_multiple_blocks_per_group_aligned().
  GroupInfoTest: ''
---
# Module: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common_test.py)

## Classes
### `GroupInfoTest`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common_test.py:25`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common_test.py#L25)
- signature: `class GroupInfoTest(parameterized.TestCase):`
- members:
  - `test_create_aligned(self, tid_size)` — [`L28`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common_test.py#L28)
  - `test_create_aligned_empty_groups(self)` — [`L44`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common_test.py#L44)
  - `test_create_aligned_multiple_blocks_per_group(self)` — [`L58`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common_test.py#L58)
  - `test_create_aligned_multiple_blocks_per_group_aligned(self)` — [`L72`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common_test.py#L72)
- uses (calls/refs, reference-scoped): [`create_aligned`](pallas_mosaic_gpu_common.md#GroupInfo.create_aligned), [`GroupInfo`](pallas_mosaic_gpu_common.md#GroupInfo)

