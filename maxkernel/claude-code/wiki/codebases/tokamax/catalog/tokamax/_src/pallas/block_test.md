---
title: 'Module: tokamax/_src/pallas/block_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/pallas/block_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.pallas.block_test`/
symbols:
  BlockTest.test_block_ref_bounds: BlockTest#test_block_ref_bounds().
  BlockTest.test_block_ref_bounds_checked: BlockTest#test_block_ref_bounds_checked().
  BlockTest.test_block_ref_inbounds_mask: BlockTest#test_block_ref_inbounds_mask().
  BlockTest.test_block_ref_at_masks: BlockTest#test_block_ref_at_masks().
  BlockTest.kernel: BlockTest#kernel().
  _SKIP_TPU_TEST_REASON: _SKIP_TPU_TEST_REASON.
  BlockTest: BlockTest#
---
# Module: [`tokamax/_src/pallas/block_test.py`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block_test.py)

## Classes
### `BlockTest`  ·  implements/extends TestCase
- def: [`tokamax/_src/pallas/block_test.py:31`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block_test.py#L31)
- signature: `class BlockTest(parameterized.TestCase):`
- members:
  - `kernel(x_ref, _)` — [`L44`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block_test.py#L44)
  - `test_block_ref_at_masks(self)` — [`L124`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block_test.py#L124)
  - `test_block_ref_bounds(self, block_shape, expected_bounds, at=None)` — [`L40`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block_test.py#L40)
  - `test_block_ref_bounds_checked(self, block_shape, expect_checked, at=None)` — [`L60`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block_test.py#L60)
  - `test_block_ref_inbounds_mask(self, block_shape, use_grid_spec=False)` — [`L79`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block_test.py#L79)
- uses (calls/refs, reference-scoped): [`pallas_call`](block.md#pallas_call), [`_SKIP_TPU_TEST_REASON`](block_test.md#_SKIP_TPU_TEST_REASON)

## Module values
- `_SKIP_TPU_TEST_REASON` — [`L26`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block_test.py#L26)

