---
title: 'Module: axlearn/common/flash_attention/common_test.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/common_test.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.common_test`/
symbols:
  BuildMaskTest: BuildMaskTest#
  BuildMaskTest.test_sliding_window_fast_path: BuildMaskTest#test_sliding_window_fast_path().
  UtilsTest: UtilsTest#
  UtilsTest.test_maybe_pad_inputs: UtilsTest#test_maybe_pad_inputs().
  SplitPrngKeysTest: SplitPrngKeysTest#
  SplitPrngKeysTest.test_no_sharding: SplitPrngKeysTest#test_no_sharding().
  SplitPrngKeysTest.test_sharding_shapes: SplitPrngKeysTest#test_sharding_shapes().
---
# Module: [`axlearn/common/flash_attention/common_test.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common_test.py)

## Classes
### `BuildMaskTest`
- def: [`axlearn/common/flash_attention/common_test.py:25`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common_test.py#L25)
- signature: `class BuildMaskTest(TestCase):`
- members:
  - `test_sliding_window_fast_path(self, sliding_window_sz, seq_len, block_size)` — [`L31`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common_test.py#L31)

### `SplitPrngKeysTest`
- def: [`axlearn/common/flash_attention/common_test.py:82`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common_test.py#L82)
- doc: Tests for split_prng_keys_for_shard_map function.
- signature: `class SplitPrngKeysTest(TestCase):`
- members:
  - `test_no_sharding(self, mesh_axes, spec)` — [`L90`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common_test.py#L90) — Returns original key when num_devices=1.
  - `test_sharding_shapes(self, mesh_axes, mesh_shape, spec, axis_sizes)` — [`L106`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common_test.py#L106) — Verifies output shape matches sharding_spec structure and keys are unique.

### `UtilsTest`
- def: [`axlearn/common/flash_attention/common_test.py:38`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common_test.py#L38)
- signature: `class UtilsTest(TestCase):`
- members:
  - `test_maybe_pad_inputs(self, block_size, input_len, output_len)` — [`L45`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/common_test.py#L45)

