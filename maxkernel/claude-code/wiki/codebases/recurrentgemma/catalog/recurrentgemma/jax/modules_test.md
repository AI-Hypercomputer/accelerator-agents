---
title: 'Module: recurrentgemma/jax/modules_test.py'
type: catalog
provenance: extracted
module: recurrentgemma/jax/modules_test.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.jax.modules_test`/
symbols:
  RecurrentBlockTest.test_recurrent_block_output_shapes: RecurrentBlockTest#test_recurrent_block_output_shapes().
  LocalAttentionTest.test_local_attention_output_shapes: LocalAttentionTest#test_local_attention_output_shapes().
  LocalAttentionTest.test_local_attention_updates_cache_correctly: LocalAttentionTest#test_local_attention_updates_cache_correctly().
  LocalAttentionTest: LocalAttentionTest#
  RecurrentBlockTest: RecurrentBlockTest#
---
# Module: [`recurrentgemma/jax/modules_test.py`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules_test.py)

## Classes
### `LocalAttentionTest`  ·  implements/extends TestCase
- def: [`recurrentgemma/jax/modules_test.py:26`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules_test.py#L26)
- signature: `class LocalAttentionTest(parameterized.TestCase):`
- members:
  - `test_local_attention_output_shapes(self, seq_len: int, seed: int = 12319843)` — [`L29`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules_test.py#L29)
  - `test_local_attention_updates_cache_correctly(self, seed: int = 874321)` — [`L71`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules_test.py#L71)
- uses (calls/refs, reference-scoped): [`LocalAttentionBlock`](modules.md#LocalAttentionBlock), [`init_cache`](modules.md#LocalAttentionBlock.init_cache), [`num_heads`](modules.md#LocalAttentionBlock.num_heads), [`width`](modules.md#LocalAttentionBlock.width), [`window_size`](modules.md#LocalAttentionBlock.window_size)

### `RecurrentBlockTest`  ·  implements/extends TestCase
- def: [`recurrentgemma/jax/modules_test.py:114`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules_test.py#L114)
- signature: `class RecurrentBlockTest(parameterized.TestCase):`
- members:
  - `test_recurrent_block_output_shapes(self, seq_len: int, seed: int = 1208743)` — [`L117`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules_test.py#L117)
- uses (calls/refs, reference-scoped): [`TemporalBlockType`](../common.md#TemporalBlockType), [`RecurrentBlock`](modules.md#RecurrentBlock), [`RECURRENT`](../common.md#TemporalBlockType.RECURRENT), [`ResidualBlock`](modules.md#ResidualBlock), [`init_cache`](modules.md#ResidualBlock.init_cache), [`width`](modules.md#RecurrentBlock.width), [`lru_width`](modules.md#RecurrentBlock.lru_width), [`num_heads`](modules.md#RecurrentBlock.num_heads), [`conv1d_temporal_width`](modules.md#RecurrentBlock.conv1d_temporal_width)

