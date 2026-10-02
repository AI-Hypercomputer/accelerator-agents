---
title: 'Module: recurrentgemma/torch/modules_test.py'
type: catalog
provenance: extracted
module: recurrentgemma/torch/modules_test.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.torch.modules_test`/
symbols:
  ResidualBlockTest.test_numerically_to_jax: ResidualBlockTest#test_numerically_to_jax().
  EmbedderTest.test_numerically_to_jax: EmbedderTest#test_numerically_to_jax().
  RecurrentBlockTest.test_numerically_to_jax: RecurrentBlockTest#test_numerically_to_jax().
  LocalAttentionTest.test_numerically_to_jax: LocalAttentionTest#test_numerically_to_jax().
  MLPBlockTest.test_numerically_to_jax: MLPBlockTest#test_numerically_to_jax().
  EmbedderTest: EmbedderTest#
  LocalAttentionTest: LocalAttentionTest#
  RecurrentBlockTest: RecurrentBlockTest#
  MLPBlockTest: MLPBlockTest#
  ResidualBlockTest: ResidualBlockTest#
---
# Module: [`recurrentgemma/torch/modules_test.py`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules_test.py)

## Classes
### `EmbedderTest`  ·  implements/extends TestCase
- def: [`recurrentgemma/torch/modules_test.py:30`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules_test.py#L30)
- signature: `class EmbedderTest(parameterized.TestCase):`
- members:
  - `test_numerically_to_jax(self, vocab_size: int, embed_dim: int, scale_by_sqrt_dim: bool, dtype: str, seed: int)` — [`L39`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules_test.py#L39)
- uses (calls/refs, reference-scoped): [`Griffin`](griffin.md#Griffin), [`Griffin`](../jax/griffin.md#Griffin), [`config`](../jax/griffin.md#Griffin.config), [`vocab_size`](../common.md#GriffinConfig.vocab_size), [`width`](../common.md#GriffinConfig.width), [`block_types`](../common.md#GriffinConfig.block_types), [`attention_window_size`](../common.md#GriffinConfig.attention_window_size), [`GriffinConfig`](../common.md#GriffinConfig), [`num_heads`](../common.md#GriffinConfig.num_heads), [`embeddings_scale_by_sqrt_dim`](../common.md#GriffinConfig.embeddings_scale_by_sqrt_dim), [`logits_soft_cap`](../common.md#GriffinConfig.logits_soft_cap), [`mlp_expanded_width`](../common.md#GriffinConfig.mlp_expanded_width), [`param_dtype`](../jax/griffin.md#Griffin.param_dtype)  (1 test-only)

### `LocalAttentionTest`  ·  implements/extends TestCase
- def: [`recurrentgemma/torch/modules_test.py:77`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules_test.py#L77)
- signature: `class LocalAttentionTest(parameterized.TestCase):`
- members:
  - `test_numerically_to_jax(self, width: int, num_heads: int, window_size: int, seq_len: int, dtype: str, seed: int, num_unroll_steps: int = 2)` — [`L87`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules_test.py#L87)
- uses (calls/refs, reference-scoped): [`LocalAttentionBlock`](../jax/modules.md#LocalAttentionBlock), [`LocalAttentionBlock`](modules.md#LocalAttentionBlock), [`param_dtype`](../jax/modules.md#LocalAttentionBlock.param_dtype), [`num_heads`](../jax/modules.md#LocalAttentionBlock.num_heads), [`width`](../jax/modules.md#LocalAttentionBlock.width), [`window_size`](../jax/modules.md#LocalAttentionBlock.window_size)  (1 test-only)

### `MLPBlockTest`  ·  implements/extends TestCase
- def: [`recurrentgemma/torch/modules_test.py:162`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules_test.py#L162)
- signature: `class MLPBlockTest(parameterized.TestCase):`
- members:
  - `test_numerically_to_jax(self, width: int, expanded_width: int, dtype: str, seed: int, num_unroll_steps: int = 2)` — [`L170`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules_test.py#L170)
- uses (calls/refs, reference-scoped): [`param_dtype`](../jax/modules.md#MLPBlock.param_dtype), [`expanded_width`](../jax/modules.md#MLPBlock.expanded_width), [`width`](../jax/modules.md#MLPBlock.width), [`MLPBlock`](../jax/modules.md#MLPBlock), [`MLPBlock`](modules.md#MLPBlock)  (1 test-only)

### `RecurrentBlockTest`  ·  implements/extends TestCase
- def: [`recurrentgemma/torch/modules_test.py:119`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules_test.py#L119)
- signature: `class RecurrentBlockTest(parameterized.TestCase):`
- members:
  - `test_numerically_to_jax(self, width: int, num_heads: int, lru_width: int, seq_len: int, dtype: str, seed: int, num_unroll_steps: int = 2)` — [`L129`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules_test.py#L129)
- uses (calls/refs, reference-scoped): [`ScanType`](../common.md#ScanType), [`RecurrentBlock`](modules.md#RecurrentBlock), [`RecurrentBlock`](../jax/modules.md#RecurrentBlock), [`param_dtype`](../jax/modules.md#RecurrentBlock.param_dtype), [`scan_type`](../jax/modules.md#RecurrentBlock.scan_type), [`LINEAR_NATIVE`](../common.md#ScanType.LINEAR_NATIVE), [`width`](../jax/modules.md#RecurrentBlock.width), [`lru_width`](../jax/modules.md#RecurrentBlock.lru_width), [`num_heads`](../jax/modules.md#RecurrentBlock.num_heads)  (1 test-only)

### `ResidualBlockTest`  ·  implements/extends TestCase
- def: [`recurrentgemma/torch/modules_test.py:198`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules_test.py#L198)
- signature: `class ResidualBlockTest(parameterized.TestCase):`
- members:
  - `test_numerically_to_jax(self, width: int, mlp_expanded_width: int, lru_width: int, num_heads: int, attention_window_size: int, temporal_block_type: common.TemporalBlockType, seq_len: int, dtype: str, seed: int, num_unroll_steps: int = 2)` — [`L214`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules_test.py#L214)
- uses (calls/refs, reference-scoped): [`TemporalBlockType`](../common.md#TemporalBlockType), [`ScanType`](../common.md#ScanType), [`RECURRENT`](../common.md#TemporalBlockType.RECURRENT), [`ResidualBlock`](../jax/modules.md#ResidualBlock), [`ResidualBlock`](modules.md#ResidualBlock), [`ATTENTION`](../common.md#TemporalBlockType.ATTENTION), [`param_dtype`](../jax/modules.md#ResidualBlock.param_dtype), [`scan_type`](../jax/modules.md#ResidualBlock.scan_type), [`LINEAR_NATIVE`](../common.md#ScanType.LINEAR_NATIVE), [`width`](../jax/modules.md#ResidualBlock.width), [`temporal_block_type`](../jax/modules.md#ResidualBlock.temporal_block_type), [`num_heads`](../jax/modules.md#ResidualBlock.num_heads), [`attention_window_size`](../jax/modules.md#ResidualBlock.attention_window_size), [`lru_width`](../jax/modules.md#ResidualBlock.lru_width), [`mlp_expanded_width`](../jax/modules.md#ResidualBlock.mlp_expanded_width)  (1 test-only)

