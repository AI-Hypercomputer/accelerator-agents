---
title: 'Module: recurrentgemma/jax/griffin_test.py'
type: catalog
provenance: extracted
module: recurrentgemma/jax/griffin_test.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.jax.griffin_test`/GriffinTest#
symbols:
  GriffinTest.test_griffin_output_shape: test_griffin_output_shape().
  GriffinTest: ''
---
# Module: [`recurrentgemma/jax/griffin_test.py`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/griffin_test.py)

## Classes
### `GriffinTest`  ·  implements/extends TestCase
- def: [`recurrentgemma/jax/griffin_test.py:23`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/griffin_test.py#L23)
- signature: `class GriffinTest(absltest.TestCase):`
- members:
  - `test_griffin_output_shape(self)` — [`L25`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/griffin_test.py#L25)
- uses (calls/refs, reference-scoped): [`TemporalBlockType`](../common.md#TemporalBlockType), [`Griffin`](griffin.md#Griffin), [`RECURRENT`](../common.md#TemporalBlockType.RECURRENT), [`ATTENTION`](../common.md#TemporalBlockType.ATTENTION), [`vocab_size`](../common.md#GriffinConfig.vocab_size), [`width`](../common.md#GriffinConfig.width), [`block_types`](../common.md#GriffinConfig.block_types), [`attention_window_size`](../common.md#GriffinConfig.attention_window_size), [`GriffinConfig`](../common.md#GriffinConfig), [`lru_width`](../common.md#GriffinConfig.lru_width), [`num_heads`](../common.md#GriffinConfig.num_heads), [`AttentionBlockCache`](modules.md#AttentionBlockCache), [`embeddings_scale_by_sqrt_dim`](../common.md#GriffinConfig.embeddings_scale_by_sqrt_dim), [`logits_soft_cap`](../common.md#GriffinConfig.logits_soft_cap), [`mlp_expanded_width`](../common.md#GriffinConfig.mlp_expanded_width), [`RecurrentBlockCache`](modules.md#RecurrentBlockCache), [`keys`](modules.md#AttentionBlockCache.keys), [`num_tokens`](modules.md#AttentionBlockCache.num_tokens), [`conv1d_state`](modules.md#RecurrentBlockCache.conv1d_state), [`values`](modules.md#AttentionBlockCache.values), [`rg_lru_state`](modules.md#RecurrentBlockCache.rg_lru_state)

