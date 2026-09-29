---
title: 'Module: recurrentgemma/torch/griffin_test.py'
type: catalog
provenance: extracted
module: recurrentgemma/torch/griffin_test.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.torch.griffin_test`/GriffinTest#
symbols:
  GriffinTest.test_numerically_to_jax: test_numerically_to_jax().
  GriffinTest: ''
---
# Module: [`recurrentgemma/torch/griffin_test.py`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/griffin_test.py)

## Classes
### `GriffinTest`  ·  implements/extends TestCase
- def: [`recurrentgemma/torch/griffin_test.py:28`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/griffin_test.py#L28)
- signature: `class GriffinTest(parameterized.TestCase):`
- members:
  - `test_numerically_to_jax(self, vocab_size: int, width: int, mlp_expanded_width: int, num_heads: int, attention_window_size: int, scale_by_sqrt_dim: bool, seq_len: int, dtype: str, seed: int, num_unroll_steps: int = 2)` — [`L41`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/griffin_test.py#L41)
- uses (calls/refs, reference-scoped): [`TemporalBlockType`](../common.md#TemporalBlockType), [`Griffin`](griffin.md#Griffin), [`ScanType`](../common.md#ScanType), [`Griffin`](../jax/griffin.md#Griffin), [`RECURRENT`](../common.md#TemporalBlockType.RECURRENT), [`config`](../jax/griffin.md#Griffin.config), [`ATTENTION`](../common.md#TemporalBlockType.ATTENTION), [`vocab_size`](../common.md#GriffinConfig.vocab_size), [`width`](../common.md#GriffinConfig.width), [`block_types`](../common.md#GriffinConfig.block_types), [`attention_window_size`](../common.md#GriffinConfig.attention_window_size), [`GriffinConfig`](../common.md#GriffinConfig), [`num_heads`](../common.md#GriffinConfig.num_heads), [`embeddings_scale_by_sqrt_dim`](../common.md#GriffinConfig.embeddings_scale_by_sqrt_dim), [`logits_soft_cap`](../common.md#GriffinConfig.logits_soft_cap), [`mlp_expanded_width`](../common.md#GriffinConfig.mlp_expanded_width), [`param_dtype`](../jax/griffin.md#Griffin.param_dtype), [`LINEAR_NATIVE`](../common.md#ScanType.LINEAR_NATIVE), [`scan_type`](../common.md#GriffinConfig.scan_type)  (1 test-only)

