---
title: 'Module: recurrentgemma/conversion_test.py'
type: catalog
provenance: extracted
module: recurrentgemma/conversion_test.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.conversion_test`/
symbols:
  ConversionTest.test_conversion_back_and_forth: ConversionTest#test_conversion_back_and_forth().
  BLOCK_TYPES: BLOCK_TYPES.
  BATCH_SIZE: BATCH_SIZE.
  SEQ_LEN: SEQ_LEN.
  VOCAB_SIZE: VOCAB_SIZE.
  WIDTH: WIDTH.
  MLP_EXPANDED_WIDTH: MLP_EXPANDED_WIDTH.
  NUM_HEADS: NUM_HEADS.
  LRU_WIDTH: LRU_WIDTH.
  NUM_SAMPLING_STEPS: NUM_SAMPLING_STEPS.
  ConversionTest: ConversionTest#
---
# Module: [`recurrentgemma/conversion_test.py`](../../../../../raw/code/recurrentgemma/recurrentgemma/conversion_test.py)

## Classes
### `ConversionTest`  ·  implements/extends TestCase
- def: [`recurrentgemma/conversion_test.py:39`](../../../../../raw/code/recurrentgemma/recurrentgemma/conversion_test.py#L39)
- signature: `class ConversionTest(absltest.TestCase):`
- members:
  - `test_conversion_back_and_forth(self, seed: int = 1287312)` — [`L41`](../../../../../raw/code/recurrentgemma/recurrentgemma/conversion_test.py#L41)
- uses (calls/refs, reference-scoped): [`Griffin`](jax/griffin.md#Griffin), [`vocab_size`](common.md#GriffinConfig.vocab_size), [`width`](common.md#GriffinConfig.width), [`block_types`](common.md#GriffinConfig.block_types), [`attention_window_size`](common.md#GriffinConfig.attention_window_size), [`GriffinConfig`](common.md#GriffinConfig), [`lru_width`](common.md#GriffinConfig.lru_width), [`num_heads`](common.md#GriffinConfig.num_heads), [`embeddings_scale_by_sqrt_dim`](common.md#GriffinConfig.embeddings_scale_by_sqrt_dim), [`logits_soft_cap`](common.md#GriffinConfig.logits_soft_cap), [`mlp_expanded_width`](common.md#GriffinConfig.mlp_expanded_width), [`BLOCK_TYPES`](conversion_test.md#BLOCK_TYPES), [`flax_params_to_pytorch_state_dict`](conversion.md#flax_params_to_pytorch_state_dict), [`pytorch_state_dict_to_flax_params`](conversion.md#pytorch_state_dict_to_flax_params), [`BATCH_SIZE`](conversion_test.md#BATCH_SIZE), [`SEQ_LEN`](conversion_test.md#SEQ_LEN), [`VOCAB_SIZE`](conversion_test.md#VOCAB_SIZE), [`LRU_WIDTH`](conversion_test.md#LRU_WIDTH), [`MLP_EXPANDED_WIDTH`](conversion_test.md#MLP_EXPANDED_WIDTH), [`NUM_HEADS`](conversion_test.md#NUM_HEADS), [`WIDTH`](conversion_test.md#WIDTH)

## Module values
- `BATCH_SIZE` — [`L25`](../../../../../raw/code/recurrentgemma/recurrentgemma/conversion_test.py#L25)
- `BLOCK_TYPES` — [`L33`](../../../../../raw/code/recurrentgemma/recurrentgemma/conversion_test.py#L33)
- `LRU_WIDTH` — [`L31`](../../../../../raw/code/recurrentgemma/recurrentgemma/conversion_test.py#L31)
- `MLP_EXPANDED_WIDTH` — [`L29`](../../../../../raw/code/recurrentgemma/recurrentgemma/conversion_test.py#L29)
- `NUM_HEADS` — [`L30`](../../../../../raw/code/recurrentgemma/recurrentgemma/conversion_test.py#L30)
- `NUM_SAMPLING_STEPS` — [`L32`](../../../../../raw/code/recurrentgemma/recurrentgemma/conversion_test.py#L32)
- `SEQ_LEN` — [`L26`](../../../../../raw/code/recurrentgemma/recurrentgemma/conversion_test.py#L26)
- `VOCAB_SIZE` — [`L27`](../../../../../raw/code/recurrentgemma/recurrentgemma/conversion_test.py#L27)
- `WIDTH` — [`L28`](../../../../../raw/code/recurrentgemma/recurrentgemma/conversion_test.py#L28)

