---
title: 'Module: recurrentgemma/jax/sampler_test.py'
type: catalog
provenance: extracted
module: recurrentgemma/jax/sampler_test.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.jax.sampler_test`/
symbols:
  SamplerTest.test_output_shapes: SamplerTest#test_output_shapes().
  SamplerTest.test_forward_equivalence: SamplerTest#test_forward_equivalence().
  SamplerTest.test_samples: SamplerTest#test_samples().
  MockVocab.EncodeAsIds: MockVocab#EncodeAsIds().
  MockVocab.GetPieceSize: MockVocab#GetPieceSize().
  MockVocab.DecodeIds: MockVocab#DecodeIds().
  MockVocab._vocab_size: MockVocab#_vocab_size.
  MockVocab: MockVocab#
  MockVocab._mapping_text_to_id: MockVocab#_mapping_text_to_id.
  MockVocab._separator: MockVocab#_separator.
  MockVocab.bos_id: MockVocab#bos_id().
  MockVocab.__init__: MockVocab#__init__().
  MockVocab._start_id: MockVocab#_start_id.
  MockVocab.pad_id: MockVocab#pad_id().
  MockVocab.eos_id: MockVocab#eos_id().
  SamplerTest: SamplerTest#
---
# Module: [`recurrentgemma/jax/sampler_test.py`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler_test.py)

## Classes
### `MockVocab`
- def: [`recurrentgemma/jax/sampler_test.py:28`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler_test.py#L28)
- signature: `class MockVocab:`
- members:
  - `DecodeIds(self, ids: Iterable[int])` — [`L61`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler_test.py#L61)
  - `EncodeAsIds(self, text: str)` — [`L65`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler_test.py#L65)
  - `GetPieceSize(self)` — [`L58`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler_test.py#L58)
  - `bos_id(self)` — [`L52`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler_test.py#L52)
  - `eos_id(self)` — [`L55`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler_test.py#L55)
  - `pad_id(self)` — [`L49`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler_test.py#L49)
- protocol/private: `__init__`[`L30`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler_test.py#L30), `_mapping_text_to_id`[`L32`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler_test.py#L32), `_separator`[`L47`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler_test.py#L47), `_start_id`[`L31`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler_test.py#L31), `_vocab_size`[`L46`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler_test.py#L46)
- used by: [`test_forward_equivalence`](sampler_test.md#SamplerTest.test_forward_equivalence), [`test_output_shapes`](sampler_test.md#SamplerTest.test_output_shapes), [`test_samples`](sampler_test.md#SamplerTest.test_samples)

### `SamplerTest`  ·  implements/extends TestCase
- def: [`recurrentgemma/jax/sampler_test.py:70`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler_test.py#L70)
- signature: `class SamplerTest(parameterized.TestCase):`
- members:
  - `test_forward_equivalence(self, dtype: str)` — [`L168`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler_test.py#L168)
  - `test_output_shapes(self, echo: bool, return_logits: bool)` — [`L106`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler_test.py#L106)
  - `test_samples(self)` — [`L72`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler_test.py#L72)
- uses (calls/refs, reference-scoped): [`TemporalBlockType`](../common.md#TemporalBlockType), [`RECURRENT`](../common.md#TemporalBlockType.RECURRENT), [`ATTENTION`](../common.md#TemporalBlockType.ATTENTION), [`vocab_size`](../common.md#GriffinConfig.vocab_size), [`width`](../common.md#GriffinConfig.width), [`block_types`](../common.md#GriffinConfig.block_types), [`attention_window_size`](../common.md#GriffinConfig.attention_window_size), [`GriffinConfig`](../common.md#GriffinConfig), [`lru_width`](../common.md#GriffinConfig.lru_width), [`num_heads`](../common.md#GriffinConfig.num_heads), [`embeddings_scale_by_sqrt_dim`](../common.md#GriffinConfig.embeddings_scale_by_sqrt_dim), [`logits_soft_cap`](../common.md#GriffinConfig.logits_soft_cap), [`mlp_expanded_width`](../common.md#GriffinConfig.mlp_expanded_width), [`Griffin`](__init__.md#Griffin), [`EncodeAsIds`](sampler_test.md#MockVocab.EncodeAsIds), [`GetPieceSize`](sampler_test.md#MockVocab.GetPieceSize), [`Sampler`](__init__.md#Sampler), [`logits`](sampler.md#SamplerOutput.logits), [`MockVocab`](sampler_test.md#MockVocab), [`tokens`](sampler.md#SamplerOutput.tokens), [`bos_id`](sampler_test.md#MockVocab.bos_id)

