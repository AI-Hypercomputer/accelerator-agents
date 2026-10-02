---
title: 'Module: recurrentgemma/torch/sampler_test.py'
type: catalog
provenance: extracted
module: recurrentgemma/torch/sampler_test.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.torch.sampler_test`/
symbols:
  SamplerTest.test_forward_equivalence: SamplerTest#test_forward_equivalence().
  SamplerTest.test_output_shapes: SamplerTest#test_output_shapes().
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
# Module: [`recurrentgemma/torch/sampler_test.py`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler_test.py)

## Classes
### `MockVocab`
- def: [`recurrentgemma/torch/sampler_test.py:26`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler_test.py#L26)
- signature: `class MockVocab:`
- members:
  - `DecodeIds(self, ids: Iterable[int])` — [`L59`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler_test.py#L59)
  - `EncodeAsIds(self, text: str)` — [`L63`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler_test.py#L63)
  - `GetPieceSize(self)` — [`L56`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler_test.py#L56)
  - `bos_id(self)` — [`L50`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler_test.py#L50)
  - `eos_id(self)` — [`L53`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler_test.py#L53)
  - `pad_id(self)` — [`L47`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler_test.py#L47)
- protocol/private: `__init__`[`L28`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler_test.py#L28), `_mapping_text_to_id`[`L30`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler_test.py#L30), `_separator`[`L45`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler_test.py#L45), `_start_id`[`L29`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler_test.py#L29), `_vocab_size`[`L44`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler_test.py#L44)
- used by: [`test_forward_equivalence`](sampler_test.md#SamplerTest.test_forward_equivalence), [`test_output_shapes`](sampler_test.md#SamplerTest.test_output_shapes), [`test_samples`](sampler_test.md#SamplerTest.test_samples)

### `SamplerTest`  ·  implements/extends TestCase
- def: [`recurrentgemma/torch/sampler_test.py:68`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler_test.py#L68)
- signature: `class SamplerTest(parameterized.TestCase):`
- members:
  - `test_forward_equivalence(self, dtype: torch.dtype)` — [`L151`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler_test.py#L151)
  - `test_output_shapes(self, echo: bool, return_logits: bool)` — [`L98`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler_test.py#L98)
  - `test_samples(self)` — [`L70`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler_test.py#L70)
- uses (calls/refs, reference-scoped): [`TemporalBlockType`](../common.md#TemporalBlockType), [`RECURRENT`](../common.md#TemporalBlockType.RECURRENT), [`ATTENTION`](../common.md#TemporalBlockType.ATTENTION), [`vocab_size`](../common.md#GriffinConfig.vocab_size), [`width`](../common.md#GriffinConfig.width), [`block_types`](../common.md#GriffinConfig.block_types), [`attention_window_size`](../common.md#GriffinConfig.attention_window_size), [`GriffinConfig`](../common.md#GriffinConfig), [`lru_width`](../common.md#GriffinConfig.lru_width), [`num_heads`](../common.md#GriffinConfig.num_heads), [`embeddings_scale_by_sqrt_dim`](../common.md#GriffinConfig.embeddings_scale_by_sqrt_dim), [`logits_soft_cap`](../common.md#GriffinConfig.logits_soft_cap), [`mlp_expanded_width`](../common.md#GriffinConfig.mlp_expanded_width), [`forward`](griffin.md#Griffin.forward), [`Griffin`](__init__.md#Griffin), [`EncodeAsIds`](sampler_test.md#MockVocab.EncodeAsIds), [`GetPieceSize`](sampler_test.md#MockVocab.GetPieceSize), [`Sampler`](__init__.md#Sampler), [`MockVocab`](sampler_test.md#MockVocab), [`bos_id`](sampler_test.md#MockVocab.bos_id)

