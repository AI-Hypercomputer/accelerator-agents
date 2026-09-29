---
title: 'Module: simply/rl_lib_test.py'
type: catalog
provenance: extracted
module: simply/rl_lib_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.rl_lib_test`/
symbols:
  RunExperimentTest.test_run_experiment_saves_checkpoint: RunExperimentTest#test_run_experiment_saves_checkpoint().
  RunExperimentTest._mock_vocab: RunExperimentTest#_mock_vocab.
  MockDeepScaleRSource._examples: MockDeepScaleRSource#_examples().
  _MOCK_VOCAB_NAME: _MOCK_VOCAB_NAME.
  RunExperimentTest.tearDown: RunExperimentTest#tearDown().
  MockDeepScaleRJSONExample.id: MockDeepScaleRJSONExample#id.
  MockDeepScaleRSource.__getitem__: MockDeepScaleRSource#__getitem__().
  RewardNormalizerTest.test_global: RewardNormalizerTest#test_global().
  RewardNormalizerTest.test_by_group: RewardNormalizerTest#test_by_group().
  MockDeepScaleRJSONExample: MockDeepScaleRJSONExample#
  MockSimplyV1Chat: MockSimplyV1Chat#
  MockDeepScaleRSource.__len__: MockDeepScaleRSource#__len__().
  UtilityFunctionsTest.test_np_safe_mean: UtilityFunctionsTest#test_np_safe_mean().
  UtilityFunctionsTest.test_np_safe_std: UtilityFunctionsTest#test_np_safe_std().
  ByteVocab: ByteVocab#
  ByteVocab.vocab_size: ByteVocab#vocab_size.
  MockDeepScaleRSource.start_index: MockDeepScaleRSource#start_index.
  MockDeepScaleRSource.end_index: MockDeepScaleRSource#end_index.
  ByteVocab.__init__: ByteVocab#__init__().
  ByteVocab.bos_id: ByteVocab#bos_id.
  ByteVocab.eos_id: ByteVocab#eos_id.
  ByteVocab.pad_id: ByteVocab#pad_id.
  ByteVocab.unk_id: ByteVocab#unk_id.
  ByteVocab.encode: ByteVocab#encode().
  ByteVocab.decode: ByteVocab#decode().
  MockSimplyV1Chat.extra_eos_tokens: MockSimplyV1Chat#extra_eos_tokens.
  MockDeepScaleRJSONExample.question: MockDeepScaleRJSONExample#question.
  MockDeepScaleRJSONExample.short_answer: MockDeepScaleRJSONExample#short_answer.
  MockDeepScaleRJSONExample.answer: MockDeepScaleRJSONExample#answer.
  MockDeepScaleRJSONExample.uid: MockDeepScaleRJSONExample#uid.
  MockDeepScaleRSource: MockDeepScaleRSource#
  UtilityFunctionsTest: UtilityFunctionsTest#
  RewardNormalizerTest: RewardNormalizerTest#
  RunExperimentTest: RunExperimentTest#
  RunExperimentTest.setUp: RunExperimentTest#setUp().
---
# Module: [`simply/rl_lib_test.py`](../../../../../raw/code/simply/simply/rl_lib_test.py)

## Classes
### `ByteVocab`
- def: [`simply/rl_lib_test.py:32`](../../../../../raw/code/simply/simply/rl_lib_test.py#L32)
- doc: Simple byte-level vocabulary for testing.
- signature: `class ByteVocab:`
- members:
  - `decode(self, token_ids: list[int])` — [`L46`](../../../../../raw/code/simply/simply/rl_lib_test.py#L46)
  - `encode(self, text: str)` — [`L42`](../../../../../raw/code/simply/simply/rl_lib_test.py#L42)
  - `bos_id` — [`L36`](../../../../../raw/code/simply/simply/rl_lib_test.py#L36)
  - `eos_id` — [`L37`](../../../../../raw/code/simply/simply/rl_lib_test.py#L37)
  - `pad_id` — [`L38`](../../../../../raw/code/simply/simply/rl_lib_test.py#L38)
  - `unk_id` — [`L39`](../../../../../raw/code/simply/simply/rl_lib_test.py#L39)
  - `vocab_size` — [`L40`](../../../../../raw/code/simply/simply/rl_lib_test.py#L40)
- protocol/private: `__init__`[`L35`](../../../../../raw/code/simply/simply/rl_lib_test.py#L35)
- used by: [`test_run_experiment_saves_checkpoint`](rl_lib_test.md#RunExperimentTest.test_run_experiment_saves_checkpoint), [`_mock_vocab`](rl_lib_test.md#RunExperimentTest._mock_vocab)

### `MockDeepScaleRJSONExample`  ·  implements/extends _TypedDict
- def: [`simply/rl_lib_test.py:63`](../../../../../raw/code/simply/simply/rl_lib_test.py#L63)
- doc: Type definition for a single example in MockDeepScaleRSource.
- signature: `class MockDeepScaleRJSONExample(TypedDict):`
- members:
  - `answer` — [`L68`](../../../../../raw/code/simply/simply/rl_lib_test.py#L68)
  - `id` — [`L70`](../../../../../raw/code/simply/simply/rl_lib_test.py#L70)
  - `question` — [`L66`](../../../../../raw/code/simply/simply/rl_lib_test.py#L66)
  - `short_answer` — [`L67`](../../../../../raw/code/simply/simply/rl_lib_test.py#L67)
  - `uid` — [`L69`](../../../../../raw/code/simply/simply/rl_lib_test.py#L69)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`DataSourceRegistry`](data_lib.md#DataSourceRegistry)
- used by: [`_examples`](rl_lib_test.md#MockDeepScaleRSource._examples), [`__getitem__`](rl_lib_test.md#MockDeepScaleRSource.__getitem__)

### `MockDeepScaleRSource`
- def: [`simply/rl_lib_test.py:77`](../../../../../raw/code/simply/simply/rl_lib_test.py#L77)
- doc: Mock DeepScaleRSource with 100 examples.
- signature: `class MockDeepScaleRSource:`
- members:
  - `end_index` — [`L81`](../../../../../raw/code/simply/simply/rl_lib_test.py#L81)
  - `start_index` — [`L80`](../../../../../raw/code/simply/simply/rl_lib_test.py#L80)
- protocol/private: `__getitem__`[`L100`](../../../../../raw/code/simply/simply/rl_lib_test.py#L100), `__len__`[`L97`](../../../../../raw/code/simply/simply/rl_lib_test.py#L97), `_examples`[`L84`](../../../../../raw/code/simply/simply/rl_lib_test.py#L84)
- uses (calls/refs, reference-scoped): [`MockDeepScaleRJSONExample`](rl_lib_test.md#MockDeepScaleRJSONExample)

### `MockSimplyV1Chat`
- def: [`simply/rl_lib_test.py:57`](../../../../../raw/code/simply/simply/rl_lib_test.py#L57)
- doc: Mock LM format for SimplyV1Chat, used in tests.
- signature: `class MockSimplyV1Chat(lm_format.SimplyV1Chat):`
- members:
  - `extra_eos_tokens` — [`L60`](../../../../../raw/code/simply/simply/rl_lib_test.py#L60)
- uses (calls/refs, reference-scoped): [`SimplyV1Chat`](utils/lm_format.md#SimplyV1Chat)

### `RewardNormalizerTest`  ·  implements/extends TestCase
- def: [`simply/rl_lib_test.py:165`](../../../../../raw/code/simply/simply/rl_lib_test.py#L165)
- signature: `class RewardNormalizerTest(absltest.TestCase):`
- members:
  - `test_by_group(self)` — [`L178`](../../../../../raw/code/simply/simply/rl_lib_test.py#L178)
  - `test_global(self)` — [`L167`](../../../../../raw/code/simply/simply/rl_lib_test.py#L167)
- uses (calls/refs, reference-scoped): [`ByGroup`](rl_lib.md#RewardNormalizer.ByGroup), [`Global`](rl_lib.md#RewardNormalizer.Global), [`RewardNormalizer`](rl_lib.md#RewardNormalizer)

### `RunExperimentTest`  ·  implements/extends TestCase
- def: [`simply/rl_lib_test.py:195`](../../../../../raw/code/simply/simply/rl_lib_test.py#L195)
- doc: Tests for RL loop with `lm_rl_test` config.
- signature: `class RunExperimentTest(absltest.TestCase):`
- members:
  - `setUp(self)` — [`L198`](../../../../../raw/code/simply/simply/rl_lib_test.py#L198)
  - `tearDown(self)` — [`L205`](../../../../../raw/code/simply/simply/rl_lib_test.py#L205)
  - `test_run_experiment_saves_checkpoint(self)` — [`L209`](../../../../../raw/code/simply/simply/rl_lib_test.py#L209)
- protocol/private: `_mock_vocab`[`L200`](../../../../../raw/code/simply/simply/rl_lib_test.py#L200)
- uses (calls/refs, reference-scoped): [`run_experiment`](rl_lib.md#run_experiment), [`DatasetConfig`](data_lib.md#DatasetConfig), [`lm_rl_test`](config_lib.md#lm_rl_test), [`TokenizerRegistry`](utils/tokenization.md#TokenizerRegistry), [`register_value`](utils/registry.md#RootRegistry.register_value), [`_MOCK_VOCAB_NAME`](rl_lib_test.md#_MOCK_VOCAB_NAME), [`unregister`](utils/registry.md#RootRegistry.unregister), [`ByteVocab`](rl_lib_test.md#ByteVocab), [`vocab_size`](rl_lib_test.md#ByteVocab.vocab_size)

### `UtilityFunctionsTest`  ·  implements/extends TestCase
- def: [`simply/rl_lib_test.py:104`](../../../../../raw/code/simply/simply/rl_lib_test.py#L104)
- signature: `class UtilityFunctionsTest(parameterized.TestCase):`
- members:
  - `test_np_safe_mean(self, x: np.ndarray, where: np.ndarray, expected: float)` — [`L126`](../../../../../raw/code/simply/simply/rl_lib_test.py#L126)
  - `test_np_safe_std(self, x: np.ndarray, where: np.ndarray, expected: float)` — [`L158`](../../../../../raw/code/simply/simply/rl_lib_test.py#L158)
- uses (calls/refs, reference-scoped): [`np_safe_mean`](rl_lib.md#np_safe_mean), [`np_safe_std`](rl_lib.md#np_safe_std)

## Module values
- `_MOCK_VOCAB_NAME` — [`L52`](../../../../../raw/code/simply/simply/rl_lib_test.py#L52)

