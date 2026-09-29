---
title: 'Module: simply/data_lib_test.py'
type: catalog
provenance: extracted
module: simply/data_lib_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.data_lib_test`/
symbols:
  MixtureDatasetTest.test_pack_before_mix: MixtureDatasetTest#test_pack_before_mix().
  TruncateAndPadTransformTest.test_composed_truncate_then_pad: TruncateAndPadTransformTest#test_composed_truncate_then_pad().
  ChatFormatTransformTest.test_trainable_roles_creates_loss_mask: ChatFormatTransformTest#test_trainable_roles_creates_loss_mask().
  CreateIterDatasetTest.test_validation_mode_produces_finite_batches: CreateIterDatasetTest#test_validation_mode_produces_finite_batches().
  CreateIterDatasetTest.test_batch_contents_correct: CreateIterDatasetTest#test_batch_contents_correct().
  MixtureDatasetTest.setUpClass: MixtureDatasetTest#setUpClass().
  MixtureDatasetTest.test_mixture_produces_batches: MixtureDatasetTest#test_mixture_produces_batches().
  MockTokenizer.decode: MockTokenizer#decode().
  TokenizeTransformTest.test_basic_tokenization_with_eos: TokenizeTransformTest#test_basic_tokenization_with_eos().
  TokenizeTransformTest.test_tokenization_with_bos: TokenizeTransformTest#test_tokenization_with_bos().
  CreateIterDatasetTest.setUpClass: CreateIterDatasetTest#setUpClass().
  CreateIterDatasetTest.test_produces_correct_batch_shape: CreateIterDatasetTest#test_produces_correct_batch_shape().
  TokenizeTransformTest.test_bytes_input_decoded: TokenizeTransformTest#test_bytes_input_decoded().
  TruncateAndPadTransformTest.test_pad_short_sequences: TruncateAndPadTransformTest#test_pad_short_sequences().
  TruncateAndPadTransformTest.test_pad_does_not_truncate: TruncateAndPadTransformTest#test_pad_does_not_truncate().
  MixtureConfigValidationTest.test_requires_positive_weights: MixtureConfigValidationTest#test_requires_positive_weights().
  TruncateAndPadTransformTest.test_truncate_long_sequences: TruncateAndPadTransformTest#test_truncate_long_sequences().
  RekeyTransformTest.test_rekeys_features: RekeyTransformTest#test_rekeys_features().
  RekeyTransformTest.test_none_key_map: RekeyTransformTest#test_none_key_map().
  NumpyTransformTest.test_type_cast_map: NumpyTransformTest#test_type_cast_map().
  NumpyTransformTest.test_no_type_cast_map: NumpyTransformTest#test_no_type_cast_map().
  GetDataSourceTest.test_unknown_source_raises_error: GetDataSourceTest#test_unknown_source_raises_error().
  MockExperimentConfig: MockExperimentConfig#
  MockExperimentConfig.dataset: MockExperimentConfig#dataset.
  MockTokenizer._id_to_special: MockTokenizer#_id_to_special.
  MockTokenizer.encode: MockTokenizer#encode().
  MixtureConfigValidationTest.test_requires_datasets: MixtureConfigValidationTest#test_requires_datasets().
  NextTokenPredTransformTest.test_shifts_tokens_correctly: NextTokenPredTransformTest#test_shifts_tokens_correctly().
  NextTokenPredTransformTest.test_loss_mask_shifted_correctly: NextTokenPredTransformTest#test_loss_mask_shifted_correctly().
  NumpyTransformTest.test_converts_to_numpy: NumpyTransformTest#test_converts_to_numpy().
  MockTokenizer._special_tokens: MockTokenizer#_special_tokens.
  MockExperimentConfig.validation_dataset: MockExperimentConfig#validation_dataset.
  MockTokenizer: MockTokenizer#
  MockTokenizer.bos_id: MockTokenizer#bos_id.
  MockTokenizer.eos_id: MockTokenizer#eos_id.
  MockTokenizer.pad_id: MockTokenizer#pad_id.
  MockExperimentConfig.seq_len: MockExperimentConfig#seq_len.
  CreateIterDatasetTest.setUpClass.TestSource: CreateIterDatasetTest#setUpClass().TestSource#
  MixtureDatasetTest.setUpClass.MixSource1: MixtureDatasetTest#setUpClass().MixSource1#
  MixtureDatasetTest.setUpClass.MixSource2: MixtureDatasetTest#setUpClass().MixSource2#
  MixtureConfigValidationTest: MixtureConfigValidationTest#
  TokenizeTransformTest: TokenizeTransformTest#
  NextTokenPredTransformTest: NextTokenPredTransformTest#
  TruncateAndPadTransformTest: TruncateAndPadTransformTest#
  ChatFormatTransformTest: ChatFormatTransformTest#
  RekeyTransformTest: RekeyTransformTest#
  NumpyTransformTest: NumpyTransformTest#
  GetDataSourceTest: GetDataSourceTest#
  MockExperimentConfig.vocab_name: MockExperimentConfig#vocab_name.
  MockExperimentConfig.batch_size: MockExperimentConfig#batch_size.
  MockExperimentConfig.dataset_seed: MockExperimentConfig#dataset_seed.
  MockExperimentConfig.prefetch_num_workers: MockExperimentConfig#prefetch_num_workers.
  MockExperimentConfig.prefetch_per_worker_buffer_size: MockExperimentConfig#prefetch_per_worker_buffer_size.
  MockExperimentConfig.validation_eval_batch_size: MockExperimentConfig#validation_eval_batch_size.
  MockExperimentConfig.validation_eval_epochs: MockExperimentConfig#validation_eval_epochs.
  MockExperimentConfig.shard_data_method: MockExperimentConfig#shard_data_method.
  CreateIterDatasetTest: CreateIterDatasetTest#
  CreateIterDatasetTest.setUpClass.TestSource.__len__: CreateIterDatasetTest#setUpClass().TestSource#__len__().
  CreateIterDatasetTest.setUpClass.TestSource.__getitem__: CreateIterDatasetTest#setUpClass().TestSource#__getitem__().
  MixtureDatasetTest: MixtureDatasetTest#
  MixtureDatasetTest.setUpClass.MixSource1.__len__: MixtureDatasetTest#setUpClass().MixSource1#__len__().
  MixtureDatasetTest.setUpClass.MixSource1.__getitem__: MixtureDatasetTest#setUpClass().MixSource1#__getitem__().
  MixtureDatasetTest.setUpClass.MixSource2.__len__: MixtureDatasetTest#setUpClass().MixSource2#__len__().
  MixtureDatasetTest.setUpClass.MixSource2.__getitem__: MixtureDatasetTest#setUpClass().MixSource2#__getitem__().
---
# Module: [`simply/data_lib_test.py`](../../../../../raw/code/simply/simply/data_lib_test.py)

## Classes
### `ChatFormatTransformTest`  ·  implements/extends TestCase
- def: [`simply/data_lib_test.py:254`](../../../../../raw/code/simply/simply/data_lib_test.py#L254)
- doc: Tests for ChatFormatTransform.
- signature: `class ChatFormatTransformTest(absltest.TestCase):`
- members:
  - `test_trainable_roles_creates_loss_mask(self)` — [`L257`](../../../../../raw/code/simply/simply/data_lib_test.py#L257) — Test that trainable_roles masks exactly the correct tokens.
- uses (calls/refs, reference-scoped): [`add_bos`](data_lib.md#ChatFormatTransform.add_bos), [`lm_format_name`](data_lib.md#ChatFormatTransform.lm_format_name), [`tokenizer_name`](data_lib.md#ChatFormatTransform.tokenizer_name), [`trainable_roles`](data_lib.md#ChatFormatTransform.trainable_roles), [`ChatFormatTransform`](data_lib.md#ChatFormatTransform)

### `CreateIterDatasetTest`  ·  implements/extends TestCase
- def: [`simply/data_lib_test.py:423`](../../../../../raw/code/simply/simply/data_lib_test.py#L423)
- doc: Tests for create_iter_dataset function.
- signature: `class CreateIterDatasetTest(absltest.TestCase):`
- members:
  - `setUpClass(cls)` — [`L427`](../../../../../raw/code/simply/simply/data_lib_test.py#L427) — Register test sources.
  - `test_batch_contents_correct(self)` — [`L467`](../../../../../raw/code/simply/simply/data_lib_test.py#L467) — Test that batch contents are tokenized correctly.
  - `test_produces_correct_batch_shape(self)` — [`L444`](../../../../../raw/code/simply/simply/data_lib_test.py#L444) — Test that output batches have correct shape.
  - `test_validation_mode_produces_finite_batches(self)` — [`L455`](../../../../../raw/code/simply/simply/data_lib_test.py#L455) — Test validation mode produces finite number of batches.
- uses (calls/refs, reference-scoped): [`create_iter_dataset`](data_lib.md#create_iter_dataset), [`DatasetConfig`](data_lib.md#DatasetConfig), [`DataSourceRegistry`](data_lib.md#DataSourceRegistry), [`keys`](utils/registry.md#RootRegistry.keys), [`register_value`](utils/registry.md#RootRegistry.register_value), [`MockExperimentConfig`](data_lib_test.md#MockExperimentConfig), [`dataset`](data_lib_test.md#MockExperimentConfig.dataset), [`validation_dataset`](data_lib_test.md#MockExperimentConfig.validation_dataset), [`TestSource`](data_lib_test.md#CreateIterDatasetTest.setUpClass.TestSource)

### `GetDataSourceTest`  ·  implements/extends TestCase
- def: [`simply/data_lib_test.py:392`](../../../../../raw/code/simply/simply/data_lib_test.py#L392)
- doc: Tests for get_data_source function.
- signature: `class GetDataSourceTest(absltest.TestCase):`
- members:
  - `test_unknown_source_raises_error(self)` — [`L395`](../../../../../raw/code/simply/simply/data_lib_test.py#L395) — Test that getting unknown source raises error.
- uses (calls/refs, reference-scoped): [`DatasetConfig`](data_lib.md#DatasetConfig), [`get_data_source`](data_lib.md#get_data_source)

### `MixSource1`
- def: [`simply/data_lib_test.py:501`](../../../../../raw/code/simply/simply/data_lib_test.py#L501)
- signature: `class MixSource1:`
- protocol/private: `__getitem__`[`L505`](../../../../../raw/code/simply/simply/data_lib_test.py#L505), `__len__`[`L502`](../../../../../raw/code/simply/simply/data_lib_test.py#L502)
- used by: [`setUpClass`](data_lib_test.md#MixtureDatasetTest.setUpClass)

### `MixSource2`
- def: [`simply/data_lib_test.py:509`](../../../../../raw/code/simply/simply/data_lib_test.py#L509)
- signature: `class MixSource2:`
- protocol/private: `__getitem__`[`L513`](../../../../../raw/code/simply/simply/data_lib_test.py#L513), `__len__`[`L510`](../../../../../raw/code/simply/simply/data_lib_test.py#L510)
- used by: [`setUpClass`](data_lib_test.md#MixtureDatasetTest.setUpClass)

### `MixtureConfigValidationTest`  ·  implements/extends TestCase
- def: [`simply/data_lib_test.py:104`](../../../../../raw/code/simply/simply/data_lib_test.py#L104)
- doc: Tests for MixtureConfig validation logic.
- signature: `class MixtureConfigValidationTest(absltest.TestCase):`
- members:
  - `test_requires_datasets(self)` — [`L107`](../../../../../raw/code/simply/simply/data_lib_test.py#L107) — Test that MixtureConfig requires at least one dataset.
  - `test_requires_positive_weights(self)` — [`L112`](../../../../../raw/code/simply/simply/data_lib_test.py#L112) — Test that MixtureConfig requires positive weights.
- uses (calls/refs, reference-scoped): [`DatasetConfig`](data_lib.md#DatasetConfig), [`MixtureConfig`](data_lib.md#MixtureConfig)

### `MixtureDatasetTest`  ·  implements/extends TestCase
- def: [`simply/data_lib_test.py:492`](../../../../../raw/code/simply/simply/data_lib_test.py#L492)
- doc: Tests for mixture datasets.
- signature: `class MixtureDatasetTest(absltest.TestCase):`
- members:
  - `setUpClass(cls)` — [`L496`](../../../../../raw/code/simply/simply/data_lib_test.py#L496) — Register test sources.
  - `test_mixture_produces_batches(self)` — [`L525`](../../../../../raw/code/simply/simply/data_lib_test.py#L525) — Test that mixture dataset produces valid batches.
  - `test_pack_before_mix(self)` — [`L539`](../../../../../raw/code/simply/simply/data_lib_test.py#L539) — Test pack_before_mix option works.
- uses (calls/refs, reference-scoped): [`create_iter_dataset`](data_lib.md#create_iter_dataset), [`DatasetConfig`](data_lib.md#DatasetConfig), [`DataSourceRegistry`](data_lib.md#DataSourceRegistry), [`keys`](utils/registry.md#RootRegistry.keys), [`register_value`](utils/registry.md#RootRegistry.register_value), [`MixtureConfig`](data_lib.md#MixtureConfig), [`MockExperimentConfig`](data_lib_test.md#MockExperimentConfig), [`dataset`](data_lib_test.md#MockExperimentConfig.dataset), [`MixSource1`](data_lib_test.md#MixtureDatasetTest.setUpClass.MixSource1), [`MixSource2`](data_lib_test.md#MixtureDatasetTest.setUpClass.MixSource2), [`seq_len`](data_lib_test.md#MockExperimentConfig.seq_len)

### `MockExperimentConfig`
- def: [`simply/data_lib_test.py:408`](../../../../../raw/code/simply/simply/data_lib_test.py#L408)
- doc: Mock experiment config for testing.
- signature: `class MockExperimentConfig:`
- members:
  - `batch_size` — [`L414`](../../../../../raw/code/simply/simply/data_lib_test.py#L414)
  - `dataset` — [`L410`](../../../../../raw/code/simply/simply/data_lib_test.py#L410)
  - `dataset_seed` — [`L415`](../../../../../raw/code/simply/simply/data_lib_test.py#L415)
  - `prefetch_num_workers` — [`L416`](../../../../../raw/code/simply/simply/data_lib_test.py#L416)
  - `prefetch_per_worker_buffer_size` — [`L417`](../../../../../raw/code/simply/simply/data_lib_test.py#L417)
  - `seq_len` — [`L413`](../../../../../raw/code/simply/simply/data_lib_test.py#L413)
  - `shard_data_method` — [`L420`](../../../../../raw/code/simply/simply/data_lib_test.py#L420)
  - `validation_dataset` — [`L411`](../../../../../raw/code/simply/simply/data_lib_test.py#L411)
  - `validation_eval_batch_size` — [`L418`](../../../../../raw/code/simply/simply/data_lib_test.py#L418)
  - `validation_eval_epochs` — [`L419`](../../../../../raw/code/simply/simply/data_lib_test.py#L419)
  - `vocab_name` — [`L412`](../../../../../raw/code/simply/simply/data_lib_test.py#L412)
- used by: [`test_pack_before_mix`](data_lib_test.md#MixtureDatasetTest.test_pack_before_mix), [`test_batch_contents_correct`](data_lib_test.md#CreateIterDatasetTest.test_batch_contents_correct), [`test_mixture_produces_batches`](data_lib_test.md#MixtureDatasetTest.test_mixture_produces_batches), [`test_validation_mode_produces_finite_batches`](data_lib_test.md#CreateIterDatasetTest.test_validation_mode_produces_finite_batches), [`test_produces_correct_batch_shape`](data_lib_test.md#CreateIterDatasetTest.test_produces_correct_batch_shape)

### `MockTokenizer`
- def: [`simply/data_lib_test.py:28`](../../../../../raw/code/simply/simply/data_lib_test.py#L28)
- doc: Byte-based mock tokenizer with special token support.
- signature: `class MockTokenizer:`
- members:
  - `decode(self, ids)` — [`L68`](../../../../../raw/code/simply/simply/data_lib_test.py#L68)
  - `encode(self, text)` — [`L49`](../../../../../raw/code/simply/simply/data_lib_test.py#L49)
  - `bos_id` — [`L41`](../../../../../raw/code/simply/simply/data_lib_test.py#L41)
  - `eos_id` — [`L42`](../../../../../raw/code/simply/simply/data_lib_test.py#L42)
  - `pad_id` — [`L43`](../../../../../raw/code/simply/simply/data_lib_test.py#L43)
- protocol/private: `_id_to_special`[`L47`](../../../../../raw/code/simply/simply/data_lib_test.py#L47), `_special_tokens`[`L46`](../../../../../raw/code/simply/simply/data_lib_test.py#L46)

### `NextTokenPredTransformTest`  ·  implements/extends TestCase
- def: [`simply/data_lib_test.py:167`](../../../../../raw/code/simply/simply/data_lib_test.py#L167)
- doc: Tests for NextTokenPredTransform.
- signature: `class NextTokenPredTransformTest(absltest.TestCase):`
- members:
  - `test_loss_mask_shifted_correctly(self)` — [`L183`](../../../../../raw/code/simply/simply/data_lib_test.py#L183) — Test that loss mask is shifted to align with targets.
  - `test_shifts_tokens_correctly(self)` — [`L170`](../../../../../raw/code/simply/simply/data_lib_test.py#L170) — Test next-token pred shifts input/target correctly.
- uses (calls/refs, reference-scoped): [`NextTokenPredTransform`](data_lib.md#NextTokenPredTransform)

### `NumpyTransformTest`  ·  implements/extends TestCase
- def: [`simply/data_lib_test.py:352`](../../../../../raw/code/simply/simply/data_lib_test.py#L352)
- doc: Tests for NumpyTransform.
- signature: `class NumpyTransformTest(absltest.TestCase):`
- members:
  - `test_converts_to_numpy(self)` — [`L355`](../../../../../raw/code/simply/simply/data_lib_test.py#L355)
  - `test_no_type_cast_map(self)` — [`L380`](../../../../../raw/code/simply/simply/data_lib_test.py#L380)
  - `test_type_cast_map(self)` — [`L363`](../../../../../raw/code/simply/simply/data_lib_test.py#L363)
- uses (calls/refs, reference-scoped): [`type_cast_map`](data_lib.md#NumpyTransform.type_cast_map), [`NumpyTransform`](data_lib.md#NumpyTransform)

### `RekeyTransformTest`  ·  implements/extends TestCase
- def: [`simply/data_lib_test.py:336`](../../../../../raw/code/simply/simply/data_lib_test.py#L336)
- doc: Tests for RekeyTransform.
- signature: `class RekeyTransformTest(absltest.TestCase):`
- members:
  - `test_none_key_map(self)` — [`L345`](../../../../../raw/code/simply/simply/data_lib_test.py#L345)
  - `test_rekeys_features(self)` — [`L339`](../../../../../raw/code/simply/simply/data_lib_test.py#L339)
- uses (calls/refs, reference-scoped): [`key_map`](data_lib.md#RekeyTransform.key_map), [`RekeyTransform`](data_lib.md#RekeyTransform)

### `TestSource`
- def: [`simply/data_lib_test.py:432`](../../../../../raw/code/simply/simply/data_lib_test.py#L432)
- signature: `class TestSource:`
- protocol/private: `__getitem__`[`L436`](../../../../../raw/code/simply/simply/data_lib_test.py#L436), `__len__`[`L433`](../../../../../raw/code/simply/simply/data_lib_test.py#L433)
- used by: [`setUpClass`](data_lib_test.md#CreateIterDatasetTest.setUpClass)

### `TokenizeTransformTest`  ·  implements/extends TestCase
- def: [`simply/data_lib_test.py:125`](../../../../../raw/code/simply/simply/data_lib_test.py#L125)
- doc: Tests for TokenizeTransform.
- signature: `class TokenizeTransformTest(absltest.TestCase):`
- members:
  - `test_basic_tokenization_with_eos(self)` — [`L128`](../../../../../raw/code/simply/simply/data_lib_test.py#L128) — Test tokenization adds EOS token.
  - `test_bytes_input_decoded(self)` — [`L154`](../../../../../raw/code/simply/simply/data_lib_test.py#L154) — Test that bytes input is decoded to string.
  - `test_tokenization_with_bos(self)` — [`L141`](../../../../../raw/code/simply/simply/data_lib_test.py#L141) — Test tokenization with BOS token.
- uses (calls/refs, reference-scoped): [`add_eos`](data_lib.md#TokenizeTransform.add_eos), [`tokenizer_name`](data_lib.md#TokenizeTransform.tokenizer_name), [`TokenizeTransform`](data_lib.md#TokenizeTransform), [`add_bos`](data_lib.md#TokenizeTransform.add_bos)

### `TruncateAndPadTransformTest`  ·  implements/extends TestCase
- def: [`simply/data_lib_test.py:196`](../../../../../raw/code/simply/simply/data_lib_test.py#L196)
- doc: Tests for TruncateTransform and PadTransform.
- signature: `class TruncateAndPadTransformTest(absltest.TestCase):`
- members:
  - `test_composed_truncate_then_pad(self)` — [`L239`](../../../../../raw/code/simply/simply/data_lib_test.py#L239) — Test composing TruncateTransform + PadTransform for fixed length.
  - `test_pad_does_not_truncate(self)` — [`L215`](../../../../../raw/code/simply/simply/data_lib_test.py#L215) — Test PadTransform does not truncate long sequences.
  - `test_pad_short_sequences(self)` — [`L199`](../../../../../raw/code/simply/simply/data_lib_test.py#L199) — Test PadTransform pads short sequences.
  - `test_truncate_long_sequences(self)` — [`L228`](../../../../../raw/code/simply/simply/data_lib_test.py#L228) — Test TruncateTransform truncates from the left (keeps end).
- uses (calls/refs, reference-scoped): [`pad_id`](data_lib.md#PadTransform.pad_id), [`seq_len`](data_lib.md#PadTransform.seq_len), [`PadTransform`](data_lib.md#PadTransform), [`TruncateTransform`](data_lib.md#TruncateTransform), [`seq_len`](data_lib.md#TruncateTransform.seq_len)

