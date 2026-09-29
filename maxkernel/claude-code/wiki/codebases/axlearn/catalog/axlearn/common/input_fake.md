---
title: 'Module: axlearn/common/input_fake.py'
type: catalog
provenance: extracted
module: axlearn/common/input_fake.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.input_fake`/
symbols:
  EmptyInput: EmptyInput#
  EmptyInput.__init__: EmptyInput#__init__().
  fake_source: fake_source().
  EmptyInput.Config: EmptyInput#Config#
  EmptyInput.batches: EmptyInput#batches().
  FakeSeq2SeqInput.Config: FakeSeq2SeqInput#Config#
  FakeSequenceClassificationInput.Config: FakeSequenceClassificationInput#Config#
  EmptyInput._num_batches: EmptyInput#_num_batches.
  EmptyInput.Config.is_training: EmptyInput#Config#is_training.
  FakeLmInput._prng_key: FakeLmInput#_prng_key.
  FakeSeq2SeqInput._prng_key: FakeSeq2SeqInput#_prng_key.
  FakeSequenceClassificationInput._prng_key: FakeSequenceClassificationInput#_prng_key.
  EmptyInput.Config.global_batch_size: EmptyInput#Config#global_batch_size.
  FakeLmInput.__next__: FakeLmInput#__next__().
  FakeSeq2SeqInput.__next__: FakeSeq2SeqInput#__next__().
  FakeSequenceClassificationInput.__next__: FakeSequenceClassificationInput#__next__().
  fake_text_source: fake_text_source().
  fake_serialized_json_source: fake_serialized_json_source().
  fake_text2text_source: fake_text2text_source().
  fake_glue_source: fake_glue_source().
  fake_classification_source: fake_classification_source().
  fake_classification_source_instruct_lm: fake_classification_source_instruct_lm().
  fake_speech_source: fake_speech_source().
  EmptyInput.__iter__: EmptyInput#__iter__().
  fake_source.fn: fake_source().fn().
  fake_serialized_json_source.ds_fn: fake_serialized_json_source().ds_fn().
  EmptyInput.dataset: EmptyInput#dataset().
  FakeLmInput: FakeLmInput#
  FakeSeq2SeqInput: FakeSeq2SeqInput#
  FakeSequenceClassificationInput: FakeSequenceClassificationInput#
  EmptyInput.Config.total_num_batches: EmptyInput#Config#total_num_batches.
  fake_source.data_gen: fake_source().data_gen().
  fake_serialized_json_source.ds_fn.data_gen: fake_serialized_json_source().ds_fn().data_gen().
  EmptyInput.Config.source_length: EmptyInput#Config#source_length.
  EmptyInput.Config.max_token_id: EmptyInput#Config#max_token_id.
  EmptyInput._prng_key: EmptyInput#_prng_key.
  EmptyInput.__next__: EmptyInput#__next__().
  FakeSeq2SeqInput.Config.target_length: FakeSeq2SeqInput#Config#target_length.
  FakeSequenceClassificationInput.Config.num_labels: FakeSequenceClassificationInput#Config#num_labels.
  fake_grain_source: fake_grain_source().
---
# Module: [`axlearn/common/input_fake.py`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py)

## Classes
### `Config`  ·  implements/extends Config
- def: [`axlearn/common/input_fake.py:138`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L138)
- doc: Configures FakeSequenceClassificationInput.
- signature: `class Config(EmptyInput.Config):`
- members:
  - `global_batch_size` — [`L30`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L30)
  - `is_training` — [`L29`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L29)
  - `max_token_id` — [`L33`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L33)
  - `num_labels` — [`L141`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L141)
  - `source_length` — [`L32`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L32)
  - `target_length` — [`L96`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L96)
  - `total_num_batches` — [`L31`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L31)
- uses (calls/refs, reference-scoped): [`config_class`](config.md#config_class), [`REQUIRED`](config.md#REQUIRED), [`Required`](config.md#Required), [`EmptyInput`](input_fake.md#EmptyInput), [`Config`](input_fake.md#EmptyInput.Config)
- used by: [`__init__`](input_fake.md#EmptyInput.__init__)

### `EmptyInput`  ·  implements/extends Input
- def: [`axlearn/common/input_fake.py:22`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L22)
- doc: Produces empty inputs.
- signature: `class EmptyInput(Input):`
- members:
  - `batches(self, it: tf.data.Iterator)` — [`L53`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L53)
  - `dataset(self)` — [`L50`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L50)
- protocol/private: `__init__`[`L35`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L35), `__iter__`[`L46`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L46), `__next__`[`L43`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L43), `_num_batches`[`L41`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L41), `_prng_key`[`L40`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L40)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`Input`](input_base.md#Input), [`__init__`](input_base.md#Input.__init__), [`Config`](input_fake.md#EmptyInput.Config), [`as_numpy_array`](utils.md#as_numpy_array), [`is_training`](input_fake.md#EmptyInput.Config.is_training), [`FakeLmInput`](input_fake.md#FakeLmInput), [`FakeSeq2SeqInput`](input_fake.md#FakeSeq2SeqInput), [`FakeSequenceClassificationInput`](input_fake.md#FakeSequenceClassificationInput), [`total_num_batches`](input_fake.md#EmptyInput.Config.total_num_batches)
- used by: [`target_to_source`](state_builder.md#BaseConverterFromPretrainedModel.target_to_source), [`Input`](input_base.md#Input), [`batches`](input_base.md#Input.batches), [`dataset`](input_base.md#Input.dataset), [`Config`](input_fake.md#FakeSeq2SeqInput.Config), [`Config`](input_fake.md#FakeSequenceClassificationInput.Config), [`__next__`](input_fake.md#FakeLmInput.__next__), [`__next__`](input_fake.md#FakeSeq2SeqInput.__next__), [`__next__`](input_fake.md#FakeSequenceClassificationInput.__next__), [`FakeLmInput`](input_fake.md#FakeLmInput), [`FakeSeq2SeqInput`](input_fake.md#FakeSeq2SeqInput), [`FakeSequenceClassificationInput`](input_fake.md#FakeSequenceClassificationInput)

### `FakeLmInput`  ·  implements/extends EmptyInput
- def: [`axlearn/common/input_fake.py:60`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L60)
- doc: Produces fake language modeling inputs.
- signature: `class FakeLmInput(EmptyInput):`
- protocol/private: `__next__`[`L63`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L63), `_prng_key`[`L68`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L68)
- uses (calls/refs, reference-scoped): [`config`](config.md#Configurable.config), [`EmptyInput`](input_fake.md#EmptyInput), [`as_tensor`](utils.md#as_tensor), [`_num_batches`](input_fake.md#EmptyInput._num_batches)
- used by: [`EmptyInput`](input_fake.md#EmptyInput)

### `FakeSeq2SeqInput`  ·  implements/extends EmptyInput
- def: [`axlearn/common/input_fake.py:91`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L91)
- doc: Produces fake sequence-to-sequence inputs.
- signature: `class FakeSeq2SeqInput(EmptyInput):`
- protocol/private: `__next__`[`L98`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L98), `_prng_key`[`L109`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L109)
- uses (calls/refs, reference-scoped): [`config`](config.md#Configurable.config), [`EmptyInput`](input_fake.md#EmptyInput), [`as_tensor`](utils.md#as_tensor), [`_num_batches`](input_fake.md#EmptyInput._num_batches)
- used by: [`EmptyInput`](input_fake.md#EmptyInput)

### `FakeSequenceClassificationInput`  ·  implements/extends EmptyInput
- def: [`axlearn/common/input_fake.py:134`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L134)
- doc: Produces fake sequence classification inputs.
- signature: `class FakeSequenceClassificationInput(EmptyInput):`
- protocol/private: `__next__`[`L143`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L143), `_prng_key`[`L148`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L148)
- uses (calls/refs, reference-scoped): [`config`](config.md#Configurable.config), [`EmptyInput`](input_fake.md#EmptyInput), [`as_tensor`](utils.md#as_tensor), [`_num_batches`](input_fake.md#EmptyInput._num_batches)
- used by: [`EmptyInput`](input_fake.md#EmptyInput)

## Functions
- `data_gen()` — [`L187`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L187)
- `data_gen()` — [`L241`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L241)
- `ds_fn()` — [`L240`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L240)
- `fake_classification_source(*, source_key: str = "source_text", target_key: str = "target_text", is_training: bool, classes: Sequence[str], shuffle_buffer_size: Optional[int] = None)` — [`L303`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L303)
- `fake_classification_source_instruct_lm(*, text_key: str = "text", is_training: bool, classes: Sequence[str] = ("yes", "no"), shuffle_buffer_size: Optional[int] = None, eoa_text: str = "<eoa>", eob_text: str = "<eob>")` — [`L326`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L326) — Returns a BuildDatasetFn containing fake classification examples in the InstructLM format.
- `fake_glue_source(*, input_key: Union[str, tuple[str, str]], label_key: str, is_training: bool, label_value: Union[int, Sequence[int]] = 0, num_examples: Optional[int] = None, shuffle_buffer_size: Optional[int] = None, spec: Optional[dict[str, tf.TypeSpec]] = None)` — [`L273`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L273)
- `fake_grain_source(examples: Sequence[Any], *, repeat: Optional[int] = 1, shuffle_seed: Optional[int] = None)` — [`L409`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L409) — Returns a fake grain input source.
- `fake_serialized_json_source(examples: Sequence[dict[str, Any]])` — [`L229`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L229) — Returns a BuildDatasetFn that returns a dataset of jsonlines of examples.
- `fake_source(is_training: bool, examples: Sequence[dict[str, tf.Tensor]], repeat: int = 1, spec: Optional[dict[str, tf.TypeSpec]] = None, shuffle_buffer_size: Optional[int] = None)` — [`L177`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L177)
- `fake_speech_source(*, is_training: bool, max_len: int = 100, num_examples: int = 100, speech_key: str = "speech", shuffle_buffer_size: Optional[int] = None)` — [`L370`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L370) — Fake speech data source.
- `fake_text2text_source(*, source_key: str = "source_text", target_key: str = "target_text", is_training: bool, shuffle_buffer_size: Optional[int] = None)` — [`L253`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L253)
- `fake_text_source(*, text_field_name: str = "text", repeat: int = 1, is_training: bool, shuffle_buffer_size: Optional[int] = None, batch_size: int = 2)` — [`L208`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L208)
- `fn()` — [`L191`](../../../../../../raw/code/axlearn/axlearn/common/input_fake.py#L191)

