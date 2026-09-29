---
title: 'Module: axlearn/common/input_tf_data.py'
type: catalog
provenance: extracted
module: axlearn/common/input_tf_data.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.input_tf_data`/
symbols:
  DatasetToDatasetFn: DatasetToDatasetFn#
  Input.__init__: Input#__init__().
  BuildDatasetFn: BuildDatasetFn#
  tfds_dataset: tfds_dataset().
  set_dispatch_config_recursively.enter_fn: set_dispatch_config_recursively().enter_fn().
  tfds_dataset.fn: tfds_dataset().fn().
  sample_from_datasets: sample_from_datasets().
  concatenate_datasets: concatenate_datasets().
  with_processor: with_processor().
  batch: batch().
  batch.fn: batch().fn().
  Input.Config.source: Input#Config#source.
  Input.Config.processor: Input#Config#processor.
  Input.Config.batcher: Input#Config#batcher.
  Input._batcher: Input#_batcher.
  pack_to_batch.fn: pack_to_batch().fn().
  per_feed_batch: per_feed_batch().
  disable_shuffle_recursively: disable_shuffle_recursively().
  Input.Config: Input#Config#
  Input._source: Input#_source.
  Input._processor: Input#_processor.
  _pad_logical_to_physical: _pad_logical_to_physical().
  per_feed_batch.fn: per_feed_batch().fn().
  rekey.fn: rekey().fn().
  set_dispatch_config_recursively: set_dispatch_config_recursively().
  Input.dataset: Input#dataset().
  Input.Config.is_training: Input#Config#is_training.
  chain: chain().
  _pad_for_evaluation: _pad_for_evaluation().
  _maybe_shard_examples: _maybe_shard_examples().
  rekey: rekey().
  disable_shuffle_recursively.enter_fn: disable_shuffle_recursively().enter_fn().
  pad_to_batch: pad_to_batch().
  tfrecord_dataset: tfrecord_dataset().
  take: take().
  unbatch: unbatch().
  select_fields: select_fields().
  remove_fields: remove_fields().
  filter_examples: filter_examples().
  squeeze_fields: squeeze_fields().
  identity: identity().
  skip_on_error: skip_on_error().
  extract_from_sequence: extract_from_sequence().
  shuffle: shuffle().
  unpack: unpack().
  ragged_to_tensor: ragged_to_tensor().
  Input.source: Input#source().
  Input.processor: Input#processor().
  Input.element_spec: Input#element_spec().
  preserve_element_spec: preserve_element_spec().
  add_static_fields: add_static_fields().
  pack_to_batch: pack_to_batch().
  trim_to_batch: trim_to_batch().
  PadExampleFn: PadExampleFn.
  _infer_cardinality: _infer_cardinality().
  tfrecord_dataset.fn: tfrecord_dataset().fn().
  squeeze_fields.example_fn: squeeze_fields().example_fn().
  chain.fn: chain().fn().
  rekey.has_path: rekey().has_path().
  Input: Input#
  preserve_element_spec.process_dataset_fn: preserve_element_spec().process_dataset_fn().
  pad_to_batch.process_example_fn: pad_to_batch().process_example_fn().
  pack_to_batch.scan_fn: pack_to_batch().scan_fn().
  pack_to_batch.define_shape: pack_to_batch().define_shape().
  default_pad_example_fn: default_pad_example_fn().
  tfds_read_config: tfds_read_config().
  pack_to_batch._init_state: pack_to_batch()._init_state().
  _infer_num_shards: _infer_num_shards().
  _infer_num_examples: _infer_num_examples().
  tfrecord_dataset._decode_record: tfrecord_dataset()._decode_record().
  sample_from_datasets.fn: sample_from_datasets().fn().
  concatenate_datasets.fn: concatenate_datasets().fn().
  take.fn: take().fn().
  unbatch.fn: unbatch().fn().
  remove_fields.process_fn: remove_fields().process_fn().
  filter_examples.fn: filter_examples().fn().
  with_processor.fn: with_processor().fn().
  default_pad_example_fn.spec_to_tensor: default_pad_example_fn().spec_to_tensor().
  identity.fn: identity().fn().
  skip_on_error.fn: skip_on_error().fn().
  extract_from_sequence.fn: extract_from_sequence().fn().
  shuffle.fn: shuffle().fn().
  unpack.fn: unpack().fn().
  ragged_to_tensor.fn: ragged_to_tensor().fn().
  preserve_element_spec.process_dataset_fn._set_shape: preserve_element_spec().process_dataset_fn()._set_shape().
  add_static_fields.fn: add_static_fields().fn().
  pad_to_batch.pad_fn: pad_to_batch().pad_fn().
  pack_to_batch.define_shape.fn: pack_to_batch().define_shape().fn().
  trim_to_batch.trim: trim_to_batch().trim().
  BuildDatasetFn.__call__: BuildDatasetFn#__call__().
  DatasetToDatasetFn.__call__: DatasetToDatasetFn#__call__().
  trim_and_pad_tensor: trim_and_pad_tensor().
---
# Module: [`axlearn/common/input_tf_data.py`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py)

## Classes
### `BuildDatasetFn`  ·  implements/extends Protocol
- def: [`axlearn/common/input_tf_data.py:125`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L125)
- doc: A function to create a tf.data.Dataset instance.
- signature: `class BuildDatasetFn(Protocol):`
- protocol/private: `__call__`[`L128`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L128)
- used by: [`tfds_dataset`](input_tf_data.md#tfds_dataset), [`concatenate_datasets`](input_tf_data.md#concatenate_datasets), [`sample_from_datasets`](input_tf_data.md#sample_from_datasets), [`with_processor`](input_tf_data.md#with_processor), [`fake_source`](input_fake.md#fake_source), [`source`](input_tf_data.md#Input.Config.source), [`fake_classification_source`](input_fake.md#fake_classification_source), [`fake_classification_source_instruct_lm`](input_fake.md#fake_classification_source_instruct_lm), [`fake_glue_source`](input_fake.md#fake_glue_source), [`fake_serialized_json_source`](input_fake.md#fake_serialized_json_source), [`fake_speech_source`](input_fake.md#fake_speech_source), [`fake_text2text_source`](input_fake.md#fake_text2text_source), [`fake_text_source`](input_fake.md#fake_text_source), [`source`](input_tf_data.md#Input.source), [`tfrecord_dataset`](input_tf_data.md#tfrecord_dataset)

### `Config`  ·  implements/extends Config
- def: [`axlearn/common/input_tf_data.py:1242`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1242)
- doc: Configures Input.
- signature: `class Config(input_base.Input.Config):`
- members:
  - `batcher` — [`L1260`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1260)
  - `is_training` — [`L1245`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1245)
  - `processor` — [`L1257`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1257)
  - `source` — [`L1252`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1252)
- uses (calls/refs, reference-scoped): [`config_class`](config.md#config_class), [`REQUIRED`](config.md#REQUIRED), [`Required`](config.md#Required), [`InstantiableConfig`](config.md#InstantiableConfig), [`config_for_function`](config.md#config_for_function), [`DatasetToDatasetFn`](input_tf_data.md#DatasetToDatasetFn), [`Input`](input_base.md#Input), [`BuildDatasetFn`](input_tf_data.md#BuildDatasetFn), [`Config`](input_base.md#Input.Config), [`batch`](input_tf_data.md#batch)
- used by: [`__init__`](input_tf_data.md#Input.__init__), [`_batcher`](input_tf_data.md#Input._batcher), [`disable_shuffle_recursively`](input_tf_data.md#disable_shuffle_recursively), [`_processor`](input_tf_data.md#Input._processor), [`_source`](input_tf_data.md#Input._source)

### `DatasetToDatasetFn`  ·  implements/extends Protocol
- def: [`axlearn/common/input_tf_data.py:131`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L131)
- doc: A function to create a tf.data.Dataset instance from the given dataset.
- signature: `class DatasetToDatasetFn(Protocol):`
- protocol/private: `__call__`[`L134`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L134)
- used by: [`batch`](input_tf_data.md#batch), [`batcher`](input_tf_data.md#Input.Config.batcher), [`processor`](input_tf_data.md#Input.Config.processor), [`per_feed_batch`](input_tf_data.md#per_feed_batch), [`chain`](input_tf_data.md#chain), [`pad_to_batch`](input_tf_data.md#pad_to_batch), [`rekey`](input_tf_data.md#rekey), [`add_static_fields`](input_tf_data.md#add_static_fields), [`extract_from_sequence`](input_tf_data.md#extract_from_sequence), [`filter_examples`](input_tf_data.md#filter_examples), [`identity`](input_tf_data.md#identity), [`pack_to_batch`](input_tf_data.md#pack_to_batch), [`preserve_element_spec`](input_tf_data.md#preserve_element_spec), [`processor`](input_tf_data.md#Input.processor), [`ragged_to_tensor`](input_tf_data.md#ragged_to_tensor), [`remove_fields`](input_tf_data.md#remove_fields), [`select_fields`](input_tf_data.md#select_fields), [`shuffle`](input_tf_data.md#shuffle), [`skip_on_error`](input_tf_data.md#skip_on_error), [`squeeze_fields`](input_tf_data.md#squeeze_fields), [`take`](input_tf_data.md#take), [`trim_to_batch`](input_tf_data.md#trim_to_batch), [`unbatch`](input_tf_data.md#unbatch), [`unpack`](input_tf_data.md#unpack)

### `Input`  ·  implements/extends Input
- def: [`axlearn/common/input_tf_data.py:1228`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1228)
- doc: A Module to generate input batches with tf.data.Dataset.
- signature: `class Input(input_base.Input):`
- members:
  - `dataset(self)` — [`L1287`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1287)
  - `element_spec(self)` — [`L1290`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1290) — Returns the tfds element spec.
  - `processor(self)` — [`L1284`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1284)
  - `source(self)` — [`L1280`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1280)
- protocol/private: `__init__`[`L1262`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1262), `_batcher`[`L1277`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1277), `_processor`[`L1276`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1276), `_source`[`L1275`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1275)
- uses (calls/refs, reference-scoped): [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`Module`](module.md#Module), [`instantiate`](config.md#InstantiableConfig.instantiate), [`maybe_set_config`](config.md#maybe_set_config), [`DatasetToDatasetFn`](input_tf_data.md#DatasetToDatasetFn), [`children`](module.md#Module.children), [`Input`](input_base.md#Input), [`input_dispatcher`](input_base.md#Input.input_dispatcher), [`BuildDatasetFn`](input_tf_data.md#BuildDatasetFn), [`__init__`](input_base.md#Input.__init__), [`batcher`](input_tf_data.md#Input.Config.batcher), [`processor`](input_tf_data.md#Input.Config.processor), [`source`](input_tf_data.md#Input.Config.source), [`Config`](input_tf_data.md#Input.Config), [`feed_logical_batch_size`](input_dispatch.md#BaseInputDispatcher.feed_logical_batch_size), [`feed_read_config`](input_dispatch.md#BaseInputDispatcher.feed_read_config), [`set_dispatch_config_recursively`](input_tf_data.md#set_dispatch_config_recursively), [`is_training`](input_tf_data.md#Input.Config.is_training)
- used by: [`Input`](input_base.md#Input), [`dataset`](input_base.md#Input.dataset), [`disable_shuffle_recursively`](input_tf_data.md#disable_shuffle_recursively), [`element_spec`](input_base.md#Input.element_spec)

## Functions
- `_decode_record(record: dict[str, tf.Tensor])` — [`L425`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L425) — Decodes a record to a TensorFlow example.
- `_infer_cardinality(dataset: tf.data.Dataset)` — [`L683`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L683) — Returns the size of the dataset, by counting examples if necessary.
- `_infer_num_examples(builder: tfds.core.DatasetBuilder, split: str)` — [`L198`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L198) — Attempts to infer the number of examples associated with the given split.
- `_infer_num_shards(builder: tfds.core.DatasetBuilder, split: str)` — [`L179`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L179) — Attempts to infer the number of shards associated with the given split.
- `_init_state(dtype)` — [`L1454`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1454)
- `_maybe_shard_examples(builder: tfds.core.DatasetBuilder, read_config: InstantiableConfig, split: str, required_shards: int, is_training: bool, dataset_name: str)` — [`L217`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L217) — Determines how to split the examples into required number of shards.
- `_pad_for_evaluation(dataset: tf.data.Dataset, *, per_feed_batch_size: int, pad_example_fn: PadExampleFn)` — [`L697`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L697) — Pad evaluation dataset.
- `_pad_logical_to_physical(dataset: tf.data.Dataset, *, global_batch_size: int, global_logical_batch_size: int, num_logical_feeds: int, logical_feed_index: Optional[int], pad_example_fn: PadExampleFn)` — [`L745`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L745) — Pad logical dataset in preparation for batching.
- `_set_shape(x)` — [`L1345`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1345)
- `add_static_fields(key_map: dict[str, Any])` — [`L1355`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1355) — Adds a predetermined set of key, value pairs to each example.
- `batch(global_batch_size: int, *, is_training: bool, pad_example_fn: PadExampleFn, global_logical_batch_size: Optional[int] = None, logical_feed_indices: Optional[Sequence[int]] = None, prefetch_buffer_size: Optional[int] = None, post_batch_processor: Optional[ConfigOr[DatasetToDatasetFn]] = None, repeat: Optional[int] = None)` — [`L839`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L839) — Returns a function that generates a tf.data.Dataset object.
- `chain(*args, is_training: Optional[bool] = None)` — [`L655`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L655)
- `concatenate_datasets(is_training: bool, *, sources: Sequence[InstantiableConfig])` — [`L532`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L532) — Concatenates the given datasets sequentially (one after another).
- `default_pad_example_fn(element_spec: Any)` — [`L673`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L673) — Returns 0 values (or empty strings) for every tensor.
- `define_shape(element_spec: Any, batch_size: int)` — [`L1488`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1488)
- `disable_shuffle_recursively(cfg: Input.Config)` — [`L1299`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1299) — Disables all shuffling on the input config.
- `enter_fn(_, value, default_kv)` — [`L1215`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1215)
- `enter_fn(_, child, default_kv)` — [`L1305`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1305)
- `example_fn(example: dict[str, Tensor])` — [`L621`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L621)
- `extract_from_sequence(in_key: str, out_key: str, idx: Union[int, slice] = 0)` — [`L1063`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1063) — Provides function to extract slice or value at index from keyed sequence.
- `filter_examples(filter_fn: Callable)` — [`L601`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L601) — Filter the dataset with the given filter function.
- `fn()` — [`L347`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L347)
- `fn()` — [`L429`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L429)
- `fn()` — [`L497`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L497)
- `fn()` — [`L559`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L559)
- `fn(ds: tf.data.Dataset)` — [`L569`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L569)
- `fn(ds: tf.data.Dataset)` — [`L576`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L576)
- `fn(ds: tf.data.Dataset)` — [`L604`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L604)
- `fn()` — [`L648`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L648)
- `fn(ds: tf.data.Dataset)` — [`L659`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L659)
- `fn(ds: tf.data.Dataset)` — [`L915`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L915)
- `fn(ds: tf.data.Dataset)` — [`L1003`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1003)
- `fn(ds: tf.data.Dataset)` — [`L1041`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1041)
- `fn(ds: tf.data.Dataset)` — [`L1057`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1057)
- `fn(example: dict[str, tf.Tensor])` — [`L1080`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1080)
- `fn(example: dict[str, tf.Tensor])` — [`L1125`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1125)
- `fn(ds: tf.data.Dataset)` — [`L1157`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1157)
- `fn(example: dict[str, tf.Tensor])` — [`L1180`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1180)
- `fn(example: dict[str, tf.Tensor])` — [`L1203`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1203)
- `fn(example: dict[str, Any])` — [`L1366`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1366)
- `fn(example: dict[str, tf.Tensor])` — [`L1489`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1489)
- `fn(ds: tf.data.Dataset)` — [`L1496`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1496)
- `has_path(x, path: str)` — [`L1118`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1118)
- `identity()` — [`L1038`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1038) — Identity function, useful for example as batcher when data is already batched.
- `pack_to_batch(batch_size: int, pad_value: int = 0)` — [`L1414`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1414) — Packs along the first (batch) dimension.
- `pad_fn(v: tf.Tensor)` — [`L1401`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1401)
- `pad_to_batch(batch_size: int, pad_value: int = 0)` — [`L1374`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1374) — Pads along the first (batch) dimension.
- `per_feed_batch(feed_batch_size: int, *, is_training: bool, pad_example_fn: PadExampleFn, prefetch_buffer_size: Optional[int] = None, post_batch_processor: Optional[ConfigOr[DatasetToDatasetFn]] = None, repeat: Optional[int] = None)` — [`L964`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L964) — Returns a DatasetToDatasetFn that batches examples by `feed_feed_batch_size`.
- `preserve_element_spec(fn: DatasetToDatasetFn, key_map: Optional[dict[str, str]] = None)` — [`L1321`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1321) — Wraps a processor by ensuring that it does not change the dataset element_spec.
- `process_dataset_fn(ds: tf.data.Dataset)` — [`L1337`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1337)
- `process_example_fn(example: dict[str, tf.Tensor])` — [`L1408`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1408)
- `process_fn(example: dict[str, tf.Tensor])` — [`L591`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L591)
- `ragged_to_tensor(feature_shapes: dict[str, Any], default_value: int = 0)` — [`L1191`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1191) — Converts ragged tensors specified in `feature_shapes`
- `rekey(key_map: dict[str, str], default_value: Optional[Any] = "", retain_original_inputs: bool = False, separator: Optional[str] = None)` — [`L1087`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1087) — Replace the feature keys according to mapping in `key_map`.
- `remove_fields(fields: Sequence[str])` — [`L588`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L588) — Filter the dataset to remove the fields specified.
- `sample_from_datasets(is_training: bool, *, sources: Sequence[InstantiableConfig], weights: Sequence[float], seed: Optional[int] = None, autotune_ram_budget_gb: Optional[int] = None)` — [`L451`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L451) — Returns a data source formed by sampling from multiple data sources without replacement.
- `scan_fn(carry: dict[str, Any], elem: dict[str, tf.Tensor])` — [`L1462`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1462)
- `select_fields(fields: Sequence[str])` — [`L582`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L582) — Filter the dataset to only select the fields specified.
- `set_dispatch_config_recursively(cfg: ConfigBase, **kwargs)` — [`L1212`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1212) — Sets **kwargs on all tfds_read_config in `cfg`.
- `shuffle(shuffle_buffer_size: int)` — [`L1145`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1145) — Shuffle dataset if given buffersize is valid (i.e. > 0).
- `skip_on_error(*, log_warning: bool = False)` — [`L1047`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1047) — Silently skip examples in the dataset that raise an error.
- `spec_to_tensor(spec: tf.TensorSpec)` — [`L676`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L676)
- `squeeze_fields(axis: Mapping[str, Optional[Union[int, tuple[int, ...]]]])` — [`L610`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L610) — Squeeze fields specified using the corresponding axis.
- `take(num_examples: int)` — [`L568`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L568)
- `tfds_dataset(dataset_name: str, *, split: str, is_training: bool, train_shuffle_buffer_size: Optional[int] = None, train_shuffle_files: Optional[bool] = None, data_dir: Optional[str] = None, download: bool = False, read_config: Optional[InstantiableConfig] = None, decoders: Optional[InstantiableConfig] = None)` — [`L285`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L285) — Returns a BuildDatasetFn for the given TFDS dataset name and split.
- `tfds_read_config(*, is_training: bool, num_shards: Optional[int] = None, shard_index: Optional[int] = None, read_parallelism: int = 1, decode_parallelism: int = 32)` — [`L137`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L137) — Constructs a ReadConfig for tfds dataset.
- `tfrecord_dataset(glob_path: str, is_training: bool, shuffle_buffer_size: int, features: dict[str, tf.io.FixedLenFeature], compression_type: Optional[str] = None, read_parallelism: int = 1)` — [`L391`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L391) — Builds a BuildDatasetFn for the given a TFRecord dataset name.
- `trim(example: dict[str, tf.Tensor])` — [`L1524`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1524)
- `trim_and_pad_tensor(t: Union[tf.Tensor, tf.RaggedTensor], max_len: int, pad_id: int = 0)` — [`L1532`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1532) — Convert a tensor to uniform length by trimming and padding the last dimension.
- `trim_to_batch(batch_size: int)` — [`L1521`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1521) — Trims the first (batch) dimension.
- `unbatch()` — [`L575`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L575)
- `unpack(key_map: dict[str, tuple[str, ...]])` — [`L1166`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L1166) — Provides function to return flattened values according to key map.
- `with_processor(source: InstantiableConfig, *, processor: InstantiableConfig, is_training: Optional[bool] = None)` — [`L629`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L629) — Returns a BuildDatasetFn that combines the given `source` and `processor`.

## Module values
- `PadExampleFn` — [`L670`](../../../../../../raw/code/axlearn/axlearn/common/input_tf_data.py#L670)

