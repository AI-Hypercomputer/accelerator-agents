---
title: 'Module: axlearn/common/elastic_input.py'
type: catalog
provenance: extracted
module: axlearn/common/elastic_input.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.elastic_input`/
symbols:
  ElasticInput.batches: ElasticInput#batches().
  ElasticDataset.__iter__: ElasticDataset#__iter__().
  ElasticInput.__init__: ElasticInput#__init__().
  ElasticInput.dataset: ElasticInput#dataset().
  ElasticSpmdInputDispatcher.__init__: ElasticSpmdInputDispatcher#__init__().
  ElasticSpmdInputDispatcher.feed_read_config: ElasticSpmdInputDispatcher#feed_read_config().
  ElasticSpmdInputDispatcher._feed_logical_batch_size: ElasticSpmdInputDispatcher#_feed_logical_batch_size.
  ElasticSpmdInputDispatcher.elastic_feed_mini_batch_size: ElasticSpmdInputDispatcher#elastic_feed_mini_batch_size.
  ElasticInput.elastic_input: ElasticInput#elastic_input.
  ElasticInput.Config.input: ElasticInput#Config#input.
  ElasticSpmdInputDispatcher.Config.partition_spec: ElasticSpmdInputDispatcher#Config#partition_spec.
  ElasticSpmdInputDispatcher.Config: ElasticSpmdInputDispatcher#Config#
  ElasticSpmdInputDispatcher._partition_spec: ElasticSpmdInputDispatcher#_partition_spec.
  ElasticInput.primary_input: ElasticInput#primary_input.
  ElasticSpmdInputDispatcher.feed_index: ElasticSpmdInputDispatcher#feed_index.
  ElasticSpmdInputDispatcher.elastic_feed_index: ElasticSpmdInputDispatcher#elastic_feed_index.
  ElasticSpmdInputDispatcher._device_physical_batch_size: ElasticSpmdInputDispatcher#_device_physical_batch_size.
  ElasticSpmdInputDispatcher.feed_count: ElasticSpmdInputDispatcher#feed_count.
  ElasticSpmdInputDispatcher.partition_spec: ElasticSpmdInputDispatcher#partition_spec().
  ElasticSpmdInputDispatcher.Config.num_max_slices: ElasticSpmdInputDispatcher#Config#num_max_slices.
  ElasticDatasetIterator.elastic_iterator: ElasticDatasetIterator#elastic_iterator.
  ElasticInput.Config: ElasticInput#Config#
  slice_count: slice_count().
  ElasticSpmdInputDispatcher: ElasticSpmdInputDispatcher#
  ElasticSpmdInputDispatcher.elastic_process_ids: ElasticSpmdInputDispatcher#elastic_process_ids.
  ElasticSpmdInputDispatcher.is_in_elastic_mode: ElasticSpmdInputDispatcher#is_in_elastic_mode().
  ElasticSpmdInputDispatcher.fid2pids: ElasticSpmdInputDispatcher#fid2pids().
  ElasticInput.is_in_elastic_mode: ElasticInput#is_in_elastic_mode().
  ElasticDatasetIterator: ElasticDatasetIterator#
  ElasticDatasetIterator.primary_iterator: ElasticDatasetIterator#primary_iterator.
  ElasticInput._padded_select: ElasticInput#_padded_select().
  ElasticSpmdInputDispatcher.logical_to_physical_batch: ElasticSpmdInputDispatcher#logical_to_physical_batch().
  ElasticSpmdInputDispatcher.physical_to_logical_batch: ElasticSpmdInputDispatcher#physical_to_logical_batch().
  ElasticSpmdInputDispatcher.logical_to_physical_shapes: ElasticSpmdInputDispatcher#logical_to_physical_shapes().
  ElasticDatasetIterator.__next__: ElasticDatasetIterator#__next__().
  ElasticSpmdInputDispatcher.elastic_feed_mini_batch_index: ElasticSpmdInputDispatcher#elastic_feed_mini_batch_index.
  ElasticSpmdInputDispatcher.num_logical_feeds: ElasticSpmdInputDispatcher#num_logical_feeds().
  ElasticInput: ElasticInput#
  ElasticDataset.elastic_dataset: ElasticDataset#elastic_dataset.
  get_process_index_and_count_and_mapping: get_process_index_and_count_and_mapping().
  ElasticDatasetIterator.elastic_process_ids: ElasticDatasetIterator#elastic_process_ids.
  ElasticDataset.primary_dataset: ElasticDataset#primary_dataset.
  ElasticDataset.elastic_process_ids: ElasticDataset#elastic_process_ids.
  ElasticDataset.is_primary_for_checkpoint: ElasticDataset#is_primary_for_checkpoint.
  ElasticSpmdInputDispatcher.is_primary: ElasticSpmdInputDispatcher#is_primary.
  ElasticSpmdInputDispatcher.logical_feed_index: ElasticSpmdInputDispatcher#logical_feed_index().
  ElasticSpmdInputDispatcher.feed_logical_batch_size: ElasticSpmdInputDispatcher#feed_logical_batch_size().
  ElasticSpmdInputDispatcher.device_physical_batch_size: ElasticSpmdInputDispatcher#device_physical_batch_size().
  ElasticSpmdInputDispatcher.Config.is_read_elastic_feed: ElasticSpmdInputDispatcher#Config#is_read_elastic_feed.
  ElasticDatasetIterator.is_primary_for_checkpoint: ElasticDatasetIterator#is_primary_for_checkpoint.
  ElasticDataset: ElasticDataset#
  NonUniformShardingError: NonUniformShardingError#
  process_count_per_slice: process_count_per_slice().
  ElasticDatasetIterator.__iter__: ElasticDatasetIterator#__iter__().
---
# Module: [`axlearn/common/elastic_input.py`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py)

## Classes
### `Config`  ·  implements/extends Config
- def: [`axlearn/common/elastic_input.py:300`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L300)
- doc: Configures ElasticInput.
- signature: `class Config(input_base.Input.Config):`
- members:
  - `input` — [`L303`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L303)
  - `is_read_elastic_feed` — [`L71`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L71)
  - `num_max_slices` — [`L66`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L66)
  - `partition_spec` — [`L62`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L62)
- uses (calls/refs, reference-scoped): [`config_class`](config.md#config_class), [`REQUIRED`](config.md#REQUIRED), [`Required`](config.md#Required), [`Input`](input_base.md#Input), [`Config`](input_base.md#Input.Config)
- used by: [`__init__`](elastic_input.md#ElasticInput.__init__), [`__init__`](elastic_input.md#ElasticSpmdInputDispatcher.__init__), [`feed_read_config`](elastic_input.md#ElasticSpmdInputDispatcher.feed_read_config), [`_feed_logical_batch_size`](elastic_input.md#ElasticSpmdInputDispatcher._feed_logical_batch_size), [`elastic_feed_mini_batch_size`](elastic_input.md#ElasticSpmdInputDispatcher.elastic_feed_mini_batch_size), [`elastic_input`](elastic_input.md#ElasticInput.elastic_input), [`_partition_spec`](elastic_input.md#ElasticSpmdInputDispatcher._partition_spec), [`primary_input`](elastic_input.md#ElasticInput.primary_input), [`feed_count`](elastic_input.md#ElasticSpmdInputDispatcher.feed_count), [`partition_spec`](elastic_input.md#ElasticSpmdInputDispatcher.partition_spec)

### `ElasticDataset`
- def: [`axlearn/common/elastic_input.py:271`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L271)
- doc: A composition of dataset from current process and processes of the elastic feed
- signature: `class ElasticDataset:`
- members:
  - `elastic_dataset` — [`L277`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L277)
  - `elastic_process_ids` — [`L280`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L280)
  - `is_primary_for_checkpoint` — [`L285`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L285)
  - `primary_dataset` — [`L274`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L274)
- protocol/private: `__iter__`[`L287`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L287)
- uses (calls/refs, reference-scoped): [`elastic_iterator`](elastic_input.md#ElasticDatasetIterator.elastic_iterator), [`ElasticDatasetIterator`](elastic_input.md#ElasticDatasetIterator), [`primary_iterator`](elastic_input.md#ElasticDatasetIterator.primary_iterator), [`elastic_process_ids`](elastic_input.md#ElasticDatasetIterator.elastic_process_ids), [`is_primary_for_checkpoint`](elastic_input.md#ElasticDatasetIterator.is_primary_for_checkpoint)
- used by: [`dataset`](elastic_input.md#ElasticInput.dataset)

### `ElasticDatasetIterator`
- def: [`axlearn/common/elastic_input.py:252`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L252)
- doc: Iterator for ElasticDataset
- signature: `class ElasticDatasetIterator:`
- members:
  - `elastic_iterator` — [`L256`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L256)
  - `elastic_process_ids` — [`L258`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L258)
  - `is_primary_for_checkpoint` — [`L259`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L259)
  - `primary_iterator` — [`L255`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L255)
- protocol/private: `__iter__`[`L261`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L261), `__next__`[`L264`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L264)
- used by: [`_get_spec`](checkpointer.md#TensorStoreStateStorage._get_spec), [`batches`](elastic_input.md#ElasticInput.batches), [`__iter__`](elastic_input.md#ElasticDataset.__iter__), [`async_save_tf_savables`](checkpointer.md#async_save_tf_savables), [`restore_tf_savables`](checkpointer.md#restore_tf_savables)

### `ElasticInput`  ·  implements/extends Input
- def: [`axlearn/common/elastic_input.py:296`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L296)
- doc: A general wrapper for Input to support elastic training
- signature: `class ElasticInput(input_base.Input):`
- members:
  - `batches(self, it: Iterator[Nested[Tensor]])` — [`L350`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L350)
  - `dataset(self)` — [`L330`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L330)
  - `is_in_elastic_mode(self)` — [`L327`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L327)
  - `elastic_input` — [`L316`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L316)
  - `primary_input` — [`L313`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L313)
- protocol/private: `__init__`[`L305`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L305), `_padded_select`[`L355`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L355)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`maybe_set_config`](config.md#maybe_set_config), [`Input`](input_base.md#Input), [`elastic_feed_mini_batch_size`](elastic_input.md#ElasticSpmdInputDispatcher.elastic_feed_mini_batch_size), [`input`](elastic_input.md#ElasticInput.Config.input), [`__init__`](input_base.md#Input.__init__), [`Config`](elastic_input.md#ElasticSpmdInputDispatcher.Config), [`input_dispatcher`](input_base.md#Input.Config.input_dispatcher), [`elastic_iterator`](elastic_input.md#ElasticDatasetIterator.elastic_iterator), [`Config`](elastic_input.md#ElasticInput.Config), [`ElasticSpmdInputDispatcher`](elastic_input.md#ElasticSpmdInputDispatcher), [`ElasticDatasetIterator`](elastic_input.md#ElasticDatasetIterator), [`primary_iterator`](elastic_input.md#ElasticDatasetIterator.primary_iterator), [`elastic_feed_mini_batch_index`](elastic_input.md#ElasticSpmdInputDispatcher.elastic_feed_mini_batch_index), [`elastic_dataset`](elastic_input.md#ElasticDataset.elastic_dataset), [`elastic_process_ids`](elastic_input.md#ElasticDataset.elastic_process_ids), [`is_primary_for_checkpoint`](elastic_input.md#ElasticDataset.is_primary_for_checkpoint), [`primary_dataset`](elastic_input.md#ElasticDataset.primary_dataset), [`ElasticDataset`](elastic_input.md#ElasticDataset), [`is_read_elastic_feed`](elastic_input.md#ElasticSpmdInputDispatcher.Config.is_read_elastic_feed)
- used by: [`eval_step`](evaler.md#SpmdEvaler.eval_step), [`Input`](input_base.md#Input), [`batches`](input_base.md#Input.batches), [`dataset`](input_base.md#Input.dataset)

### `ElasticSpmdInputDispatcher`  ·  implements/extends BaseInputDispatcher
- def: [`axlearn/common/elastic_input.py:55`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L55)
- doc: Similar to SpmdInputDispatcher, but to support elastic training
- signature: `class ElasticSpmdInputDispatcher(BaseInputDispatcher):`
- members:
  - `device_physical_batch_size(self)` — [`L221`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L221)
  - `feed_logical_batch_size(self)` — [`L212`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L212)
  - `feed_read_config(self)` — [`L224`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L224)
  - `fid2pids(feed_id)` — [`L134`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L134)
  - `is_in_elastic_mode(self)` — [`L74`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L74)
  - `logical_feed_index(self)` — [`L208`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L208)
  - `logical_to_physical_batch(self, logical_feed_batch: Nested[Tensor])` — [`L236`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L236)
  - `logical_to_physical_shapes(self, logical_feed_shapes: Nested[jax.ShapeDtypeStruct])` — [`L242`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L242) — Maps per-feed logical shapes to per-feed physical shapes for AOT compilation.
  - `num_logical_feeds(self)` — [`L204`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L204)
  - `partition_spec(self)` — [`L216`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L216)
  - `physical_to_logical_batch(self, global_physical_batch: Nested[Tensor])` — [`L239`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L239)
  - `elastic_feed_index` — [`L189`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L189)
  - `elastic_feed_mini_batch_index` — [`L171`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L171)
  - `elastic_feed_mini_batch_size` — [`L177`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L177)
  - `elastic_process_ids` — [`L193`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L193)
  - `feed_count` — [`L157`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L157)
  - `feed_index` — [`L158`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L158)
  - `is_primary` — [`L198`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L198)
- protocol/private: `__init__`[`L90`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L90), `_device_physical_batch_size`[`L127`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L127), `_feed_logical_batch_size`[`L161`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L161), `_partition_spec`[`L111`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L111)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`Module`](module.md#Module), [`__init__`](module.md#Module.__init__), [`global_logical_batch_size`](input_dispatch.md#BaseInputDispatcher.Config.global_logical_batch_size), `env`, [`BaseInputDispatcher`](input_dispatch.md#BaseInputDispatcher), [`partition_spec`](elastic_input.md#ElasticSpmdInputDispatcher.Config.partition_spec), [`Config`](elastic_input.md#ElasticSpmdInputDispatcher.Config), [`num_max_slices`](elastic_input.md#ElasticSpmdInputDispatcher.Config.num_max_slices), [`slice_count`](elastic_input.md#slice_count), [`_validate_logical_feed_shapes`](input_dispatch.md#_validate_logical_feed_shapes), [`get_process_index_and_count_and_mapping`](elastic_input.md#get_process_index_and_count_and_mapping), [`is_read_elastic_feed`](elastic_input.md#ElasticSpmdInputDispatcher.Config.is_read_elastic_feed), [`process_count_per_slice`](elastic_input.md#process_count_per_slice)
- used by: [`batches`](elastic_input.md#ElasticInput.batches), [`__init__`](elastic_input.md#ElasticInput.__init__), [`BaseInputDispatcher`](input_dispatch.md#BaseInputDispatcher), [`logical_to_physical_batch`](input_dispatch.md#BaseInputDispatcher.logical_to_physical_batch), [`physical_to_logical_batch`](input_dispatch.md#BaseInputDispatcher.physical_to_logical_batch), [`logical_to_physical_shapes`](input_dispatch.md#BaseInputDispatcher.logical_to_physical_shapes), [`feed_logical_batch_size`](input_dispatch.md#BaseInputDispatcher.feed_logical_batch_size), [`feed_read_config`](input_dispatch.md#BaseInputDispatcher.feed_read_config), [`partition_spec`](input_dispatch.md#BaseInputDispatcher.partition_spec), [`logical_feed_index`](input_dispatch.md#BaseInputDispatcher.logical_feed_index), [`num_logical_feeds`](input_dispatch.md#BaseInputDispatcher.num_logical_feeds), [`_padded_select`](elastic_input.md#ElasticInput._padded_select)

### `NonUniformShardingError`  ·  implements/extends ValueError
- def: [`axlearn/common/elastic_input.py:422`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L422)
- doc: Raised when sharding is not uniform across processes.
- signature: `class NonUniformShardingError(ValueError):`
- used by: [`get_process_index_and_count_and_mapping`](elastic_input.md#get_process_index_and_count_and_mapping)

## Functions
- `get_process_index_and_count_and_mapping(tensor_sharding: jsharding.Sharding, dim: int, ndims: int)` — [`L431`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L431) — Get current process index and number of unique processes for given dimension.
- `process_count_per_slice()` — [`L408`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L408) — Returns the number of processes per slice.
- `slice_count()` — [`L403`](../../../../../../raw/code/axlearn/axlearn/common/elastic_input.py#L403) — Returns the number of slices.

