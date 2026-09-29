---
title: 'Module: simply/utils/seqio_wrapper.py'
type: catalog
provenance: extracted
module: simply/utils/seqio_wrapper.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.seqio_wrapper`/
symbols:
  create_dataset: create_dataset().
  _SeqIOIterator.__next__: _SeqIOIterator#__next__().
  SeqIOConfig.__post_init__: SeqIOConfig#__post_init__().
  _SeqIOIterator.set_state: _SeqIOIterator#set_state().
  SeqIODataset.__iter__: SeqIODataset#__iter__().
  SeqIOConfig.feature_converter: SeqIOConfig#feature_converter().
  make_train_data: make_train_data().
  make_eval_data: make_eval_data().
  SeqIOConfig: SeqIOConfig#
  SeqIOConfig.feature_converter_name: SeqIOConfig#feature_converter_name.
  SeqIOConfig.task_feature_lengths: SeqIOConfig#task_feature_lengths().
  _SeqIOIterator._example_counter: _SeqIOIterator#_example_counter.
  _SeqIOIterator._seqio_dataiter: _SeqIOIterator#_seqio_dataiter.
  _SeqIOIterator.get_state: _SeqIOIterator#get_state().
  _SeqIOIterator.__str__: _SeqIOIterator#__str__().
  SeqIODataset.set_slice: SeqIODataset#set_slice().
  _SeqIOIterator._config: _SeqIOIterator#_config.
  SeqIOConfig.seq_len: SeqIOConfig#seq_len.
  _SeqIOIterator.__init__: _SeqIOIterator#__init__().
  SeqIODataset.__init__: SeqIODataset#__init__().
  SeqIODataset.__str__: SeqIODataset#__str__().
  SeqIOConfig.use_packing: SeqIOConfig#use_packing.
  SeqIOConfig.use_cached: SeqIOConfig#use_cached.
  SeqIOConfig.shuffle: SeqIOConfig#shuffle.
  SeqIOConfig.num_epochs: SeqIOConfig#num_epochs.
  SeqIOConfig.seed: SeqIOConfig#seed.
  SeqIOConfig.from_config: SeqIOConfig#from_config().
  _SeqIOIterator: _SeqIOIterator#
  SeqIODataset: SeqIODataset#
  SeqIODataset._config: SeqIODataset#_config.
  SeqIODataset._num_workers: SeqIODataset#_num_workers.
  SeqIODataset._worker_index: SeqIODataset#_worker_index.
  SeqIOConfig.dataset_name: SeqIOConfig#dataset_name.
  SeqIOConfig.batch_size: SeqIOConfig#batch_size.
  SeqIOConfig.split: SeqIOConfig#split.
  SeqIOConfig.bos_id: SeqIOConfig#bos_id.
  _SeqIOIterator._worker_index: _SeqIOIterator#_worker_index.
  _SeqIOIterator._num_workers: _SeqIOIterator#_num_workers.
---
# Module: [`simply/utils/seqio_wrapper.py`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py)

## Classes
### `SeqIOConfig`
- def: [`simply/utils/seqio_wrapper.py:14`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L14)
- doc: Configuration for SeqIO dataset.
- signature: `class SeqIOConfig:`
- members:
  - `feature_converter(self)` — [`L53`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L53)
  - `from_config(cls, config, **overrides)` — [`L64`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L64)
  - `task_feature_lengths(self)` — [`L42`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L42)
  - `batch_size` — [`L19`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L19)
  - `bos_id` — [`L23`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L23)
  - `dataset_name` — [`L17`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L17)
  - `feature_converter_name` — [`L18`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L18)
  - `num_epochs` — [`L27`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L27)
  - `seed` — [`L28`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L28)
  - `seq_len` — [`L20`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L20)
  - `shuffle` — [`L26`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L26)
  - `split` — [`L21`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L21)
  - `use_cached` — [`L25`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L25)
  - `use_packing` — [`L22`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L22)
- protocol/private: `__post_init__`[`L30`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L30)
- used by: [`create_dataset`](seqio_wrapper.md#create_dataset), [`set_state`](seqio_wrapper.md#_SeqIOIterator.set_state), [`make_eval_data`](seqio_wrapper.md#make_eval_data), [`make_train_data`](seqio_wrapper.md#make_train_data), [`__init__`](seqio_wrapper.md#SeqIODataset.__init__), [`__init__`](seqio_wrapper.md#_SeqIOIterator.__init__)

### `SeqIODataset`  ·  implements/extends IterDataset
- def: [`simply/utils/seqio_wrapper.py:144`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L144)
- doc: Batch transformation for IterDatasets, using a batch function.
- signature: `class SeqIODataset(grain.IterDataset[dict[str, Any]]):`
- members:
  - `set_slice(self, sl: slice, sequential_slice: bool = False)` — [`L153`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L153)
- protocol/private: `__init__`[`L147`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L147), `__iter__`[`L162`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L162), `__str__`[`L165`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L165), `_config`[`L149`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L149), `_num_workers`[`L150`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L150), `_worker_index`[`L151`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L151)
- uses (calls/refs, reference-scoped): [`SeqIOConfig`](seqio_wrapper.md#SeqIOConfig), [`_SeqIOIterator`](seqio_wrapper.md#_SeqIOIterator)
- used by: [`make_eval_data`](seqio_wrapper.md#make_eval_data), [`make_train_data`](seqio_wrapper.md#make_train_data)

### `_SeqIOIterator`  ·  implements/extends DatasetIterator
- def: [`simply/utils/seqio_wrapper.py:103`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L103)
- doc: Iterator that batches elements with a given batch function.
- signature: `class _SeqIOIterator(grain.DatasetIterator[dict[str, Any]]):`
- members:
  - `get_state(self)` — [`L126`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L126)
  - `set_state(self, state)` — [`L132`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L132)
- protocol/private: `__init__`[`L106`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L106), `__next__`[`L114`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L114), `__str__`[`L137`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L137), `_config`[`L108`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L108), `_example_counter`[`L111`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L111), `_num_workers`[`L110`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L110), `_seqio_dataiter`[`L112`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L112), `_worker_index`[`L109`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L109)
- uses (calls/refs, reference-scoped): [`create_dataset`](seqio_wrapper.md#create_dataset), [`SeqIOConfig`](seqio_wrapper.md#SeqIOConfig)
- used by: [`__iter__`](seqio_wrapper.md#SeqIODataset.__iter__)

## Functions
- `create_dataset(config: SeqIOConfig, worker_index: int, num_workers: int)` — [`L74`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L74) — Fork of simply/data_lib.py:create_dataset_split().
- `make_eval_data(config, num_workers: int = 32, worker_buffer_size: int = 2)` — [`L181`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L181) — Returns a PyGrain dataset for evaluation, from Simply experiment config.
- `make_train_data(config, num_workers: int = 32, worker_buffer_size: int = 2)` — [`L169`](../../../../../../raw/code/simply/simply/utils/seqio_wrapper.py#L169) — Returns a PyGrain dataset for training, from Simply experiment config.

