---
title: 'Module: src/maxtext/common/data_loader.py'
type: catalog
provenance: extracted
module: src/maxtext/common/data_loader.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.common.data_loader`/
symbols:
  RampUpDataLoader.load_next_batch: RampUpDataLoader#load_next_batch().
  DataLoader.load_next_batch_pre_sharding: DataLoader#load_next_batch_pre_sharding().
  DataLoader.load_next_batch: DataLoader#load_next_batch().
  DataLoader.data_iterator: DataLoader#data_iterator.
  DataLoader.update_data_iterator: DataLoader#update_data_iterator().
  DataLoader.config: DataLoader#config.
  DataLoader.check_example_batch: DataLoader#check_example_batch().
  RampUpDataLoader.batch_buffer: RampUpDataLoader#batch_buffer.
  DataLoader.last_batch: DataLoader#last_batch.
  create_dataloader: create_dataloader().
  DataLoader: DataLoader#
  DataLoader.data_iterator_index: DataLoader#data_iterator_index.
  RampUpDataLoader.buffer_start: RampUpDataLoader#buffer_start.
  RampUpDataLoader: RampUpDataLoader#
  DataLoader.data_iterator_list: DataLoader#data_iterator_list.
  RampUpDataLoader.__init__: RampUpDataLoader#__init__().
  DataLoader.input_data_shardings: DataLoader#input_data_shardings.
  RampUpDataLoader.rampup_active: RampUpDataLoader#rampup_active.
  DataLoader.__init__: DataLoader#__init__().
  DataLoader.goodput_recorder: DataLoader#goodput_recorder.
  RampUpDataLoader._slice_and_concat: RampUpDataLoader#_slice_and_concat().
  RampUpDataLoader._slice: RampUpDataLoader#_slice().
---
# Module: [`src/maxtext/common/data_loader.py`](../../../../../../../raw/code/maxtext/src/maxtext/common/data_loader.py)

## Classes
### `DataLoader`
- def: [`src/maxtext/common/data_loader.py:31`](../../../../../../../raw/code/maxtext/src/maxtext/common/data_loader.py#L31)
- doc: Loads preprocessed data for training.
- signature: `class DataLoader:`
- members:
  - `check_example_batch(self)` — [`L79`](../../../../../../../raw/code/maxtext/src/maxtext/common/data_loader.py#L79)
  - `load_next_batch(self, *args, **kwargs)` — [`L72`](../../../../../../../raw/code/maxtext/src/maxtext/common/data_loader.py#L72) — Loads the next batch with sharding hint
  - `load_next_batch_pre_sharding(self)` — [`L54`](../../../../../../../raw/code/maxtext/src/maxtext/common/data_loader.py#L54) — Loads the next batch w/o sharding. Can keep reusing the same batch for performance reasons.
  - `update_data_iterator(self)` — [`L48`](../../../../../../../raw/code/maxtext/src/maxtext/common/data_loader.py#L48) — Update to the next data iterator in the list, if applicable.
  - `config` — [`L37`](../../../../../../../raw/code/maxtext/src/maxtext/common/data_loader.py#L37)
  - `data_iterator` — [`L42`](../../../../../../../raw/code/maxtext/src/maxtext/common/data_loader.py#L42)
  - `data_iterator_index` — [`L41`](../../../../../../../raw/code/maxtext/src/maxtext/common/data_loader.py#L41)
  - `data_iterator_list` — [`L40`](../../../../../../../raw/code/maxtext/src/maxtext/common/data_loader.py#L40)
  - `goodput_recorder` — [`L38`](../../../../../../../raw/code/maxtext/src/maxtext/common/data_loader.py#L38)
  - `input_data_shardings` — [`L46`](../../../../../../../raw/code/maxtext/src/maxtext/common/data_loader.py#L46)
  - `last_batch` — [`L45`](../../../../../../../raw/code/maxtext/src/maxtext/common/data_loader.py#L45)
- protocol/private: `__init__`[`L36`](../../../../../../../raw/code/maxtext/src/maxtext/common/data_loader.py#L36)
- uses (calls/refs, reference-scoped): [`load_next_batch`](data_loader.md#RampUpDataLoader.load_next_batch), [`RampUpDataLoader`](data_loader.md#RampUpDataLoader)
- used by: [`load_next_batch`](data_loader.md#RampUpDataLoader.load_next_batch), [`create_dataloader`](data_loader.md#create_dataloader), [`RampUpDataLoader`](data_loader.md#RampUpDataLoader), [`__init__`](data_loader.md#RampUpDataLoader.__init__)

### `RampUpDataLoader`  ·  implements/extends DataLoader
- def: [`src/maxtext/common/data_loader.py:88`](../../../../../../../raw/code/maxtext/src/maxtext/common/data_loader.py#L88)
- doc: A DataLoader that implements batch size ramp-up.
- signature: `class RampUpDataLoader(DataLoader):`
- members:
  - `load_next_batch(self, *args, rampup_manager=None, **kwargs)` — [`L106`](../../../../../../../raw/code/maxtext/src/maxtext/common/data_loader.py#L106) — Updates the batch size based on the schedule and then loads the next
  - `batch_buffer` — [`L103`](../../../../../../../raw/code/maxtext/src/maxtext/common/data_loader.py#L103)
  - `buffer_start` — [`L104`](../../../../../../../raw/code/maxtext/src/maxtext/common/data_loader.py#L104)
  - `rampup_active` — [`L102`](../../../../../../../raw/code/maxtext/src/maxtext/common/data_loader.py#L102)
- protocol/private: `__init__`[`L98`](../../../../../../../raw/code/maxtext/src/maxtext/common/data_loader.py#L98), `_slice`[`L146`](../../../../../../../raw/code/maxtext/src/maxtext/common/data_loader.py#L146), `_slice_and_concat`[`L127`](../../../../../../../raw/code/maxtext/src/maxtext/common/data_loader.py#L127)
- uses (calls/refs, reference-scoped): [`load_next_batch_pre_sharding`](data_loader.md#DataLoader.load_next_batch_pre_sharding), [`load_next_batch`](data_loader.md#DataLoader.load_next_batch), [`config`](data_loader.md#DataLoader.config), [`DataLoader`](data_loader.md#DataLoader), [`input_data_shardings`](data_loader.md#DataLoader.input_data_shardings), [`__init__`](data_loader.md#DataLoader.__init__)
- used by: [`load_next_batch`](data_loader.md#DataLoader.load_next_batch), [`create_dataloader`](data_loader.md#create_dataloader), [`DataLoader`](data_loader.md#DataLoader)

## Functions
- `create_dataloader(config, mesh, data_iterator, goodput_recorder, rampup_manager)` — [`L162`](../../../../../../../raw/code/maxtext/src/maxtext/common/data_loader.py#L162) — Create the dataloader

