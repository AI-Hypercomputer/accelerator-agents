---
title: 'Module: src/maxtext/input_pipeline/multihost_dataloading.py'
type: catalog
provenance: extracted
module: src/maxtext/input_pipeline/multihost_dataloading.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.input_pipeline.multihost_dataloading`/
symbols:
  MultiHostDataLoadIterator._get_next_batch_sharded: MultiHostDataLoadIterator#_get_next_batch_sharded().
  RemoteIteratorWrapper.dummy_array: RemoteIteratorWrapper#dummy_array.
  MultiHostDataLoadIterator.reset: MultiHostDataLoadIterator#reset().
  MultiHostDataLoadIterator.dataloader: MultiHostDataLoadIterator#dataloader.
  MultiHostDataLoadIterator.local_iterator: MultiHostDataLoadIterator#local_iterator.
  RemoteIterator.iterator: RemoteIterator#iterator.
  RemoteIterator.get_next: RemoteIterator#get_next().
  RemoteIterator.save_state: RemoteIterator#save_state().
  RemoteIterator.restore_state: RemoteIterator#restore_state().
  RemoteIteratorWrapper.__next__: RemoteIteratorWrapper#__next__().
  RemoteIteratorWrapper.cpu_mesh: RemoteIteratorWrapper#cpu_mesh.
  RemoteIterator.reset: RemoteIterator#reset().
  RemoteIteratorWrapper.save_state: RemoteIteratorWrapper#save_state().
  RemoteIteratorWrapper.restore_state: RemoteIteratorWrapper#restore_state().
  _TF_RETRYABLE_ERRORS: _TF_RETRYABLE_ERRORS.
  MultiHostDataLoadIterator._make_padding_batch: MultiHostDataLoadIterator#_make_padding_batch().
  RemoteIterator.elastic: RemoteIterator#elastic.
  MultiHostDataLoadIterator.last_local_data: MultiHostDataLoadIterator#last_local_data.
  RemoteIteratorWrapper.local_iterator: RemoteIteratorWrapper#local_iterator.
  _form_global_array: _form_global_array().
  RemoteIteratorWrapper.cpu_devices: RemoteIteratorWrapper#cpu_devices.
  RemoteIteratorWrapper.cpu_sharding: RemoteIteratorWrapper#cpu_sharding.
  MultiHostDataLoadIterator.out_of_data: MultiHostDataLoadIterator#out_of_data.
  MultiHostDataLoadIterator.__iter__: MultiHostDataLoadIterator#__iter__().
  MultiHostDataLoadIterator.__next__: MultiHostDataLoadIterator#__next__().
  RemoteIteratorWrapper.reset: RemoteIteratorWrapper#reset().
  MultiHostDataLoadIterator.generate_padding_batch: MultiHostDataLoadIterator#generate_padding_batch.
  MultiHostDataLoadIterator.expansion_loading_factor_for_grain: MultiHostDataLoadIterator#expansion_loading_factor_for_grain.
  RemoteIterator.checkpoint_path: RemoteIterator#checkpoint_path.
  _build_global_shape_and_sharding: _build_global_shape_and_sharding().
  MultiHostDataLoadIterator.global_mesh: MultiHostDataLoadIterator#global_mesh.
  _colocated_cpu_devices: _colocated_cpu_devices().
  _colocated_cpu_mesh: _colocated_cpu_mesh().
  RemoteIterator: RemoteIterator#
  RemoteIterator.get_ds_fn: RemoteIterator#get_ds_fn.
  RemoteIterator.preprocessing_fn: RemoteIterator#preprocessing_fn.
  RemoteIterator.global_shape: RemoteIterator#global_shape.
  RemoteIterator.form_global_array_colocated_python: RemoteIterator#form_global_array_colocated_python().
  RemoteIteratorWrapper.tpu_sharding: RemoteIteratorWrapper#tpu_sharding.
  MultiHostDataLoadIterator: MultiHostDataLoadIterator#
  MultiHostDataLoadIterator.__init__: MultiHostDataLoadIterator#__init__().
  RemoteIterator.__init__: RemoteIterator#__init__().
  RemoteIteratorWrapper: RemoteIteratorWrapper#
  RemoteIteratorWrapper.__init__: RemoteIteratorWrapper#__init__().
  RemoteIteratorWrapper.__iter__: RemoteIteratorWrapper#__iter__().
---
# Module: [`src/maxtext/input_pipeline/multihost_dataloading.py`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py)

## Classes
### `MultiHostDataLoadIterator`
- def: [`src/maxtext/input_pipeline/multihost_dataloading.py:77`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L77)
- doc: fold get_next_batch_sharded into a iterator class.
- signature: `class MultiHostDataLoadIterator:`
- members:
  - `_get_next_batch_sharded(self)` — [`L119`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L119) — Splits the host loaded data equally over all devices.
  - `reset(self)` — [`L102`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L102)
  - `dataloader` — [`L90`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L90)
  - `expansion_loading_factor_for_grain` — [`L100`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L100)
  - `generate_padding_batch` — [`L99`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L99)
  - `global_mesh` — [`L89`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L89)
  - `last_local_data` — [`L98`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L98)
  - `local_iterator` — [`L92`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L92)
  - `out_of_data` — [`L97`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L97)
- protocol/private: `__init__`[`L82`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L82), `__iter__`[`L112`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L112), `__next__`[`L116`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L116), `_make_padding_batch`[`L164`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L164)
- uses (calls/refs, reference-scoped): [`_TF_RETRYABLE_ERRORS`](multihost_dataloading.md#_TF_RETRYABLE_ERRORS), [`_form_global_array`](multihost_dataloading.md#_form_global_array)

### `RemoteIterator`
- def: [`src/maxtext/input_pipeline/multihost_dataloading.py:182`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L182)
- doc: iterator class for using colocated python class
- signature: `class RemoteIterator:`
- members:
  - `form_global_array_colocated_python(path, array, devices, global_shape, sharding)` — [`L208`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L208)
  - `get_next(self, dummy_array)` — [`L204`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L204) — Gets the next batch of data and forms a global array.
  - `reset(self)` — [`L194`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L194)
  - `restore_state(self, step_array)` — [`L250`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L250)
  - `save_state(self, step_array)` — [`L230`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L230) — Saves the iterator state to a file.
  - `checkpoint_path` — [`L189`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L189)
  - `elastic` — [`L190`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L190)
  - `get_ds_fn` — [`L186`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L186)
  - `global_shape` — [`L188`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L188)
  - `iterator` — [`L198`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L198)
  - `preprocessing_fn` — [`L187`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L187)
- protocol/private: `__init__`[`L185`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L185)
- used by: [`dummy_array`](multihost_dataloading.md#RemoteIteratorWrapper.dummy_array)

### `RemoteIteratorWrapper`
- def: [`src/maxtext/input_pipeline/multihost_dataloading.py:262`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L262)
- doc: Wrapper for RemoteIterator that handles device placement.
- signature: `class RemoteIteratorWrapper:`
- members:
  - `reset(self)` — [`L287`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L287)
  - `restore_state(self, step)` — [`L301`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L301)
  - `save_state(self, step)` — [`L295`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L295)
  - `cpu_devices` — [`L266`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L266)
  - `cpu_mesh` — [`L267`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L267)
  - `cpu_sharding` — [`L269`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L269)
  - `dummy_array` — [`L270`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L270)
  - `local_iterator` — [`L275`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L275)
  - `tpu_sharding` — [`L268`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L268)
- protocol/private: `__init__`[`L265`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L265), `__iter__`[`L284`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L284), `__next__`[`L290`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L290)
- uses (calls/refs, reference-scoped): [`RemoteIterator`](multihost_dataloading.md#RemoteIterator), [`_colocated_cpu_devices`](multihost_dataloading.md#_colocated_cpu_devices), [`_colocated_cpu_mesh`](multihost_dataloading.md#_colocated_cpu_mesh)

## Functions
- `_build_global_shape_and_sharding(local_shape: tuple[int, ...], global_mesh: Mesh)` — [`L50`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L50)
- `_colocated_cpu_devices(devices: Sequence[jax.Device])` — [`L170`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L170) — Returns CPU devices colocated with the given devices.
- `_colocated_cpu_mesh(mesh: Mesh)` — [`L177`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L177) — Returns a CPU mesh that has colocated CPU devices.
- `_form_global_array(path, array: np.ndarray, global_mesh: Mesh)` — [`L60`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L60) — Put local sharded array into local devices

## Module values
- `_TF_RETRYABLE_ERRORS` — [`L32`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/multihost_dataloading.py#L32)

