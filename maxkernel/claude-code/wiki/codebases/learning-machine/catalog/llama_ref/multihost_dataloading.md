---
title: 'Module: llama_ref/multihost_dataloading.py'
type: catalog
provenance: extracted
module: llama_ref/multihost_dataloading.py
status: fresh
symbol_base: scip-python python learning-machine 0.0.0 `llama_ref.multihost_dataloading`/
symbols:
  MultiHostDataLoadIterator.dataloader: MultiHostDataLoadIterator#dataloader.
  MultiHostDataLoadIterator.local_iterator: MultiHostDataLoadIterator#local_iterator.
  MultiHostDataLoadIterator.__next__: MultiHostDataLoadIterator#__next__().
  MultiHostDataLoadIterator.reset: MultiHostDataLoadIterator#reset().
  _form_global_array: _form_global_array().
  get_next_batch_sharded: get_next_batch_sharded().
  MultiHostDataLoadIterator.__iter__: MultiHostDataLoadIterator#__iter__().
  _build_global_shape_and_sharding: _build_global_shape_and_sharding().
  MultiHostDataLoadIterator.global_mesh: MultiHostDataLoadIterator#global_mesh.
  MultiHostDataLoadIterator: MultiHostDataLoadIterator#
  MultiHostDataLoadIterator.__init__: MultiHostDataLoadIterator#__init__().
---
# Module: [`llama_ref/multihost_dataloading.py`](../../../../../raw/code/learning-machine/llama_ref/multihost_dataloading.py)

## Classes
### `MultiHostDataLoadIterator`
- def: [`llama_ref/multihost_dataloading.py:92`](../../../../../raw/code/learning-machine/llama_ref/multihost_dataloading.py#L92)
- doc: fold get_next_batch_sharded into a iterator class
- signature: `class MultiHostDataLoadIterator:`
- members:
  - `reset(self)` — [`L105`](../../../../../raw/code/learning-machine/llama_ref/multihost_dataloading.py#L105)
  - `dataloader` — [`L97`](../../../../../raw/code/learning-machine/llama_ref/multihost_dataloading.py#L97)
  - `global_mesh` — [`L96`](../../../../../raw/code/learning-machine/llama_ref/multihost_dataloading.py#L96)
  - `local_iterator` — [`L99`](../../../../../raw/code/learning-machine/llama_ref/multihost_dataloading.py#L99)
- protocol/private: `__init__`[`L95`](../../../../../raw/code/learning-machine/llama_ref/multihost_dataloading.py#L95), `__iter__`[`L113`](../../../../../raw/code/learning-machine/llama_ref/multihost_dataloading.py#L113), `__next__`[`L117`](../../../../../raw/code/learning-machine/llama_ref/multihost_dataloading.py#L117)
- uses (calls/refs, reference-scoped): [`get_next_batch_sharded`](multihost_dataloading.md#get_next_batch_sharded)

## Functions
- `_build_global_shape_and_sharding(local_shape: tuple[int, ...], global_mesh: Mesh)` — [`L39`](../../../../../raw/code/learning-machine/llama_ref/multihost_dataloading.py#L39)
- `_form_global_array(path, array: np.ndarray, global_mesh: Mesh)` — [`L49`](../../../../../raw/code/learning-machine/llama_ref/multihost_dataloading.py#L49) — Put local sharded array into local devices
- `get_next_batch_sharded(local_iterator: Iterator, global_mesh: Mesh)` — [`L66`](../../../../../raw/code/learning-machine/llama_ref/multihost_dataloading.py#L66) — Splits the host loaded data equally over all devices.

