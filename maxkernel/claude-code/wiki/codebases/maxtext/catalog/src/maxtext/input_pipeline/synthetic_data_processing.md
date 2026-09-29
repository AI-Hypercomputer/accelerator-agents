---
title: 'Module: src/maxtext/input_pipeline/synthetic_data_processing.py'
type: catalog
provenance: extracted
module: src/maxtext/input_pipeline/synthetic_data_processing.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.input_pipeline.synthetic_data_processing`/
symbols:
  SyntheticDataIterator.__next__: SyntheticDataIterator#__next__().
  SyntheticDataIterator.config: SyntheticDataIterator#config.
  PlaceHolderDataIterator.mesh: PlaceHolderDataIterator#mesh.
  PlaceHolderDataIterator.data_generator: PlaceHolderDataIterator#data_generator.
  PlaceHolderDataIterator.get_place_holder_synthetic_data: PlaceHolderDataIterator#get_place_holder_synthetic_data().
  PlaceHolderDataIterator.__iter__: PlaceHolderDataIterator#__iter__().
  PlaceHolderDataIterator.__next__: PlaceHolderDataIterator#__next__().
  SyntheticDataIterator.data_generator: SyntheticDataIterator#data_generator.
  SyntheticDataIterator: SyntheticDataIterator#
  SyntheticDataIterator.mesh: SyntheticDataIterator#mesh.
  SyntheticDataIterator.data: SyntheticDataIterator#data.
  SyntheticDataIterator.raw_generate_synthetic_data: SyntheticDataIterator#raw_generate_synthetic_data().
  PlaceHolderDataIterator: PlaceHolderDataIterator#
  PlaceHolderDataIterator.infinite_iterator: PlaceHolderDataIterator#infinite_iterator().
  SyntheticDataIterator.__init__: SyntheticDataIterator#__init__().
  SyntheticDataIterator.reset: SyntheticDataIterator#reset().
  SyntheticDataIterator.__iter__: SyntheticDataIterator#__iter__().
  PlaceHolderDataIterator.__init__: PlaceHolderDataIterator#__init__().
  PlaceHolderDataIterator.reset: PlaceHolderDataIterator#reset().
---
# Module: [`src/maxtext/input_pipeline/synthetic_data_processing.py`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/synthetic_data_processing.py)

## Classes
### `PlaceHolderDataIterator`
- def: [`src/maxtext/input_pipeline/synthetic_data_processing.py:84`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/synthetic_data_processing.py#L84)
- doc: Creates a place holder synthetic data iterator for loading on subset of hosts
- signature: `class PlaceHolderDataIterator:`
- members:
  - `get_place_holder_synthetic_data(config: pyconfig.HyperParameters)` — [`L102`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/synthetic_data_processing.py#L102) — fill negative value in synthetic data
  - `infinite_iterator()` — [`L115`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/synthetic_data_processing.py#L115)
  - `reset(self)` — [`L98`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/synthetic_data_processing.py#L98)
  - `data_generator` — [`L90`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/synthetic_data_processing.py#L90)
  - `mesh` — [`L88`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/synthetic_data_processing.py#L88)
- protocol/private: `__init__`[`L87`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/synthetic_data_processing.py#L87), `__iter__`[`L92`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/synthetic_data_processing.py#L92), `__next__`[`L95`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/synthetic_data_processing.py#L95)

### `SyntheticDataIterator`
- def: [`src/maxtext/input_pipeline/synthetic_data_processing.py:31`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/synthetic_data_processing.py#L31)
- doc: Creates a synthetic data iterator for performance testing work
- signature: `class SyntheticDataIterator:`
- members:
  - `raw_generate_synthetic_data(config: pyconfig.HyperParameters, data)` — [`L70`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/synthetic_data_processing.py#L70) — Generates a single batch of synthetic data
  - `reset(self)` — [`L59`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/synthetic_data_processing.py#L59)
  - `config` — [`L38`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/synthetic_data_processing.py#L38)
  - `data` — [`L57`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/synthetic_data_processing.py#L57)
  - `data_generator` — [`L34`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/synthetic_data_processing.py#L34)
  - `mesh` — [`L37`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/synthetic_data_processing.py#L37)
- protocol/private: `__init__`[`L36`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/synthetic_data_processing.py#L36), `__iter__`[`L62`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/synthetic_data_processing.py#L62), `__next__`[`L65`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/synthetic_data_processing.py#L65)

