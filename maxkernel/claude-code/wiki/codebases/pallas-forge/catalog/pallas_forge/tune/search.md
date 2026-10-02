---
title: 'Module: pallas_forge/tune/search.py'
type: catalog
provenance: extracted
module: pallas_forge/tune/search.py
status: fresh
symbol_base: scip-python python pallas-forge 0.0.0 `pallas_forge.tune.search`/
symbols:
  RandomSearch.generate: RandomSearch#generate().
  GridSearch.generate: GridSearch#generate().
  RandomSearch: RandomSearch#
  GridSearch: GridSearch#
  SearchStrategy.generate: SearchStrategy#generate().
  SearchStrategy: SearchStrategy#
  RandomSearch.n_trials: RandomSearch#n_trials.
  RandomSearch.seed: RandomSearch#seed.
  RandomSearch.__init__: RandomSearch#__init__().
---
# Module: [`pallas_forge/tune/search.py`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/search.py)

## Classes
### `GridSearch`
- def: [`pallas_forge/tune/search.py:24`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/search.py#L24)
- doc: Exhaustive search over all valid parameter combinations.
- signature: `class GridSearch:`
- members:
  - `generate(self, config: TuneConfig)` — [`L27`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/search.py#L27) — documented in [pallas_forge-tune-config](../../../concepts/pallas_forge-tune-config.md)
- uses (calls/refs, reference-scoped): [`TuneConfig`](config.md#TuneConfig), [`grid`](config.md#TuneConfig.grid)
- used by: [`tune`](__init__.md#tune)  (2 test-only)

### `RandomSearch`
- def: [`pallas_forge/tune/search.py:31`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/search.py#L31)
- doc: Random sampling from the configuration space.
- signature: `class RandomSearch:`
- members:
  - `generate(self, config: TuneConfig)` — [`L43`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/search.py#L43) — documented in [pallas_forge-tune-config](../../../concepts/pallas_forge-tune-config.md)
  - `n_trials` — [`L40`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/search.py#L40)
  - `seed` — [`L41`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/search.py#L41)
- protocol/private: `__init__`[`L39`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/search.py#L39)
- uses (calls/refs, reference-scoped): [`TuneConfig`](config.md#TuneConfig), [`sample`](config.md#TuneConfig.sample)
- used by: [`tune`](__init__.md#tune)  (3 test-only)

### `SearchStrategy`  ·  implements/extends Protocol
- def: [`pallas_forge/tune/search.py:14`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/search.py#L14)
- doc: Protocol for tuning search strategies.
- signature: `class SearchStrategy(Protocol):`
- members:
  - `generate(self, config: TuneConfig)` — [`L21`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/search.py#L21) — documented in [pallas_forge-tune-config](../../../concepts/pallas_forge-tune-config.md)
- uses (calls/refs, reference-scoped): [`TuneConfig`](config.md#TuneConfig)
- used by: [`tune`](__init__.md#tune)

