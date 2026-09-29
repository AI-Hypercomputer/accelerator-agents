---
title: 'Module: tests/test_tune_search.py'
type: catalog
provenance: extracted
module: tests/test_tune_search.py
status: fresh
symbol_base: scip-python python pallas-forge 0.0.0 `tests.test_tune_search`/Test
symbols:
  TestGridSearch.test_respects_constraints: GridSearch#test_respects_constraints().
  TestGridSearch.test_returns_all_combos: GridSearch#test_returns_all_combos().
  TestRandomSearch.test_returns_requested_count: RandomSearch#test_returns_requested_count().
  TestRandomSearch.test_reproducible: RandomSearch#test_reproducible().
  TestRandomSearch.test_different_seeds_differ: RandomSearch#test_different_seeds_differ().
  TestGridSearch: GridSearch#
  TestRandomSearch: RandomSearch#
---
# Module: [`tests/test_tune_search.py`](../../../../../raw/code/pallas-forge/tests/test_tune_search.py)

## Classes
### `TestGridSearch`
- def: [`tests/test_tune_search.py:9`](../../../../../raw/code/pallas-forge/tests/test_tune_search.py#L9)
- signature: `class TestGridSearch:`
- members:
  - `test_respects_constraints(self)` — [`L16`](../../../../../raw/code/pallas-forge/tests/test_tune_search.py#L16)
  - `test_returns_all_combos(self)` — [`L10`](../../../../../raw/code/pallas-forge/tests/test_tune_search.py#L10)
- uses (calls/refs, reference-scoped): [`TuneConfig`](../pallas_forge/tune/config.md#TuneConfig), [`from_dict`](../pallas_forge/tune/config.md#TuneConfig.from_dict), [`generate`](../pallas_forge/tune/search.md#GridSearch.generate), [`add_constraint`](../pallas_forge/tune/config.md#TuneConfig.add_constraint), [`GridSearch`](../pallas_forge/tune/search.md#GridSearch)

### `TestRandomSearch`
- def: [`tests/test_tune_search.py:24`](../../../../../raw/code/pallas-forge/tests/test_tune_search.py#L24)
- signature: `class TestRandomSearch:`
- members:
  - `test_different_seeds_differ(self)` — [`L37`](../../../../../raw/code/pallas-forge/tests/test_tune_search.py#L37)
  - `test_reproducible(self)` — [`L31`](../../../../../raw/code/pallas-forge/tests/test_tune_search.py#L31)
  - `test_returns_requested_count(self)` — [`L25`](../../../../../raw/code/pallas-forge/tests/test_tune_search.py#L25)
- uses (calls/refs, reference-scoped): [`TuneConfig`](../pallas_forge/tune/config.md#TuneConfig), [`from_dict`](../pallas_forge/tune/config.md#TuneConfig.from_dict), [`generate`](../pallas_forge/tune/search.md#RandomSearch.generate), [`RandomSearch`](../pallas_forge/tune/search.md#RandomSearch)

