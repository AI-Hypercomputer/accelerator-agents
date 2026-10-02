---
title: 'Module: tests/test_tune_config.py'
type: catalog
provenance: extracted
module: tests/test_tune_config.py
status: fresh
symbol_base: scip-python python pallas-forge 0.0.0 `tests.test_tune_config`/TestTune
symbols:
  TestTuneConfig.test_from_dict: Config#test_from_dict().
  TestTuneConfig.test_constraints_filter: Config#test_constraints_filter().
  TestTuneConfig.test_from_yaml: Config#test_from_yaml().
  TestTuneParam.test_basic: Param#test_basic().
  TestTuneParam.test_empty_values_raises: Param#test_empty_values_raises().
  TestTuneConfig.test_grid_exhaustive: Config#test_grid_exhaustive().
  TestTuneConfig.test_total_combinations: Config#test_total_combinations().
  TestTuneConfig.test_sample_count: Config#test_sample_count().
  TestTuneConfig.test_sample_reproducibility: Config#test_sample_reproducibility().
  TestTuneConfig.test_single_param: Config#test_single_param().
  TestTuneParam: Param#
  TestTuneConfig: Config#
---
# Module: [`tests/test_tune_config.py`](../../../../../raw/code/pallas-forge/tests/test_tune_config.py)

## Classes
### `TestTuneConfig`
- def: [`tests/test_tune_config.py:24`](../../../../../raw/code/pallas-forge/tests/test_tune_config.py#L24)
- signature: `class TestTuneConfig:`
- members:
  - `test_constraints_filter(self)` — [`L56`](../../../../../raw/code/pallas-forge/tests/test_tune_config.py#L56)
  - `test_from_dict(self)` — [`L25`](../../../../../raw/code/pallas-forge/tests/test_tune_config.py#L25)
  - `test_from_yaml(self)` — [`L89`](../../../../../raw/code/pallas-forge/tests/test_tune_config.py#L89)
  - `test_grid_exhaustive(self)` — [`L35`](../../../../../raw/code/pallas-forge/tests/test_tune_config.py#L35)
  - `test_sample_count(self)` — [`L68`](../../../../../raw/code/pallas-forge/tests/test_tune_config.py#L68)
  - `test_sample_reproducibility(self)` — [`L78`](../../../../../raw/code/pallas-forge/tests/test_tune_config.py#L78)
  - `test_single_param(self)` — [`L102`](../../../../../raw/code/pallas-forge/tests/test_tune_config.py#L102)
  - `test_total_combinations(self)` — [`L47`](../../../../../raw/code/pallas-forge/tests/test_tune_config.py#L47)
- uses (calls/refs, reference-scoped): [`TuneConfig`](../pallas_forge/tune/config.md#TuneConfig), [`from_dict`](../pallas_forge/tune/config.md#TuneConfig.from_dict), [`sample`](../pallas_forge/tune/config.md#TuneConfig.sample), [`grid`](../pallas_forge/tune/config.md#TuneConfig.grid), [`params`](../pallas_forge/tune/config.md#TuneConfig.params), [`param_names`](../pallas_forge/tune/config.md#TuneConfig.param_names), [`total_combinations`](../pallas_forge/tune/config.md#TuneConfig.total_combinations), [`from_yaml`](../pallas_forge/tune/config.md#TuneConfig.from_yaml), [`add_constraint`](../pallas_forge/tune/config.md#TuneConfig.add_constraint)

### `TestTuneParam`
- def: [`tests/test_tune_config.py:13`](../../../../../raw/code/pallas-forge/tests/test_tune_config.py#L13)
- signature: `class TestTuneParam:`
- members:
  - `test_basic(self)` — [`L14`](../../../../../raw/code/pallas-forge/tests/test_tune_config.py#L14)
  - `test_empty_values_raises(self)` — [`L19`](../../../../../raw/code/pallas-forge/tests/test_tune_config.py#L19)
- uses (calls/refs, reference-scoped): [`name`](../pallas_forge/tune/config.md#TuneParam.name), [`values`](../pallas_forge/tune/config.md#TuneParam.values), [`TuneParam`](../pallas_forge/tune/config.md#TuneParam)

