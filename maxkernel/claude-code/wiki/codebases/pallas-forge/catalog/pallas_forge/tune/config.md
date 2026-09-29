---
title: 'Module: pallas_forge/tune/config.py'
type: catalog
provenance: extracted
module: pallas_forge/tune/config.py
status: fresh
symbol_base: scip-python python pallas-forge 0.0.0 `pallas_forge.tune.config`/Tune
symbols:
  TuneConfig: Config#
  TuneConfig.from_dict: Config#from_dict().
  TuneConfig.sample: Config#sample().
  TuneConfig.grid: Config#grid().
  TuneConfig.params: Config#params.
  TuneParam.name: Param#name.
  TuneParam.values: Param#values.
  TuneConfig.param_names: Config#param_names().
  TuneConfig.total_combinations: Config#total_combinations().
  TuneConfig.from_yaml: Config#from_yaml().
  TuneConfig.add_constraint: Config#add_constraint().
  TuneParam: Param#
  TuneParam.__post_init__: Param#__post_init__().
  TuneConfig._is_valid: Config#_is_valid().
  TuneConfig.constraints: Config#constraints.
---
# Module: [`pallas_forge/tune/config.py`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/config.py)

## Classes
### `TuneConfig`
- def: [`pallas_forge/tune/config.py:32`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/config.py#L32) — documented in [pallas_forge-tune-config](../../../concepts/pallas_forge-tune-config.md)
- doc: Defines the search space for auto-tuning.
- signature: `class TuneConfig:`
- members:
  - `add_constraint(self, fn: Callable[[dict[str, Any]], bool])` — [`L72`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/config.py#L72) — Add a constraint function. It receives a config dict and returns True if valid.
  - `from_dict(cls, d: dict[str, list])` — [`L53`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/config.py#L53) — Create a TuneConfig from a dict mapping param names to value lists. — documented in [pallas_forge-tune-config](../../../concepts/pallas_forge-tune-config.md)
  - `from_yaml(cls, path: str | Path)` — [`L59`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/config.py#L59) — Load a TuneConfig from a YAML file. — documented in [pallas_forge-tune-config](../../../concepts/pallas_forge-tune-config.md)
  - `grid(self)` — [`L91`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/config.py#L91) — Generate all valid parameter combinations (cartesian product filtered by constraints). — documented in [pallas_forge-tune-config](../../../concepts/pallas_forge-tune-config.md)
  - `param_names(self)` — [`L77`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/config.py#L77)
  - `sample(self, n: int, seed: int = 42)` — [`L99`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/config.py#L99) — Randomly sample n valid parameter combinations. — documented in [pallas_forge-tune-config](../../../concepts/pallas_forge-tune-config.md)
  - `total_combinations(self)` — [`L81`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/config.py#L81) — Total combinations before constraint filtering. — documented in [pallas_forge-tune-config](../../../concepts/pallas_forge-tune-config.md)
  - `constraints` — [`L50`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/config.py#L50) — documented in [pallas_forge-tune-config](../../../concepts/pallas_forge-tune-config.md)
  - `params` — [`L49`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/config.py#L49) — documented in [pallas_forge-tune-config](../../../concepts/pallas_forge-tune-config.md)
- protocol/private: `_is_valid`[`L88`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/config.py#L88)
- uses (calls/refs, reference-scoped): [`name`](config.md#TuneParam.name), [`values`](config.md#TuneParam.values), [`TuneParam`](config.md#TuneParam)
- used by: [`tune`](__init__.md#tune), [`generate`](search.md#RandomSearch.generate), [`generate`](search.md#GridSearch.generate), [`generate`](search.md#SearchStrategy.generate)  (16 test-only)

### `TuneParam`
- def: [`pallas_forge/tune/config.py:20`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/config.py#L20) — documented in [pallas_forge-tune-config](../../../concepts/pallas_forge-tune-config.md)
- doc: A single tunable parameter with its possible values.
- signature: `class TuneParam:`
- members:
  - `name` — [`L23`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/config.py#L23) — documented in [pallas_forge-tune-config](../../../concepts/pallas_forge-tune-config.md)
  - `values` — [`L24`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/config.py#L24) — documented in [pallas_forge-tune-config](../../../concepts/pallas_forge-tune-config.md)
- protocol/private: `__post_init__`[`L26`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/config.py#L26)
- used by: [`from_dict`](config.md#TuneConfig.from_dict), [`sample`](config.md#TuneConfig.sample), [`grid`](config.md#TuneConfig.grid), [`params`](config.md#TuneConfig.params), [`param_names`](config.md#TuneConfig.param_names), [`total_combinations`](config.md#TuneConfig.total_combinations)  (2 test-only)

