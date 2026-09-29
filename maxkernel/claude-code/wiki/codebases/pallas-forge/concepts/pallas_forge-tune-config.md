---
title: pallas_forge/tune/config — the auto-tuner's search-space definition
type: concept
provenance: mixed
concept: pallas_forge-tune-config
updated: 2026-07-03
status: fresh
---
# pallas_forge/tune/config — the auto-tuner's search-space definition

<!-- connect:up:begin -->
> **Cross-repo concept:** part of [autotuning](../../../concepts/autotuning.md) across this wiki's repos.
<!-- connect:up:end -->
Defines [`TuneConfig`](../catalog/pallas_forge/tune/config.md#TuneConfig), the declarative search-space object every `pallas-forge` benchmark script builds before calling `tune()` — a list of named parameters with candidate values, plus optional constraint predicates to prune invalid combinations, exposed either as an exhaustive grid or a reproducible random sample.

## Entry points
- [`TuneConfig.from_dict`](../catalog/pallas_forge/tune/config.md#TuneConfig.from_dict) — the common construction path (e.g. `{"block_m": [64,128,256], "block_k": [64,128,256], "block_n": [64,128,256]}` in [`main`](../catalog/benchmarks/bench_matmul.md#main)); wraps each key/value-list pair into a [`TuneParam`](../catalog/pallas_forge/tune/config.md#TuneParam).
- [`TuneConfig.grid`](../catalog/pallas_forge/tune/config.md#TuneConfig.grid) / [`TuneConfig.sample`](../catalog/pallas_forge/tune/config.md#TuneConfig.sample) — reached from [`GridSearch.generate`](../catalog/pallas_forge/tune/search.md#GridSearch.generate) and [`RandomSearch.generate`](../catalog/pallas_forge/tune/search.md#RandomSearch.generate) respectively, which [`tune`](../catalog/pallas_forge/tune/__init__.md#tune) dispatches to based on its `strategy` argument — these are the two concrete ways a `TuneConfig` becomes an actual list of configurations to benchmark.

## Mechanism (step-by-step)
1. [`TuneConfig.from_dict`](../catalog/pallas_forge/tune/config.md#TuneConfig.from_dict) builds one [`TuneParam`](../catalog/pallas_forge/tune/config.md#TuneParam) per dict key, and [`TuneConfig.from_yaml`](../catalog/pallas_forge/tune/config.md#TuneConfig.from_yaml) loads the same shape from a YAML file via `yaml.safe_load` before delegating to `from_dict` — both produce an identical [`TuneConfig`](../catalog/pallas_forge/tune/config.md#TuneConfig) regardless of source format.
2. [`TuneConfig.grid`](../catalog/pallas_forge/tune/config.md#TuneConfig.grid) takes the cartesian product of every [`TuneParam`](../catalog/pallas_forge/tune/config.md#TuneParam)'s [`values`](../catalog/pallas_forge/tune/config.md#TuneParam.values) list (via `itertools.product`), builds one dict per combination keyed by [`name`](../catalog/pallas_forge/tune/config.md#TuneParam.name), then filters through [`_is_valid`](../catalog/pallas_forge/tune/config.md#TuneConfig._is_valid) — which requires *every* registered [`constraints`](../catalog/pallas_forge/tune/config.md#TuneConfig.constraints) callable to return true for that combination — before returning the surviving list.
3. [`TuneConfig.sample`](../catalog/pallas_forge/tune/config.md#TuneConfig.sample) picks between two strategies based on space size: if [`total_combinations`](../catalog/pallas_forge/tune/config.md#TuneConfig.total_combinations) is small relative to the requested `n` (`<= n*2`), it just calls [`grid`](../catalog/pallas_forge/tune/config.md#TuneConfig.grid) and shuffles/truncates — exhaustive enumeration is cheaper than rejection sampling when the space is nearly as small as the sample size. Otherwise it does rejection sampling: draw a random combination per parameter, skip if already seen (deduplicated via a `set` of tuples) or invalid (checked with [`_is_valid`](../catalog/pallas_forge/tune/config.md#TuneConfig._is_valid)), and stop once `n` valid unique combos are collected or `max_attempts = n * 20` tries are exhausted.
4. [`GridSearch.generate`](../catalog/pallas_forge/tune/search.md#GridSearch.generate) is a one-line pass-through to [`TuneConfig.grid`](../catalog/pallas_forge/tune/config.md#TuneConfig.grid); [`RandomSearch.generate`](../catalog/pallas_forge/tune/search.md#RandomSearch.generate) is a one-line pass-through to [`TuneConfig.sample`](../catalog/pallas_forge/tune/config.md#TuneConfig.sample) using the strategy's own `n_trials`/`seed` — both search strategies are thin adapters over this file's two enumeration methods, and [`SearchStrategy`](../catalog/pallas_forge/tune/search.md#SearchStrategy.generate) is declared as a `typing.Protocol` so a caller can plug in a custom strategy without subclassing anything, as long as it exposes a matching `generate(config) -> list[dict]` method.
5. [`tune`](../catalog/pallas_forge/tune/__init__.md#tune) — the top-level auto-tuner entry point — normalizes whatever `config` argument it's given (a `TuneConfig`, a plain `dict`, or a YAML path) into a [`TuneConfig`](../catalog/pallas_forge/tune/config.md#TuneConfig) via [`from_dict`](../catalog/pallas_forge/tune/config.md#TuneConfig.from_dict)/[`from_yaml`](../catalog/pallas_forge/tune/config.md#TuneConfig.from_yaml), then calls the chosen search strategy's [`generate`](../catalog/pallas_forge/tune/search.md#SearchStrategy.generate) — this file's two enumeration methods are therefore the actual mechanism behind every `tune()` call regardless of which of the three shipped benchmark scripts ([`main`](../catalog/benchmarks/bench_matmul.md#main) for matmul, [`main`](../catalog/benchmarks/bench_swiglu.md#main) for SwiGLU, [`main`](../catalog/benchmarks/bench_rmsnorm.md#main) for RMSNorm) invokes it.

## Key data structures
- [`TuneParam`](../catalog/pallas_forge/tune/config.md#TuneParam) — a frozen-shape `@dataclass` of [`name`](../catalog/pallas_forge/tune/config.md#TuneParam.name) plus a [`values`](../catalog/pallas_forge/tune/config.md#TuneParam.values) list of `int | float | str`; its `__post_init__` (not itself in this packet's subgraph, but visible in source) raises `ValueError` if `values` is empty, so every tunable parameter is guaranteed at least one candidate before any enumeration runs.
- [`TuneConfig.params`](../catalog/pallas_forge/tune/config.md#TuneConfig.params) / [`TuneConfig.constraints`](../catalog/pallas_forge/tune/config.md#TuneConfig.constraints) — the two lists that fully define a search space: the parameters to vary, and the predicate functions (each `dict[str, Any] -> bool`) used to reject known-bad combinations (e.g. `bench_matmul.py`'s `config.add_constraint(lambda p: all(v >= 64 for v in p.values()))`).

## Dynamics (design intent)
> [!inferred] `sample`'s dual strategy (exhaustive-then-shuffle for small spaces, rejection sampling for large ones) exists to avoid the pathological cost of rejection sampling when the valid fraction of the space is large relative to `n` — for a space barely larger than `2n`, generating and filtering the whole grid is guaranteed-terminating and about as cheap as sampling would be anyway, whereas for a much larger space, materializing the full grid just to take `n` samples would waste memory and time the rejection-sampling path avoids.

## Edge cases
- [`TuneConfig.sample`](../catalog/pallas_forge/tune/config.md#TuneConfig.sample)'s rejection-sampling path can return **fewer than `n`** results if the valid fraction of the space is small enough that `max_attempts = n * 20` random draws don't turn up `n` valid unique combinations — this is a silent partial result, not an error.
- [`TuneConfig.grid`](../catalog/pallas_forge/tune/config.md#TuneConfig.grid) and [`TuneConfig.sample`](../catalog/pallas_forge/tune/config.md#TuneConfig.sample) both call [`_is_valid`](../catalog/pallas_forge/tune/config.md#TuneConfig._is_valid), which requires *all* constraints to pass (`all(c(config) for c in self.constraints)`) — constraints compose by AND, with no way to express an OR relationship between two constraint functions short of writing one combined lambda.

## Open questions
> [!inferred] Whether `sample`'s `max_attempts = n * 20` heuristic was empirically tuned or chosen arbitrarily isn't stated in source or docstrings — for a search space with a very low valid fraction (a possible outcome of a poorly-chosen constraint), this could return well under the requested `n` configs without any warning to the caller.

## See also
- [pallas_forge/tune/runner](pallas_forge-tune-runner.md) — consumes the list of config dicts this file produces, benchmarking each one with proper warmup/timing discipline.
- [pallas_forge/tune/report](pallas_forge-tune-report.md) — aggregates the benchmark results produced from these configs into rankings, exports, and heatmaps.
