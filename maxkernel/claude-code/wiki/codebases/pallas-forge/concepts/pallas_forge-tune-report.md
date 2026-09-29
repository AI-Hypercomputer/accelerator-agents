---
title: pallas_forge/tune/report — rankings, exports, and heatmaps over a tuning sweep
type: concept
provenance: mixed
concept: pallas_forge-tune-report
updated: 2026-07-03
status: fresh
---
# pallas_forge/tune/report — rankings, exports, and heatmaps over a tuning sweep
[`TuneReport`](../catalog/pallas_forge/tune/report.md#TuneReport) is the object every `pallas-forge` `tune()` call returns — it wraps a completed sweep's [`BenchmarkResult`](../catalog/pallas_forge/tune/runner.md#BenchmarkResult) list with best/worst lookups, a headline "speedup range" statistic, JSON/CSV export, and — per the module docstring — the heatmap visualization that is "the visual centerpiece" making a block-size sweep's performance spread "tangible."

## Entry points
- [`TuneReport`](../catalog/pallas_forge/tune/report.md#TuneReport) — constructed once per sweep by [`tune`](../catalog/pallas_forge/tune/__init__.md#tune) from the benchmark runner's sorted results; if `param_names` isn't given explicitly it infers them from the first result's config keys.
- [`TuneReport.heatmap`](../catalog/pallas_forge/tune/report.md#TuneReport.heatmap) — the visualization entry point a caller invokes after a sweep to produce the block-size-vs-performance figure referenced in the library's README.
- [`TuneReport.best`](../catalog/pallas_forge/tune/report.md#TuneReport.best) / [`TuneReport.worst`](../catalog/pallas_forge/tune/report.md#TuneReport.worst) — the ranking accessors [`tune`](../catalog/pallas_forge/tune/__init__.md#tune) itself calls (`report.best(1)[0]`) to print the winning configuration at the end of a sweep.

## Mechanism (step-by-step)
1. [`TuneReport.best`](../catalog/pallas_forge/tune/report.md#TuneReport.best) and [`TuneReport.worst`](../catalog/pallas_forge/tune/report.md#TuneReport.worst) both sort [`results`](../catalog/pallas_forge/tune/report.md#TuneReport.results) by [`median_ms`](../catalog/pallas_forge/tune/runner.md#BenchmarkResult.median_ms) (ascending / descending respectively) and slice the top `n` — since the benchmark runner already returns results pre-sorted fastest-first (visible in the surrounding source), these re-sorts are redundant work for the common case but make `TuneReport` correct even if constructed from an out-of-order or concatenated results list.
2. [`TuneReport.speedup_range`](../catalog/pallas_forge/tune/report.md#TuneReport.speedup_range) computes `max(times) / max(min(times), 1e-9)` over every result's [`median_ms`](../catalog/pallas_forge/tune/runner.md#BenchmarkResult.median_ms) — the `1e-9` floor guards against a division by zero if the fastest config happened to measure as exactly `0.0` ms, and this ratio is the headline number [`tune`](../catalog/pallas_forge/tune/__init__.md#tune) prints as "Speedup range: {x}x" — quantifying how much a block-size choice matters for this kernel, independent of any XLA-baseline comparison.
3. [`TuneReport.to_json`](../catalog/pallas_forge/tune/report.md#TuneReport.to_json) and [`TuneReport.to_csv`](../catalog/pallas_forge/tune/report.md#TuneReport.to_csv) both call [`BenchmarkResult.to_dict`](../catalog/pallas_forge/tune/runner.md#BenchmarkResult.to_dict) on every result before writing; `to_csv`'s own docstring states its fieldname strategy explicitly — "built from the union of keys across all rows, preserving first-seen order... heterogeneous configs... export cleanly instead of raising a ValueError" — directly addressing the case where different results in one report carry different optional fields (e.g. some with `tflops`, some without).
4. [`TuneReport.heatmap`](../catalog/pallas_forge/tune/report.md#TuneReport.heatmap) validates that both requested axis parameters are in [`param_names`](../catalog/pallas_forge/tune/report.md#TuneReport.param_names), collects the sorted unique values along each axis, then builds a 2D grid where cell `(x, y)` takes the *best* value among all results sharing that `(x_param, y_param)` pair — "best" means `max` for throughput metrics (`tflops`, `bandwidth_gb_s`) and `min` for time metrics, so a 3rd/4th tuned parameter beyond the two plotted axes is implicitly optimized-over rather than averaged.
5. The rest of [`TuneReport.heatmap`](../catalog/pallas_forge/tune/report.md#TuneReport.heatmap) renders the grid via `matplotlib.pyplot.imshow` with a configurable `cmap` (default `"YlOrRd_r"`, chosen so red=slow/yellow=fast reads intuitively), optional per-cell numeric annotation, and an optional `save_path` — `matplotlib`/`numpy` are imported lazily inside the method body so `matplotlib` is not a hard dependency of the rest of the package (consistent with the README's `pip install "pallas-forge[viz]"` optional extra).

## Key data structures
- [`TuneReport.results`](../catalog/pallas_forge/tune/report.md#TuneReport.results) — the raw list of [`BenchmarkResult`](../catalog/pallas_forge/tune/runner.md#BenchmarkResult) the report wraps; every other method on this class is a view or transformation over this one list.
- [`TuneReport.param_names`](../catalog/pallas_forge/tune/report.md#TuneReport.param_names) — the ordered list of tunable parameter names used both for `heatmap`'s axis validation and, when not explicitly passed to the constructor, inferred from `results[0].config.keys()`.

## Dynamics (design intent)
> [!inferred] The heatmap method's per-cell "take the best over any un-plotted dimensions" reduction is what lets a caller sweep three parameters (say `block_m`, `block_k`, `block_n`) but visualize only two at a time — each cell answers "what's the best achievable time/throughput at this (x,y) if the third parameter is free to vary," which is the natural question when the third parameter has a comparatively small effect (or when the report author simply wants a 2D slice of a higher-dimensional sweep).

## Edge cases
- [`TuneReport.to_csv`](../catalog/pallas_forge/tune/report.md#TuneReport.to_csv) and the implicit `param_names` inference in [`TuneReport`](../catalog/pallas_forge/tune/report.md#TuneReport)'s constructor both silently no-op / default to empty on an empty `results` list — `to_csv` returns immediately without writing a file at all (not even a header row), so a caller checking for the CSV's existence after an all-failed sweep would find nothing.
- [`TuneReport.heatmap`](../catalog/pallas_forge/tune/report.md#TuneReport.heatmap) raises `ValueError` if either axis parameter isn't in [`param_names`](../catalog/pallas_forge/tune/report.md#TuneReport.param_names) — this only catches a typo'd or nonexistent parameter name, not the case where the parameter exists but every result happens to share the same single value for it (which would still plot, just as a degenerate 1-row-or-column heatmap).

## Open questions
> [!inferred] Whether `speedup_range`'s ratio is meant to be compared against pallas-forge's other headline metric — the Pallas-vs-XLA speedup reported by each benchmark script's `xla_baseline` comparison — or is purely an internal tuning-sensitivity signal isn't stated; the two numbers answer different questions (how much does block-size choice matter, vs. does the custom kernel beat XLA at all) but both live in the same README table.

## See also
- [pallas_forge/tune/runner](pallas_forge-tune-runner.md) — produces the `BenchmarkResult` list this file aggregates, and defines the `to_dict` flattening this file's exports depend on.
- [pallas_forge/tune/config](pallas_forge-tune-config.md) — defines the search space whose parameter names flow through to `param_names` and the heatmap's axis labels.
