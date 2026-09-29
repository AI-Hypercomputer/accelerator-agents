---
title: 'Module: pallas_forge/tune/report.py'
type: catalog
provenance: extracted
module: pallas_forge/tune/report.py
status: fresh
symbol_base: scip-python python pallas-forge 0.0.0 `pallas_forge.tune.report`/TuneReport#
symbols:
  TuneReport.heatmap: heatmap().
  TuneReport.results: results.
  TuneReport.best: best().
  TuneReport: ''
  TuneReport.to_json: to_json().
  TuneReport.to_csv: to_csv().
  TuneReport.worst: worst().
  TuneReport.speedup_range: speedup_range().
  TuneReport.param_names: param_names.
  TuneReport.__init__: __init__().
---
# Module: [`pallas_forge/tune/report.py`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/report.py)

## Classes
### `TuneReport`
- def: [`pallas_forge/tune/report.py:18`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/report.py#L18) — documented in [pallas_forge-tune-report](../../../concepts/pallas_forge-tune-report.md)
- doc: Aggregates benchmark results and generates reports and visualizations.
- signature: `class TuneReport:`
- members:
  - `best(self, n: int = 1)` — [`L32`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/report.py#L32) — Return the top-n fastest configurations. — documented in [pallas_forge-tune-report](../../../concepts/pallas_forge-tune-report.md)
  - `heatmap(self, x_param: str, y_param: str, metric: str = "median_ms", *, title: str | None = None, save_path: str | Path | None = None, figsize: tuple[int, int] = (10, 8), cmap: str = "YlOrRd_r", annotate: bool = True)` — [`L83`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/report.py#L83) — Generate a 2D heatmap showing how two parameters affect performance. — documented in [pallas_forge-tune-report](../../../concepts/pallas_forge-tune-report.md)
  - `speedup_range(self)` — [`L43`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/report.py#L43) — Ratio of slowest to fastest median time. — documented in [pallas_forge-tune-report](../../../concepts/pallas_forge-tune-report.md)
  - `to_csv(self, path: str | Path)` — [`L58`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/report.py#L58) — Export results to CSV. — documented in [pallas_forge-tune-report](../../../concepts/pallas_forge-tune-report.md)
  - `to_json(self, path: str | Path)` — [`L50`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/report.py#L50) — Export results to JSON. — documented in [pallas_forge-tune-report](../../../concepts/pallas_forge-tune-report.md)
  - `worst(self, n: int = 1)` — [`L37`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/report.py#L37) — Return the n slowest configurations. — documented in [pallas_forge-tune-report](../../../concepts/pallas_forge-tune-report.md)
  - `param_names` — [`L30`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/report.py#L30) — documented in [pallas_forge-tune-report](../../../concepts/pallas_forge-tune-report.md)
  - `results` — [`L27`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/report.py#L27) — documented in [pallas_forge-tune-report](../../../concepts/pallas_forge-tune-report.md)
- protocol/private: `__init__`[`L26`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/report.py#L26)
- uses (calls/refs, reference-scoped): [`median_ms`](runner.md#BenchmarkResult.median_ms), [`to_dict`](runner.md#BenchmarkResult.to_dict), [`BenchmarkResult`](runner.md#BenchmarkResult), [`config`](runner.md#BenchmarkResult.config)
- used by: [`tune`](__init__.md#tune)  (12 test-only)

