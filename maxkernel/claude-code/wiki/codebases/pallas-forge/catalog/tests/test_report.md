---
title: 'Module: tests/test_report.py'
type: catalog
provenance: extracted
module: tests/test_report.py
status: fresh
symbol_base: scip-python python pallas-forge 0.0.0 `tests.test_report`/
symbols:
  _make_results: _make_results().
  TestTuneReport.test_best: TestTuneReport#test_best().
  TestTuneReport.test_worst: TestTuneReport#test_worst().
  TestTuneReport.test_speedup_range: TestTuneReport#test_speedup_range().
  TestTuneReport.test_to_json: TestTuneReport#test_to_json().
  TestTuneReport.test_to_csv: TestTuneReport#test_to_csv().
  TestTuneReport.test_heatmap_creates_figure: TestTuneReport#test_heatmap_creates_figure().
  TestTuneReport.test_heatmap_save: TestTuneReport#test_heatmap_save().
  TestTuneReport.test_heatmap_invalid_param: TestTuneReport#test_heatmap_invalid_param().
  TestTuneReport.test_empty_results: TestTuneReport#test_empty_results().
  TestTuneReport: TestTuneReport#
---
# Module: [`tests/test_report.py`](../../../../../raw/code/pallas-forge/tests/test_report.py)

## Classes
### `TestTuneReport`
- def: [`tests/test_report.py:51`](../../../../../raw/code/pallas-forge/tests/test_report.py#L51)
- signature: `class TestTuneReport:`
- members:
  - `test_best(self)` — [`L52`](../../../../../raw/code/pallas-forge/tests/test_report.py#L52)
  - `test_empty_results(self)` — [`L130`](../../../../../raw/code/pallas-forge/tests/test_report.py#L130)
  - `test_heatmap_creates_figure(self)` — [`L96`](../../../../../raw/code/pallas-forge/tests/test_report.py#L96) — Heatmap should return a matplotlib Figure.
  - `test_heatmap_invalid_param(self)` — [`L118`](../../../../../raw/code/pallas-forge/tests/test_report.py#L118) — Invalid parameter name should raise ValueError.
  - `test_heatmap_save(self)` — [`L106`](../../../../../raw/code/pallas-forge/tests/test_report.py#L106) — Heatmap should save to file.
  - `test_speedup_range(self)` — [`L64`](../../../../../raw/code/pallas-forge/tests/test_report.py#L64)
  - `test_to_csv(self)` — [`L83`](../../../../../raw/code/pallas-forge/tests/test_report.py#L83)
  - `test_to_json(self)` — [`L68`](../../../../../raw/code/pallas-forge/tests/test_report.py#L68)
  - `test_worst(self)` — [`L58`](../../../../../raw/code/pallas-forge/tests/test_report.py#L58)
- uses (calls/refs, reference-scoped): [`median_ms`](../pallas_forge/tune/runner.md#BenchmarkResult.median_ms), [`heatmap`](../pallas_forge/tune/report.md#TuneReport.heatmap), [`best`](../pallas_forge/tune/report.md#TuneReport.best), [`TuneReport`](../pallas_forge/tune/report.md#TuneReport), [`to_csv`](../pallas_forge/tune/report.md#TuneReport.to_csv), [`to_json`](../pallas_forge/tune/report.md#TuneReport.to_json), [`worst`](../pallas_forge/tune/report.md#TuneReport.worst), [`speedup_range`](../pallas_forge/tune/report.md#TuneReport.speedup_range)  (1 test-only)

## Functions
- `_make_results()` — [`L13`](../../../../../raw/code/pallas-forge/tests/test_report.py#L13) — Create sample results for testing.

