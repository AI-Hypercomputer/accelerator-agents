---
title: 'Module: pallas_forge/tune/runner.py'
type: catalog
provenance: extracted
module: pallas_forge/tune/runner.py
status: fresh
symbol_base: scip-python python pallas-forge 0.0.0 `pallas_forge.tune.runner`/BenchmarkR
symbols:
  BenchmarkRunner.run_single: unner#run_single().
  BenchmarkResult.median_ms: esult#median_ms.
  BenchmarkResult.to_dict: esult#to_dict().
  BenchmarkResult: esult#
  BenchmarkResult.config: esult#config.
  BenchmarkRunner: unner#
  BenchmarkRunner.run_all: unner#run_all().
  BenchmarkResult.mean_ms: esult#mean_ms.
  BenchmarkResult.std_ms: esult#std_ms.
  BenchmarkResult.min_ms: esult#min_ms.
  BenchmarkResult.max_ms: esult#max_ms.
  BenchmarkResult.tflops: esult#tflops.
  BenchmarkResult.bandwidth_gb_s: esult#bandwidth_gb_s.
  BenchmarkRunner.kernel_fn: unner#kernel_fn.
  BenchmarkRunner.flops_fn: unner#flops_fn.
  BenchmarkRunner.bytes_fn: unner#bytes_fn.
  BenchmarkResult.all_times_ms: esult#all_times_ms.
  BenchmarkRunner.input_fn: unner#input_fn.
  BenchmarkRunner.n_warmup: unner#n_warmup.
  BenchmarkRunner.n_repeat: unner#n_repeat.
  BenchmarkRunner.__init__: unner#__init__().
---
# Module: [`pallas_forge/tune/runner.py`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/runner.py)

## Classes
### `BenchmarkResult`
- def: [`pallas_forge/tune/runner.py:21`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/runner.py#L21) — documented in [pallas_forge-tune-report](../../../concepts/pallas_forge-tune-report.md)
- doc: Result of benchmarking a single kernel configuration.
- signature: `class BenchmarkResult:`
- members:
  - `to_dict(self)` — [`L37`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/runner.py#L37) — Convert to a flat dictionary for JSON/CSV export. — documented in [pallas_forge-tune-report](../../../concepts/pallas_forge-tune-report.md)
  - `all_times_ms` — [`L33`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/runner.py#L33) — documented in [pallas_forge-tune-runner](../../../concepts/pallas_forge-tune-runner.md)
  - `bandwidth_gb_s` — [`L35`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/runner.py#L35) — documented in [pallas_forge-tune-runner](../../../concepts/pallas_forge-tune-runner.md)
  - `config` — [`L27`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/runner.py#L27) — documented in [pallas_forge-tune-runner](../../../concepts/pallas_forge-tune-runner.md)
  - `max_ms` — [`L32`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/runner.py#L32) — documented in [pallas_forge-tune-runner](../../../concepts/pallas_forge-tune-runner.md)
  - `mean_ms` — [`L29`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/runner.py#L29) — documented in [pallas_forge-tune-runner](../../../concepts/pallas_forge-tune-runner.md)
  - `median_ms` — [`L28`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/runner.py#L28) — documented in [pallas_forge-tune-report](../../../concepts/pallas_forge-tune-report.md)
  - `min_ms` — [`L31`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/runner.py#L31) — documented in [pallas_forge-tune-runner](../../../concepts/pallas_forge-tune-runner.md)
  - `std_ms` — [`L30`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/runner.py#L30) — documented in [pallas_forge-tune-runner](../../../concepts/pallas_forge-tune-runner.md)
  - `tflops` — [`L34`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/runner.py#L34) — documented in [pallas_forge-tune-runner](../../../concepts/pallas_forge-tune-runner.md)
- used by: [`run_single`](runner.md#BenchmarkRunner.run_single), [`tune`](__init__.md#tune), [`heatmap`](report.md#TuneReport.heatmap), [`results`](report.md#TuneReport.results), [`best`](report.md#TuneReport.best), [`run_all`](runner.md#BenchmarkRunner.run_all), [`to_csv`](report.md#TuneReport.to_csv), [`to_json`](report.md#TuneReport.to_json), [`worst`](report.md#TuneReport.worst), [`speedup_range`](report.md#TuneReport.speedup_range), [`__init__`](report.md#TuneReport.__init__)  (15 test-only)

### `BenchmarkRunner`
- def: [`pallas_forge/tune/runner.py:56`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/runner.py#L56)
- doc: Runs benchmarks for kernel configurations with proper timing methodology.
- signature: `class BenchmarkRunner:`
- members:
  - `run_all(self, configs: list[dict[str, Any]], *, verbose: bool = True)` — [`L135`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/runner.py#L135) — Benchmark all configurations. — documented in [pallas_forge-tune-runner](../../../concepts/pallas_forge-tune-runner.md)
  - `run_single(self, config: dict[str, Any])` — [`L85`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/runner.py#L85) — Benchmark a single configuration. — documented in [pallas_forge-tune-runner](../../../concepts/pallas_forge-tune-runner.md)
  - `bytes_fn` — [`L83`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/runner.py#L83) — documented in [pallas_forge-tune-runner](../../../concepts/pallas_forge-tune-runner.md)
  - `flops_fn` — [`L82`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/runner.py#L82) — documented in [pallas_forge-tune-runner](../../../concepts/pallas_forge-tune-runner.md)
  - `input_fn` — [`L79`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/runner.py#L79) — documented in [pallas_forge-tune-runner](../../../concepts/pallas_forge-tune-runner.md)
  - `kernel_fn` — [`L78`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/runner.py#L78) — documented in [pallas_forge-tune-runner](../../../concepts/pallas_forge-tune-runner.md)
  - `n_repeat` — [`L81`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/runner.py#L81) — documented in [pallas_forge-tune-runner](../../../concepts/pallas_forge-tune-runner.md)
  - `n_warmup` — [`L80`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/runner.py#L80) — documented in [pallas_forge-tune-runner](../../../concepts/pallas_forge-tune-runner.md)
- protocol/private: `__init__`[`L68`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/runner.py#L68)
- uses (calls/refs, reference-scoped): [`median_ms`](runner.md#BenchmarkResult.median_ms), [`BenchmarkResult`](runner.md#BenchmarkResult), [`config`](runner.md#BenchmarkResult.config), [`mean_ms`](runner.md#BenchmarkResult.mean_ms), [`std_ms`](runner.md#BenchmarkResult.std_ms), [`max_ms`](runner.md#BenchmarkResult.max_ms), [`min_ms`](runner.md#BenchmarkResult.min_ms), [`tflops`](runner.md#BenchmarkResult.tflops), [`bandwidth_gb_s`](runner.md#BenchmarkResult.bandwidth_gb_s), [`all_times_ms`](runner.md#BenchmarkResult.all_times_ms)
- used by: [`tune`](__init__.md#tune)  (7 test-only)

