---
title: 'Module: tests/test_tune_runner.py'
type: catalog
provenance: extracted
module: tests/test_tune_runner.py
status: fresh
symbol_base: scip-python python pallas-forge 0.0.0 `tests.test_tune_runner`/TestBenchmarkR
symbols:
  TestBenchmarkResult.test_to_dict: esult#test_to_dict().
  TestBenchmarkResult.test_to_dict_no_throughput: esult#test_to_dict_no_throughput().
  TestBenchmarkRunner.test_basic_timing: unner#test_basic_timing().
  TestBenchmarkRunner.test_run_all_sorted: unner#test_run_all_sorted().
  TestBenchmarkRunner.test_flops_computation: unner#test_flops_computation().
  TestBenchmarkRunner.test_bandwidth_computation: unner#test_bandwidth_computation().
  TestBenchmarkRunner.kernel: unner#kernel().
  TestBenchmarkRunner.input_fn: unner#input_fn().
  TestBenchmarkResult: esult#
  TestBenchmarkRunner: unner#
---
# Module: [`tests/test_tune_runner.py`](../../../../../raw/code/pallas-forge/tests/test_tune_runner.py)

## Classes
### `TestBenchmarkResult`
- def: [`tests/test_tune_runner.py:10`](../../../../../raw/code/pallas-forge/tests/test_tune_runner.py#L10)
- signature: `class TestBenchmarkResult:`
- members:
  - `test_to_dict(self)` — [`L11`](../../../../../raw/code/pallas-forge/tests/test_tune_runner.py#L11)
  - `test_to_dict_no_throughput(self)` — [`L27`](../../../../../raw/code/pallas-forge/tests/test_tune_runner.py#L27)
- uses (calls/refs, reference-scoped): [`median_ms`](../pallas_forge/tune/runner.md#BenchmarkResult.median_ms), [`to_dict`](../pallas_forge/tune/runner.md#BenchmarkResult.to_dict), [`BenchmarkResult`](../pallas_forge/tune/runner.md#BenchmarkResult), [`config`](../pallas_forge/tune/runner.md#BenchmarkResult.config), [`mean_ms`](../pallas_forge/tune/runner.md#BenchmarkResult.mean_ms), [`std_ms`](../pallas_forge/tune/runner.md#BenchmarkResult.std_ms), [`max_ms`](../pallas_forge/tune/runner.md#BenchmarkResult.max_ms), [`min_ms`](../pallas_forge/tune/runner.md#BenchmarkResult.min_ms), [`tflops`](../pallas_forge/tune/runner.md#BenchmarkResult.tflops)

### `TestBenchmarkRunner`
- def: [`tests/test_tune_runner.py:41`](../../../../../raw/code/pallas-forge/tests/test_tune_runner.py#L41)
- signature: `class TestBenchmarkRunner:`
- members:
  - `input_fn(config)` — [`L48`](../../../../../raw/code/pallas-forge/tests/test_tune_runner.py#L48)
  - `kernel(x, *, multiplier=2)` — [`L45`](../../../../../raw/code/pallas-forge/tests/test_tune_runner.py#L45)
  - `test_bandwidth_computation(self)` — [`L97`](../../../../../raw/code/pallas-forge/tests/test_tune_runner.py#L97) — Bandwidth should be computed when bytes_fn is provided.
  - `test_basic_timing(self)` — [`L42`](../../../../../raw/code/pallas-forge/tests/test_tune_runner.py#L42) — Runner should produce valid timing results with a trivial kernel.
  - `test_flops_computation(self)` — [`L80`](../../../../../raw/code/pallas-forge/tests/test_tune_runner.py#L80) — TFLOPS should be computed when flops_fn is provided.
  - `test_run_all_sorted(self)` — [`L59`](../../../../../raw/code/pallas-forge/tests/test_tune_runner.py#L59) — Results should be sorted by median time (fastest first).
- uses (calls/refs, reference-scoped): [`run_single`](../pallas_forge/tune/runner.md#BenchmarkRunner.run_single), [`median_ms`](../pallas_forge/tune/runner.md#BenchmarkResult.median_ms), [`config`](../pallas_forge/tune/runner.md#BenchmarkResult.config), [`BenchmarkRunner`](../pallas_forge/tune/runner.md#BenchmarkRunner), [`run_all`](../pallas_forge/tune/runner.md#BenchmarkRunner.run_all), [`mean_ms`](../pallas_forge/tune/runner.md#BenchmarkResult.mean_ms), [`std_ms`](../pallas_forge/tune/runner.md#BenchmarkResult.std_ms), [`tflops`](../pallas_forge/tune/runner.md#BenchmarkResult.tflops), [`bandwidth_gb_s`](../pallas_forge/tune/runner.md#BenchmarkResult.bandwidth_gb_s)

