---
title: 'Module: pallas_forge/profile/analysis.py'
type: catalog
provenance: extracted
module: pallas_forge/profile/analysis.py
status: fresh
symbol_base: scip-python python pallas-forge 0.0.0 `pallas_forge.profile.analysis`/
symbols:
  compute_operational_intensity: compute_operational_intensity().
  compute_mxu_utilization: compute_mxu_utilization().
  compute_bandwidth_utilization: compute_bandwidth_utilization().
  classify_boundedness: classify_boundedness().
  TPU_SPECS: TPU_SPECS.
---
# Module: [`pallas_forge/profile/analysis.py`](../../../../../../raw/code/pallas-forge/pallas_forge/profile/analysis.py)

## Functions
- `classify_boundedness(operational_intensity: float, peak_tflops: float, peak_bandwidth_gb_s: float)` — [`L55`](../../../../../../raw/code/pallas-forge/pallas_forge/profile/analysis.py#L55) — Classify a kernel as compute-bound or memory-bound.
- `compute_bandwidth_utilization(attained_gb_s: float, peak_gb_s: float)` — [`L37`](../../../../../../raw/code/pallas-forge/pallas_forge/profile/analysis.py#L37) — Compute memory bandwidth utilization as a percentage of peak.
- `compute_mxu_utilization(attained_tflops: float, peak_tflops: float)` — [`L22`](../../../../../../raw/code/pallas-forge/pallas_forge/profile/analysis.py#L22) — Compute MXU utilization as a percentage of peak.
- `compute_operational_intensity(total_flops: int, total_bytes: int)` — [`L10`](../../../../../../raw/code/pallas-forge/pallas_forge/profile/analysis.py#L10) — Compute operational intensity: FLOPS per byte accessed.

## Module values
- `TPU_SPECS` — [`L84`](../../../../../../raw/code/pallas-forge/pallas_forge/profile/analysis.py#L84)

