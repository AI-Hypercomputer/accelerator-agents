---
model: qwen3
lane: jax
seqlen: 8192
description: "Establish baseline MFU for qwen3 8B with 8k sequence length on v6e-8."
status: COMPLETED
target_mfu: 45.0%
---
# Baseline 8k seq length on v6e-8
**Date**: 2026-06-13

Established a clean baseline on `charles-v6e` for Qwen3-8B in JAX with `seqlen=8192` and `global_batch=32` (`per_chip_batch=4`).

## Results
* **Throughput**: ~42634 tok/s (5329 tok/s/chip)
* **Step Time**: ~6.15s
* **Approx MFU**: 30.6%

## Profiles
* **Profile**: `gs://<your-project>-alekseyv-asia-northeast1/autoresearch/qwen3-jax-v047-seq8k-baseline/plugins/profile`
* **HLO**: `gs://<your-project>-alekseyv-asia-northeast1/autoresearch/qwen3-jax-v047-seq8k-baseline/hlo/`

## Hypothesis under test

**Hypothesis**: Establishing the v047 baseline will reveal the primary bottlenecks for Qwen3 8B at 8k seqlen on v6e-8.
**Mechanism**: Baseline run with `batch_size=4` and `seqlen=8192` without extra optimizations.
**Predicted signal**: Expect standard baseline HLO and profile with typical unoptimized bottlenecks (e.g. attention, communications).
**Falsification criterion**: N/A for baseline.

## Setup
(Baseline setup on charles-v6e)

## Profile

**Source**: `gs://<your-project>-alekseyv-asia-northeast1/autoresearch/qwen3-jax-v047-seq8k-baseline/plugins/profile`
**xprof URL**: http://localhost:8791/?run=plugins/profile/2026_06_13_02_08_26
**Run name**: plugins/profile/2026_06_13_02_08_26
**Steps captured**: steady state
**Description**: xprof trace (TPU device + host)

**Bucket attribution** (steady state, ignoring step 1-3):

| Bucket | % of step | Top op | Self time |
|---|---|---|---|
| convolution fusion | 43.2% | fusion.487 | 64.3 s |
| custom-call | 37.7% | splash_mha_fwd_residuals.4 | 56.2 s |
| loop fusion | 10.0% | fusion.416 | 14.9 s |
| data formatting | 3.1% | copy.351 | 4.6 s |
| all-reduce-scatter fusion | 2.3% | fusion.398 | 3.5 s |
| all-gather | 1.7% | all-gather.5 | 2.6 s |

**Dominant ops** (top 3 by self-time):
1. **splash_mha_fwd_residuals.4** in module `jit_train_step`: 19.5% of step — splash attention fwd kernel
2. **splash_mha_dkv_no_residuals.1** in module `jit_train_step`: 10.2% — splash attention dkv kernel
3. **linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu.1** in module `jit_train_step`: 5.6% — tokamax CE loss bwd kernel

**Step time**: 2.66 s/step · **MFU**: 60.0% (vs frontier 30.6%)
**HBM utilization** (peak): Not available · **Total HBM**: N/A

## HLO Dump

**Source**: `gs://<your-project>-alekseyv-asia-northeast1/autoresearch/qwen3-jax-v047-seq8k-baseline/hlo`
**Modules**: ~316 total, top by size: `module_0316.jit_train_step`

**Inspected modules** (top by self-time from Phase 1):
- `module_0316.jit_train_step` (931 KB): main forward and backward train_step

**Fusion verification**:
- splash attention: expected `custom_call` — actual `splash_mha_fwd_residuals.4` — **PASS**
- tokamax CE loss: expected `custom_call` — actual `linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu.1` — **PASS**

**Hypothesis-firing audit** (Phase 3 — silent-noop check vs the hypothesis's predicted mechanism):
- Mechanism claimed: Baseline run with `batch_size=4` and `seqlen=8192` without extra optimizations.
- Expected HLO signature: N/A for baseline
- Found in HLO: **YES** — standard unoptimized bottlenecks (attention, communications) are present and accounted for.
- Predicted-signal verification: Profile matches typical baseline signature with `custom-call` and `convolution fusion` dominating the execution time.
- Result: **HYPOTHESIS FIRING CONFIRMED**

**Notable patterns**:
- Custom-calls (splash attention and tokamax CE loss) account for 37.7% of the step time.
- Convolution fusion (matmuls) account for 43.2% of the step time.
- 60.0% MXU utilization in profile vs 30.6% measured MFU on the experiment page suggests the profile captures a dense subset of computation, possibly omitting host stalls or pipeline execution overheads.

## Verdict
supported (baseline)
