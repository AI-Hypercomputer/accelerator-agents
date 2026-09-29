---
title: "Qwen3 JAX v017 - splash TP2 tokamax remat scan bs8"
type: experiment
hypothesis: "Adding the scan-over-layers variant at the v015 TP2 batch-8 shape will preserve the tokamax/remat throughput while reducing HLO temp pressure or compile/runtime overhead."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8
verdict: refuted
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax, remat, scan, tp2, refuted, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v017 - splash TP2 tokamax remat scan bs8

## Hypothesis under test

**Hypothesis**: Adding the scan-over-layers variant at the v015 TP2 batch-8 shape will preserve the tokamax/remat throughput while reducing HLO temp pressure or compile/runtime overhead.

**Mechanism**: The actual accepted workload uses the v015 image with `--tp_parallelism=2` and `--batch_size=8`, and writes to a `v017-splash-tp2-tokamax-remat-scan-bs8` run directory. This preserves the v015 mesh shape (`fsdp=4,tp=2`, global batch 32) while testing the scan/remat variant represented by the launched run.

**Predicted signal**: The run should complete, keep throughput near or above v015's 47,846 tok/s / 30.8% MFU, and show lower HLO temp pressure or cleaner profile structure than v015. If scan materially helps, XProf should not show a large increase in loop-fusion or collective share.

**Falsification criterion**: Compile/runtime failure, HBM OOM, or a clean run with lower throughput/MFU than v015 and no memory/profile improvement.

## Setup

GKE workload `alekseyv-qwen3-v017`.

Note: an earlier `v017` submission attempt using a TP4 command received SIGTERM (`EXIT_CODE=143`) before first train step and disappeared from the API. The active accepted workload with the same name was recreated with the TP2 scan run path below; this page tracks the active workload.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v017-splash-tp2-tokamax-remat-scan-bs8`
- **Mesh**: `fsdp=4,tp=2`
- **Command shape**: `--batch_size=8 --seqlen=2048 --tp_parallelism=2 --use_splash=True`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Both pods completed cleanly; `EXIT_CODE=0`.

| Metric | Worker 0 | Worker 1 | Notes |
|--------|----------|----------|-------|
| Steady step time | ~1.369 s | ~1.369 s | step 0 cold/cache setup excluded |
| Throughput | 47,864 tok/s | 47,820 tok/s | 5,983 and 5,977 tok/s/chip |
| MFU | 30.8% | 30.7% | trainer estimate, v6e bf16 peak |
| Global batch | 32 | 32 | seq_len=2048 |
| Exit code | 0 | 0 | 20 train steps |

Comparison:

| Run | Throughput | MFU | Peak HBM | Notes |
|-----|------------|-----|----------|-------|
| v015 frontier | 47,846 tok/s | 30.8% | 31.23 GiB | formal current best |
| v017 scan variant | 47,864 tok/s | 30.8% | 31.23 GiB | no meaningful delta |

## Profile

- **XProf run**: `2026-06-03-qwen3-jax-v017-splash-tp2-tokamax-remat-scan-bs8/2026_06_03_01_08_28`
- **XProf URL**: `http://localhost:8791/?run=2026-06-03-qwen3-jax-v017-splash-tp2-tokamax-remat-scan-bs8`
- **MXU utilization**: 35.4% (v015: 35.5%)
- **Op profile**: `jit_train_step` accounts for 99.2% of device time; idle is 0.8%.

Top HLO op families by time:

| Op family | Time share | v015 share | Notes |
|-----------|------------|------------|-------|
| convolution fusion | 35.9% | 35.8% | effectively unchanged |
| loop fusion | 12.9% | 12.8% | slightly worse, not a scan win |
| all-gather | 11.6% | 11.5% | unchanged TP communication |
| all-reduce-scatter fusion | 11.1% | 11.0% | unchanged |
| custom-call | 10.5% | 10.4% | Splash/Pallas attention |
| all-to-all | 5.4% | 5.4% | unchanged |
| data formatting | 5.0% | 5.0% | unchanged |

Memory profile:

```text
capacity: 31.25 GiB
peak:    31.23 GiB
stack:   25.2969 GiB
heap:     5.9352 GiB
free:     0.014 GiB
util:    99.94%
```

## HLO Dump

No HLO dump objects were found under the v017 run path when checked with `gsutil ls`. The short step-0 time relative to v015 suggests the train-step compilation reused the persistent compilation cache, so this run has XProf data but no fresh HLO text dump in its run directory.

## Verdict

`refuted`.

The scan-labeled variant completes, but it does not produce the predicted profile or memory improvement. Throughput is a measurement-level tie with v015, MXU is slightly lower, loop-fusion share is slightly higher, and peak HBM remains 31.23 GiB with only ~14 MiB free. v015 remains the current best.
