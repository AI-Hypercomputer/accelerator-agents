---
title: "Qwen3 JAX v018 - splash TP2 tokamax remat fused bs6"
type: experiment
hypothesis: "Reducing per-chip batch from 8 to 6 in the v015 TP2 stack will recover meaningful HBM margin while preserving most of the MFU gain from tokamax/remat/fused kernels."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8
status: supported
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax, remat, fused-rope, tp2, batch-scaling, supported, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v018 - splash TP2 tokamax remat fused bs6

## Hypothesis under test

**Hypothesis**: Reducing per-chip batch from 8 to 6 in the v015 TP2 stack will recover meaningful HBM margin while preserving most of the MFU gain from tokamax/remat/fused kernels.

**Mechanism**: Reuse the v015 image and TP2 sharding, but run `--batch_size=6` instead of 8. The mesh stays `fsdp=4,tp=2`; global batch becomes 24, trading peak throughput for memory headroom.

**Predicted signal**: The run should complete with materially lower peak HBM than v015/v017's 31.23 GiB while preserving at least ~28% MFU. If the batch reduction is a useful memory-relief point, XProf should show the same op mix but with lower stack/heap pressure and enough free HBM to make longer-sequence probes plausible.

**Falsification criterion**: Compile/runtime failure, peak HBM still near 31.25 GiB, or clean completion with a disproportionate MFU collapse relative to the recovered memory.

## Setup

Planned GKE workload `alekseyv-qwen3-v018`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v018-splash-tp2-tokamax-remat-fused-bs6`
- **Mesh**: `fsdp=4,tp=2`
- **Command shape**: `--batch_size=6 --seqlen=2048 --tp_parallelism=2 --use_splash=True`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Clean completion on `<your-cluster>`.

| Metric | Value |
|--------|-------|
| Exit code | 0 |
| Per-chip batch | 6 |
| Global batch | 24 |
| Sequence length | 2048 |
| Steady step time | ~1056 ms |
| Worker 0 avg throughput | 46,526 tok/s |
| Worker 0 tok/s/chip | 5,816 |
| Worker 0 approximate MFU | 29.9% |

This is not a new throughput/MFU frontier versus v015, but it preserves most of
the v015 speed while recovering material HBM headroom.

## Profile

XProf run:
`2026-06-03-qwen3-jax-v018-splash-tp2-tokamax-remat-fused-bs6/2026_06_03_01_15_20`

| Metric | Value |
|--------|-------|
| MXU utilization | 33.8% |
| `jit_train_step` device time | 98.4% |
| Device idle time | 1.6% |
| Peak HBM | 28.32 GiB / 31.25 GiB |
| Free HBM at peak | 2.92 GiB |
| Stack reservation | 20.92 GiB |
| Heap allocations | 7.40 GiB |

Top HLO self-time remains the same shape as v015/v017:

| HLO op family | Self-time share |
|---------------|-----------------|
| Convolution fusion | 35.2% |
| Loop fusion | 12.7% |
| All-reduce-scatter fusion | 11.5% |
| All-gather | 11.2% |
| Custom-call | 10.1% |
| All-to-all | 5.2% |
| Data formatting | 4.9% |
| Custom fusion | 4.1% |

Compared with v015/v017, peak HBM drops from 31.23 GiB to 28.32 GiB while MFU
drops from 30.8% to 29.9%. The result supports using smaller per-chip batches
as a memory-relief knob before attempting longer sequences.

## HLO Dump

HLO artifacts were written under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v018-splash-tp2-tokamax-remat-fused-bs6/hlo/`

The dump contains 45 objects totaling 280.84 MiB, including
`module_0092.jit_train_step.cl_854318611` configs and per-step
`jit_train_step_*` directories.

## Verdict

Supported as a non-frontier memory-margin point.

Reducing per-chip batch from 8 to 6 recovers ~2.9 GiB of free HBM while retaining
29.9% MFU. The current formal frontier remains v015 at 47,846 tok/s / 30.8% MFU,
but v018 establishes that batch reduction can buy enough margin to justify a
nearby bs7 Pareto probe and then longer-sequence experiments.
