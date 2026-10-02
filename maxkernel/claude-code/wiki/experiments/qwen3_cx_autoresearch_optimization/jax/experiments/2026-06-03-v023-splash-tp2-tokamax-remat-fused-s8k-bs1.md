---
title: "Qwen3 JAX v023 - splash TP2 tokamax remat fused s8k bs1"
type: experiment
hypothesis: "Seq8192 may fit the v015/v018 TP2 stack at the smallest per-chip batch, batch 1, after v022 showed seq4096 batch 2 has 5.35 GiB HBM margin."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8
status: supported
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax, remat, fused-rope, tp2, long-seq, supported, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v023 - splash TP2 tokamax remat fused s8k bs1

## Hypothesis under test

**Hypothesis**: Seq8192 may fit the v015/v018 TP2 stack at the smallest per-chip
batch, batch 1, after v022 showed seq4096 batch 2 has 5.35 GiB HBM margin.

**Mechanism**: Reuse the v015 image, successful XLA flag stack, TP2 sharding,
Splash attention, tokamax CE, remat, and fused RoPE. Increase `--seqlen` from
4096 to 8192 and reduce per-chip batch from 2 to 1, keeping global tokens per
step equal to v022 while doubling context length.

**Predicted signal**: The run should compile if Splash/tokamax/remat scale
cleanly with context length. If it completes, it establishes the first target
seq8192 point for the lane.

**Falsification criterion**: Compile/runtime OOM, or a clean run with severe MFU
collapse that points to a new long-sequence bottleneck.

## Setup

Planned GKE workload `alekseyv-qwen3-v023`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v023-splash-tp2-tokamax-remat-fused-s8k-bs1`
- **Mesh**: `fsdp=4,tp=2`
- **Command shape**: `--batch_size=1 --seqlen=8192 --tp_parallelism=2 --use_splash=True`
- **Global batch**: 4
- **Profile window**: steps 12-14

## Results

Clean completion on `<your-cluster>`.

| Metric | Worker 0 | Worker 1 |
|--------|----------|----------|
| Exit code | 0 | 0 |
| Per-chip batch | 1 | 1 |
| Global batch | 4 | 4 |
| Sequence length | 8192 | 8192 |
| Steady step time | ~881.9 ms | ~881.9 ms |
| Avg throughput | 37,113 tok/s | 37,148 tok/s |
| Tok/s/chip | 4,639 | 4,644 |
| Approx MFU | 26.6% | 26.6% |

This establishes the first supported target seq8192 point for the lane.

## Profile

XProf run:
`2026-06-03-qwen3-jax-v023-splash-tp2-tokamax-remat-fused-s8k-bs1/2026_06_03_01_53_42`

| Metric | Value |
|--------|-------|
| MXU utilization | 34.0% |
| `jit_train_step` device time | 98.8% |
| Device idle time | 1.2% |
| Peak HBM | 26.57 GiB / 31.25 GiB |
| Free HBM at peak | 4.67 GiB |
| Stack reservation | 19.19 GiB |
| Heap allocations | 7.38 GiB |

Top HLO self-time:

| HLO op family | Self-time share |
|---------------|-----------------|
| Convolution fusion | 28.7% |
| Custom-call | 20.4% |
| All-reduce-scatter fusion | 14.2% |
| All-gather | 10.4% |
| Loop fusion | 10.4% |
| Custom fusion | 4.4% |
| All-to-all | 4.1% |
| Data formatting | 4.0% |

## HLO Dump

HLO artifacts were written under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v023-splash-tp2-tokamax-remat-fused-s8k-bs1/hlo/`

The dump contains 45 objects totaling 287.31 MiB, including
`module_0092.jit_train_step.cl_854318611` configs and per-step
`jit_train_step_*` directories.

## Verdict

Supported.

The v015/v018 stack reaches the target seq8192 length at per-chip batch 1 with
26.6% MFU and 4.67 GiB free HBM. The profile shows the long-sequence bottleneck
shifting toward Splash custom-calls (20.4% self-time) plus collectives. The next
probe should test seq8192 batch 2 to locate the batch frontier.
