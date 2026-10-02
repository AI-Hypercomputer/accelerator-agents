---
title: "Qwen3 JAX v022 - splash TP2 tokamax remat fused s4k bs2"
type: experiment
hypothesis: "Seq4096 may fit the v015/v018 TP2 stack if per-chip batch is reduced to the smallest even useful point, batch 2."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8
status: supported
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax, remat, fused-rope, tp2, long-seq, supported, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v022 - splash TP2 tokamax remat fused s4k bs2

## Hypothesis under test

**Hypothesis**: Seq4096 may fit the v015/v018 TP2 stack if per-chip batch is
reduced to the smallest even useful point, batch 2.

**Mechanism**: Reuse the v015 image, successful XLA flag stack, TP2 sharding,
Splash attention, tokamax CE, remat, and fused RoPE. Keep `--seqlen=4096` and
reduce per-chip batch from v021's 4 to 2, giving global batch 8.

**Predicted signal**: The run should compile. If it completes, it establishes
the first valid seq4096 point and tells whether longer-sequence MFU is worth
improving with memory work.

**Falsification criterion**: Compile/runtime OOM, or a clean run with too little
MFU to justify this stack at seq4096 without new memory mechanisms.

## Setup

Planned GKE workload `alekseyv-qwen3-v022`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v022-splash-tp2-tokamax-remat-fused-s4k-bs2`
- **Mesh**: `fsdp=4,tp=2`
- **Command shape**: `--batch_size=2 --seqlen=4096 --tp_parallelism=2 --use_splash=True`
- **Global batch**: 8
- **Profile window**: steps 12-14

## Results

Clean completion on `<your-cluster>`.

| Metric | Worker 0 | Worker 1 |
|--------|----------|----------|
| Exit code | 0 | 0 |
| Per-chip batch | 2 | 2 |
| Global batch | 8 | 8 |
| Sequence length | 4096 | 4096 |
| Steady step time | ~787.6 ms | ~787.6 ms |
| Avg throughput | 41,511 tok/s | 41,592 tok/s |
| Tok/s/chip | 5,189 | 5,199 |
| Approx MFU | 27.7% | 27.8% |

This is the first supported seq4096 point in the current v015/v018 TP2 stack.
It is not a new MFU frontier, but it confirms the long-sequence path is viable
when per-chip batch is reduced to 2.

## Profile

XProf run:
`2026-06-03-qwen3-jax-v022-splash-tp2-tokamax-remat-fused-s4k-bs2/2026_06_03_01_46_34`

| Metric | Value |
|--------|-------|
| MXU utilization | 32.3% |
| `jit_train_step` device time | 98.4% |
| Device idle time | 1.6% |
| Peak HBM | 25.90 GiB / 31.25 GiB |
| Free HBM at peak | 5.35 GiB |
| Stack reservation | 18.59 GiB |
| Heap allocations | 7.31 GiB |

Top HLO self-time:

| HLO op family | Self-time share |
|---------------|-----------------|
| Convolution fusion | 32.0% |
| All-reduce-scatter fusion | 15.4% |
| Custom-call | 13.9% |
| All-gather | 11.8% |
| Loop fusion | 11.0% |
| All-to-all | 4.6% |
| Data formatting | 4.1% |
| Custom fusion | 3.9% |

## HLO Dump

HLO artifacts were written under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v022-splash-tp2-tokamax-remat-fused-s4k-bs2/hlo/`

The dump contains 45 objects totaling 283.26 MiB, including
`module_0092.jit_train_step.cl_854318611` configs and per-step
`jit_train_step_*` directories.

## Verdict

Supported.

Seq4096 is feasible in this stack at per-chip batch 2, global batch 8. The
supported point has meaningful HBM margin (5.35 GiB free) and keeps MFU at
27.8%, but v021 shows that batch 4 at seq4096 still OOMs. The next feasibility
rung is seq8192 with per-chip batch 1.
