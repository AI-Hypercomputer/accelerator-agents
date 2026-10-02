---
title: "Qwen3 JAX v042 - shard acts s8k bs3 no scan"
type: experiment
hypothesis: "At v041's target shape, disabling scan may reduce loop-fusion overhead and improve throughput if the unrolled TP1 graph still fits HBM."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, remat, maxtext-ce, activation-sharding, no-scan, long-seq, invalid, oom, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v042 - shard acts s8k bs3 no scan

## Hypothesis under test

**Hypothesis**: At v041's target shape, disabling scan may reduce loop-fusion
overhead and improve throughput if the unrolled TP1 graph still fits HBM.

**Mechanism**: Keep v041's supported target-sequence shape fixed
(`fsdp=8,tp=1`, seq8192, per-chip batch 3, global batch 24) and change only
`--use_scan=False`. Keep remat, Splash, MaxText CE, activation sharding, and
the XLA/libtpu flag stack fixed.

**Predicted signal**: A clean run should reduce the loop-fusion bucket or step
time versus v041's 48,312 tok/s / 34.6% MFU. A compile-time OOM means scan is a
required memory layout at this shape, not just a convenience.

**Falsification criterion**: Compile/runtime failure, or clean completion at or
below v041's 48,312 tok/s / 34.6% MFU.

## Setup

GKE workload `alekseyv-qwen3-v042`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v042-shard-acts-s8k-bs3-noscan`
- **Mesh**: `fsdp=8,tp=1`
- **Command shape**: `--use_scan=False --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Compile failed before the first training step with HBM OOM:

- `RESOURCE_EXHAUSTED`: used 34.11 GiB of 31.25 GiB HBM, exceeded by
  2.86 GiB.
- Total HBM usage >= 34.36 GiB: program 28.39 GiB, arguments 5.72 GiB.
- HLO temp requirement 28.37 GiB, with 2.7% fragmentation / 783.90 MiB.
- Largest allocations:
  - 6.96 GiB `bf16[3,8192,151936]` from
    `transpose(jvp())/convert_element_type`, XLA label
    `multiply_convert_fusion`.
  - 768 MiB `bf16[3,4,32,8192,128]` Splash DKV pallas-call temp.
  - Repeated 576 MiB `bf16[3,8192,12288]` MLP remat temps across unrolled
    layers.

This directly falsifies the idea that scan can be disabled at v041's
target-sequence shape for a runtime win. The TP1/no-scan graph is too large
even with the same global batch as v041.

## Profile

No runtime profile: compile failed before training steps.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v042-shard-acts-s8k-bs3-noscan/hlo/`
- **Size**: 539.16 MiB
- **Object count**: 32

## Verdict

**Invalid / refuted.** Scan is a required memory layout at the v041 target
shape, not just a source of loop-fusion overhead. Disabling scan exposes a
6.96 GiB full-vocab logit temp and repeated unrolled MLP/Splash temps, pushing
compile-time HBM 2.86 GiB over capacity. Keep `--use_scan=True` for this
frontier stack.
