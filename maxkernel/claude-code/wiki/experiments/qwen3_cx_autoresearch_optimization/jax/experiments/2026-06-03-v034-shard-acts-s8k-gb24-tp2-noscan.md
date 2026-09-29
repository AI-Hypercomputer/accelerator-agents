---
title: "Qwen3 JAX v034 - shard acts s8k gb24 TP2 no scan"
type: experiment
hypothesis: "Removing scan may avoid v033's 13.50 GiB stacked-layer broadcast and make the TP2/global-batch-24 MaxText-CE shape compile."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, remat, maxtext-ce, activation-sharding, tp2, no-scan, long-seq, sharding, invalid, oom, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v034 - shard acts s8k gb24 TP2 no scan

## Hypothesis under test

**Hypothesis**: Removing scan may avoid v033's 13.50 GiB stacked-layer
broadcast and make the TP2/global-batch-24 MaxText-CE shape compile.

**Mechanism**: Reuse v033's TP2/global-batch-24 shape (`fsdp=4,tp=2`,
seq8192, per-FSDP batch 6, global batch 24), but set `--use_scan=False`.
Keep remat, Splash, MaxText CE, and activation sharding fixed.

**Predicted signal**: If v033 failed specifically because scan stacked all
decoder-layer activations/weights into the `[36,6,8192,4096]` broadcast, the
unrolled path should compile. It must still beat v041's 48,312 tok/s / 34.6% MFU
to be supported; a slower clean run is a refutation.

**Falsification criterion**: Compile/runtime failure, or clean completion at or
below v041's 48,312 tok/s / 34.6% MFU.

## Setup

GKE workload `alekseyv-qwen3-v034`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v034-shard-acts-s8k-gb24-tp2-noscan`
- **Mesh**: `fsdp=4,tp=2`
- **Command shape**: `--use_scan=False --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=6 --seqlen=8192 --tp_parallelism=2`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Compile failed before the first training step with HBM OOM:

- `RESOURCE_EXHAUSTED`: used 34.66 GiB of 31.25 GiB HBM, exceeded by
  3.42 GiB.
- Total HBM usage >= 34.92 GiB: program 28.94 GiB, arguments 5.72 GiB.
- HLO temp requirement 28.91 GiB, with 2.4% fragmentation / 703.82 MiB.
- Largest allocations:
  - 6.96 GiB `bf16[6,8192,75968]` from
    `transpose(jvp())/convert_element_type`, XLA label
    `multiply_convert_fusion`.
  - 6.96 GiB `bf16[6,8192,75968]` from `jvp()/dot_general`, XLA label
    `fusion.8048`.
  - Repeated 384 MiB `bf16[6,8192,4096]` async collective/output fusions
    across unrolled layers.

The no-scan variant did remove v033's 13.50 GiB
`bf16[36,6,8192,4096]` stacked-layer broadcast, but the resulting unrolled
program is still larger and now exposes half-vocab logit temps plus many
per-layer hidden-state temps.

## Profile

No runtime profile: compile failed before training steps.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v034-shard-acts-s8k-gb24-tp2-noscan/hlo/`
- **Size**: 341.9 MiB
- **Object count**: 32

## Verdict

**Invalid / refuted.** TP2 at the v041 global-batch-24 target shape is not a
viable layout, with or without scan. Scan fails on a stacked-layer broadcast;
no-scan fails on half-vocab logits and unrolled layer temps. Keep v041's
`fsdp=8,tp=1` scan layout as the target-sequence frontier unless a CE/logit
implementation changes the memory shape directly.
