---
title: "Qwen3 JAX v043 - shard acts s8k bs3 label CE"
type: experiment
hypothesis: "Replacing MaxText one-hot CE with an integer-label custom VJP will reduce CE residual/temp pressure and improve v041 throughput without changing loss semantics."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v043-label-ce
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, remat, maxtext-ce, label-ce, activation-sharding, long-seq, invalid, oom, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v043 - shard acts s8k bs3 label CE

## Hypothesis under test

**Hypothesis**: Replacing MaxText one-hot CE with an integer-label custom VJP
will reduce CE residual/temp pressure and improve v041 throughput without
changing loss semantics.

**Mechanism**: Layer a single `train.py` change onto the v041 image. Add
`--use_label_ce=True`, which computes the same `logsumexp(logits) -
logits[label]` loss and `softmax - one_hot(label)` gradient as MaxText CE, but
passes integer labels through the custom VJP instead of materializing/saving the
forward one-hot/log-softmax target path.

**Predicted signal**: Clean run at v041's shape with loss trajectory matching
the same synthetic-data run, lower CE/logit memory pressure, and throughput
above v041's 48,312 tok/s / 34.6% MFU.

**Falsification criterion**: Compile/runtime failure, invalid loss trajectory,
or clean completion at or below v041's 48,312 tok/s / 34.6% MFU.

## Setup

GKE workload `alekseyv-qwen3-v043`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v043-label-ce`
- **Base image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Image digest**: `sha256:d5f651ec93e5d6052e26781730e334ba8b6c42f133cdc1e972362bc5e4c87003`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v043-shard-acts-s8k-bs3-label-ce`
- **Mesh**: `fsdp=8,tp=1`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_label_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Compile failed before the first training step with HBM OOM:

- `RESOURCE_EXHAUSTED`: used 34.88 GiB of 31.25 GiB HBM, exceeded by
  3.64 GiB.
- Total HBM usage >= 35.14 GiB: program 29.16 GiB, arguments 5.72 GiB.
- HLO temp requirement 29.15 GiB, with high fragmentation: 22.21 GiB unpadded
  and 6.94 GiB fragmentation.
- Largest allocations:
  - 13.91 GiB `f32[3,8192,151936]` from the LM-head `dot_general`.
  - 6.96 GiB `bf16[3,8192,151936]` from the same `dot_general`.
  - 1.16 GiB `bf16[151936,4096]` lm-head weight temp.

The implementation accidentally made the label-CE path worse than MaxText CE:
passing `logits.astype(jnp.float32)` into the custom VJP let XLA promote or
materialize the full lm-head dot output in f32. That produces a 13.91 GiB temp
not present in the v041 frontier.

## Profile

No runtime profile: compile failed before training steps.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v043-shard-acts-s8k-bs3-label-ce/hlo/`
- **Size**: 21.22 MiB
- **Object count**: 32

## Verdict

**Invalid / refuted for this implementation.** The integer-label CE idea did
not get a runtime signal because the external f32 logits cast changed the
compiler memory shape and OOMed. A narrower follow-up can keep the custom VJP
primal logits in bf16 and cast inside the CE body to test whether the idea is
viable without promoting the lm-head dot itself.
