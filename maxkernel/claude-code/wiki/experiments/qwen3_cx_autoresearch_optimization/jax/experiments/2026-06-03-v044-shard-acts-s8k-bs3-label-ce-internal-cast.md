---
title: "Qwen3 JAX v044 - shard acts s8k bs3 label CE internal cast"
type: experiment
hypothesis: "Keeping label-CE primal logits in bf16 while casting inside the custom VJP may avoid v043's f32 lm-head-dot materialization and recover a valid batch3 signal."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v044-label-ce-internal-cast
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, remat, maxtext-ce, label-ce, activation-sharding, long-seq, invalid, oom, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v044 - shard acts s8k bs3 label CE internal cast

## Hypothesis under test

**Hypothesis**: Keeping label-CE primal logits in bf16 while casting inside the
custom VJP may avoid v043's f32 lm-head-dot materialization and recover a valid
batch3 signal.

**Mechanism**: Same integer-label CE custom VJP as v043, but call it with bf16
logits. The CE body casts to f32 internally for the logsumexp/loss math and
returns the logits cotangent in the primal logits dtype, matching the original
bf16-logits -> f32-CE cast boundary.

**Predicted signal**: Compile should no longer allocate the 13.91 GiB
`f32[3,8192,151936]` lm-head dot temp seen in v043. If it runs cleanly, compare
throughput directly against v041's 48,312 tok/s / 34.6% MFU.

**Falsification criterion**: Compile/runtime failure, invalid loss trajectory,
or clean completion at or below v041's 48,312 tok/s / 34.6% MFU.

## Setup

GKE workload `alekseyv-qwen3-v044`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v044-label-ce-internal-cast`
- **Base image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Image digest**: `sha256:9fce87c68a31a854b9643b15f63b2a98b294d68c8cd04614e48aaf24d416eccf`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v044-shard-acts-s8k-bs3-label-ce-internal-cast`
- **Mesh**: `fsdp=8,tp=1`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_label_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Compile failed before the first training step with the same HBM OOM signature
as v043:

- `RESOURCE_EXHAUSTED`: used 34.88 GiB of 31.25 GiB HBM, exceeded by
  3.64 GiB.
- Total HBM usage >= 35.14 GiB: program 29.16 GiB, arguments 5.72 GiB.
- HLO temp requirement 29.15 GiB, with 22.21 GiB unpadded and 6.94 GiB
  fragmentation.
- Largest allocations:
  - 13.91 GiB `f32[3,8192,151936]` from the LM-head `dot_general`.
  - 6.96 GiB `bf16[3,8192,151936]` from the same `dot_general`.
  - 1.16 GiB `bf16[151936,4096]` lm-head weight temp.

Moving the f32 cast inside the custom VJP did not change the compiler memory
shape. XLA still promoted/materialized the lm-head dot output in f32 for this
label-CE implementation.

## Profile

No runtime profile: compile failed before training steps.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v044-shard-acts-s8k-bs3-label-ce-internal-cast/hlo/`
- **Size**: 21.23 MiB
- **Object count**: 32

## Verdict

**Invalid / refuted.** The integer-label custom VJP path is not viable in this
form. Both external-cast and internal-cast variants create a 13.91 GiB f32
lm-head-dot temp and fail at v041's batch3 target shape. Do not continue this
branch unless the implementation can avoid a custom VJP over full logits or
stream the CE without tokamax's observed runtime cost.
