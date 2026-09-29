---
title: "Qwen3 JAX v026 - shard acts s8k bs4"
type: experiment
hypothesis: "The v041 activation-sharded stack may have enough HBM margin to increase seq8192 from global batch 24 to global batch 32."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, scan, remat, maxtext-ce, activation-sharding, tp1, long-seq, batch-scaling, invalid, oom, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v026 - shard acts s8k bs4

## Hypothesis under test

**Hypothesis**: The v041 activation-sharded stack may have enough HBM margin to
increase seq8192 from global batch 24 to global batch 32.

**Mechanism**: Reuse the `v041-shard-acts` image and flag stack exactly, add HLO
dumping for analysis, and increase only `--batch_size` from 3 to 4. Keep
`--tp_parallelism=1`, `--seqlen=8192`, `--use_scan=True`, `--use_remat=True`,
`--use_maxtext_ce=True`, `--shard_acts=True`, and Splash attention.

**Predicted signal**: If batch 4 fits, it should improve target-sequence
throughput and MFU over v041. If it OOMs, v041 is the activation-sharded batch
frontier and attention/custom-call tuning becomes the next target.

**Falsification criterion**: Compile/runtime OOM, or clean completion with no
MFU gain over v041's 34.6%.

## Setup

Planned GKE workload `alekseyv-qwen3-v026`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v026-shard-acts-s8k-bs4`
- **Mesh**: `fsdp=8,tp=1`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Invalid. Workload `alekseyv-qwen3-v026` failed during train-step compilation
on both workers with `RESOURCE_EXHAUSTED`.

- **Exit**: `EXIT_CODE=1` at `Wed Jun 3 02:13:40 UTC 2026` on worker 0 and
  `Wed Jun 3 02:13:41 UTC 2026` on worker 1
- **OOM**: used 33.53 GiB of 31.25 GiB HBM; exceeded capacity by 2.29 GiB
- **Memory accounting**: total HBM usage at least 33.79 GiB, reserved 260.00
  MiB, program 27.81 GiB, arguments 5.72 GiB, HLO temp 27.80 GiB
- **Fragmentation**: HLO temp padded 27.80 GiB vs unpadded 18.55 GiB; 33.3%
  fragmentation / 9.25 GiB
- **Largest allocations**: two 9.27 GiB vocab/logit tensors of shape
  `bf16[4,8192,151936]`

## Profile

No runtime profile; the workload failed before executing training steps.

## HLO Dump

OOM artifacts were dumped under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v026-shard-acts-s8k-bs4/hlo/`

The prefix contains 45 objects totaling 18.44 MiB, including
`module_0112.jit_train_step.cl_854318611.oom_intermediate_module.txt` and
`module_0112.jit_train_step.cl_854318611.oom_top_memory_consuming_instructions`.

## Verdict

Refuted. v041 is the activation-sharded batch frontier at seq8192: per-chip
batch 3 / global batch 24 fits with 34.6% MFU, but per-chip batch 4 / global
batch 32 materializes two 9.27 GiB `[4,8192,151936]` tensors and OOMs. Further
gains at target sequence length should tune the attention/custom-call path or
change the CE/logit materialization, not increase batch under the current
MaxText CE path.
