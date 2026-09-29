---
title: "Qwen3 JAX v203 - no-scan MaxText CE batch 5"
type: experiment
hypothesis: "The confirmed no-scan MaxText CE frontier may be close enough to the HBM limit that per-chip batch 5/global batch 40 fits; if it does, the faster CE path could beat the batch-4 throughput frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-maxtext-ce-bs5
status: invalid
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, ce, maxtext-ce, batch-scaling, splash, tokamax-splash, collective-matmul, scoped-vmem, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v203 - no-scan MaxText CE batch 5

## Hypothesis under test

**Hypothesis**: v182/v183 established the current no-scan MaxText CE frontier
at batch 4/global batch 32 with 27.66 GiB compiled memory and 58,288-58,330
tok/s. v202 showed that batch 5/global batch 40 can fit with Tokamax CE, but
the CE path is too slow at 54.9k tok/s. The faster MaxText CE path may be a
near-fit at batch 5; if it compiles, the larger global batch could beat the
current throughput frontier.

**Mechanism**: Reuse the v183 no-scan MaxText CE frontier stack and change only
`--batch_size=4` to `--batch_size=5`.

Keep the v169 no-MLP-intermediate-`_sac` image, no-scan, remat, tokamax Splash,
MaxText CE, activation sharding, `TOKAMAX_MAX_LOGIT_CONST=30.0`, scoped VMEM
100352, collective-matmul modes disabled, and the profile window fixed.

**Predicted signal**: Support requires clean compile/loss and throughput above
the v182/v183 confirmed 58,288-58,330 tok/s average / 41.8% MFU band.

**Falsification criterion**: Compile HBM OOM, runtime failure, invalid loss, or
clean completion below the frontier.

## Setup

GKE workload `alekseyv-qwen3-v203-noscan-maxtextce-bs5`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v203-no-scan-maxtext-ce-bs5`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=5 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 40
- **Profile window**: steps 12-14 if the run reaches training.

## Equivalence

No model-code edit is made. This changes only runtime batch size relative to
v183.

## Results

The job failed before step 0 with `EXIT_CODE=1`.

XLA reported a permanent TPU compile HBM OOM:

- Used 32.10 GiB of 31.25 GiB HBM.
- Exceeded capacity by 877.10 MiB.
- No training step or profile was produced.

## Profile

No profile was produced because compilation failed before step 0.

## HLO Dump

Failed train-step HLO was emitted:

- Optimized HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v203-no-scan-maxtext-ce-bs5/hlo/module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- OOM top-memory report:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v203-no-scan-maxtext-ce-bs5/hlo/module_0109.jit_train_step.cl_854318611.oom_top_memory_consuming_instructions`
- Size: 18,965,922 bytes
- SHA256: `741eb3be6ff0b4d55273479788cac32efc30cf51fd0e8cf6ceb36014c9bbcefd`
- Memory report: 32.11 GiB total bytes used, with a 26.35 GiB preallocated
  temp.
- Largest allocation: 11.59 GiB
  `bf16[5,8192,151936]` from `jit(train_step)/forward_backward/jvp()/dot_general`,
  i.e. the full-vocab lm-head/logit path at batch 5.
- Text counts: all-gather 6,964, all-reduce 1,148, collective-permute 6,
  convolution 3,082, custom-call 2,804, copy 21,780, fusion 31,399,
  `log_softmax` 0, `_one_hot` 11.

## Verdict

Invalid/refuted. Direct batch-5 scaling of the fast no-scan MaxText CE frontier
misses HBM by under 1 GiB, dominated by the batch-5 full-vocab lm-head/logit
temp. Future batch-5 attempts need a real memory reduction before raising
batch size again.
