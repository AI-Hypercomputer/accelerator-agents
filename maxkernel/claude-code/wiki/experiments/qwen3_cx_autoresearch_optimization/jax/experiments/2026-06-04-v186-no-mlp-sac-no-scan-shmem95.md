---
title: "Qwen3 JAX v186 - no MLP intermediate sac, no scan, scheduler shared memory 95"
type: experiment
hypothesis: "The unrolled no-scan graph has much larger HLO/fusion pressure than the scan graph, so a slightly lower TPU scheduler shared-memory limit may improve placement without the throughput cost seen in stronger caps."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-no-scan-shmem95
status: refuted/tie
tags: [qwen3-cc, jax, flax-nnx, scan, no-scan, remat, activation-sharding, mlp, sharding-constraint, splash, tokamax-splash, collective-matmul, scoped-vmem, scheduler, shared-memory, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v186 - no MLP intermediate sac, no scan, scheduler shared memory 95

## Hypothesis under test

**Hypothesis**: The confirmed no-scan schedule is faster but has a much larger
unrolled HLO and higher copy/fusion pressure. Lowering
`--xla_tpu_scheduler_percent_shared_memory_limit` from 100 to 95 may improve
scheduling/placement for this unrolled graph, even though the same knob was a
no-gain tie in the scan family.

**Mechanism**: Runtime-only XLA flag change on top of the confirmed v183
frontier:
`--xla_tpu_scheduler_percent_shared_memory_limit=95`.

**Predicted signal**: Support requires clean compile/loss and steady throughput
above the confirmed no-scan band (v182/v183, with v184/v185 closing local VMEM
retunes). A changed HLO/hash plus higher step throughput would justify carrying
the lower scheduler shared-memory limit.

**Falsification criterion**: Compile/runtime failure, invalid loss, same-HLO
no-op, or clean completion at or below the no-scan frontier band.

## Setup

GKE workload `alekseyv-qwen3-v186-noscan-shmem95`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v186-no-mlp-sac-no-scan-shmem95`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v183 stack except `--xla_tpu_scheduler_percent_shared_memory_limit=95`.
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit was made. This is a runtime scheduling flag change only and
preserves model math. The current lane equivalence test was rerun in the v169
dependency image before the no-scan retune sequence and passed.

## Results

The run completed cleanly on worker0 with normal loss and `EXIT_CODE=0`.

Worker0 stdout:

- Step 0 compile/warmup: 134.2 s.
- Step 1 warmup: 10.6 s.
- Steady steps: 4.503-4.518 s.
- Best observed step: 58,210 tok/s at step 2.
- Average measured throughput: 58,104 tok/s, 7,263 tok/s/chip.
- Approx MFU: 41.7%.
- Final loss: 12.0459.
- Exit code: 0.

This is below the confirmed no-scan frontier and below v184's no-gain high-side
VMEM result. The shared-memory cap does not improve the no-scan schedule.

## Profile

Profile files:

- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v186-no-mlp-sac-no-scan-shmem95/plugins/profile/2026_06_04_22_10_33/gke-tpu-964065d9-c180.trace.json.gz`
- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v186-no-mlp-sac-no-scan-shmem95/plugins/profile/2026_06_04_22_10_33/gke-tpu-964065d9-9vh8.trace.json.gz`
- matching `.xplane.pb` files for both hosts.

## HLO Dump

Optimized train-step HLO:

- GCS:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v186-no-mlp-sac-no-scan-shmem95/hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Bytes: 19,043,435.
- SHA256: `1ba7015c0111bd3ce0a8a9238ca32e25359e7f45c92033b921cf321da227dc48`.
- Counts: all-gather 4,656, all-reduce 960, collective-permute 3,
  convolution 2,096, custom-call 1,476, copy 9,401, fusion 23,300.
- Memory report: 27.66 GiB.

This is byte-identical to v182/v183's optimized no-scan HLO.

## Verdict

Refuted/tie. Lowering
`--xla_tpu_scheduler_percent_shared_memory_limit` from 100 to 95 does not change
the optimized train-step HLO and regresses observed throughput to 58,104 tok/s.
Do not carry the shared-memory cap on the current no-scan frontier.
