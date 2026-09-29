---
title: "Qwen3 JAX v191 - no MLP intermediate sac, no scan, Tokamax Splash experimental scheduler"
type: experiment
hypothesis: "The no-scan unrolled graph may expose different attention custom-call scheduling pressure, so Tokamax Splash's experimental scheduler could improve the no-scan frontier despite regressing in the older scan family."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-no-scan-tokamax-exp-sched
status: refuted
tags: [qwen3-cc, jax, flax-nnx, scan, no-scan, remat, activation-sharding, mlp, sharding-constraint, splash, tokamax-splash, scheduler, collective-matmul, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v191 - no MLP intermediate sac, no scan, Tokamax Splash experimental scheduler

## Hypothesis under test

**Hypothesis**: v046 refuted `TOKAMAX_USE_EXP_SCHED=1` in an older scan-family
stack, but v182/v183 moved the frontier into a very different unrolled no-scan
HLO. The no-scan train-step HLO is much larger and recent BKV probes show
attention tiling choices strongly change fusion/copy/custom-call pressure.
Tokamax Splash's experimental scheduler may select a better custom-call
schedule in this no-scan graph.

**Mechanism**: Runtime-only Tokamax Splash scheduler toggle on top of the
confirmed v183 frontier:
`TOKAMAX_USE_EXP_SCHED=1`.

**Predicted signal**: Support requires clean compile/loss and steady throughput
above the confirmed no-scan band: v182/v183 at 58,288-58,330 tok/s confirmed
average and 58,443 best observed. A fresh HLO/hash or profile with lower
attention custom-call time would justify carrying this only in the no-scan
schedule.

**Falsification criterion**: Compile/runtime failure, invalid loss, HLO no-op
with no throughput gain, or clean completion at/below the no-scan frontier band.

## Setup

GKE workload `alekseyv-qwen3-v191-noscan-expsched`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v191-no-mlp-sac-no-scan-tokamax-exp-sched`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_USE_EXP_SCHED=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v183 stack.
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit was made. This is a runtime attention-kernel scheduler
toggle only and preserves model math. The current v169 lane equivalence test
was rerun in the dependency image on CPU and passed: logits max delta
`2.682e-07`, loss delta `0`, all 25 gradients OK, max gradient delta
`7.823e-08`.

## Results

Refuted. Workload `alekseyv-qwen3-v191-noscan-expsched` completed cleanly with
valid loss ending at 12.0459 and `EXIT_CODE=0`, but stayed below the confirmed
no-scan frontier:

- Worker0 stdout: 57,516 tok/s average / 7,189 tok/s/chip / 41.2% MFU.
- Worker1 stdout: 57,495 tok/s average / 7,187 tok/s/chip / 41.2% MFU.
- Best observed worker0 steady step: 57,663 tok/s at step 2.
- Profile-derived host averages: `gke-tpu-964065d9-9vh8` 4.572703 s / 57,328
  tok/s, `gke-tpu-964065d9-c180` 4.562751 s / 57,453 tok/s.
- Final loss: 12.0459.

## Profile

Profile files:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v191-no-mlp-sac-no-scan-tokamax-exp-sched/plugins/profile/2026_06_04_22_50_36/`

The profile validates the stdout regression. Device-side
`jit_train_step(6674237032253463936)` spans are about 4.563-4.573 s across the
two hosts, slower than the v183 no-scan 4.497 s class.

## HLO Dump

Fresh train-step HLO:

- Optimized HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v191-no-mlp-sac-no-scan-tokamax-exp-sched/hlo/module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 19,043,039 bytes.
- SHA256: `4a924f0315e80fecb00e1484b43911ac902f8112d8f447ec50baf75085aea731`.
- Memory report: 27.66 GiB total bytes used.
- Text counts: all-gather 7,039, all-reduce 1,148, collective-permute 6,
  convolution 3,071, custom-call 2,806, copy 21,271, fusion 31,686.

Compared with the v182/v183 frontier HLO (`1ba7015c...`, 19,043,435 bytes),
Tokamax experimental scheduling selects a real changed program. It has the same
large-count pattern as the refuted BKV512 family rather than the lower-count
frontier pattern, and wall time regresses.

## Verdict

Refuted. Do not carry `TOKAMAX_USE_EXP_SCHED=1` in the no-scan frontier. The
experimental scheduler is valid and better than the smaller-tile probes, but it
does not beat the confirmed no-scan schedule.
