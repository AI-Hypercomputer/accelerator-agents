---
title: "Qwen3 JAX v192 - no MLP intermediate sac, no scan, no activation sharding constraints"
type: experiment
hypothesis: "The no-scan unrolled graph may no longer benefit from explicit layer-boundary activation sharding constraints; removing them could let XLA choose a better unrolled layout."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-no-scan-no-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, scan, no-scan, remat, activation-sharding, sharding-constraint, splash, tokamax-splash, collective-matmul, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v192 - no MLP intermediate sac, no scan, no activation sharding constraints

## Hypothesis under test

**Hypothesis**: `--shard_acts=True` was introduced for the scan-family target
sequence runs to pin layer-boundary activations to FSDP batch layout. The
confirmed no-scan frontier is a much larger unrolled HLO with different
fusion/copy pressure. Removing explicit activation constraints may let XLA
choose a better unrolled layout or reduce forced relayout/copy pressure.

**Mechanism**: Runtime-only command flag change on top of the confirmed v183
frontier:
`--shard_acts=True` -> `--shard_acts=False`.

All model math, MaxText CE, remat, no-scan, tokamax Splash, scoped VMEM, and XLA
flags remain fixed.

**Predicted signal**: Support requires clean compile/loss and steady throughput
above the confirmed no-scan band: v182/v183 at 58,288-58,330 tok/s confirmed
average and 58,443 best observed. A fresh HLO/hash with fewer relayout/copy or
collective artifacts would justify carrying no activation constraints in the
no-scan schedule.

**Falsification criterion**: Compile/runtime failure, HBM OOM, invalid loss, or
clean completion at or below the no-scan frontier without a profile/memory win.

## Setup

GKE workload `alekseyv-qwen3-v192-noscan-noshardacts`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v192-no-mlp-sac-no-scan-no-shard-acts`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v183 stack.
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=False --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit was made. This removes layout constraints only and preserves
model math. The current v169 lane equivalence test was rerun in the dependency
image on CPU and passed: logits max delta `2.682e-07`, loss delta `0`, all 25
gradients OK, max gradient delta `7.823e-08`.

## Results

Refuted/near-tie. Workload `alekseyv-qwen3-v192-noscan-noshardacts` completed
cleanly with valid loss ending at 12.0459 and `EXIT_CODE=0`, but stayed below
the confirmed no-scan frontier:

- Worker0 stdout: 58,088 tok/s average / 7,261 tok/s/chip / 41.6% MFU.
- Worker1 stdout: 58,067 tok/s average / 7,258 tok/s/chip / 41.6% MFU.
- Best observed worker0 steady step: 58,244 tok/s at step 2.
- Profile-derived host averages: `gke-tpu-964065d9-9vh8` 4.527451 s / 57,901
  tok/s, `gke-tpu-964065d9-c180` 4.517729 s / 58,026 tok/s.
- Final loss: 12.0459.

This is closer than v189-v191, but still below v182/v183's 58,288-58,330
confirmed average and v182's 58,443 best observed step.

## Profile

Profile files:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v192-no-mlp-sac-no-scan-no-shard-acts/plugins/profile/2026_06_04_22_58_44/`

The profile confirms the near-tie/regression. Device-side
`jit_train_step(14016405483169477827)` spans are about 4.518-4.527 s across the
two hosts, slower than the v183 no-scan 4.497 s class.

## HLO Dump

Fresh train-step HLO:

- Optimized HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v192-no-mlp-sac-no-scan-no-shard-acts/hlo/module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 19,042,581 bytes.
- SHA256: `a05d8bbc890d3894410361052661d359e95898e9ae3365121b1d1426db227dc7`.
- Memory report: 27.66 GiB total bytes used.
- Text counts: all-gather 7,039, all-reduce 1,148, collective-permute 6,
  convolution 3,071, custom-call 2,806, copy 21,271, fusion 31,686.

Removing all activation constraints selects the same high-count family as
v189/v191 rather than the lower-count v182/v183 frontier HLO. It slightly
reduces pre-optimization HLO size, but the final optimized graph remains slower.

## Verdict

Refuted/near-tie. Do not carry `--shard_acts=False` wholesale in the no-scan
frontier. The result suggests individual no-scan activation-boundary probes may
still be worth testing, but removing all layer-boundary constraints selects a
slower high-count schedule.
