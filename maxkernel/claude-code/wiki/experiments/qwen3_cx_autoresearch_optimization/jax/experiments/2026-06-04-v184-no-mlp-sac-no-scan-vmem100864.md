---
title: "Qwen3 JAX v184 - no MLP intermediate sac, no scan, VMEM 100864"
type: experiment
hypothesis: "The confirmed no-scan frontier has a much larger unrolled HLO and higher compiled memory than the scan frontier, so scoped VMEM may need retuning; raising the limit from 100352 to 100864 may improve the schedule."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-no-scan-vmem100864
status: refuted/tie
tags: [qwen3-cc, jax, flax-nnx, scan, no-scan, remat, activation-sharding, mlp, sharding-constraint, splash, tokamax-splash, collective-matmul, scoped-vmem, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v184 - no MLP intermediate sac, no scan, VMEM 100864

## Hypothesis under test

**Hypothesis**: v182/v183 confirmed that disabling scan on top of the v169
no-MLP-intermediate-sac schedule improves steady throughput, but the unrolled
program changes the scheduler problem substantially: HLO grows to 19,043,435
bytes and compiled memory rises to 27.66 GiB. Raising scoped VMEM from 100352 to
100864 KiB may select a better unrolled schedule even though the same change was
byte-identical/no-gain in the scan family.

**Mechanism**: Runtime-only XLA flag change on top of the confirmed v183
frontier:
`--xla_tpu_scoped_vmem_limit_kib=100864`.

**Predicted signal**: Support requires clean compile/loss and steady throughput
above the confirmed no-scan rerun band (profile-derived 58,288-58,291 tok/s,
v182 stdout 58,307-58,330 tok/s, 58,443 best observed). A fresh HLO/hash would
show that this flag matters for the unrolled schedule; a byte-identical HLO and
in-band throughput should be treated as no-gain.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v182/v183 no-scan band.

## Setup

GKE workload `alekseyv-qwen3-v184-noscan-vmem100864`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v184-no-mlp-sac-no-scan-vmem100864`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v183 stack except `--xla_tpu_scoped_vmem_limit_kib=100864`.
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit was made. This is a runtime scheduling flag change only and
preserves model math. The current lane equivalence test was rerun in the v169
dependency image before this submission and passed: logits max delta
`2.682e-07`, loss delta `0`, all 25 gradients OK, max gradient delta
`7.823e-08`.

## Results

The run completed cleanly on worker0 with normal loss and `EXIT_CODE=0`. The
worker1 pod was cleaned up before delayed stdout capture, but the profile trace
was present and was used for the second-host timing check.

Worker0 stdout:

- Step 0 compile/warmup: 138.5 s.
- Step 1 warmup: 10.6 s.
- Steady steps: mostly 4.487-4.501 s.
- Best observed step: 58,421 tok/s at step 15.
- Average measured throughput: 58,311 tok/s, 7,289 tok/s/chip.
- Approx MFU: 41.8%.
- Final loss: 12.0459.
- Exit code: 0.

This is inside the confirmed no-scan frontier band and does not beat v182's
58,443 best observed step or 58,330 best-worker average. It is a changed-HLO
tie, not a new frontier.

## Profile

Profile files:

- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v184-no-mlp-sac-no-scan-vmem100864/plugins/profile/2026_06_04_21_50_04/gke-tpu-46dd3e54-g30s.trace.json.gz`
- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v184-no-mlp-sac-no-scan-vmem100864/plugins/profile/2026_06_04_21_50_04/gke-tpu-46dd3e54-rkr2.trace.json.gz`
- matching `.xplane.pb` files for both hosts.

Device-side `jit_train_step` spans:

| Host trace | Captured spans | Avg captured step | Avg TPS | Best span TPS |
|------------|----------------|-------------------|---------|---------------|
| `g30s` | 12/12 | 4.498196 s | 58,278 tok/s | 58,293 tok/s |
| `rkr2` | 12/12 | 4.510664 s | 58,116 tok/s | 58,287 tok/s |

The first `rkr2` profile span is slower at 4.5356 s; excluding that edge gives
the same 4.498 s class as `g30s`. Either way, the profile does not show an
improvement over v182/v183.

## HLO Dump

Fresh optimized train-step HLO:

- GCS:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v184-no-mlp-sac-no-scan-vmem100864/hlo/module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Bytes: 19,043,156.
- SHA256: `8997b4c6b68aa332204162465ff4801465786cb170952864b3614ad637fcf16b`.
- Counts: all-gather 4,656, all-reduce 960, collective-permute 3,
  convolution 2,096, custom-call 1,476, copy 9,397, fusion 23,300.
- Memory report: 27.65 GiB.

The HLO differs from v182/v183 (`1ba7015c...`, 19,043,435 bytes), but the
schedule is not faster.

## Verdict

Refuted/tie. `--xla_tpu_scoped_vmem_limit_kib=100864` changes the unrolled
no-scan HLO slightly and still fits, but it does not improve throughput over
the confirmed 100352 KiB no-scan frontier. Keep scoped VMEM at 100352 KiB for
the current no-scan schedule.
