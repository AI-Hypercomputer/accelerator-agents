---
title: "Qwen3 JAX v185 - no MLP intermediate sac, no scan, VMEM 99328"
type: experiment
hypothesis: "After v184 showed the high-side VMEM point changes the no-scan HLO but does not improve throughput, the lower-side point may reduce scheduling pressure for the unrolled graph and recover a faster steady schedule."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-no-scan-vmem99328
status: refuted
tags: [qwen3-cc, jax, flax-nnx, scan, no-scan, remat, activation-sharding, mlp, sharding-constraint, splash, tokamax-splash, collective-matmul, scoped-vmem, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v185 - no MLP intermediate sac, no scan, VMEM 99328

## Hypothesis under test

**Hypothesis**: The no-scan unrolled graph has a different scheduling surface
from the scan graph. v184 proved that VMEM 100864 changes the unrolled HLO but
does not improve throughput. Lowering scoped VMEM to 99328 KiB may reduce
custom-call/temporary pressure enough to improve steady step time.

**Mechanism**: Runtime-only XLA flag change on top of the confirmed v183
frontier:
`--xla_tpu_scoped_vmem_limit_kib=99328`.

**Predicted signal**: Support requires clean compile/loss and steady throughput
above the confirmed no-scan band (v182/v183/v184: roughly 58.28k-58.33k tok/s
steady averages, 58,443 best observed). A fresh HLO/hash would show the flag
matters for the unrolled schedule; a same-HLO or in-band result closes this
local VMEM retune.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the no-scan frontier band.

## Setup

GKE workload `alekseyv-qwen3-v185-noscan-vmem99328`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v185-no-mlp-sac-no-scan-vmem99328`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v183 stack except `--xla_tpu_scoped_vmem_limit_kib=99328`.
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit was made. This is a runtime scheduling flag change only and
preserves model math. The current lane equivalence test was rerun in the v169
dependency image before the VMEM retune sequence and passed.

## Results

The run completed cleanly on worker0 with normal loss and `EXIT_CODE=0`.

Worker0 stdout:

- Step 0 compile/warmup: 138.1 s.
- Step 1 warmup: 10.7 s.
- Steady steps: 4.519-4.542 s.
- Best observed step: 58,004 tok/s at step 2.
- Average measured throughput: 57,815 tok/s, 7,227 tok/s/chip.
- Approx MFU: 41.5%.
- Final loss: 12.0460.
- Exit code: 0.

This is below the confirmed no-scan frontier band and below even the old
v169/v170 best observed step of 58,051 tok/s. Lowering scoped VMEM to 99328 KiB
is a real schedule change, but the schedule is slower.

## Profile

Profile files:

- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v185-no-mlp-sac-no-scan-vmem99328/plugins/profile/2026_06_04_21_59_51/gke-tpu-964065d9-c180.trace.json.gz`
- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v185-no-mlp-sac-no-scan-vmem99328/plugins/profile/2026_06_04_21_59_51/gke-tpu-964065d9-9vh8.trace.json.gz`
- matching `.xplane.pb` files for both hosts.

## HLO Dump

Fresh optimized train-step HLO:

- GCS:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v185-no-mlp-sac-no-scan-vmem99328/hlo/module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Bytes: 19,043,751.
- SHA256: `6e904d246b2893d10d93b978661be51bdaf2aa04f4ee5d1305fadfb99a971d2f`.
- Counts: all-gather 4,656, all-reduce 960, collective-permute 3,
  convolution 2,096, custom-call 1,476, copy 9,405, fusion 23,300.
- Memory report: 27.66 GiB.

## Verdict

Refuted. `--xla_tpu_scoped_vmem_limit_kib=99328` changes the unrolled no-scan
HLO but slows steady throughput to 57,815 tok/s / 41.5% MFU. Keep scoped VMEM at
100352 KiB for the current no-scan schedule. Together with v184, this closes the
immediate lower/high-side VMEM retune around the v182/v183 frontier.
