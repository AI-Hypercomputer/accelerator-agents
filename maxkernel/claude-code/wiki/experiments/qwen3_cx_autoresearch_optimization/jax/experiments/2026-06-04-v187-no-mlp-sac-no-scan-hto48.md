---
title: "Qwen3 JAX v187 - no MLP intermediate sac, no scan, host transfer overlap 48"
type: experiment
hypothesis: "The confirmed no-scan frontier may expose a different host/device overlap profile than the scan family, so raising host-transfer overlap from 24 to 48 could improve runtime scheduling without changing model semantics."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-no-scan-hto48
status: refuted/tie
tags: [qwen3-cc, jax, flax-nnx, scan, no-scan, remat, activation-sharding, mlp, sharding-constraint, splash, tokamax-splash, collective-matmul, host-transfer-overlap, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v187 - no MLP intermediate sac, no scan, host transfer overlap 48

## Hypothesis under test

**Hypothesis**: The unrolled no-scan program has different compile/runtime
behavior from the scan frontier. Raising
`--xla_tpu_host_transfer_overlap_limit` from 24 to 48 may improve runtime
overlap for the no-scan graph even though it was a no-gain tie in the scan
family.

**Mechanism**: Runtime-only XLA flag change on top of the confirmed v183
frontier:
`--xla_tpu_host_transfer_overlap_limit=48`.

**Predicted signal**: Support requires clean compile/loss and steady throughput
above the confirmed no-scan band (v182/v183: ~58.29k-58.33k tok/s steady,
58,443 best observed). Same-HLO in-band throughput should be treated as no
gain.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the no-scan frontier band.

## Setup

GKE workload `alekseyv-qwen3-v187-noscan-hto48`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v187-no-mlp-sac-no-scan-hto48`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v183 stack except `--xla_tpu_host_transfer_overlap_limit=48`.
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit was made. This is a runtime scheduling flag change only and
preserves model math.

## Results

The run completed cleanly on worker0 with normal loss and `EXIT_CODE=0`.

Worker0 stdout:

- Step 0 compile/warmup: 133.3 s.
- Step 1 warmup: 10.6 s.
- Steady steps: 4.486-4.501 s.
- Best observed step: 58,440 tok/s at step 2.
- Average measured throughput: 58,324 tok/s, 7,290 tok/s/chip.
- Approx MFU: 41.8%.
- Final loss: 12.0459.
- Exit code: 0.

This is in the v182/v183 no-scan frontier band and still just below v182's
58,443 best observed step. It is not a new frontier.

## Profile

Profile files:

- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v187-no-mlp-sac-no-scan-hto48/plugins/profile/2026_06_04_22_22_15/gke-tpu-46dd3e54-g30s.trace.json.gz`
- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v187-no-mlp-sac-no-scan-hto48/plugins/profile/2026_06_04_22_22_15/gke-tpu-46dd3e54-rkr2.trace.json.gz`
- matching `.xplane.pb` files for both hosts.

## HLO Dump

Optimized train-step HLO:

- GCS:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v187-no-mlp-sac-no-scan-hto48/hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Bytes: 19,043,435.
- SHA256: `1ba7015c0111bd3ce0a8a9238ca32e25359e7f45c92033b921cf321da227dc48`.
- Counts: all-gather 4,656, all-reduce 960, collective-permute 3,
  convolution 2,096, custom-call 1,476, copy 9,401, fusion 23,300.
- Memory report: 27.66 GiB.

This is byte-identical to v182/v183's optimized no-scan HLO.

## Verdict

Refuted/tie. Raising `--xla_tpu_host_transfer_overlap_limit` from 24 to 48 does
not change the optimized train-step HLO and does not beat the confirmed no-scan
frontier. Keep host-transfer overlap limit 24.
