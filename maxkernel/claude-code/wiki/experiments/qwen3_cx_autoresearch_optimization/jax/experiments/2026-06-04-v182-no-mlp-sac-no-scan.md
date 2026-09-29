---
title: "Qwen3 JAX v182 - no MLP intermediate sac without scan"
type: experiment
hypothesis: "On the current v169 frontier stack, disabling scan may remove loop/body overhead and expose more scheduling freedom; if the newer collective-matmul-none plus no-MLP-sac memory savings are enough, it may now fit where old v042 did not."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-no-scan
status: provisional-supported
tags: [qwen3-cc, jax, flax-nnx, scan, remat, activation-sharding, mlp, sharding-constraint, splash, tokamax-splash, collective-matmul, scoped-vmem, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v182 - no MLP intermediate sac without scan

## Hypothesis under test

**Hypothesis**: v042 disabled scan on the older v041 stack and OOMed, dominated
by a full-vocab logit temp plus unrolled MLP/Splash temps. The current frontier
has since added collective-matmul-none, scoped VMEM 100352, scheduler
simplifications, and v169's no-MLP-intermediate `_sac` schedule. Those changes
may reduce enough memory or reshape enough scheduling pressure for no-scan to
fit and avoid scan/while overhead.

**Mechanism**: Reuse `v169-no-mlp-sac` and change only `--use_scan=True` to
`--use_scan=False`.

Keep remat, tokamax Splash, MaxText CE, activation sharding, batch/sequence
shape, sharding plan, and XLA flags fixed.

**Predicted signal**: Support requires clean compile/loss and throughput above
the confirmed v169/v170 frontier band (57,935-57,947 tok/s average, 58,051 best
observed). A compile-time HBM OOM refutes the hypothesis and keeps scan
mandatory at batch-4 seq8192.

**Falsification criterion**: Compile/runtime OOM, invalid loss, or clean
completion at/below the v169/v170 frontier.

## Setup

GKE workload `alekseyv-qwen3-v182-noscan-mlpsac`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v182-no-mlp-sac-no-scan`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v170 stack with `--xla_tpu_scoped_vmem_limit_kib=100352`.
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14 if the run reaches training

## Equivalence

No model-code edit was made. This is a runtime lowering/scheduling expression
change only and preserves model math.

## Results

Completed cleanly on both workers with normal loss and `EXIT_CODE=0`. Startup
confirmed scan was disabled: there is no `[scan]` line and the log prints one
Splash/Tokamax-Splash configuration per unrolled layer.

- Worker 0: 58,307 tok/s average, 7,288 tok/s/chip, 41.8% MFU.
- Worker 1: 58,330 tok/s average, 7,291 tok/s/chip, 41.8% MFU.
- Best observed steady step: 58,443 tok/s.
- Final loss: 12.0459 on both workers.

This exceeds the confirmed v169/v170 frontier (57,935-57,947 tok/s repeated
average, 58,051 best observed, 41.5% MFU). Compile/warmup cost is much higher:
step 0 took 134.6 s and step 1 took 10.7 s, but steady steps settle around
4.49 s versus v169/v170 around 4.52 s.

## Profile

Profile files were written under
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v182-no-mlp-sac-no-scan/plugins/profile/2026_06_04_21_31_37/`
for both TPU hosts:

- `gke-tpu-46dd3e54-g30s.xplane.pb`
- `gke-tpu-46dd3e54-rkr2.xplane.pb`

The `.xplane.pb` files are much larger than scan runs (~71.7 MiB each), matching
the unrolled HLO/program shape.

## HLO Dump

Fresh train-step HLO was emitted:

- Optimized HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v182-no-mlp-sac-no-scan/hlo/module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 19,043,435 bytes
- SHA256: `1ba7015c0111bd3ce0a8a9238ca32e25359e7f45c92033b921cf321da227dc48`
- Text counts: all-gather 4,656, all-reduce 960, collective-permute 3,
  convolution 2,096, custom-call 1,476, copy 9,401, fusion 18,276.
- Memory report: 27.66 GiB total bytes used. This is higher than v169's
  26.86 GiB but still fits under the v6e HBM limit.

## Verdict

Provisional supported. The old no-scan OOM was specific to the earlier schedule
family; on the current v169 frontier stack, no-scan fits and improves
steady-state throughput. Rerun exactly before carrying because the gain is
meaningful but still small enough to require confirmation.
