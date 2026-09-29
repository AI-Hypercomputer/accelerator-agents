---
title: "Qwen3 JAX v180 - no MLP intermediate sac with remat offload"
type: experiment
hypothesis: "If no-remat is impossible because scan retains full layer activations, selective remat offload may reduce recompute cost without the HBM explosion and improve wall time on the v169 schedule."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-offload-remat
status: refuted
tags: [qwen3-cc, jax, flax-nnx, remat, offload, activation-sharding, mlp, sharding-constraint, splash, tokamax-splash, collective-matmul, scoped-vmem, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v180 - no MLP intermediate sac with remat offload

## Hypothesis under test

**Hypothesis**: v179 showed that disabling remat on top of v169 is impossible:
scan retains full stacked layer activations and XLA reports 340.87 GiB HBM.
The trainer already has a narrower offload-remat policy that saves/offloads only
named decoder inputs and attention projections to pinned host memory. On the
v169 no-MLP-intermediate schedule, this may reduce recompute work while avoiding
the no-remat HBM explosion.

**Mechanism**: Reuse `v169-no-mlp-sac` and change only the runtime flag
`--offload_remat=True`, keeping `--use_remat=True`.

Keep scan, tokamax Splash, MaxText CE, activation sharding, batch/sequence
shape, sharding plan, and XLA flags fixed.

**Predicted signal**: Support requires clean compile/loss and throughput above
the confirmed v169/v170 frontier band (57,935-57,947 tok/s average, 58,051 best
observed). The profile must show that reduced recompute outweighs offload/data
formatting cost.

**Falsification criterion**: Compile/runtime OOM, invalid loss, or clean
completion at/below the confirmed frontier.

## Setup

GKE workload `alekseyv-qwen3-v180-offload-mlpsac`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v180-no-mlp-sac-offload-remat`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v170 stack with `--xla_tpu_scoped_vmem_limit_kib=100352`.
- **Command shape**: `--use_scan=True --use_remat=True --offload_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

This is a runtime rematerialization/offload policy change only; it does not
change model math, parameter sharding, CE implementation, or weights. Per user
request, the CPU equivalence gate was rerun anyway against the current JAX lane
code using the v169 dependency image:

```bash
docker run --rm -e JAX_PLATFORMS=cpu \
  -v /mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki_cx/wiki/experiments/qwen3_cx_autoresearch_optimization/jax:/workspace \
  -w /workspace \
  <your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac \
  python -u test_equivalence.py
```

Result: `==== EQUIVALENCE PASS ====`.

- Forward logits max |delta|: `2.682e-07`
- Loss delta: `0.000e+00`
- Gradients: all 25 HF parameters OK
- Max observed gradient delta: `7.823e-08`

## Results

Completed cleanly on both workers with normal loss and `EXIT_CODE=0`. Startup
confirmed the intended policy:
`[remat] per-layer jax.checkpoint (save_and_offload(proj+mlpwi -> pinned_host)) ON`.

- Worker 0: 49,761 tok/s average, 6,220 tok/s/chip, 35.7% MFU.
- Worker 1: 49,743 tok/s average, 6,218 tok/s/chip, 35.7% MFU.
- Best observed steady step: 49,888 tok/s.
- Final loss: 12.0457 on both workers.

This is a large regression from the confirmed v169/v170 frontier
(57,935-57,947 tok/s repeated average, 58,051 best observed, 41.5% MFU).
Offload-remat makes the program valid but adds too much transfer / scheduling
overhead for this shape.

## Profile

Profile files were written under
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v180-no-mlp-sac-offload-remat/plugins/profile/2026_06_04_21_11_31/`
for both TPU hosts:

- `gke-tpu-46dd3e54-g30s.xplane.pb`
- `gke-tpu-46dd3e54-rkr2.xplane.pb`

The step log already shows the core signal: post-warmup steady steps are around
5.27 s versus v169/v170 around 4.52 s.

## HLO Dump

Fresh train-step HLO was emitted:

- Optimized HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v180-no-mlp-sac-offload-remat/hlo/module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 1,303,770 bytes
- SHA256: `262e50f1fe353d98136e328b51018e20df05b4ee151ee27ecb6b1403020501c1`
- Text counts: all-gather 226, all-reduce 62, collective-permute 3,
  convolution 110, custom-call 78, copy 694, fusion 964.
- Memory report: 54.71 GiB total bytes used, versus 26.86 GiB for v169. The
  largest temp allocations are offload/remat-shaped activation blocks:
  31.50 GiB color-5 preallocated temp and 17.46 GiB default preallocated temp.

## Verdict

Refuted. Selective host offload avoids v179's compile-time 340.87 GiB HBM OOM,
but the valid program is far slower than the v169/v170 frontier. Do not carry
`--offload_remat=True` for the current batch-4, seq8192, fsdp8 schedule. Keep
normal remat and move to a new mechanism.
