---
title: "Qwen3 JAX v202 - no-scan Tokamax CE batch 5"
type: experiment
hypothesis: "Tokamax CE under no-scan is too slow at batch 4, but its 22.50 GiB compiled-memory footprint may make per-chip batch 5 fit and recover enough total throughput to beat the MaxText CE batch-4 frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-tokamax-ce-bs5
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, ce, tokamax-ce, batch-scaling, splash, tokamax-splash, collective-matmul, scoped-vmem, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v202 - no-scan Tokamax CE batch 5

## Hypothesis under test

**Hypothesis**: v201 showed that Tokamax streamed CE under no-scan is not a
batch-4 throughput improvement, reaching only about 54.3k tok/s, but it lowered
compiled memory from the v182/v183 no-scan MaxText CE frontier's 27.66 GiB to
22.50 GiB. That memory relief may allow per-chip batch 5/global batch 40 to fit
and recover enough total throughput to beat the confirmed batch-4 frontier.

**Mechanism**: Reuse the v201 no-scan Tokamax CE stack and change only
`--batch_size=4` to `--batch_size=5`.

Keep the v169 no-MLP-intermediate-`_sac` image, no-scan, remat, tokamax Splash,
activation sharding, `TOKAMAX_MAX_LOGIT_CONST=30.0`, scoped VMEM 100352,
collective-matmul modes disabled, and the profile window fixed.

**Predicted signal**: Support requires clean compile/loss and throughput above
the v182/v183 confirmed 58,288-58,330 tok/s average / 41.8% MFU band. A clean
but slower run closes Tokamax CE batch scaling for this frontier.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion below the frontier.

## Setup

GKE workload `alekseyv-qwen3-v202-noscan-tokamaxce-bs5`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v202-no-scan-tokamax-ce-bs5`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v201 stack with `--xla_tpu_scoped_vmem_limit_kib=100352`,
  selective resources, custom-call VMEM adjustment, scheduler rerun 3, and
  both collective-matmul modes set to `none`.
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_tokamax_ce=True --tokamax_ce_impl=mosaic_tpu --use_maxtext_ce=False --shard_acts=True --batch_size=5 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 40
- **Profile window**: steps 12-14 if the run reaches training.

## Equivalence

No model-code edit is made. This changes only runtime batch size relative to
v201.

## Results

The workload completed cleanly on both workers with `EXIT_CODE=0` and normal
loss through step 19.

- Step 0/1 compile and first cached step: ~147.7 s each, 2,219-2,227 tok/s.
- Worker 0 steady steps 2-19: average 54,915 tok/s, range 54,844-55,069
  tok/s.
- Worker 1 steady steps 2-19: average 54,905 tok/s, range 54,684-55,068
  tok/s.
- Final loss at step 19: 12.0496 on both workers.

Batch 5/global batch 40 fits, but throughput is still below both v201 batch-4
Tokamax CE (~54.3k tok/s) by only a small amount and far below the v182/v183
MaxText CE no-scan frontier of 58,288-58,330 tok/s / 41.8% MFU.

## Profile

Profile files were written under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v202-no-scan-tokamax-ce-bs5/plugins/profile/2026_06_05_00_43_21/`

- `gke-tpu-964065d9-9vh8.xplane.pb`
- `gke-tpu-964065d9-c180.xplane.pb`
- matching `.trace.json.gz` files for both hosts.

Trace JSON spans for `jit_train_step(3503685562718697460)` confirm the stdout
throughput:

- `gke-tpu-964065d9-9vh8`: average profiled span 5.977850 s / 54,816 tok/s,
  best span 5.969598 s / 54,891 tok/s.
- `gke-tpu-964065d9-c180`: average profiled span 5.972015 s / 54,869 tok/s,
  best span 5.969459 s / 54,893 tok/s.

## HLO Dump

Fresh train-step HLO was emitted:

- Optimized HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v202-no-scan-tokamax-ce-bs5/hlo/module_0112.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 19,048,795 bytes
- SHA256: `e0dfbacf45659f1c3a07a30cd0736e4b80f16bcf7e401765015647d3341ccc7c`
- Memory report: 25.62 GiB total bytes used, with a 19.87 GiB preallocated
  temp. This proves the batch-5 Tokamax CE train step fits under the 31.25 GiB
  HBM capacity.
- Text counts: all-gather 7,038, all-reduce 1,147, collective-permute 6,
  convolution 3,060, custom-call 2,804, copy 21,400, fusion 31,679, `tokamax`
  32, `linear_softmax_cross_entropy_loss` 34, `log_softmax` 0, `_one_hot` 0.

## Verdict

Refuted for throughput. Tokamax CE's memory relief is enough to fit
batch 5/global batch 40 under no-scan, but the larger batch does not recover the
CE path's wall-time cost. Do not carry Tokamax CE batch scaling for this
frontier.
