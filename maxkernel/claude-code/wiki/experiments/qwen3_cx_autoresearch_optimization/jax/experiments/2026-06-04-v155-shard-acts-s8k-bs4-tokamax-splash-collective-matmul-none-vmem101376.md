---
title: "Qwen3 JAX v155 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 101376"
type: experiment
hypothesis: "If the scoped-VMEM optimum lies between confirmed 100352 and refuted 102400, then 101376 may retain v153/v154's faster schedule or expose the upper-edge regression."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, scoped-vmem, latency-hiding, scheduler, selective-resources, async-collective-fusion, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v155 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 101376

## Hypothesis under test

**Hypothesis**: The high-side scoped-VMEM bracket has a local optimum above the
confirmed 100352 KiB point but below the refuted 102400 KiB point.

**Mechanism**: Keep the confirmed v154 command, image, XLA flag stack, mesh,
batch, and profile window fixed. Change only
`--xla_tpu_scoped_vmem_limit_kib=100352` to `101376`.

**Predicted signal**: Support requires clean loss and a best-worker average
above the confirmed v153/v154 band, ideally above 57,757 tok/s. A regression
toward v152's 57,410 tok/s means 100352 is the carried high-side frontier.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed
command shape outside the scoped-VMEM limit, or clean completion below the
confirmed 100352 frontier.

## Setup

GKE workload `alekseyv-qwen3-v155-bs4-vmem101376`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v155-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem101376`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: exact v154 stack except `--xla_tpu_scoped_vmem_limit_kib=101376`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Worker 0 completed cleanly.

- Worker 0: average 57,588 tok/s (7,198/chip), approx 41.3% MFU; final loss 12.0463.
- Best steady worker-0 step: 57,740 tok/s (step 2).
- Kubernetes JobSet/pods were removed before worker 1 logs could be fetched;
  profile files for both TPU hosts are present in GCS.

This is below the confirmed v153/v154 100352 KiB band:

- v153: 57,757 tok/s best-worker average, 57,888 best observed.
- v154: 57,754 tok/s best-worker average, 57,862 best observed.

## Profile

Profile files were written under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v155-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem101376/plugins/profile/2026_06_04_17_40_29/`

Files present:

- `gke-tpu-964065d9-c180.xplane.pb`
- `gke-tpu-964065d9-c180.trace.json.gz`
- `gke-tpu-964065d9-9vh8.xplane.pb`
- `gke-tpu-964065d9-9vh8.trace.json.gz`

Immediate XProf HTTP/MCP queries again returned no-xplane/500 despite the GCS
files being present.

## HLO Dump

Fresh HLO dump was emitted for the train-step program, but the optimized HLO is
byte-identical to v153:

- HLO: `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 783,949 bytes
- SHA256: `3734fae066fc35c4b9c6c4cce9b0015413ab4513c34a09ba33a0e1862bff3a3d`
- Compiled memory: 28,841,540,536 bytes / 26.86 GiB
- Flagfile confirms `--xla_tpu_scoped_vmem_limit_kib=101376`.

## Verdict

**Refuted/no gain.** `101376` is valid, but it does not improve the confirmed
100352 frontier. Since the optimized HLO is identical to v153 while throughput
is lower in the available worker log, carry `100352` and do not pursue the
high-side bracket further without a new mechanism.
