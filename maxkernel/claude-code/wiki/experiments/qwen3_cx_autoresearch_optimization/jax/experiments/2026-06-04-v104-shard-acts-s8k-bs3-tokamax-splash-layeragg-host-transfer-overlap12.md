---
title: "Qwen3 JAX v104 - shard acts s8k bs3 tokamax splash layeragg host transfer overlap12"
type: experiment
hypothesis: "Lowering the TPU host-transfer overlap limit on the v090 frontier may remove unhelpful host send/recv scheduling pressure without changing model semantics."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, host-transfer-overlap, latency-hiding, scheduler, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v104 - shard acts s8k bs3 tokamax splash layeragg host transfer overlap12

## Hypothesis under test

**Hypothesis**: The v090 frontier includes
`--xla_tpu_host_transfer_overlap_limit=24`. The profile is still dominated by
device work, but the frontier also carries aggressive host send/recv overlap
and async collective settings. A smaller host-transfer overlap limit may reduce
scheduler pressure or compile/runtime overhead without affecting correctness.

**Mechanism**: Keep the valid v090 qseq1 frontier fixed and change only
`--xla_tpu_host_transfer_overlap_limit=24` to `12`.

**Predicted signal**: Loss stays finite and throughput exceeds the v090
51,033 tok/s best-observed / 51,026+ repeated frontier, or the profile shows a
useful reduction in data-formatting/collective overhead without increasing
custom-call or loop-fusion time.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v090 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v104-hto12`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v104-shard-acts-s8k-bs3-tokamax-splash-layeragg-host-transfer-overlap12`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: v090 scheduler frontier, except
  `--xla_tpu_host_transfer_overlap_limit=12`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v104-hto12` completed cleanly on 2026-06-04 with
valid loss, but did not exceed the v090 frontier band.

- Worker 0: 51,007 tok/s, 6,376 tok/s/chip, 36.6% MFU
- Worker 1: 50,964 tok/s, 6,371 tok/s/chip, 36.5% MFU
- Final loss: 12.0508 on both workers
- Exit code: 0

This is below v090's 51,033 best observed and below the 51,026+ repeated
v089/v090 band.

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v104-shard-acts-s8k-bs3-tokamax-splash-layeragg-host-transfer-overlap12/2026_06_04_08_30_23`

Profile:

- Total time: 92,910.3 ms; idle 394.5 ms / 0.4%.
- `jit_train_step`: 92,515.1 ms / 99.6%.
- Convolution fusion: 45,719.0 ms / 49.2%.
- Custom-call: 22,055.8 ms / 23.7%.
  - `splash_mha_dkv_no_residuals.1`: 11,641.1 ms / 12.5%.
  - `splash_mha_fwd_residuals.2/3`: 10,414.7 ms / 11.2%.
- Loop fusion: 16,720.0 ms / 18.0%.
- Data formatting: 3,767.1 ms / 4.1%.
- Peak HBM: 28.32 GiB of 31.25 GiB, with 2.93 GiB free.

The profile is effectively the same as v089/v090/v102/v103 and does not show
a data-formatting or custom-call reduction from lowering the host-transfer
overlap limit.

## HLO Dump

HLO dump:
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v104-shard-acts-s8k-bs3-tokamax-splash-layeragg-host-transfer-overlap12/hlo/`

- Train-step optimized HLO:
  `module_0264.jit_train_step.cl_854318611.after_optimizations.txt` and
  `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.
- Train-step optimized text size: 1,375,271 bytes, matching v089/v090/v102/v103.
- Pod command confirms `--xla_tpu_host_transfer_overlap_limit=12`.

## Verdict

**Refuted / tie.** Lowering the host-transfer overlap limit from 24 to 12 is
valid, but it does not improve throughput or profile shape. Keep
`--xla_tpu_host_transfer_overlap_limit=24` in the operational frontier unless a
later host-transfer sweep finds a real effect.
