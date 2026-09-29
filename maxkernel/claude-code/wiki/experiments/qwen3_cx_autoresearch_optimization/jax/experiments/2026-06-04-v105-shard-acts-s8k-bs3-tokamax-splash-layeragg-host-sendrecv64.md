---
title: "Qwen3 JAX v105 - shard acts s8k bs3 tokamax splash layeragg host sendrecv64"
type: experiment
hypothesis: "Reducing max concurrent host send/recv on the v090 frontier may reduce host-transfer scheduling pressure without changing model semantics."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, host-send-recv, latency-hiding, scheduler, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v105 - shard acts s8k bs3 tokamax splash layeragg host sendrecv64

## Hypothesis under test

**Hypothesis**: The v090 frontier carries
`--xla_max_concurrent_host_send_recv=100`. The valid frontier profile is
device-dominated, and v104 showed that reducing the adjacent host-transfer
overlap limit from 24 to 12 does not help. A less aggressive global host
send/recv concurrency limit may still reduce scheduler overhead or data
formatting noise without altering the train-step semantics.

**Mechanism**: Keep the valid v090 qseq1 frontier fixed and change only
`--xla_max_concurrent_host_send_recv=100` to `64`.

**Predicted signal**: Loss stays finite and throughput exceeds the v090
51,033 tok/s best-observed / 51,026+ repeated frontier, or the profile shows a
useful data-formatting / collective reduction without increasing custom-call or
loop-fusion time.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v090 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v105-hsr64`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v105-shard-acts-s8k-bs3-tokamax-splash-layeragg-host-sendrecv64`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: v090 scheduler frontier, except
  `--xla_max_concurrent_host_send_recv=64`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v105-hsr64` completed cleanly on 2026-06-04 with
valid loss, but did not exceed the v090 frontier band.

- Worker 0: 51,006 tok/s, 6,376 tok/s/chip, 36.6% MFU
- Worker 1: 50,994 tok/s, 6,374 tok/s/chip, 36.6% MFU
- Final loss: 12.0508 on both workers
- Exit code: 0

This is below v090's 51,033 best observed and below the 51,026+ repeated
v089/v090 band.

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v105-shard-acts-s8k-bs3-tokamax-splash-layeragg-host-sendrecv64/2026_06_04_08_40_49`

Profile:

- Total time: 92,735.3 ms; idle 221.6 ms / 0.2%.
- `jit_train_step`: 92,513.1 ms / 99.8%.
- Convolution fusion: 45,715.6 ms / 49.3%.
- Custom-call: 22,056.0 ms / 23.8%.
  - `splash_mha_dkv_no_residuals.1`: 11,641.1 ms / 12.6%.
  - `splash_mha_fwd_residuals.2/3`: 10,414.9 ms / 11.2%.
- Loop fusion: 16,716.3 ms / 18.0%.
- Data formatting: 3,771.2 ms / 4.1%.
- Peak HBM: 28.32 GiB of 31.25 GiB, with 2.93 GiB free.

The profile is effectively unchanged from v089/v090/v102-v104.

## HLO Dump

HLO dump:
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v105-shard-acts-s8k-bs3-tokamax-splash-layeragg-host-sendrecv64/hlo/`

- Train-step optimized HLO:
  `module_0264.jit_train_step.cl_854318611.after_optimizations.txt` and
  `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.
- Train-step optimized text size: 1,375,271 bytes, matching v089/v090/v102-v104.
- Pod command confirms `--xla_max_concurrent_host_send_recv=64`.

## Verdict

**Refuted / tie.** Reducing max concurrent host send/recv from 100 to 64 is
valid, but it does not improve throughput or profile shape. Keep
`--xla_max_concurrent_host_send_recv=100` in the operational frontier.
