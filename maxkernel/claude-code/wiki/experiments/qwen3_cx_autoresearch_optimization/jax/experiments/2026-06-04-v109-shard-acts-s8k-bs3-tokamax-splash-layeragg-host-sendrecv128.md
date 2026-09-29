---
title: "Qwen3 JAX v109 - shard acts s8k bs3 tokamax splash layeragg host sendrecv128"
type: experiment
hypothesis: "Increasing max concurrent host send/recv on the v090 frontier may expose more overlap than the current limit without changing model semantics."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, host-send-recv, latency-hiding, scheduler, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v109 - shard acts s8k bs3 tokamax splash layeragg host sendrecv128

## Hypothesis under test

**Hypothesis**: v105 showed that reducing
`--xla_max_concurrent_host_send_recv=100` to 64 does not help. The opposite
direction may still improve overlap if the v090 stack is constrained by host
send/recv concurrency around collectives and data formatting.

**Mechanism**: Keep the valid v090 qseq1 frontier fixed and change only
`--xla_max_concurrent_host_send_recv=100` to `128`.

**Predicted signal**: Loss stays finite and throughput exceeds the v090
51,033 tok/s best-observed / 51,026+ repeated frontier, or the profile shows a
useful reduction in idle/data-formatting/collective overhead without increasing
custom-call or loop-fusion time.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v090 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v109-hsr128`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v109-shard-acts-s8k-bs3-tokamax-splash-layeragg-host-sendrecv128`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: v090 scheduler frontier, except
  `--xla_max_concurrent_host_send_recv=128`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v109-hsr128` completed cleanly on 2026-06-04
with normal loss, but did not improve the v089/v090 frontier.

- **Worker 0 pod**: `alekseyv-qwen3-v109-hsr128-slice-job-0-0-sj4d4`
- **Worker 0 throughput**: 51,008 tok/s, 6,376 tok/s/chip
- **Worker 0 MFU**: 36.6%
- **Worker 1 pod**: `alekseyv-qwen3-v109-hsr128-slice-job-0-1-7tvr6`
- **Worker 1 throughput**: 50,984 tok/s, 6,373 tok/s/chip
- **Worker 1 MFU**: 36.6%
- **Final loss**: 12.0508
- **Exit status**: 0 on both workers

The pod command confirms `--xla_max_concurrent_host_send_recv=128`. This is
valid, but below v090's 51,033 tok/s best-observed worker and below the
51,026+ tok/s repeated v089/v090 band.

## Profile

- **XProf run**: `2026-06-04-qwen3-jax-v109-shard-acts-s8k-bs3-tokamax-splash-layeragg-host-sendrecv128/2026_06_04_09_23_09`
- **Profile host used**: `gke-tpu-964065d9-c180`
- **Total time**: 46,337.291 ms
- **Idle**: 76.035 ms / 0.2%
- **Compute**: 46,261.256 ms
- **Top program**: `jit_train_step`, 46,260.926 ms / 99.8%
- **Convolution fusion**: 22,838.037 ms / 49.3%
- **Custom-call**: 11,026.432 ms / 23.8%
- **DKV backward bucket**: 5,820.330 ms / 12.6%
- **Forward residual custom-calls**: 2,605.671 ms + 2,600.427 ms / 11.2%
- **Loop fusion**: 8,351.451 ms / 18.0%
- **Data formatting**: 1,878.272 ms / 4.1%
- **Peak HBM**: 28.32 GiB, with 2.9278 GiB free
- **Step average**: 1654.9 ms
- **MXU utilization**: 56.8%

The profile shape matches the v089/v090/v102-v108 in-band frontier family:
custom-call remains about 23.8%, DKV about 12.5-12.6%, loop fusion about 18%,
and data formatting about 4.1%. Raising host send/recv concurrency does not
expose a measurable overlap win.

## HLO Dump

The run wrote the normal train-step HLO:

- **Optimized train-step HLO**: `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- **Train-step HLO size**: 1,375,271 bytes
- **HLO object count**: 1,619
- **Program hash**: `13194376450853388418`

This matches the normal layeragg frontier HLO family, so the flag did not select
a materially different compiled program.

## Verdict

Refuted/tie. Increasing `--xla_max_concurrent_host_send_recv` from 100 to 128 is
valid, but reaches only 51,008 tok/s best worker and leaves the HLO/profile shape
unchanged. Keep the v090 frontier value, `--xla_max_concurrent_host_send_recv=100`.
