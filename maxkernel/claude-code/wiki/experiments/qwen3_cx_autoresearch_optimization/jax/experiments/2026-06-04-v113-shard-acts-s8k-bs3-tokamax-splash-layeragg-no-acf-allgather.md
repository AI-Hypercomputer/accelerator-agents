---
title: "Qwen3 JAX v113 - shard acts s8k bs3 tokamax splash layeragg no acf allgather"
type: experiment
hypothesis: "Async collective fusion's all-gather sub-pass may be neutral or harmful under the v090 layeragg scheduler even though base async collective fusion is required."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, async-collective-fusion, all-gather, latency-hiding, scheduler, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v113 - shard acts s8k bs3 tokamax splash layeragg no acf allgather

## Hypothesis under test

**Hypothesis**: v101 showed disabling base async collective fusion is harmful
because all-gather time reappears, but it did not isolate the subtype flags.
The v090 frontier explicitly enables both all-gather and reduce-scatter async
collective fusion. Under the layer-scheduler/sparse-core-aggregator pair, the
all-gather fusion subtype might be over-constraining the final schedule while
base async collective fusion and reduce-scatter fusion still provide the useful
overlap.

**Mechanism**: Keep the valid v090 qseq1 frontier fixed and change only
`--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true` to `false`.
Leave base async collective fusion, reduce-scatter fusion, and multiple-steps
fusion enabled.

**Predicted signal**: Loss stays finite and throughput exceeds the v090
51,033 tok/s best-observed / 51,026+ repeated frontier, or XProf shows lower
train-step wall time with no compensating all-gather/custom-call regression.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v090 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v113-noacfag`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v113-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-acf-allgather`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v090 scheduler frontier, except
  `--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v113-noacfag` completed cleanly on
`<your-cluster>`.

- Worker 0: 50,335 tok/s average, 36.1% MFU.
- Worker 1: 50,355 tok/s average, 36.1% MFU.
- Final loss: 12.0511 on both workers.
- XProf run:
  `2026-06-04-qwen3-jax-v113-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-acf-allgather/2026_06_04_10_11_19`.

This is valid but clearly below both the v090 best-observed 51,033 tok/s and
the 51,026+ repeated v089/v090 frontier band.

## Profile

XProf confirms that disabling the all-gather fusion subtype restores visible
all-gather cost and lowers MXU.

- Total profiled time: 94,003.8 ms.
- MXU utilization: 56.1%.
- Peak HBM: 28.03 GiB.
- Custom-call: 22,080.4 ms / 23.5%.
- DKV custom-call: 11,654.9 ms / 12.4%.
- Forward residual custom-calls: 10,425.5 ms / 11.1% combined.
- Convolution fusion: 45,385.7 ms / 48.3%.
- Loop fusion: 16,686.7 ms / 17.8%.
- Data formatting: 3,700.8 ms / 3.9%.
- All-gather: 2,136.2 ms / 2.3%.

The lower HBM mirrors v101's lower-memory/no-ACF schedule, but the all-gather
bucket and lower MXU make it slower in wall time.

## HLO Dump

Train-step HLO artifacts were written under the run's `hlo/` prefix.

- Optimized train-step text:
  `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 1,187,819 bytes.
- HLO object count under the run prefix: 1,619.
- Content hash: `4cdb9bf855debf8559a8e2f93c316d993b555fdc92b80cd380116317e29cf41d`.
- XProf train-step program hash: `16920192351436248925`.

This is a real schedule change with a smaller train-step HLO than the normal
layeragg frontier family.

## Verdict

Refuted. The all-gather async collective fusion subtype is required for the
current qseq1 layeragg frontier. Do not carry
`--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false` forward.
