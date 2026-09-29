---
title: "Qwen3 JAX v099 - shard acts s8k bs3 tokamax splash qseq2 no fuse recip"
type: experiment
hypothesis: "The fast q_seq_shards=2 Tokamax Splash path NaNs because the q-sharded fused reciprocal path is numerically unstable; disabling fused reciprocal may restore finite loss while preserving some qseq2 scheduling gain."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, qseq-shards, fuse-reciprocal, custom-call, remat, maxtext-ce, activation-sharding, long-seq, invalid, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v099 - shard acts s8k bs3 tokamax splash qseq2 no fuse recip

## Hypothesis under test

**Hypothesis**: v050/v051/v064/v097/v098 all show a large invalid throughput
upside from `TOKAMAX_Q_SEQ_SHARDS=2`, with NaN loss beginning at step 1. Dense
masking and disabling base2 exp do not restore correctness. The next isolated
numeric-control probe is disabling Tokamax's fused reciprocal path.

**Mechanism**: Use the supported v041 image and change only runtime Tokamax
numeric controls relative to the v090 scheduler frontier:

`TOKAMAX_Q_SEQ_SHARDS=2 TOKAMAX_MAX_LOGIT_CONST=30.0 TOKAMAX_FUSE_RECIPROCAL=0`

**Predicted signal**: Loss remains finite through the profiling window. If
valid, throughput should remain above the v090 valid frontier despite any
reciprocal cost regression.

**Falsification criterion**: NaN/invalid loss, compile/runtime failure, or a
clean run that falls below the v090 51,033 tok/s / 36.6% MFU best observed
frontier.

## Setup

GKE workload `alekseyv-qwen3-v099-qseqnorecip`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v099-shard-acts-s8k-bs3-tokamax-splash-qseq2-no-fuse-recip`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_Q_SEQ_SHARDS=2 TOKAMAX_MAX_LOGIT_CONST=30.0 TOKAMAX_FUSE_RECIPROCAL=0`
- **XLA flags**: v090 scheduler frontier with scoped VMEM 81920, custom-call
  VMEM adjustment, selective resources, scheduler rerun 3, async collective
  fusion flags, layer scheduler, and sparse-core collective aggregator.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v099-qseqnorecip` completed on 2026-06-04 with exit
code 0, but both workers produced NaN loss from step 0. This is invalid and
does not count as a frontier move.

- **Worker 0**: step 0 loss NaN, summary 54,909 tok/s / 39.4% nominal MFU.
- **Worker 1**: step 0 loss NaN, summary 54,882 tok/s / 39.4% nominal MFU.
- **Best valid comparison**: v090 remains 51,033 tok/s best observed, 51,026+
  repeated / 36.6% MFU.

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v099-shard-acts-s8k-bs3-tokamax-splash-qseq2-no-fuse-recip/2026_06_04_07_40_14`

The profile is invalid-fast diagnostic only. It reports total device time
86,222.3 ms, idle 0.3%, train-step time 85,944.3 ms, convolution fusion 52.5%,
custom-call 17.1%, data formatting 4.4%, and collective-permute-start 1.4%.
Within custom-call, `splash_mha_dkv_no_residuals.1` is 11,532.6 ms / 13.4%,
and the two forward residual calls total 3,222.1 ms / 3.7%.

Memory profile reports peak HBM 28.32 GiB of 31.25 GiB, with 22.51 GiB stack
reservation, 5.81 GiB heap allocation, and 90.62% utilization.

## HLO Dump

HLO was regenerated under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v099-shard-acts-s8k-bs3-tokamax-splash-qseq2-no-fuse-recip/hlo/`

- 28 HLO objects, 18.11 MiB total.
- Train-step module:
  `module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Tokamax custom-call metadata confirms `fuse_reciprocal: false`,
  `use_base2_exp: true`, and `max_logit_const: 30.0`.

## Verdict

**Invalid/refuted.** Disabling fused reciprocal alone makes qseq2 worse
numerically: loss is NaN at step 0 rather than step 1. The fused reciprocal path
is not the isolated qseq2 correctness bug.
