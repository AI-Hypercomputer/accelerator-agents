---
title: "Qwen3 JAX v098 - shard acts s8k bs3 tokamax splash qseq2 no base2"
type: experiment
hypothesis: "The fast q_seq_shards=2 Tokamax Splash path NaNs because the q-sharded base2-exp softmax path is numerically unstable; disabling base2 exp may restore finite loss while preserving some qseq2 scheduling gain."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, qseq-shards, base2-exp, custom-call, remat, maxtext-ce, activation-sharding, long-seq, invalid, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v098 - shard acts s8k bs3 tokamax splash qseq2 no base2

## Hypothesis under test

**Hypothesis**: v050/v051/v064/v097 all show a large invalid throughput upside
from `TOKAMAX_Q_SEQ_SHARDS=2`, with NaN loss beginning at step 1. v097 refutes
the mask-representation-only explanation. The next minimal numeric-control
probe is disabling Tokamax's base2-exp path in the qseq2 kernel, because qseq1
ran valid with no-base2 in earlier ablations.

**Mechanism**: Use the supported v041 image and change only runtime Tokamax
numeric controls relative to the v090 scheduler frontier:

`TOKAMAX_Q_SEQ_SHARDS=2 TOKAMAX_MAX_LOGIT_CONST=30.0 TOKAMAX_USE_BASE2_EXP=0`

**Predicted signal**: Loss remains finite through the profiling window. If
valid, throughput should remain above the v090 valid frontier despite any
softmax cost regression.

**Falsification criterion**: NaN/invalid loss, compile/runtime failure, or a
clean run that falls back below the v090 51,033 tok/s / 36.6% MFU best observed
frontier.

## Setup

GKE workload `alekseyv-qwen3-v098-qseqnob2`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v098-shard-acts-s8k-bs3-tokamax-splash-qseq2-no-base2`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_Q_SEQ_SHARDS=2 TOKAMAX_MAX_LOGIT_CONST=30.0 TOKAMAX_USE_BASE2_EXP=0`
- **XLA flags**: v090 scheduler frontier with scoped VMEM 81920, custom-call
  VMEM adjustment, selective resources, scheduler rerun 3, async collective
  fusion flags, layer scheduler, and sparse-core collective aggregator.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v098-qseqnob2` completed on 2026-06-04 with exit code
0, but both workers produced NaN loss starting at step 1. This is invalid and
does not count as a frontier move.

- **Worker 0**: step 0 loss 12.0973, step 1 loss NaN, summary 55,574 tok/s /
  39.8% nominal MFU.
- **Worker 1**: step 0 loss 12.0973, step 1 loss NaN, summary 55,549 tok/s /
  39.8% nominal MFU.
- **Best valid comparison**: v090 remains 51,033 tok/s best observed, 51,026+
  repeated / 36.6% MFU.

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v098-shard-acts-s8k-bs3-tokamax-splash-qseq2-no-base2/2026_06_04_07_34_21`

The profile is invalid-fast diagnostic only. It reports total device time
85,186.2 ms, idle 0.3%, train-step time 84,900.0 ms, convolution fusion 52.9%,
loop fusion 19.6%, custom-call 17.7%, data formatting 4.5%, and
collective-permute-start 1.1%. Within custom-call,
`splash_mha_dkv_no_residuals.1` is 11,593.1 ms / 13.6%, and the two forward
residual calls total 3,457.1 ms / 4.0%.

Memory profile reports peak HBM 28.32 GiB of 31.25 GiB, with 22.51 GiB stack
reservation, 5.81 GiB heap allocation, and 90.62% utilization.

## HLO Dump

HLO was regenerated under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v098-shard-acts-s8k-bs3-tokamax-splash-qseq2-no-base2/hlo/`

- 32 HLO objects, 18.26 MiB total.
- Train-step module:
  `module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Flagfile confirms scoped VMEM 81920, custom-call scoped VMEM adjustments,
  selective resources, scheduler rerun 3, layer scheduler, sparse-core
  collective aggregator, and async collective fusion flags.
- Tokamax custom-call metadata confirms `use_base2_exp: false`,
  `fuse_reciprocal: true`, and `max_logit_const: 30.0`.

## Verdict

**Invalid/refuted.** Disabling Tokamax base2 exp does not restore qseq2
correctness. The NaN survives the q-sharded non-base2 softmax path, so the
qseq2 failure is not isolated to base2 exponentiation.
