---
title: "Qwen3 JAX v100 - shard acts s8k bs3 tokamax splash qseq2 conservative softmax"
type: experiment
hypothesis: "The fast q_seq_shards=2 Tokamax Splash path may require disabling both base2 exp and fused reciprocal together; the combined conservative softmax path may restore finite loss where either single ablation did not."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, qseq-shards, conservative-softmax, custom-call, remat, maxtext-ce, activation-sharding, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v100 - shard acts s8k bs3 tokamax splash qseq2 conservative softmax

## Hypothesis under test

**Hypothesis**: v098 (`use_base2_exp=false`) keeps step 0 finite but NaNs at
step 1, while v099 (`fuse_reciprocal=false`) NaNs immediately at step 0. The
remaining small qseq2 numeric-control interaction is disabling both paths
together, forcing the most conservative Tokamax softmax variant.

**Mechanism**: Use the supported v041 image and change the qseq2 softmax mode:

`TOKAMAX_Q_SEQ_SHARDS=2 TOKAMAX_MAX_LOGIT_CONST=30.0 TOKAMAX_USE_BASE2_EXP=0 TOKAMAX_FUSE_RECIPROCAL=0`

**Predicted signal**: Loss remains finite through the profiling window. If it
is still invalid, retire qseq2 runtime numeric-control probes and pivot back to
structural valid-run optimizations.

**Falsification criterion**: NaN/invalid loss, compile/runtime failure, or a
clean run that falls below the v090 51,033 tok/s / 36.6% MFU best observed
frontier.

## Setup

Planned GKE workload `alekseyv-qwen3-v100-qseqsafe`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v100-shard-acts-s8k-bs3-tokamax-splash-qseq2-conservative-softmax`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_Q_SEQ_SHARDS=2 TOKAMAX_MAX_LOGIT_CONST=30.0 TOKAMAX_USE_BASE2_EXP=0 TOKAMAX_FUSE_RECIPROCAL=0`
- **XLA flags**: v090 scheduler frontier with scoped VMEM 81920, custom-call
  VMEM adjustment, selective resources, scheduler rerun 3, async collective
  fusion flags, layer scheduler, and sparse-core collective aggregator.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v100-qseqsafe` completed with exit code 0 on the
v6e-8 slice, but both workers produced NaN loss from step 0. The combined
conservative Tokamax qseq2 softmax path is therefore invalid.

- **Worker 0**: avg throughput 54,732 tok/s, 6,841 tok/s/chip, 39.2% nominal
  MFU; loss NaN on steps 0-19.
- **Worker 1**: avg throughput 54,766 tok/s, 6,846 tok/s/chip, 39.3% nominal
  MFU; loss NaN on steps 0-19.
- **Best worker**: 54,766 tok/s / 39.3% nominal MFU.
- **Validity**: invalid; NaN from the first measured step makes the apparent
  speedup unusable.

## Profile

XProf run:
`2026-06-04-qwen3-jax-v100-shard-acts-s8k-bs3-tokamax-splash-qseq2-conservative-softmax/2026_06_04_07_46_13`

Device profile summary:

- Total device time: 86,481.5 ms; train step: 86,179.6 ms / 99.7%.
- Idle: 301.2 ms / 0.3%.
- Convolution fusion: 45,249.9 ms / 52.3%.
- Custom-call: 14,976.2 ms / 17.3%.
- `splash_mha_dkv_no_residuals.1`: 11,587.9 ms / 13.4%.
- Forward residual custom calls: 3,388.3 ms / 3.9%.
- Data formatting: 3,778.1 ms / 4.4%.
- Collective-permute-start: 1,337.3 ms / 1.5%.
- Peak HBM: 28.32 GiB of 31.25 GiB, with 22.51 GiB stack reservation and
  5.81 GiB heap allocation.

The invalid-fast signature matches v098/v099: qseq2 reduces the custom-call
bucket relative to the valid v090 stack, but the loss is already NaN before
the profile window.

## HLO Dump

HLO dump:
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v100-shard-acts-s8k-bs3-tokamax-splash-qseq2-conservative-softmax/hlo/`

- Train-step optimized HLO:
  `module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Optimized train-step text size: 1,386,661 bytes.
- Runtime metadata confirms the conservative qseq2 mode:
  `q_seq_shards=2`, `max_logit_const=30.0`, `use_base2_exp=false`,
  `fuse_reciprocal=false`.

## Verdict

**Refuted / invalid.** Disabling both base2 exp and fused reciprocal does not
restore qseq2 correctness; it makes the path NaN from step 0, like v099. The
qseq2 runtime numeric-control axis is exhausted for this frontier:

- v050/v051: qseq2 NaNs at step 1.
- v097: dense static mask still NaNs at step 1.
- v098: disabling base2 exp still NaNs at step 1.
- v099: disabling fused reciprocal NaNs at step 0.
- v100: disabling both base2 exp and fused reciprocal NaNs at step 0.

Do not count the 39.3% nominal MFU. Keep v090 as the best valid frontier and
pivot to valid-run structural attention/CE work or small scheduler probes that
preserve the qseq1 valid path.
