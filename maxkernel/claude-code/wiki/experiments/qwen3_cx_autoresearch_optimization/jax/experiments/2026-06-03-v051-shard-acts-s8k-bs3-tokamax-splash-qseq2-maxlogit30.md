---
title: "Qwen3 JAX v051 - shard acts s8k bs3 tokamax splash qseq2 maxlogit30"
type: experiment
hypothesis: "A max-logit bound may make the fast q_seq_shards=2 tokamax Splash path numerically valid."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, remat, maxtext-ce, activation-sharding, long-seq, invalid, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v051 - shard acts s8k bs3 tokamax splash qseq2 maxlogit30

## Hypothesis under test

**Hypothesis**: Adding `TOKAMAX_MAX_LOGIT_CONST=30.0` may stabilize the fast
but invalid `TOKAMAX_Q_SEQ_SHARDS=2` path from v050.

**Mechanism**: Keep v045 fixed and set `TOKAMAX_Q_SEQ_SHARDS=2
TOKAMAX_MAX_LOGIT_CONST=30.0`. v050 showed the q-sequence split cuts
custom-call time but produces NaN loss from step 1; this run tests whether the
max-logit bound restores the normal loss trajectory.

**Predicted signal**: Loss stays near the v045/v049 trajectory, ending around
12.05, while throughput remains above v045's 49,069 tok/s / 35.2% MFU.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion at or below v045's 49,069 tok/s / 35.2% MFU.

## Setup

GKE workload `alekseyv-qwen3-v051`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v051-shard-acts-s8k-bs3-tokamax-splash-qseq2-maxlogit30`
- **Mesh**: `fsdp=8,tp=1`
- **Environment change**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_Q_SEQ_SHARDS=2 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Invalid run:

- Worker 0: 54,956 tok/s total, 6,869 tok/s/chip, 39.4% nominal MFU.
- Worker 1: 54,930 tok/s total, 6,866 tok/s/chip, 39.4% nominal MFU.
- Loss became NaN at step 1 on both workers and stayed NaN through step 19.

`TOKAMAX_MAX_LOGIT_CONST=30.0` did not stabilize the q-sequence split from
v050. The high throughput remains unusable.

## Profile

- **XProf run**:
  `2026-06-03-qwen3-jax-v051-shard-acts-s8k-bs3-tokamax-splash-qseq2-maxlogit30/2026_06_03_04_16_31`
- **MXU utilization**: 54.4%.
- **Peak HBM**: 28.33 GiB, 2.914 GiB free.
- **Device idle**: 0.3%.

Top device buckets:

- Convolution fusion: 51.8%.
- Loop fusion: 20.3%.
- Custom-call: 17.2%.

The max-logit bound reduces the invalid q-seq custom-call profile further than
v050, but it does not fix numerical correctness.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v051-shard-acts-s8k-bs3-tokamax-splash-qseq2-maxlogit30/hlo/`
- **Size**: 19.02 MiB
- **Object count**: 26

## Verdict

**Invalid.** `TOKAMAX_MAX_LOGIT_CONST=30.0` does not rescue
`TOKAMAX_Q_SEQ_SHARDS=2`; the branch remains fast but numerically unusable.
