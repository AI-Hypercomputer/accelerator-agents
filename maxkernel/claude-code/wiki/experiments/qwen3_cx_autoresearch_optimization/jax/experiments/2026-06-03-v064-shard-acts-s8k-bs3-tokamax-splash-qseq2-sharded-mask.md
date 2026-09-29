---
title: "Qwen3 JAX v064 - shard acts s8k bs3 tokamax splash qseq2 sharded mask"
type: experiment
hypothesis: "Passing q_seq_shards through to Tokamax's causal-mask shard_count may make the fast qseq2 path numerically valid."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v064-qseq2-sharded-mask
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, qseq2, remat, maxtext-ce, activation-sharding, long-seq, invalid, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v064 - shard acts s8k bs3 tokamax splash qseq2 sharded mask

## Hypothesis under test

**Hypothesis**: v050/v051's fast `TOKAMAX_Q_SEQ_SHARDS=2` path produced NaNs
because the wrapper split the q-sequence kernel but left Tokamax's causal mask
at `shard_count=1`. Setting `CausalMask(..., shard_count=q_seq_shards)` may
make the qseq2 path numerically valid.

**Mechanism**: Keep the v056 frontier stack fixed and change only
`splash_attn.py` so the Tokamax causal mask receives the same shard count as
`q_seq_shards`. Runtime keeps `TOKAMAX_Q_SEQ_SHARDS=2` and
`TOKAMAX_MAX_LOGIT_CONST=30.0`.

**Predicted signal**: Loss stays near the v056 trajectory, ending around
12.05, while throughput remains closer to v051's 54,930 tok/s / 39.4% nominal
MFU than to v056's 50,383 tok/s / 36.1% MFU.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion at or below v056's 50,383 tok/s / 36.1% MFU.

## Setup

GKE workload `alekseyv-qwen3-v064`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v064-qseq2-sharded-mask`
- **Base image**: `v041-shard-acts`
- **Code change**: in `/app/trainer/splash_attn.py`, construct Tokamax
  `CausalMask(shape=..., shard_count=_q_seq_shards)` instead of leaving
  `shard_count=1`.
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v064-shard-acts-s8k-bs3-tokamax-splash-qseq2-sharded-mask`
- **Mesh**: `fsdp=8,tp=1`
- **Environment change**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_Q_SEQ_SHARDS=2 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag**: keep v056's `--xla_latency_hiding_scheduler_rerun=3`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Invalid run:

- Worker 0: average 54,795 tok/s total, 6,849 tok/s/chip, 39.3% nominal MFU.
- Worker 1: average 54,778 tok/s total, 6,847 tok/s/chip, 39.3% nominal MFU.
- Step 0 loss was finite at 12.0979 on both workers.
- Loss became NaN at step 1 on both workers and stayed NaN through step 19.

Passing `q_seq_shards` through to the causal mask's `shard_count` does not fix
the qseq2 correctness failure. The throughput remains in the same fast but
invalid band as v050/v051.

## Profile

- **XProf run**:
  `2026-06-03-qwen3-jax-v064-shard-acts-s8k-bs3-tokamax-splash-qseq2-sharded-mask/2026_06_03_21_22_46`
- **MXU utilization**: 54.1%.
- **Peak HBM**: 28.33 GiB, 2.914 GiB free.
- **Device idle**: 0.3%.

Top device buckets:

- Convolution fusion: 51.5%.
- Loop fusion: 20.0%.
- Custom-call: 17.2%.
- Data formatting: 4.4%.

The invalid qseq2 profile still cuts forward residual custom-call time sharply:
`splash_mha_fwd_residuals.{2,3}` total about 3.8%, versus about 11.1% on the
valid v056 frontier. DKV remains 13.3%, slightly above v056's 12.3%.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v064-shard-acts-s8k-bs3-tokamax-splash-qseq2-sharded-mask/hlo/`
- **Size**: 17.55 MiB
- **Object count**: 24 objects plus two empty JIT subdirectories.

## Verdict

**Invalid.** The causal-mask shard count was not the missing qseq2 correctness
piece. `TOKAMAX_Q_SEQ_SHARDS=2` remains unusable for this Qwen3 target shape
because the loss becomes NaN from the first optimizer update.

Next: stop trying qseq2 as a runtime knob. The useful signal from v050/v051/v064
is that forward residual custom-call time is the remaining exploitable bucket,
but it needs a correctness-preserving mechanism on the qseq1 path.
