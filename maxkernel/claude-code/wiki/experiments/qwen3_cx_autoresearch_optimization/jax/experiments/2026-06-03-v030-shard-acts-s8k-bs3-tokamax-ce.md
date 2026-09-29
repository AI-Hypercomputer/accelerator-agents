---
title: "Qwen3 JAX v030 - shard acts s8k bs3 tokamax CE"
type: experiment
hypothesis: "Replacing MaxText CE with streamed tokamax CE at the v041 target shape may remove logit materialization and improve MFU or HBM headroom."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, scan, remat, tokamax-ce, activation-sharding, tp1, long-seq, ce, refuted, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v030 - shard acts s8k bs3 tokamax CE

## Hypothesis under test

**Hypothesis**: Replacing MaxText CE with streamed tokamax CE at the v041
target shape may remove logit materialization and improve MFU or HBM headroom.

**Mechanism**: Reuse the v041 stack exactly at the known-fitting shape
(`fsdp=8,tp=1`, seq8192, per-chip batch 3/global batch 24), but set
`--use_tokamax_ce=True --tokamax_ce_impl=mosaic_tpu` and remove
`--use_maxtext_ce=True`. This is a single CE implementation change; scan,
remat, Splash, activation sharding, and batch shape stay fixed.

**Predicted signal**: If streamed CE avoids the v041/v026 logit materialization
cost without introducing a larger fp32 weight/scratch wall, the run should
complete with lower peak HBM and MFU at or above v041's 34.6%. If tokamax's
fp32 CE weight or backward dot dominates, it will OOM or regress.

**Falsification criterion**: Compile/runtime failure, or clean completion at or
below v041's 48,312 tok/s / 34.6% MFU.

## Setup

Planned GKE workload `alekseyv-qwen3-v030`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v030-shard-acts-s8k-bs3-tokamax-ce`
- **Mesh**: `fsdp=8,tp=1`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_tokamax_ce=True --tokamax_ce_impl=mosaic_tpu --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Refuted as a direct replacement for v041 at batch 3. Workload
`alekseyv-qwen3-v030` completed cleanly and greatly reduced peak HBM, but
throughput and MFU regressed versus v041.

- **Worker 0 summary**: 45,077 tok/s, 5,635 tok/s/chip, 32.3% MFU
- **Worker 1 summary**: 45,103 tok/s, 5,638 tok/s/chip, 32.3% MFU
- **v041 comparison**: 48,312 tok/s, 6,039 tok/s/chip, 34.6% MFU
- **Delta**: -6.6% throughput and -2.3 pp MFU
- **Tokamax confirmation**: startup emitted tokamax autotuning for
  `x=(24576,4096)`, `labels=(24576,)`, `w=(4096,151936)`, all fp32, and no
  MaxText CE enable line.

## Profile

- **XProf run**: `2026-06-03-qwen3-jax-v030-shard-acts-s8k-bs3-tokamax-ce/2026_06_03_02_42_59`
- **MXU utilization**: 51.0% vs v041's 53.7%
- **Idle time**: 0.3% from op profile
- **Peak HBM**: 20.77 GiB of 31.25 GiB; 10.4725 GiB free
- **Stack / heap**: 14.9576 GiB stack reservation, 5.8159 GiB heap allocation

Top HLO op self-time from XProf:

| Op family | Self-time |
|-----------|-----------|
| convolution fusion | 40.4% |
| custom-call | 34.0% |
| loop fusion | 16.1% |
| data formatting | 3.7% |
| collective-permute-start | 1.5% |
| all-gather | 0.8% |
| reduce | 0.7% |
| all-reduce | 0.7% |

The CE swap removes most of the logit-memory pressure, but the runtime cost
moves into custom calls. The custom-call bucket is 14.0% forward Splash
residuals, 10.9% Splash DKV, and 7.2% tokamax
`linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu`. Direct batch-3 MFU
therefore regresses even though HBM improves substantially.

## HLO Dump

HLO artifacts were dumped under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v030-shard-acts-s8k-bs3-tokamax-ce/hlo/`

The prefix contains 63 objects totaling 35.69 MiB.

## Verdict

Refuted for same-shape performance: streamed tokamax CE is slower than v041's
MaxText CE at batch 3. However, it is a useful memory mechanism: peak HBM drops
by about 7.56 GiB and leaves more than 10 GiB free, so the next targeted probe is
the same tokamax-CE stack at per-chip batch 4 / global batch 32.
