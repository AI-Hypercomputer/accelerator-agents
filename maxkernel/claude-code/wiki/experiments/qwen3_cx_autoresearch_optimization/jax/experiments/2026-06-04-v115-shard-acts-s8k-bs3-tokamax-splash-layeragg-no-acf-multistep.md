---
title: "Qwen3 JAX v115 - shard acts s8k bs3 tokamax splash layeragg no acf multistep"
type: experiment
hypothesis: "Async collective fusion multiple-steps mode may be neutral or harmful under the v090 layeragg scheduler after base all-gather and reduce-scatter fusion remain enabled."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, async-collective-fusion, multiple-steps, latency-hiding, scheduler, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v115 - shard acts s8k bs3 tokamax splash layeragg no acf multistep

## Hypothesis under test

**Hypothesis**: v101 proved base async collective fusion is required, v113
proved the all-gather subtype is required, and v114 showed the reduce-scatter
subtype ablation is only a tie. The remaining unisolated ACF flag in the v090
frontier is `--xla_tpu_enable_async_collective_fusion_multiple_steps=true`.
Disabling multiple-step fusion while keeping base/all-gather/reduce-scatter
fusion enabled may relax scheduling without reintroducing the all-gather
bucket.

**Mechanism**: Keep the valid v090 qseq1 frontier fixed and change only
`--xla_tpu_enable_async_collective_fusion_multiple_steps=true` to `false`.

**Predicted signal**: Loss stays finite and throughput exceeds the v090
51,033 tok/s best-observed / 51,026+ repeated frontier, or XProf shows lower
train-step wall time without all-gather/custom-call regression.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v090 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v115-noacfms`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v115-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-acf-multistep`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v090 scheduler frontier, except
  `--xla_tpu_enable_async_collective_fusion_multiple_steps=false`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Completed cleanly.

- **Workload**: `alekseyv-qwen3-v115-noacfms`
- **Worker 0**: 50,998 tok/s (6,375 tok/s/chip), approx 36.6% MFU
- **Worker 1**: 51,023 tok/s (6,378 tok/s/chip), approx 36.6% MFU
- **Final loss**: 12.0510 on both workers
- **Profile window**: steps 12-14
- **XProf run**: `2026-06-04-qwen3-jax-v115-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-acf-multistep/2026_06_04_10_37_12`

## Profile

XProf shows a valid in-band profile, but no wall-time win over the repeated
v089/v090 frontier.

- **Total time**: 92,836.8 ms
- **Idle time**: 290.2 ms (0.3%)
- **Compute time**: 92,546.7 ms
- **MXU utilization**: 56.9%
- **Peak HBM**: 28.32 GiB
- **Top op buckets**:
  - convolution fusion: 45,794.2 ms (49.3%)
  - custom-call: 22,080.0 ms (23.8%)
  - DKV custom-call: 11,649.0 ms (12.5%)
  - forward residuals: 5,228.9 ms + 5,202.1 ms (11.2% combined)
  - loop fusion: 16,688.4 ms (18.0%)
  - data formatting: 3,690.8 ms (4.0%)
  - collective-permute-start: 1,179.2 ms (1.3%)

## HLO Dump

Train-step optimized HLO was regenerated under the run prefix:

- **Train-step HLO**: `hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- **Size**: 1,188,888 bytes
- **Object count**: 1,619
- **Content hash**: `e5cf4844bafcebfacbeb1ef3d883dc3fcd0368005d33d4a123afae4efc1fb657`
- **Train-step program hash**: `6644865601235731379`

## Verdict

**Refuted/tie.** Disabling only
`--xla_tpu_enable_async_collective_fusion_multiple_steps` changes the compiled
train-step hash/HLO shape, but it does not beat the v090 51,033 tok/s
best-observed worker or the 51,026+ repeated frontier band. Keep multiple-step
async collective fusion enabled in the qseq1 layeragg frontier. The ACF subtype
sweep is now closed: base ACF and all-gather fusion are required, reduce-scatter
fusion and multiple-steps ablations are valid but do not improve throughput.
