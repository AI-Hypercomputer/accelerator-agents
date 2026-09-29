---
title: "Qwen3 JAX v181 - no MLP intermediate sac with Tokamax CE"
type: experiment
hypothesis: "After collective-matmul-none, scoped VMEM tuning, and the v169 no-MLP-intermediate schedule, Tokamax streamed CE may recover enough memory/layout benefit to beat MaxText CE despite its earlier custom-call overhead."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-tokamax-ce
status: refuted
tags: [qwen3-cc, jax, flax-nnx, tokamax-ce, ce, mlp, sharding-constraint, splash, tokamax-splash, collective-matmul, scoped-vmem, activation-sharding, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v181 - no MLP intermediate sac with Tokamax CE

## Hypothesis under test

**Hypothesis**: v031 refuted Tokamax CE on the older v041 stack: it fit
batch-4 seq8192 but only reached 45,461 tok/s / 32.6% MFU because the custom-call
bucket grew. Since then, the current stack added collective-matmul-none, scoped
VMEM 100352, scheduler simplifications, and v169's no-MLP-intermediate `_sac`
schedule. Retesting Tokamax CE on this newer frontier may show whether the CE
overhead was partly schedule-family dependent.

**Mechanism**: Reuse `v169-no-mlp-sac` and change only the CE implementation:
replace `--use_maxtext_ce=True` with
`--use_tokamax_ce=True --tokamax_ce_impl=mosaic_tpu --use_maxtext_ce=False`.

Keep scan, remat, tokamax Splash, activation sharding, batch/sequence shape,
sharding plan, and XLA flags fixed.

**Predicted signal**: Support requires clean compile/loss and throughput above
the confirmed v169/v170 frontier band (57,935-57,947 tok/s average, 58,051 best
observed). Partial support would be a real HLO/profile change with CE/custom-call
time lower than v031 and throughput at least in the current frontier band.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion below the v169/v170 frontier.

## Setup

GKE workload `alekseyv-qwen3-v181-tokamaxce-mlpsac`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v181-no-mlp-sac-tokamax-ce`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v170 stack with `--xla_tpu_scoped_vmem_limit_kib=100352`.
- **Command shape**: `--use_scan=True --use_remat=True --offload_remat=False --use_splash=True --use_tokamax_ce=True --tokamax_ce_impl=mosaic_tpu --use_maxtext_ce=False --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit was made. The model/loader CPU equivalence gate was rerun
immediately before this experiment during v180 documentation and passed:
logits max delta `2.682e-07`, loss delta `0`, all 25 gradients matched, max
gradient delta `7.823e-08`.

Tokamax CE is a mathematically equivalent fused/streamed CE replacement for
Qwen3's ordinary CE objective; correctness signal for this TPU-only path is
finite loss matching the known synthetic-loss curve.

## Results

Completed cleanly on both workers with normal loss and `EXIT_CODE=0`.

- Worker 0: 53,280 tok/s average, 6,660 tok/s/chip, 38.2% MFU.
- Worker 1: 53,265 tok/s average, 6,658 tok/s/chip, 38.2% MFU.
- Best observed steady step: 53,515 tok/s.
- Final loss: 12.0526 on both workers.

This is a large improvement over old v031 Tokamax CE (45,461 tok/s / 32.6% MFU)
because the surrounding schedule family is much better, but it remains far
below the v169/v170 MaxText-CE frontier (57,935-57,947 tok/s repeated average,
58,051 best observed, 41.5% MFU). The loss curve is finite and close to the
known synthetic-random curve, so this is a performance refutation rather than a
correctness failure.

## Profile

Profile files were written under
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v181-no-mlp-sac-tokamax-ce/plugins/profile/2026_06_04_21_20_51/`
for both TPU hosts:

- `gke-tpu-46dd3e54-g30s.xplane.pb`
- `gke-tpu-46dd3e54-rkr2.xplane.pb`

The step log shows steady-state steps around 4.90-4.95 s versus v169/v170
around 4.52 s.

## HLO Dump

Fresh train-step HLO was emitted. The duplicated train-step dumps
`module_0109` and `module_0308` are byte-identical.

- Optimized HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v181-no-mlp-sac-tokamax-ce/hlo/module_0308.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 791,173 bytes
- SHA256: `caca667e8d8e01a478379131a36d943f4007cbd428cdfab5db7ee5b3d3b71ac3`
- Text counts: all-gather 139, all-reduce 36, collective-permute 3,
  convolution 58, custom-call 43, copy 423, fusion 558, `tokamax` metadata 18.
- Memory report: 23.15 GiB total bytes used, down from v169's 26.86 GiB.
  Tokamax CE does reduce train-step memory, but the lower memory footprint does
  not translate into faster wall time.

## Verdict

Refuted. Tokamax CE is valid and materially reduces HBM versus MaxText CE on the
current v169 schedule family, but it is still too slow on v6e-8 at this shape.
Do not carry `--use_tokamax_ce=True`; keep MaxText CE unless a different
streamed/fused CE implementation removes the custom-call overhead.
