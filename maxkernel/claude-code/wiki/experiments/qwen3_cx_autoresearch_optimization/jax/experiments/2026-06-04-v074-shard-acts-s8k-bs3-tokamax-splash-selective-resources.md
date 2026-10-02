---
title: "Qwen3 JAX v074 - shard acts s8k bs3 tokamax splash selective resources"
type: experiment
hypothesis: "Enabling latency-hiding selective resources on top of the v072/v073 frontier may improve scheduler placement around custom-calls and collectives."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: supported
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, scheduler, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, supported, frontier, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v074 - shard acts s8k bs3 tokamax splash selective resources

## Hypothesis under test

**Hypothesis**: v072/v073 showed a tiny but repeatable gain from forcing TPU
custom-call scoped-VMEM adjustments. The flagfile still has
`--xla_latency_hiding_scheduler_enable_selective_resources=false`. Enabling
selective resources may improve latency-hiding placement around the remaining
Splash custom-call, loop-fusion, and collective regions.

**Mechanism**: Keep the v072/v073 frontier fixed and add only
`--xla_latency_hiding_scheduler_enable_selective_resources=true`.

**Predicted signal**: Loss stays valid and throughput exceeds the v072/v073
band, ideally above 50,793 tok/s / 36.4% MFU or with lower profile total time.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion at or below the v072/v073 band.

## Setup

Planned GKE workload `alekseyv-qwen3-v074-selective`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v074-shard-acts-s8k-bs3-tokamax-splash-selective-resources`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: add
  `--xla_latency_hiding_scheduler_enable_selective_resources=true` to the
  v072/v073 bundle.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v074-selective` completed cleanly on
`<your-cluster>`.

| Worker | Avg step | Avg throughput | Tok/s/chip | MFU | Loss end |
| --- | ---: | ---: | ---: | ---: | ---: |
| worker0 | 3.85 s | 51,013 tok/s | 6,377 | 36.6% | 12.0508 |
| worker1 | 3.85 s | 51,021 tok/s | 6,378 | 36.6% | 12.0508 |

This is a material improvement over the v072/v073 50.78k tok/s band and the
first clear move past 36.4% MFU.

## Profile

XProf run:
`2026-06-04-qwen3-jax-v074-shard-acts-s8k-bs3-tokamax-splash-selective-resources/2026_06_04_03_26_50`.

- Profile summary: MXU 56.8% on TPU v6 Lite.
- Op profile: 92,685.0 ms total, 0.2% idle, `jit_train_step` 99.8%.
- Top train-step buckets: convolution fusion 49.3%, custom-call 23.8%, loop
  fusion 18.0%, data formatting 4.1%, collective-permute-start 1.0%, reduce
  0.8%.
- Splash split: `splash_mha_dkv_no_residuals.1` 12.6%;
  `splash_mha_fwd_residuals.3` 5.6%;
  `splash_mha_fwd_residuals.2` 5.6%.
- Memory profile: peak HBM 28.32 GiB, stack reservation 22.508 GiB, heap
  5.8103 GiB, free 2.9278 GiB, utilization 90.62%.

The op mix is broadly unchanged from v072/v073, but the profile window total
time falls and MXU rises. This supports the scheduler-placement hypothesis.

## HLO Dump

HLO dump:
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v074-shard-acts-s8k-bs3-tokamax-splash-selective-resources/hlo/`.

The train-step dump is `module_0267.jit_train_step.cl_854318611` with a
1,375,271-byte `after_optimizations.txt`; the dump prefix contains 1,615
objects. The flagfile confirms `--xla_tpu_scoped_vmem_limit_kib=81920`,
`--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments='true'`,
`--xla_latency_hiding_scheduler_enable_selective_resources='true'`,
`--xla_latency_hiding_scheduler_rerun=3`, and
`--xla_tpu_enable_ag_backward_pipelining=true`.

## Verdict

**Supported.** Enabling selective resources on top of the v072/v073 scheduler
stack improves throughput to 51,021 tok/s / 36.6% MFU. v075 reruns the same
configuration and confirms the 51k tok/s band, so carry this stack forward as
the scheduler frontier.
