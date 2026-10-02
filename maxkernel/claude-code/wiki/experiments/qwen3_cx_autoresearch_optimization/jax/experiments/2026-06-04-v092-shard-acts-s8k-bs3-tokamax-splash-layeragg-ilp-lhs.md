---
title: "Qwen3 JAX v092 - shard acts s8k bs3 tokamax splash layeragg ILP LHS"
type: experiment
hypothesis: "ILP latency-hiding scheduling may compose with the layer-scheduler plus sparse-core-aggregator frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted/tie
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, latency-hiding, layer-scheduler, sparse-core-collective-aggregator, ilp-scheduler, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, refuted, tie, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v092 - shard acts s8k bs3 tokamax splash layeragg ILP LHS

## Hypothesis under test

**Hypothesis**: v089/v090 establish the layer-scheduler plus sparse-core
collective aggregator pair as the current supported scheduler frontier. v086
showed ILP LHS alone is valid and in-band. Forcing ILP on top of the new
layeragg frontier may select a better schedule than the auto scheduler used by
v089/v090.

**Mechanism**: Keep the v090 frontier fixed and add only
`--xla_tpu_enable_ilp_latency_hiding_scheduler=true`. Do not carry BRKGA from
v091.

**Predicted signal**: Loss stays valid and throughput exceeds the v090
51,033 tok/s best-observed worker value or at least repeats the 51,026+
supported band with a profile win.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v089/v090 throughput band without a profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v092-layerilp`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v092-shard-acts-s8k-bs3-tokamax-splash-layeragg-ilp-lhs`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v090 bundle plus
  `--xla_tpu_enable_ilp_latency_hiding_scheduler=true`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v092-layerilp` completed cleanly on 2026-06-04
with normal loss, but did not improve the v089/v090 frontier.

- Worker 0: 51,016 tok/s, 36.6% MFU
- Worker 1: 50,991 tok/s, 36.6% MFU
- Final loss: 12.0508 on both workers
- Exit code: 0

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v092-shard-acts-s8k-bs3-tokamax-splash-layeragg-ilp-lhs/2026_06_04_06_27_38`

Profile:

- MXU utilization: 56.8%
- Total time: 92,768.7 ms; idle 275.0 ms / 0.3%
- `jit_train_step`: 92,493.0 ms / 99.7%
- Convolution fusion: 45,720.6 ms / 49.3%
- Custom-call: 22,056.2 ms / 23.8%
  - `splash_mha_dkv_no_residuals.1`: 11,641.2 ms / 12.5%
  - `splash_mha_fwd_residuals.2/3`: 10,415.0 ms / 11.2%
- Loop fusion: 16,713.5 ms / 18.0%
- Peak HBM: 28.32 GiB of 31.25 GiB; 2.93 GiB free; 0.05% fragmentation

## HLO Dump

HLO dump exists under
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v092-shard-acts-s8k-bs3-tokamax-splash-layeragg-ilp-lhs/hlo/`.

- HLO object count: 1,615
- Train-step after-optimization dump:
  - `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Train-step after-optimization size: 1,375,271 bytes
- Flagfile confirms:
  - `--xla_tpu_enable_latency_hiding_layer_scheduler='true'`
  - `--xla_tpu_enable_sparse_core_collective_aggregator='true'`
  - `--xla_tpu_enable_ilp_latency_hiding_scheduler='true'`
  - `--xla_tpu_enable_brkga_latency_hiding_scheduler=auto`
  - `--xla_tpu_enable_collective_compute_fusion=auto`
  - `--xla_tpu_scoped_vmem_limit_kib=81920`
  - `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments='true'`
  - `--xla_latency_hiding_scheduler_enable_selective_resources='true'`

## Verdict

Refuted/tie. ILP LHS is valid on top of the layeragg frontier, but it does not
improve throughput or profile shape and falls below the v089/v090 repeated
51,026+ tok/s band. Do not carry ILP forward.
