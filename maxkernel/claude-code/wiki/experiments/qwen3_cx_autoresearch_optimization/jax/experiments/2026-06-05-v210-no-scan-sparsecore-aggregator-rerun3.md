---
title: "Qwen3 JAX v210 - SparseCore collective aggregator corrected confirmation rerun"
type: experiment
hypothesis: "The marginal v208 SparseCore collective-aggregator lift over v206 is reproducible when relaunched with the exact accepted v208 flag names."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-all-collectives-aggregator-rerun3
status: supported
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collective-aggregator, collectives, all-gather, reduce-scatter, all-reduce, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8, rerun]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v210 - SparseCore collective aggregator corrected confirmation rerun

## Hypothesis under test

**Hypothesis**: v208 reported a marginal new best of 59,372 tok/s / 42.6% MFU
after enabling `--xla_tpu_enable_sparse_core_collective_aggregator=true` on top
of the v206 all-collective SparseCore-offload stack. A corrected exact rerun
should reproduce the same throughput band if that runtime-level lift is real.

**Mechanism**: Corrected relaunch after v209 failed before compile due two
invalid flag names. Use the accepted v208 flag names from the train-step
flagfile, including
`--xla_tpu_enable_latency_hiding_layer_scheduler=false` and
`--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=true`.

**Predicted signal**: Support requires clean compile/loss and best-worker
throughput at or above the v208 band, roughly 59,36x-59,38x tok/s, with the
same train-step HLO or a clearly related cached program.

**Falsification criterion**: Clean completion back in the v206 band
(approximately 59,33x-59,35x tok/s), invalid loss, missing profile, or
compile/runtime failure.

## Setup

GKE workload `alekseyv-qwen3-v210-noscan-scagg-r3`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v210-no-scan-sparsecore-aggregator-rerun3`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This is a corrected runtime-flag rerun and
preserves model math.

## Results

Completed cleanly on `alekseyv-qwen3-v210-noscan-scagg-r3` with `EXIT_CODE=0`
on both workers.

- Worker 0 reported **59,370 tok/s** aggregate, **7,421 tok/s/chip**, and
  **42.6% MFU**.
- Worker 1 reported **59,342 tok/s** aggregate, **7,418 tok/s/chip**, and
  **42.5% MFU**.
- Final measured steady steps after profile export stayed in the v208 band:
  worker0 steps 15-19 were 59,528, 59,439, 59,381, 59,364, and 59,331 tok/s;
  worker1 steps 15-19 were 59,527, 59,434, 59,379, 59,360, and 59,328 tok/s.
- Loss decreased normally through step 19, ending at **12.0459**.

This confirms v208's marginal best-worker lift over the v206 all-collective
SparseCore-offload frontier. The confirmed aggregator frontier is now
59,370-59,372 tok/s / 42.6% MFU best-worker reported, versus the v206 confirmed
floor of 59,351 tok/s / 42.6% MFU.

## Profile

Profile capture completed on both hosts:

- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v210-no-scan-sparsecore-aggregator-rerun3/plugins/profile/2026_06_05_02_15_12/gke-tpu-964065d9-9vh8.{trace.json.gz,xplane.pb}`
- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v210-no-scan-sparsecore-aggregator-rerun3/plugins/profile/2026_06_05_02_15_12/gke-tpu-964065d9-c180.{trace.json.gz,xplane.pb}`

## HLO Dump

Fresh train-step HLO was emitted:

- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v210-no-scan-sparsecore-aggregator-rerun3/hlo/module_0294.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `d67433daa35f81edbfbd0fb25d79bd4cd210fc9697bd39133cd50dfe11da9573`
- Compiled memory: `29700367320` bytes / **27.66 GiB**.

The optimized HLO is byte-identical to v205 and v208. The aggregator effect is
runtime scheduling/placement only, not an optimized-HLO structural change.

## Verdict

**Supported.** v210 confirms v208's marginal aggregator improvement. Carry
`--xla_tpu_enable_sparse_core_collective_aggregator=true` as part of the
frontier, with the caveat that its measured gain is only about +0.03-0.04% over
the v206 no-aggregator stack.
