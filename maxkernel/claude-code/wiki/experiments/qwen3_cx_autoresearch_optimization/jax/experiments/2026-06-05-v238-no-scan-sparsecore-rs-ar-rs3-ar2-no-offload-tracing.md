---
title: "Qwen3 JAX v238 - RS3+AR2 without SparseCore offload tracing"
type: experiment
hypothesis: "The confirmed RS3+AR2 schedule still enables reduce-scatter and all-reduce SparseCore offload tracing; disabling those tracing flags may remove runtime/instrumentation overhead or select a cleaner schedule without changing model math."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs3-ar2-no-offload-tracing
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, offload-tracing, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v238 - RS3+AR2 without SparseCore offload tracing

## Hypothesis under test

**Hypothesis**: v236 confirmed the RS3+AR2 schedule as a frontier-band tie, but
the active SparseCore reduce-scatter and all-reduce offloads still carry tracing
flags. If those tracing flags add overhead or bias scheduling, disabling them
may preserve the RS3+AR2 HLO family while improving throughput/profile.

**Mechanism**: Keep v236 fixed except change:

- `--xla_tpu_enable_reduce_scatter_offload_tracing=true` to `false`
- `--xla_tpu_enable_all_reduce_offload_tracing=true` to `false`

All-gather offload tracing is already `false` because all-gather offload is
disabled in the RS/AR split.

**Predicted signal**: Support requires clean compile/loss and throughput above
the v230/v236 band (**59,749-59,750 tok/s**) or the same throughput with lower
profile step time and unchanged HBM.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed HLO
with worse profile, or clean completion at or below the v230/v236 band.

## Setup

GKE workload `alekseyv-q3-v238-notrace`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v238-no-scan-sparsecore-rs-ar-rs3-ar2-no-offload-tracing`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False
  --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4
  --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This is an XLA runtime tracing-flag isolate and
preserves model math.

## Result

Completed cleanly as workload `alekseyv-q3-v238-notrace` with `EXIT_CODE=0`
on both workers and normal loss ending at 12.0459.

- Worker0 pod: `alekseyv-q3-v238-notrace-slice-job-0-0-fnhkk` averaged
  **59,749 tok/s** over steps 2-19, **59,796 tok/s** over steps 15-19, and
  **59,766 tok/s** excluding profiled steps. Best post-warmup step:
  **59,929 tok/s**.
- Worker1 pod: `alekseyv-q3-v238-notrace-slice-job-0-1-dpr5p` averaged
  **59,730 tok/s** over steps 2-19, **59,793 tok/s** over steps 15-19, and
  **59,766 tok/s** excluding profiled steps. Best post-warmup step:
  **59,929 tok/s**.

This ties the v230 full-window result and is one tok/s below v236, so it is not
a throughput gain. Profile artifacts were much smaller than v236 because the
SparseCore offload tracing records were removed.

## Profile

XProf run:
`2026-06-05-qwen3-jax-v238-no-scan-sparsecore-rs-ar-rs3-ar2-no-offload-tracing/2026_06_05_08_34_43`.

- Step time: **4401.5 ms** average.
- MXU utilization: **67.5%**.
- Peak HBM: **30.93 GiB**.
- Top train-step buckets include `convolution fusion`
  **56,767.8 ms / 53.7%**, `custom-call` **29,411.2 ms / 27.8%**, and
  `loop fusion` **10,682.9 ms / 10.1%**.

## HLO Dump

Optimized train-step HLO is byte-identical to v235/v236:

- SHA256:
  `d4d46797cc4508667e0aad8b0a75bf6485f046806db6ced027a159f89ee80b5c`
- Size: **19,909,611 bytes**
- Lines: **119,349**

## Verdict

Refuted/tie for runtime. Disabling active SparseCore offload tracing reduces
profile volume but does not improve the train-step schedule, throughput, HBM,
or XProf step time. Keep tracing enabled when detailed offload diagnostics are
needed; disabling it is acceptable for smaller profiles but is not a frontier
performance lever.
