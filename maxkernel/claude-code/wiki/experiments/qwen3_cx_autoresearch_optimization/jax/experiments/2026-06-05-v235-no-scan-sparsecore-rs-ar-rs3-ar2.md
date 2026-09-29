---
title: "Qwen3 JAX v235 - RS3+AR2 SparseCore latency interaction"
type: experiment
hypothesis: "The v230 frontier combines RS latency multiplier 2 with AR latency multiplier 2; RS latency multiplier 3 had an unreproduced high-water before AR2, so testing RS3+AR2 isolates whether the joint latency model can recover that schedule."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs3-ar2
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, latency-model, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v235 - RS3+AR2 SparseCore latency interaction

## Hypothesis under test

**Hypothesis**: v225's RS latency multiplier `3` produced a one-run high-water
near the current frontier before AR2 was added, but v226 failed to reproduce
that exact throughput. v230 later confirmed AR latency multiplier `2` as a real
gain on RS2. Testing RS3+AR2 isolates whether the combined latency model can
recover a better schedule than either RS3 alone or RS2+AR2.

**Mechanism**: Keep v230 fixed except change
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=2` to `3`.

**Predicted signal**: Support requires clean compile/loss and throughput above
v230's **59,749 tok/s** confirmed value, or a lower profile step time with no
HBM/MXU regression.

**Falsification criterion**: Compile/runtime failure, invalid loss, HLO/memory
regression, or clean completion at or below v230's confirmed band.

## Setup

GKE workload `alekseyv-q3-v235-rs3-ar2`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v235-no-scan-sparsecore-rs-ar-rs3-ar2`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False
  --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4
  --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This is an XLA runtime latency-model flag isolate
and preserves model math.

## Result

Completed cleanly as workload `alekseyv-q3-v235-rs3-ar2` with `EXIT_CODE=0`
on both workers and normal loss ending at 12.0459.

- Worker0 pod: `alekseyv-q3-v235-rs3-ar2-slice-job-0-0-fp4rb` averaged
  **59,724 tok/s** over steps 2-19, **59,789 tok/s** over steps 15-19, and
  **59,763 tok/s** excluding profiled steps. Best post-warmup step:
  **59,912 tok/s**.
- Worker1 pod: `alekseyv-q3-v235-rs3-ar2-slice-job-0-1-54zm8` averaged
  **59,747 tok/s** over steps 2-19, **59,788 tok/s** over steps 15-19, and
  **59,762 tok/s** excluding profiled steps. Best post-warmup step:
  **59,906 tok/s**.

This is a near-tie with v230: the best full-window average is **2 tok/s**
below v230's **59,749 tok/s**, while late/no-trace windows are effectively
matched or slightly better.

## Profile

XProf run:
`2026-06-05-qwen3-jax-v235-no-scan-sparsecore-rs-ar-rs3-ar2/2026_06_05_07_58_34`.

- Step time: **4400.9 ms** average.
- MXU utilization: **67.4%**.
- Peak HBM: **30.93 GiB**.
- Top train-step buckets include `convolution fusion`
  **56,762.9 ms / 53.7%**, `custom-call` **29,411.8 ms / 27.8%**, and
  `loop fusion` **10,683.9 ms / 10.1%**.

## HLO Dump

Optimized train-step HLO changed from v230 and is the RS3+AR2 schedule:

- SHA256:
  `d4d46797cc4508667e0aad8b0a75bf6485f046806db6ced027a159f89ee80b5c`
- Size: **19,909,611 bytes**
- Lines: **119,349**

## Verdict

Near-tie / rerun. RS3+AR2 did not beat v230 on the primary full-window metric,
but it matched the profile band and had slightly stronger late/no-trace
throughput. Run an exact rerun before closing this interaction.
