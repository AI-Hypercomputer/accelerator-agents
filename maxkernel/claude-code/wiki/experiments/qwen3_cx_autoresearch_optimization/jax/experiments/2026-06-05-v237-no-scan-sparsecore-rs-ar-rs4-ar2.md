---
title: "Qwen3 JAX v237 - RS4+AR2 SparseCore latency bracket"
type: experiment
hypothesis: "RS3+AR2 reproduced a same-band alternative to RS2+AR2; raising only the reduce-scatter SparseCore latency multiplier to 4 checks whether the compiler has another useful scheduling transition above the RS2/RS3 plateau."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs4-ar2
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, latency-model, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v237 - RS4+AR2 SparseCore latency bracket

## Hypothesis under test

**Hypothesis**: v236 confirmed RS3+AR2 as an equivalent schedule to the v230
RS2+AR2 frontier, while RS1+AR2 was slower. A one-step high-side bracket at
RS4+AR2 checks whether the SparseCore reduce-scatter latency model has another
useful transition above `3`, or whether the RS2/RS3 plateau is the local peak.

**Mechanism**: Keep v236 fixed except change
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3` to `4`. Keep
`--xla_tpu_sparse_core_all_reduce_latency_multiplier=2`.

**Predicted signal**: Support requires clean compile/loss and throughput above
the v230/v236 band (**59,749-59,750 tok/s**) with no HBM/MXU regression, or a
lower profile step time with the same memory profile.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed
HLO with worse profile, or clean completion at or below the v230/v236 band.

## Setup

GKE workload `alekseyv-q3-v237-rs4-ar2`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v237-no-scan-sparsecore-rs-ar-rs4-ar2`
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

Completed cleanly as workload `alekseyv-q3-v237-rs4-ar2` with `EXIT_CODE=0`
on both workers and normal loss ending at 12.0459.

- Worker0 pod: `alekseyv-q3-v237-rs4-ar2-slice-job-0-0-mb9rc` averaged
  **59,725 tok/s** over steps 2-19, **59,773 tok/s** over steps 15-19, and
  **59,741 tok/s** excluding profiled steps. Best post-warmup step:
  **59,912 tok/s**.
- Worker1 pod: `alekseyv-q3-v237-rs4-ar2-slice-job-0-1-8lpdh` averaged
  **59,704 tok/s** over steps 2-19, **59,776 tok/s** over steps 15-19, and
  **59,745 tok/s** excluding profiled steps. Best post-warmup step:
  **59,913 tok/s**.

This is below v230/v236 on the primary full-window metric and does not improve
late/no-trace windows.

## Profile

XProf run:
`2026-06-05-qwen3-jax-v237-no-scan-sparsecore-rs-ar-rs4-ar2/2026_06_05_08_22_46`.

- Step time: **4404.0 ms** average.
- MXU utilization: **67.5%**.
- Peak HBM: **31.16 GiB**.
- Top train-step buckets include `convolution fusion`
  **56,799.2 ms / 53.7%**, `custom-call` **29,413.4 ms / 27.8%**, and
  `loop fusion` **10,694.2 ms / 10.1%**.

## HLO Dump

Optimized train-step HLO changed from v236:

- SHA256:
  `6277ba8867f83440a8606ec916b3e2ab108955756ee053ee29d3e03d80fbba17`
- Size: **19,895,634 bytes**
- Lines: **119,264**

## Verdict

Refuted. RS4+AR2 selected a different HLO, raised peak HBM from the RS2/RS3
band's **30.93 GiB** to **31.16 GiB**, worsened XProf step time to
**4404.0 ms**, and landed below the v230/v236 throughput band. Close the
high-side reduce-scatter latency bracket; keep RS2+AR2 or same-band RS3+AR2.
