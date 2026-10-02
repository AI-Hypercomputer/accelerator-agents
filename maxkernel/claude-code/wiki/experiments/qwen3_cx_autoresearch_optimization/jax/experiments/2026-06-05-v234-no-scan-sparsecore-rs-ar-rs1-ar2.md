---
title: "Qwen3 JAX v234 - RS1+AR2 SparseCore latency interaction"
type: experiment
hypothesis: "The v230 frontier combines RS latency multiplier 2 with AR latency multiplier 2; after AR2 is enabled, lowering only the RS latency multiplier to 1 may reveal whether RS2 is still required or whether the joint latency model is over-penalizing reduce-scatter."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs1-ar2
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, latency-model, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v234 - RS1+AR2 SparseCore latency interaction

## Hypothesis under test

**Hypothesis**: v230's confirmed frontier uses both
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=2` and
`--xla_tpu_sparse_core_all_reduce_latency_multiplier=2`. Earlier v227/v228
showed RS2 helps before AR2 is added, but the interaction has not been isolated.
With AR2 present, reducing only the RS latency multiplier to `1` may retain the
AR2 benefit while recovering a better reduce-scatter schedule.

**Mechanism**: Keep v230 fixed except change
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=2` to `1`.

**Predicted signal**: Support requires clean compile/loss and throughput above
v230's **59,749 tok/s** confirmed value, or a lower profile step time with no
HBM/MXU regression.

**Falsification criterion**: Compile/runtime failure, invalid loss, HLO/memory
regression, or clean completion at or below v230's confirmed band.

## Setup

GKE workload `alekseyv-q3-v234-rs1-ar2`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v234-no-scan-sparsecore-rs-ar-rs1-ar2`
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

Completed cleanly as workload `alekseyv-q3-v234-rs1-ar2` with `EXIT_CODE=0`
on both workers and normal loss ending at 12.0459.

- Worker0 pod: `alekseyv-q3-v234-rs1-ar2-slice-job-0-0-5zg2r` averaged
  **59,643 tok/s** over steps 2-19, **59,683 tok/s** over steps 15-19, and
  **59,656 tok/s** excluding profiled steps. Best post-warmup step:
  **59,801 tok/s**.
- Worker1 pod: `alekseyv-q3-v234-rs1-ar2-slice-job-0-1-vktn5` averaged
  **59,618 tok/s** over steps 2-19, **59,685 tok/s** over steps 15-19, and
  **59,657 tok/s** excluding profiled steps. Best post-warmup step:
  **59,804 tok/s**.

The late/no-trace windows are healthy, but the full-window average remains
below v230's confirmed **59,749 tok/s** frontier.

## Profile

XProf run:
`2026-06-05-qwen3-jax-v234-no-scan-sparsecore-rs-ar-rs1-ar2/2026_06_05_07_47_40`.

- Step time: **4408.3 ms** average.
- MXU utilization: **67.3%**.
- Peak HBM: **30.93 GiB**.
- Top train-step buckets include `convolution fusion`
  **56,752.0 ms / 53.6%**, `custom-call` **29,412.9 ms / 27.8%**, and
  `loop fusion` **10,690.4 ms / 10.1%**.

## HLO Dump

Optimized train-step HLO changed from v230:

- SHA256:
  `02d404f309bfe92d40d098b49274b39bfd5e41e42370f1ea5a9bf8110f50258b`
- Size: **19,956,545 bytes**
- Lines: **119,563**

## Verdict

Refuted/tie. Lowering the reduce-scatter SparseCore latency multiplier from
`2` to `1` under AR2 produces a distinct HLO and slightly worse profile/throughput.
Keep v230's RS2+AR2 stack as the confirmed native-JAX frontier.
