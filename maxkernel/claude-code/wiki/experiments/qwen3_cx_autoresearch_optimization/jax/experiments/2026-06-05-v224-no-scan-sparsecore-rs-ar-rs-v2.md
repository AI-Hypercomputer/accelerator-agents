---
title: "Qwen3 JAX v224 - no-scan SparseCore RS/AR with reduce-scatter v2"
type: experiment
hypothesis: "The confirmed v220/v221 frontier depends on reduce-scatter SparseCore offload; enabling the TPU recipe reduce-scatter v2 flag may improve RS offload scheduling without the invalid ND reduce-scatter path."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs-ar-rsv2
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, all-gather, reduce-scatter-v2, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v224 - no-scan SparseCore RS/AR with reduce-scatter v2

## Hypothesis under test

**Hypothesis**: v220/v221 confirm that reduce-scatter and all-reduce should stay
on SparseCore while all-gather should stay on async collective fusion. TPU
recipe stacks often enable `--xla_tpu_enable_sparse_core_reduce_scatter_v2=true`
with SparseCore reduce-scatter offload. Adding only this v2 flag may improve
reduce-scatter scheduling without using v223's invalid ND reduce-scatter path.

**Mechanism**: Keep v220/v221 fixed and add only
`--xla_tpu_enable_sparse_core_reduce_scatter_v2=true`.

**Predicted signal**: Support requires clean compile/loss and throughput above
v220/v221, preferably with lower `async-done`, lower all-reduce-scatter bucket,
or higher MXU in XProf.

**Falsification criterion**: Compile/runtime failure, invalid loss, unchanged
throughput, or regression below v220/v221.

## Setup

GKE workload `alekseyv-qwen3-v224-sc-rsar-rsv2`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v224-no-scan-sparsecore-rs-ar-rs-v2`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False
  --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4
  --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This is an XLA runtime/offload flag isolate and
preserves model math.

## Result

Completed cleanly as workload `alekseyv-qwen3-v224-sc-rsar-rsv2` with
`EXIT_CODE=0` on both workers. Loss stayed normal and ended at 12.0461.

- Worker0 pod: `alekseyv-qwen3-v224-sc-rsar-rsv2-slice-job-0-0-ttptw` on
  `gke-tpu-0d0013c1-l5jf`
  - Avg steps 2-19: **59,204 tok/s**
  - Best step 2-19: **59,308 tok/s**
  - Avg steps 15-19: **59,241 tok/s**
  - Avg steps 2-19 excluding profiled steps 12-14: **59,247 tok/s**
  - Tok/s/chip from full average: **7,401**
  - Nominal MFU: **42.5%**
- Worker1 pod: `alekseyv-qwen3-v224-sc-rsar-rsv2-slice-job-0-1-lgw4f` on
  `gke-tpu-0d0013c1-9rkl`
  - Avg steps 2-19: **59,133 tok/s**
  - Best step 2-19: **59,310 tok/s**
  - Avg steps 15-19: **58,895 tok/s** (step 16 outlier at 57,487 tok/s)
  - Avg steps 2-19 excluding profiled steps 12-14: **59,131 tok/s**
  - Tok/s/chip from full average: **7,392**
  - Nominal MFU: **42.4%**

Even the cleaner worker0 no-trace window is below v220/v221 by roughly
**216 tok/s**, and the full-window result is below the v208/v210/v220 frontier
band.

## Profile

Profile capture completed with one timestamp per host:

- `2026-06-05-qwen3-jax-v224-no-scan-sparsecore-rs-ar-rs-v2/2026_06_05_05_52_17`
  (`gke-tpu-0d0013c1-l5jf`)
- `2026-06-05-qwen3-jax-v224-no-scan-sparsecore-rs-ar-rs-v2/2026_06_05_05_52_18`
  (`gke-tpu-0d0013c1-9rkl`)

XProf summaries:

- Average step time: **4442.6 ms** on both profile timestamps
- MXU utilization: **66.8-66.9%**
- Peak memory: **30.93 GiB / 31.25 GiB**

Top buckets for `2026_06_05_05_52_17`:

| Bucket | Time | Share | Note |
|---|---:|---:|---|
| `convolution fusion` | 28,355.4 ms | 53.1% | one-host profile, about half v221 aggregate count |
| `custom-call` | 14,711.6 ms | 27.6% | Splash attention |
| `loop fusion` | 5,363.0 ms | 10.0% | bandwidth-heavy fusions |
| `custom fusion` | 1,739.0 ms | 3.3% | worse share than v221 |
| `data formatting` | 1,457.5 ms | 2.7% | copies/formatting |
| `async-done` | 711.4 ms | 1.3% | worse share than v221's 1.0% |
| `all-reduce-scatter fusion` | 155.3 ms | 0.3% | flat share |
| `all-gather` | 111.3 ms | 0.2% | flat share |

The profile confirms the stdout regression: step time is worse than v221
(4429.0 ms) and MXU is lower than v221 (67.4%).

## HLO Dump

- Optimized train-step HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v224-no-scan-sparsecore-rs-ar-rs-v2/hlo/xla-dump/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: **19,954,794 bytes**
- SHA256:
  `a33c7940efea974ff9a7dcf2704fa4d08f79e5d82e1821a59a553805ceca81d5`

The optimized HLO text is byte-identical to v220/v221/v222. The regression is a
runtime scheduling effect of the reduce-scatter v2 flag, not a graph rewrite.

## Verdict

**Refuted.** Adding
`--xla_tpu_enable_sparse_core_reduce_scatter_v2=true` to the confirmed v220/v221
frontier compiles and runs but regresses throughput, step time, and MXU. Do not
carry reduce-scatter v2 in this native-JAX shape.
