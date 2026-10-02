---
title: "Qwen3 JAX v279 - Confirmed RS3 selective-resources frontier with Tokamax CE batch 5"
type: experiment
hypothesis: "The existing Tokamax streaming CE path may compose better with the newer v273 MLP up-proj-first + RS3/AR2 selective-resources runtime stack than it did on the older v245 RS2/AR2 stack, allowing batch 5/global 40 to fit without the full-vocab logits temp and recover enough throughput to challenge the MaxText-CE batch-4 frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first-runtime-rs3-ar2-selective-resources-tokamax-ce-bs5
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, sparsecore, collectives, scheduler, selective-resources, latency, rs3, ar2, tokamax-ce, ce, batch-scaling, splash, tokamax-splash, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v279 - Confirmed RS3 selective-resources frontier with Tokamax CE batch 5

## Hypothesis under test

**Hypothesis**: v278 showed that direct batch 5 with MaxText CE still fails
because the full `bf16[5,8192,151936]` logits temp alone consumes **11.59
GiB**. v245 showed that Tokamax streaming CE can fit batch 5 on an older
RS2/AR2 SparseCore stack, but was too slow (**55,885 tok/s / 40.0% MFU**).
The newer v273 stack carries MLP up-proj-first, RS3/AR2, selective resources,
and a better profiled step-time schedule. If those runtime improvements reduce
non-CE overhead enough, Tokamax CE batch 5 may recover useful throughput while
avoiding the logits OOM.

**Mechanism**: Reuse the v273 image/runtime stack and switch only the CE path
from MaxText/T5X full-logits CE to existing Tokamax streaming CE, while raising
per-chip batch from `4` to `5`. Keep scoped VMEM `100352`, scheduler
shared-memory limit `100`, RS latency multiplier `3`, AR latency multiplier
`2`, selective resources, SparseCore reduce-scatter/all-reduce offload,
SparseCore collective aggregator, Tokamax Splash, no-scan, remat, MLP
up-proj-first image, and activation sharding.

**Predicted signal**: Support requires clean finite loss, a valid XProf, no
full-vocab logits OOM, and throughput above the confirmed v273 MaxText-CE
frontier (**60,346 tok/s / 43.3% MFU**). A result materially above v245 but
still below v273 is a refutation of Tokamax CE as the next frontier, but useful
evidence that the remaining gap is CE-kernel cost rather than memory.

**Falsification criterion**: Compile/runtime failure, invalid loss, missing
Tokamax CE HLO markers, or a clean run below the v273 batch-4 MaxText-CE
frontier.

## Setup

GKE workload: `alekseyv-q3-v279-tkce-bs5`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v279-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-tokamax-ce-bs5`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_tokamax_ce=True --tokamax_ce_impl=mosaic_tpu --use_maxtext_ce=False --shard_acts=True --batch_size=5 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 40
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This changes only the CE implementation and batch
size relative to v273, using an existing CE path previously validated in this
lane.

## Results

Completed cleanly on both workers with finite loss and `EXIT_CODE=0`.

- Worker0: **56,318 tok/s**, **7,040 tok/s/chip**, **40.4% MFU** over steps
  2-19; loss ended at **12.0496**.
- Worker1: **56,306 tok/s**, **7,038 tok/s/chip**, **40.4% MFU** over steps
  2-19; loss ended at **12.0496**.
- Steady-state post-compile steps were about **5.81-5.82 s**.

This is slightly faster than the older v245 Tokamax-CE batch-5 run
(**55,885 tok/s / 40.0% MFU**) but far below the confirmed v273 MaxText-CE
batch-4 frontier (**60,346 tok/s / 43.3% MFU**).

## Profile

XProf run:
`2026-06-05-qwen3-jax-v279-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-tokamax-ce-bs5/2026_06_05_18_23_40`.

- Step time: **5829.3 ms** average, **7.0 ms** standard deviation.
- MXU utilization: **64.4%**.
- Op profile: `jit_train_step` accounts for **99.9%** of compute time;
  device idle is **0.1%**, so the slowdown is on-device work/scheduling rather
  than cluster starvation.
- Top op families by time: convolution fusion **32,185 ms / 46.0%**,
  custom-call **26,349 ms / 37.7%**, loop fusion **6,791 ms / 9.7%**, data
  formatting **1,927 ms / 2.8%**, custom fusion **1,170 ms / 1.7%**.
- Memory: **29.50 GiB / 31.25 GiB** peak HBM, **1.7418 GiB** free;
  stack reservation **21.7067 GiB**, heap allocation **7.7976 GiB**.

The streaming CE path successfully avoids the v278 logits OOM, but the
custom-call and loop-fusion cost is too large to recover the lost batch-4
MaxText-CE throughput.

## HLO Dump

Optimized train-step HLO was copied to `/tmp/qwen3-v279-hlo/`. The two worker
dumps are byte-identical:

- SHA256:
  `f205e693bc93ac8db8e11c71199fbe2b562d62cdebe6d3c6410cf9924ebe77bb`
- Size: **19,922,224 bytes / 119,222 lines**
- Key counts: `PallasMosaicTpuLinearSoftmaxCrossEntropyLoss` **13**,
  `linear_softmax_cross_entropy_loss` **16**, `tokamax` **20**,
  `all-gather` **4630**, `reduce-scatter` **1766**, `all-reduce` **154**,
  `async-start`/`async-done` **393** each, `convolution` **2088**,
  `custom-call` **2254**, `fusion` **22690**, `copy` **9942**.
- HLO markers confirm the Tokamax Mosaic TPU CE custom calls:
  `linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu.1` and
  `linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu.1`.

## Verdict

**Refuted.** Tokamax streaming CE batch 5 now fits comfortably and improves
over the older Tokamax-CE v245 baseline, but it remains **4,028 tok/s** and
**2.9 MFU percentage points** below v273. Do not carry Tokamax CE as the
frontier path. Direct MaxText-CE batch 5 is blocked by the full-logits temp,
while existing streaming CE implementations are compute/scheduling-bound; the
next useful work should return to the v273 MaxText-CE batch-4 frontier and
target non-CE HLO/layout/kernel costs, or design a materially better fused
lm-head+CE kernel rather than reusing the current Tokamax/Levanter CE paths.
