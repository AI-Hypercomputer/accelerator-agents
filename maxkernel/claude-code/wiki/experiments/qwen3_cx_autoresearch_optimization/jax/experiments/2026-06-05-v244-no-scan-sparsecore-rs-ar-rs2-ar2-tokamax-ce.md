---
title: "Qwen3 JAX v244 - RS2+AR2 with Tokamax CE"
type: experiment
hypothesis: "Tokamax streamed CE was too slow before SparseCore RS/AR latency tuning, but its much lower HBM footprint may interact differently with the current post-SparseCore frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs2-ar2-tokamax-ce
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, tokamax-ce, ce, splash, tokamax-splash, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v244 - RS2+AR2 with Tokamax CE

## Hypothesis under test

**Hypothesis**: v201/v202 refuted Tokamax streamed CE on the older no-scan
frontier: it reduced HBM substantially but lost too much wall time. The current
v230 frontier has a different collective schedule, with reduce-scatter and
all-reduce on SparseCore and latency multipliers tuned to `2/2`. Retesting
Tokamax CE on this newer schedule closes whether the lower-memory CE path
becomes useful after the collective backend changed.

**Mechanism**: Keep v230 fixed except replace MaxText CE with Tokamax CE:
`--use_tokamax_ce=True --tokamax_ce_impl=mosaic_tpu --use_maxtext_ce=False`.

**Predicted signal**: Support requires clean compile/loss and throughput above
the v230/v236 band (**59,749-59,750 tok/s**) or a profile showing a credible
new memory/perf tradeoff that justifies a follow-up batch-size probe.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion far below the current frontier without a compelling memory-opening
signal.

## Setup

GKE workload `alekseyv-q3-v244-tokamaxce-sc`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v244-no-scan-sparsecore-rs-ar-rs2-ar2-tokamax-ce`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False
  --use_splash=True --use_tokamax_ce=True --tokamax_ce_impl=mosaic_tpu
  --use_maxtext_ce=False --shard_acts=True --batch_size=4 --seqlen=8192
  --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. Tokamax CE is a mathematically equivalent fused
implementation of the same integer-label mean CE objective; correctness signal
is finite loss matching the known synthetic-loss curve.

## Result

Workload `alekseyv-q3-v244-tokamaxce-sc` completed cleanly with `EXIT_CODE=0`
on both workers and normal loss ending at 12.0524.

- Worker0: **55,473 tok/s** average over steps 2-19, **55,632 tok/s** best
  step, **55,491 tok/s** average over steps 15-19, **55,507 tok/s** excluding
  profiled steps.
- Worker1: **55,493 tok/s** average over steps 2-19, **55,632 tok/s** best
  step, **55,491 tok/s** average over steps 15-19, **55,507 tok/s** excluding
  profiled steps.

This improves on v201's older no-scan Tokamax CE run by about 1.2k tok/s, but
it remains far below the v230/v236 **59,749-59,750 tok/s** frontier band.
Nominal MFU is about **39.8%**.

## Profile

XProf run
`2026-06-05-qwen3-jax-v244-no-scan-sparsecore-rs-ar-rs2-ar2-tokamax-ce/2026_06_05_09_53_42`
shows:

- Step time: **4745.4 ms**
- MXU utilization: **63.3%**
- Peak HBM: **27.02 GiB** / 31.25 GiB
- Stack reservation: **19.1602 GiB**
- Heap allocation: **7.8634 GiB**
- Free memory at peak: **4.2225 GiB**

The memory relief is real versus v230's **30.93 GiB**, but runtime is worse.
The all-host op profile shows:

- convolution fusion: **51,429.3 ms** / **45.2%**
- custom-call: **42,245.0 ms** / **37.1%**
- loop fusion: **10,593.3 ms** / **9.3%**

Within custom-call, `linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu.1`
takes **10,181.2 ms** / **8.9%** of the profiled train-step time.

## HLO Dump

Optimized train-step HLO:

- Local dump:
  `/tmp/qwen3-v244-hlo/module_0477.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `a76111c8e5f40ce1e82cc1f8ce28f0656e7c9f41f1a446c10b2f06edeed81d33`
- Size: **19,909,919 bytes**
- Lines: **119,104**

The HLO is distinct from the v230 MaxText CE frontier because the fused
Tokamax linear CE path replaces the MaxText CE lowering.

## Verdict

**Refuted.** Post-SparseCore collectives improve Tokamax CE relative to the
older v201 no-scan run, and Tokamax CE opens about **3.9 GiB** of HBM headroom,
but the fused CE custom-call cost still dominates the tradeoff. Throughput is
about **4.3k tok/s** below the current frontier and MXU falls to **63.3%**. Do
not carry Tokamax CE for the current frontier; only revisit it if a tuned
Tokamax CE config or different CE kernel specifically targets the
`linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu` cost.
