---
title: "Qwen3 JAX current v063 - v037 frontier rerun"
type: experiment
hypothesis: "An exact rerun of the unmodified v258/v037 current frontier checks whether the cluster/cache still reproduces the 59.33k tok/s band after source probes selected slower HLO families."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, sparsecore, collectives, scheduler, shared-memory, rerun, control, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v063 - v037 frontier rerun

## Hypothesis under test

**Hypothesis**: v061 and v062 each changed only one source line but selected the
same slower large-HLO family. An exact rerun of the unmodified v258 image with
the v037 runtime flags should reproduce the current-cluster frontier HLO and
throughput band. If it does not, the observed gap may be cluster/cache/runtime
noise rather than source-change signal.

**Mechanism**: Reuse `qwen3-8b-jax:v258-mlp-up-first` without code changes and
run the v037 carried runtime exactly: no-scan/remat, activation sharding,
Tokamax Splash max-logit 30, MaxText CE, scoped VMEM 100352, selective
resources, rerun3, SparseCore RS/AR offload, RS3+AR2, and shared-memory limit
90.

**Predicted signal**: A healthy rerun should land near v036/v037:
**59,330 tok/s / 4433.0 ms**, with the tighter current-HLO family and peak HBM
around **29.03 GiB**.

**Falsification criterion**: Clean completion materially below v037 or HLO
counts matching the slower v061/v062 large-HLO family would indicate the current
environment is no longer reproducing the carried frontier.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v063-v037rerun`
- Image: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- Image digest: `sha256:3a08bd533913355b79d8d9c27243646bf8dd76809878cf41372fee0c6407071e`
- Run dir: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v063-v037-frontier-rerun`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.

## Result

Completed cleanly on `2026-06-12`.

- Worker0 summary: **58,963 tok/s**, **7,370 tok/s/chip**, **42.3% old MFU**
- XProf: `2026-06-12-qwen3-jax-v063-v037-frontier-rerun/2026_06_12_13_36_40`
- XProf step time: **4461.2 ms** average, **1.2 ms** stddev
- MXU utilization: **66.6%**
- Peak HBM: **29.03 GiB / 31.25 GiB**, **2.2202 GiB** free
- Op profile: convolution fusion **57,030.6 ms / 53.2%**; custom-call
  **29,412.2 ms / 27.5%**; loop fusion **10,713.5 ms / 10.0%**
- Fresh HLO: none found under this run's `hlo/` prefix, consistent with a
  persistent compile-cache hit.

## Verdict

**Refuted as a reproduction of the 59.33k frontier.** The unmodified v258/v037
control now lands in the same **58.9k tok/s / 4460 ms** live band as v061 and
v062. That means those two source probes were not clearly worse than the live
control; the current gap to the earlier v036/v037 measurement may involve the
compile cache, emitted HLO family, or cluster/runtime placement. Next action:
force a fresh compile/HLO for the exact frontier before attributing the
regression to source edits.
