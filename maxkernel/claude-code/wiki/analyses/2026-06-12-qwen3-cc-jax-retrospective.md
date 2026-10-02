---
title: "Qwen3 8B JAX retrospective update"
type: analysis
model: qwen3-cc
lane: jax
created: 2026-06-12
updated: 2026-06-12
mode: incremental
tags: [qwen3-cc, jax, retrospective, v6e-8, autoresearch]
---

# Qwen3 8B JAX retrospective update

Mode: INCREMENTAL. Prior: `2026-06-05-qwen3-cc-jax-retrospective.md`.
This update deep-reads the post-prior frontier work, current `qwen3_cc5`
reruns, the fresh MaxText baseline/profile, and the new MFU normalization.

Current historical frontier: v273/v277 family, Qwen3-8B on v6e-8, seq8192,
global batch 32, native JAX/Flax NNX, no-scan + remat + activation sharding,
Tokamax Splash max-logit 30, MaxText/T5X CE, no MLP-intermediate `_sac`,
SparseCore reduce-scatter/all-reduce offload with aggregator, all-gather on
async collective fusion, MLP up-proj first, selective resources, RS latency
multiplier 3, AR latency multiplier 2, scoped VMEM 100352. Best reproduced
2026-06-05 runtime: **60,346 tok/s**, **7,543 tok/s/chip**, about **49.2%
MaxText-style MFU**, XProf **4358.1 ms**.

Current-cluster frontier: v036/v037 current-root rerun of the same material
HLO band, with shared-memory limit 90, reaches **59,330 tok/s**, **7,416
tok/s/chip**, about **48.4% MaxText-style MFU**, XProf **4433.0 ms**, and peak
HBM **29.03/31.25 GiB**. This is below the historical v273/v277 runtime despite
matching the same HLO hash family, so treat current-run comparisons against
v036/v037 unless the environment gap is explained.

Fresh MaxText reference: `qwen3_cc5` MaxText v001-base-b reaches **6,956-6,998
tok/s/chip** and about **45.4-45.7% MFU** by MaxText's printed formula. After
normalizing the native JAX MFU counter to the same full-attention FLOP
convention, native JAX is not behind MaxText on the full-throughput shape. The
remaining work is real runtime improvement, not accounting correction.

## Executive Readout

The lane is not exhausted, but the old scalar surface is exhausted. The
post-v273 and current-root runs show repeated refutations for VMEM,
shared-memory, rerun count, MaxText-style checkpoint save policies, pinned-host
offload, RMSNorm replication, local attention ordering, CE variants that reuse
existing Tokamax/Levanter-style paths, and MaxText-style MLP remat name
transfers.

The strongest remaining opportunity is still the logits/lm-head/CE path, but
the failure mode is now sharper: any layer-remat save policy that increases
live values runs into the full `[B,T,V]` logits allocation, while the existing
Tokamax streaming CE fits but is too slow. A successful CE experiment must
avoid full logits without introducing the current Tokamax backward custom-call
cost or the naive tiled-XLA backward live-range explosion.

The second opportunity is a real MLP/SwiGLU kernel or source rewrite informed
by HLO, not another naming/ordering cosmetic probe. MaxText-style names and
explicit elementwise rewrites were refuted; a useful experiment needs to change
the expensive convolution-fusion bucket or memory schedule.

## Mechanism Tree Update

### CE / Logits / lm-head

Supported historically: MaxText/T5X CE remains part of the best batch-4
frontier. MFU accounting has been normalized to MaxText's full-attention FLOP
convention for reporting only.

Refuted or invalid: Levanter CE variants v248-v251 did not improve the
frontier. Direct MaxText CE batch 5 v278 OOMed with an **11.59 GiB** full-vocab
logits temp. Tokamax CE batch 5 v279 fit with **1.74 GiB** free HBM but
regressed to **56,318 tok/s / 40.4% old MFU** and **5829.3 ms**. Earlier tiled
CE/custom-VJP attempts v213-v216 either OOMed or inflated backward live ranges.

Open: build a real streaming/fused lm-head+CE path with a local equivalence
gate. It must avoid materializing full logits and avoid the current Tokamax
backward cost.

### Activation Checkpointing / Remat Save Policies

Refuted or invalid: v048 MaxText `save_dot_except_mlpwi` on-device remat save
OOMed at **44.38 GiB**. v049 attention-projection-only save policy produced a
byte-identical OOM. v050/v051 pinned-host attention offload nearly fit, but
v052 only compiled at shared-memory limit 80 and regressed to **40,056 tok/s**
with **43.6% MXU**.

Conclusion: close remat save/offload variants for the current CE/logits
structure. They are blocked by logits live range or host-transfer cost.

### MLP / SwiGLU / Elementwise Fusion

Supported: v258 MLP up-proj first was the only recent MLP source-order signal
that entered the historical frontier family.

Refuted or invalid: v252/v253 fused gate-up variants OOMed. v255/v256 explicit
SwiGLU variants refuted. v047 MaxText-style MLP checkpoint names selected no
useful HLO/profile improvement and regressed slightly.

Open: if attacking MLP next, use an HLO-driven kernel/source experiment that
targets the convolution-fusion bucket directly. Do not repeat source-name or
pure elementwise expression variants.

### Attention / Layout

Refuted: v254/v257 QK norm after transpose and current-root v044 value-proj
first did not improve the frontier. Prior Splash tile sweeps were bounded.

Open: qseq2 remains a separate correctness branch. Do not count qseq2
throughput until loss is valid.

### Sharding / Collectives / Runtime Flags

Supported historically: no-scan, no MLP-intermediate `_sac`, SparseCore
reduce-scatter/all-reduce offload with aggregator, all-gather kept on async
collective fusion, selective resources, RS3+AR2.

Refuted or bounded: v259-v277 bracketed RS/AR, selective resources, VMEM, and
shared-memory around the historical frontier. Current-root v038-v043 repeated
the shared-memory/VMEM/rerun neighborhood and did not recover the historical
60.3k runtime. v046 replicated RMSNorm parameters refuted.

Open: source-level collective/layout changes may still help, but more backend
flag sweeps on the same material HLO should be avoided.

### MaxText Comparison

MaxText was rerun and profiled. The best fresh MaxText reference is about
**45.4-45.7% MFU**, while native JAX historical v273 is about **49.2%
MaxText-style MFU** and current-root v037 is about **48.4%**. MaxText is
valuable as an implementation comparator, but it no longer represents an
unclosed MFU target for this exact native-JAX full-throughput lane.

## Ranked Next Directions

1. Implement a real streaming/fused lm-head+CE path. Success signal: batch 4
   stays at or above v036/v037 on the current cluster while logits/CE HBM or
   custom-call time drops, or batch 5 fits without the Tokamax CE slowdown.

2. Investigate the v273/v277 versus v036/v037 same-HLO environment gap.
   Success signal: recover the historical **60.3k tok/s** band or identify the
   libtpu/compiler/cache/cluster condition causing the current **59.3k tok/s**
   band. This should be a short diagnostic, not a long rerun sweep.

3. Prototype an HLO-driven MLP/SwiGLU/Pallas kernel or source rewrite that
   changes the convolution-fusion bucket. Success signal: lower
   convolution-fusion profile time without raising HBM enough to block v6e.

4. Source-level collective/layout simplification. Success signal: preserve the
   SparseCore RS/AR benefit with fewer brittle backend-flag dependencies or a
   cleaner HLO schedule.

5. qseq2 correctness branch, isolated from the performance lane.

## Cross-Lane Brief

The native JAX lane remains the performance lane for Qwen3-8B on v6e-8.
MaxText is now a reference for formula and implementation details, not a higher
frontier to chase. TorchAX remains useful for cross-lane regressions but has
not shown a better Qwen3-8B v6e-8 frontier.

## Incremental Experiment Ledger

| Experiments | Topic | Outcome | Frontier shift |
|---|---|---|---|
| v248-v251 | Levanter / XLA CE variants | refuted | no CE frontier improvement |
| v252-v253 | fused MLP gate-up | invalid/OOM | no viable MLP fusion |
| v254/v257 | QK norm after transpose | refuted | no attention-layout gain |
| v255-v256 | explicit SwiGLU expression | refuted | no MLP expression gain |
| v258 | MLP up-proj first | supported | entered historical frontier family |
| v259-v277 | RS/AR, selective resources, VMEM, shmem brackets | supported then bounded | historical frontier v273/v277 around 60.3k tok/s |
| v278 | batch 5 with MaxText CE | invalid/OOM | logits temp remains blocker |
| v279 | batch 5 with Tokamax CE | refuted | fits but too slow |
| cc5 v001-v043 | current-root baseline/rerun/scalar brackets | mixed/refuted | current cluster best v036/v037 around 59.33k tok/s |
| cc5 v044 | attention value projection first | refuted | no gain |
| cc5 v045 | CE backward residual recompute | refuted | no gain |
| cc5 v046 | replicate RMSNorm params | refuted | no gain |
| cc5 v047 | MaxText-style MLP checkpoint names | refuted | no gain |
| cc5 v048-v049 | on-device remat save policies | invalid/OOM | logits live range blocker |
| cc5 v050-v052 | pinned-host attention remat offload | invalid/refuted | fits only with severe slowdown |
| MaxText v001-base-b | reference baseline | supported reference | 45.4-45.7% MFU target beaten by native JAX |
