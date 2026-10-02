---
title: "Retrospective — qwen3_cc5 / jax as of 2026-06-12 (post-MaxText-phase)"
type: analysis
tags: [retrospective, qwen3-cc5, jax]
model: qwen3_cc5
lane: jax
created: 2026-06-12
updated: 2026-06-12
---

# qwen3_cc5 / jax retrospective — 2026-06-12 (#2, post-MaxText-phase)

**Mode**: INCREMENTAL (prior: [2026-06-12-qwen3_cc5-jax-retrospective.md](2026-06-12-qwen3_cc5-jax-retrospective.md), v000–v019; +13 new experiments v020–v032 deep-read)
**Scope**: 8B/v6e-8 (sole variant row)
**Period**: 2026-06-02 → 2026-06-12
**Experiments**: 33 total — 11 supported, 15 refuted, 3 inconclusive, 4 invalid
**Frontier**: **6,040 tok/s/chip / 34.6% MFU / 5,425 ms @ bs4/seq8192** (v027; scan + offload_attn + splash + chunked CE f32-x + 25-flag stack)

## Prior-retrospective accountability

All six recommendations of retrospective #1 were acted on — none remain open:

| # | Prior recommendation | Outcome |
|---|---|---|
| 1 | Scan-over-layers | ✅ landed (v020, −8 GiB live-set; became frontier substrate) |
| 2 | Host offload of layer inputs/activations | ✅ landed via offload_attn (v024–v026 arc; part of frontier) |
| 3 | bs2+ @ seq8192 | ✅ exceeded — bs3 (v026), bs4 frontier (v027); closed at bs4 (v029) |
| 4 | tokamax splash knobs | ❌ refuted (v022) |
| 5 | Fused QK-norm+RoPE Pallas | 🅿 stays parked (HLO pre-filter: ceiling ≤1–2% at L) |
| 6 | Data-formatting drilldown | ✅ done — bucket shrank to 2.0% at bs4 under RS-layout flags |

## Search-strategy assessment

- ✅ **Frontier moved 4× since prior retrospective** (v020/v026/v027 shifts; 33.6% → 34.6% MFU; +2.4% tok/s/chip). Not stuck.
- ✅ **Kernel-vs-probe balance healthy**: the phase mixed structural work (scan, offload remat, flag-stack archaeology, HLO forensics) with axis probes; no 5:1 imbalance.
- ✅ **Gap fully attributed**: the −13.1% residual to MaxText is decomposed to the millisecond ([bucket diff](2026-06-12-maxtext-vs-jax-bucket-diff.md)); this is the strongest possible position from which to judge exhaustion.
- ⚠️ **Comprehensively closed topics** (≥3 refutations each, diverse mechanisms): splash kernel config (v016/v022/v028), batch axis (v021/v029 + 4 earlier cliffs), AG-overlap flags (v030/v031/v032), unrolled+offload (v024/v030-B).
- 🔵 **Genuinely unexplored axes remain**: Tensor parallelism (ZERO experiments on this lane), int8/AQT weight-only quantization (zero), sequence parallelism (zero), manual collective restructuring in the scan body (the named L route).

**Inference**: the lane is NOT exhausted — but every remaining direction is
M/L effort or carries negative cross-model priors. The S-effort,
single-axis space IS exhausted (that's what v028–v032 proved). Remaining
upside concentrates on the quantified 624 ms AG bucket.

## Mechanism tree (delta since prior; v020–v032)

- **FSDP / collective optimization** (7 new) — THE phase topic
  - 25-flag MaxText stack completion — 🏆 v026 +10.3% (windowed-einsum disable −341 ms; lane standard)
  - AG-overlap flags — ❌ v030 (flag absent) / ❌ v031 (pipeliner fire-and-hurt −1.8%) / ⚠️ v032 (loop-unroll silent no-op) — flag path CLOSED
  - 🔵 manual weight prefetch in scan body — UNEXPLORED (the 624 ms bucket; MaxText parity route)
  - 🔵 int8 weight AG (AQT weight-only) — UNEXPLORED (halves AG bytes AND weight HBM traffic; conv bucket is 49% BW-util)
- **HBM offload / remat** (3 new)
  - offload_attn unrolled — ❌ v024 (liveness OOM, structural — confirmed again v030-B at bs4)
  - offload_attn + scan — ❌ v025 standalone (5,381; DMA stalls) → ✅ part of v026 frontier (DMA 0.7 ms WITH flags)
  - MLP-only remat — retired (purpose was bs6; batch closed)
- **Batch/seq** (3 new) — bs3 ✅ v026 / bs4 🏆 v027 / bs5 ❌ v029 — **closed at bs4** (ladder +4.9% → +1.8% → −0.5%; bs6 = OOM at 31.86 GiB projected)
- **torch.compile/scan** (1 new) — 🏆 v020 scan substrate (−8 GiB live-set, 3.4× compile)
- **Splash attention** (1 new) — ❌ v028 block sweep (3rd flat; bq=2048 strictly > 1024 by +2.6%; kernel config CLOSED; splash measured FASTER than MaxText flash −89 ms)
- **Measurement/reference** (2 new) — ✅ mt-v001 MaxText baseline (maxtext lane); ✅ bucket-diff + phase-summary analyses
- 🔵 **Tensor parallelism** — UNEXPLORED on this lane (tp=1 in all 33 experiments). Cross-model prior: llama3-jax TP=2 = −14%. v6e ICI 80 GB/s; taxonomy: TP ≤ 8 intra-island fine. Launch-only probe possible (`--tp_parallelism=2`).
- 🔵 **Sequence parallelism** — UNEXPLORED (needs shard_as plumbing through scan; M-L).
- 🔵 **int8 weight-only (AQT/qwix)** — UNEXPLORED (taxonomy: "recommended starting point" for lower precision; needs loss-parity validation per SCHEMA rule 8).

## Coverage gaps

| Topic | Tried/catalog | Status |
|---|---|---|
| FSDP/collectives | flags exhaustively; structural restructuring 0/2 (manual prefetch, int8-AG) | the live frontier direction |
| Tensor/sequence parallelism | 0/2 | unexplored; negative cross-model prior on TP |
| Precision (int8/fp8) | 0/2 | unexplored; semantics-validation burden |
| Pallas authoring | 0/1 viable (QK-norm+RoPE parked on pre-filter) | correctly parked |
| Splash, batch, remat ladder, vmem, CE impls, scheduler flags | saturated | CLOSED with multi-refutation evidence |

## Recommended next directions

| # | Direction | Topic | Gain | Conf | Effort | Score | Source |
|---|---|---|---|---|---|---|---|
| 1 | **Manual weight prefetch in scan body** (double-buffered AG of layer N+1 under layer N compute) | fsdp-collectives | ≈ +3.8 pp MFU (+11% tok/s; the 624 ms bucket) | medium | L | 0.30 | local-gap: [bucket diff](2026-06-12-maxtext-vs-jax-bucket-diff.md) |
| 2 | **TP=2 probe at the frontier shape** (fsdp4×tp2; launch-only) | tensor-parallelism | ±0–1 pp (restructures the AG pattern entirely; axis never probed) | low | S | 0.20 | local-gap: zero TP coverage; conflict noted: llama3-jax −14% |
| 3 | **int8 weight-only AQT** (halves AG bytes + weight HBM traffic) | precision | ~+2 pp | low-med | L | 0.12 | local-gap: taxonomy precision section; attacks same AG bucket as #1 |
| 4 | Sequence parallelism through scan | sharding | speculative | low | M-L | <0.1 | local-gap |

## Anti-recommendations (don't pursue)

- **Any further AG-overlap flag probes** — three-way closure (absent / fire-and-hurt / no-op); the build's flag surface is mapped.
- **Splash kernel config** — triple-flat; splash already beats the MaxText reference kernel.
- **Batch/remat/vmem/CE-impl re-probes** — all closed with measured walls (see phase summary ledger).
- **Unrolled loop in any offload configuration** — structural liveness OOM, twice confirmed.
- All anti-recommendations from retrospective #1 remain in force.

## Cross-lane brief

- **qwen3-cc5-maxtext**: reference lane only (mt-v001 = 6,953 tok/s/chip @ bs3). Its entire advantage = scan-body AG overlap (74%) + MXU layout (29%, partly stall feedback) − our splash advantage (−15%). No further transferable knobs — the flag stack and remat policy are already ported.
- **qwen3-cc5-torchax**: frontier 19.2% MFU @ seq2048 baseline; dormant; no transfer candidates toward jax.
- **llama3-8b-jax** (cross-model): frontier stack (scan + tokamax CE + SC-offload flags @ 43.6% reported MFU on its own accounting) — its SC-offload lever is compile-fatal on current libtpu (mt-v001 attempt 1 + v009); its TP=2 result (−14%) is the negative prior for direction #2.

## Experiment ledger (machine-readable)

| v-ID | Date | Variant | Topic | Verdict | Frontier-shift |
|---|---|---|---|---|---|
| v000 | 2026-06-02 | 8B/v6e-8 | baseline | supported | yes (initial: 20.5% @ seq2048) |
| v001 | 2026-06-12 | 8B/v6e-8 | batch-seq | refuted | no |
| v002 | 2026-06-12 | 8B/v6e-8 | splash-attention | supported | yes (+1.7 pp → 22.3%) |
| v003 | 2026-06-12 | 8B/v6e-8 | ce-softmax-loss | refuted | no |
| v004 | 2026-06-12 | 8B/v6e-8 | batch-seq | refuted | no |
| v005 | 2026-06-12 | 8B/v6e-8 | vmem-scratch | inconclusive | no |
| v006 | 2026-06-12 | 8B/v6e-8 | ce-softmax-loss | invalid | no |
| v007 | 2026-06-12 | 8B/v6e-8 | batch-seq | supported | yes (31.4% @ seq2048) |
| v008 | 2026-06-12 | 8B/v6e-8 | ce-softmax-loss | invalid | no |
| v009 | 2026-06-12 | 8B/v6e-8 | libtpu-xla-flags | refuted | no |
| v010 | 2026-06-12 | 8B/v6e-8 | libtpu-xla-flags | refuted | no |
| v011 | 2026-06-12 | 8B/v6e-8 | ce-softmax-loss | supported | no (enabler) |
| v012 | 2026-06-12 | 8B/v6e-8 | activation-checkpointing | invalid | no |
| v013 | 2026-06-12 | 8B/v6e-8 | batch-seq | refuted | no |
| v014 | 2026-06-12 | 8B/v6e-8 | batch-seq | refuted | no |
| v015 | 2026-06-12 | 8B/v6e-8 | activation-checkpointing | supported | yes (seq-8192 unlock, 31.8%) |
| v016 | 2026-06-12 | 8B/v6e-8 | splash-attention | refuted | no |
| v017 | 2026-06-12 | 8B/v6e-8 | activation-checkpointing | supported | yes (33.6% settled @ seq8192) |
| v018 | 2026-06-12 | 8B/v6e-8 | activation-checkpointing | refuted | no |
| v019 | 2026-06-12 | 8B/v6e-8 | measurement-validation | supported | no (validates frontier) |
| v020 | 2026-06-12 | 8B/v6e-8 | torch-compile-scan | supported | no (substrate: −8 GiB, 3.4× compile) |
| v021 | 2026-06-12 | 8B/v6e-8 | batch-seq | refuted | no |
| v022 | 2026-06-12 | 8B/v6e-8 | splash-attention | refuted | no |
| v023 | 2026-06-12 | 8B/v6e-8 | activation-checkpointing | refuted | no |
| v024 | 2026-06-12 | 8B/v6e-8 | hbm-offload | refuted | no (mechanism proven; unrolled liveness OOM) |
| v025 | 2026-06-12 | 8B/v6e-8 | hbm-offload | refuted | no (fits; 5,381 < bar) |
| v026 | 2026-06-12 | 8B/v6e-8 | fsdp-collectives | supported | yes (5,933 → 34.0% @ bs3; flags lane-standard) |
| v027 | 2026-06-12 | 8B/v6e-8 | batch-seq | supported | yes (6,040 → 34.6% @ bs4) |
| v028 | 2026-06-12 | 8B/v6e-8 | splash-attention | refuted | no |
| v029 | 2026-06-12 | 8B/v6e-8 | batch-seq | refuted | no (batch closed at bs4) |
| v030 | 2026-06-12 | 8B/v6e-8 | fsdp-collectives | inconclusive | no (flag absent; unrolled-OOM corroborated) |
| v031 | 2026-06-12 | 8B/v6e-8 | fsdp-collectives | refuted | no (pipeliner −1.8%) |
| v032 | 2026-06-12 | 8B/v6e-8 | fsdp-collectives | inconclusive | no (flag no-op; flat; spot-exhausted) |

## Sources

- Per-lane log: `wiki/experiments/qwen3_cc5_autoresearch_optimization/jax/log.md`
- Model page: `wiki/models/qwen3-cc5-jax.md`
- Taxonomy: `wiki/model-optimization-index.md`; Pallas catalog: `wiki/analyses/2026-04-23-pallas-kernel-directory.md`
- Prior retrospective: [2026-06-12-qwen3_cc5-jax-retrospective.md](2026-06-12-qwen3_cc5-jax-retrospective.md)
- Phase docs: [phase summary](2026-06-12-maxtext-gap-closing-phase-summary.md), [bucket diff](2026-06-12-maxtext-vs-jax-bucket-diff.md)
- All 13 new experiment pages v020–v032 under `wiki/experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/`

## See also

- [Model page](../models/qwen3-cc5-jax.md) · [Phase summary](2026-06-12-maxtext-gap-closing-phase-summary.md) · [Prior retrospective](2026-06-12-qwen3_cc5-jax-retrospective.md)
