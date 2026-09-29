---
title: "Retrospective — qwen3_cc / jax as of 2026-06-02 (#3, FULL --full-redo audit)"
type: analysis
tags: [retrospective, qwen3-cc, jax]
model: qwen3_cc
lane: jax
created: 2026-06-02
updated: 2026-06-02
---

# qwen3_cc / jax retrospective #3 — 2026-06-02 (FULL re-read / audit)

**Mode**: FULL (`--full-redo`) — every experiment page re-read FROM DISK by an independent reader,
not inherited from the prior ledger. Purpose: verify retrospective
[#2](2026-06-02-qwen3_cc-jax-retrospective-2.md) (which was built INCREMENTAL from in-session memory)
against ground truth on disk.
**Scope**: 8B/v6e-8. **Period**: 2026-06-02. **Experiments**: 24 (baseline + v001–v024; v015 never existed).
**Frontier (verified)**: seq2048 = **v018 35.8% MFU / 6,964 tok/s/chip**; seq8192 = **v009 30.4% / 5,305 tok/s/chip**.

## Audit result — retrospective #2 CONFIRMED, zero discrepancies

The independent full-disk re-read reproduces retrospective #2's ledger and conclusions exactly:

- **Verdicts ↔ numbers consistent** for all 24: every `supported` is above the prior frontier; every
  `refuted` is below it; every `invalid` is a pre-step-0 crash (OOM/flag error); v013 correctly
  `supported`-as-canary (21.0%, not a frontier-mover, explicitly noted).
- **Frontier lineage intact**: 20.5%→22.0%(v004)→25.1%(v005)→32.4%(v008)→**35.8%(v018)**; seq8192
  target 30.4%(v009). Model page "Current best" cell + "Frontier exp" link = v018, matches.
- **LINT clean**: all 19 non-invalid experiments carry a `## Profile` section with GCS/xprof paths; all 24
  carry `## Next hypotheses`; all 13 referenced `qwen3-jax-*` hypothesis files exist (no broken links);
  all `variant: "8B/v6e-8"`.
- **Model page**: Open hyps = 3 (async-collective-fusion→now refuted, scan, AMP); Retired list =
  batch-scaling, tokamax-ce, SparseCore-offload. Consistent.

**Discrepancies found: NONE.** Two immaterial categorization refinements vs #2 (do not change any verdict
or conclusion):
1. **v005** — #2 filed it under *activation-checkpointing*; the independent reader labeled it *batch-size
   (remat+bs)*. Both correct: it is remat **enabling** bs=3 (without remat, bs≥2 OOMs — v001/v002). The
   unlocking lever is remat; the knob varied is batch. Either tag is defensible.
2. **v020** — #2 (and this page) treat it as the **tokamax-splash kernel swap** (`USE_TOKAMAX_SPLASH=1`,
   default blocks); the reader's auto-tag said "splash-block-tuning (tokamax)". The precise reading is:
   v020 = kernel swap; v021/v022 = the actual block-size sweep. Kept as kernel-swap here.

## Mechanism tree (verified counts)

- **Splash attention** (6: v006,v007,v008,v009,v020*,v021/v022 block-tune) — 🏆 v008 32.4%, 🏆 v009 seq8192 30.4%; tokamax-splash ❌ v020 −8.5%; block-tuning ❌/⚠️ v021/v022. *(v020 kernel-swap; v021/v022 block-tune.)*
- **libtpu/XLA flags** (3: v018,v019,v024) — 🏆 **v018 +3.4 pp → 35.8%**; ⚠️ v019 neutral @ seq8192; ❌ v024 ablation (umbrella flag alone ≈ baseline; the 6 explicit flags carry the win).
- **tokamax CE** (6: v010–v014,v016) — ✅ v013 correct; ❌ v014 (seq2048), v016 (seq8192); v010–v012 invalid (impl/absl). Memory enabler, never a throughput lever. Closed.
- **Activation checkpointing / remat** (2–3: v004,v005) — 🏆 v004 22.0%, 🏆 v005 25.1%. The batch unlock; baseline since.
- **FSDP / collective** (3: v003,v017,v023) — SparseCore-offload ❌ v003 (bs1)/v017 (bs4); async-collective-fusion ❌ v023 (−3.5 pp, over-subscribes the v018 scheduler). Closed.
- **Batch/seq** (2: v001,v002) — ❌ pre-remat OOM; subsumed by remat.
- 🔵 **Data-transformation** (0 exp) — QKV→splash transpose **intrinsic** (investigated, closed; obs `qwen3-jax-qkv-splash-transpose-intrinsic.md`).
- 🔵 **scan / torch.compile** (0) — profile says non-dominant; low prior. 🔵 **fused-QKV Pallas kernel** (0) — effort-L, low prior. 🔵 **seq16384** (0) — coverage.

## Search-strategy assessment

Confirmed unchanged from #2: lane climbed 20.5%→35.8% (+74% tok/s/chip) on **compiler/scheduler levers**
(remat, splash, the explicit XLA scheduler bundle), then 6 consecutive post-v018 non-wins exhausted the
cheap/medium space. Profile (v018) is **non-matmul-bound** (MXU 48.3%, 48.6% non-matmul) — but each
non-matmul bucket resisted its obvious lever (RS-overlap backfired v023; splash kernel-swaps regress;
norms already XLA-fused; layout intrinsic). **Verdict: seq2048 frontier 35.8% at practical ceiling for
non-deep-kernel levers** — independently re-confirmed.

## Recommended next directions (unchanged from #2)

| # | Direction | Topic | Gain | Conf | Effort | Source |
|---|---|---|---|---|---|---|
| 1 | **Pivot to torchax sibling lane (19.2%)** — transfer remat + splash + the EXPLICIT 7-flag XLA scheduler bundle | cross-lane | large (19.2%→~30%+) | medium-high | M (new lane) | sibling: torchax untouched; v018 levers are framework-agnostic |
| 2 | scan-over-layers | scan | 0–2% | low | M | profile: non-dominant |
| 3 | fused QK-norm+RoPE+QKV Pallas kernel (HLO-prefilter first) | Pallas | +1–3% | low | L | local-gap; kernel-swaps regress on qwen3 |
| 4 | seq16384 coverage | seq | coverage | low | S | local-gap |

The #1 actionable item is the **torchax pivot** (`/start-experiment qwen3_cc torchax`) — the jax lane is done.

## Anti-recommendations (verified closed)

tokamax-CE, SparseCore-offload, tokamax-splash, splash block-tuning, async-collective-fusion,
XLA-scheduler-flags@seq8192, raw batch-scaling, custom RMSNorm Pallas, `xla_jf_spmd_threshold` (crashes),
QKV-layout reshaping (intrinsic). All have direct refuting/neutral experiments or structural impossibility.

## Cross-lane brief

- **qwen3-cc-torchax**: 19.2% MFU, optimization loop NOT yet run — the high-value target. The full v018
  jax stack (remat + splash + explicit 7-flag scheduler bundle) is framework-agnostic libtpu/compiler
  work and should transfer; flag-ablation (v024) established that the **6 explicit flags** (not the
  umbrella) are load-bearing, so apply the explicit bundle.
- **llama3-8b-jax**: 43.3% MFU @ bs4/seq8192. The ~7.5 pp gap is a seq8192 batch-occupancy regime qwen3
  can't reach (CE refuted as its batch enabler, v016); not closable by the levers qwen3 already refuted.

## Experiment ledger (machine-readable — verified from disk)

| v-ID | Date | Variant | Topic | Verdict | MFU | Frontier-shift |
|---|---|---|---|---|---|---|
| baseline | 2026-06-02 | 8B/v6e-8 | baseline | supported | 20.5% | ref |
| v001 | 2026-06-02 | 8B/v6e-8 | batch-size | invalid | — | no (OOM) |
| v002 | 2026-06-02 | 8B/v6e-8 | batch-size | invalid | — | no (OOM) |
| v003 | 2026-06-02 | 8B/v6e-8 | fsdp-collective | refuted | 16.5% | no |
| v004 | 2026-06-02 | 8B/v6e-8 | activation-checkpointing | supported | 22.0% | yes |
| v005 | 2026-06-02 | 8B/v6e-8 | activation-checkpointing (remat+bs3) | supported | 25.1% | yes |
| v006 | 2026-06-02 | 8B/v6e-8 | splash-attention | invalid | — | no (VMEM) |
| v007 | 2026-06-02 | 8B/v6e-8 | splash-attention | invalid | — | no (VMEM) |
| v008 | 2026-06-02 | 8B/v6e-8 | splash-attention | supported | 32.4% | yes |
| v009 | 2026-06-02 | 8B/v6e-8 | splash-attention (seq8192) | supported | 30.4% | yes (seq8192) |
| v010 | 2026-06-02 | 8B/v6e-8 | tokamax-ce | invalid | — | no (impl) |
| v011 | 2026-06-02 | 8B/v6e-8 | tokamax-ce | invalid | — | no (absl) |
| v012 | 2026-06-02 | 8B/v6e-8 | tokamax-ce | invalid | — | no (absl) |
| v013 | 2026-06-02 | 8B/v6e-8 | tokamax-ce | supported | 21.0% | no (canary) |
| v014 | 2026-06-02 | 8B/v6e-8 | tokamax-ce | refuted | 30.5% | no |
| v016 | 2026-06-02 | 8B/v6e-8 | tokamax-ce (seq8192) | refuted | 29.5% | no |
| v017 | 2026-06-02 | 8B/v6e-8 | fsdp-collective | refuted | 31.7% | no |
| v018 | 2026-06-02 | 8B/v6e-8 | libtpu-xla-flags | supported | **35.8%** | **yes** |
| v019 | 2026-06-02 | 8B/v6e-8 | libtpu-xla-flags (seq8192) | inconclusive | 30.7% | no |
| v020 | 2026-06-02 | 8B/v6e-8 | splash-attention (tokamax swap) | refuted | 32.8% | no |
| v021 | 2026-06-02 | 8B/v6e-8 | splash-block-tuning | refuted | 32.5% | no |
| v022 | 2026-06-02 | 8B/v6e-8 | splash-block-tuning (seq8192) | inconclusive | 30.5% | no |
| v023 | 2026-06-02 | 8B/v6e-8 | fsdp-collective (async-fusion) | refuted | 32.3% | no |
| v024 | 2026-06-02 | 8B/v6e-8 | libtpu-xla-flags (ablation) | refuted | 31.9% | no |

## Sources

- Prior retrospectives: [#1 FULL](2026-06-02-qwen3_cc-jax-retrospective.md), [#2 INCREMENTAL](2026-06-02-qwen3_cc-jax-retrospective-2.md)
- Independent full-disk audit of all 24 experiment pages (this run)
- Model page `wiki/models/qwen3-cc-jax.md`; lane log `wiki/experiments/qwen3_cc_autoresearch_optimization/jax/log.md`
- Observations: `qwen3-jax-fsdp-rs-not-overlapped.md`, `qwen3-jax-qkv-splash-transpose-intrinsic.md`
- Topic taxonomy `wiki/model-optimization-index.md`; Pallas catalog `wiki/analyses/2026-04-23-pallas-kernel-directory.md`

## See also

- [Qwen3 8B — jax](../models/qwen3-cc-jax.md)
- [Retrospective #1](2026-06-02-qwen3_cc-jax-retrospective.md), [#2](2026-06-02-qwen3_cc-jax-retrospective-2.md)
