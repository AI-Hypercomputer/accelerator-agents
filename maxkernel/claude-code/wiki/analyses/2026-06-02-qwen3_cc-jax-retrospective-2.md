---
title: "Retrospective — qwen3_cc / jax as of 2026-06-02 (#2, post-v018 arc)"
type: analysis
tags: [retrospective, qwen3-cc, jax]
model: qwen3_cc
lane: jax
created: 2026-06-02
updated: 2026-06-02
---

# qwen3_cc / jax retrospective #2 — 2026-06-02 (post-v018 arc)

**Mode**: INCREMENTAL (prior: [2026-06-02-qwen3_cc-jax-retrospective.md](2026-06-02-qwen3_cc-jax-retrospective.md); +8 experiments since: v016–v023, all deep-read).
**Scope**: 8B/v6e-8 (only variant). **Period**: 2026-06-02 (single day, 23 experiments).
**Experiments**: 23 — 7 supported, 6 refuted, 6 invalid (integration/OOM), 3 inconclusive, +1 deleted-dup (v015).
**Frontier**: seq2048 = **35.8% MFU / 6,964 tok/s/chip** (v018, remat+splash+XLA-scheduler-flags bs4); seq8192 = **30.4%** (v009). Climb 20.5%→22.0%→25.1%→32.4%→**35.8%** (+74% tok/s/chip over baseline).

## Search-strategy assessment

🟢 **The prior retrospective worked exactly as intended.** Its #1 recommendation (MaxText HOST_OFFLOAD XLA scheduler flag stack) landed **v018 = +3.4 pp / +10.6%**, the single biggest qwen3-jax win — confirming the "act on the retrospective's top rec" discipline.

🔴 **Then the frontier stalled: 5 consecutive post-v018 non-wins** (v019 inconclusive, v020 refuted, v021 refuted, v022 inconclusive, v023 refuted). Frontier-stuck-for = 5 experiments since v018.

🔬 **Profile-driven, NOT single-axis-blind.** Single-axis : kernel ratio is healthy (~2:1); after the 4th non-win I dispatched a profile-analyzer on v018 rather than keep guessing — the disciplined response to a stall. The profile (below) **refuted the matmul-ceiling hypothesis** (MXU 48.3%, 48.6% non-matmul) and named the bottleneck. The 5th experiment (v023) was then a *profile-directed* attack on that bottleneck — and still failed.

🔑 **Refined lever-transferability law for qwen3-jax** (the central finding of this arc):
- **COMPILER/SCHEDULER levers transfer** from llama3-jax: the XLA scheduler flag bundle gave +3.4 pp (v018). ✅
- **KERNEL-SWAP / COLLECTIVE-OFFLOAD levers do NOT**: SparseCore-offload (v003/v017), tokamax-splash (v020), splash block-tuning (v021/v022) — all refuted. ❌
- **NEW nuance**: a *second overlap mechanism* stacked on an already-overlap-tuned schedule is **not additive** — async-collective-fusion (v023) regressed −3.5 pp because the v018 latency-hiding scheduler already saturates the overlap budget. Overlap levers transfer only as the *first* such mechanism.

**Inference**: the seq2048 frontier (35.8%, MXU 48.3%) is **near its practical ceiling for flag/scheduler/kernel-knob levers**. The measured non-matmul wall (RS 12.9% + splash 12.3% + norms-HBM 11.0%) is real but each chunk has resisted its obvious lever. Remaining upside requires **deep custom-kernel work** (effort L, and the kernel-swap pattern says low transfer prior) or is **structurally capped** (qwen3's collective share is already low; norms are already XLA-fused). The gap to llama3-jax 43.3% is largely a seq8192-batch-occupancy difference qwen3 can't replicate (CE refuted as the batch enabler).

## v018 frontier profile (the evidence base)

| Bucket | % step | Bound | Addressable? |
|--------|--------|-------|--------------|
| matmul (conv fusion) | 51.4% | compute (AI 1,777 ≫ ridge 578) | at ceiling |
| **FSDP grad reduce-scatter** | **12.9%** | ICI, synchronous | ❌ async-fusion backfired (v023); SparseCore-offload refuted (v003/v017) |
| splash custom-call | 12.3% | VMEM/compute | ❌ tokamax refuted (v020); block-tuning neutral (v021/v022) |
| loop-fusion (norms/acts) | 11.0% | HBM (AI 1.2) | 🔵 already XLA-fused; custom RMSNorm Pallas is catalog-counterproductive on TPU |
| data-formatting (GQA layout) | 4.5% | HBM | 🔵 UNEXPLORED — QKV→splash transposes; a layout/sharding change might absorb these |
| MXU util 48.3% · TC idle 3.87 ms · HBM peak 65.4% | | | |

## Mechanism tree (incremental — new/changed topics since prior)

- **libtpu / XLA flags** (4 exp)
  - MaxText HOST_OFFLOAD scheduler bundle (7 flags) — 🏆 **v018 +3.4 pp → 35.8%** (the win).
  - same bundle @ seq8192 — ⚠️ v019 neutral (+0.3 pp, within noise — seq2048-specific lever).
  - async-collective-fusion (4 flags, stacked) — ❌ v023 −3.5 pp (over-subscribes the v018 scheduler).
  - 🔵 `xla_jf_spmd_threshold...` — INVALID on this build (crashes; banned).
  - 🔵 flag-ablation (which of the 7 carry the +3.4 pp) — UNEXPLORED (attribution, not frontier-moving).
- **Splash attention** (6 exp) — frontier kernel = upstream JAX splash, bq=2048/bkv=1024
  - 🏆 v008 (32.4%), 🏆 v009 (seq8192 30.4%); tokamax-splash ❌ v020 (−8.5%); block-tuning ❌/⚠️ v021 (seq2048 −9.4%) / v022 (seq8192 neutral). **Topic exhausted.**
- **tokamax CE** (6 exp) — ✅ v013 correct, ❌ v014/v016 (memory enabler, never a throughput lever). **Closed.**
- **FSDP / collective** (2 exp) — SparseCore-offload ❌ v003 (bs1)/v017 (bs4). **Closed** (qwen3 collective share only 6.3%).
- **Activation checkpointing** (3 exp) — 🏆 v004/v005 (remat, the batch unlock). Baseline.
- **Batch/seq** (2 exp) — ❌ v001/v002 (pre-remat OOM). Subsumed by remat.
- 🔵 **Data-transformation overhead** (0 exp) — UNEXPLORED; the 4.5% GQA-layout-bridge bucket.
- 🔵 **torch.compile / scan** (0 exp) — UNEXPLORED; profile says NOT the dominant bottleneck (fully-unrolled but cost is runtime).
- 🔵 **Pallas kernels (Qwen3-specific)** (0 exp) — fused QK-norm+RoPE+QKV; QK-norm already XLA-fused, so only wins if it also absorbs the layout bridges (effort L).

## Coverage gaps — what's NOT been explored

| Topic | Tried | Unexplored | Prior |
|-------|-------|------------|-------|
| data-transformation overhead | none | layout/sharding to avoid QKV→splash GQA transposes (4.5% bucket) | medium |
| torch.compile/scan | none | scan-over-layers (profile: not dominant; compile-time lever) | low |
| Pallas kernels | splash, CE | fused QK-norm+RoPE+QKV absorbing layout bridges (effort L) | low-med |
| seq scaling | 2048, 8192 | seq16384 (coverage probe) | low |
| XLA flag attribution | full bundle | flag-ablation of the 7 (reusable prior, not frontier) | n/a |

## Recommended next directions

| # | Direction | Topic | Gain | Conf | Effort | Score | Source |
|---|---|---|---|---|---|---|---|
| 1 | Layout/sharding to avoid QKV→splash GQA transposes | data-transformation | +1–4% | medium | M | med | local-gap: v018 profile data-formatting 4.5% |
| 2 | scan-over-layers (compile/scheduling-quality probe) | scan | 0–2% | low | M | low | sibling: llama3-jax used scan; profile says not dominant |
| 3 | Fused QK-norm+RoPE+QKV Pallas kernel (HLO-prefilter FIRST) | Pallas | +1–3% | low | L | low | local-gap: Qwen3-specific; must absorb layout bridges to beat XLA fusion |
| 4 | XLA flag-ablation of the 7 scheduler flags | libtpu/XLA | 0 (attribution) | high | S | — | local-gap: reusable prior for torchax lane |
| 5 | seq16384 splash-scaling coverage probe | seq | coverage | low | S | low | local-gap |

**Honest assessment**: no high-confidence frontier-mover remains. #1 (layout) is the best-prior untested lever — it targets a real, unexplored 4.5% bucket via a compiler/layout change (the transferable category), at medium effort. Everything else is low-prior or attribution-only. The seq2048 frontier at 35.8% (MXU 48.3%) is a strong result near the practical ceiling for non-deep-kernel levers.

## Anti-recommendations (don't pursue)

- **More XLA scheduler/overlap flags** — async-collective-fusion (v023) proved a 2nd overlap mechanism over-subscribes the v018 scheduler. The scheduler-overlap budget is saturated.
- **tokamax-splash, splash block-tuning** — refuted/neutral at both seqs; the kernel is at its plateau.
- **SparseCore-offload, tokamax-CE, raw batch-scaling** — all closed (refuted across shapes).
- **Custom RMSNorm Pallas kernel** — Pallas catalog + Gemma4 exp33 (−8.1%) say XLA-fusion is sufficient on TPU; the norms bucket is already fused.
- **`xla_jf_spmd_threshold_for_windowed_einsum_mib`** — invalid on this XLA build (crashes).

## Cross-lane brief

- **qwen3-cc-torchax**: baseline 19.2% MFU, optimization loop not yet run. The jax frontier (35.8%) is +16.6 pp ahead. Transferable TO torchax once it iterates: remat, splash, **the XLA scheduler flag bundle** (the cleanest universal win), CE (memory only). The **flag-ablation (rec #4)** would hand torchax a pre-validated minimal flag set.
- **llama3-8b-jax** (same arch class, same v6e-8): frontier **43.3% MFU** at bs4/seq8192. The ~7.5 pp gap to qwen3-jax is NOT closable by the levers llama3 used that qwen3 already tried-and-refuted (tokamax-splash, SparseCore, CE-for-batch). It reflects a seq8192 batch-occupancy regime qwen3 can't reach (CE refuted as its batch enabler, v016). Conclusion: qwen3-jax's ceiling on this hardware is structurally below llama3's for the explored lever set.
  - **Conflict resolved**: llama3-jax got +4.4% from tokamax-splash and +3.4% from SparseCore-offload; qwen3-jax refuted both. Stack/head-config difference (qwen3 32q/8kv hd128, low 6.3% collective share) — these llama3 levers do not port. Already documented; not re-importable.

## Experiment ledger (machine-readable)

| v-ID | Date | Variant | Topic | Verdict | Frontier-shift |
|---|---|---|---|---|---|
| baseline | 2026-06-02 | 8B/v6e-8 | batch-seq | supported | yes (20.5%) |
| v001 | 2026-06-02 | 8B/v6e-8 | batch-size | invalid | no (OOM) |
| v002 | 2026-06-02 | 8B/v6e-8 | batch-size | invalid | no (OOM) |
| v003 | 2026-06-02 | 8B/v6e-8 | fsdp-collective | refuted | no |
| v004 | 2026-06-02 | 8B/v6e-8 | activation-checkpointing | supported | yes (→22.0%) |
| v005 | 2026-06-02 | 8B/v6e-8 | activation-checkpointing | supported | yes (→25.1%) |
| v006 | 2026-06-02 | 8B/v6e-8 | splash-attention | invalid | no (VMEM) |
| v007 | 2026-06-02 | 8B/v6e-8 | splash-attention | invalid | no (VMEM) |
| v008 | 2026-06-02 | 8B/v6e-8 | splash-attention | supported | yes (→32.4%) |
| v009 | 2026-06-02 | 8B/v6e-8 | splash-attention | supported | yes (seq8192 30.4%) |
| v010 | 2026-06-02 | 8B/v6e-8 | tokamax-ce | invalid | no (impl) |
| v011 | 2026-06-02 | 8B/v6e-8 | tokamax-ce | invalid | no (absl) |
| v012 | 2026-06-02 | 8B/v6e-8 | tokamax-ce | invalid | no (absl) |
| v013 | 2026-06-02 | 8B/v6e-8 | tokamax-ce | supported | no (canary) |
| v014 | 2026-06-02 | 8B/v6e-8 | tokamax-ce | refuted | no |
| v016 | 2026-06-02 | 8B/v6e-8 | tokamax-ce | refuted | no (seq8192 29.5%) |
| v017 | 2026-06-02 | 8B/v6e-8 | fsdp-collective | refuted | no |
| v018 | 2026-06-02 | 8B/v6e-8 | libtpu-xla-flags | supported | **yes (→35.8%)** |
| v019 | 2026-06-02 | 8B/v6e-8 | libtpu-xla-flags | inconclusive | no (seq8192 neutral) |
| v020 | 2026-06-02 | 8B/v6e-8 | splash-attention (tokamax) | refuted | no |
| v021 | 2026-06-02 | 8B/v6e-8 | splash-attention (blocks) | refuted | no |
| v022 | 2026-06-02 | 8B/v6e-8 | splash-attention (blocks) | inconclusive | no (seq8192 neutral) |
| v023 | 2026-06-02 | 8B/v6e-8 | fsdp-collective (async-fusion) | refuted | no |

## Sources

- Prior retrospective: [2026-06-02-qwen3_cc-jax-retrospective.md](2026-06-02-qwen3_cc-jax-retrospective.md)
- v018 frontier profile + bottleneck verdict: [v018 page](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v018-xla-flag-stack.md); observation [fsdp-rs-not-overlapped](../observations/qwen3-jax-fsdp-rs-not-overlapped.md)
- Per-lane log: `wiki/experiments/qwen3_cc_autoresearch_optimization/jax/log.md`
- Model page: `wiki/models/qwen3-cc-jax.md`
- Topic taxonomy: `wiki/model-optimization-index.md`; Pallas catalog: `wiki/analyses/2026-04-23-pallas-kernel-directory.md`

## See also

- [Qwen3 8B — jax](../models/qwen3-cc-jax.md)
- [Prior retrospective](2026-06-02-qwen3_cc-jax-retrospective.md)
