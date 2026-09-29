---
title: "Retrospective — qwen3_cc / jax as of 2026-06-02 (#4, post maxtext-CE arc)"
type: analysis
tags: [retrospective, qwen3-cc, jax]
model: qwen3_cc
lane: jax
created: 2026-06-02
updated: 2026-06-02
---

# qwen3_cc / jax retrospective #4 — 2026-06-02 (post maxtext-CE / offload arc)

**Mode**: INCREMENTAL (prior: [#3](2026-06-02-qwen3_cc-jax-retrospective-3.md), 24 exp through v024; +13 new
v025–v037). Taxonomy + Pallas catalog mtimes predate the prior → no force-FULL. Variant matrix unchanged (8B/v6e-8).
**Scope**: 8B/v6e-8. **Period**: 2026-06-02 (v025–v037). **Experiments**: 37 total (baseline + v001–v037, v015 never existed) — +13 since #3.
**Frontier (current)**: seq2048 = **v018 35.8% / 6,964 tok/s/chip**; seq8192 = **v035 34.6% / 6,030 tok/s/chip** (was v009 30.4% / 5,305 at #3 → **+13.7% tok/s/chip**).

## Headline — retrospective #3's "lane is done" conclusion was OVERTURNED

#3 (and #2, #1) concluded the jax lane was at its "practical ceiling for non-deep-kernel levers" (seq2048
35.8%) and the #1 action was **pivot to torchax**. That was **premature** — exactly the failure mode this
skill exists to prevent (cf. 2026-05-21 gemma4-jax → tokamax-CE +4.18 pp a week later). The post-#3 arc
(v025–v037), driven by the user directive *"use the same approach as MaxText for CE"*, moved the **seq8192**
frontier **30.4% → 34.6% (+13.7% tok/s/chip, 5,305 → 6,030)** and **cracked the seq8192 batch wall** the
v028 closing analysis had wrongly called a "documented hard wall." The lever was a **cross-entropy-kernel
swap** (MaxText's T5X `@jax.custom_vjp`), a kernel-level topic #3 had not surfaced as frontier-relevant.

**Lesson**: #3's exhaustion verdict was confined to *seq2048* + *the levers tried so far*; it missed that
(a) the **seq8192 regime was under-explored** for batch amortization, and (b) **CE-kernel choice** was a
frontier-moving axis, not just a memory enabler. The never-stop retrospective gate caught this.

## What the new arc (v025–v037) established

- **MaxText's CE is the T5X `@jax.custom_vjp cross_entropy_with_logits`** (full logits + one-hot + explicit
  fused `softmax−onehot` backward) — NOT vocab-tiled, NOT the tokamax streamed kernel. Ported as
  `--use_maxtext_ce`; CPU bit-identical to `_ce`.
- **It cracked the seq8192 batch wall**: v034 bs2 5,992 > v028 bs1 5,632 (the prior bs1>bs2>bs3
  anti-amortization was the *tokamax/autodiff CE-backward transient*, not structural). v035 bs3 6,030 (frontier),
  fits **without offload** (the CE removed tokamax's 4.6G f32[H,V] weight-gather).
- **scan-over-layers + collective-overlap** (v028) was the *first* +6.2% (the stacked-param `lax.scan` made the
  async-collective-fusion flags productive — they had regressed on the unrolled graph at v023).
- **Seq-specificity**: both scan (v032) and maxtext-CE (v037) **regress at seq2048** — they are seq8192-specific
  (large-logit / large-collective-tail regime). seq2048 keeps plain `_ce`, no scan.
- **Named host-offload is refuted twice** (v030 w/ tokamax-CE −18%; v036 w/ maxtext-CE −18.6%): our
  `pinned_host` offload is **not pipelined** (host round-trip on the critical path), and it tags proj/mlpwi,
  not the RMSNorms that dominate the recompute.

## Mechanism tree (new + updated topics; see #3 for v001–v024 detail)

- **Cross-entropy kernel** (NEW frontier-moving topic; v010–v014/v016 tokamax + v029/v033–v037 maxtext-CE)
  - 🏆 **maxtext-CE T5X custom_vjp** — v034 bs2 5,992 (wall cracked), 🏆 v035 bs3 **6,030 frontier**; ✅ v033 bs1 parity; ❌ v037 seq2048 −6.9% (seq8192-specific)
  - ❌ tokamax streamed CE — v014/v016 below frontier; 💥 v029 bf16-weight (mosaic kernel hardcodes f32 scratch)
  - 🔵 **norm-inclusive save/offload CE-adjacent remat** — UNEXPLORED (the v035 profile says RMSNorm recompute is the residual)
- **scan / torch.compile** (NEW: v028,v030,v031,v032) — 🏆 v028 +6.2% seq8192 (makes overlap flags productive); ❌ v032 seq2048 −3.4% (seq-specific)
- **FSDP / collective + HBM offload** (v025,v026,v027,v030,v036 + prior v003,v017,v023) — ❌ named-offload refuted twice (v030/v036, not pipelined); ⚠️ v025 offload fits bs2 (logit-wall finding); 💥 v026/v027 OOM bs3; reduce-scatter via windowed-einsum is **already better than MaxText** (v035 profile)
- **Batch/seq amortization** (v031 + prior v001,v002) — bs2 amortizes at seq8192 ONLY with maxtext-CE (v034); plateaus at bs3 (v035 +0.6%); ❌ v031 (bs2 + tokamax-CE) anti-amortizes
- 🔵 **MXU occupancy / logical-axis-rules tile alignment** — UNEXPLORED; v035 profile: MXU 53.6% vs MaxText 61.2% (+156 ms, ~+3-4% of the residual gap)
- 🔵 **fused QK-norm+RoPE+QKV Pallas** (0) — still unexplored, low prior. 🔵 **seq16384** (0) — coverage.

## Search-strategy assessment

- **Frontier moved 2× since #3** (v028, then v034/v035) — NOT stuck. The lane is in an *active-progress* state,
  the opposite of #3's read.
- **Kernel work is now the productive axis**: the +7% seq8192 gain came from a CE-kernel swap + scan (structural),
  not single-axis flags. #3's single-axis-heavy history was real, but the breakthrough was kernel-level — vindicating
  "when flags plateau, go to kernels."
- **seq8192 at 86.9% of MaxText** (was 81% at v028, 76% at v009). Residual ~13% attributed (v035 profile):
  ~85% remat-recompute (norms), ~29% MXU occupancy; reduce-scatter already better than MaxText.

## Coverage gaps — what's NOT been explored (ranked remaining levers)

| Topic | Unexplored option | Evidence | Effort |
|---|---|---|---|
| MXU / logical-axis | tile-alignment via `logical_axis_rules` to match MaxText's larger GeMM tiles | v035 profile: MXU 53.6% vs 61.2%, +156 ms (~+3-4%) | M–L (sharding-layout rewrite, uncertain payoff) |
| Remat policy | norm-inclusive **selective save** (keep RMSNorm outputs on-device/cheap) at bs2 where HBM has headroom | v035 profile: norm recompute = 17.4% vs MaxText 7.2%; v036 showed offload-of-proj doesn't cover norms | M (remat policy edit) |
| Pipelined offload | overlap the pinned_host copy with compute (MaxText <0.1%) | v036 refuted stock offload (not pipelined) | **L+ (XLA/kernel-authoring — out of scope)** |
| Pallas | fused QK-norm+RoPE+QKV kernel | local-gap; qwen3 kernel-swaps have regressed | L, low prior |
| seq | seq16384 coverage | local-gap | S, coverage-only |

## Recommended next directions

| # | Direction | Topic | Gain | Conf | Effort | Score | Source |
|---|---|---|---|---|---|---|---|
| 1 | **MXU occupancy via `logical_axis_rules` tile alignment** (match MaxText's GeMM tiling at bs3 seq8192) | MXU/sharding | +3-4% | medium | M–L | ~0.6 | local-gap: v035 profile-analyzer attribution |
| 2 | **Norm-inclusive selective-save remat** at bs2 seq8192 (save RMSNorm outputs; cut the 17.4% recompute) | remat | +2-4% | low-medium | M | ~0.5 | local-gap: v035 profile (norms dominate recompute); v036 showed proj-offload misses norms |
| 3 | maxtext-CE on the **torchax** sibling lane (framework-agnostic; torchax still at 19.2%) | cross-lane | large | medium | M (new lane) | — | sibling: torchax untouched; CE+remat+splash+flags all transfer |
| 4 | fused QK-norm+RoPE+QKV Pallas (HLO-prefilter first) | Pallas | +1-3% | low | L | low | local-gap |
| 5 | seq16384 coverage | seq | coverage | low | S | low | local-gap |

**The lane is NOT exhausted.** Directions #1 and #2 are real, profile-attributed, in-scope-for-config/refactor
levers toward the remaining 13% MaxText gap. Both are model-code edits (medium effort) with uncertain payoff;
#1 is the higher-gain. Per the active session constraint, the MXU/logical-axis rewrite (#1) should be confirmed
with the user before dispatch (it is a non-trivial sharding-layout change). #2 (selective-save remat) is the
lighter, also-untried alternative.

## Anti-recommendations (verified closed)

- **Named host-offload** (`--offload_remat`) — refuted TWICE (v030, v036); our pinned_host path is not pipelined. A
  *pipelined* offload is kernel-authoring (out of scope), not the same lever.
- **maxtext-CE @ seq2048** (v037) and **scan @ seq2048** (v032) — both regress; seq8192-specific. Keep plain `_ce`, no scan at seq2048.
- **bs4+ @ seq8192** — amortization saturated at bs2–bs3 (v035 +0.6% over bs2); higher batch won't help.
- Carried from #3: tokamax-CE-as-throughput-lever, SparseCore-offload, tokamax-splash, splash block-tuning,
  async-collective-fusion *on the unrolled graph* (it works WITH scan — v028), `xla_jf_spmd_threshold` (crashes),
  QKV-layout reshaping (intrinsic), raw pre-remat batch scaling.

## Cross-lane brief

- **qwen3-cc-torchax**: 19.2% MFU, loop still not run — now an even stronger target: the *entire* validated jax
  stack (remat + splash + 7-flag scheduler bundle + scan + **maxtext-CE**) is framework-agnostic and should
  transfer. This is the largest single opportunity in the model family.
- **qwen3-cc-maxtext**: reference ceiling 45.3% / 6,942 (seq8192 bs3), 38.0% / 7,116 (seq2048 bs4). jax now at
  86.9% (seq8192) / 97.9% (seq2048).
- **llama3-8b-jax**: 43.3% seq8192 — the maxtext-CE + scan levers that just worked here may transfer (llama3-jax
  was already higher; check whether it already has a custom_vjp CE).

## Experiment ledger (machine-readable — v025–v037 appended to #3's verified v001–v024)

| v-ID | Date | Variant | Topic | Verdict | MFU | Frontier-shift |
|---|---|---|---|---|---|---|
| v025 | 2026-06-02 | 8B/v6e-8 | hbm-offload | inconclusive | — | no (logit-wall finding) |
| v026 | 2026-06-02 | 8B/v6e-8 | hbm-offload | invalid | — | no (OOM +2.34G) |
| v027 | 2026-06-02 | 8B/v6e-8 | hbm-offload | invalid | — | no (OOM, CE-weight wall) |
| v028 | 2026-06-02 | 8B/v6e-8 | scan + fsdp-overlap | supported | 32.3% | yes (seq8192) |
| v029 | 2026-06-02 | 8B/v6e-8 | tokamax-ce (bf16 weight) | invalid | — | no (kernel f32 scratch) |
| v030 | 2026-06-02 | 8B/v6e-8 | scan+offload+ce (bs3) | inconclusive | 26.4% | no (fits, regresses) |
| v031 | 2026-06-02 | 8B/v6e-8 | batch/seq (bs2 no-offload) | refuted | 31.9% | no |
| v032 | 2026-06-02 | 8B/v6e-8 | scan (seq2048) | refuted | 34.6%@s2k | no (−3.4% vs v018) |
| v033 | 2026-06-02 | 8B/v6e-8 | cross-entropy (maxtext-CE bs1) | supported | 32.4% | no (parity) |
| v034 | 2026-06-02 | 8B/v6e-8 | cross-entropy (maxtext-CE bs2) | supported | 34.4% | yes (seq8192, wall cracked) |
| v035 | 2026-06-02 | 8B/v6e-8 | cross-entropy (maxtext-CE bs3) | supported | **34.6%** | **yes (seq8192 frontier 6,030)** |
| v036 | 2026-06-02 | 8B/v6e-8 | hbm-offload (maxtext-CE+offload) | refuted | 28.2% | no (−18.6%) |
| v037 | 2026-06-02 | 8B/v6e-8 | cross-entropy (maxtext-CE seq2048) | refuted | 33.4% | no (−6.9% vs v018) |

## Sources

- Prior retrospectives: [#1](2026-06-02-qwen3_cc-jax-retrospective.md), [#2](2026-06-02-qwen3_cc-jax-retrospective-2.md), [#3](2026-06-02-qwen3_cc-jax-retrospective-3.md)
- maxtext-CE climb closing: [2026-06-02-qwen3-cc-jax-maxtext-ce-climb.md](2026-06-02-qwen3-cc-jax-maxtext-ce-climb.md)
- Deep-read of v025–v037 experiment pages; model page `wiki/models/qwen3-cc-jax.md`; lane log.
- Topic taxonomy `wiki/model-optimization-index.md`; Pallas catalog `wiki/analyses/2026-04-23-pallas-kernel-directory.md`

## See also

- [Qwen3 8B — jax](../models/qwen3-cc-jax.md) · [maxtext-CE climb](2026-06-02-qwen3-cc-jax-maxtext-ce-climb.md) · retrospectives [#1](2026-06-02-qwen3_cc-jax-retrospective.md)/[#2](2026-06-02-qwen3_cc-jax-retrospective-2.md)/[#3](2026-06-02-qwen3_cc-jax-retrospective-3.md)
