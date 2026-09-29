---
title: "Qwen3-CC-JAX autonomous session — process retrospective (session 5d03ff24)"
type: analysis
tags: [retrospective, process, autoresearch-loop, qwen3, claude-opus-4-8]
created: 2026-07-08
updated: 2026-07-08
---

Retrospective on the Claude Code (Opus 4.8) session `5d03ff24-35cc-4e5e-89b4-9b985a700fb4` that ran the autoresearch loop on **`qwen3_cc` / jax lane**, 2026-06-02 → 2026-06-21. This page judges the run against the SCHEMA + `program.md` process: what the process is, how faithfully the agent followed it, what went well, what went badly (with concrete examples), and a prioritized backlog of **fixable process gaps**. It is a *process* analysis, not a results analysis — the science is summarized only where it illustrates a process point.

The transcript itself (14,327 lines, 35 MB) lives at `~/.claude/projects/-mnt-disks-persist-torch-tpu-tpu-performance-autoresearch-wiki/5d03ff24-35cc-4e5e-89b4-9b985a700fb4.jsonl`; it is not under `raw/` and is not a citable wiki artifact, but every claim below is anchored to a filed experiment/analysis page that is.

## The process it was supposed to follow

Per [SCHEMA](../../SCHEMA.md) + the [program.md hierarchy](../experiments/program.md):

1. `/create-experiment` — bootstrap the model family (folders, model-level `program.md`, model page stubs).
2. `/start-experiment` — resolve `program.md` (root → model → lane), select hardware, discover clusters, occupancy check, then start `/loop` in never-stop mode.
3. **The loop**, per iteration: formulate a falsifiable hypothesis → (optional AOT/HLO pre-flight screen) → run on GKE via `gke-cluster-runner` → capture profile → `profile-analyzer` attributes the bucket → record observations → update the model page's variant row → **file next-hypothesis stubs** → update `index.md` / `log.md`.
4. `/stop-experiment` — reap orphan workloads, file missing pages, run LINT, append a clean-shutdown marker.

Verdict discipline (SCHEMA rules 6–8): `supported` requires improvement beyond noise **and** a semantics check **and** no tracked-metric regression; a crash or a semantic change is `invalid` and its "speedup" is **not** reported.

## What actually happened — the shape of the run

The "19-day session" is misleading. Tool-call activity clusters into three regimes:

| Phase | Days | Tool calls | Character |
|-------|------|-----------|-----------|
| **Autonomous burst** | 2026-06-02 | 947 | `/start-experiment` → `/loop` → **50 never-stop auto-continuations**; v001→v040 unsupervised |
| **Human-steered climb** | 06-03 / 04 / 05 | ~370 | directive "inspect MaxText, close the gap, try all" → v041–v045 + closing analyses |
| **Meta / ops** | 06-12 / 15 / 16 | ~415 | cross-lane merges, MFU-formula reconciliation, visualizations, blog, promotion |

So the *research loop* was **~1 intense autonomous day + 3 steered days**; the remaining two calendar weeks were sync/visualization/publishing, not the loop. Days 06-06 → 06-11 have zero activity (the session was idle, then manually resumed).

**Output artifacts** (filed, on disk):

- **48 experiment pages** — 45 jax, 2 maxtext reference, 1 torchax baseline.
- jax verdict distribution: **12 supported, 16 refuted, 11 invalid, 5 inconclusive, 1 baseline**.
- **Result:** seq2048 20.5% → **35.8% MFU** (v018, +74% tok/s/chip); seq8192 → **34.8% MFU / 6,068 tok/s/chip** (v045) = **87.4% of the MaxText reference**, converging to a ceiling correctly attributed at op level to MaxText's host-offload *pipelining* (a runtime/build capability, out of config scope).

## What went well

### 1. Structural compliance was near-perfect
Across the 45 jax experiment pages: **44/45 carry a `## Profile` section, 45/45 carry `## Next hypotheses`.** The queue-refill discipline — the single most important habit for keeping an autonomous loop from stalling — held on every page.

### 2. Honest verdict taxonomy; failures were never laundered
11 `invalid` + 16 `refuted` verdicts, and *not one* was reported as a win. Examples of correctly-rejected results:
- [v001](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v001-batch-bs2.md) / [v002](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v002-batch-bs4.md) — compile-time OOM, no run → `invalid`, filed as an HBM-ceiling marker rather than a "regression."
- [v043](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-05-v043-tp2-s8k.md) — TP=2 "regressed," but the run had a batch-halving confound, so it was called `inconclusive`, **not** `refuted`, and a clean control was dispatched (see §3).

### 3. Real methodology under autonomy
- **Confound control.** [v043](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-05-v043-tp2-s8k.md) (tp=2) halved the global batch, confounding the regression. The agent recognized it, filed the run as inconclusive, and dispatched [v043b](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-05-v043b-tp2-bs6.md) — tp=2 at bs6 to match global batch — which cleanly refuted TP. Textbook.
- **Contradiction discipline.** When the CE port ([v034](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v034-maxtext-ce-s8k-bs2.md)) overturned the earlier "seq8192 hard wall" conclusion, the agent used `[!warning]` supersession blocks on the model page rather than silently overwriting — the strikethrough+warning is preserved in [qwen3-cc-jax.md](../models/qwen3-cc-jax.md).
- **Failures fed the loop.** The invalid OOM chain [v025](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v025-offload-remat-s8k-bs2.md) → [v026](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v026-offload-ce-s8k-bs3.md) → [v027](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v027-named-offload-ce-s8k-bs3.md) is what *discovered* that the seq8192 batch ceiling is the LM-head logit/CE-weight wall, not layer activations — the insight that motivated the whole MaxText-CE arc.

### 4. Self-correction when told it was "done"
Retrospective [#4](2026-06-02-qwen3_cc-jax-retrospective-4.md) explicitly overturned [#3](2026-06-02-qwen3_cc-jax-retrospective-3.md)'s "lane exhausted, pivot to torchax" call — the maxtext-CE arc then moved the frontier +13.7%. The premature-exhaustion trap fired, and the loop caught it (this is the exact failure the `create-retrospective` skill exists to prevent).

## What went badly — with concrete examples

### P1 — ~¼ of GKE runs were avoidable no-step crashes
Of 45 jax experiments, **11 were `invalid` with zero steps executed.** Several were not physics — they were integration bugs that a CPU pre-flight would have caught:

| Exp | Failure | Was it avoidable on CPU? |
|-----|---------|--------------------------|
| [v001](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v001-batch-bs2.md), [v002](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v002-batch-bs4.md) | compile-time OOM, no run | **Yes** — AOT `.lower().compile()` OOMs on CPU |
| [v006](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v006-splash-bs4.md), [v007](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v007-splash-s8k.md) | splash-backward scoped-VMEM overrun; v007 additionally **forgot `--xla_tpu_scoped_vmem_limit_kib`** | Partly — v007 was a config-carry omission |
| [v010](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v010-ce-bs2.md) | wrong tokamax impl name (`chunked_xla` absent in build) | **Yes** — a build-capability check |
| [v011](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v011-ce-mtpu-bs2.md), [v012](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v012-splash-ce-bs6.md) | **absl argv-parsing collision** — the CE kernel never ran; same bug twice | **Yes** — a 1-step CPU smoke test |
| [v025](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v025-offload-remat-s8k-bs2.md), [v026](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v026-offload-ce-s8k-bs3.md), [v027](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v027-named-offload-ce-s8k-bs3.md), [v029](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v029-full-stack-s8k-bs3.md) | compile-OOM / trace-time dtype crash | Mostly **yes** via AOT |

SCHEMA's own **AOT screening (RUN-EXPERIMENT step 1b)** and **HLO pre-filter (FORMULATE step 2b)** exist to catch exactly these on CPU before spending TPU time — and they were not applied systematically. Note also **the same bug shipped twice** (v011 then v012 = identical absl-flags collision) because the fix wasn't verified before the next dispatch reused the path.

### P2 — the comparison denominator was wrong during the entire climb
The frontier was tracked as "% of MaxText MFU" throughout 06-02→06-05, but the MaxText reference was **non-causal (45.3% MFU), and the causal-adjusted value is 39.8%** — a fact not reconciled until **06-12 → 06-16**, two weeks later. The user had to force the reconciliation manually, repeatedly:
- 06-12: *"check that mfu formula for all experiments is correct and same, and let me know how it compares to maxtext"*
- 06-15: *"cc result (this session) for 8k seems low, is it under-explored?"*
- 06-16: *"if I check mfu graph, neither variation crosses maxtext 8k mfu line over 45%, but when I switch to tps cx and cc5 cross it. What is wrong here? Is maxtext reference incorrect?"*

Every "N% of MaxText" milestone on the model page was computed against an unpinned, ultimately-incorrect denominator. The science underneath was fine; the *reported comparison* was not, and nothing in the loop caught it.

### P3 — one mega-session did the loop *and* all the ops
The single session hit **context compaction 5 times** ("continued from a previous conversation that ran out of context"). The never-stop research loop, the 4-way cross-lane repo merge (cc/ag/cx/cc5), the interactive HTML visualization builds, the blog post, and even open-source promotion were all crammed into the same conversation. This is what caused the compactions and buried the actual research under publishing work.

### P4 — single lane, single variant
`/create-experiment` said "start with torchax," but 45/48 experiments went to jax; the **torchax lane never got past its baseline** and maxtext was used only as a reference. All 45 jax runs targeted a single matrix row (`8B/v6e-8`) — no hardware scale-up row was ever exercised, so nothing tested cross-variant transfer (the model page's Knobs-translation matrix stayed hypothetical for this lane).

### P5 — mandatory `## Sources` missing on most run pages
Only **19/45** jax experiment pages carry the `## Sources` section that SCHEMA rule 7 mandates (the profile path repeated for lineage). Some omissions are legitimate (no-run `invalid` pages have no profile), but many run pages simply skipped it. LINT would have flagged this — but LINT only ran at the very end, not mid-loop.

### P6 — the discrete audit gates were collapsed into inline reasoning
`formulate-hypothesis` was invoked **0 times**; `profile-analyzer` only **4 times for 45 experiments**. Hypotheses were formulated and profiles read inline by the master. (Caveat: those skills likely post-date early June — but the effect is that the pre-filter and profile-attribution gates were not discrete, re-auditable steps; they lived in prose that context-compaction can erase.)

### P7 — `stop-experiment` never cleanly closed the autonomous phase
The loop was left open and manually re-entered on later days rather than closed with `/stop-experiment` (reap + LINT + shutdown marker). There is no clean-shutdown boundary between "the loop" (ending ~06-05) and "the ops work" (06-12+).

## Problems to address later — prioritized backlog

Ranked by (impact on loop integrity × frequency observed / effort to fix).

### A. Make CPU pre-flight a hard, un-skippable gate before every GKE dispatch  *(fixes P1)*
The single highest-value fix. On this run it would have eliminated ~9–11 wasted TPU launches (v001/v002/v006/v007/v010/v011/v012/v025/v026/v027/v029).
- **Where:** add a mandatory step to `wiki/experiments/program.md` (loop step, pre-dispatch): `jax.jit(fn).lower(*args).compile()` on CPU; on OOM/shape/flag error, do **not** dispatch — file the `invalid` page from the CPU result and move on.
- **Add a build-capability probe:** before using a tokamax impl name or an XLA flag, assert it exists in the *installed wheel* (not the checkout) — v010's `chunked_xla` and the [qwen3-jax libtpu flag-name](../../SCHEMA.md) `xla_` vs `xla_tpu_` class of crash are both cheap to pre-check.
- **Enforce "fix-then-verify" between dispatches:** v011→v012 shipped the same absl bug twice. A 1-step CPU smoke run after any harness/flag fix, before the next dispatch, prevents duplicate-crash pairs.
- **Make it gate-able in the hook**, so the never-stop loop physically cannot dispatch an un-screened run.

### B. Pin the metric contract *before* the climb starts  *(fixes P2)*
- **Where:** the model page must carry, in frontmatter or a fixed "Target metrics" block, the **exact MFU/TPS formula** and the **causal-adjusted** reference value, filed *before the first optimization experiment*.
- Add a LINT check: every experiment's reported "% of reference" must be recomputed from the pinned formula; flag drift. This turns P2 from a two-week-late manual catch into a mechanical gate.
- Record whether the reference is causal or non-causal explicitly — the 45.3% → 39.8% adjustment was the root of a full day of 06-16 confusion.

### C. Split "loop" sessions from "ops" sessions  *(fixes P3, P7)*
- Operational policy: a never-stop `/loop` session runs *only* the loop. Repo merges, visualization builds, blog/promotion → separate sessions.
- End every autonomous phase with `/stop-experiment` so there is a clean boundary and a LINT pass before the context is contaminated with ops work. This directly reduces the compaction rate (5× here).

### D. Run LINT mid-loop, not only at shutdown  *(fixes P5, and surfaces P2/P6 early)*
- Wire a lightweight LINT into the loop cadence (e.g. every N experiments or on each retrospective). It would have caught the 26 missing `## Sources` sections while context was fresh, and can host the metric-drift check from (B).

### E. Budget across the matrix instead of 45-deep on one row  *(fixes P4)*
- The `program.md` iteration ladder should *schedule* the second lane (torchax) and at least one hardware scale-up row rather than letting the loop exhaust a single `(size, hw)` cell. 45 experiments on `8B/v6e-8` with zero cross-variant transfer tests is over-concentration.
- Add a stop-signal: when K consecutive experiments on one row are sub-noise (the v038–v045 tail was almost all parity), the loop should *rotate the variant/lane*, not keep probing.

### F. Restore the discrete audit gates now that the skills exist  *(fixes P6)*
- Route hypothesis creation through `/formulate-hypothesis` (it owns the falsifiability + HLO pre-filter checklist) and profile attribution through the `profile-analyzer` agent, every iteration. Discrete tool calls survive compaction; inline prose does not.

## Bottom line

Methodologically the run was **strong where it counts**: falsifiability, confound control (v043→v043b), contradiction discipline, honest failure accounting, and self-correction from a premature "exhausted" call were all genuinely good, and it produced a real, correctly-bounded optimization result (+74% seq2048, +7% seq8192 CE win, converged to a real ceiling) largely autonomously. The weaknesses are **operational, not scientific**: it burned roughly a quarter of its TPU launches on crashes that a CPU pre-flight would have caught, tracked its headline metric against an unpinned/incorrect denominator for two weeks, and ran the research loop and all the publishing work in one overloaded session that compacted five times. Fixes A–D are cheap, mechanical, and would remove most of the waste without touching what already works.

## See also
- [qwen3-cc-jax model page](../models/qwen3-cc-jax.md) — the live variant matrix + climb narrative this session produced.
- [2026-06-05 gap-closing "try all" retrospective](2026-06-05-qwen3-cc-jax-gap-closing-attempt.md) — the end-of-climb state.
- [2026-06-02 qwen3_cc-jax retrospective #4](2026-06-02-qwen3_cc-jax-retrospective-4.md) — the premature-exhaustion catch.
- [2026-06-02 MaxText vs jax Qwen3-8B MFU gap](2026-06-02-maxtext-vs-jax-qwen3-8b-mfu-gap.md) — the reference-gap teardown (note: its 45.3% figure is the *non-causal* MaxText number — see P2).
- [2026-04-24 Gemma4 JAX ceiling + process retrospective](2026-04-24-gemma4-jax-ceiling-and-process-retrospective.md) — the prior process retrospective this one parallels.
- [SCHEMA.md](../../SCHEMA.md) · [wiki/experiments/program.md](../experiments/program.md) — the process being audited.

## Sources
- Session transcript (not a `raw/` artifact): `~/.claude/projects/-mnt-disks-persist-torch-tpu-tpu-performance-autoresearch-wiki/5d03ff24-35cc-4e5e-89b4-9b985a700fb4.jsonl` — Claude Code / Opus 4.8, 2026-06-02 → 2026-06-21, 14,327 lines.
- All filed `qwen3_cc_autoresearch_optimization/jax/experiments/*.md` pages (v001–v045) and the lane `log.md`.
