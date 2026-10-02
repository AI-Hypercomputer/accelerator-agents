---
title: "Qwen3-CX-JAX autonomous session — process retrospective (Codex / GPT-5.5 lane, session 019e8ab6)"
type: analysis
tags: [retrospective, process, autoresearch-loop, qwen3, codex, gpt-5.5]
created: 2026-07-08
updated: 2026-07-08
---

Retrospective on the **OpenAI Codex CLI (GPT-5.5, reasoning effort high)** session `019e8ab6-81c0-7d32-a52a-1c261499495a` that ran the autoresearch loop on **`qwen3_cx` / jax lane**, 2026-06-02 → 2026-06-13. Unlike the [`cc` (Claude Opus 4.8) session](2026-07-08-qwen3-cc-jax-session-retrospective.md), this one ran in Codex, so the transcript is the Codex *rollout* schema (JSON-line `response_item` / `event_msg` / `compacted` records), the harness has **no subagents** and **no `DO NOT STOP` hook**, and the process leaned on a different autonomy mechanism (below). This page judges the run against the same [SCHEMA](../../SCHEMA.md) + [`program.md`](../experiments/program.md) loop and anchors every claim to a filed experiment page. It is a *process* analysis; the science is summarized only where it illustrates a process point — except for the decisive-win section, which the comparison demands.

The transcript (76,776 lines, 155 MB) lives at `~/.codex/sessions/2026/06/02/rollout-2026-06-02T23-41-17-019e8ab6-81c0-7d32-a52a-1c261499495a.jsonl`; it is not a citable `raw/` artifact, but every claim below resolves to a filed page that is.

## The process it was supposed to follow

Per [SCHEMA](../../SCHEMA.md) + the [program.md hierarchy](../experiments/program.md): `/create-experiment` → `/start-experiment` (resolve program, hardware, occupancy) → the never-stop loop `[formulate falsifiable hypothesis → optional AOT/HLO CPU pre-flight → run on GKE → capture profile → attribute buckets → record observations → update the model page's variant row → **file next-hypothesis stubs** → update index/log]` → `/stop-experiment`. Verdict discipline (rules 6–8): `supported` needs gain-beyond-noise **and** a semantics check **and** no regression; a crash or a semantic change is `invalid` and its "speedup" is **not** reported.

### The Codex autonomy mechanism (the missing `DO NOT STOP` hook analog)

Codex has no Claude Stop hook. Continuous autonomy came from three things, visible in the rollout:
- **`approval_policy: never` + `sandbox: danger-full-access`** (`codex --dangerously-bypass-approvals-and-sandbox`, per `notex_codex.md`) — no per-action approval gate at all.
- **The thread "goal" object.** 5,874 `thread_goal_updated` events, and the model was repeatedly re-fed `<codex_internal_context source="goal">Continue working toward the active thread goal…` — roughly hourly through 06-04/05. This re-injection *is* the structural analog to the never-stop hook: it re-primes the model to keep iterating without a human turn.
- **A `.codex/.loop_active.json` stop hook** that blocks stopping until a recent retrospective exists (the user's 06-03 note flagged it as likely misconfigured: *"this seems to be wrong"*). It never functioned as the cc hook did.

Two structural harness differences matter for the audit: Codex drove **`kubectl` (3,373 calls), `xpk` (199), `gsutil` (910) directly** from the master (no `gke-cluster-runner` subagent), and it called the **xprof MCP tools inline** (`get_memory_profile` 290×, `get_op_profile` 286×, `get_profile_summary` 195×, `get_overview` 134×) — there is no `profile-analyzer` subagent in this harness. Discrete audit gates that SCHEMA assumes exist as subagents were folded into the master's own turn.

## What actually happened — the shape of the run

Tool activity by day (event counts) clusters into three regimes:

| Phase | Days | Events | Character |
|-------|------|--------|-----------|
| **Setup / adaptation** | 06-02 | 468 | adapt the Claude-Code repo to Codex (keep Claude+Antigravity compat), wire xprof MCP, reproduce the jax baseline |
| **Autonomous seq8192 climb** | 06-03 / 04 / 05 | 6,440 / 23,518 / 21,022 | `/start-experiment` → goal-driven never-stop; v001→v279; MaxText self-discovered ~v204; SparseCore arc v205→v279 |
| **Meta / 2k push** | 06-12 / 13 | 15,524 / 9,804 | MaxText gap re-analysis, MFU-formula reconciliation, numerical/equivalence test, then a *second* start-experiment for seq2048 on `v6e-demo-hjajoo` (fresh v-series to ~v137) |

Days 06-06 → 06-11 are idle (session resumed manually). The *research loop* was thus ~3 intense autonomous days (06-03→06-05) plus a steered 2-day 2k coda. The human was a periodic supervisor: **48 real user messages**, overwhelmingly `status, what's the best MFU?` pulses plus a handful of directives (start-experiment; "find maxtext… close the gap"; "start-experiment for 2k context"; "stop experiment"). The session **compacted 116 times** (`context_compacted`) — vs cc's 5 — the price of one very long goal-driven thread.

**Output artifacts** (filed, on disk): **349 jax experiment pages** + 5 maxtext reference + 1 torchax baseline. The seq8192 series ran v001–**v279** (272 distinct v-numbers); the 06-12/13 2k push **restarted numbering** (v037→~v137), so v-numbers collide across series (e.g. two `v116`).

**Result** (model page [qwen3-cx-jax.md](../models/qwen3-cx-jax.md), variant `8B/v6e-8`):
- **seq8192**: baseline-arc → **7,543–7,544 tok/s/chip / 43.3% old-MFU / ~49.2% MaxText-style MFU** at **global batch 32** (per-chip 4), frontier [v273](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v273-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-rerun.md), peak HBM 30.93/31.25 GiB (0.318 GiB free — nearly full). This **exceeds the MaxText reference** ([bs3 no-vocab-tiling](../experiments/qwen3_cx_autoresearch_optimization/maxtext/experiments/2026-06-05-maxtext-qwen3-8b-v6e8-bs3-no-vocabtiling.md), 6,883 tok/s/chip / 44.9% MFU; prompt cites ~6,953).
- **seq2048** (2k coda): batch-16/17/18 scaling reached **~47.3% MaxText-style MFU**, up from the 20.5% baseline.

## What went well

### 1. Persistence and breadth — and it directly caused the win
**349 experiments** vs cc's 45. The exhaustive flag/kernel-space sweep is *why* cx found the two levers cc never reached (see the decisive-win section), and it **self-discovered MaxText mid-run** (~v204, "somehow it discovered maxtext") without being told. Breadth was not busywork here — it was the mechanism of success.

### 2. Genuine rerun-before-carry discipline under autonomy
Every accepted frontier move was **rerun and required byte-identical optimized HLO** before being carried: [v220](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v220-no-scan-sparsecore-rs-ar-only.md)→[v221](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v221-no-scan-sparsecore-rs-ar-only-rerun.md), [v263](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v263-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first.md)→[v264](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v264-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-rerun.md), [v272](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v272-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources.md)→[v273](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v273-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-rerun.md). Provisional-vs-confirmed was a real, consistently-applied distinction — the source of the `provisional-supported` status values.

### 3. Honest failure accounting + correct semantics gating
NaN runs were **not** laundered despite tempting numbers: [v097–v100](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-04-v097-shard-acts-s8k-bs3-tokamax-splash-qseq2-dense-mask.md) reached ~39% nominal MFU but NaN'd from step 1 and were all marked `invalid`. cx also **empirically killed the host-offload route** ([v180](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-04-v180-no-mlp-sac-offload-remat.md): `offload_remat=True` → 35.7% MFU regression) and wrote **33 CPU-equivalence-tested pages**, so accepted wins carried a semantics check.

### 4. Strong profile grounding, per-lever isolation
The xprof MCP was used heavily and inline; **259/349** pages carry a `## Profile`. Each collective lever was isolated individually rather than bundled: [v218](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v218-no-scan-sparsecore-ag-rs-only.md)/[v219](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v219-no-scan-sparsecore-ag-ar-only.md)/[v220](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v220-no-scan-sparsecore-rs-ar-only.md) turned AG/RS/AR offload on one-at-a-time to prove each was load-bearing.

## The decisive win — how cx beat MaxText where cc stalled

This is the crux the comparison exists for. **cc stalled at 6,068 tok/s/chip (34.8% MFU, 87.4% of MaxText)**, concluded batch could not amortize past bs3 at seq8192, and attributed the residual to an **out-of-scope "host-offload pipelining wall"** (MaxText's edge = host-memory offload pipelining, a runtime capability cc judged it lacked). **cx reached 7,543 tok/s/chip and passed MaxText.** The gap was closed by two mechanisms cc never reached, layered on the shared stack:

**Shared with cc — the memory stack that removes the seq8192 CE wall:** MaxText CE (one-hot/log-softmax-free — dissolves the `bf16[B,8192,151936]` ≈ 11.6 GiB logits temp both sessions identified as the batch ceiling), activation sharding (`shard_acts`), remat, tokamax-Splash + `MAX_LOGIT_CONST=30`, and **no-scan / unrolled layers** ([v182/v183](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-04-v182-no-mlp-sac-no-scan.md) base, 41.8% MFU) — unrolling avoids scan's stacked-layer `[36,B,8192,…]` broadcast that OOM'd at [v033/v034](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-03-v034-shard-acts-s8k-gb24-tp2-noscan.md) and lets XLA free activations per layer.

**Lever 1 — the batch-scaling memory unlock cc did not find: `DISABLE_COLLECTIVE_MATMUL`.** [v116/v117](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-04-v116-shard-acts-s8k-bs3-tokamax-splash-layeragg-collective-matmul-none.md) set `xla_tpu_{all_gather,reduce_scatter}_collective_matmul_mode=none`, which **dropped peak HBM from 28.32 → 22.21 GiB** at batch 3 (≈9 GiB freed). That headroom is exactly what let [v118](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-04-v118-shard-acts-s8k-bs4-tokamax-splash-layeragg-collective-matmul-none.md) **reopen per-chip batch 4 / global batch 32 at seq8192** (26.91 GiB, fits) — the page states outright that "collective-matmul-none creates enough HBM margin for batch 4." This is the direct answer to *did cx push batch scaling cc declared refuted*: **yes.** cc's bs2/bs4 OOMs (v001/v002) were early, pre-stack; cx grew batch by *lowering HBM per step* rather than by host offload, so it never needed the capability cc thought it lacked. (cx confirmed the host route was a dead end at [v180](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-04-v180-no-mlp-sac-offload-remat.md).)

**Lever 2 — the throughput lever that beat MaxText: SparseCore collective offload.** [v205](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v205-no-scan-sparsecore-all-collectives.md) onward offloaded all-gather / reduce-scatter / all-reduce onto the v6e **SparseCore — an *on-chip* engine, not the host** — freeing TensorCore issue slots so collectives overlap matmul. This is the on-chip equivalent of the pipelining cc declared out of scope. It moved MFU 41.8% ([v183](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v205-no-scan-sparsecore-all-collectives.md)) → 42.6% ([v206](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v206-no-scan-sparsecore-all-collectives-rerun.md)) → 43.3% ([v273](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v273-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-rerun.md)), followed by a long RS/AR latency-multiplier + selective-resources micro-tuning tail.

**In one paragraph:** cx beat MaxText not by finding more amortization at bs3 but by *changing the memory budget so batch could grow* — `DISABLE_COLLECTIVE_MATMUL` freed ~9 GiB (v116), which fit global batch 32 (v118) — and then *overlapping the now-larger collective volume on-chip* via SparseCore offload (v205→v273), which is the same overlap MaxText gets from host-offload pipelining but realized on the SparseCore. cc, probing far less of the XLA-flag space, never tried collective-matmul-off, so it never got the HBM headroom to leave bs3, and it framed the remaining gap as an out-of-scope host-runtime capability rather than an on-chip flag it hadn't reached. Breadth, not a smarter single move, is what put cx on the far side of that wall.

## What went badly — with concrete examples

### P1 — the mandatory queue-refill and lineage sections almost entirely collapsed
**`## Next hypotheses`: 3/349. `## Sources`: 2/349.** (cc: 45/45 and 19/45.) The single most important loop-integrity habit — capturing the next move at the cheapest moment — is essentially absent. Codex ran on the re-injected *goal* object instead of a filed hypothesis queue, so next-move reasoning lived in ephemeral prose that **116 compactions** repeatedly erased. There is no persistent hypothesis backlog for this lane.

### P2 — frontmatter integrity is broken lane-wide
**All 349 pages carry `model: qwen3-cc-jax`** (the *wrong* lane page — this lane is `qwen3-cx-jax`) and `qwen3-cc` tags, copied wholesale from the cc scaffold; the frontier [v273](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v273-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-rerun.md) shows it. There is **no SCHEMA `verdict:` enum**; instead **21 distinct ad-hoc `status:` strings** (`refuted`, `completed`, `refuted/tie`, `supported/tie`, `provisional-supported`, `supported_tie`, `invalid_refuted`, …). And the 06-12/13 2k restart **duplicated v-numbers** already used by the seq8192 series (two `v116`). LINT would fail on the variant/model mismatch and the non-enum verdicts.

### P3 — low signal-to-noise; heavy rerun / micro-probe tax
**241/349** pages are `refuted`/`tie`/`completed` noise-level outcomes. Whole stretches are single-scalar sweeps that tie or regress: the [v124–v160](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-04-v124-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-layeragg.md) scoped-VMEM sweep, the v220–v279 RS/AR latency-multiplier neighborhood, the v274–v277 shared-memory sweep — dozens of TPU launches for sub-noise deltas, with **no CPU/AOT pre-filter** to prune them beforehand and **no stop-signal** when K consecutive runs are sub-noise. This is cc's P1 waste class at ~7× the volume; breadth's cost.

### P4 — no CPU pre-flight; ~46 invalid pages, many avoidable on CPU
**46 `status: invalid`** pages. Some are physics (compile-OOM markers like [v034](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-03-v034-shard-acts-s8k-gb24-tp2-noscan.md), [v043](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-03-v043-shard-acts-s8k-bs3-label-ce.md)), but the class of libtpu compile-time SIGSEGV/crash (v083 `async_all_gather_emitter`, v085 `CollectiveComputeDecomposer`, v088 startup, per the Knobs matrix) and the env-omission invalids ([v254](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v205-no-scan-sparsecore-all-collectives.md)/v255 launched **without the frontier-required Tokamax Splash env**, producing `tokamax` count 0 and forcing full reruns v256/v257) are exactly what SCHEMA's AOT screen (RUN step 1b) and a build/flag-capability probe exist to catch on CPU. They were not applied systematically.

### P5 — the discrete audit gates don't exist as re-auditable steps
This is partly harness-structural: Codex has no `profile-analyzer`/`gke-cluster-runner` subagents and no `/formulate-hypothesis` skill invocation, so hypothesis formulation and profile attribution lived inline in the master turn. Inline reasoning is strong for grounding (P-good #4) but **does not survive compaction**, and with 116 compactions the audit trail for *why* each hypothesis was chosen is largely gone.

### P6 — the metric denominator churned, same as cc
The 06-12/13 block exists largely because the human had to force an **MFU-formula reconciliation** ("analyze our mfu formula vs maxtext mfu formula") — the frontier had been reported as `43.3% old MFU` vs `~49.2% MaxText-style MFU` against an unpinned reference. Same root cause as cc's P2: the metric contract was not pinned before the climb.

## Problems to address later — prioritized backlog

Ranked by (loop-integrity impact × frequency / effort).

### A. Restore `## Next hypotheses` + `## Sources` as hard per-page gates *(fixes P1)*
The highest-value fix. A Codex-side page template (or a `.codex` stop-hook check) must refuse to close an experiment page without a non-empty `## Next hypotheses` (or literal `None — <reason>`) and the profile path repeated under `## Sources`. Without a persistent queue, a 349-experiment run leaves *no* backlog for the next session.

### B. Fix frontmatter at the scaffold and add a LINT gate *(fixes P2)*
Correct `model:`/`tags:` to `qwen3-cx-jax` lane-wide (a mechanical sed), adopt the SCHEMA `verdict:` enum instead of 21 free-text `status:` values, and namespace v-numbers per series so the 2k restart cannot collide with seq8192. Add the variant/model-mismatch and verdict-enum checks to LINT so this can't recur.

### C. Make CPU pre-flight + a config-capability probe un-skippable before GKE dispatch *(fixes P3, P4)*
`jax.jit(fn).lower(*args).compile()` on CPU before every launch (catches the compile-OOM and libtpu-crash invalids), plus assert required env/flags/impl-names exist in the *installed image* before dispatch (would have prevented the v254/v255 Tokamax-env-omission reruns). Add a stop-signal: after K consecutive sub-noise runs on one lever, rotate the axis instead of continuing the sweep (would have trimmed the v124–v160 / v220–v279 tails).

### D. Pin the metric contract before the climb *(fixes P6)*
Put the exact MFU/TPS formula and the causal-adjusted MaxText reference in the model page's Target metrics block *before* experiment 1, and LINT every "% of MaxText" against it — turning the 06-12 manual reconciliation into a mechanical check.

### E. Persist the audit trail against compaction *(fixes P5)*
Given no subagents, at minimum write the hypothesis rationale + profile-bucket attribution *into the experiment page* every iteration (structured, not just prose in the model turn), so 100+ compactions can't erase why a move was made.

## Bottom line

The Codex/GPT-5.5 run is the **stronger scientific result and the weaker process artifact** — the mirror image of the cc session. Its breadth (349 experiments vs 45) and persistence are the *direct cause* of beating MaxText: brute-force coverage of the XLA-flag space found `DISABLE_COLLECTIVE_MATMUL` (−9 GiB HBM, [v116](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-04-v116-shard-acts-s8k-bs3-tokamax-splash-layeragg-collective-matmul-none.md)) which reopened global batch 32 ([v118](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-04-v118-shard-acts-s8k-bs4-tokamax-splash-layeragg-collective-matmul-none.md)), then SparseCore collective offload ([v205](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v205-no-scan-sparsecore-all-collectives.md)→[v273](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v273-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-rerun.md)) overlapped the larger batch on-chip — exactly the wall cc declared out of scope. The rerun-before-carry, HLO-hash, NaN-honesty, and per-lever isolation discipline were all genuinely good. But the *loop bookkeeping* nearly collapsed: no filed hypothesis queue (3/349 `## Next hypotheses`), no lineage sections (2/349 `## Sources`), lane-wide wrong frontmatter, non-enum verdicts, colliding v-numbers, and ~241/349 sub-noise reruns with no CPU pre-filter. The lesson across the pair: **cc had the discipline and stopped short; cx had the persistence and beat the target but left the wiki barely lint-able.** Fixes A–C would let the next Codex run keep the breadth that won while producing a queue and a lineage the loop can actually build on.

## See also
- [qwen3-cx-jax model page](../models/qwen3-cx-jax.md) — the live variant matrix + climb narrative (v001–v279 + the 2k coda).
- [Qwen3-CC-JAX session retrospective](2026-07-08-qwen3-cc-jax-session-retrospective.md) — the Opus-4.8 counterpart that stalled at 87% of MaxText; the direct comparison.
- [SCHEMA.md](../../SCHEMA.md) · [wiki/experiments/program.md](../experiments/program.md) — the process being audited.
- MaxText reference: [bs3 no-vocab-tiling](../experiments/qwen3_cx_autoresearch_optimization/maxtext/experiments/2026-06-05-maxtext-qwen3-8b-v6e8-bs3-no-vocabtiling.md) (44.9% MFU / 6,883 tok/s/chip), the ~45% recipe cx exceeded.

## Sources
- Session transcript (not a `raw/` artifact): `~/.codex/sessions/2026/06/02/rollout-2026-06-02T23-41-17-019e8ab6-81c0-7d32-a52a-1c261499495a.jsonl` — OpenAI Codex CLI / GPT-5.5 (effort high), 2026-06-02 → 2026-06-13, 76,776 lines, 155 MB.
- All filed `qwen3_cx_autoresearch_optimization/jax/experiments/*.md` pages (v001–v279 seq8192 + the 06-12/13 2k series) and the `qwen3_cx_autoresearch_optimization/maxtext/experiments/*.md` reference pages.
