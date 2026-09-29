---
title: "Qwen3-CC5-JAX autonomous session — process retrospective (Fable 5, session 3e7df36d)"
type: analysis
tags: [retrospective, process, autoresearch-loop, qwen3, fable-5]
created: 2026-07-08
updated: 2026-07-08
---

Retrospective on the Claude Code (**Fable 5**) session `3e7df36d-a825-4b07-a589-a017131a0d1c` that ran the autoresearch loop on **`qwen3_cc5` / jax lane**, 2026-06-12 → 2026-06-13. This page judges the run against the SCHEMA + `program.md` process: what the process is, how faithfully the agent followed it, what went well, what went badly (with concrete examples), and a prioritized backlog of **fixable process gaps**. It is a *process* analysis; the science is summarized only where it illustrates a process point. It is also — deliberately — a **comparison** against the sibling [Opus-4.8 `qwen3_cc` retrospective](2026-07-08-qwen3-cc-jax-session-retrospective.md), because the two sessions attacked the same target and cc5 crossed a ceiling that cc declared out of scope.

The transcript (6,784 lines, 15 MB) lives at `~/.claude/projects/-mnt-disks-persist-torch-tpu-tpu-performance-autoresearch-wiki-cc5/3e7df36d-a825-4b07-a589-a017131a0d1c.jsonl`; it is not a citable `raw/` artifact, but every claim below is anchored to a filed experiment/analysis page that is.

## The process it was supposed to follow

Per [SCHEMA](../../SCHEMA.md) + the [program.md hierarchy](../experiments/program.md):

1. `/create-experiment` — bootstrap the model family (folders, model-level `program.md`, model page stubs).
2. `/start-experiment` — resolve `program.md` (root → model → lane), select hardware, discover clusters, occupancy check, then start `/loop` in never-stop mode.
3. **The loop**, per iteration: formulate a falsifiable hypothesis → (optional AOT/HLO CPU pre-flight) → run on GKE via `gke-cluster-runner` → capture profile → `profile-analyzer` attributes the bucket → record observations → update the model page's variant row → **file next-hypothesis stubs** → update `index.md` / `log.md`.
4. `/stop-experiment` — reap orphan workloads, file missing pages, run LINT, append a clean-shutdown marker.

Verdict discipline (SCHEMA rules 6–8): `supported` requires improvement beyond noise **and** a semantics check **and** no tracked-metric regression; a crash or a semantic change is `invalid` and its "speedup" is **not** reported.

## What actually happened — the shape of the run

Unlike the 19-calendar-day cc session, cc5 was **one tight ~29-hour burst**: `2026-06-12T01:18` → `2026-06-13T06:30`, 1,092 tool calls (1,024 on 06-12, 68 on 06-13), **2 context compactions** (vs cc's 5). The never-stop loop was driven by the Stop hook (`.claude/stop_hook.sh`, re-injecting "next iteration" / "never-stop" continuations — 30/24 occurrences across 220 hook lines) plus `ScheduleWakeup` used as a **safety-net poll while up to 3 GKE workloads ran in parallel** ("*up to 3 v6e8 clusters in parallel, enable stop hook*", 01:23). It fanned out to subagents heavily: **90 Agent dispatches — 72 `gke-cluster-runner`, 17 `profile-analyzer`, 1 general-purpose** — plus 374 Edit, 363 Bash, 105 Write, 91 ScheduleWakeup, 54 Read, and 6 Skill calls (`start-experiment`, `loop`, `formulate-hypothesis`, `edit-model-code`, ×2 `create-retrospective`).

Three phases, all driven by short human directives:

| Phase | Window (06-12/13) | Character |
|-------|-------------------|-----------|
| **Autonomous seq8192 climb** | 01:23 → 05:54 | `/start-experiment` → `/loop`, 3 parallel clusters; v001→~v023 (splash, batch, CE, remat, scan) unsupervised |
| **Human-steered "match MaxText"** | 05:54 → ~20:18 | directive "*find maxtext recipe… run it… then find a way to match its performance on jax lane*"; MaxText reference [mt-v001](../experiments/qwen3_cc5_autoresearch_optimization/maxtext/experiments/2026-06-12-mt-v001-base.md) (6,953 tok/s/chip) + gap-closing v024–v051 — **this phase produced the win** |
| **seq-2048 phase (charles-v6e)** | 06-12 22:06 → 06-13 06:30 | new cluster, user-directed; v052–v063 batch/remat arc; bs5 40.5% MFU certified |

**Output artifacts** (filed, on disk): **62 jax pages** (1 baseline + v001–v063). jax verdict distribution: **21 supported, 29 refuted, 4 invalid, 6 inconclusive, 1 baseline.**

**Result:** seq2048 → **7,871 tok/s/chip / 40.5% MFU** (bs5, [v058](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v058-2kbs5.md)/[v060](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v060-2kbs5val.md), 50-step certified; a statistical tie with MaxText's 7,915). seq8192 → **6,959 tok/s/chip / 39.9% MFU** ([v050](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v050-weight-io.md)/[v051](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v051-wioval.md), 50-step certified: median 6,958.9, +0.08% over MaxText 6,953) — **matching and exceeding the MaxText reference**, at peak HBM **22.03 GiB (70.5%)**.

## The decisive question: how cc5 cracked seq8192 where cc gave up

This is the point of the retrospective. The [Opus-4.8 `cc` session](2026-07-08-qwen3-cc-jax-session-retrospective.md) stalled at seq8192 = **6,068 tok/s/chip (87.4% of MaxText)**, concluded the residual gap was an out-of-scope **"host-offload *pipelining* wall"** (a MaxText runtime capability), and declared the attention/SiLU-recompute lever **"walled — can't save on-device 21.7G."** cc5 crossed the same line in a **three-step chain, all on exactly the on-device activation-save / remat-policy lever cc dismissed:**

1. **[v044](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v044-context-checkpoint.md)/[v045](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v045-ctxval.md) — splash context checkpointing** (`SPLASH_RESIDUAL_CKPT_NAME=context` + `--remat_policy=save_qkv_ctx`). The splash kernel emits out+lse tagged `"context"`; the remat policy **saves those residuals instead of re-running splash-forward in the backward pass** — the kernel-side analog of the exact "save the attention residual instead of recompute it" lever cc called walled. HLO confirmed splash bwd uses `splash_mha_dkv_no_residuals` with **no** `splash_mha_fwd_no_residuals` re-run. Result: +7.3% step (6,265 → 6,723 tok/s/chip), **and HBM dropped 27 → 22.03 GiB**; MaxText gap −9.9% → −3.3%.
2. **[v047](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v047-ctx-hbm.md)/[v048](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v048-ctxh-val.md) — `save_qkv_ctx_hbm`**: cc5 *measured* that the 4.57 GiB context residuals **fit in HBM outright** (projected ~26.6 GiB, real frontier peak 22.03 = 70.5%), dropped the host offload entirely, and recovered the +25 ms DMA residual → 6,780 tok/s/chip. **This directly refutes cc's premise on both counts:** cc treated the residual gap as an out-of-scope *host-offload pipelining* wall — but cc5 needed no host pipeline, because the save fit on-device; and cc's "can't save on-device 21.7G" HBM-budget estimate was simply **wrong** — cc5 shows ~9 GiB of headroom at the frontier.
3. **[v050](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v050-weight-io.md)/[v051](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v051-wioval.md) — (in,out) weight layout** (`JAX_WEIGHT_LAYOUT=io`): transpose-free `x@W` on every decoder matmul — the conv-fusion / MXU-layout lever the bucket-diff predicted — pushed **past** MaxText: 6,969 tok/s/chip 20-step (+0.24%), certified **6,958.9 / chip 50-step (+0.08%)**, 39.9% MFU.

**Why cc5 succeeded where cc gave up, in one line:** cc reasoned about the HBM budget and declared a wall; cc5 *ran the remat-policy experiment*, discovered the on-device save fit with ~9 GiB to spare, and then closed the final margin with a **config-scope weight layout** — not the runtime capability cc had ruled out of scope. The "wall" was an untested assumption; the win was one falsifiable experiment ([v044](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v044-context-checkpoint.md)) plus a memory measurement ([v047](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v047-ctx-hbm.md)) away.

## What went well

### 1. The discrete audit gates cc's retro asked for were actually used
cc's retrospective (its problem P6) flagged that `formulate-hypothesis` ran **0 times** and `profile-analyzer` only **4 times for 45 experiments**. cc5 fixed exactly this: **17 `profile-analyzer` dispatches** across 62 experiments, plus `formulate-hypothesis` and `edit-model-code` skills invoked. Profile attribution and hypothesis construction were discrete, re-auditable tool calls that survive compaction — not inline prose. And CPU pre-flight was actually run: [v044](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v044-context-checkpoint.md) ("CPU smoke: grads bit-identical across save_qkv / save_qkv_ctx"), [v050](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v050-weight-io.md) ("CPU smoke: loss bit-identical").

### 2. 50-step certification discipline — every frontier flip double-verified
Every frontier claim was re-run at 50 steps before adoption: [v019](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v019-val50.md), [v038](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v038-bs2-saveattn-val.md), [v043](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v043-bs2-svqkv-val.md), [v045](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v045-ctxval.md), [v048](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v048-ctxh-val.md), [v051](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v051-wioval.md), [v058](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v058-2kbs5.md)/[v060](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v060-2kbs5val.md). The headline "MaxText exceeded" number is a certified 48-measured-step median (6,958.9, σ=6.6, spread 0.41%), not a lucky 20-step read.

### 3. Near-perfect structural compliance
**62/62 carry `## Next hypotheses`, 62/62 carry `## Sources`, 58/62 carry `## Profile`** — and the three run pages without a Profile are all legitimate no-measurement cases ([v053](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v053-2kflags.md)/[v054](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v054-2kscanflags.md) compile-budget kills, [v059](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v059-2kbs6.md) bs6 OOM). This is a step-change over cc's **19/45 `## Sources`** — the mandatory lineage section held on every page here.

### 4. Honest verdict taxonomy under numerics pressure
Only **4 `invalid`** (vs cc's 11), and all four were real, correctly rejected, never laundered:
- [v006](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v006-tkmce-bf16.md) — tokamax `mosaic_tpu` bwd hardcodes an f32 scratch Ref; kernel-support gap, not a refutation.
- [v008](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v008-chunked-xla-ce.md) — chunked-CE bf16-x drifts loss by **−0.73 (≈ ln 2)**; parity fail. The corrective (cast x-only to f32) was derived and dispatched as [v011](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v011-cxla-f32x.md).
- [v012](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v012-remat.md) — remat-over-splash NaN from step 4 + 27% overhead; the OOM breakdown was retained as the decisive observation that redirected to [v014](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v014-bs3-ce.md).
- [v061](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v061-2kce8.md) — chunked-CE's +1.5% batch-unlock **not reported** per rule 8, because the same-data A/B ([v062](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v062-2kce5par.md)) showed a systematic +0.006 loss offset.

### 5. Parallel dispatch + confound control at higher throughput
The 3-cluster fan-out (72 `gke-cluster-runner` subagents) produced 62 experiments in ~29 hours — higher throughput than cc's 45-over-4-active-days, *without* losing rigor: the 20-step probe → 50-step cert pattern is itself a confound control, and the [v061→v062](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v062-2kce5par.md) same-data parity arm is textbook.

## What went badly — with concrete examples

### P1 — some no-step launches still burned TPU time
Fewer than cc, but present: [v006](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v006-tkmce-bf16.md) (kernel-support gap, discoverable from the installed wheel), [v053](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v053-2kflags.md)/[v054](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v054-2kscanflags.md) (compile-budget kills before step 0 at 2k), [v059](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v059-2kbs6.md) (bs6 OOM), [v012](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v012-remat.md) (NaN). A build-capability probe (does this tokamax impl / flag exist in the installed wheel?) and an AOT compile-budget/OOM screen on CPU would have pre-empted several — SCHEMA's RUN-EXPERIMENT step 1b exists for exactly this and was applied unevenly.

### P2 — a numerics bug shipped *into* the certified frontier before it was caught
The chunked-CE +0.006 loss offset was only surfaced by the 2k parity arm [v062](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v062-2kce5par.md) — **after** the same `chunked_xla` kernel was already baked into the certified **8k** frontier ([v044](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v044-context-checkpoint.md)+). The model page correctly escalates this ("⚠️ same kernel is in the certified 8k frontier — human review requested"), but the process ordering is backwards: a same-data parity A/B should **precede** frontier adoption of any custom-CE kernel, not follow it two phases later. This is a genuine correctness risk that a mid-loop numerics gate would have caught at v011, not v062.

### P3 — one mega-session did the loop, the retrospectives, the pivot, and the equivalence test
Better than cc (2 compactions vs 5), but still not split: the same conversation ran the never-stop seq8192 loop, two `create-retrospective` passes, a MaxText reference build, a user-directed seq-2048 pivot onto a *different* cluster (charles-v6e), and a local equivalence-test run ("*copy … test_tpu_equivalence.py … and run it locally*", 20:15). The two compactions both fell in this overloaded stretch. Per cc's backlog item C, the loop and the ops/pivot work should be separate sessions.

### P4 — single lane, mostly single variant, again
**62/63** experiments went to jax; the [torchax lane](../models/qwen3-cc5-torchax.md) never got past its baseline and MaxText was reference-only. Within jax, only two matrix rows were exercised (seq8192 and, when the user directed it, seq2048) — both on `8B/v6e-8`; no hardware scale-up row tested cross-variant transfer. The [Knobs translation matrix](../models/qwen3-cc5-jax.md) is rich *within* the row (and did prove per-shape knobs like io-layout: +2.8% at 8k, −0.18% at 2k via [v056](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v056-2kio.md)) but empty across hardware.

### P5 — no clean `/stop-experiment` boundary
The user manually managed the hook ("*enable stop hook*" 01:23, "*cleanup stop hook*" 20:18) rather than the loop closing itself with `/stop-experiment` (reap + LINT + shutdown marker). There is no clean-shutdown marker separating the 8k phase, the 2k pivot, and the tail — the phases blur into one open session, and LINT was not run mid-loop.

## Problems to address later — prioritized backlog

Ranked by (impact on loop integrity × frequency observed / effort to fix).

### A. Gate custom-kernel numerics with a same-data parity A/B *before* frontier adoption *(fixes P2)*
The single highest-value fix, because it is the one place cc5 shipped a real risk. Any experiment that swaps in a custom CE / attention / matmul kernel must run a same-data, same-seed A/B against the reference kernel and pass the parity bound **before** the config is eligible to become the frontier — not as a later-phase afterthought. On this run it would have caught the chunked-CE +0.006 offset at [v011](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v011-cxla-f32x.md)/v044, not [v062](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v062-2kce5par.md). Wire it into `program.md` as a hard gate on any `use_tokamax_ce` / custom-kernel experiment.

### B. Make the build-capability + AOT-OOM probe an un-skippable pre-dispatch gate *(fixes P1)*
Before any GKE dispatch: (i) assert the tokamax impl name / XLA flag exists in the **installed wheel**; (ii) `jax.jit(fn).lower(*args).compile()` on CPU to catch OOM / shape / compile-budget failures. Would have pre-empted [v006](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v006-tkmce-bf16.md), [v053](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v053-2kflags.md), [v054](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v054-2kscanflags.md), [v059](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v059-2kbs6.md). cc5 already runs CPU smoke tests for numerics ([v044](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v044-context-checkpoint.md)/[v050](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v050-weight-io.md)) — extend the same habit to compile/OOM screening and make it mandatory, not optional.

### C. Split "loop" sessions from "ops/pivot" sessions and close with `/stop-experiment` *(fixes P3, P5)*
A never-stop `/loop` runs only the loop. Retrospectives, MaxText reference builds, the 2k cluster pivot, and local equivalence tests → separate sessions, each closed with `/stop-experiment` (reap + LINT + clean-shutdown marker). This directly reduces the compaction rate (2 here, all in the overloaded stretch) and gives a clean phase boundary.

### D. Run LINT mid-loop, not only at shutdown *(surfaces P2/P4 early)*
A lightweight LINT every N experiments would host the numerics-parity check from (A), flag the missing-Profile no-run pages while context is fresh, and surface the single-variant over-concentration before 60 experiments accrete on one row.

### E. Budget across the matrix and schedule the second lane *(fixes P4)*
The `program.md` iteration ladder should schedule the torchax lane and at least one hardware scale-up row rather than exhausting `8B/v6e-8`. The io-layout per-shape result ([v050](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v050-weight-io.md) +2.8% at 8k vs [v056](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v056-2kio.md) −0.18% at 2k) proves knobs do *not* transfer for free — which is exactly the argument for testing the transfer deliberately.

## Bottom line

cc5/Fable 5 is the stronger of the two sessions on nearly every process axis its sibling flagged: it used the discrete `profile-analyzer` and `formulate-hypothesis` gates cc skipped, ran CPU pre-flight, certified every frontier flip at 50 steps, and filed `## Sources` + `## Next hypotheses` on 62/62 pages — all in a single tight 29-hour burst with 2 compactions instead of cc's 19 days and 5. And it produced the result cc could not: it **matched and exceeded the MaxText reference at seq8192 (6,959 vs 6,953 tok/s/chip, 39.9% MFU, 50-step certified)** where cc stalled at 6,068 (87%). The reason is the cleanest possible vindication of the loop discipline itself: cc *reasoned* that the on-device activation save was "walled at 21.7G" and the gap was an out-of-scope host-offload pipeline; cc5 *ran the experiment* ([v044](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v044-context-checkpoint.md)), measured the HBM ([v047](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v047-ctx-hbm.md): 22 GiB, ~9 GiB headroom), found the save fit on-device with no host pipeline needed, and closed the last margin with an in-scope weight layout ([v050](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v050-weight-io.md)). The remaining weaknesses are operational and one is a real risk — a custom-CE numerics bug reached the certified frontier before the parity arm caught it — but fixes A–E are cheap, mechanical, and would remove the risk without touching what already works well.

## See also
- [qwen3-cc5-jax model page](../models/qwen3-cc5-jax.md) — the live variant matrix + the seq8192/seq2048 frontier this session produced.
- [2026-07-08 qwen3-cc-jax session retrospective](2026-07-08-qwen3-cc-jax-session-retrospective.md) — the Opus-4.8 sibling that stalled at 6,068; the counterfactual this page is measured against.
- [2026-06-12 MaxText feature-gap inventory](2026-06-12-maxtext-feature-gap-inventory.md) — the inventory whose items #1 (context checkpoint) and the layout item this session executed to close the gap.
- [2026-06-12 MaxText vs jax bucket diff](2026-06-12-maxtext-vs-jax-bucket-diff.md) — the op-level gap decomposition that predicted the (in,out)-layout lever.
- [2026-06-13 qwen3 2k stack vs MaxText](2026-06-13-qwen3-2k-stack-vs-maxtext.md) — the seq-2048 phase close (statistical tie).
- [SCHEMA.md](../../SCHEMA.md) · [wiki/experiments/program.md](../experiments/program.md) — the process being audited.

## Sources
- Session transcript (not a `raw/` artifact): `~/.claude/projects/-mnt-disks-persist-torch-tpu-tpu-performance-autoresearch-wiki-cc5/3e7df36d-a825-4b07-a589-a017131a0d1c.jsonl` — Claude Code / Fable 5, 2026-06-12 → 2026-06-13, 6,784 lines.
- All filed `qwen3_cc5_autoresearch_optimization/jax/experiments/*.md` pages (baseline + v001–v063), the MaxText reference [mt-v001](../experiments/qwen3_cc5_autoresearch_optimization/maxtext/experiments/2026-06-12-mt-v001-base.md), and the [qwen3-cc5-jax model page](../models/qwen3-cc5-jax.md).
