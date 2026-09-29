---
title: "Qwen3-8B four-lane cross-model comparison — why Fable-5 and Codex crossed MaxText where Opus-4.8 and Antigravity stalled"
type: analysis
tags: [retrospective, cross-model, qwen3, maxtext, claude-opus-4-8, claude-fable-5, codex, antigravity, process]
created: 2026-07-08
updated: 2026-07-08
---

The same optimization task — take Qwen3-8B on v6e-8 to (and past) the MaxText reference — was run autonomously by four different (model × harness) lanes. Two crossed MaxText; two stalled ~13–23% short. This page synthesizes the four session retrospectives to answer the question the human asked: **why did Fable-5 (`cc5`) and Codex (`cx`) beat MaxText while Opus-4.8 (`cc`) got stuck and Antigravity/Gemini (`ag`) also stalled?** The short answer is not "the winning models are smarter." It is a **memory-budget-skepticism + search-discipline** story, with a **metric-accounting artifact** layered on top.

Per-lane process detail lives in the four sibling retrospectives:
[cc / Opus 4.8](2026-07-08-qwen3-cc-jax-session-retrospective.md) ·
[cc5 / Fable 5](2026-07-08-qwen3-cc5-jax-session-retrospective.md) ·
[cx / Codex](2026-07-08-qwen3-cx-jax-session-retrospective.md) ·
[ag / Antigravity](2026-07-08-qwen3-ag-jax-session-retrospective.md).

## The scoreboard (accounting-free: tok/s/chip)

Because the lanes reported MFU under **different FLOP conventions** (see §"The metric artifact"), the only honest cross-lane number is **tokens/sec/chip** on identical hardware (v6e-8) and shape.

### seq8192 (the program target)

| Lane | Model × harness | Best tok/s/chip | Config route | vs MaxText 6,953 |
|------|-----------------|-----------------|--------------|------------------|
| **cx** | Codex / GPT-5.5 · Codex CLI | **7,543** (global batch 32) | `DISABLE_COLLECTIVE_MATMUL` → batch 32 + on-chip SparseCore offload ([v273](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v273-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-rerun.md)) | **+8.5% — beat** |
| **cc5** | Fable 5 · Claude Code | **6,959** (bs2) | scan + splash context-checkpoint + measured in-HBM residuals (no offload) + (in,out) layout ([v050/v051](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v051-wioval.md)) | **+0.08% — tie/beat** |
| — | **MaxText reference** | 6,953 (bs3) | their integrated recipe | — |
| **cc** | Opus 4.8 · Claude Code | 6,068 (bs3) | scan + maxtext-CE + norm-stat save ([v045](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-05-v045-save-norm-stats.md)) | **−12.7% — stalled** |
| **ag** | Antigravity / Gemini | 5,329 (baseline) | reached the right levers, none compiled clean ([v047 seq8k baseline](../experiments/qwen3_ag_autoresearch_optimization/jax/experiments/2026-06-13-qwen3-jax-v047-seq8k-baseline.md) tier) | **−23% — stalled** |

### seq2048

| Lane | Best tok/s/chip | vs MaxText 7,915 |
|------|-----------------|------------------|
| **cc5** | **7,871** (bs5, 50-step certified) | −0.56% — statistical tie |
| **cc** | 6,964 (bs4) | −12% |
| **ag** | 4,515 (bs?, v042 gate-up fusion) | ~−43% raw (though ag reports ~90% by its MFU) |

## The one thing the winners did and the losers didn't: they changed the memory budget

Every lane hit the same physical fact — at seq8192 the frontier is **HBM-bound**, and batch/activation-save levers are gated by how much of the 31.25 GiB fits. The four lanes split on **what they did about that budget**:

- **cx (won) — freed ~9 GiB with a flag, then grew batch.** `xla_tpu_{all_gather,reduce_scatter}_collective_matmul_mode=none` ([v116/v117](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v273-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-rerun.md)) cut peak HBM **28.32 → 22.21 GiB**, which reopened per-chip batch 4 / global batch 32 (v118 fits at 26.91 GiB). Batch then amortized the per-token overhead → 7,543 tok/s/chip.
- **cc5 (won) — *measured* that ~9 GiB was already free, then saved activations on-device.** Splash context-checkpointing ([v044/v045](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v051-wioval.md)) skips the backward splash-forward re-run (+7.3%, HBM 27→22 GiB); then [v047/v048](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v051-wioval.md) **measured** the residuals at only 4.57 GiB and dropped host offload entirely — ~9 GiB of headroom that was there all along.
- **cc (stalled) — *assumed* the budget was full and quit.** It declared the SiLU double-recompute *"walled — can't save on-device 21.7 G"* and the residual an *"out-of-scope host-offload pipelining wall"* ([v045 log](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-05-v045-save-norm-stats.md), [gap-closing retrospective](2026-06-05-qwen3-cc-jax-gap-closing-attempt.md)). It never measured the actual residual footprint. Its own bs2/bs4 OOMs ([v001/v002](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v001-batch-bs2.md)) led it to close the batch axis — the axis cx reopened with one flag.
- **ag (stalled) — found both levers, but every run crashed.** ag mined the *identical* context-checkpointing vein (v053/v055/v057-*) and the SparseCore-offload vein, but not one compiled to a clean measured result (VMEM/HBM/shard_map crashes; offload died on an unsupported XLA flag v002 and a silent no-op v033). It reached the right ideas and never got a number.

**The most striking convergence:** the two winners independently discovered the *same ~9 GiB of HBM headroom* by two *different* mechanisms — a compiler flag (cx) and a remat measurement (cc5) — and both spent it **on-chip**. That is the exact opposite of the direction cc searched: cc was hunting for a **host-DMA offload pipeline** (and correctly found host offload doesn't pipeline on this libtpu — [v039](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v039-maxtext-offload-recipe-s8k-bs3.md)), then generalized "host offload is the only remaining lever, and it's out of scope." The real lever was to **avoid host DMA entirely** and keep the freed memory on-chip. cc had the residual diagnosis right and the conclusion exactly backwards.

## Two ways to win, two ways to lose

The win/loss did **not** correlate with process hygiene. It correlated with whether the lane *tested the memory budget* instead of *reasoning about it*.

| Lane | Outcome | How | Process hygiene | One-line character |
|------|---------|-----|-----------------|--------------------|
| **cc5** | Beat | measure-and-target | **Best** (17 profile-analyzer, formulate-hypothesis, 50-step cert, 62/62 Sources, 2 compactions) | Disciplined *and* skeptical of the wall |
| **cx** | Beat | brute-breadth | **Worst** (Next-hyps 3/349, Sources 2/349, 116 compactions, all frontmatter mis-slugged) | Persistence substituted for discipline — 349 experiments brute-forced the two flags cc never reached |
| **cc** | Stalled | reason-a-wall-and-quit | Good (44/45 Profile, 45/45 Next-hyps) | Clean process, **wrong conclusion** — premature exhaustion |
| **ag** | Stalled | find-the-lever-but-crash-loop | Collapsing (Profile 33/90, Next-hyps 4/90, Sources 2/90) | Right ideas, **execution + harness-persistence failure** |

- **cc5 is the model to emulate:** it won *and* left the cleanest wiki. Discrete audit gates (profile-analyzer, formulate-hypothesis) survive context compaction; its lineage held (2 compactions vs cc's 5, cx's 116).
- **cx proves breadth can substitute for depth** — but at a cost: 241/349 experiments were sub-noise reruns, its lineage was destroyed by 116 compactions, and every page is mis-slugged `model: qwen3-cc-jax`. It won the number and lost the knowledge base.
- **cc is the cautionary tale for *smart* agents:** good methodology (confound controls, contradiction blocks, honest verdicts) gave it false confidence in a wall it never measured. Discipline without empiricism about the budget.
- **ag is the cautionary tale for *harness*:** with no hard Stop hook (self-enforced never-stop via Antigravity's `schedule`), the loop degenerated on 06-15 into ~30 un-pre-flighted compile-crash retries, the queue-refill discipline collapsed, and the model page went stale (40/90 pages still slugged `qwen3-cc-jax`, broken frontier links). It found the levers cc5 won with and could not land one.

## The metric artifact (why cc's gap *looked* worse than it was)

Layered on top of the real throughput gap is a measurement confusion that made cc look further behind than it was. Per the [MFU-normalization analysis](2026-06-12-qwen3-maxtext-rerun-mfu-normalization.md): the native JAX trainers (cc, cc5) computed MFU with **causal (÷2) attention FLOPs**, while MaxText's MFU counts the **full non-causal** QK/AV term (~15% more FLOPs/token). So cc's headline *"34.8% MFU = 87% of MaxText's ~39.8%"* compared two different denominators. Under a common convention, cx's frontier is **~49.2% MFU vs MaxText's ~45.4%** — a genuine win that the causal MFU had hidden. The lesson: **the gap was real in tok/s/chip but exaggerated in MFU**, and nothing in cc's loop caught the apples-to-oranges denominator until the human forced the reconciliation two weeks later — the acute form of the "unpinned metric contract" problem (cc-retrospective fix B).

## What to fix — the unified backlog

The four lanes' failures collapse to a small set of fixes. The single highest-value one resolves **both** stall modes at once.

### 1. A hard CPU/AOT pre-flight gate before every GKE dispatch — fixes cc *and* ag
This is the master fix. `jax.jit(fn).lower(*args).compile()` + `.cost_analysis()` + a `.memory_analysis()`/peak-HBM read on CPU, gating every dispatch, would have:
- **handed cc the residual-footprint number** (4.57 GiB, ~9 GiB free) that directly refutes its "21.7 G wall" — the wall was never measured, and a mandatory measurement makes "reason a wall and quit" impossible;
- **caught ag's ~30 compile crashes** (VMEM/HBM/shard_map/unsupported-flag) on CPU before they burned the loop and collapsed its queue-refill discipline.
Both stalls were the *absence of a cheap measurement*. Make it un-skippable in `program.md` and enforce it in the never-stop hook.

### 2. Pin the metric contract — causal-vs-non-causal, before the climb
Put the exact MFU/TPS formula **and the FLOP convention (causal? non-causal?)** and the causal-adjusted MaxText reference in the model page *before* experiment 1, with a LINT check recomputing every "% of MaxText" from the pinned formula. Removes the two-week-late reconciliation that hit cc, cx, and ag alike.

### 3. Make discrete audit gates mandatory — they survive compaction; prose doesn't
cc5 (2 compactions) kept its lineage because hypotheses/profiles went through `formulate-hypothesis` and `profile-analyzer` **tool calls**; cx (116 compactions) and ag lost theirs because the reasoning lived in inline prose that compaction erased (Next-hyps 3/349 and 4/90). Route every hypothesis and profile through the skill/agent, not the chat.

### 4. A consecutive-tie stop-signal + variant rotation
cx ran 241/349 sub-noise reruns; cc's v038–v045 tail was almost all parity; ag crash-looped one lever. When K consecutive experiments on a row are within noise (or K consecutive crashes), the loop should **rotate the variant/lane or escalate to a retrospective**, not keep probing the same cell. All four lanes over-concentrated on `8B/v6e-8`.

### 5. Enforce frontmatter/model-slug integrity in LINT
cx (all 349) and ag (40/90) filed experiments under `model: qwen3-cc-jax` regardless of lane, with non-enum `status:` strings and colliding v-numbers. A LINT gate on `model:` matching the folder's lane, and on the `verdict:` enum, would have kept the merged wiki navigable.

### 6. Split loop from ops; end phases with `/stop-experiment`
Three of four lanes (cc, cc5, ag) ran the loop and the meta-work (merges, viz, equivalence tests, 2k pivot) in one open-ended session and never cleanly closed the autonomous phase. This drives the compaction counts and buries the research. One phase, one session, one clean shutdown.

## Bottom line

Fable-5 and Codex did not beat MaxText because they are more capable models in the abstract — they beat it because they **treated the HBM budget as a measurable quantity to be tested, not a wall to be reasoned about**, and (Codex) because sheer 349-experiment persistence stumbled onto the two flags that freed the memory. Opus-4.8 had the sharpest methodology of the four and still lost, because it reasoned itself into an unmeasured "out-of-scope" wall — the same residual it had diagnosed correctly but concluded about backwards (the fix was on-chip, not host offload). Antigravity found the right levers and could not land one, undone by a harness whose self-enforced never-stop loop collapsed into a crash-retry spiral. The decisive differentiators are **empiricism about the memory budget**, **persistence/breadth**, and **harness robustness** — in that order — not raw model horsepower. And a single fix, a mandatory CPU/AOT pre-flight that *measures the budget*, would have flipped both stalled lanes.

## See also
- [cc / Opus 4.8 session retrospective](2026-07-08-qwen3-cc-jax-session-retrospective.md)
- [cc5 / Fable 5 session retrospective](2026-07-08-qwen3-cc5-jax-session-retrospective.md)
- [cx / Codex session retrospective](2026-07-08-qwen3-cx-jax-session-retrospective.md)
- [ag / Antigravity session retrospective](2026-07-08-qwen3-ag-jax-session-retrospective.md)
- [MFU normalization — the causal-vs-non-causal FLOP convention](2026-06-12-qwen3-maxtext-rerun-mfu-normalization.md)
- [Qwen3-8B @ 2k: jax lane vs MaxText — best-vs-best](2026-06-13-qwen3-2k-stack-vs-maxtext.md)
- [cc5 retrospective #3 — the save_qkv/context-checkpoint frontier](2026-06-12-qwen3_cc5-jax-retrospective-3.md)
- Model pages: [qwen3-cc-jax](../models/qwen3-cc-jax.md) · [qwen3-cc5-jax](../models/qwen3-cc5-jax.md) · [qwen3-cx-jax](../models/qwen3-cx-jax.md) · [qwen3-ag-jax](../models/qwen3-ag-jax.md)

## Sources
- The four session transcripts (not `raw/` artifacts): `cc` = `~/.claude/projects/-mnt-disks-persist-torch-tpu-tpu-performance-autoresearch-wiki/5d03ff24-…jsonl` (Opus 4.8); `cc5` = `…-wiki-cc5/3e7df36d-…jsonl` (Fable 5); `cx` = `~/.codex/sessions/2026/06/02/rollout-2026-06-02T23-41-17-019e8ab6-…jsonl` (Codex/GPT-5.5); `ag` = `~/.gemini/antigravity-cli/brain/…` (Antigravity/Gemini).
- All filed `qwen3_{cc,cc5,cx,ag}_autoresearch_optimization/**` experiment pages and lane model pages.
