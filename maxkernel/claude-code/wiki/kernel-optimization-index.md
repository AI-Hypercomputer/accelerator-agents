---
title: "Kernel-Optimization Index — the knowledge MAP + load mandate for optimizing any Pallas/Mosaic kernel"
type: index
tags: [kernel-optimization, pallas, llo, hypothesis-generation, router, load-mandate]
created: 2026-07-09
updated: 2026-07-22
---

> **Purpose.** This file is the **router**, not the knowledge. It (a) names every kernel-knowledge source in this wiki and what each is for, and (b) **mandates — non-negotiable — exactly which sources get pasted into an author brief for a given kernel category** (see [Load mandate](#load-mandate-non-negotiable)). Its own body is a *thin, transferable decision procedure* (which class, which order, which category) — a regeneratable synthesis, not the deep knowledge, which lives in the pages it points at.
>
> **Value.** It guarantees an author never writes from the model prior. The deep knowledge (Pallas mechanics, mechanism classes, the earned traps) travels *in the brief*, pulled by category — the fix for the 2026-07-11 path-less-brief failure (2/8 vs 8/8). `/formulate-hypothesis` reads it to generate a falsifiable hypothesis; the master reads its [Load mandate](#load-mandate-non-negotiable) to assemble the pasted pack.
>
> **Regeneratable.** Everything here is *derived* from the wiki's source pages — throw it away and rebuild it via [`kernel-optimization-index-regenerate-prompt.md`](kernel-optimization-index-regenerate-prompt.md) whenever sources change. Nothing original lives here. (Contrast [`kernel_experiments/BRIEFS.md`](kernel_experiments/BRIEFS.md) — the *earned rules*, which are NOT regeneratable and accrete one hard-won lesson at a time. This index routes to BRIEFS; it does not restate it.)

The kernel-family analog of [`model-optimization-index.md`](model-optimization-index.md), read by `/formulate-hypothesis` for a kernel family. The body below is the transferable decision procedure — the mental model + the ordering for a *new* kernel. It carries **no deep knowledge of its own**: *how to measure* the signals → [`profile-analyzer-index.md`](profile-analyzer-index.md); *how to write a TPU Pallas kernel without fighting the compiler* (memory model, the (8,128) rule, scalar prefetch, the compile gotchas) → [`concepts/pallas-kernel.md`](concepts/pallas-kernel.md); *the earned parity/measurement/portability rules* → [`kernel_experiments/BRIEFS.md`](kernel_experiments/BRIEFS.md). The [Load mandate](#load-mandate-non-negotiable) says which of these to paste for which kernel.


## Read order for a new run

[design-class catalog](kernel_experiments/design-class-catalog.md) (mechanism classes + applicability tests) → [BRIEFS](kernel_experiments/BRIEFS.md) (the earned rules: measurement, K1 diagnosis, author discipline, parity/oracles, evidence, platform gotchas, lever catalog) → [kernel-root program.md](kernel_experiments/program.md) (the K0–K9 loop) → this index for anything not covered. Per-family worked examples exist only on archival branches — OPTIMIZATION runs may consult them under the reuse policy (mechanisms transfer; tiles/flags/ratios do not); cold `capability-eval` runs MUST NOT open them (the auditor's contamination class).

## Cardinal rule

Only material that helps generate a falsifiable kernel hypothesis for an arbitrary kernel: a principle, the profile signal that triggers it, its expected direction of effect, or a refuted pattern that saves you from writing a doomed candidate.

---

## The core thesis

**A Pallas kernel earns its keep by changing the *memory traffic*, *materialization*, or *work-grouping* that the compiler cannot avoid on its own — not by doing the same arithmetic "faster."** XLA already lowers dense arithmetic near-optimally. So the question is never "can I write a faster matmul?" — it is "**what is the naive form forced to move through HBM (or serialize) that a kernel could avoid?**" If the answer is "nothing" — the compiler is already fusion-complete and near its roofline — then no kernel can win, and the honest verdict is *refute*.

Corollary: **profile the bound before you hypothesize.** A kernel targeting the wrong bound is wasted effort even if it compiles and is correct.

## Generation scope (what transfers, what to re-derive)

The reasoning in this page is **hardware-invariant**: the core thesis, the three sinks, the intervention-class decision, the ordering, the dispatch floor, and the parity/measurement discipline apply on any TPU generation (v5p, v6e, v7/Ironwood, …) and mostly transfer to GPU/Triton too. What is **generation-specific and must be re-derived per target**: the MXU tile edge (128 on ≤v5p, 256 on v6e+ — sets block alignment and the small-block crossover), the **number of MXUs per TensorCore** (multi-MXU generations want tiles sized to feed all of them), the VMEM budget and its OOM wall and any scoped-VMEM flag knee, the VMEM:HBM bandwidth ratio (sets how much streaming lowers the required intensity), the functional-unit issue-slot counts (the utilization denominators), the HBM bandwidth (the memory-bound roofline), the available profiling signals (v7/Ironwood adds runtime perf-counter sampling that v6e lacks — v6e is static-LLO only), and **every measured block size and speedup below — all v6e**. Prefer *querying* these over hardcoding: `pltpu.get_tpu_info_for_chip` returns the per-generation constants programmatically (see [pallas-tpu-hardware](codebases/jax/doc-concepts/pallas-tpu-hardware.md)). Treat the numbers as calibration for the *shape* of the answer, not as targets; re-measure on the target hardware.

## Load mandate (non-negotiable)

**This is the index's primary job.** The master assembles the author brief by *pasting content* — never paths (the authoring model is assumed to have little or no Pallas/TPU in its training set), and **SECTIONS, not whole files** (the router/meta/evidence in this index is master-side and must NOT reach the author — pasting whole files is what balloons + dilutes the brief). What gets pasted is not discretionary; it is mandated by the kernel's category.

**Always paste (every kernel-family author brief), no exceptions:**
1. **This index's decision procedure — these sections only:** *The core thesis · Generation scope · The three sinks · THE WORKFLOW · Pick the intervention class · Kernel categories → strategy · Tiling theory · The dispatch/orchestration floor · Reading the profile: bound → lever* (+ the one-line Parity-&-measurement → BRIEFS pointer). **Do NOT paste** the header, this Load-mandate section, the **Hardware envelope** (a utilization-*reading* table — analysis/verifier side; the functional-unit mechanics the author needs are already in `pallas-kernel.md`, so pasting it would duplicate), Operational gotchas, the Kernel catalog, or Sources.
2. [`concepts/pallas-kernel.md`](concepts/pallas-kernel.md) **through `## Debugging`** — the authoring mechanics (annotated kernel skeleton, memory spaces, the (8,128) rule, scalar prefetch, functional units, the compile-error gotcha table). NOT its `## See also` / `## Sources` / `connect:auto` repo-link tail.
3. [`kernel_experiments/BRIEFS.md`](kernel_experiments/BRIEFS.md) — the earned rules, whole (the canonical `bench()` timing helper is its §1). §1 Measurement + §3 Author discipline also go into every *verifier* brief.
4. The kernel's **class page** from the routing table (`wiki/kernels/classes/<category>.md`) + its **category doc distillation(s)** from the map below.

**The category doc distillations** are ~100-line pages under [`codebases/jax/doc-concepts/`](codebases/jax/doc-concepts/), distilled from the pinned checkout's `raw/code/jax/docs/pallas/`; open the raw doc only when a distillation's pointer says the full prose/notebook is needed.

The **canonical category slugs** (see the routing table in [Kernel categories](#kernel-categories--the-routing-table-which-class-page-an-author-loads)): `attention` · `gemm-conv-epilogue` · `streaming-reduction` · `grouped-ragged-indirection` · `state-carry-scan` · `dense-near-roofline` · `cross-chip-collective`. For an author with no Pallas background, also paste the quickstart + grid/BlockSpec docs.

**Paste when retrieval is allowed** — 1–2 exemplar kernel pages from the [catalog](#kernel-catalog--reuse-before-you-write-unless-authorship-is-the-point) matching the category (a splash/GMM silo page) as *structure* examples, not answers.

**Never paste wholesale:** the raw docs tree, silo catalogs, or past experiment pages — link them; the transferable lesson is already distilled into the pages above. Per-kernel *answers* (`verified_briefs/`, `retrospectives-digest.md`) are **warm-tier only** — paste them only in demo/beat-MK mode for that exact kernel, never in a cold/holdout run (they would leak the answer).

| Doc distillation | Teaches | Paste for these categories |
|---|---|---|
| [pallas-tpu-details](codebases/jax/doc-concepts/pallas-tpu-details.md) | TPU lowering rules: tiling/layout constraints, dimension semantics, unsupported ops | **always**, before authoring |
| [pallas-quickstart](codebases/jax/doc-concepts/pallas-quickstart.md) + [pallas-grid-blockspec](codebases/jax/doc-concepts/pallas-grid-blockspec.md) | `pallas_call`/Ref semantics; the grid + BlockSpec `index_map` contract | model's first Pallas contact; any nontrivial `index_map` |
| [pallas-tpu-pipelining](codebases/jax/doc-concepts/pallas-tpu-pipelining.md) (+ [pallas-pipelining](codebases/jax/doc-concepts/pallas-pipelining.md)) | memory spaces, multiple buffering, `emit_pipeline`, dynamic block shapes | streaming-reduction kernels; DMA-starved profiles |
| [pallas-tpu-matmul](codebases/jax/doc-concepts/pallas-tpu-matmul.md) | canonical block matmul + roofline analysis walkthrough | dense-compute kernels; accumulator patterns |
| [pallas-tpu-sparse](codebases/jax/doc-concepts/pallas-tpu-sparse.md) (+ [pallas-tpu-core-map](codebases/jax/doc-concepts/pallas-tpu-core-map.md)) | scalar prefetch, block-sparse, data-dependent block indexing | grouped / ragged / paged / indirection kernels |
| [pallas-tpu-hardware](codebases/jax/doc-concepts/pallas-tpu-hardware.md) | TPU architecture background (MXU/VPU/VMEM, chip topology) | the model has no TPU background |
| [pallas-tpu-distributed](codebases/jax/doc-concepts/pallas-tpu-distributed.md) (+ [pallas-async-note](codebases/jax/doc-concepts/pallas-async-note.md)) | RDMA, DMA semaphores, collective kernel patterns | cross-chip kernels (ring attention, in-kernel collectives) |
| [pallas-design](codebases/jax/doc-concepts/pallas-design.md) | why the API is shaped this way (Ref/grid/BlockSpec rationale) | rarely — persistent API confusion |

## The three sinks of lost time (and their canonical fixes)

Almost every kernel win removes one of three things. Classify which one the naive form suffers, and the fix follows:

1. **Materialized intermediates** — the naive computes a large temporary and round-trips it through HBM (a score matrix, a normalization buffer, a logits tensor). **Fix: keep it in VMEM via a streaming/online algorithm** — compute-and-consume in tiles so the intermediate never lands in HBM (online-softmax, single-pass reductions, chunked accumulation). This is the highest-yield kernel move and usually the memory-bound case. *Why it's such a large lever*: VMEM bandwidth exceeds HBM bandwidth by an order of magnitude+ (~22× on v5e — generation-specific ratio, re-derive per target), so an op that stays in VMEM needs ~that-factor less arithmetic intensity to be compute-bound — even a low-intensity elementwise epilogue streams at near-peak once it never leaves VMEM ([scaling book ch.2](sources/2025-scaling-book-ch2-tpus.md)).
2. **Redundant reads** — the naive materializes a broadcast/repeat/concat before the real compute, reading the same bytes many times (a group-expansion feeding matmuls, a padded concatenation). **Fix is usually *not* a kernel** — an *algebraic restructuring* that folds the expansion into the contraction lets XLA emit minimal traffic, bit-exact and cheap. A kernel only helps here if it also changes the traffic pattern.
3. **Dispatch / orchestration overhead** — the work is fragmented into many tiny ops or DMAs, or the per-call work sits below the kernel dispatch floor. **Fix: coarsen** — block/group so each program does substantial work with large DMAs; if the op is fundamentally fine-grained (per-element gather, single-token decode), a hand kernel will *lose* to XLA's fused path — see [the dispatch floor](#the-dispatch--orchestration-floor).

---

## THE WORKFLOW (and why the order is the order)

> This is the CONCEPTUAL view of the same procedure `kernel_experiments/program.md` operationalizes as K0–K9 — one procedure, two zoom levels, edited together: step 1 ↔ K1 · step 2 ↔ K2 (+ the `/author-kernel` escalation ladder) · step 3 ↔ K5 · steps 4–5 ↔ K4/BRIEFS §Sweeps · step 6 ↔ K7 + the Stop rule. This section carries the WHY; program.md carries the files/commits/gates. If you change the order in one place, change it in both.

Order matters because each step assumes the previous one is settled; doing them out of order measures noise.

1. **Measure the naive + find the bound.** Benchmark (jit, warmup, compare **p50**), and **dump post-optimization HLO** (`XLA_FLAGS=--xla_dump_to`). Read it to answer: what does it *materialize*? is it compute-bound (near roofline), memory-bound (materialization/redundant-read), or overhead-bound (many tiny ops)? Is it **already fusion-complete**? Estimate the roofline ceiling so you know how much is even on the table — and compute the arithmetic intensity for the *exact shape/dtype combination*, not the op class: mixed precision moves the ridge (weights narrower than compute ≈ halves the compute-bound batch threshold; uniformly-narrow leaves it roughly unchanged), and some structures are *intrinsically* memory/comm-bound at any batch (e.g. a per-example-weights batched matmul, where the "weight" tensor scales with batch — no batching, and no kernel, fixes that; refute) ([scaling book ch.1](sources/2025-scaling-book-ch1-roofline.md)).
2. **Try the cheapest, most reversible intervention first.** In increasing cost/risk: **(a) refute** — if HLO shows XLA is already fusion-complete/near-roofline, stop now; **(b) flag** — if XLA's kernel is right but resource-limited, a compiler knob may buy the gap for free (bit-exact); **(c) xla-rewrite** — an algebraic change removing a redundant read, bit-exact, no kernel; **(d) author a Pallas kernel** — the expensive, risky option, justified only when a materialization or grouping needs a hand kernel (load [`concepts/pallas-kernel.md`](concepts/pallas-kernel.md) first — it prevents most of the compile-fix churn). Authoring last, not first, is what makes the loop efficient.
3. **Get correctness before speed.** Write the parity gate against a *high-precision* recompute first; verify the kernel on a *tiny* shape (correct indexing/masking/grouping) before scaling. A fast wrong kernel is worthless, and ragged/indirection bugs hide until you check a boundary case.
4. **Structural change before tiling before flags.** Fix the algorithm (streaming, grouping, fusion scope) first; only then sweep block sizes; only then chase flags. A block sweep on a kernel with the wrong structure optimizes the wrong thing.
5. **Sweep tiling to the wall** (see [tiling theory](#tiling-theory)).
6. **If nothing beats the naive at parity, REFUTE — with the bound and the specific blocker.** This is a first-class outcome, not a failure; it is more valuable than a forced marginal kernel, and it is what distinguishes directed optimization from black-box search.

## Pick the intervention class (the load-bearing decision)

| Class | Use when | Cost | Parity |
|---|---|---|---|
| **refute** | HLO shows XLA already fusion-complete / at minimal traffic / near roofline | free | — |
| **flag** | XLA's kernel is structurally right but resource-limited (e.g. VMEM budget) | trivial | bit-exact |
| **xla-rewrite** | the waste is a redundant read (repeat/broadcast/concat) removable by restructuring the algebra | cheap | bit-exact |
| **kernel-authored** | a materialized intermediate must be kept in VMEM, or ragged/grouped work must be regridded | expensive/risky | tolerance-gated |

The discriminating skill is **not forcing a kernel where a flag, a rewrite, or a refute is correct** — and, having written a kernel, **refuting your own kernel when it loses** rather than shipping a correct-but-slower result.

## Kernel categories → the routing table (which class page an author loads)

Classify by **K1 bound-and-structure, not the op's name**; the class page is a REQUIRED load (recorded in the K3 stub's `category:` + `class page:` fields, auditor-checked). One primary page; at most one secondary for straddlers.

| Category | K1 signature | Required page |
|---|---|---|
| **attention** | S²-scaling score materialization (or its fused absence), softmax + mask structure, shared/paged KV | [classes/attention.md](kernels/classes/attention.md) |
| **gemm-conv-epilogue** | matmul/conv producer + pointwise/reduction epilogue chain (the dominant suite shape) | [classes/gemm-conv-epilogue.md](kernels/classes/gemm-conv-epilogue.md) |
| **streaming-reduction** | big reduction over a materialized axis, no dominating matmul producer (norms, CE, standalone softmax) | [classes/streaming-reduction.md](kernels/classes/streaming-reduction.md) |
| **grouped-ragged-indirection** | per-group serialization, offset/length metadata, gathers, page tables | [classes/grouped-ragged-indirection.md](kernels/classes/grouped-ragged-indirection.md) |
| **state-carry-scan** | linear recurrence / chunked scan with carried state (SSM, retention, linear attention) | [classes/state-carry-scan.md](kernels/classes/state-carry-scan.md) |
| **dense-near-roofline** | one large dense contraction already near compute roofline, no epilogue | [classes/dense-near-roofline.md](kernels/classes/dense-near-roofline.md) |
| **cross-chip-collective** | op spans chips (ring attention, in-kernel collectives) | [classes/cross-chip-collective.md](kernels/classes/cross-chip-collective.md) |

*already-optimal* is not a category but a terminal OUTCOME any category can reach (HLO shows fusion-complete/near-roofline): refute cheaply with the HLO as evidence.

---

## Tiling theory

Block sizes are the **last** structural decision and the primary tuning knob, but they are an *interior optimum found by sweeping* — not a value to guess or copy from another op-point. A block size trades five competing pressures:

- **VMEM residency** — bigger blocks keep more operand resident (fewer reloads) but risk `CompileTimeScopedVmemOom`; the OOM boundary is a hard wall you should map, not avoid blindly.
- **Pipelining headroom** — the block must leave room for ≥2 buffers to overlap DMA with compute; too big and there is no room to double-buffer.
- **MXU-tile alignment** — contracting/output dims should be multiples of the MXU tile (128 on ≤v5p, 256 on v6e+); misalignment wastes MXU lanes.
- **Grid-program count** — smaller blocks mean more programs, and each program pays scalar setup + dispatch; overhead can swamp the compute.
- **DMA granularity** — smaller blocks mean smaller, more numerous DMAs; below a size threshold the DMA orchestration cost dominates the transfer.

**Generalizable heuristics:** the optimum is almost never the smallest block — per-program + tiny-DMA overhead dominates below roughly the MXU-tile edge, so the small-block crossover *scales with the MXU/VREG geometry* and is smaller on a smaller-MXU generation (re-find it per target, don't copy a v6e block size). There is also a *hardware* reason small contractions lose, independent of DMA overhead: the MXU is a weight-stationary systolic array with a fixed pipeline-fill bubble — back-to-back large matmuls amortize it, thin matmuls (contraction below the MXU edge) pay it as a large fraction of their time ([scaling book ch.2](sources/2025-scaling-book-ch2-tpus.md)). And generations with **multiple MXUs per TensorCore** (e.g. 4 on v4/v5 — generation-specific, re-derive) want tile dims sized to feed all of them, not just one MXU edge. Start around the MXU tile and sweep upward until VMEM OOM; the useful range is a few sizes; the win from right-sizing is real (often tens of percent) but secondary to getting the structure right first. Sweep only *after* the algorithm and grid are fixed. When knobs interact and the directed sweep doesn't pin the optimum, fall back to a constrained enumerative/random search with median-of-repeats ranking — the methodology is already grounded in [pallas-forge's tune harness](codebases/pallas-forge/concepts/pallas_forge-tune-runner.md).

## The dispatch / orchestration floor

A Pallas kernel pays a fixed dispatch + marshalling cost per call, and each program pays scalar-setup + DMA-orchestration cost. **Below a per-call (and per-program) work threshold, a hand kernel loses to XLA's fused path regardless of how good its inner loop is.** Signs you are near the floor: the op does little arithmetic per byte moved (bandwidth-bound); the grid is enormous with tiny per-program work; the kernel issues many small DMAs (per-element/per-page gather); or the op is a single-token decode. In these cases the correct move is a rewrite or accepting XLA, and the honest verdict for a hand kernel is *refute*. Every kernel hypothesis must name its **operating point** (shapes/dtype/batch) — a kernel that wins at a synthetic op-point but sits below the floor at the real one is not a real win.

## Parity & measurement discipline → the earned rules are in BRIEFS

The **generative principles** (what to be aware of when forming a hypothesis) are below; the **binding earned rules** — the exact gate calibration, the ULP-at-magnitude math, the non-degenerate probe, the artifact-or-void audit, the co-measurement protocol, the portability gate — are in [`kernel_experiments/BRIEFS.md`](kernel_experiments/BRIEFS.md) — §1 Measurement, §4 Parity & Oracles, §5 Evidence/Verification/Portability — which the [load mandate](#load-mandate-non-negotiable) pastes into every brief. Do not restate them here; that duplication was the BRIEFS↔index overlap. The principles a hypothesis must respect:

- **Parity gates a high-precision recompute, not the low-precision baseline** — an fp32-accumulating kernel is *more accurate* than a fused-bf16 baseline, so a naive max-abs-vs-baseline reads a false ULP; gate against an fp32 (auto-fp64) oracle that replicates the baseline's own casts, absolute-not-relative near zero denominators, on a tiny shape first, at a **non-degenerate** probe point. (Exact rules + the gate-calibration math: BRIEFS §4.)
- **Read the reference's conventions from its code, not the op name** — pre-scaling, layout, reduction semantics; a wrong assumption is a silent parity fail.
- **A number exists only if measured honestly**: p50 not mean; flags in a separate process; the kernel-fired (silent-noop) HLO audit; the win attributed to a profile-visible mechanism; watch for compiler-inserted relayout `copy`/`bitcast` (a fourth sink) and for a trailing reduction/collective with its *own* roofline. (Exact protocol + every calibration incident: BRIEFS §1/§5.)

## Reading the profile: bound → lever

Measurement recipes live in [`profile-analyzer-index.md`](profile-analyzer-index.md); the *interpretation* is generalizable. Per-functional-unit utilization (from LLO counter tracks) tells you which resource is the bottleneck, and the bottleneck implies the lever:

| Dominant profile signal | What it means | Lever |
|---|---|---|
| a large intermediate visible in HLO / high HBM traffic, low compute util | memory-bound on a materialization | streaming kernel (keep it in VMEM) |
| a `repeat`/`broadcast`/`concat` feeding a matmul in HLO | redundant read | algebraic rewrite |
| vector-load util ≫ MXU util; high wait between compute | DMA-starved compute | more pipeline buffering, then bigger blocks |
| MXU util low with aligned shapes and no wait | blocks too small / grid overhead | bigger blocks; check the dispatch floor |
| scalar-unit util > MXU | scalar-core serialization (address math, control flow, DMA setup) | hoist index math out of the loop, coarsen DMAs |
| register spill/fill activity > 0 | register pressure | shrink live ranges, smaller unroll/last-dims |
| MXU util high and steady, still short of roofline | near the achievable ceiling | consider precision (bf16 compute/fp32 accumulate) or refute |
| kernel fast in isolation but neutral in the real workload | dispatch-bound at the real op-point — OR the surrounding jit inserted collectives/relayouts around the call | validate at the true op-point; check HLO around the custom_call for XLA-inserted all-gathers/copies — pin with `with_sharding_constraint` or move the region to `shard_map` ([scaling book ch.10](sources/2025-scaling-book-ch10-jax.md)) |

Note the *available* signal depends on the kernel style (pipeline-stage/wait signals exist only for `emit_pipeline` kernels; hand-rolled grids expose only the counter/bundle view) — read whichever the trace carries, and fall back to the HLO custom-call firing audit when counter tracks are absent.

## Hardware envelope (how to read utilization anywhere)

Utilization is *slots-used ÷ issue-slot capacity* per functional unit, so you need the capacity row for the target generation (re-derive from a per-bundle-utilization dump's CAPACITY line). The units and what they issue are generation-invariant; the counts change:

| Unit | Issues | v6e slots (example) |
|---|---|---|
| MXU | matrix-multiply pipelines | 2 |
| VALU / VPOP | vector elementwise / permute | 4 / 2 |
| EUP | exponent/transcendental (softmax `exp`) | 1 |
| VLOAD / VSTORE (+ FILL/SPILL) | vector load/store (+ register spill traffic) | 3 / 2 |
| XLU | cross-lane shuffle/reduction | 2 |
| SALU | scalar core (address math, control flow, DMA orchestration) | 2 |

Three unit *mechanisms* worth knowing beyond the slot counts (all generation-invariant in kind, [scaling book ch.2](sources/2025-scaling-book-ch2-tpus.md)): the **scalar core issues at most ~1 DMA per cycle** — no amount of MXU/VPU headroom raises the HBM↔VMEM issue rate, which is *why* "hoist index math, coarsen DMAs" works; the **VPU is roughly an order of magnitude+ below the MXU in FLOPs/s** (~30× on recent gens), so a dense elementwise component starves a matmul kernel; and **cross-lane (last-dim) reductions route through the XLU** and cost markedly more than sublane (second-to-last-dim) reductions — the mechanism behind the layout cost-ordering in [pallas-kernel](concepts/pallas-kernel.md).

Generation constants that gate tiling: MXU tile edge (128 ≤v5p / 256 v6e+), MXUs per TensorCore, VMEM budget and its OOM wall, HBM bandwidth (sets the memory-bound roofline). See [tpu-hardware-generations](concepts/tpu-hardware-generations.md), [vmem-budget](concepts/vmem-budget.md), or query via `pltpu.get_tpu_info_for_chip`.

## Operational gotchas (transferable)

- **The profiling flags that actually work** are narrow and version-sensitive: counter tracks require the two custom-call trace flags in `LIBTPU_INIT_ARGS` (not `XLA_FLAGS`); LLO IR dumps use their own dump-dir flag distinct from the HLO dump dir; several plausible-looking dump flags silently do nothing on a given libtpu. Verify capture worked (non-empty, expected tracks) rather than trusting the flag name — see [[llo-dumps-verified-v6e]].
- **HLO pre-filter is the cheapest triage** — always dump and read `after_optimizations` before authoring; it both finds the bound and catches "XLA already did it."
- **Resource flags are startup/process-scoped** (they re-init the device and contaminate a co-resident reference) → measure in separate processes; and they are generation-gated (a VMEM budget valid on one gen is rejected on another).
- **Framework-version kernel-API drift is real** — primitive names and in-kernel-legal ops move between releases; assume the target version's API, not memory. Prefer ref-slicing primitives over dynamic-slice inside kernels.

## Kernel catalog — reuse before you write (unless authorship is the point)

When retrieval is allowed, an existing well-tuned kernel usually beats a fresh one; when the goal is to *test authorship* (or no library covers the op), author from scratch. Sources: Pallas kernel directory (archival record — prior-campaign branches only; cold runs must not retrieve it), tokamax (silo not on this branch), JAXBench (silo not on this branch) (patterns + roofline grader), [jax pallas](codebases/jax/overview.md), ejkernel (silo not on this branch), pallas-forge (silo not on this branch), maxtext (silo not on this branch). Note coverage gaps are common (e.g. no TPU norm kernel in a library that ships attention + CE) — absence of a library kernel is itself a signal about where authorship pays.

---

## Regeneration

This page drifts as kernel-family experiments land measured lessons. Rebuild it end-to-end (rather than patch forever) via [`kernel-optimization-index-regenerate-prompt.md`](kernel-optimization-index-regenerate-prompt.md) when 4+ new kernel families have landed, a new TPU generation ships, the Pallas API changes materially, or the structure needs rethinking.

## Sources

- [profile-analyzer-index § Deep kernel profiling](profile-analyzer-index.md), [llo-dumps verified on v6e](observations/llo-dumps-and-kernel-profiling-tracks-verified-on-v6e.md), [ep_* scopes = emit_pipeline only](observations/ep-stage-scopes-only-exist-for-emit-pipeline-kernels.md), [chunked_xla bf16 accum](observations/tokamax-chunked-xla-ce-bf16-accumulation-error.md).
- [concepts/llo-utilization.md](concepts/llo-utilization.md), [perf-counters.md](concepts/perf-counters.md), [pallas-kernel.md](concepts/pallas-kernel.md), [arithmetic-intensity.md](concepts/arithmetic-intensity.md).
- Scaling book (already-ingested source pages): [ch.1 rooflines](sources/2025-scaling-book-ch1-roofline.md), [ch.2 TPUs](sources/2025-scaling-book-ch2-tpus.md), [ch.3 sharding](sources/2025-scaling-book-ch3-sharding.md), [ch.9 profiling](sources/2025-scaling-book-ch9-profiling.md), [ch.10 JAX](sources/2025-scaling-book-ch10-jax.md) — mechanism layers folded in 2026-07-11 (deep-research sweep).
- [Ragged Paged Attention (arXiv 2604.15464)](https://arxiv.org/abs/2604.15464) — workload-specialized kernel variants.
