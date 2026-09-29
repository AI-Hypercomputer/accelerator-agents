# Regeneration prompt — `kernel-optimization-index.md`

Use this prompt to rebuild [`kernel-optimization-index.md`](kernel-optimization-index.md) from the current wiki state. The kernel index is a **router + load-mandate** — it names the wiki's kernel-knowledge sources and mandates which get pasted into an author brief per kernel category, wrapped around a *thin, transferable decision procedure* (which class, which order, which category). It is **not** the deep knowledge and **not** the earned rules: the authoring mechanics live in `concepts/pallas-kernel.md`, the measurement recipes in `profile-analyzer-index.md`, and the hard-earned parity/measurement/portability rules in `kernel_experiments/BRIEFS.md` — the index *routes to* all three, it does not restate them (restating BRIEFS's strategy was the historical overlap this structure removes). It drifts as new kernel-family experiments land measured lessons; periodically regenerate end-to-end so the *decision procedure* stays sharp, the *load-mandate rows* stay resolved, and the *evidence base* stays current, rather than patching forever.

**Role split (do not blur):** the index is *regeneratable* (derived from source pages — this prompt rebuilds it). BRIEFS is *not* regeneratable (it accretes one hard-won rule at a time; never rebuild it from this prompt, never inline it here — link + mandate it). `design-class-catalog.md` is an independently-maintained mechanism-class source (link, never inline). All three are **load-mandate targets** of the index, not sections of it.

**Heuristic for when to regenerate**: 4+ new kernel families / verified verdicts have landed since the last regen, OR a new TPU generation has shipped (new envelope constants + a new "re-derive per gen" datum), OR the Pallas framework API changed materially, OR the structure itself needs rethinking. For 1–2 new verdicts, just fold the transferable lesson into the right category + evidence base by hand.

---

## The prompt

```
You are rebuilding wiki/kernel-optimization-index.md from the current
wiki state. The page is the ROUTER + LOAD-MANDATE for kernel
optimization (SCHEMA "Kernel experiment families"): the MASTER reads
it to assemble an author brief (paste the mandated sources per the
kernel's category), and /formulate-hypothesis reads its decision
procedure to generate a falsifiable hypothesis. The AUTHOR does not
read this file — the author reads the PASTED PACK the mandate
produced (this index's category section + concepts/pallas-kernel.md +
kernel_experiments/BRIEFS.md + the per-category doc distillations +
the timing helper).

Two things this file must have, in priority order:
  1. THE LOAD MANDATE — the "always paste" list + the per-category
     table of which source pages MUST be pasted into a brief. This is
     the file's primary job; it is what forces relevant knowledge INTO
     the brief (non-negotiable) instead of leaving the author to write
     from its prior. Elevate it; keep every row resolving to a real page.
  2. THE THIN DECISION PROCEDURE — the transferable synthesis that has
     no other single home (core thesis, three sinks, intervention-class
     decision, category→strategy map, tiling theory, dispatch floor,
     signal→lever, generation-scope, hardware envelope reference).

THE CARDINAL RULE — what belongs in this file:

ONLY (1) the load-mandate, or (2) material that helps optimize an
ARBITRARY (possibly never-seen) Pallas/Mosaic kernel. For (2),
content earns a place only if it is:

  (a) a TRANSFERABLE PRINCIPLE — a mechanism/insight that applies
      across kernels (the core thesis, the three sinks, the class
      decision, the ordering, tiling theory, the dispatch floor), or
  (b) a PROFILE SIGNAL → LEVER mapping (bound signature → what to
      try), or
  (c) a REFUTED PATTERN stated as a principle (what NOT to try, and
      why — "a from-scratch Pallas dense GEMM loses to XLA", not
      "8p lost by 4.4%"), or
  (d) a GENERATION-INVARIANT-vs-SPECIFIC distinction (what transfers
      across TPU gens / to GPU, vs what must be re-derived per
      target), or
  (e) a POINTER to where depth lives (authoring mechanics ->
      concepts/pallas-kernel.md; measurement recipes ->
      profile-analyzer-index.md; existing kernels -> the Pallas
      catalog).

What does NOT belong (this is the anti-overfit discipline — the
page's whole value is that it is NOT a lookup table of solved
kernels):
- A specific block size or speedup presented as a PRESCRIPTION or
  TARGET. Numbers appear ONLY in the compact Evidence base at the
  end, generation-labeled, framed as "calibration for the shape of
  the answer, not a target."
- Per-kernel narrative / per-experiment detail (lives in the
  kernel-family experiment pages).
- Authoring mechanics — the Pallas API, memory spaces, the (8,128)
  tiling rule, dynamic_slice-is-illegal, scalar prefetch. Those live
  in concepts/pallas-kernel.md; MANDATE it, don't duplicate.
- Measurement recipes (which xprof-mcp tool, which flags). Those
  live in profile-analyzer-index.md; cite it.
- The EARNED parity / measurement / portability / process rules —
  the exact gate calibration, ULP-at-magnitude math, non-degenerate
  probe, artifact-or-void audit, co-measurement protocol, fleet
  portability gate. Those live in kernel_experiments/BRIEFS.md and are
  NOT regeneratable. Keep ONLY the one-line generative PRINCIPLE a
  hypothesis must respect (e.g. "gate a high-precision recompute, not
  the bf16 baseline") + a pointer to BRIEFS; MANDATE BRIEFS in the
  load list. Restating BRIEFS's rules here was the historical overlap
  — do not reintroduce it.
- Reproduction of the JAX Pallas docs. The official docs are ingested
  as per-doc distillation pages under
  wiki/codebases/jax/doc-concepts/pallas-*.md; the index carries only
  the READING MAP (which distillation to load for which kernel
  category) — never the content itself.

If a candidate line is a specific result rather than a transferable
principle, either generalize it into a principle + refuted-pattern,
or put the number in the Evidence base — never state it as a rule.

READ FIRST (to understand what consumers expect):
- SCHEMA.md, especially "Kernel experiment families"
- The current contents of wiki/kernel-optimization-index.md
- .claude/skills/formulate-hypothesis/SKILL.md (kernel-family mode —
  the consumer; it reads this file in full)
- wiki/concepts/pallas-kernel.md (the authoring-mechanics companion —
  so you know what to cite vs. duplicate)
- wiki/experiments/program.md "Roles" (the evidence feeding this file
  comes from the INDEPENDENT VERIFIER's numbers, not self-certified
  author claims — trust verified verdicts, discount unverified ones)

INPUTS (walk the wiki — don't guess; the evidence is measured):

1. wiki/experiments/wiki/kernel_experiments/*/ — every
   kernel-family experiment page. For each FILED verdict (supported /
   refuted / invalid): extract the TRANSFERABLE lesson, not the
   number — the kernel CATEGORY it belongs to, the winning
   intervention CLASS (kernel / xla-rewrite / flag / refute), the
   mechanism, and any what-LOST datum (a refuted-pattern principle).
   Prefer verdicts whose evidence came from the kernel-verifier
   (per program.md Roles); flag unverified ones.
2. wiki/analyses/ — the kernel sweep / comparison syntheses (e.g.
   the strict-authorship sweep, the MaxKernel comparison). These
   aggregate the measured findings; re-derive the category strategies
   and the evidence base from them.
3. wiki/observations/ — kernel-relevant reusable findings (LLO tracks,
   ep_* scopes = emit_pipeline only, precision/accumulation traps,
   dispatch-floor observations). Each often becomes a
   parity/measurement/gotcha line or a refuted principle.
4. wiki/concepts/pallas-kernel.md + profile-analyzer-index.md — the
   two companion references. Verify the cross-links resolve; move any
   authoring-mechanics or measurement-recipe content that leaked into
   the index OUT to these pages.
5. wiki/concepts/tpu-hardware-generations.md + vmem-budget.md +
   the per-gen cloud-tpu source pages — re-derive the Generation
   scope section and the "how to read utilization" envelope. Update
   when a new TPU gen ships (its MXU tile edge, VMEM budget/wall,
   slot counts, HBM BW, and which profiling signals it exposes —
   e.g. v7/Ironwood runtime perf-counters vs v6e static-LLO).
6. wiki/analyses/2026-04-23-pallas-kernel-directory.md + the
   connect:auto silo links on concepts/pallas-kernel.md — the
   catalog of existing kernels to reuse-before-writing.
7. wiki/codebases/jax/doc-concepts/pallas-*.md — the official-doc
   distillation pages. Verify the reading-map rows all resolve; if
   the jax checkout gained/lost Pallas docs since the last regen,
   re-ingest the delta (one distillation page per doc, same format)
   before rebuilding the map.
8. wiki/index.md — catalog; identify kernel-relevant material added
   since the last regen.

OUTPUT SCHEMA (preserve this structure — it is the current file's
shape, and it front-loads the highest-value transferable content):

1. Header + intro — a PURPOSE/VALUE header block: this file is the
   ROUTER (not the knowledge); it forces relevant sources into the
   brief; it is REGENERATABLE (vs BRIEFS, the earned rules, which is
   not). Then who reads it (master to assemble, /formulate-hypothesis
   for the decision procedure; the author reads the pasted pack, not
   this file) and the pointers to the routed sources (pallas-kernel.md
   mechanics, profile-analyzer-index measurement, BRIEFS earned rules).
2. THE LOAD MANDATE (non-negotiable) — ELEVATE this near the top; it
   is the file's primary job. The "always paste" list (this index's
   category section + concepts/pallas-kernel.md + kernel_experiments/
   BRIEFS.md + the bench() timing helper) + the per-category table
   mapping each doc-concepts/pallas-* distillation to the categories
   that need it + the catalog-exemplar (retrieval-allowed) rung +
   the never-paste-wholesale list + the WARM-TIER caveat
   (verified_briefs/ and retrospectives-digest.md are per-kernel
   answers — paste ONLY in demo/beat-MK mode for that exact kernel,
   never in a cold/holdout run). Keep it a MAP — content stays in the
   pages it points at.
3. Cardinal rule.
4. Generation scope — what transfers (all the reasoning) vs what is
   generation-specific and must be re-derived (MXU tile, VMEM
   budget/wall/knee, slot counts, HBM BW, available profiling
   signals, all measured numbers).
5. The core thesis — a kernel wins by changing memory
   traffic/materialization/grouping the compiler can't avoid, not by
   "faster math"; profile the bound first.
6. The three sinks of lost time (materialized intermediates /
   redundant reads / dispatch overhead) + the canonical fix for each.
7. THE WORKFLOW + why the order is the order (cheapest-reversible
   first: refute -> flag -> xla-rewrite -> author; correctness before
   speed; structure before tiling before flags; refute honestly).
8. Pick the intervention class — the refute/flag/xla-rewrite/kernel
   decision table.
9. Kernel categories -> strategy — the generalization of the solved
   kernels into a few recognizable categories (streaming-reduction /
   redundant-expansion / grouped-ragged-indirection /
   dense-near-roofline / already-optimal). Add a category ONLY when
   ≥2 kernels' evidence supports a genuinely distinct bound-and-
   structure; don't fragment into per-kernel entries.
10. Tiling theory — the competing pressures + the interior-optimum
   sweep, with the small-block crossover relativized to the MXU/VREG
   geometry (re-find per gen), not an absolute block size.
11. The dispatch / orchestration floor — when NOT to Pallas.
12. Parity & measurement discipline → BRIEFS pointer. Keep ONLY the
    one-line generative PRINCIPLES a hypothesis must respect (gate a
    high-precision recompute not the bf16 baseline; abs-not-rel near
    zero; read the reference's conventions; p50 not mean; separate-
    process flags; the kernel-fired audit; attribute the win to a
    mechanism) + a pointer to kernel_experiments/BRIEFS.md for the
    binding earned rules (exact gate calibration, ULP-at-magnitude,
    non-degenerate probe, artifact-or-void, co-measurement, portability
    gate). Do NOT restate BRIEFS's rules — that was the overlap.
13. Reading the profile: bound -> lever — the utilization-pattern ->
    lever table (generalizable interpretation; recipes are cited to
    profile-analyzer-index).
14. Hardware envelope — how to read utilization on ANY generation
    (units gen-invariant, counts change; re-derive the capacity row);
    the generation constants that gate tiling.
15. Operational gotchas — transferable (flag namespaces + capture
    verification, HLO pre-filter, resource-flag process-scoping,
    API-version drift).
16. Kernel catalog — reuse-before-writing pointers; note coverage
    gaps are themselves a signal.
17. Signal -> lever quick map.
18. Evidence base — the measured runs, COMPACT, generation-labeled,
    as SUPPORT for the principles. Each bullet: category + the
    transferable finding + the pointer; numbers are calibration.
19. Sources.
20. Pointer to this regeneration-prompt file.

CONSTRAINTS (the page's design principles):

- TRANSFERABLE, NOT OVERFIT. This is the primary discipline. The
  value is a decision procedure + a category map for a NEW kernel,
  not answers to memorize. Specific block sizes / speedups live in
  the Evidence base, framed as calibration.
- GENERATION-INVARIANT reasoning vs GENERATION-SPECIFIC constants —
  keep the split explicit (the Generation scope section) and label
  every hardware number with its generation.
- ROUTER FIRST. The load-mandate is the file's primary job — every
  "always paste" item and every per-category row must resolve to a
  real page. If a brief could be assembled wrong from this file, the
  file failed regardless of how good its prose is.
- DEPTH BY CITATION (now MANDATE). Don't duplicate concepts/pallas-
  kernel.md (authoring mechanics), profile-analyzer-index.md
  (measurement recipes), or kernel_experiments/BRIEFS.md (earned
  parity/measurement/portability rules). Route to them via the load
  mandate; keep only the one-line generative principle + pointer.
  Push any leakage back to those pages. Restating BRIEFS's rules is
  the specific overlap this structure exists to prevent.
- REFUTED PATTERNS AS PRINCIPLES. "A from-scratch Pallas dense GEMM
  loses to XLA (dispatch-bound)"; "a 1-token-decode paged kernel
  loses (orchestration-bound)"; "small blocks below ~the MXU tile
  lose". State the failure mode + why, not the experiment ID.
- CATEGORIES OVER CASES. Generalize the solved kernels into
  bound-and-structure categories; a reader should recognize a NEW
  kernel's category and inherit the strategy.
- DENSE, NOT SPRAWLING. Target ~150-260 lines. The master reads it to
  assemble a brief and /formulate-hypothesis reads its decision
  procedure, so tightness matters; it is NOT the 700-900-line topic
  catalog the model index is. If a line isn't the load-mandate, a
  transferable principle, a cited pointer, or a compact evidence
  bullet, cut it.

PROCESS:

1. Read the existing file. Mark which principles are stable and which
   the new evidence sharpens or contradicts.
2. For each kernel-family verdict filed since the last regen: derive
   the transferable lesson (category, class, mechanism, refuted
   principle). Fold it into the right category-strategy /
   refuted-pattern / evidence-base line — as a PRINCIPLE, with the
   number only in the evidence base. Prefer verifier-produced
   evidence.
3. Re-evaluate the kernel categories. Add one only when a genuinely
   new bound-and-structure has ≥2 kernels behind it. Merge/rename to
   keep the map small.
4. Update Generation scope + the envelope when a new TPU gen ships
   or new hardware constants are measured. Re-check every hardware
   number is generation-labeled.
5. Rebuild the Evidence base from the current experiment pages /
   analyses — compact, generation-labeled, one bullet per category
   with pointers.
6. Move any authoring-mechanics or measurement-recipe content that
   crept in OUT to concepts/pallas-kernel.md /
   profile-analyzer-index.md, and cite instead.
7. Run LINT per SCHEMA.

VALIDATION (before committing):

- Every cited path resolves (no broken links); the Generation-scope
  anchor that concepts/pallas-kernel.md links to still exists.
- Every row of the docs reading map resolves to an existing
  codebases/jax/doc-concepts/pallas-*.md page, and every distillation
  page under doc-concepts/ appears in the map (or is deliberately
  excluded with a reason).
- ANTI-OVERFIT CHECK: no specific block size or speedup appears
  OUTSIDE the Evidence base as a prescription/target; every hardware
  number is generation-labeled; the Generation-scope section names
  what transfers vs what to re-derive.
- Every kernel category has a strategy + at least one what-LOST /
  refuted note (or "none observed yet").
- Authoring mechanics are MANDATED to concepts/pallas-kernel.md, not
  duplicated; measurement recipes cited to profile-analyzer-index;
  earned parity/measurement/portability rules MANDATED to
  kernel_experiments/BRIEFS.md, not restated (only the one-line
  generative principle + pointer remains).
- LOAD-MANDATE RESOLVES: the "always paste" list names this index's
  category section + pallas-kernel.md + BRIEFS + the timing helper;
  every per-category doc row resolves; the warm-tier caveat
  (verified_briefs/, retrospectives-digest.md = per-kernel answers,
  demo-mode only) is present.
- The Evidence base cites the kernel-family experiment pages /
  analyses it rests on, and prefers verified verdicts.
- Length ~150-260 lines; every line is a transferable principle, a
  cited pointer, or a compact evidence bullet.
- Update wiki/index.md and wiki/log.md per SCHEMA.
```

## When NOT to use this prompt

- Folding in 1–2 new kernel-family verdicts — just add the transferable lesson to the right category + a compact Evidence-base bullet, by hand.
- Fixing links / renaming — normal edit + LINT.
- Adding authoring mechanics or measurement recipes — those go in `concepts/pallas-kernel.md` / `profile-analyzer-index.md`, not here.

Regenerate end-to-end only when 4+ new kernel families have landed, a new TPU generation shipped, the Pallas API changed materially, or the structure needs rethinking.

## See also

- [`kernel-optimization-index.md`](kernel-optimization-index.md) — the file this prompt rebuilds
- [`concepts/pallas-kernel.md`](concepts/pallas-kernel.md) — authoring-mechanics load-mandate target (mandate, don't duplicate)
- [`kernel_experiments/BRIEFS.md`](kernel_experiments/BRIEFS.md) — the earned-rules load-mandate target (mandate, never restate; NOT regeneratable)
- [`profile-analyzer-index.md`](profile-analyzer-index.md) — the measurement companion (cite, don't duplicate)
- [`.claude/skills/formulate-hypothesis/SKILL.md`](../.claude/skills/formulate-hypothesis/SKILL.md) — the consumer (reads the decision procedure; the master reads the load mandate to assemble briefs)
- [`SCHEMA.md`](../SCHEMA.md) — "Kernel experiment families" + the INGEST / LINT operations
- [`model-optimization-index-regenerate-prompt.md`](model-optimization-index-regenerate-prompt.md) — the sibling prompt this mirrors

## Amendment (2026-07-15): campaign sources are mandatory regeneration inputs

Any regeneration of kernel-optimization-index.md MUST:
1. Read and incorporate as evidence sources: `kernel_experiments/design-class-catalog.md` (9 mechanism classes — keep it a SEPARATE document and LINK to it; do not inline it, it is independently maintained), `kernel_experiments/BRIEFS.md` (trap catalog + rules R1–R8 — link, never inline), the 2026-07 verified physics results (12p HBM-roofline tie, 11p/14p uniform-GMM einsum ~68% MXU ceiling, 16p intrinsic-precision proof, 4p exclusion chain), and `analyses/2026-07-15-multi-arm-kernel-benchmark-campaign.md`.
2. Preserve a "campaign layer" section equivalent to the 2026-07-15 one: the reading order (catalog → BRIEFS → kernel-root program.md incl. its 2026-07 amendments → this index) and the campaign_lanes reuse policy (worked examples, not answers; tiles/flags/ratios are op-point-bound).
3. Keep the separation of roles: this index = **router + load-mandate + thin decision procedure** (regeneratable — this prompt rebuilds it); BRIEFS = **hard-earned distilled rules** (NOT regeneratable — accretes one lesson at a time; link + mandate, never inline or rebuild); design-class-catalog = curated mechanism-class source (independently maintained; link, never inline). All three are **load-mandate targets** of the index, not sections of it. The index restating BRIEFS's strategy/rules was the historical overlap this split removes — do not reintroduce it.
