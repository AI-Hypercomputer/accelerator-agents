# Regeneration prompt — `profile-analyzer-index.md`

Use this prompt next time to rebuild [`profile-analyzer-index.md`](profile-analyzer-index.md) from the current wiki state. The page is meant to drift as the wiki accumulates new xprof / HLO / Pallas / TPU-device material; rather than incrementally patching forever, periodically regenerate end-to-end.

**Heuristic for when to regenerate**: 10+ new reference sources have landed since the last regeneration, OR a new TPU generation has shipped, OR the mcp__xprof__ server has added/renamed tools, OR the structure itself needs rethinking.

---

## The prompt

```
You are rebuilding wiki/profile-analyzer-index.md from the current
wiki state. The page is the reference catalog that the
profile-analyzer agent reads in full at Step 0 of every analysis.

THE CARDINAL RULE — what belongs in this file:

ONLY information that helps an agent INTERPRET a captured xprof trace
or HLO dump. Concretely, content earns a place on this page only if it
helps the agent answer:

  (a) WHICH tool to call to get a specific piece of data out of a trace
      or HLO dump.
  (b) WHAT a metric / bucket / counter MEANS when it comes back from
      the tool.
  (c) HOW the device or compiler structurally produces what we observe
      (HLO op semantics, fusion patterns, collective forms, sharding
      annotations, custom_call shapes).
  (d) WHICH GENERATION-SPECIFIC CONSTANTS to apply when interpreting
      numbers (MXU shape, VMEM cap, HBM BW, ICI BW) so the agent
      doesn't have to look them up mid-analysis.
  (e) HOW to format an observation page so findings extracted from a
      profile follow the wiki schema.

Content that is NOT analysis-reference material does NOT belong:

- HYPOTHESIS LOGIC of any kind. "If you see X, try Y" is the
  profile-analyzer's Phase 3 job, driven by the EXPERIMENT PAGE'S
  hypothesis context — NOT by this index. The index gives the tools
  and the meaning of the data; the agent supplies the judgment per
  experiment.
- OPTIMIZATION RECOMMENDATIONS. Those live in
  wiki/model-optimization-index.md. The two indexes are strictly
  separated: model-optimization-index says "if memory-bound, try
  Pallas kernels with these tradeoffs"; profile-analyzer-index says
  "memory-bound means HBM BW util > 70%; here's how to get that
  number; here's what HLO ops typically appear in a memory-bound
  workload".
- Per-experiment narratives, per-model frontier history, refuted-
  pattern lists with v-IDs — those live in wiki/experiments/,
  wiki/models/, per-model refuted-patterns.md.
- Hypothesis-firing recipes ("for kernel-port hypothesis, grep HLO
  for target X"). These are tempting, but they're hypothesis-class
  prescriptions. KEEP THEM OUT. The agent forms them in Phase 3 from
  the experiment page; the index supports that work by documenting
  the structural primitives (how to grep HLO, what custom_call
  targets exist) WITHOUT the prescriptive mapping.
- Tutorials / pedagogical material. Link out; don't duplicate.
- Hardware roadmap speculation that isn't tied to a current
  generation in production.

If a candidate addition doesn't fit (a)-(e), it goes in a concept page,
source page, codebase page, observation, or analysis — not here.

READ FIRST (to understand what consumers expect):
- SCHEMA.md
- The current contents of wiki/profile-analyzer-index.md
- .claude/agents/profile-analyzer.md (the consumer)
- wiki/model-optimization-index-regenerate-prompt.md (the sibling
  regen prompt; mirror its discipline, but DO NOT replicate its
  content scope — they're strictly separated)

INPUTS (walk the wiki — don't guess):

1. wiki/index.md — catalog. Identify all xprof / HLO / Pallas /
   tpu-device / mcp sources / concepts / codebases added since the
   last regeneration.
2. wiki/sources/2026-xprof-*.md — every xprof tool / view doc. Group
   into: triage screen (overview), op profile, timeline, memory,
   HLO viewer, roofline, megascale, captures, framework-specific.
3. wiki/sources/2026-cloud-tpu-*-docs.md — per-gen TPU device docs.
   Re-derive the per-generation envelope table. Include MXU shape,
   VMEM cap, HBM BW, ICI BW, critical FLOP/byte.
4. wiki/sources/2025-scaling-book-ch1-*.md (roofline), ch2-*.md
   (TPUs), ch9-*.md (profiling) — substrate references.
5. wiki/concepts/ — read filenames + lead paragraphs. The relevant
   subset:
   - HLO + XLA: hlo, hlo-op, xla-fusion, xla-compiler-flags,
     xla-cost-model, hlo-dumping-and-diffing, custom-call,
     scan-over-layers, jaxpr-hash-cache-keys,
     torch-tpu-compilation-cache, manual-mlir-dialect-pallas,
     mosaic-kernel
   - Device + memory: vmem, vmem-budget, hbm, mxu, memory-hierarchy,
     mxu-utilization, memory-bandwidth-utilization, llo-utilization
   - Profiling: roofline-model, ridge-point, ici-roofline,
     trace-event-categories, perf-counters, profile-capture
   - Collectives: all-reduce, all-gather, reduce-scatter,
     async-collectives, collective-bucketing-prefetch,
     collective-communication, send-recv-done
   - libtpu: libtpu-flags
6. wiki/codebases/xprof-mcp.md — THIS IS THE SINGLE SOURCE OF TRUTH
   for the tool catalog. The index's tool table cites the xprof-mcp
   page line-for-line (file paths + line numbers + 1-line uses).
   Mirror its tool table verbatim with anchor IDs.
7. wiki/codebases/xprof.md — xprof server internals.
8. wiki/codebases/stablehlo.md — StableHLO spec + passes ref.
9. wiki/codebases/pallas-forge.md, tokamax.md, ejkernel.md — kernel
   sources whose custom_call_targets show up in HLO.
10. wiki/analyses/2026-04-23-pallas-kernel-directory.md — Pallas
    kernel catalog.
11. wiki/observations/ — any observation that documents an xprof
    bucket attribution pattern or HLO fingerprint pattern (NOT for
    citing as "lessons", but for grounding "here's what these
    patterns look like in practice").

OUTPUT SCHEMA (preserve this structure — the agent depends on it):

1. Header + lead paragraph — what this index is, who reads it, link
   to the consumer agent.
2. Cardinal rule — explicit statement of (a)-(e), explicit list of
   what does NOT belong, strict separation from
   model-optimization-index.
3. TLDR — analysis flow — the canonical generic 10-step walk through
   the tools. Table format: step / tool / question answered. Anchor
   each tool name to its row in the tool catalog (§ 4).
4. mcp__xprof__ tool catalog — full per-tool reference. Each row:
   tool name (with anchor ID), file path + line in raw/code/xprof-mcp,
   data path (HTTP / .xplane.pb / XLA dump dir), 1-line use.
   Mirror exactly what wiki/codebases/xprof-mcp.md says — don't
   re-derive.
5. Tool-call ergonomics — small subsection on return shapes,
   error handling, caching, format-awareness, multi-host hints.
6. xprof bucket taxonomy — time decomposition, headline metrics
   from get_overview (table with key / meaning / healthy band),
   op-profile metrics, roofline interpretation, standard timeline
   regexes.
7. TPU device model — per-generation envelope table (cited from
   model-optimization-index — it's canonical there), XPK naming
   caveats, practical VMEM caps observed, memory hierarchy, compute
   units (MXU / VPU / scalar / SparseCore).
8. HLO IR reference — module structure, op categories table (op,
   what it does, what attrs to read), sharding annotations, fusion
   patterns, collective forms table, special metadata to look at.
9. StableHLO reference — why the agent reads StableHLO, spec + passes
   pointers, practical tooling.
10. Pallas + custom_call signatures — naming table by target,
    reading "did the kernel fire", backend_config inspection,
    kernel sources in raw/code.
11. HLO dumping + diffing — XLA_FLAGS for dump, filename conventions,
    stage aliases, diff workflow.
12. Reading recipes (how-to patterns) — HLO module count fingerprint,
    eager vs compiled detection, recompilation detection, collectives
    gap, finding a specific op in HLO, reading the memory profile,
    detecting misaligned MXU shape, SparseCore stats. Each recipe is
    a HOW (not a WHAT-to-look-for).
13. Generation-specific gotchas — per-gen behaviors that affect
    interpretation (v5p / v6e / v7x / v8-preview).
14. Observation page template — schema reminder for how to file
    findings (so the agent doesn't drift from SCHEMA).
15. See also — pointers to SCHEMA, profile-analyzer agent,
    model-optimization-index, xprof / xprof-mcp / stablehlo codebase
    pages, Pallas kernel catalog, this regen prompt.
16. Sources — substrate sources.
17. How to maintain — guidance for incremental updates.

CONSTRAINTS (the page's design principles):

- ANALYSIS-REFERENCE FOCUS. Each line earns its place via the
  cardinal rule. If it's prescriptive, cut it.
- STRICT SEPARATION FROM MODEL-OPTIMIZATION-INDEX. That index says
  what to try; this index says what to read and what it means. Do
  NOT replicate optimization recommendations here.
- DEPTH BY CITATION. Don't duplicate xprof-mcp.md's tool catalog —
  cite it line-for-line. Don't duplicate concept page content — link.
  Don't duplicate per-gen envelope numbers — cite model-optimization-
  index as canonical.
- TOOL CATALOG IS LOOKUP-FREE. The agent should never need to grep
  xprof-mcp source mid-analysis to find a tool's args.
- TLDR IS THE FRONT DOOR. The 10-step canonical walk lets the agent
  reach a triage verdict in <10 calls before deciding whether to go
  deeper.
- HARDWARE ENVELOPE IS LOOKUP-FREE FOR INTERPRETATION. The per-gen
  numbers (MXU shape, VMEM cap, critical FLOP/byte) must be in the
  index so the agent doesn't context-switch to look them up while
  reasoning about a roofline question.
- READING RECIPES ARE HYPOTHESIS-AGNOSTIC. "How to find collectives
  gap" yes. "How to know if your FSDP-bucketing experiment worked"
  no.
- HYPOTHESIS-FIRING / SILENT-NOOP CHECKS belong in the agent's
  Phase 3 spec (in .claude/agents/profile-analyzer.md), NOT here.
  The index supports Phase 3 with structural primitives (how to read
  a custom_call_target, how to list HLO modules) but does NOT
  enumerate the per-hypothesis-class verification recipes. That
  mapping would be a prescriptive layer the agent should generate
  per-experiment from the experiment page.

PROCESS:

1. Read the existing file. Mark sections that haven't changed and
   sections that need refresh.
2. Walk wiki/codebases/xprof-mcp.md first. If the tool catalog has
   changed (new tools added, renamed, removed, signatures changed),
   refresh § 4 to match exactly.
3. Walk wiki/sources/2026-xprof-*.md. For each new source: identify
   which § it deepens (bucket taxonomy / dumping / recipes / etc.).
   Add citation; do NOT inline its content unless it's a new TPU
   constant or a new tool.
4. Walk wiki/sources/2026-cloud-tpu-*-docs.md. If a new generation
   has shipped, add it to the per-gen envelope and add a generation-
   specific-gotchas subsection.
5. Walk wiki/concepts/ for new pages in the relevant set (HLO /
   device / profiling / collectives / libtpu). Cite each from its
   appropriate § without inlining content.
6. Walk wiki/codebases/ for new kernel sources (pallas-forge,
   tokamax, ejkernel updates). If new custom_call_targets are
   documented, add rows to § 10.
7. Walk wiki/analyses/2026-04-23-pallas-kernel-directory.md for
   updates; the catalog reference in § 10 stays current.
8. Re-evaluate the recipes (§ 12). Add a recipe only when a new HOW-
   TO pattern emerges that's hypothesis-agnostic and reusable across
   multiple experiments. Do not add hypothesis-class verification
   recipes.
9. Run the LINT operation per SCHEMA after — particularly check that
   every cited concept / source / codebase page exists.

VALIDATION (before committing the regenerated file):

- Every cited path resolves (no broken links).
- The tool catalog (§ 4) row count matches the count in
  wiki/codebases/xprof-mcp.md (or differs only by tools intentionally
  excluded from the agent's surface).
- Per-gen envelope (§ 7) is consistent with
  wiki/model-optimization-index.md's envelope.
- No prescriptive content snuck in. Grep the page for "should try",
  "consider", "recommend" — any such phrasing is a smell. Either
  remove or rephrase to be descriptive ("rms-norm appears in HLO
  as a sequence of convert + multiply + reduce" yes; "rms-norm
  fusion is suboptimal — consider Pallas port" no).
- No model-specific experiment IDs (v391y, v520, etc.) — those
  belong in per-model files. Memory references for canonical
  illustrative cases are OK (e.g. "see v675b memory for the
  canonical eager-vs-compiled detection case") because they're
  factual.
- The TLDR step count is roughly 8-12; if it's growing, that's a
  signal the analysis flow is becoming too prescriptive.
- Update wiki/index.md and wiki/log.md per SCHEMA's INGEST
  conventions.

The output is intentionally long (~600-800 lines) because it has wide
surface area. The agent reads in full at Step 0; total length matters
for the per-analysis cost. Keep it ≤ 900 lines. Cut tutorial
prose ruthlessly. Cite instead of duplicating.
```

## When NOT to use this prompt

- Adding 1-2 new sources — just edit the relevant § in place.
- Updating per-experiment narratives — those don't belong on the page; they go in `wiki/experiments/`.
- Renaming files / fixing links — use a normal edit + LINT, not a regen.
- Adding hypothesis-class verification recipes — those don't belong here at all (they go in the agent's Phase 3 spec, or are formed ad-hoc per experiment).

Regenerate end-to-end only when the structure itself needs rethinking, a new TPU generation has shipped, the mcp__xprof__ tool catalog has materially changed, or 10+ reference sources have landed since the last regen.

## See also

- [`profile-analyzer-index.md`](profile-analyzer-index.md) — the file this prompt rebuilds
- [`.claude/agents/profile-analyzer.md`](../.claude/agents/profile-analyzer.md) — the consumer
- [`model-optimization-index-regenerate-prompt.md`](model-optimization-index-regenerate-prompt.md) — the sibling index's regen prompt (strict-separation reference)
- [`SCHEMA.md`](../SCHEMA.md) — wiki structure + the INGEST / LINT operations the prompt references
