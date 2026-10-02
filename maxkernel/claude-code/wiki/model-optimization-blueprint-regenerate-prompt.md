# Regeneration prompt — `model-optimization-blueprint.md`

Use this prompt next time to rebuild [`model-optimization-blueprint.md`](model-optimization-blueprint.md) from the current wiki state. The page is meant to drift as the experiment ledger accumulates new supported wins and refuted patterns; rather than incrementally patching forever, periodically regenerate end-to-end.

**Heuristic for when to regenerate**: a new model family has reached a frontier OR a new lane has become live OR 100+ new experiments have been filed since the last regeneration OR a new TPU generation has shipped OR the structural rewrite pattern catalog has gained ≥2 new patterns (`§ Pattern N`).

This prompt is the **procedural** sibling of [`model-optimization-index-regenerate-prompt.md`](model-optimization-index-regenerate-prompt.md). The index regenerates from sources / concepts / hardware docs; this blueprint regenerates from **the experiment ledger** + supporting concepts.

---

## The prompt

```
You are rebuilding wiki/model-optimization-blueprint.md from the
current wiki state. The page is the procedural, ordered playbook that
pairs with wiki/model-optimization-index.md (the topic-keyed depth
reference). Both are consulted by the /formulate-hypothesis skill at
hypothesis-generation time; this one tells the agent the ORDER in
which to try things, and the structural-rewrite patterns to apply.

THE CARDINAL RULE — what belongs in this file:

ONLY procedural, ORDERED, ACTIONABLE guidance distilled from the
experiment ledger. Content earns a place on this page only if it
helps an agent decide:

  (a) WHICH PHASE to attack next (the 0–6 ladder),
  (b) WHICH KNOWN-GOOD LEVERS apply at that phase (with magnitude +
      experiment citation),
  (c) WHICH STRUCTURAL REWRITE PATTERN to apply when touching model
      code (the catalog of 7 patterns + anti-patterns),
  (d) WHICH TRAPS / REFUTED PATTERNS to avoid before launching a
      candidate hypothesis (the anti-playbook), or
  (e) WHICH LANE-SPECIFIC CAVEAT applies (JAX-stack vs
      torchax-thin vs MaxText reference).

Content that is NOT procedural-blueprint material does NOT belong:
- Per-experiment narratives in detail (link to the experiment page;
  cite only the v-ID + the headline lift)
- Per-model frontier history (lives in wiki/models/<model>-<lane>.md)
- Pure topic depth (lives in wiki/model-optimization-index.md or in
  wiki/concepts/*.md — link to those, don't duplicate)
- Tutorials / pedagogical / academic prose (link out)
- Mechanism / theory (the model-optimization-index.md sibling owns it)
- Hardware envelope numeric tables (the index owns them) — though a
  table reference IS appropriate when explaining a phase decision
- Inference-side optimization details when the autoresearch loop is
  training-focused

If a candidate addition doesn't fit one of (a)-(e), it goes in the
sibling index, a concept page, an observation, or an analysis — not
here.

READ FIRST (to understand what consumers expect):
- SCHEMA.md
- The current contents of wiki/model-optimization-blueprint.md
- wiki/model-optimization-index.md — the topic-depth sibling
- .claude/skills/formulate-hypothesis/SKILL.md (the consumer)
- .claude/skills/edit-model-code/SKILL.md (the consumer for
  Phase 3 / Structural rewrite pattern catalog)

INPUTS (walk the wiki — don't guess):

1. wiki/experiments/<model>_autoresearch_optimization/<lane>/*.md —
   the experiment ledger. Group by verdict:
   - SUPPORTED → which mechanism category did they win on?
     (Pallas port, kernel removal, structural refactor, scan/compile,
      flag bundle, AC mode, batch amortize, image/build, numerics,
      bring-up). The biggest single-experiment lifts populate Phase 3
      and the Structural rewrite pattern catalog.
   - REFUTED / INCONCLUSIVE → which failure-mode bucket did they fall
     into? (Inert flag at op-point, cross-lane single-axis port, OOM
     treated as flag-tunable, numerics drift, splash+multi-host hang,
     kernel choice wrong for regime, dispatch overhead dominates,
     image/launch infrastructure, calibration). These populate the
     anti-playbook + the Pattern 7 anti-patterns table.
   - INVALID → semantics-changing or infrastructure-only; mostly
     instructive for the silent-no-op-audit concept.
2. wiki/models/<model>-<lane>.md — per-model frontier + Knobs
   translation matrix. The matrix rows where a technique is marked
   "universal: win" become Phase 1/2/3 lever rows.
3. wiki/observations/*.md — reusable findings. The repeated ones
   (across 2+ models) seed the universal-levers section.
4. wiki/analyses/*.md — session retrospectives + multi-model surveys.
   The 2026-05-25 cross-lane retrospective is the canonical cross-
   model evidence base for "universal levers".
5. wiki/concepts/optimization-ladder.md +
   wiki/concepts/compounding-optimizations.md +
   wiki/concepts/silent-noop-audit.md +
   wiki/concepts/numerics-trajectory-validation.md +
   wiki/concepts/inert-flag-flip-trap.md +
   wiki/concepts/cross-lane-transfer-trap.md — the methodology
   concepts that anchor each phase. Phase descriptions cite them; if
   a new methodology concept emerges, add it here AND cite it from
   the relevant phase.
6. wiki/model-optimization-index.md — the topic-depth sibling.
   Every Phase 2 flag bundle, Phase 3 kernel mention, Phase 4 batch
   rule should cross-link into the index's matching topic section.

OUTPUT SCHEMA (preserve this structure — the skill depends on it):

1. Header + preamble — generic, model-agnostic, sibling to the
   index, who reads this and when.
2. TL;DR — the ladder. A 7-row table (Phase 0–6) with effort,
   typical lift, and a one-line citation set per phase.
3. Phase 0 — substrate sanity (mandatory pre-flight). The
   invariants that must hold or every downstream measurement is
   suspect (profiler off; megacore on; libtpu version compat;
   fp32 master non-LoRA; loss trajectory check; silent-no-op
   audit; FSDP divisibility).
4. Phase 1 — cheap memory + precision. The S-effort levers that
   compose cleanly across model families (AC=selective,
   adamw_bf16_state, fp32 master + bf16 compute, bf16 CE / drop
   spurious upcast).
5. Phase 2 — known-good high-yield flag bundle. SPLIT by lane
   where the levers differ: Universal pre-flight, On JAX-stack
   (MaxText bundles + VMEM), with any thinner-lane bundle called
   out where it diverges. Dedicated VMEM-flag-value sweep
   procedure (hyperparameter, 65k–131k range, per-gen cap).
   Definitively-refuted flags table. The flag-sweep trap.
6. Phase 3 — kernel + structural code work. SPLIT by lane into
   JAX-stack sub-ladder (3A ports, 3C structural, 3D scan + cache,
   3E torchax-thin), plus the model-agnostic structural-rewrite
   catalog. Eight of the top-10 unitary experiment lifts live here.
7. Phase 4 — batch + sequence amortization. lbs-doubling at
   FSDP ≥ 8, total batch multiple of 1024, FSDP-divisibility +
   per-chip alignment checks. Sequence sweep at fixed per-shard
   tokens.
8. Phase 5 — cross-variant + cross-generation transfer. Knobs
   translation matrix update rule; AC=selective is NOT universal;
   Pallas block size re-tune per gen; lane → lane never 1:1.
9. Phase 6 — exploratory flag sweep (last resort). When to sweep
   (profile-anchored, narrow, bundle-aware) and when NOT (refuted
   list, default-on, no profile, wrong libtpu version).
10. Structural rewrite pattern catalog. SEVEN H3 subsections:
    Pattern 1 — op-count reduction
    Pattern 2 — weight fusion + matmul folding
    Pattern 3 — reshape / data-formatting elimination
    Pattern 4 — loop / unroll → scan and fold
    Pattern 5 — sharding-aware structural patterns
    Pattern 6 — numerical fidelity during refactors
    Pattern 7 — anti-patterns (rejected rewrites)
    Each pattern carries: diagnostic / when-to-apply, a what-you-
    change column, win citations (v-IDs + magnitude), and mechanism
    explanation. Closing "How to choose which pattern to apply"
    subsection ties patterns to xprof bucket signatures.
11. Universal levers — confirmed multi-model. The patterns that
    repeated across ≥2 model families with the same mechanism
    (MaxText XLA bundle, fp32 master + bf16 compute, splash
    block_q=block_kv ∈ {1024, 2048}, selective remat, bf16 CE +
    drop upcast, adamw_bf16_state, JAX compile cache).
12. Anti-playbook — top 5 traps that repeated. The refuted-pattern
    categories from the ledger, each with the trap concept page
    cross-link.
13. The compounding principle. Canonical walks (a 3B MoE jax
    18→26.6%, Llama 3 8B jax beats MaxText, DeepSeek V3 v5p-512
    7→27%). Cite compounding-optimizations concept.
14. Lane-specific caveats. JAX, torchax, MaxText — one short list
    per lane with what's different.
15. Where to look next — open research questions. The structural
    gaps the ledger has called out (e.g., Zig-Zag ring attention,
    TP=2 code path, MaxText Maverick EP-collective-permute port,
    sharding-in-types splash re-test, v7x/v8 re-tuning,
    per-model refuted-patterns.md layer seeding).
16. How to use this blueprint — the consumer protocol (capture
    profile → walk ladder → cross-check anti-playbook → update
    Knobs translation matrix → promote universal lessons).
17. How to maintain this file — guidance for keeping it generic
    and procedural; what to add and what to defer to per-model
    files / refuted-patterns layer / the sibling index.
18. See also — model-optimization-index / SCHEMA / index / program /
    formulate-hypothesis skill / edit-model-code skill.
19. Sources — the corpora distilled (Gemma 4 + Llama 3 8B +
    Llama 4 + DeepSeek V3 + Flux) with counts. Then the substrate
    sources cited (link to the index's canonical reference stacks).
20. Pointer to this regeneration-prompt file.

CONSTRAINTS (the page's design principles):

- PROCEDURAL, not topical. Every section is an ordered "do this
  next" or "don't do this." Topic depth lives in the
  model-optimization-index sibling and the concept pages.
- GENERIC, not model-specific. Per-experiment data is cited only
  by v-ID + magnitude; the narrative belongs in the experiment
  page. Per-model frontier history belongs in
  wiki/models/<model>-<lane>.md.
- LANE-SPLIT WHERE IT MATTERS. Phase 2 and Phase 3 carry explicit
  per-lane subsections wherever the levers diverge (JAX-stack vs
  torchax-thin vs MaxText reference). Phase 0/1/4/5/6 are mostly
  shared. Cross-lane levers that flip sign across stacks (a lever
  that wins on one lane and regresses on another) must be called
  out at both subsections.
- VMEM FLAG VALUE IS A HYPERPARAMETER. The flag
  xla_tpu_scoped_vmem_limit_kib is near-mandatory; the VALUE
  (sweep 65k–131k per generation × workload) is not a constant.
  Do not regress this distinction.
- THE PATTERN CATALOG IS EVIDENCE-BACKED. Every pattern row cites
  at least one supported experiment with v-ID + magnitude. New
  pattern row must have ≥1 supporting experiment in the ledger.
- ANTI-PATTERNS NEED ≥2 REFUTING EXPERIMENTS to land in Pattern 7
  or the anti-playbook. Single-experiment refutes go in the
  per-model refuted-patterns.md layer, not here.
- COMPOUNDING WALKS ARE LIVING DATA. When a frontier moves on a
  cited variant (e.g., a 3B MoE jax variant moves past 26.6%),
  refresh the walk table; don't accumulate stale numbers.
- SELECTIVE-READ FRIENDLY. The skill reads the TL;DR ladder always
  and ≤2 specific phase sections per hypothesis. Don't add cross-
  section forward references that force a full read.
- LINK TO CONCEPTS FOR DEPTH. Phase descriptions reference the
  concept pages (optimization-ladder, compounding-optimizations,
  silent-noop-audit, numerics-trajectory-validation,
  inert-flag-flip-trap, cross-lane-transfer-trap). Don't duplicate
  their content.
- UNIVERSAL LEVERS ARE THE ≥2-FAMILY RULE. Promote a pattern from
  per-model evidence to a universal lever only when it has won on
  ≥2 model families with the same mechanism. Otherwise it's a
  per-model finding (lives in wiki/models/<model>-<lane>.md).

PROCESS:

1. Read the existing blueprint. Mark sections that haven't changed
   and sections that need refresh.
2. Walk wiki/experiments/<model>_autoresearch_optimization/<lane>/
   for each (model, lane). Count supported vs refuted/inconclusive/
   invalid since the last regeneration.
3. Refresh the top-10 single-experiment lifts. The TL;DR ladder
   row for Phase 3 cites these.
4. Refresh the Structural rewrite pattern catalog:
   - For each of the 7 patterns: are there new supporting wins?
     Update the magnitude + v-ID columns.
   - Are there new pattern categories with ≥3 supporting
     experiments across ≥2 lanes? Add a new pattern row.
   - Are there new anti-patterns (≥2 refute experiments)? Add to
     Pattern 7 or the anti-playbook.
5. Refresh the compounding-walks tables. Are the cited variant
   frontiers still current? If a row has moved, update the span;
   if it has decayed (a refuted experiment retired a step in the
   walk), retire the step from the table.
6. Refresh the Universal levers section. Has any lever been
   refuted on a new lane / model since the last regen? Demote to
   per-lane caveat or remove. Has any new ≥2-family pattern
   emerged? Add.
7. Refresh the Anti-playbook. Are the top 5 traps still the most
   populous failure modes? Re-rank by refuted-experiment count.
8. Refresh Lane-specific caveats. Have any structural ceilings
   moved? Have new lane-specific levers emerged (e.g., a new
   collective-coalescing path on one stack)?
9. Refresh Where to look next. Close any open research items
   that have been validated; add new gaps the ledger has called
   out.
10. Run the LINT operation per SCHEMA — every cited concept /
    experiment / observation page resolves.

VALIDATION (before committing the regenerated file):

- Every cited path resolves (no broken links). Wiki LINT clean.
- The TL;DR ladder has all 7 phases with effort + typical lift +
  citations.
- Every Phase 2 / Phase 3 section carries per-lane subsections
  wherever the levers differ across stacks.
- Every Pattern in the catalog (1–7) cites at least one v-ID with
  magnitude. Pattern 7 cites at least two refutes per row.
- Compounding walks reference current frontiers (no stale numbers).
- Universal levers section enumerates only ≥2-family wins; per-
  model wins are demoted to caveats or per-model files.
- Anti-playbook ranks failure modes by refuted-experiment count
  (or by repetition across model families if counts are close).
- Open research questions are concrete (cite a v-ID where the gap
  was observed; cite a port target where applicable).
- No model-specific narrative prose (it goes to wiki/models/ or
  wiki/experiments/).
- Update wiki/index.md and wiki/log.md per SCHEMA's INGEST /
  ANALYZE conventions.
- Every line passes the cardinal-rule test: it earns its place by
  helping an agent pick the next move along the ladder or by
  steering away from a documented trap. If a line is narrative or
  pedagogical without operational consequence, cut it.

The output is intentionally medium-long (~600 lines) — long enough
to encode the lane-split, the 7-pattern catalog, and the universal-
levers + anti-playbook + compounding walks, but short enough that a
human can read it end-to-end. The skill reads selectively, so total
length doesn't hurt per-hypothesis cost.
```

## When NOT to use this prompt

- Adding a single new supported experiment — append to the appropriate Phase 3 sub-ladder row or pattern catalog cell in place.
- Adding a single new refuted experiment — append to the per-model `refuted-patterns.md` layer (not here) unless it forms a third refute that promotes a pattern to Pattern 7 / the anti-playbook.
- Updating a compounding walk after one frontier move — edit the relevant row in place.
- Renaming files / fixing links — use a normal edit + LINT, not a regen.

Regenerate end-to-end only when a new model family has reached a frontier OR a new lane has become live OR 100+ new experiments have landed OR a new TPU generation has shipped OR the pattern catalog has gained ≥2 new patterns.

## What this prompt is NOT

- A regeneration of `model-optimization-index.md` — that has its own prompt at [`model-optimization-index-regenerate-prompt.md`](model-optimization-index-regenerate-prompt.md). The two prompts are siblings: the index regenerates from sources / concepts / hardware docs; this one regenerates from the experiment ledger + supporting methodology concepts.
- A guide for writing a per-model page or `refuted-patterns.md` layer — those have different consumers (model-specific frontier work + per-variant precedent checks) and don't go through this prompt.
- A consumer-side discipline guide — for the operational discipline of using the blueprint at hypothesis time, see [`/formulate-hypothesis`](../.claude/skills/formulate-hypothesis/SKILL.md) and [`/edit-model-code`](../.claude/skills/edit-model-code/SKILL.md).

## See also

- [`model-optimization-blueprint.md`](model-optimization-blueprint.md) — the file this prompt rebuilds
- [`model-optimization-index.md`](model-optimization-index.md) — the topic-depth sibling
- [`model-optimization-index-regenerate-prompt.md`](model-optimization-index-regenerate-prompt.md) — the sibling regeneration prompt
- [`.claude/skills/formulate-hypothesis/SKILL.md`](../.claude/skills/formulate-hypothesis/SKILL.md) — the consumer
- [`.claude/skills/edit-model-code/SKILL.md`](../.claude/skills/edit-model-code/SKILL.md) — the consumer for the Structural rewrite pattern catalog
- [`SCHEMA.md`](../SCHEMA.md) — wiki structure + the INGEST / ANALYZE / LINT operations the prompt references
