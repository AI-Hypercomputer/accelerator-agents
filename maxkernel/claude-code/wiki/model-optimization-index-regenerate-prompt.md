# Regeneration prompt — `model-optimization-index.md`

Use this prompt next time to rebuild [`model-optimization-index.md`](model-optimization-index.md) from the current wiki state. The page is meant to drift as the wiki accumulates new sources / concepts / hardware generations; rather than incrementally patching forever, periodically regenerate end-to-end.

**Heuristic for when to regenerate**: 10+ new sources have landed since the last regeneration, OR a new TPU generation has shipped, OR the structure itself needs rethinking.

---

## The prompt

```
You are rebuilding wiki/model-optimization-index.md from the current
wiki state. The page is the topic-keyed lookup table that the
/formulate-hypothesis skill consults at hypothesis-formulation time.

THE CARDINAL RULE — what belongs in this file:

ONLY information about MODEL OPTIMIZATION that is USEFUL FOR HYPOTHESIS
GENERATION. Concretely, content earns a place on this page only if it
helps an agent decide:

  (a) WHICH optimization direction to try next (TLDR mapping from
      profile signature → topic → first-line hypothesis class), or
  (b) HOW the optimization is achieved at the mechanism level
      (concept page + source citations + generic principles), or
  (c) WHAT NOT to try because the principle is structurally refuted
      (generic refuted-pattern principles), or
  (d) WHICH HARDWARE the hypothesis runs on, with the exact constants
      (VMEM caps, HBM BW, bytes/flop, MXU shape, ICI BW) that the
      agent shouldn't have to look up mid-hypothesis, or
  (e) WHERE the kernel / recipe already exists (Pallas catalog quick-
      pick) so the agent ports instead of re-writing.

Content that is NOT hypothesis-generation material does NOT belong:
- Per-experiment narratives (those live in wiki/experiments/)
- Per-model frontier history (lives in wiki/models/<model>-<lane>.md)
- Per-model refuted experiments with v-IDs (lives in
  wiki/experiments/<model>_autoresearch_optimization/refuted-patterns.md)
- Tutorials / pedagogical material (link out; don't duplicate)
- Inference-side optimization details when training is the loop's focus
  (mention in TLDR for completeness; don't expand into a topic section
  unless inference enters the autoresearch loop)
- Hardware roadmap speculation that isn't tied to a current decision
- Compiler internals that don't surface as a flag, code pattern, or
  observable profile signature

If a candidate addition doesn't fit one of (a)-(e), it goes in a
concept page / observation / source page / analysis — not here.

READ FIRST (to understand what consumers expect):
- SCHEMA.md
- The current contents of wiki/model-optimization-index.md
- .claude/skills/formulate-hypothesis/SKILL.md (the consumer)

INPUTS (walk the wiki — don't guess):

1. wiki/index.md — catalog. Identify all sources / codebases / concepts
   added since the last regeneration.
2. wiki/sources/ — read titles + lead paragraphs of each source. Group
   by topic. Filter to those that influence hypothesis generation
   (most do; some are pure reference material that the skill cites by
   URL without needing prose in this index).
3. wiki/concepts/ — read filenames + lead paragraphs. Note which are
   generic principles vs which are TPU-hardware-specific. Both count.
4. wiki/observations/ — these are reusable findings; cross-reference
   per topic. Observations are often the source of new TLDR rows.
5. wiki/analyses/2026-04-23-pallas-kernel-directory.md + subpages —
   the Pallas catalog. Re-derive the quick-pick-by-function table.
6. wiki/concepts/tpu-hardware-generations.md +
   wiki/sources/2026-cloud-tpu-v*-docs.md +
   wiki/sources/2026-cloud-tpu-8t-8i-announcement.md — re-derive the
   hardware envelope tables. Include critical arithmetic intensity per
   generation.
7. wiki/sources/2026-xprof-mcp-tpu-optimization.md — the canonical
   decision tree. The TLDR table descends from this.
8. wiki/sources/2025-scaling-book-ch3-sharding.md — the 4-case
   matmul taxonomy. The Sharding strategy topic descends from this.
9. wiki/concepts/libtpu-flags.md, wiki/concepts/xla-compiler-flags.md
   — the universal libtpu stack. Note which flags are deprecated.

OUTPUT SCHEMA (preserve this structure — the skill depends on it):

1. Header + preamble — generic, model-agnostic, how the
   /formulate-hypothesis skill uses this file.
2. TLDR — profile-signature → topic → first-hypothesis-class table.
   Add a row whenever a recurring profile signature lands in
   observations or recent experiments.
3. TPU hardware envelope — per-generation tables (v5p, v6e, v7x,
   v8 preview, …). Per generation: bf16 peak, FP8/FP4 peak where
   applicable, HBM cap, HBM BW, VMEM cap, TCs/chip, MXU shape, ICI BW,
   critical arithmetic intensity, bytes/flop. XPK naming caveat per
   gen. Roofline implications. Topology cheat sheet (XPK type → chip
   count → slice shape → common use).
4. Canonical reference stacks — pointer table to substrates
   (MaxText / tokamax / tpu-recipes / scaling-book / xprof-mcp /
   ultrascale / JAX / torchtitan / pallas-forge /
   ejkernel / ringattention …).
5. Checkpointing + persistence — Orbax + multi-tier patterns
   (HBM → host pinned via jax.experimental.layout.Format → GCS).
   Async-save scale numbers.
6. Optimization topic index — one section per topic. Each section:
   - Mechanism — concept + source + codebase citations. May have
     theory / partitioning-systems / sequence-parallelism / concepts
     sub-sections when the topic is large.
   - Generic principles — model-agnostic lessons.
   - Generic refuted-pattern principles — high-level traps stated
     without specific experiment IDs.

   Current topics (refresh as new categories emerge):
   - Sharding strategy
   - Dimension alignment + dtype (incl. FP8/FP4)
   - Profile-driven attribution (incl. PT2-side diagnostic
     complements: tlparse + HTA + ezyang)
   - CE / softmax loss
   - MoE expert dispatch
   - Splash attention (incl. NSA / FA3 / TyphoonMLA / RPA cross-refs)
   - Tensor parallelism
   - FSDP / collective optimization
   - VMEM / scratch memory
   - Activation checkpointing
   - Batch-size (lbs) amortization
   - Pallas kernel patterns (split: Kernel catalog + Authoring
     principles)
   - torch.compile + scan-over-layers
   - HBM transient memory + I/O
   - Hardware envelope (v5p vs v6e vs v7x) — the "cross-gen recipe
     transfer" version of the envelope (the upfront tables are the
     numeric envelope; this section is the principles).
   - libtpu / XLA flags

7. How to maintain this file — guidance for keeping it generic and
   not per-experiment.
8. See also — SCHEMA / wiki/index / program.md / the consumer skill
   (formulate-hypothesis) / edit-model-code skill (the implementation
   discipline).
9. Sources — the substrate sources this index rests on (the small
   list of "always cited" foundations).
10. Pointer to this regeneration-prompt file.

CONSTRAINTS (the page's design principles):

- GENERIC, not model-specific. Per-experiment data goes in the model
  page and per-model refuted-patterns.md. Per-model lessons that
  GENERALIZE are promoted here as principles (e.g., "FLOPs-only cost
  analysis is incomplete" yes; "v391y refuted on 3B/v5p-16" no).
- SELECTIVE-READ FRIENDLY. The skill picks 1-3 topic sections, not
  the whole file. Don't add cross-section forward references that
  force a full read.
- DEPTH BY CITATION. Don't duplicate concept page content. Link to it.
- HARDWARE ENVELOPE IS LOOKUP-FREE. Include exact numbers (VMEM caps,
  HBM BW, bytes/flop) so agents aren't guessing mid-hypothesis.
- REFUTED-PATTERN PRINCIPLES are high-level, model-agnostic. Cite
  the failure mode, not the experiment ID.
- TLDR is the front door. Maintain the profile-signature → topic
  mapping as new signatures are observed in the wiki.
- HYPOTHESIS-GENERATION FOCUS. Each line should justify itself
  against the (a)-(e) cardinal rule above. If it doesn't, cut it.

PROCESS:

1. Read the existing file. Mark sections that haven't changed and
   sections that need refresh.
2. For each new source / codebase / concept ingested since last
   regeneration: ask whether it's hypothesis-generation material
   (cardinal rule). If yes, identify which topic section it belongs
   to and cite it under the right "Mechanism" subsection.
3. Update the Hardware envelope when a new TPU gen ships
   (currently v5p / v6e / v7x / v8 preview).
4. Update the Canonical reference stacks pointer table when a new
   substrate is ingested.
5. Update the Pallas kernel catalog quick-pick table if a new kernel
   source enters the catalog (the catalog itself is in
   analyses/2026-04-23-pallas-kernel-directory.md and subpages).
6. Re-evaluate the topic list. Add a topic section only when an
   optimization category has ≥3 supporting sources / observations /
   concept pages that aren't already covered by an existing topic.
   Don't fragment.
7. Run the LINT operation per SCHEMA after — particularly check
   that every cited concept / source / codebase page exists.

VALIDATION (before committing the regenerated file):

- Every cited path resolves (no broken links).
- The TLDR table has a row for every dominant profile signature
  observed in recent experiments / observations.
- The Hardware envelope reflects the latest TPU generation docs
  ingested under wiki/sources/2026-cloud-tpu-*-docs.md.
- Every topic section has both Mechanism citations AND Generic
  refuted-pattern principles (or an explicit "none documented yet"
  for young topics).
- No model-specific experiment IDs (v391y, v520, etc.) — those
  belong in per-model files.
- The Pallas catalog quick-pick table cites both TPU and GPU sources
  per functional category where both exist.
- Update wiki/index.md and wiki/log.md per SCHEMA's INGEST
  conventions.
- Every line passes the cardinal-rule test: it earns its place by
  helping an agent generate or filter a hypothesis. If a line is
  tutorial / commentary / narrative without operational consequence,
  cut it.

The output is intentionally long (700-900 lines) because it has wide
surface area. The skill reads selectively, so total length doesn't
hurt per-hypothesis cost. But every line must earn its place via the
cardinal rule.
```

## When NOT to use this prompt

- Adding 1-2 new sources — just edit the relevant topic section in place.
- Updating per-experiment narratives — those don't belong on the page; they go in `wiki/experiments/`.
- Renaming files / fixing links — use a normal edit + LINT, not a regen.

Regenerate end-to-end only when the structure itself needs rethinking, a new TPU generation has shipped, or 10+ sources have landed since the last regen.

## See also

- [`model-optimization-index.md`](model-optimization-index.md) — the file this prompt rebuilds
- [`.claude/skills/formulate-hypothesis/SKILL.md`](../.claude/skills/formulate-hypothesis/SKILL.md) — the consumer
- [`SCHEMA.md`](../SCHEMA.md) — wiki structure + the INGEST / LINT operations the prompt references
