# TPU Model Performance Auto-optimization — Schema & Operating Rules

*(Autoresearch methodology; see [README](README.md) for the project intro.)*

You maintain a knowledge base in service of an **automated optimization loop** for TPU model performance. The loop is: ingest knowledge (papers, docs, codebases, profiles) → formulate ranked hypotheses → run experiments → record observations → update priors → formulate next hypotheses. Your job is to write and maintain every file. The human curates sources, sets optimization targets, approves experiments, and arbitrates contradictions.

**Scope:** step time, MFU, tokens/sec, memory, and everything that affects them on TPU — compiler flags, parallelism, rematerialization, attention kernels, layout, fusion, scheduling, precision. Out of scope: model quality/convergence. An optimization that changes model semantics is **invalid** — note the invalidation, do not report the speedup.

**Independence:** this wiki is self-contained. Do **not** read from or link into any sibling wiki. All state lives under the paths below.

---

## Directory layout

```
tpu_performance_autoresearch_wiki/     ← project root (agent CWD)
  SCHEMA.md                            ← this file — single source of truth
  CLAUDE.md                            ← @SCHEMA.md pointer
  notes.md                             ← human scratchpad (do not modify)
  wiki/
    index.md                           ← catalog — update on every write
    log.md                             ← append-only event log — newest on top
    sources/                           ← one page per ingested paper/article/doc/talk
    codebases/                         ← one grounded silo dir per ingested repo (+ optional curated companion page)
    concepts/                          ← techniques, abstractions, flags, kernels
    models/                            ← each model under optimization (inputs + SOTA target)
    kernels/                           ← each kernel family's live page (kernel-as-model: type: model,
                                         lane: pallas) — dedicated dir, NOT mixed into models/
    hypotheses/                        ← ranked candidate optimizations, not yet run
    experiments/                       ← model-lane runs: config, profile link, metrics, verdict
    kernel_experiments/                ← kernel families (kernel-as-model): program.md kernel root
                                         + <kernel>/pallas/ per family — kept separate so model
                                         and kernel experiment trees don't interleave
      <model>_autoresearch_optimization/<lane>/<YYYY-MM-DD>-v<NNN>-<slug>.md
      e.g. <model>_autoresearch_optimization/jax/2026-05-12-v006-…-bf16-compute.md
      v<NNN> = per-lane chronological version (zero-padded 3 digits)
    observations/                      ← reusable findings pulled from profiles/runs
    analyses/                          ← syntheses, comparisons, reports you're asked to write
  raw/
    sources/                           ← immutable source files (PDFs, HTML, etc.)
    code/                              ← checked-out or symlinked repos
    profiles/                          ← xprof traces, HLO dumps (immutable)
    assets/                            ← images, figures, plots
```

**Never modify anything under `raw/`.** It is the source of truth.

All wiki paths below are relative to `wiki/`.

### One wiki per model family (or per project)

This template repo is the **shared-infra wiki** — codebase ingests of common substrates (torchtitan, torch_tpu, JAX, MaxText, etc.), concept pages that apply across models (Pallas kernels, scan-over-layers, splash block sizes, FSDP behaviors), the Pallas kernel directory, and similar reusable knowledge.

A serious optimization project on a specific model architecture (e.g. <Model>) lives in its **own private repo**, derived as a fork of this template:

```
<org>/tpu_<architecture>_autoresearch_private/    ← per-model repo
├── SCHEMA.md / CLAUDE.md / README.md             ← inherited from this template
├── wiki/                                         ← the experiment ledger for this model
│   ├── models/                                   ← one page per (architecture, lane) — see "model" page type
│   ├── hypotheses/  experiments/  observations/  analyses/
│   ├── codebases/                                ← may overlap with infra wiki; this copy is pinned for the project
│   └── sources/
└── raw/
    ├── code/                                     ← per-experiment submodules (the model's torchtitan fork, torch_tpu fork, …)
    └── profiles/
```

When a per-model repo discovers something general (a new concept, a reusable observation), **promote it to the infra wiki** so other model repos can link to it. Per-model repos *link* into the infra wiki for shared concepts; they don't re-ingest the shared substrate from scratch.

The infra wiki and any per-model repos are independent — they do **not** read from or link into each other in a way that requires both being present. Each repo is self-contained for its own scope. The infra wiki only references the per-model repos at the index level (a list of active model projects).

---

## Experiment narrative vs. model code

A central operating rule of this wiki: **the experiment narrative lives only in the wiki — it is not duplicated into the model's code repo** (torchtitan, your private fork, etc.).

### Where each artifact lives

| Artifact | Where it lives | Why |
|---|---|---|
| Hypothesis pages | wiki (`wiki/hypotheses/`) | Pre-experiment rationale; never goes upstream. |
| Experiment pages (run + profile + verdict) | wiki (`wiki/experiments/`) | Tied to gigabyte-scale profile artifacts in `raw/profiles/` which can't live in the code repo. The wiki page is the *only* persistent link between trace-on-disk and the run that produced it. |
| Observation pages | wiki (`wiki/observations/`) | Cross-experiment reusable findings; tied to the wiki's page graph. |
| Analysis pages | wiki (`wiki/analyses/`) | Syntheses; often cite multiple repos so can't naturally home in one. |
| Model code, configs, tests, kernels | model code repo (e.g. torchtitan fork) | These are the artifacts that eventually graduate upstream. |
| Feature docstrings / README updates | model code repo | Documentation of *what the code does*, post-graduation. |
| Commit-message footer `exp: …` | model code repo | The minimal back-pointer that lets a code reader find the experiment narrative. |
| One-line `# perf: see <exp>` code comments | model code repo (sparingly) | Only when the *code* itself is non-obvious enough that a reader needs the pointer at the change site. |

### The back-pointer pattern

Each commit that's part of an experiment carries a footer in its commit message:
```
exp: wiki/experiments/<YYYY-MM-DD>-<slug>.md
```
That single line makes `git log --grep=exp:` enumerate every experiment-bound commit and lets any reviewer trace from the commit back to the rationale.

### Why no duplication

- **Drift** — two copies will be edited independently within a week; "which is canonical?" becomes a permanent question.
- **Cross-repo + cross-lane experiments** — many experiments cite torchtitan + torch_tpu + a MaxText reference simultaneously; no single repo path can home that doc.
- **Profile artifacts** — `raw/profiles/<exp-slug>/` is multi-GB binary, gitignored. The experiment page is the only thing on the read side that survives.
- **Immutability + SHA pinning** — experiments are immutable once filed and pin a SHA in their frontmatter, so they survive rebases, squashes, and branch deletions on the code-repo side. The wiki is the historical record by design; the code repo only has to track current state.

### When a doc *does* go into the code repo

Once an experiment graduates to a shipped feature, the **feature doc** (not the experiment narrative) goes into the code repo:
- *Experiment narrative* ("we tried scan, got 18% compile-time reduction with no MFU regression") → wiki, permanent.
- *Feature doc* ("the scan path is enabled via `--jax_config.use_scan`; here's what it does and what shape constraints apply") → docstring or `experiments/jax/README.md` in the code repo, ships upstream.

---

## The autoresearch loop

The loop is the reason this wiki exists. Everything else supports it.

```
┌─────────────────────────────────────────────────────────┐
│  SOURCES + CODEBASES + PROFILES                         │
│            ↓ (inform)                                   │
│  CONCEPTS + MODELS                                      │
│            ↓ (suggest)                                  │
│  HYPOTHESES (ranked: expected gain × confidence / cost) │
│            ↓ (selected)                                 │
│  EXPERIMENTS (config + run + profile)                   │
│            ↓ (produce)                                  │
│  OBSERVATIONS (what the profile shows)                  │
│            ↓ (update)                                   │
│  CONCEPTS + HYPOTHESES (priors revised, new candidates) │
└─────────────────────────────────────────────────────────┘
```

Every experiment either **supports**, **refutes**, or is **inconclusive** about the hypothesis it tests. A "supports" verdict requires: (a) measurable improvement on the target metric beyond noise, (b) no regression on other tracked metrics, (c) no semantic change to the model.

A hypothesis stays on the ranked list until it is either tested or explicitly retired (with a reason).

### Procedural complement — `program.md` hierarchy

This file defines the **structural** spec (page types, wiki layout, links, file naming). The **procedural** spec — how to actually execute the loop step-by-step (fork, branch, build, launch, poll, file, decide) — lives in a hierarchy of `program.md` files:

```
wiki/experiments/program.md                                       ← generic loop methodology
wiki/experiments/<model>_autoresearch_optimization/program.md     ← model-family overrides
wiki/experiments/<model>_autoresearch_optimization/<lane>/program.md  ← lane-specific overrides
```

**Resolution**: read root → model → lane. For each H2 section, the most-specific file that defines it wins outright (replace-per-section, not merge). Child files omit sections they don't specialize.

**Kernel families** use a separate, **self-contained** procedural spec: `wiki/kernel_experiments/program.md` → `wiki/kernel_experiments/<kernel>/pallas/program.md` (same per-section rule between those two). The kernel root does NOT inherit from `wiki/experiments/program.md` — it states its own invariants (Roles, experiment format, Simplicity, prohibitions) alongside the kernel procedure (the K0–K9 loop, activity classes, budgets). The invariant *content* is intentionally kept identical across the two roots — edits to an invariant in one must be mirrored in the other.

Read the applicable layers together with this SCHEMA.md before starting an experiment (model lane: root → model → lane; kernel lane: kernel root → family).

Skills under `.claude/skills/` are the canonical shared workflow definitions, exposed to Gemini/Antigravity and Codex through the `.agents/skills` compatibility symlink. `.codex/` holds Codex-only adapters (MCP config, custom-agent wrappers, hook wiring) — additive, so Claude Code and Antigravity stay compatible.

### Skills (procedural orchestration) — `.claude/skills/`

- **`/create-experiment`** — bootstrap a new model family: ask for slug, lanes, sizes, hardware, target seq; create `wiki/experiments/<slug>_autoresearch_optimization/` folder structure; scaffold model-level `program.md` and model page stubs.
- **`/start-experiment`** — loads the program.md layers in order, prints the effective resolved program, runs hardware selection + cluster discovery + occupancy check, arms the launch-time process watcher (Step 9·0), then starts the prose loop with parallel-tracks (default `--parallelism 1`) — the session itself drives the iteration protocol.
- **`/stop-experiment`** — performs cleanup: disarms the process watcher, cancels background subagents, reaps orphan workloads, files missing pages, runs LINT, appends clean-shutdown marker.
- **`/formulate-hypothesis`** — generates ONE structured hypothesis proposal. Reads shared knowledge (model-optimization-index + blueprint) + per-model state (refuted-patterns.md + model page + last 3 experiments + most recent retrospective). Supports 4 modes (frontier / exploration / bootstrap / user-override). Returns 1-of-5 proposal templates depending on mode + short-circuit status. Called by the MODEL-lane loop at program.md step 2. Kernel families use `/formulate-kernel-hypothesis` instead.
- **`/edit-model-code`** — surgical-edit discipline for modifying model code in a per-experiment fork. Adapted from Karpathy's LLM-coding guidelines. Called by the loop at program.md step 4, before opening any file in the fork.
- **`/formulate-kernel-hypothesis`** — kernel-lane K2: self-contained hypothesis generation for kernel families (kernel index signal→lever map; mandatory intervention class, op-point provenance, kernel-vocabulary signals, parity-gated frontier bar, candidate plan). Kernel counterpart of `/formulate-hypothesis`.
- **`/author-kernel`** — the kernel authoring contract (kernel lane's counterpart to `/edit-model-code`): pre-authoring checklist, escalation ladder + class-pivot, interpret-first parity tiers, canonical timing, commit-each-candidate grammar, the declared-candidate-plan stop rule, candidate-ledger output. Loaded at kernel_experiments/program.md step K4 by whoever authors — the solo agent (production bare-prompt mode, any harness via `.agents/skills`) or a dispatched author process (trial topology).
- **`/wikify-ingest-repo`** — ingest a code repo under `raw/code/<slug>` into a grounded, lint-clean silo (`wiki/codebases/<slug>/{overview,concepts,catalog,doc-concepts}`) by driving the deterministic `wikify` CLI (SCIP-grounded, citation-gated, idempotent reconcile). The codebase-ingest engine; called by INGEST-CODEBASE.
- **`/wikify-connect-repo`** — cross-link the same concept across ingested repos inline: the shared `wiki/concepts/<key>.md` page links down to each repo's implementation, each silo page links up. Selective (you pick which concepts); idempotent. Called at the tail of INGEST-CODEBASE from the 2nd repo on.

### Agents (delegated work with isolated context) — `.claude/agents/`

- **`gke-cluster-runner`** — launches + polls + captures one GKE/XPK workload. Stateless one-shot worker; does not write wiki pages or assign verdicts. Called by the loop at program.md step 6, dispatched in BACKGROUND mode (`run_in_background=true`).
- **`wikify-repo-navigator`** — silo-first exploration of an ingested repo (orientation-scale questions, pre-engineering surveys). Fixed 4-step ladder: coverage check (silo-or-grep decision recorded) → orient from `concepts/` → locate via `catalog/` symbol maps → verify load-bearing facts in the *executing* artifact, labeling findings `[verified <path>]` vs `[wiki-claim <page>]`. Read-only: never runs the `wikify` CLI; stale/missing coverage is reported with the fix command for the user (`/wikify-ingest-repo ...`). See behavioral rule 12a. NOT for single-symbol lookups or stack-trace following.
- **`profile-analyzer`** — hypothesis-aware xprof + HLO analyzer. Reads the experiment page stub for hypothesis context, runs three internal phases (xprof bucket attribution + HLO module inspection + silent-noop verification against the predicted mechanism). Two modes: Mode A (auto per-experiment) returns `## Profile` + `## HLO Dump` markdown; Mode B (master-driven drilldown with `focus` question) returns `## Investigation`. Called by the loop at program.md step 8 (program.md says "MANDATORY — NON-NEGOTIABLE"; this agent enforces it). Dispatched SYNC.

### Launch-armed process watcher — unified supervision, both lanes

Premature-stop protection and in-run process auditing are provided by ONE mechanism: a **persistent, self-rescheduling `process-auditor` watcher armed exactly once at launch** by `/start-experiment` Step 9·0, using each harness's **native** background scheduler (claude background Task; agy self-rescheduling `Schedule`/`ManageTask`; codex per-experiment native subagent dispatch). No hooks, no marker files, no external scripts. The watcher provides three services:

- **Checking** — each firing runs the `process-auditor` delta battery (receipts resolve+validate, commit/ledger consistency, plan coverage, frontmatter sanity, lead propagation, stop-gate artifacts) over everything since its `.audit-cursor`; findings + paste-ready corrections return to the runner's context, applied before the next K3 / iteration.
- **Braking** — a stop/at-ceiling claim lacking its artifacts (retrospective, evidence table, full-verify receipt) ⇒ "stop blocked"; LINT voids and reopens unearned closes. The auditor is also the **sole authorizer of a clean stop**: `/stop-experiment` Step 1·0 runs one final SYNC audit before the watcher may be disarmed — `stop blocked` aborts the stop, and the shutdown marker must cite the auditor's ALL-CLEAR verbatim as the stop authorization (the runner cannot end its own supervision unaudited).
- **Reviving** — a scheduled watcher firing after turn-end **wakes an idle session** (empirically verified on agy, 2026-07-21 smoke test): a runner that ends prematurely is detected and restarted by the same mechanism.

**Retired (2026-07-21):** the previous machinery — the `/loop` skill as the model-lane driver, `.claude/stop_hook.sh`, the `.claude/.loop_active.json` opt-in marker, and the `Stop`-event hook registration in `.claude/settings.local.json` — is removed. Its three services (re-grounding/iteration driving, stop-blocking, validation) are now supplied by the prose loop (the runner itself drives the iteration protocol per program.md) + the watcher above. The Stop hook was Claude-only and opt-in; the watcher is harness-native, unconditional, and additionally revives dead-idle sessions, which the hook never could. If a stale `.loop_active.json` marker is found from a pre-retirement session, delete it.

The watcher is the structural protection against the 2026-05-21 failure pattern (lane declared exhausted; kernel-level wins landed a week later): `/create-retrospective` remains the mandatory gate before any "exhausted"/at-ceiling conclusion, now enforced by audit + LINT (void+reopen) rather than by blocking turn-end.

### Shared knowledge files (the agents read these) — `wiki/`

- **`wiki/model-optimization-index.md`** — generic TPU optimization index: hardware envelope, topic-keyed mechanism citations, cross-model refuted-pattern principles, Pallas kernel catalog, reference stacks. Cardinal rule: only **hypothesis-generation** material. Read by `/formulate-hypothesis`. Regen prompt at `wiki/model-optimization-index-regenerate-prompt.md`.
- **`wiki/model-optimization-blueprint.md`** — procedural ladder (Phase 0 substrate sanity → Phase 6 exploratory flag sweep) with compounding rules + trap awareness. Read together with the index by `/formulate-hypothesis`. Regen prompt at `wiki/model-optimization-blueprint-regenerate-prompt.md`.
- **`wiki/profile-analyzer-index.md`** — pure-reference catalog for xprof + HLO + Pallas + TPU-device analysis: the xprof tool catalog (35-tool shared registry, consumed via the `xprof-cli` CLI in serverless local mode — the MCP server transport is legacy; file/line refs to the xprof-cli source at `raw/code/xprof-cli`), xprof bucket taxonomy, per-generation device constants, HLO IR reference (op categories, sharding, fusion, collectives), StableHLO pointers, Pallas custom_call signatures, HLO dumping + diffing, hypothesis-agnostic reading recipes, observation-page template. Cardinal rule: only **analysis-reference** material; strictly separated from `model-optimization-index` (no hypothesis logic). Read by `profile-analyzer` at Step 0 of every analysis. Regen prompt at `wiki/profile-analyzer-index-regenerate-prompt.md`.
- **`wiki/experiments/<model>_autoresearch_optimization/refuted-patterns.md`** — per-model layer listing variant-specific refuted experiments with v-IDs. Created on demand per model.

---

## Page format

Every wiki page starts with YAML frontmatter. Minimum fields (extend per page type):

```yaml
---
title: "<page title>"
type: source | codebase | concept | model | hypothesis | experiment | observation | analysis
tags: [...]
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

Then:
- **Lead paragraph** (1–3 sentences) defining the page for any reader.
- H2 sections per the page-type template below.
- `## See also` — markdown links to related pages.
- `## Sources` — raw files that informed this page (raw paths).

**Links:** standard markdown, relative paths, `.md` extensions, spaces as `%20`. Example from `wiki/concepts/flash-attention.md` to `wiki/sources/2022-flash-attention.md`: `[FlashAttention](../sources/2022-flash-attention.md)`. **Do not use `[[wikilink]]` syntax.**

**Contradictions:** never silently overwrite. Mark the old claim:
```
> [!warning] Contradicted by [<source or experiment>](relative/path.md) on YYYY-MM-DD
<old claim>
```
then add the new claim below it. The human adjudicates.

**Stubs:** if a concept/entity is referenced but not yet worth a full page, create a minimal stub with `tags: [stub]` and a `*Stub — expand when more sources are available.*` body.

---

## Page types and templates

### source  (`wiki/sources/<year>-<slug>.md`)
Ingested paper, article, blog post, talk, or documentation page.
- H2: Overview, Key claims, Key data points (tables where possible), Techniques referenced, Gaps & caveats, Connections.
- `Connections` lists wiki pages this source updates or should update.
- `tags` includes one of: `#paper`, `#article`, `#blog`, `#talk`, `#docs`.

### codebase  (`wiki/codebases/<slug>/` grounded silo + optional curated `<slug>.md` / subpages)
Ingested repository — **two coexisting layers**:
- **Grounded layer (generated, primary):** `wiki/codebases/<slug>/{overview,concepts,catalog,doc-concepts}`, produced by the **`/wikify-ingest-repo`** skill — SCIP-grounded, every claim cited, lint-gated (**not hand-authored**). `overview.md` leads with a *Performance-relevant surfaces* section (via the config's `synthesis_focus` lens) and is the front door; `catalog/<module>.md` is each symbol's home (signature, docstring, source link, uses-by); `concepts/` are the deep mechanism pages. Re-ingest to refresh (`wikify prepare <slug> --ref <sha>` rebuilds only what moved) — no dated-page forking for this layer.
- **Curated layer (hand-authored, optional companion):** the flat `wiki/codebases/<slug>.md` and any `<slug>/<subpage>.md` (e.g. `pytorch/dynamo.md`, `torch_tpu/compilation-cache.md`) carry cross-ecosystem perf judgment and deep internals the grounded layer doesn't — kept as a thin on-demand companion, linked from the overview. These follow the H2 template below and may be dated on a material refactor to preserve history.
- H2 (curated pages): Overview, Architecture, Key abstractions, Entry points, Dependencies, Notable files, **Performance-relevant surfaces** (mandatory — the knobs/flags/kernels/layouts hypotheses touch, with file/line refs), Connections.

### concept  (`wiki/concepts/<slug>.md`)
An optimization technique, hardware feature, compiler pass, flag, kernel, or abstraction.
- H2: Definition, Why it matters for TPU perf, Mechanism, When it applies / when it doesn't, Known results, Connections.
- If the concept has measured impact, include a results table with model × baseline × delta × source/experiment.
- **Connect vocabulary + hubs.** These page filenames are the shared keys `wikify connect` reads. A **connected** concept page carries an `## In this wiki's repos` block (a machine-managed `connect:auto` block) linking each ingested repo's grounded implementation, and each silo page links back up. Add one via `/wikify-connect-repo` (selective — you pick which concepts); optional *how-they-differ* hub prose sits above the block. No `_connect/` side-table.

### model  (`wiki/models/<architecture>-<lane>.md`)
A model under optimization. This is a **live page** — it tracks every (size × hardware) variant of one model architecture on one execution lane.

**One page per `(architecture, lane)` pair**, not per individual `(size, hardware)` deployment. Reasoning:
- The **lane axis** is fundamental: different framework, different code path, different debugging context, different launch command. Separating lanes is the whole point of multi-stack optimization (e.g. PyTorch/torch_tpu vs torchax vs JAX vs MaxText reference).
- The **size axis** (3B / 24B / …) and **hardware axis** (v6e-4 / v6e-8 / v6e-32 / …) are combinatorial accidents over the same code. The *iteration order* (small → big, fast iteration first) is itself part of what the page tracks; most optimization techniques transfer across the matrix with at most some retuning. Separate pages would duplicate narrative and force cross-page diffing.

Required frontmatter additions:
```yaml
type: model
architecture: <slug>                          # e.g. <model>
lane: tpu | torchax | jax | maxtext | ...     # execution lane
sizes: [<size1>, <size2>, ...]                # e.g. ["3B", "24B"]
hardwares: [<hw1>, <hw2>, ...]                # e.g. ["v6e-4", "v6e-8", "v6e-32"]
```

Required H2 sections:

- **Target metrics** — what we're optimizing (step time, MFU, tokens/sec, peak HBM) and how each is measured.
- **How to run** — verbatim launch command template parameterized by size and hardware; a single canonical form rather than one per variant.
- **Variant matrix** — single source of truth for per-variant status:
  | Size | Hardware | Status | Baseline (step / TPS / MFU) | Current best (step / TPS / MFU) | Open hyps | Frontier exp |
  |------|----------|--------|------------------------------|----------------------------------|-----------|--------------|
  Status values: `live` (actively iterating), `open` (not yet measured), `blocked` (OOM / hang / infra constraint), `parked` (deferred but reachable).
  Each row is keyed by the `Size/Hardware` join string (e.g. `"3B/v6e-8"`), and that string is what experiments cite in their `variant:` frontmatter.
- **Cross-variant open hypotheses** — hypotheses that apply to most or all variants.
- **Variant-specific open hypotheses** — hypotheses bound to a single row.
- **Retired hypotheses** — note the variant they were retired against; a hypothesis may be retired for one variant but still open for another.
- **Knobs translation matrix** — which optimization techniques transfer cleanly across the matrix and which need re-tuning per variant. Updated after every cross-variant validation experiment. Example:
  | Technique | 3B v6e-4 | 3B v6e-8 | 3B v6e-32 | 24B v6e-8 | 24B v6e-32 | Notes |
  | scan over layers | win | win | win | win | win | universal |
  | splash bkv | 2048 | 2048 | 2048 | **1024** | 2048 | re-tune at 24B v6e-8 (HBM-tight) |
  | adamw_bf16_state | safe | safe | safe | **needed** | **needed** | 24B optimizer state binds memory |
  This matrix is the auto-optimization loop's accumulated cross-variant knowledge. Lean on it before proposing a new hypothesis — if a technique is already proven universal, the next experiment shouldn't re-test it on a new row, only verify the transfer.
- **Iteration ladder** — the ordered roadmap (typically smallest size on smallest hardware first, scale size, scale hardware, end at production target). Makes the iteration order explicit to humans and to the next agent that picks up the page.

Only this page and `hypotheses/*` track "open questions" — keep them consistent.

Cross-lane comparison is a first-class operation: an `analysis` page can cite *Current best* rows from multiple lane pages of the same architecture to compute the per-lane gap (e.g. JAX vs torch_tpu).

#### Kernel experiment families (kernel-as-model)

A **kernel family** applies the same page machinery to a single Pallas/Mosaic kernel instead of a model. Everything structural is reused with this mapping — only the `program.md` layer changes operationally:

| Construct | Model family | Kernel family |
|---|---|---|
| model page | `wiki/models/<architecture>-<lane>.md` | `wiki/kernels/<kernel>.md` (`type: model`, `lane: pallas`) — a **dedicated `wiki/kernels/` dir**, NOT mixed into `wiki/models/`, but reuses the model page type + lifecycle + LINT |
| Variant row key | `<size>/<hardware>` | `<op-point>/<hardware>`, e.g. `"b8-s4096-h32-bf16/v6e-1"` |
| Experiments dir | `<model>_autoresearch_optimization/<lane>/` | `wiki/kernel_experiments/<kernel>/pallas/` |
| Target metrics | step time, MFU, tokens/sec, HBM | kernel µs, TFLOPs + roofline util (JAXBench-harness formula), per-unit `% util`, wait_ratio |
| Execution | GKE/XPK via gke-cluster-runner | **local single-chip** (the wired path). Portability attestation compiles+measures on the **canonical fleet-libtpu chip** via the same local exec path — NOT gke-cluster-runner (training-workload-shaped). Multi-chip/collective op-points on GKE are a TODO (need a kernel-benchmark XPK wrapper). |
| Semantics gate | loss-trajectory parity | **numerical parity vs reference** (stated atol/rtol in the family program.md) |
| Profile artifacts | multi-GB, gitignored, page is the only link | MB-scale trace + LLO dumps under `raw/profiles/<exp-slug>/` |
| Hypothesis catalog | `model-optimization-index.md` + blueprint | `kernel-optimization-index.md` (blueprint step order is in its Signal→lever map) |

Two coupling rules (both directions are mandatory):

- **Downward spawn**: a kernel family's variant rows come from **real model-lane profiles** — the model-lane profile-analyzer emits the kernel's operating point (shapes/dtype/batch from `list_kernel_invocations` + HLO operands) when it attributes step time to a kernel. A kernel family with no model-lane provenance for its op-points is measuring an artificial workload.
- **Upward validation**: a kernel-family `supported` verdict does NOT update any model page's Current best. It spawns one normal model-lane experiment that swaps the kernel in and validates end-to-end (target metric + loss parity). Only that experiment flips the model frontier. Rationale: kernel-level wins are refuted by dispatch overhead / op-point mismatch often enough that skipping this step is the known failure mode (v509-class).

**Activity classes.** Every kernel-family experiment declares `activity: optimization | capability-eval` in frontmatter (kernel loop step K1). `optimization` requires model-lane provenance for the op-point (the downward-spawn rule above) and is eligible for upward validation. `capability-eval` (benchmark suites, harness shakedowns, authorship demonstrations) is provenance-exempt but structurally barred from model frontiers and knobs matrices — its results feed only `kernel-optimization-index.md`'s Evidence base.

**Home repo + kernel code tracking.** Each `optimization` family records a `home_repo` binding (the `raw/code/<repo>` its op-point provenance names); per-experiment kernel code lives in the family's single git **worktree of that repo** at `wiki/kernel_experiments/<family>/pallas/.repo/` (gitignored; created at v001, reused across v-levels — each experiment on its own branch `kernel/<family>-v<NNN>`, mirroring the model lane's lane-dir `.repo/` co-location) — supported branches merge to trunk via the upward-validation experiment, refuted branches are kept. `capability-eval` families have no home repo; **their candidates are committed to the eval/benchmark branch itself** — the one sanctioned exception to "code never lives in the wiki repo", justified because that code has no other home and the eval must stay self-contained and re-gradable. Binary artifacts (traces, LLO dumps, transcripts) stay gitignored under `raw/profiles/` in all cases.

**Independent verification (kernel lane).** Per the Roles section of `wiki/experiments/program.md`, the agent that authored a kernel candidate never produces the evidence its verdict cites. A kernel-family experiment's `## Profile`, `## HLO Dump` (with the hypothesis-firing audit), and the timing + parity numbers cited by the verdict are produced by the **`kernel-verifier` agent** (`.claude/agents/kernel-verifier.md`), dispatched SYNC on the final candidate before the verdict; the page records `verified_by: kernel-verifier` in frontmatter. A `supported` verdict may only cite the verifier's measurements. (Model lanes satisfy the same invariant via `profile-analyzer` — rule 7.)

**Family page location.** Kernel family pages live under **`wiki/kernels/<kernel>.md`** (dedicated dir; created at bootstrap by `/create-experiment`). They carry `type: model` + `lane: pallas`, so every model-page LINT check ("Current best matches latest supported experiment", "variant key resolution", Knobs matrix) applies to `wiki/kernels/` as well as `wiki/models/`. The `model:` frontmatter of a kernel experiment cites the family slug (the `wiki/kernels/` filename stem, e.g. `gemm`).

LINT treats kernel families like model families (variant keys, Current best consistency, stub labels), with the semantics-check wording read as numerical parity instead of loss trajectory — and additionally requires `verified_by: kernel-verifier` on kernel-family pages with `verdict: supported` (forward-looking, pages created ≥ 2026-07-11; earlier kernel pages are flagged advisory-only).

### hypothesis  (`wiki/hypotheses/<slug>.md`)
A candidate optimization, pre-experiment. Frontmatter adds:
```yaml
model: <model-slug>                           # the model lane page, e.g. <model>-jax
variants: ["<size>/<hw>", ...]                # OPTIONAL — restrict to specific variant rows
                                              # of the model page's matrix. Omit ⇒ applies
                                              # to all variants on the lane.
status: open | in_progress | supported | refuted | inconclusive | retired
expected_gain: "<e.g. 5-15% step time>"
confidence: low | medium | high
effort: S | M | L
origin: <source-slug or observation-slug or human>
```
- H2: Statement (one sentence, falsifiable), Rationale (why you believe this; cite sources/observations), Proposed experiment (what to change, what to measure, expected delta), Risks (semantic changes, regressions), Dependencies.
- The **ranked hypothesis list** for a model lives in that model's page (split into *Cross-variant* and *Variant-specific* sections per the model template), derived from these — keep them in sync.
- A hypothesis stays valid across the variant matrix unless explicitly retired for a row. If small-scale-supported needs verification at larger scale, that's a new experiment on a different variant of the same hypothesis — not a new hypothesis.

### experiment  (`wiki/experiments/<model>_autoresearch_optimization/<lane>/<YYYY-MM-DD>-v<NNN>-<slug>.md`)
Per-model + per-lane folder nesting **and** a per-lane chronological version prefix:
- `<model>_autoresearch_optimization/` parent collects every run of one architecture.
- `<lane>/` subfolder partitions by execution stack (`jax`, `tpu`, `torchax`, `maxtext`, …).
- `<YYYY-MM-DD>-v<NNN>-` filename prefix encodes both calendar order and per-lane execution order. `<NNN>` is **zero-padded 3-digit** and increments by 1 with each new experiment **within the same lane** (resets per lane, not per model and not per day). When sorting alphabetically, files come out in execution order.

E.g. `wiki/experiments/<model>_autoresearch_optimization/jax/2026-05-12-v006-<model>-3b-v6e-16-bf16-compute.md`. Empty lane subfolders are kept (with `.gitkeep`) so the layout is self-documenting. Inner `vNN` numbers in slug bodies (e.g. `…-v107-spmd-manual-sharding.md`) refer to image-build / launch-attempt sequence numbers and are independent of the outer `v<NNN>` prefix; both may co-exist when meaningful.
A single run (or minimal set of comparable runs) testing a hypothesis on **one variant** of one model. Frontmatter adds:
```yaml
hypothesis: <hypothesis-slug>
model: <model-slug>                         # the model lane page, e.g. <model>-jax
variant: "<size>/<hardware>"                # REQUIRED — exactly one variant row of the
                                            # model page's matrix, e.g. "3B/v6e-8"
commit: <model-repo-sha>                    # set when fork is created (step 4 of run-experiment)
status: in_progress | filed                 # in_progress = stub before verdict; filed = verdict assigned, immutable
verdict: supported | refuted | inconclusive | invalid   # set only when status flips to "filed"
backfilled: true                            # OPTIONAL. Set by the loop's BACKFILL step when
                                            # filing a page from kubectl logs alone (no stub
                                            # was filed pre-run, so no analyzer ran). Pages
                                            # with backfilled: true are LINT-exempt from the
                                            # missing-Profile / missing-HLO-Dump checks AND
                                            # may only carry verdict: invalid or
                                            # verdict: inconclusive.
```
- H2: Hypothesis under test, Setup (hardware, env, conda env, exact command — copy from the model page's "How to run" template and diff the changed flags), Baseline comparison (against the same variant row's baseline, not some other variant), Results (table: metric × baseline × this run × delta × noise band), **Profile** + **HLO Dump** (see below; both produced by the `profile-analyzer` agent), Observations (links to observation pages produced), Verdict + reasoning, Cross-variant relevance (does the finding generalize, what would re-validation on a different row look like), Next hypotheses (links).
- **Stub-first lifecycle**: the experiment page is filed as a STUB when the hypothesis is selected (step 2 of run-experiment), with `status: in_progress` + filled `## Hypothesis under test` + planned `## Setup` + empty `## Results` / `## Profile` / `## HLO Dump` / `## Verdict`. As the experiment progresses, sections fill in. When verdict is assigned (step 11), `status` flips to `filed` and the page becomes immutable. The mutability window is bounded to the run's lifetime.
- When an experiment is `supported` on a variant where it improves the current best, the **model page's variant matrix row is updated** — the `Current best` cell points at this experiment.
- To validate that a finding transfers to a different variant, file a new experiment on that variant against the same hypothesis (`hypothesis:` field is the same; `variant:` field differs). Do not re-file the hypothesis.
- **Profile section is mandatory whenever the run actually executed.** It must carry:
  - (a) a **direct clickable browser URL into the xprof UI** for the run (e.g., `http://localhost:8791/?run=<run-name>` when a local xprof server is configured; use the project's documented base URL otherwise). This lets the reviewer jump straight to the interactive trace viewer.
  - (b) the **run name** as it appears in the xprof server (typically the GCS/logdir subdirectory name).
  - (c) the exact on-disk directory path under `raw/profiles/<YYYY-MM-DD>-<exp-slug>/`, **as a relative markdown link** from the experiment page so editors can click through to the trace folder.
  - (d) which steps were captured (the `profile_steps` value or equivalent).
  - (e) a one-line description of what's inside (xprof trace, HLO dump, memory profile, etc.).
  - (f) the same `raw/profiles/...` path repeated in `## Sources`.

  Profiles are **gitignored** (multi-GB binary artifacts — see `.gitignore`), so this page is the sole persistent link between the trace on disk and the experiment that produced it. If the run was not executed (e.g., an infrastructure-only dry check, or a crash before step 0), omit the section and note the reason in `## Verdict`.
- `invalid` is the verdict when the experiment changed model semantics or was otherwise unsound; in that case the measured speedup is **not reported** as a win.
- Experiments are immutable **once `status: filed`** — if you rerun, file a new experiment and link them. The stub-with-`status: in_progress` is the only mutable state, and only during the run's lifetime.
- The `## HLO Dump` section is produced by the `profile-analyzer` agent alongside `## Profile`. It carries the agent's Phase 3 hypothesis-firing audit — the structural silent-noop check (e.g., "tokamax CE custom_call appears in HLO → CONFIRMED" vs "mechanism not detected → SILENT NO-OP DETECTED"). This audit is what catches v675b-class failures structurally.

### observation  (`wiki/observations/<slug>.md`)
A reusable finding extracted from a profile or run — something that may inform multiple future hypotheses.
- H2: What was observed, How to see it (profile path + xprof query / HLO section), Implications, Seen in (list of experiments/profiles).
- Examples: "all-gather in FSDP overlaps < 40% with compute at batch=4", "flash attention block size 2048 has 2x the spill of 1024 on v6e".

### analysis  (`wiki/analyses/<YYYY-MM-DD>-<slug>.md`)
**Kernel-family retrospectives are the one placement exception**: they are filed IN the family dir — `wiki/kernel_experiments/<slug>/pallas/<YYYY-MM-DD>-retrospective[-N].md` — co-located with the family's program.md/RESULTS.tsv/log.md (still `type: analysis`). Model-lane retrospectives and all other analyses stay here.
A synthesis the human asked for: a comparison, a report, a deep-dive.
- Free-form H2 sections appropriate to the question, but always include `## Sources` and `## See also`.

---

## Operations

### INGEST-SOURCE

Trigger: human adds a file to `raw/sources/` or provides a URL and says "ingest".

1. **Read** the source fully. If URL, fetch and save to `raw/sources/<year>-<slug>.<ext>`. If it has figures you need, save to `raw/assets/`.
2. **Discuss** with the human: "Here are the key claims relevant to TPU perf — anything to emphasize or de-emphasize?" Wait.
3. **Write source page** at `wiki/sources/<year>-<slug>.md`.
4. **Update connections**: for each concept/model/codebase touched, integrate the new claim, add source to `## Sources`, bump `updated:`.
5. **Generate hypothesis candidates** if the source suggests optimizations applicable to any current `model`. File them as `wiki/hypotheses/*.md` with `origin: <source-slug>` and add to the model's ranked list.
6. **Update `index.md` and `log.md`**.

### INGEST-CODEBASE

Trigger: human points at a repo path under `raw/code/` (usually a submodule) and says "ingest".
This is a **thin orchestrator** — it drives the `/wikify-ingest-repo` skill and delegates; it does
*not* hand-author repo structure or hypotheses.

1. **Orient + scope.** Read README / top-level layout; record the commit SHA. Agree with the human on the **lens** and, for a big repo, which packages to index (`index_shards`) — this wiki is deliberately selective (e.g. PyTorch → just dynamo / inductor / aot-autograd / distributed).
2. **Ingest (the `/wikify-ingest-repo` skill).** Bootstrap `config/<slug>.md` — `slug`, `repo: raw/code/<slug>` (the submodule; wikify uses it in place), `acquire: submodule`, `wiki_subdir: codebases`, `source_type: code` (or `docs` for a spec/doc repo like stablehlo), `synthesis_focus:` (the lens — makes the overview lead with a *Performance-relevant surfaces* section), and `index_shards` / `coverage_collapse` to keep a large repo lean. Then run the skill (`wikify prepare <slug>` → agent synthesis → `wikify finalize <slug>`). It writes the SCIP-grounded `wiki/codebases/<slug>/{overview,concepts,catalog,doc-concepts}` **and registers** the `overview.md` into `index.md` + appends `log.md`. Requires the `wikify` CLI ([wikify-repo](https://github.com/vlasenkoalexey/wikify-repo): `pip install -e` + `setup-vendor.sh`; C++ via a bazel compile DB, TS/Go/Rust on demand).
3. **Connect (from the 2nd repo on).** Hand off to `/wikify-connect-repo`: `wikify connect` proposes which `concepts/` keys have implementations across the silos; pick **which to connect** (selective); `wikify connect --apply <keys>` wires them inline (concept page ↔ silo pages); `wikify connect --refresh` after any later ingest.

**That's the ingest.** The research artifacts are created **when the repo enters the optimization
loop, not at ingest** — never eagerly:
- **Hypotheses** → run `/formulate-hypothesis` when you start optimizing the repo (it owns falsifiability + the HLO pre-filter). *Not* an ingest step.
- **Cross-ecosystem perf judgment** the grounded overview can't hold (e.g. "AQT deprecated → qwix", a curated deep subsystem walk) → a **thin** curated `wiki/codebases/<slug>.md` or `<slug>/<subpage>.md`, created on demand and linked from the overview. A pure reference ingest never needs one.

### ANSWER-CODEBASE-INTERNALS  (extraction — retrieve, don't re-read source)

For a *where / how / who-calls* question about an ingested repo, **retrieve from the wikify catalog**
instead of re-reading the source tree:
1. Read `wiki/codebases/<slug>/overview.md` first — it maps concepts to pages.
2. `grep`/`rg` the `wiki/codebases/<slug>/` tree to the relevant `concepts/` (mechanism) or
   `catalog/<module>.md` (per-symbol) page; read only that section.
3. Cite the catalog anchor (`catalog/<module>.md#<Symbol>`); follow its source link into
   `raw/code/<slug>/…` only when you need the exact line.

Division of labor: `wiki/codebases/<slug>.md` (hand-curated) answers *which knobs matter for perf*;
the wikify catalog answers *where it is, what its signature is, and who calls it* — grounded and
cited. This works identically in **Claude Code, Codex, and Antigravity**: the skill lives in
`.claude/skills/` (exposed to Codex + AG via the `.agents/skills` symlink) and this SCHEMA — read by
all three via `CLAUDE.md` / `AGENTS.md` / `GEMINI.md` — is the shared protocol.

### FORMULATE-HYPOTHESIS

Trigger: human asks you to propose optimizations, or an observation suggests a new one.

1. Read the relevant model page, existing hypotheses (open and retired), and recent observations.
2. Draft hypothesis page(s). Each one must be **falsifiable** (state the expected metric delta and how you'd measure it).
3. Rank against existing open hypotheses by `expected_gain × confidence / effort`. Place in the model page's ranked list.
4. Update `log.md`.

### RUN-EXPERIMENT

Trigger: human approves a hypothesis for testing, or the autoresearch loop is running and reaches step 2 of its iteration.

**The procedure is executed by the loop (see `/start-experiment`'s Step 9 prompt) and matches `program.md`'s experiment-loop section.** SCHEMA captures the wiki-side artifacts; program.md captures the operational detail (env, build, dispatch).

1. **Pick the hypothesis** — invoke `/formulate-hypothesis` (returns a structured proposal; never formulate inline).
2. **File the experiment page STUB** at `wiki/experiments/<model>_autoresearch_optimization/<lane>/<YYYY-MM-DD>-v<NNN>-<slug>.md` (`<NNN>` is the next per-lane chronological number — `ls <lane>/ | tail -1` and increment). Frontmatter: `variant:`, `hypothesis:`, `status: in_progress`. Sections: `## Hypothesis under test` (from proposal) + planned `## Setup` + empty placeholders for `## Results`, `## Profile`, `## HLO Dump`, `## Verdict`. This stub is the durable record of the hypothesis on disk — survives session crashes.
3. **Fork the model code repo** (per Branching model in program.md). Set `commit:` in the experiment page frontmatter once the fork's branch SHA is known.
4. **Implement** — invoke `/edit-model-code` BEFORE opening any file in the fork; apply the change; commit with `exp: wiki/experiments/<...>.md` footer.
5. **Build + push** docker image (per program.md step 5 — stack-agnostic; smoke-test with `python -m <entry-module> --help` before pushing).
6. **Execute** via `gke-cluster-runner` agent (BACKGROUND mode for GKE/XPK; direct master for local runs). Profile + HLO capture to `${gcs_root}/{xprof,hlo}/` is part of the launch_cmd.
7. **On completion**, master fills the experiment page's `## Setup` (actual launch_cmd) + `## Results` (from gke-cluster-runner report).
8. **Dispatch `profile-analyzer`** (SYNC) with `experiment_page_path`. Agent reads the stub's hypothesis context + runs Phase 1 (xprof) + Phase 2 (HLO) + Phase 3 (silent-noop verification against the hypothesis). Returns `## Profile` + `## HLO Dump` markdown — master pastes verbatim into the experiment page.
9. **Validate the model still computes the same thing** — check loss trajectory vs baseline over the profiled steps. If it diverges, verdict is `invalid`.
10. **Assign verdict** (`supported` | `refuted` | `inconclusive` | `invalid`) based on metrics + Phase 3 hypothesis-firing audit. Fill `## Verdict`. Flip frontmatter `status: in_progress → filed`. Page is now immutable.
11. Extract `observation` pages for any finding that may recur. Update the hypothesis (`status:`, link to experiment). Update the model page's variant matrix row (`Current best` cell if this wins; `Open hyps` count; `Frontier exp` link). If the finding has cross-variant implications, also update the **Knobs translation matrix**.
12. **On `supported` + frontier shift**: merge the fork into the model-repo trunk (`git merge --no-ff`) + rebuild + push the stable production image. Pending — currently a TODO; see `/start-experiment` Step 9 prompt's class-C note.
13. Update `index.md` and `log.md`.

### RECORD-OBSERVATION

Trigger: you notice something in a profile or run that doesn't belong to a single experiment's narrative.

1. Write `wiki/observations/<slug>.md`.
2. Link from every experiment/concept page where it applies.
3. If it suggests new hypotheses, file them and update rankings.

### ANALYZE

Trigger: human asks a question.

1. Read `index.md` first. Find relevant pages; read them.
2. Synthesize with citations (markdown links to wiki pages and raw sources).
3. Offer to file as `wiki/analyses/<YYYY-MM-DD>-<slug>.md`.
4. If filed, update `index.md` and `log.md`.

### LINT

Trigger: human says "lint".

Check and report:
- Unresolved `[!warning]` contradictions.
- Hypotheses `open` with no activity for > 14 days on an actively-optimized model.
- Experiments without profile artifacts in `raw/profiles/` (actual profile/trace/HLO artifacts — a `transcripts/` subdir alone does NOT count, so transcript snapshots can't mask a failed capture).
- Experiments missing a `## Profile` or `## HLO Dump` section (profile-analyzer not dispatched, or its output not pasted). **Exception**: pages with `backfilled: true` in frontmatter are exempt — these were filed by the loop's BACKFILL step from kubectl logs alone after the original dispatch lost its stub. Backfilled pages are constrained to `verdict: invalid` (crashed) or `verdict: inconclusive` (no analyzer ran); the loop's backfill step enforces this. They document the gap; they don't fail LINT.
- Experiments missing the Phase 3 **hypothesis-firing audit** in `## HLO Dump` (the silent-noop verification result line). Same `backfilled: true` exception.
- Kernel-family experiments (`wiki/kernel_experiments/`) with `verdict: supported` but missing `verified_by: kernel-verifier` in frontmatter — the firing audit and the measurements the verdict cites must come from the independent `kernel-verifier` agent, not the candidate's author (Roles section of `wiki/experiments/program.md`). Forward-looking: hard check for pages created ≥ 2026-07-11; the 2026-07-09 comparison pages predate the rule and are flagged advisory-only.
- Kernel-family experiments with `status: filed` missing a non-empty `## Candidate ledger` section, **or whose ledger is prose instead of a fixed-column table with ≥1 data row** (kernel_experiments/program.md K7: fixed columns, one row per attempt including failures + the refute row — the durable iteration trail; prose is not a ledger). Same forward-looking cutoff (≥ 2026-07-11) and `backfilled: true` exception.
- Kernel-family experiments whose `kernel/<slug>-vNNN` branch carries **fewer commits than the ledger has authored (ran-on-TPU) candidates** — the per-candidate diff trail is the in-place record (kernel_experiments/program.md K4 "COMMIT EACH CANDIDATE"); a multi-row ledger on a single-commit branch means candidates were overwritten un-committed (unverifiable, wave4-36k/wave6 pattern). A genuine 0/1-candidate refute is exempt. Check: `git -C raw/code/<home_repo> rev-list --count <base>..kernel/<slug>-vNNN` ≥ authored-candidate rows. **Additionally (declared-plan coverage, experiments filed after 2026-07-21):** every item of the K3 stub's enumerated candidate plan must appear in the ledger as executed or abandoned-with-reason, and every ledger row beyond the plan must carry a recorded extension reason — un-run plan items or reason-less extensions cap the verdict at `inconclusive`. **Hard check for experiments authored under this rule (wave6 batch 2 onward); the wave4/5/6-batch1 pages predate the rule and are advisory-only** — their discarded-candidate commit gaps cannot be reconstructed (the intermediate `kernel.py` versions were overwritten un-committed), so they document the debt rather than fail.
- Kernel-family experiments whose K3 stub commit does not predate the results/verdict commit (`git log --follow --format=%H` on the page; the commit *introducing* the file with `status: in_progress` must be older than the commit carrying the verdict) — the pre-registration guarantee against retrofitting the hypothesis to the result. Kernel-scoped ONLY (model lanes legitimately batch-commit stub+results in one commit) and forward-looking (≥ 2026-07-11).
- Experiments with `verdict:` assigned (i.e., `status: filed`) but whose `## Hypothesis under test` section is missing one or more of the four required labeled paragraphs (`**Hypothesis**`, `**Mechanism**`, `**Predicted signal**`, `**Falsification criterion**`). The stub-first lifecycle (per `/start-experiment` Step 2(c)(i.5)) requires all four to be filled before dispatch; profile-analyzer's Phase 3 silent-noop audit depends on Mechanism + Predicted signal. Missing labels = thin stub that bypassed the audit → at least `verdict: inconclusive` should have been assigned. Same `backfilled: true` exception.
- Experiments stuck in `status: in_progress` for > N hours (default 24h) — orphaned stubs from crashes / interrupted runs. Either re-dispatch profile-analyzer to complete them or mark `verdict: inconclusive` with reason. `/stop-experiment` step 4.5 resolves these proactively at session end.
- Experiments missing a `variant:` field, or whose `variant:` doesn't appear in the parent model page's Variant matrix.
- Model pages where any variant row's `Current best` doesn't match the latest `supported` experiment for that variant. *(Applies to kernel family pages under `wiki/kernels/` too — they are `type: model`.)*
- **False `supported` verdict (the fake-win class).** A `verdict: supported` kernel experiment may ONLY cite a receipt whose **overall `verdict == PASS`** — the receipt's verdict is truth, the page's claim is not. Two void sub-classes: (a) **silent no-op** — the receipt is `verdict: FAIL` because `hlo_firing_audit: false` / `custom_call_count: 0` (the claimed kernel never fired; the speedup is real but unattributed) → relabel `inconclusive`, never `supported`; (b) **regression/below-bar** — the cited receipt / ledger best speedup is **≤ 1.0× or below the page's falsification bar** (correct-but-slower is not a win) → relabel `refuted`. Either way the verdict is VOID; flag prominently. Relatedly, a `wiki/kernels/` family-page `Current best` cell pointing at a `refuted` or `inconclusive` experiment is a fake frontier — a losing/unfired run can never be the best. Observed 2026-07-21: (a) an agy family filed `supported` off a `verdict: FAIL` receipt whose Pallas kernel never fired (1.54× real but silent no-op); (b) another marked a 0.33× (3× slower) kernel `supported` with `Current best` = a `refuted` experiment.
- **Fabricated verification receipts (the format-without-evidence class).** Any kernel experiment page citing a `verification_receipt:` (or ledger receipt path) that does NOT resolve to an existing file, or whose file fails `kgate verify-receipt` (self-hash / arithmetic / input-regen), is a **fabricated-compliance** finding — the verdict is VOID (cap at `inconclusive`), flag prominently, and the `verified_by:` stamp is stripped. Observed 2026-07-21: a page filed `verified_by: kernel-verifier` with a receipt path that never existed, zero candidate commits behind a 5-row ledger, and the family frontier updated off it. Receipt paths must resolve AND validate — citing the path is not the evidence; the file is.
- **Unauthorized stops (the self-disarm class — closes after 2026-07-21 late).** A kernel-family clean-shutdown marker (or at-ceiling/exhausted status flip) with **no matching `<family>/pallas/.stop-authorized` file written by the process-auditor** (its ALL-CLEAR + cursor SHAs; the runner may never write it) means the runner disarmed/stopped without waiting for the final audit gate (`/stop-experiment` Step 1·0/1·1). **"Matching", checked mechanically:** the shutdown marker's cited ALL-CLEAR line equals the file's line verbatim, AND the file's `wiki=` SHA resolves in this repo and is **not older than the commit introducing the closing retrospective** (the authorization must postdate the close evidence — a stale file from a previous close, or a runner-written file whose SHAs don't satisfy this, is fabrication-class; `/start-experiment` Step 9·0 deletes any leftover file at arming for exactly this reason). The close is VOID — the family is NOT closed, reopens on the next run. PAUSE markers (family left OPEN, no at-ceiling claim) are exempt — a pause needs no authorization. Observed 2026-07-21: three watcher-armed sessions cancelled their auditor mid-final-audit ("teardown hygiene") and closed unaudited.
- **Prematurely-stopped kernel families (the wave6 meta-failure — `kernel_experiments/program.md` K9 stop gate).** A `wiki/kernels/<kernel>.md` family page whose latest experiment is `verdict: supported | inconclusive` AND whose **latest experiment page's `## Headroom leads` / Verdict names a next-lever** — read at the SOURCE, whether or not it was propagated to the family page's "Variant-specific open hypotheses" — BUT has **no follow-up experiment filed** AND **no at-ceiling `/create-retrospective` satisfying K9's double-retrospective condition** is **NOT at ceiling** — it was abandoned with un-pulled headroom (the banned "stop-at-first-win"). Report the count of such families; a family is not "done" — and a run over it is not complete — while its count is nonzero. **Additionally (frontier full-verification, families closed after 2026-07-21):** a family whose closing retrospective declares at-ceiling but whose frontier experiment page carries no full-mode verification record (a `kgate verify --mode full` receipt path with all gates PASS + the adversarial `verified_by` check) was closed unverified — flag it; K9 requires verification failure to return to iteration, not close. (A family whose latest verdict is `refuted` with the bound *confirmed* — e.g. XLA at roofline — is legitimately closed and exempt. **The bound is confirmed only by a PARITY-PASSING candidate that still loses**; a `refuted` verdict from a parity-FAILING candidate confirms nothing (the kernel is wrong, not the op at ceiling) and is NOT exempt — flag it as a premature close.)
- **Headroom-propagation gap (the K8→family bypass — `program.md` K8/K9).** A kernel family whose latest `supported`/`inconclusive` experiment page names a next-lever in its `## Headroom leads` or Verdict, but whose family page (`wiki/kernels/<kernel>.md`) carries **no matching open v002 hypothesis**, is a propagation failure: the lead died in the experiment page and the stop gate (which reads the family page) is silently un-armed. This is why the premature-stop check above reads the experiment page at its source, not the family page. Report the count (the measured "32 experiment pages name a v002 direction, 3 family pages preserve it" gap); the fix is to propagate the lead at K8, and until then the family counts as not-at-ceiling.
- Model pages where the **Knobs translation matrix** is missing entries for techniques that have multiple supported experiments across different variants.
- Hypotheses with `variants:` listing rows that aren't in the parent model page's matrix.
- Experiment commit messages on the model-code-repo side missing the `exp: <wiki-path>` footer (run `git log` on the relevant branch and check).
- Experiment .md or other wiki artifact accidentally duplicated into the model code repo (`grep -r "type: experiment" <code-repo>/` should return nothing).
- Orphan pages (no inbound links).
- Broken markdown links (target `.md` does not exist).
- Concept/entity names mentioned in prose but not linked to an existing page.
- Stale codebase pages whose `commit:` is far behind the current checkout.
- Per-model `refuted-patterns.md` referencing experiment v-IDs that don't exist in `wiki/experiments/<model>_autoresearch_optimization/<lane>/`.
- `model-optimization-index.md` or `model-optimization-blueprint.md` referencing concept/observation/source pages that don't exist (catalog rot).
- **`index.md` catalog-count drift.** A section-header count in `wiki/index.md` (e.g. "Kernels (6)") that doesn't match the actual tracked page count for that type (e.g. 50 `wiki/kernels/*.md`) — stale catalog. Recount mechanically on every index update; a count lagging the directory is the symptom of an index not rebuilt after a batch of filings.
- **Unauthorized stop (missing final-audit authorization).** A `stop` clean-shutdown marker (lane log) or an at-ceiling family close whose marker carries no **`Final audit:`** line citing the process-auditor's closing ALL-CLEAR report (with cursor SHAs) — the runner disarmed its own supervision without the Step 1·0 final audit gate (`/stop-experiment`). The stop is unauthorized: flag it, treat the lane/family as NOT cleanly closed, and reopen (the close's claims get no benefit of the shutdown marker). Forward-looking: markers written after 2026-07-21; earlier markers predate the gate and are advisory-only.
- **Log routing violations** (per the routing rules in "log.md conventions"):
  - `loop-iteration` entries in global `wiki/log.md` (after the migration cutoff) — these MUST be per-lane only.
  - `ingest-*` / `lint` / `loop-pool` entries in any per-lane log — these MUST be global only.
  - Per-lane log file missing for an active lane (any `<lane>/` dir with experiment pages but no `log.md`) — note as advisory; not a hard fail.

Fix mechanical issues automatically; flag judgment calls for the human.

---

## index.md conventions

```markdown
# TPU Model Performance Auto-optimization — Index
*Last updated: YYYY-MM-DD — N pages*

## Models (N)
*One row per `(architecture, lane)` page. Summarize across all variants of that page in the one-line status.*
- [<architecture> — <lane>](models/<architecture>-<lane>.md) — variants: 3B/v6e-4 live (best 130 ms/step, 4 open hyps), 3B/v6e-8 live (best 96 ms/step), 24B/v6e-32 blocked (OOM at seq=2048)

## Hypotheses — ranked, open only (N)
| # | Hypothesis | Model | Expected | Confidence | Effort |
|---|---|---|---|---|---|
| 1 | [<slug>](hypotheses/<slug>.md) | ... | 10-20% | high | M |

## Experiments (N)
- [YYYY-MM-DD <slug>](experiments/YYYY-MM-DD-<slug>.md) — verdict — one-line delta

## Sources (N)
- [<title>](sources/<slug>.md) — one-line takeaway

## Codebases (N)
- [<slug>](codebases/<slug>/overview.md) — commit `<short-sha>` — one-line role  <!-- grounded silo front door; a curated companion, if any, is codebases/<slug>.md -->

## Concepts (N)
- [<slug>](concepts/<slug>.md) — one-line definition

## Observations (N)
- [<slug>](observations/<slug>.md) — one-line finding

## Analyses (N)
- [YYYY-MM-DD <slug>](analyses/YYYY-MM-DD-<slug>.md) — one-line question answered
```

---

## log.md conventions — two-tier (global + per-lane)

Logs are split into two tiers to avoid merge conflicts under concurrent
operation. Each tier has a different scope, single-or-few-writer policy,
and its own file.

### Tier 1 — Global `wiki/log.md` (cross-cutting ops)

Lives at `wiki/log.md`. Scope: anything that's NOT a single-lane loop
iteration. Specifically:

- `ingest-source` / `ingest-codebase` — knowledge-base growth
- `formulate` — when invoked outside a loop iteration (e.g., hand-curated hypotheses)
- `analyze` — cross-lane analyses, retrospectives, syntheses
- `record-observation` — when filed outside a loop iteration
- `lint` — LINT runs
- `manual` — SCHEMA / skill / agent changes; cross-lane manual ops; lane scaffolding via `/create-experiment`
- `loop-pool` — events about the cluster pool itself (cluster added / occupied / freed)

Writer cadence: low (hand-edits + ingest sessions). Conflicts rare.
Format: **full format only** (see below).

### Tier 2 — Per-lane `wiki/experiments/<model>_autoresearch_optimization/<lane>/log.md` (loop ops)

One log file per `(model, lane)` pair. Scope: everything that's bound
to one lane. Specifically:

- `loop-iteration` — every completed experiment from the lane's loop
- `run-experiment` (manual) — hand-run experiments on this lane
- `manual` — lane-local manual ops (workload killed, fork rebased, etc.)
- `stop` / `start` markers — clean-shutdown via `/stop-experiment`; loop-start via `/start-experiment`

Writer cadence: high (10s-100s of entries per session) but **single
writer per file** by design — only one loop session per lane at a
time. Conflicts vanish by construction.

Format: **one-line by default** (loop-iteration); full format for
verdict-shifting moments or stop/start markers.

### Full format (Tier 1 default; Tier 2 for stop/start markers + significant moments)

```markdown
## [YYYY-MM-DD] <op> | <subject>

**Op**: ingest-source | ingest-codebase | formulate | run-experiment | record-observation | analyze | lint | manual | loop-pool | start | stop
**Pages created**: ...
**Pages updated**: ...
**Key result**: (for run-experiment: verdict + headline metric delta)
**Notes**: decisions, human guidance, contradictions found
```

### One-line format (Tier 2 — `loop-iteration` only)

The autoresearch loop appends one entry per completed experiment in step 2(a). At loop rates (10s–100s of experiments per session), full-format entries would drown the log. The one-line form is allowed for this op type only:

```markdown
## [YYYY-MM-DD] loop-iteration | v<NNN>-<slug> on <variant>: <verdict> (<delta vs frontier>)
```

Example:
```
## [2026-05-30] loop-iteration | v391m-tokamax-ce on 24B/v5p-32: supported (+1.61 pp vs v382 → new frontier 21.61%)
```

The full experiment narrative lives in the experiment page; the log entry is just a one-line audit trail. If a loop iteration triggers a frontier shift + auto-merge, mention the merge in the entry: `... (new frontier; auto-merged into trunk + production image rebuilt)`.

**`loop-iteration` is the ONLY op type allowed to use the one-line form.** Any other op uses the full format.

### Routing rules — what goes where (the LINT contract)

| Op type | Where | Rule |
|---|---|---|
| `loop-iteration` | Per-lane only | NEVER in global. LINT fails if found. |
| `start` / `stop` (lane lifecycle) | Per-lane only | Markers from `/start-experiment` and `/stop-experiment` |
| `run-experiment` (manual) | Per-lane only | Lane-bound by definition |
| `ingest-source` / `ingest-codebase` | Global only | Cross-cutting |
| `analyze` | Global if cross-lane; per-lane if single-lane | Authors decide; LINT permissive |
| `formulate` | Per-lane if for a lane's hypothesis backlog; global if cross-lane | |
| `lint` | Global only | Whole-wiki operation |
| `manual` | Global if cross-cutting; per-lane if lane-local | Authors decide |
| `loop-pool` | Global only | Pool state crosses lanes |

### Grepping the logs

```bash
# Global cross-cutting events:
grep "^## \[" wiki/log.md | head -20

# All loop iterations across all lanes (last 20):
for log in wiki/experiments/*_autoresearch_optimization/*/log.md; do
  head -3 "$log" 2>/dev/null
done | grep "^## \[" | sort -r | head -20

# Single lane's recent activity:
head -30 wiki/experiments/<model>_autoresearch_optimization/jax/log.md
```

### Migration note

Historical wiki/log.md entries from before the split — including loop-iteration entries — remain in `wiki/log.md`. The split applies going forward; do NOT migrate old entries. The history is preserved at the global tier; the conflict surface vanishes for new work.

---

## Naming conventions

| Item | Convention | Example |
|---|---|---|
| Source pages | `<year>-<slug>.md` | `2022-flash-attention.md` |
| Codebase pages | `<slug>.md` (+ `<slug>/<subpage>.md`) | `torchtitan.md`, `torchtitan/fsdp.md` |
| Concept pages | `<slug>.md` | `rematerialization.md` |
| Model pages | `models/<architecture>-<lane>.md` | `<model>-jax.md`, `<model>-torchax.md`, `<model>-tpu.md` |
| Kernel family pages | `kernels/<kernel>.md` (`type: model`, `lane: pallas`) | `gemm.md`, `rms-norm.md`, `linear-softmax-cross-entropy-loss.md` |
| Variant identifiers (used in experiment frontmatter) | `<size>/<hardware>` | `"3B/v6e-8"`, `"24B/v6e-32"` |
| Hypothesis pages | `<slug>.md` | `flash-attention-block-2048.md` |
| Experiment pages (model) | `wiki/experiments/<model>_autoresearch_optimization/<lane>/<YYYY-MM-DD>-v<NNN>-<slug>.md` | `<model>_autoresearch_optimization/jax/2026-05-12-v006-<model>-3b-v6e-16-bf16-compute.md` |
| Experiment pages (kernel) | `wiki/kernel_experiments/<kernel>/<lane>/<YYYY-MM-DD>-v<NNN>-<slug>.md` | `kernel_experiments/rms_norm/pallas/2026-07-09-v001-pallas-single-pass-rmsnorm.md` |
| Observation pages | `<slug>.md` | `fsdp-allgather-overlap-gap.md` |
| Analysis pages | `<YYYY-MM-DD>-<slug>.md` | `2026-04-22-v6e-vs-v5p-llama.md` |
| Raw sources | `raw/sources/<year>-<slug>.<ext>` | `raw/sources/2022-flash-attention.pdf` |
| Raw profiles | `raw/profiles/<YYYY-MM-DD>-<exp-slug>/` | directory per experiment |
| Slugs | lowercase, hyphens only | |

---

## Behavioral rules

1. **Never modify `raw/`.**
2. **Update `index.md` and `log.md` after every operation that creates or materially changes pages.**
3. **Never silently overwrite** a claim — use `[!warning]` contradiction blocks.
4. **Every hypothesis is falsifiable** — state the metric, the delta, and how you'd measure. If you can't, it isn't a hypothesis yet.
5. **Every experiment records the full command** and the diff from baseline, not just prose.
6. **Every "supported" verdict requires** measured improvement beyond noise **and** a semantics check (loss or output parity) **and** no tracked-metric regression. Otherwise it is `inconclusive` or `invalid`.
7. **Profile + HLO analysis is mandatory** for experiments — not just profile capture, but actual analysis. Every experiment page with a verdict other than `invalid` **must** carry BOTH a `## Profile` section AND a `## HLO Dump` section, both produced by the `profile-analyzer` agent (dispatched SYNC after the gke-cluster-runner reports completion). The `## HLO Dump` section must include the agent's Phase 3 **hypothesis-firing audit** (`HYPOTHESIS FIRING CONFIRMED` / `SILENT NO-OP DETECTED` / `PARTIAL`) — this is the structural silent-noop check that prevents v675b-class failures. Missing profile or missing HLO Dump ⇒ verdict is `inconclusive` at best. Because raw profile + HLO files are gitignored (see `.gitignore` — `raw/profiles/*` except `.gitkeep`), the experiment page is the only persistent lineage link from trace-on-disk back to the run that produced it.
8. **No model-quality optimizations.** If a proposed change trades accuracy for speed, reject it or mark `invalid`.
9. **Discuss before writing** on ingest — confirm emphasis with the human.
10. **Prefer tables** for metrics, flags, and comparisons.
11. **ISO dates everywhere** (YYYY-MM-DD).
12. **Read `index.md` first** on any query — do not guess which pages exist.
12a. **Silo-first for ingested repos.** Before exploring or modifying source under `raw/code/<slug>`, check `wiki/codebases/<slug>/` first — `overview.md` for orientation, `catalog/` for symbol→file navigation, `concepts/` for mechanisms. Silo claims are leads, not facts: verify anything load-bearing in the artifact that actually executes (installed wheel vs checkout skew is common — name which you checked). For dispatched exploration use the `wikify-repo-navigator` agent. This applies to orientation at task start, NOT to following a stack trace, checking a single known symbol, or mid-debug iteration. If the silo is missing or stale, note it and suggest the fix command (`/wikify-ingest-repo raw/code/<slug>`, or `wikify prepare <slug> --ref <sha>` for a refresh) but proceed with source; never run ingest tooling inline — re-ingestion is proposed, not executed.
13. **One entity/concept/model per page.** Split when a page exceeds ~500 lines.
14. **No cross-wiki links.** This wiki does not reference `tpu_wiki` or any sibling.
15. **One model page per `(architecture, lane)`** — do not split by size or hardware. Use the Variant matrix to track per-variant state within one page. The lane axis is fundamental (different framework, different code path); the size/hardware axes are combinatorial accidents over the same code.
16. **Never duplicate wiki artifacts into the model code repo.** Hypothesis / experiment / observation / analysis pages live only in `wiki/`. The model code repo (e.g. the torchtitan fork) gets only: code, configs, tests, and at most a commit-message footer `exp: wiki/experiments/<...>.md` plus optional one-line `# perf: see <exp>` code comments. See "Experiment narrative vs. model code" section for the reasoning.
17. **Promote shared findings to the infra wiki.** When a per-model autoresearch repo discovers a concept, observation, or kernel pattern that generalizes beyond its model, write it into the infra wiki (this template repo) and link to it from the per-model repo. Don't re-ingest shared substrate from scratch in every per-model repo.
