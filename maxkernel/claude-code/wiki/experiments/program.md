# autoresearch — generic loop program (root)

The procedural specification for the autoresearch optimization loop. An AI agent iteratively modifies a training stack — model code, sharding, kernels, compiler flags, runtime configuration — to maximize throughput (MFU / tokens-per-second) on a fixed hardware budget on TPU.

This is the **generic, model-agnostic** layer. Model-family and lane-specific details are defined in lower-level `program.md` files that override the generic sections. See **Inheritance model** below.

## Inheritance model

The program is hierarchical. Up to three levels of `program.md` apply per experiment:

```
wiki/experiments/program.md                                           ← root: generic
wiki/experiments/<model>_autoresearch_optimization/program.md         ← model-family
wiki/experiments/<model>_autoresearch_optimization/<lane>/program.md  ← lane (optional)
```

**Resolution rule**: child **replaces** parent per H2 section; omitted sections inherit unchanged.

**Additive convention**: for content that is naturally additive rather than replacing (e.g. extra prohibitions on top of generic ones), the child uses a distinct section name like "<MODEL>-specific CAN additions" — these are *new* sections (taken as-is) and the parent's section continues to apply.

The `/start-experiment` skill orchestrates resolution at session start (prints which level each section came from) and the loop re-reads layers every iteration for drift prevention. See the skill for the exact mechanics.

## Concurrency model

**Serial per cluster, parallel across clusters via independent tracks.**

- *Within* one cluster: experiments run **serially**. One workload at a time. The gke-cluster-runner subagent's launch + poll + capture lifecycle blocks the cluster until done.
- *Across* clusters: each cluster is an **independent track**. Tracks proceed at their own pace; there is **no iteration-wide synchronization point** — track A doesn't wait for track B to finish before track A's next experiment dispatches. The `/start-experiment` skill's `--parallelism N` selects how many tracks run concurrently (default 1).

The master orchestrates by dispatching gke-cluster-runner subagents with `run_in_background=true` (one per idle cluster per iteration), processing completed-subagent notifications as they arrive, and re-dispatching on now-idle clusters. No "wait for slowest cluster" anywhere.

Each parallel experiment must operate on its own forked copy of the model code repository to avoid stepping on others' branches, in-progress edits, and docker build contexts. The fork mechanism in **Branching model** below provides that isolation.

## Setup

Each model family defines (in its `program.md`):
- Execution lanes available (e.g. `tpu`, `jax`, `torchax`, `maxtext`), trainer entry point per lane, model code path, model sizes, architecture-specific constants.
- Lane-specific operational env vars (LIBTPU, XLA, JAX cache, etc.), kernels available, hang surfaces — typically grouped under a "Per-lane operational details" section. Optional: split per-lane content into separate `<lane>/program.md` files when it grows large.

Reading orchestration (which files to load in what order: program layers, lane README, model page, recent experiments) is handled by the `/start-experiment` skill — see that skill for the exact read sequence.

## What you CAN do

All code edits happen inside the **per-experiment fork** of the model code repo (see **Branching model**). Within that fork, you have broad edit latitude — the fork is throwaway by default and only merged back to trunk on a `supported` verdict.

Common edit targets inside the fork:

- **Model code** — anything under `<model-code-repo>/torchtitan/experiments/<lane>/<model>/` (trainer, model wrappers, sharding setup, custom ops).
- **torchtitan core** when the optimization requires it — parallelism (FSDP/TP/DDP), config loaders, optimizer implementation, activation checkpointing. Be conservative; favor experiment-folder edits when possible.
- **Training config TOMLs** — batch size, sequence length, parallelism degrees, dtype, profiling, scheduler.
- **Compiler/runtime flags** — TPU runtime args (`LIBTPU_INIT_ARGS`), XLA flags (`XLA_FLAGS`), lane-specific env vars (see lane-level `program.md`).
- **Sharding** — FSDP/TP/DDP split, mesh shape.
- **Existing Pallas kernels** — splash attention, segment_matmul, fused linear+CE loss. Encouraged where a profile signal supports it.
- **New Pallas kernels** — when profile evidence shows a memory-bound or poorly-fused op dominating step time. Encouraged — Pallas kernels that keep tiled intermediates in VMEM and avoid HBM round-trips are a primary optimization lever on TPU.
- **Framework internals** — torchax bridge, JAX runtime knobs — via the appropriate per-experiment image build (which may include a fresh wheel build for C++/MLIR changes).
- **Upstream library monkey-patches** — when the fix is small and semantics-preserving (e.g. TAMM dispatcher patches in `train_minimal.py`).
- **Optimizer / precision mix / activation checkpointing** strategy.

Model-family-specific additions or narrower scoping live in the model-level `program.md` (e.g. "edits should stay under `torchtitan/experiments/<lane>/<model>/` unless explicitly justified").

## What you CANNOT do

The boundary is **the per-experiment fork**. Inside the fork: broad latitude. Outside: ask first.

- **Change model math without human approval.** The output distribution must be identical to the baseline for a fixed seed. If a change affects numerics, verify loss trajectory matches the baseline within bf16 rounding noise over ≥ 20 steps. Changes that fail this check get verdict `invalid`.
- **Some math changes are acceptable with human approval.** Examples: switching to a numerically-equivalent attention kernel; using a fused linear+CE loss kernel; changing a RoPE implementation to a mathematically identical one. Ask first if unsure.
- **Do NOT write to the model-code-repo trunk during a run.** The shared trunk (`raw/code/<repo>/`) is read-only during steps 3–10 of the loop. Only step 11 (Decision, `supported` only) merges from fork into trunk.
- **Do NOT write to the wiki repo from the model-code-repo side**, and vice versa. The two are separate per `SCHEMA.md` — wiki holds narrative, model code holds code. Cross-edits create drift.
- **Do NOT modify `raw/`** other than via the controlled `raw/code/<repo>/` trunk merge above (and `raw/profiles/` writes by the gke-cluster-runner). `raw/sources/`, `raw/assets/`, and the rest are immutable.
- **Do NOT touch other model families' folders** from this experiment's fork. If a fix is needed in shared infra used by another model, that's a separate cross-model change — surface to the user.
- **Do NOT quantize weights below bf16.**

Model-family-specific additional prohibitions (e.g. "do not change MoE routing", "do not modify the attention pattern assignment") live in the model-level `program.md`.

## The goal

**Maximize MFU (Model FLOP Utilization)** at each model size on each hardware target.

MFU = `model FLOPs per step / (peak hardware FLOPs × step wall time × chip count)`. Higher is better. Use detailed matmul-FLOP accounting; if the architecture has routing or sparse structure that makes naive `6PT` overcounting likely, flag it.

Secondary metrics: tokens/sec (`global_batch × seq_len / median_step_time_s`), step time, peak HBM. All must be tracked per experiment. A change that improves MFU but regresses peak HBM past the OOM cliff is **not** a win.

Target sequence length and any cross-lane reference points are model-specific (defined in model-level `program.md`).

## Branching model — per-experiment forks

Code changes live in **per-experiment forks** of the model code repository, NOT in any shared checkout. The wiki repo is narrative only — experiment pages, model pages, observations commit linearly on whatever branch the wiki is on. Branches are never created in the wiki repo for experiments.

The shared trunk of the model code repo is **read-only while an experiment is running** (steps 3–10 of the loop — fork, implement, build, run, measure, analyze). All code edits during a run commit to the fork's branch, never to trunk. Trunk is **only written at step 11 (Decision)**, and only by experiments with verdict `supported` — via `git merge --no-ff` from the fork. Multiple serial experiments share the trunk as their starting point and may merge back into it sequentially, but never edit it concurrently mid-run.

### Structure

```
raw/code/<model-repo>/                                          ← shared trunk (kept wins only)
wiki/experiments/<model>_autoresearch_optimization/
└── <lane>/
    └── .repo/                                                  ← gitignored; per-experiment forks live here
        ├── 2026-MM-DD-v<NNN>-<slug>/<model-repo>/              ← experiment fork (independent working copy + branch)
        ├── 2026-MM-DD-v<NNN>-<slug>/<model-repo>/              ← parallel serial experiment, isolated fork
        └── 2026-MM-DD-v<NNN>-<slug>/<model-repo>/              ← forked from trunk after previous merged
```

- **Trunk**: shared kept-wins history. Only `supported` experiments get merged here. Docker images for production runs are built from trunk.
- **Per-experiment fork**: a **git worktree** of the trunk checked out at `<lane>/.repo/<exp-name>/<model-repo>/`, on its own branch (model-specific naming convention — see model-level `program.md`):

  ```bash
  git -C raw/code/<model-repo> worktree add \
    <path-to>/<lane>/.repo/<exp-name>/<model-repo> -b <branch>
  ```

  The worktree shares the trunk's object store (no per-fork `.git` history duplication) but has its own working tree, index, and HEAD — the isolation forks require. **Format-compatible with the older `cp -r` forks**: identical `.repo/<exp-name>/<model-repo>/` path and file layout, so existing manual copies stay valid and coexist; the one structural difference is that a worktree's `.git` is a gitlink *file*, not a full directory.
- The `<lane>/.repo/` tree is gitignored — forks never enter the wiki repo's history. Reclaim a fork with `git -C raw/code/<model-repo> worktree remove <fork-path>` (or `rm -rf <fork-path>` then `git -C raw/code/<model-repo> worktree prune`); past experiments' worktrees remain on disk until then.

Why per-experiment worktrees instead of shared-checkout branches: multiple serial experiments may be live concurrently on different clusters. They share the trunk as a starting point but must NOT contend on the trunk's working tree, index, or branch state. A worktree gives each run a private working copy while keeping every branch in one shared object store — so merge-back at step 11 is a direct `git merge --no-ff` from trunk, with no cross-directory fetch.

> **Worktree caveats** (vs `cp -r`): (a) the model repo is a git **submodule** — `worktree add` on a submodule works in modern git but has more edge cases than a plain repo; validate once on the real submodule before relying on it. (b) a given branch can be checked out in only one worktree at a time (fine — every experiment has a unique branch). (c) a worktree's `.git` is a gitlink file, so a Docker build that needs a real `.git` directory for version-stamping would break — the layered `COPY <changed>.py` flow doesn't, but smoke-build once to confirm.

**Per-experiment flow** (fork → branch → implement → build → launch → decide) is documented step-by-step in **The experiment loop** below (steps 3–11). The Structure diagram above shows where the artifacts live; the loop tells you what to do with them.

### The link between repos

The model-repo commit's `exp:` footer points to the wiki experiment page. The wiki experiment page's `commit:` frontmatter field points to the model-repo SHA. This is the bidirectional link per `SCHEMA.md`.

## Workload naming convention

Every workload submitted to a shared cluster (GKE / XPK or any multi-tenant environment) MUST follow this format so anyone inspecting the cluster can attribute jobs at a glance:

```
<USER_PREFIX>-<MODEL_NAME>-<LANE>-v<NNN>-<SLUG>[-<RETRY_SUFFIX>]
```

| Field | Source | Example |
|---|---|---|
| `USER_PREFIX` | auto-derived (see below) | `<USER_PREFIX>` |
| `MODEL_NAME` | auto-derived from model folder name (strip `_autoresearch_optimization` suffix, lowercase, replace `_` with `-`) | `gemma4`, `llama3-8b` |
| `LANE` | lane folder name | `tpu`, `jax`, `torchax` |
| `v<NNN>` | per-lane chronological experiment ID, zero-padded 3 digits | `v541` |
| `<SLUG>` | short kebab-case hypothesis description | `gate-up-fusion`, `splash-1024` |
| `<RETRY_SUFFIX>` (optional) | single letter `b`, `c`, … on resubmission of the same v-id | `b` |

Full examples:
- `<USER_PREFIX>-gemma4-tpu-v541-gate-up-fusion` (first attempt, 43 chars)
- `<USER_PREFIX>-gemma4-tpu-v541-gate-up-fusion-b` (retry after the first failed due to e.g. wrong-cluster landing)
- `<USER_PREFIX>-gemma4-jax-v021-rope-fix` (33 chars)

### Length limit (HARD — enforced by XPK at submission)

GKE workload names map to k8s JobSet → Job → Pod resources, each of which has a 63-char DNS label limit. **XPK enforces its own stricter cap** because it appends suffixes like `-slice-job-0-0-0-<rand>` (~24 chars) on top of the user-supplied name:

```
len(workload_name) < 40    → HARD limit. XPK rejects at parse time with:
                             "Name must be less than 40 characters and match
                              the pattern '[a-z]([-a-z0-9]*[a-z0-9])?'"
len(workload_name) ≤ 30    → recommended. Comfortable headroom for retry suffixes.
```

The `gke-cluster-runner` subagent MUST validate `len(workload_name) < 40` before submitting and refuse otherwise (the subagent does not mutate attribution-bearing segments — it returns to the master for regeneration). The master agent (in `/start-experiment`'s loop prompt, step 4) MUST validate at construction time and shorten the slug if at or over the limit.

**Budget breakdown** for typical Gemma 4 workload names (use as a planning aid):

| Segment | Example value | Chars |
|---|---|---|
| `<USER_PREFIX>-` | `<USER_PREFIX>-` | 9 |
| `<MODEL_NAME>-` | `gemma4-` | 7 |
| `<LANE>-` | `jax-` / `tpu-` / `torchax-` | 4 / 4 / 8 |
| `v<NNN>-` | `v316-` | 5 |
| `<SLUG>` | (e.g. `ds-stack`) | budget = `39 - prefix_total` |
| Optional `-<RETRY>` | `-b` | 2 |

For `gemma4` on JAX with a typical 8-char `USER_PREFIX` (`<USER_PREFIX>-gemma4-jax-v<NNN>-`), fixed prefix ≈ 25 chars → **slug budget ≈ 14 chars** (or 12 if reserving room for a retry suffix). For the `torchax` lane, slug budget shrinks to ≈ 10 chars. (A longer `USER_PREFIX` or `MODEL_NAME` shrinks the slug budget further — recompute against the `< 40` hard limit.)

**Recovery when too long**:
1. Shorten the `<SLUG>` — drop articles, abbreviate. Standard abbreviations: `deepseek`→`ds`, `sparsecore`→`sc`, `attention`→`attn`, `fusion`→`fus`, `tensor-parallel`→`tp`, `expert-parallel`→`ep`, `track-parallel`→`pt`, `host-offload`→`ho`, `megablox`→`mblx`, `selective`→`sel`. Slug under 10 chars is usually achievable.
2. If shortening the slug still doesn't fit, the issue is upstream: `<USER_PREFIX>` or `<MODEL_NAME>` is too long. Either set an explicit shorter `USER_PREFIX` in the model-level `program.md`, or revisit the `MODEL_NAME` derivation (e.g. `llama3-8b` instead of `llama3-8b-instruct-something`).
3. Never strip required segments (`USER_PREFIX`, `MODEL_NAME`, `LANE`, `v<NNN>`) to fit — those are the attribution-critical fields.

`USER_PREFIX` and `MODEL_NAME` are **auto-derived at session start** by the `/start-experiment` skill — see the skill for the resolution chain (override → `$USER` segment → git first-name → refuse). They remain constant for the session and are passed through to every dispatch.

### Resubmission and the retry suffix

`xpk workload delete` followed by immediate resubmit on the same name occasionally races the jobset teardown. When you need to retry an experiment with the same `v<NNN>` (e.g. the first attempt landed on the wrong cluster due to kubectl context drift, or the workload was killed and you want to rerun the exact same code):

- Append a single-letter suffix `b`, `c`, `d`, … to the workload name.
- The experiment page filename / wiki notation can mirror this: `2026-MM-DD-v541b-<slug>.md` for the retried attempt. This matches the existing `v391b`, `v412b`, `v418c` convention seen in past experiment pages.

This convention powers cluster-occupancy attribution: any workload not starting with `<USER_PREFIX>-` is treated as a foreign job and the cluster is considered occupied. Workloads starting with `<USER_PREFIX>-` are mine; the model and lane segments allow same-prefix conflict detection without inspecting docker images.

## Experiment format

Each experiment = one wiki page + one run (or minimal set of comparable runs).

Page location: `wiki/experiments/<model>_autoresearch_optimization/<lane>/experiments/<YYYY-MM-DD>-v<NNN>-<slug>.md` where `v<NNN>` is per-lane chronological (zero-padded 3 digits).

Every experiment page must include:
- YAML frontmatter: title, type, hypothesis, model, variant, commit, verdict, tags, created, updated.
- Hypothesis (one sentence, falsifiable).
- Setup: cluster/hardware, docker image, exact launch command or diff from baseline.
- Results table: step × loss × step_time × tps × mfu for the measured range.
- **Profile section**: xprof browser URL, GCS path, local on-disk path under `raw/profiles/`, steps captured.
- **HLO section**: GCS path, local path, count of HLO modules, pre/post-optimization size of the dominant module, headline observations + diff-against-prior-HLO when applicable. If the post-optimization HLO is byte-identical to a prior experiment, flag explicitly — XLA elided the change at compile time, the experiment cannot have produced a runtime difference, and the verdict should reflect that.
- Verdict: `supported` / `refuted` / `inconclusive` / `invalid`.
- Next hypotheses: what this run suggests trying next.

Experiments are **immutable once filed**. If you rerun, file a new experiment page.

**Do NOT batch experiments into summary pages.** One experiment = one page.

## The experiment loop

LOOP until the user interrupts:

1. **Read context.** Re-read all three `program.md` layers (root, model, lane). Check the model page's variant matrix for the current frontier. Read the last 2–3 experiment pages in your lane. Read the lane README for known bugs and operational notes.

2. **Generate a hypothesis via [`/formulate-hypothesis`](../../.claude/skills/formulate-hypothesis/SKILL.md).** The skill reads the shared knowledge layers ([`model-optimization-index.md`](../model-optimization-index.md) + [`model-optimization-blueprint.md`](../model-optimization-blueprint.md) + per-model `refuted-patterns.md` + model page + last 3 experiments + most recent retrospective) and emits a structured proposal with refuted-pattern checks. It supports four modes — `frontier` (default), `exploration` (hunches + stalled variants + novel directions), `bootstrap` (new models with thin state), `user-override` (validate a caller-specified candidate). NEVER skip this skill — v391y (200× regression) came from proposing without the wiki-precedent check.

   Priority order for which candidate to invoke the skill with (this is judgment, not procedure):
   - **Biggest wins first** — prioritize optimizations addressing critical bottlenecks.
   - **Profile-driven** — the highest-signal gap in the most recent profile (top slow op, collective wait, missing fusion, HBM-bound op below ridge point).
   - **Follow-up** — an idea from a previous experiment's "Next hypotheses" section.
   - **Cross-lane gap** (if multi-lane) — if another lane achieves higher MFU on the same hardware, what does it do differently?
   - **Pallas kernel opportunity** — an op that's memory-bound, poorly fused by XLA, with no existing Pallas coverage. Profile first, then propose.
   - **Novel direction** — a hypothesis that doesn't fit any cataloged topic or blueprint phase. Allowed; invoke the skill with `<mode>=exploration` (or accept `Topic: novel-mechanism` / `Phase: novel` in frontier mode with first-principles justification).

   Maintain the knowledge layers incrementally: cross-model lessons land in [`model-optimization-index.md`](../model-optimization-index.md) as generic principles; per-model refuted experiments land in `wiki/experiments/<model>_autoresearch_optimization/refuted-patterns.md` as entries.

3. **Fork the codebase.** Create a per-experiment worktree fork (`git worktree add -b <branch>`) per the **Branching model** snippet. All code edits commit to the fork's branch; the shared trunk is never touched until step 11.

4. **Implement.** Invoke [`/edit-model-code`](../../.claude/skills/edit-model-code/SKILL.md) BEFORE opening any file in the fork — it loads the surgical-edit discipline (one mechanism per experiment, no semantics drift, no kernel fallbacks, no while-I'm-here cleanups). Then make the code/config change inside the fork and commit. Commit message MUST carry the `exp: wiki/experiments/<...>.md` footer per the **Branching model** "link between repos" section. Keep it minimal — one hypothesis per experiment.

5. **Build and push.** Build docker image from the fork, tag with the branch name, push to artifact registry.

   **For layered Python-only patches** (the common case — `FROM <previous-image> + COPY changed.py`): the file you COPY MUST be derived from the image's bundled version, not from current trunk. Trunk may have diverged from the image's commit (shared dataclasses change shape, config-parser annotations evolve, module paths move), and overlaying current-trunk files onto an older image causes config-parse crashes at startup — a 10-second pre-compile failure that wastes cluster dispatch time.

   The exact crash signature varies by stack:
   - **torchtitan**: tyro CLI parser raises on mismatched dataclass shape
   - **MaxText**: pyconfig / ml_collections asserts on missing or extra fields
   - **Paxml / Praxis**: Fiddle config-tree resolution fails on type mismatch
   - **JAX / Flax NNX trainers**: pydantic-style validation fails on config schema mismatch
   - **General**: import-time `TypeError` / `AttributeError` when shared modules diverged

   Workflow (stack-agnostic — substitute `<MODEL_REPO>` and `<entry-module>` per stack):

   ```bash
   # 1. Extract the image's bundled version of the file
   #    <MODEL_REPO> e.g. torchtitan / MaxText / paxml / flax — whatever the image bundles
   docker run --rm <BASE_IMAGE> cat /<MODEL_REPO>/<path>/<file>.py > /tmp/build/<file>.py

   # 2. Apply your patch on top (Edit tool, sed, or diff/patch). Validate AST:
   python -m py_compile /tmp/build/<file>.py

   # 3. Smoke-test the rebuilt image BEFORE pushing:
   #    <entry-module> e.g. torchtitan.train / MaxText.train / your_trainer.main
   docker build -t <new_tag> .
   docker run --rm <new_tag> python -m <entry-module> --help 2>&1 | head -3
   # If --help / config-load fails, the parser is broken — fix the rebase before pushing.
   ```

6. **Run.** For GKE/XPK launches, **always dispatch the launch + poll + capture via the `gke-cluster-runner` subagent** (see `.claude/agents/gke-gke-cluster-runner.md`). Do not run XPK launches directly from the master session. Non-GKE single-host runs (e.g. on the local TPU VM) are still done from the master session.

   Before launching, check that no other workload occupies the target cluster (see `/start-experiment` skill for the discovery + occupancy check). Always include profiling flags + HLO dump — both write to the same experiment-slug folder on GCS (path defined in lane-level `program.md`).

   **GKE polling discipline** (per gke-cluster-runner agent): poll pod status every 30 s. Use the layered hang heuristic (5 min idle log → check CPU + HLO module generation; declare hung only if both indicate inactive). Hard ceiling 60 min wall time unless explicitly authorized for a long run.

7. **Measure.** Extract metrics from the trainer log + xprof. Compare against the baseline for the same variant.

8. **ANALYZE PROFILE + HLO DUMP (MANDATORY — NON-NEGOTIABLE).** Every experiment that runs to completion MUST have BOTH its xprof trace AND its HLO dump analyzed. An experiment without both analyses is incomplete. Do NOT dispatch the next experiment until both are analyzed and the findings recorded.

   **Mechanism**: in the orchestrated loop (`/start-experiment`), this analysis is performed by the [`profile-analyzer`](../../.claude/agents/profile-analyzer.md) subagent, dispatched **synchronously** when a `gke-cluster-runner` completes — it returns the `## Profile` + `## HLO Dump` sections that get pasted into the experiment page. For local single-host runs (no subagent), the master performs the steps below directly. Either way, both artifacts must be analyzed before the verdict is assigned. The tool references below document what that analysis covers.

   **(a) xprof analysis** — use the xprof MCP tools directly against GCS (no local mirror needed): `list_runs`, `get_overview`, `get_top_hlo_ops`, `get_memory_profile`, `get_op_profile`, etc.

   **(b) HLO dump analysis** — same principle: **inspect directly from GCS** for the routine path. HLO dumps are 100–500 MB per experiment and mirroring every one fills `raw/profiles/` fast for no benefit. Use:

   ```bash
   gsutil ls -l ${EXP_GCS}/hlo/                          # enumerate modules, find dominant by size
   gsutil cat ${EXP_GCS}/hlo/module_NNNN.*.after_optimizations.txt | grep/head/awk  # inspect inline
   gsutil cat ${EXP_GCS}/hlo/module_NNNN.*_buffer-assignment.txt | head -100        # OOM attribution
   ```

   Files to look at:
   - `module_NNNN.*.before_optimizations.txt` — what the frontend produced.
   - `module_NNNN.*.after_optimizations.txt` — what XLA actually compiles. Source of truth for what runs on TPU.
   - `*.html` — interactive op-graph viewer. **Requires local file** — pull selectively with `gsutil cp gs://.../module_NNNN.html /tmp/`.
   - `*_buffer-assignment.txt` — HBM layout and live-set, useful for OOM attribution.

   **Mirror selectively, not by default**, when:
   - You'll do extensive cross-module exploration with many tools and GCS round-trip latency hurts.
   - You need the interactive `.html` viewer.
   - You're diffing across runs and want both module trees locally for `diff -r`.

   When you do mirror, scope it: `gsutil -m cp -r ${EXP_GCS}/hlo raw/profiles/<exp-slug>/hlo/` (just the HLO subdir, not the whole `${EXP_GCS}`). The local path is then `raw/profiles/<exp-slug>/hlo/` and can be referenced from the experiment page.

   **Patterns to look for**: unexpected fp32 in a bf16-compute run; cross-tile copies from suboptimal sharding-constraint placement; collectives in the inner loop that should have been hoisted; fusion misses.

   **Diff against prior experiment** when the hypothesis claims a structural change. Two runs with byte-identical post-optimization HLO mean XLA elided the change at compile time — the flag/code change cannot have produced a runtime difference. This is a critical signal.

   Draw conclusions from BOTH artifacts: compute-bound, memory-bound, or collective-bound? Which specific op should the next experiment target? **The next hypothesis MUST be informed by both profile and HLO analysis, not guesswork.**

9. **Validate semantics.** If the change could affect numerics, verify loss matches baseline within bf16 noise over the profiled steps. If it diverges, verdict is `invalid`.

10. **FILE THE EXPERIMENT PAGE (MANDATORY — NON-NEGOTIABLE).** Every single experiment, without exception, gets its own wiki page **before** the next experiment is dispatched on that cluster. No experiment page = cluster stays idle. One experiment = one page.

11. **Decision.**
    - **Keep (supported)** — merge fork's branch into trunk via `git merge --no-ff` (with worktrees the branch already lives in trunk's object store — no fetch needed; from a legacy `cp -r` fork, `git fetch <fork-path> <branch>` first). Then rebuild the shared docker image from trunk under a stable tag so subsequent experiments build on the new baseline.
    - **Discard (refuted/inconclusive)** — do nothing to trunk. Leave the fork in place under `.repo/` (reclaim later with `git worktree remove`).

12. **Loop.** Use the profile-informed "Next hypotheses" you just wrote to feed the next iteration. Every hypothesis should cite a specific profile signal.

**NEVER STOP.** Once the experiment loop has begun (after initial setup), do NOT pause to ask the human if you should continue. The human may be asleep or away and expects you to continue working indefinitely until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read referenced papers, consult internet, re-read in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you.

## Metric

**Primary: MFU.** Detailed matmul-FLOP accounting (don't naively use `6PT` if the architecture is sparse/routed).

**Secondary: TPS** = `global_batch × seq_len / median_step_time_s`. Invariant under batch/seq tuning; useful for cross-variant comparison.

**Diagnostic: step_time.** Human-readable, not a decision metric on its own.

**Measurement protocol**: median step_time over steady-state steps (typically steps 3+ for short runs, steps 6–15 for longer). Step 1 includes XLA compilation and is excluded from steady-state metrics. Compile time is recorded separately as a diagnostic.

## Output format — per run

Extracted from trainer log + xprof:

```
tps:             <global_batch × seq_len / median step_time>
mfu_percent:     <detailed matmul FLOPs / (peak × wall time × n_chips)>
step_time_ms:    <median over steady-state steps>
peak_hbm_gib:    <from xprof get_memory_profile>
compile_seconds: <step 1 wall time — diagnostic only>
config:          <key flags / code change vs baseline, one-line>
```

## Ledger — `<lane>/RESULTS.tsv`

One ledger per lane. Append one row per run (tab-separated):

```
exp_id  date  tps  mfu_percent  step_time_ms  peak_hbm_gib  config  status  description
```

- `exp_id` — `v<NNN>` (matching the experiment page's version number). Numbering is per-lane.
- `status` ∈ `{keep, discard, crash, parked, inconclusive}`.

## Simplicity criterion

All else being equal, simpler is better. A small MFU improvement that adds fragile complexity is not worth it. Removing code and getting equal or better results is a win. Large MFU lifts from small, targeted changes are the ideal.

## Pallas kernels

Pallas kernels are a primary optimization lever on TPU. The decision framework:

- **Compute-bound above the HBM ridge point**: Pallas rarely wins. XLA's MXU lowering is hard to beat for dense GEMM.
- **Memory-bound below the ridge point**: Pallas candidate. Keep tiles in VMEM, fuse consecutive HBM round-trips.
- **Always profile first.** Proposing a Pallas kernel without a matched profile signal is speculation.
- **Check existing kernels first.** tokamax ships ~200 kernels across ~30 repos. Check the Pallas kernel directory before writing from scratch.
- **HLO pre-filter**: before writing a replacement kernel, dump post-optimization HLO and check if XLA already fuses your target pattern. If it does, retire the hypothesis.

## Sources

- `SCHEMA.md` — wiki structure and schema rules.
- `wiki/experiments/<model>_autoresearch_optimization/program.md` — model-family overrides.
- `wiki/experiments/<model>_autoresearch_optimization/<lane>/program.md` — lane-specific overrides.
- `.claude/agents/gke-gke-cluster-runner.md` — subagent definition for GKE workload launches.
- `.claude/skills/create-experiment/` — bootstrap a new model family's folder structure and program.md stubs.
- `.claude/skills/start-experiment/` — loop entry: hardware selection, cluster discovery, occupancy check, parallel-tracks dispatch.
- `.claude/skills/stop-experiment/` — clean shutdown: orphan reaping, missing-page filing, lint.
