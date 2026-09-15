<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# program.md — MaxPerf optimization of Qwen3-Coder-480B vLLM inference on TPU v6e

<!-- PERMISSIONS:
     READ-ONLY sections: not editable by any agent.
     APPEND-ONLY sections: accept new entries but no edits to existing entries.
     R/W sections: editable by the orchestrator (MaxPerf) only.
     Sub-agents propose queue items and author experiment pages;
     only MaxPerf merges proposals into the queue. -->

This file governs the MaxPerf optimization phase. Phase 0 (vLLM
flag-level tuning) is a prerequisite and is governed by a separate
program file. MaxPerf engages **after** Phase 0 has produced a stable,
near-optimal configuration-tuning result. The Phase 0 best config is
the baseline this document optimizes against.

---

## Gemini Operating Notes

MaxPerf is a multi-agent system with one orchestrator and six sub-agents.
In practice, a single Gemini instance plays all roles by switching context.

**How role-switching works:**

1. The orchestrator role is always active. Load `agents/maxperf_orchestrator.md`
   as the default system context alongside this file.
2. When the orchestrator routes a hypothesis to a sub-agent, load that agent's
   prompt from the `agents/` directory as role context for the duration of that
   task. The agent prompts are:

   | Agent | Prompt file |
   |-------|-------------|
   | MaxPerf (orchestrator) | `agents/maxperf_orchestrator.md` |
   | TPUDiagnoseAgent | `agents/tpu_diagnose.md` |
   | DeepResearch | `agents/deep_research.md` |
   | GraphArchitect | `agents/graph_architect.md` |
   | MaxKernel | `agents/max_kernel.md` |
   | MaxTile | `agents/max_tile.md` |
   | MaxInference | `agents/max_inference.md` |
   | MaxSync | `agents/max_sync.md` |
   | MaxFlow | `agents/max_flow.md` |
   | MaxAlign | `agents/max_align.md` |

3. When acting as a sub-agent, follow that agent's prompt exclusively.
   Do not mix sub-agent roles within a single task.
4. After the sub-agent task completes, return to the orchestrator role
   and resume the loop protocol from the step where you left off.

**Decision tree for role selection:**

```
Hypothesis pulled from queue
  ├── What is the hypothesis class?
  │   ├── graph-rewrite      → Load graph_architect.md
  │   ├── kernel-autotune    → Load max_tile.md
  │   ├── kernel-novel       → Load max_kernel.md
  │   ├── code-refactor      → Load graph_architect.md
  │   ├── inference-sampling → Load max_inference.md
  │   ├── transport-sync     → Load max_sync.md
  │   ├── pipeline-flow      → Load max_flow.md
  │   ├── numeric-align      → Load max_align.md
  │   └── composite          → Load primary agent, then secondary
  │
  ├── Does it need symbolic grounding first?
  │   ├── Yes → Load deep_research.md before the implementer
  │   └── No  → Proceed to implementer directly
  │
  └── After implementation → Load tpu_diagnose.md for profiling
```

---

## Target — READ-ONLY

| Field | Value |
|-------|-------|
| **Model** | `Qwen3-8B` (loaded from local `qwen3-8b-dummy` config) |
| **Stack** | vLLM TPU backend (V1 engine) on top of `tpu-inference` (Pallas kernels) and JAX/XLA |
| **Optimization layers** | JAX graph, HLO, Pallas kernel, ISA modeling (not vLLM flags — that is Phase 0) |
| **Hardware** | TPU v6e-8 for optimization sprint; v6e-16 reserved for validation runs |
| **Scenario** | Serving inference; workload regime resolved in Phase 0 and inherited as fixed input |
| **Workload — fixed** | Max tokens: 8192. Max requests: 64. These do not vary across experiments. |
| **Workload — variable** | All other workload parameters (request mix, prompt length distribution, concurrency, etc.) may vary per experiment. |

## Metric — READ-ONLY

**Primary metric:** tokens/sec/chip at the workload's representative request
mix, measured against the Phase 0 best config.

**Secondary diagnostics** (every experiment captures all; not optimized directly):

| Diagnostic | Description |
|------------|-------------|
| p50/p99 TPOT | Per-token output time percentiles |
| p50/p99 TTFT | Time to first token percentiles |
| HBM peak per chip | Memory high-water mark |
| KV cache occupancy | Cache utilization |
| Diagnostic vector | Roofline position, Headroom %, DMA idle %, HBM bandwidth %, VREG spill count, collective latency breakdown (see TPUDiagnoseAgent spec) |

**Correctness gate:** HumanEval + MBPP accuracy must not drop more than
1.0pp below the Phase 0 baseline. Numeric-equivalence checks (see binding 7)
are a stricter additional gate for graph-rewrite and code-refactor experiments.

---

## Fixed Bindings — READ-ONLY

These constraints govern what any agent is allowed to do.

### Binding 1 — Director's directive

Do not import performance-improvement ideas previously published *for
Qwen3-Coder-480B specifically*. Ideas published for other models, general
MoE inference, the vLLM TPU backend, `tpu-inference` Pallas kernels,
JAX/XLA compiler internals, or TPU ISA documentation are permitted.

### Binding 2 — Three hypothesis origination paths only

Every hypothesis filed into the queue must be tagged with its origination
path and carry the corresponding evidence:

| Origination path | Source agent | Required evidence |
|------------------|-------------|-------------------|
| **Profile-grounded** | TPUDiagnoseAgent | Offending xprof segment + diagnostic-vector entry that flagged it |
| **Symbolic-grounded** | DeepResearch | Derivation (commutativity proof or algebraic ISA model) |
| **Iterative-debug** | MaxKernel | Numeric corruption trace + hardware constraint violated |

No fourth origination path exists. "I think this might be faster" is not
a valid origination — it has no evidence to file.

### Binding 3 — One change per experiment

A graph rewrite is one change. A tile-size sweep is one change (the sweep
is the experiment, not five experiments). A multi-flag combination is one
experiment only if the agent states explicitly why the flags must be tested
together (typically a known-coupled bundle from upstream documentation).

### Binding 4 — Permitted code modification patterns

Code modifications are permitted within these three documented patterns only:

| Pattern | Agent | Description |
|---------|-------|-------------|
| HLO-barrier removal | GraphArchitect | Push static logic into `@dataclass(frozen=True)` properties, replace Python `if` with `jnp.where`/`lax.cond`, eliminate dynamic indexing overhead; inject dynamic FP8 scaling factor tracking and speculative tree-verification kernels |
| Graph refactor | GraphArchitect | Relocate reductions, eliminate redundant collectives, pass indexing metadata in place of materialized payloads; refactor tensor sharding mesh layouts and multi-dimensional mesh configurations (TP/DP/EP/PP sizes); implement Ring-Attention context parallelism and Pathways disaggregated prefill/decode meshes |
| Novel kernel insertion | MaxKernel | Pallas kernels for operations XLA cannot fuse (see binding 5 for scope test) |

Modifications outside these three patterns require human approval.

### Binding 5 — Pallas custom-kernel scope test

A Pallas custom kernel is in scope **if and only if** it expresses an
operation XLA does not already fuse. The test is empirical:

```
Before writing a kernel:
  1. Inspect HLO output for the fusion pattern in question.
  2. If XLA already produced a fused HLO op → kernel is OUT OF SCOPE.
  3. If no such fused op exists → kernel is IN SCOPE.
```

| Status | Kernel families |
|--------|----------------|
| **In scope** | Ragged gather/scatter, dynamic-tiled GMM, pipelined quantized-matmul/dequantize, custom ragged attention with bin-packing |
| **Out of scope** | RMSNorm replacement, SwiGLU replacement, fused cross-entropy (training-only) |

### Binding 6 — Steady-state TPS only

Throughput measured after ≥200-request warm-up. Cold-start numbers are
diagnostic only.

### Binding 7 — Numeric-equivalence gate for structural changes

Graph rewrites (binding 4 pattern 2) and code refactors (binding 4
pattern 1) must pass numeric equivalence: same inputs produce outputs
within bf16 tolerance of the Phase 0 baseline, tested on a fixed
100-prompt corpus. Failures are **rejected** regardless of throughput.

### Binding 8 — Semantic preservation

No technique that changes the model's output distribution beyond bf16
numeric tolerance is permitted. Speculative decoding (including tree-verification) is permitted
(output distribution preserved by acceptance check). Dynamic FP8/INT4 scaling factor tracking
is permitted provided numeric equivalence checks pass. Aggressive
quantization beyond FP8 baseline (e.g., INT4 weight-only) is exploratory
only and requires human approval per experiment unless dynamically scaled.

---

## Baseline — READ-ONLY (filled in after Phase 0 settles)

| Field | Value |
|-------|-------|
| Phase 0 best branch | `qwen3coder-maxperf-20260527-ghostfish-training` |
| Phase 0 best commit | `75a05436214b6299a519c5c7dfce674ebe87f4a1` |
| Phase 0 best config | see `runs/phase0-best.sh` |
| TPS/chip | `4909.07` |
| p50/p99 TPOT | `13.43` / `13.43` |
| p50/p99 TTFT | `1377.16` / `1695.53` |
| HBM peak/chip | `<TBD>` |
| Eval accuracy | HumanEval pass@1 `<TBD>`, MBPP pass@1 `<TBD>` |
| Numeric-equiv reference | `gs://<bucket>/qwen3coder/numeric-ref/` |
| Phase 0 diagnostic vector | `<TBD>` |

MaxPerf optimizes against the residual bottlenecks the diagnostic vector
reveals — Phase 0 best is already configuration-tuned.

---

## Sub-Agent Roster — READ-ONLY

This section defines who does what. The orchestrator reads this section
to route each experiment to the correct authoring agent. Each sub-agent
has a full system prompt in the `agents/` directory.

### MaxPerf — orchestrator

- **Prompt**: `agents/maxperf_orchestrator.md`
- **Role**: Pulls top hypothesis from queue, routes to authoring sub-agent,
  manages experiment lifecycle, merges proposed hypotheses into the queue.
- **Edit scope**: `Hypothesis queue`, `Wins so far`, and `Refuted` sections
  of this file. Cannot edit READ-ONLY sections.
- **Halt rule**: After three consecutive non-wins on the same hypothesis
  class, pause that class and rotate. After three consecutive non-wins
  across classes, stop and surface ceiling analysis.

### TPUDiagnoseAgent

- **Prompt**: `agents/tpu_diagnose.md`
- **Role**: Runs profiling on every experiment's measurement run, produces
  the diagnostic vector, files profile-grounded hypotheses.
- **Inputs**: xprof trace, XLA compiler logs, HBM bandwidth counters.
- **Outputs (the diagnostic vector)**:

  | Entry | Description |
  |-------|-------------|
  | Roofline analysis | Compute-bound vs memory-bound, distance from peak, binding resource |
  | Headroom report | Per-op time breakdown; ops >5% of step time flagged with fusion status, input shape, category (FFN/attention/collective/quant/other) |
  | DMA idle % | Time spent waiting on memory transfers |
  | HBM bandwidth utilization % | Memory bandwidth usage |
  | VREG spill count | Parsed from XLA compiler logs (LDST thrashing indicator) |
  | Collective latency breakdown | Time per all-to-all, all-reduce, all-gather, by source op |

- **Hypothesis-filing rule**: When a diagnostic-vector entry shows a
  regression vs Phase 0 baseline, OR when an entry shows >5% inefficiency
  that no current queue item targets, file a profile-grounded hypothesis
  at the top of the queue with the diagnostic-vector entry as evidence.

### DeepResearch

- **Prompt**: `agents/deep_research.md`
- **Role**: Produces symbolic-grounded hypotheses through commutativity
  reasoning over the JAX graph, and algebraic modeling of the TPU ISA.
- **Inputs**: HLO dumps (pre/post XLA optimization), JAX program trace,
  TPU ISA documentation, instruction-latency tables.
- **Outputs**:
  1. **Graph-rewrite proposals**: commutativity proof + expected payload-size delta
  2. **Tile-size derivations**: algebraic derivation from ISA latency tables
- **Authoring scope**: Writes hypothesis filings only. Does not author
  experiment pages. Implementations of graph rewrites and layouts go to GraphArchitect.

### GraphArchitect

- **Prompt**: `agents/graph_architect.md`
- **Role**: Originates, implements, and refactors JAX/Flax Python-level graph layouts, sharding PartitionSpecs, weight synchronization, and structural barrier-removal optimizations.
- **Inputs**: HLO dumps, JAX/Flax source code, Mosaic MLIR modules, LLO IR assemblies, commutativity proofs, and structural barrier hypotheses.
- **Outputs**: JAX/Flax code changes, numeric-equivalence results on 100-prompt corpus, compilation diffs (HLO/MLIR/LLO), experiment pages.
- **Authoring scope**: Experiment pages for graph-rewrite and code-refactor experiments.
- **Failure recovery**: For graph changes, files iterative-debug hypothesis (max 3 iterations). For structural changes, systematically narrows constraints (max 5 iterations) and records them in `constraint_log.md`.

### MaxKernel

- **Prompt**: `agents/max_kernel.md`
- **Role**: Writes Pallas custom kernels for in-scope operations (binding 5).
- **Inputs**: Profile-grounded or iterative-debug hypothesis identifying
  kernel-level bottleneck or correctness issue.
- **Outputs**: Pallas kernel implementation, numeric-equivalence result,
  experiment page.
- **Iterative self-debug**: When numeric check fails (e.g., FP8 race on
  32-bit sublane writes), reads error log, identifies hardware constraint
  violated, rewrites kernel, reruns.
- **Iteration limit**: 5 numeric-check failures terminates the attempt.
  Experiment filed as rejected with constraint analysis as the lesson.

### MaxTile

- **Prompt**: `agents/max_tile.md`
- **Role**: Implements tile-size and autotuning experiments, preferring
  algebraic derivation over empirical search.
- **Inputs**: Tile-size derivation from DeepResearch (preferred), or
  profile-grounded hypothesis indicating tile-level inefficiency (fallback).
- **Outputs**: Tile configuration, verification run, experiment page.
- **Method preference order**:
  1. Algebraic derivation against ISA model — preferred
  2. Bounded empirical sweep (≤8 configurations) — fallback
  3. Brute-force grid search — **explicitly rejected**. If (1) and (2)
     don't apply, file a clarifying question to the human.

### ExecutionConfigAgent

- **Prompt**: `agents/execution_config_agent.md`
- **Role**: Deterministic compiler co-design system that runs semantic profile classification and applies hard overrides / optimization safeguards to define the safe search space.
- **Inputs**: `model_config`, `execution_config`, and `hardware_config` JSON strings.
- **Outputs**: Validated JSON configuration containing `"classification"`, `"frozen_parameters"`, and `"open_optimization_variables"`.

---

## Sub-Agent Interface Contract — READ-ONLY

This section defines the structured handoff between the orchestrator and
each sub-agent: what goes in, what comes back, and where artifacts live.

### Artifact storage layout

```
experiments/
  <YYYYMMDD>-<slug>/
    experiment.md              # Experiment page (authored by implementing agent)
    metadata.json              # Structured metadata (created by orchestrator)
    numeric_equiv_result.json  # Numeric-equiv output (if applicable)
    hlo_scope_test.json        # HLO scope test output (kernel-novel only)
    diagnostic_vector.json     # TPUDiagnoseAgent output
    hlo_pre.txt                # Pre-change HLO fragment (code-refactor, graph-rewrite)
    hlo_post.txt               # Post-change HLO fragment (code-refactor, graph-rewrite)
    mosaic_mlir.mlir           # Mosaic MLIR module dump (debug=True outputs)
    llo_dump.llo               # LLO IR assembly (VREG spill & dual-issue density analysis)
    derivation.md              # DeepResearch derivation (symbolic-grounded only)
    constraint_log.md          # Iterative constraint narrowing log (kernel-novel, graph-rewrite)
```

### Handoff schema: orchestrator → sub-agent

The orchestrator assembles this context before loading the sub-agent prompt.
Every field is explicitly provided — the sub-agent does not search for inputs.

```
HANDOFF {
  experiment_id:    "<YYYYMMDD>-<slug>"
  hypothesis:       "<name from queue>"
  class:            "<graph-rewrite | kernel-autotune | kernel-novel | code-refactor>"
  origination:      "<profile | symbolic | iterative-debug>"
  branch:           "qwen3coder-maxperf-<YYYYMMDD>-<slug>"
  artifact_dir:     "experiments/<YYYYMMDD>-<slug>/"

  # Evidence pointers (use standardized paths from Artifact paths section)
  evidence: {
    xprof_trace_dir:    "raw/profiles/<YYYYMMDD>-<slug>/" or null
    xla_log_path:       "raw/xla_logs/<YYYYMMDD>-<slug>.log" or null
    hlo_dump_dir:       "raw/hlo/<YYYYMMDD>-<slug>/" or null
    mosaic_mlir_path:   "experiments/<prior-id>/mosaic_mlir.mlir" or null
    llo_path:           "experiments/<prior-id>/llo_dump.llo" or null
    derivation_path:    "experiments/<prior-id>/derivation.md" or null
    constraint_log_path: "experiments/<prior-id>/constraint_log.md" or null
    source_file_paths:  ["<file:line>", ...] or []
  }

  # Reference data (standardized paths)
  reference: {
    numeric_ref_corpus: "raw/numeric_ref/prompts.jsonl"
    numeric_ref_outputs: "raw/numeric_ref/reference_outputs/"
    isa_doc:            "raw/isa/v6e_instruction_set.pdf"
    latency_tables:     "raw/isa/latency_tables/latency_tables.json"
  }

  # Wiki knowledge base (read-only — paths relative to this directory)
  reference_wiki_paths: {
    wiki_index:         "wiki/index.md"
    wiki_sources:       "wiki/sources/"
    wiki_codebases:     "wiki/codebases/"
    wiki_concepts:      "wiki/concepts/"
    wiki_observations:  "wiki/observations/"
    wiki_analyses:      "wiki/analyses/"
    wiki_hypotheses:    "wiki/hypotheses/"
    kernel_directory:   "wiki/analyses/2026-04-23-pallas-kernel-directory.md"
    raw_code:           "raw/code/"
    raw_sources:        "raw/sources/"
  }

  baseline: {
    tps_per_chip:   <number>
    p50_tpot_ms:    <number>
    p99_tpot_ms:    <number>
    hbm_peak_gib:   <number>
    diagnostic_vector_path: "<path to baseline diagnostic_vector.json>"
  }

  iteration:        <1-5 for iterative-debug, null otherwise>
  prior_constraints: ["<constraint from prior iteration>", ...] or []
  accept_rule:      "<from taxonomy table>"
  reject_rule:      "<from taxonomy table>"
}
```

### Return schema: sub-agent → orchestrator

Every sub-agent returns this structured result. The orchestrator reads it
to issue the verdict, update the queue, and append to RESULTS.tsv.

```
RETURN {
  experiment_id:    "<YYYYMMDD>-<slug>"
  agent:            "<agent name>"
  status:           "<complete | failed | blocked>"

  # Artifact paths (relative to artifact_dir)
  experiment_page:  "experiment.md"
  artifacts: {
    numeric_equiv:      "numeric_equiv_result.json" or null
    hlo_scope_test:     "hlo_scope_test.json" or null
    diagnostic_vector:  "diagnostic_vector.json" or null
    hlo_pre:            "hlo_pre.txt" or null
    hlo_post:           "hlo_post.txt" or null
    mosaic_mlir:        "mosaic_mlir.mlir" or null
    llo_dump:           "llo_dump.llo" or null
    derivation:         "derivation.md" or null
    constraint_log:     "constraint_log.md" or null
  }

  # Verdict inputs (orchestrator uses these to apply accept/reject template)
  measurements: {
    tps_per_chip:       <number or null>
    p50_tpot_ms:        <number or null>
    p99_tpot_ms:        <number or null>
    hbm_peak_gib:       <number or null>
    compile_time_s:     <number or null>
    eval_humaneval:      <number or null>
    eval_mbpp:           <number or null>
    vreg_spill_delta:   <number or null>
    diagnostic_vector_delta: "<structured delta string>"
  }

  numeric_equiv_pass:   <true | false | "N/A">
  hlo_barrier_resolved: <true | false | null>  # code-refactor only
  hlo_scope_verdict:    "<in-scope | out-of-scope>" or null  # kernel-novel only

  # Queue proposals (zero or more new hypotheses for orchestrator to merge)
  proposed_hypotheses: [
    {
      name:         "<short name>"
      class:        "<hypothesis class>"
      origination:  "<profile | symbolic | iterative-debug>"
      evidence:     "<path or description>"
      expected_gain: "<range>"
      priority:     "<top | normal>"
    }
  ]

  # For iterative-debug: what constraint was identified this iteration
  constraint_identified: "<description>" or null
  iteration_exhausted:   <true | false>

  notes:  "<free text for orchestrator>"
}
```

### Per-agent handoff details

What the orchestrator provides beyond the base HANDOFF, and what each
agent is responsible for reading from disk vs receiving in context:

| Agent | Additional context provided | Agent reads from disk | Agent writes to artifact_dir | Wiki references used |
|-------|---------------------------|----------------------|------------------------------|----------------------|
| TPUDiagnoseAgent | xprof trace path, XLA log path | xprof trace, XLA compiler logs, HBM counters | `diagnostic_vector.json` | `wiki_observations` (past profiling patterns) |
| DeepResearch | HLO dump paths (pre/post), ISA doc paths, target collective/op to analyze | HLO dumps, ISA latency tables | `derivation.md` | `wiki_sources`, `wiki_concepts`, `wiki_observations`, `wiki_codebases` |
| GraphArchitect | HLO dump paths, Python source file paths, derivation.md, Mosaic MLIR, LLO | HLO dumps, Python source, Mosaic MLIR, LLO IR, derivation | `experiment.md`, `numeric_equiv_result.json`, `hlo_pre.txt`, `hlo_post.txt`, `constraint_log.md` (if iterative) | `wiki_concepts`, `wiki_observations`, `wiki_codebases`, `wiki_sources` |
| MaxKernel | HLO dump path for scope test, xprof trace path, ISA doc paths | HLO dump, xprof trace, ISA docs, prior `constraint_log.md` | `experiment.md`, `hlo_scope_test.json`, `numeric_equiv_result.json`, `constraint_log.md` | `kernel_directory`, `wiki_codebases` (tokamax, pallas-forge), `wiki_concepts`, `wiki_observations` |
| MaxTile | `derivation.md` path (algebraic) or xprof trace path (fallback), current tile config values | Derivation or trace, ISA tables | `experiment.md` | `wiki_concepts`, `wiki_codebases` |

### Where results are captured

| What | Where | Who writes | When |
|------|-------|-----------|------|
| Per-experiment narrative | `experiments/<id>/experiment.md` | Implementing agent | Loop step 9 |
| Structured experiment metadata | `experiments/<id>/metadata.json` | Orchestrator (created at step 2) | Loop step 2 |
| Numeric-equiv result | `experiments/<id>/numeric_equiv_result.json` | Implementing agent | Loop step 4 |
| Diagnostic vector | `experiments/<id>/diagnostic_vector.json` | TPUDiagnoseAgent | Loop step 6 |
| HLO scope test | `experiments/<id>/hlo_scope_test.json` | MaxKernel | Before loop step 3 |
| HLO diffs | `experiments/<id>/hlo_pre.txt`, `hlo_post.txt` | GraphArchitect | Loop step 3 |
| Derivation | `experiments/<id>/derivation.md` | DeepResearch | Before loop step 3 |
| Constraint log | `experiments/<id>/constraint_log.md` | MaxKernel or GraphArchitect | Each iterative-debug iteration |
| Cumulative ledger | `RESULTS.tsv` | Orchestrator | Loop step 10 |
| Proposed Hypotheses | `wiki/hypotheses/<name>.md` | Proposing agent (TPUDiagnoseAgent or DeepResearch) | Loop step 6 or before step 3 |
| Queue state | `program.md` Hypothesis Queue section | Orchestrator | Loop step 11 |
| Win/loss record | `program.md` Wins So Far / Refuted sections | Orchestrator | Loop step 11 |

### Artifact paths and formats — READ-ONLY

Standard paths for all artifacts the system produces or consumes. Agents
must use these paths exactly — no ad-hoc locations.

#### Directory layout

```
max_perf_g/
  program.md
  RESULTS.tsv
  agents/                                 # Agent prompts (read-only)
  experiments/
    <YYYYMMDD>-<slug>/                    # One directory per experiment
      experiment.md                       # Narrative page
      metadata.json                       # Orchestrator-created metadata
      diagnostic_vector.json              # TPUDiagnoseAgent output
      numeric_equiv_result.json           # Numeric-equiv test output
      hlo_scope_test.json                 # MaxKernel HLO scope test
      hlo_pre.txt                         # Pre-change HLO fragment
      hlo_post.txt                        # Post-change HLO fragment
      derivation.md                       # DeepResearch derivation
      constraint_log.md                   # Iterative constraint narrowing log
  raw/
    profiles/
      <YYYYMMDD>-<slug>/                  # xprof trace directory (gitignored)
        plugins/profile/...               # Standard xprof directory structure
    hlo/
      <YYYYMMDD>-<slug>/                  # HLO dumps per experiment (gitignored)
        pre_opt/                           # Pre-XLA-optimization HLO
        post_opt/                          # Post-XLA-optimization HLO
    xla_logs/
      <YYYYMMDD>-<slug>.log              # XLA compiler log per experiment (gitignored)
    isa/                                  # ISA documentation (read-only, human-provided)
      v6e_instruction_set.pdf             # TPU v6e ISA reference
      latency_tables/
        latency_tables.json               # Instruction latencies: structured JSON
    numeric_ref/                          # Numeric-equivalence reference corpus
      prompts.jsonl                       # 100-prompt corpus (one JSON object per line)
      reference_outputs/                  # Baseline outputs, one file per prompt
        prompt_000.json ... prompt_099.json
```

#### Artifact format specifications

**xprof traces** (`raw/profiles/<YYYYMMDD>-<slug>/`)
- Format: Binary xprof/TensorBoard trace directory
- Contains: `plugins/profile/` subtree with TPU trace events
- Produced by: Measurement run with profiling enabled
- Consumed by: TPUDiagnoseAgent (reads via xprof tools)
- Note: Gitignored. The experiment page is the persistent link.

**XLA compiler logs** (`raw/xla_logs/<YYYYMMDD>-<slug>.log`)
- Format: Plain text log
- Contains: Compilation passes, VREG allocation, spill indicators
  (search for `spill` or `LDST` to find spill counts)
- Produced by: XLA compiler during `jax.jit` compilation
- Consumed by: TPUDiagnoseAgent (VREG spill count), MaxKernel
  (constraint analysis)
- Note: Gitignored.

**HLO dumps** (`raw/hlo/<YYYYMMDD>-<slug>/`)
- Format: HLO IR text (`.txt` files)
- Structure: `pre_opt/` contains HLO before XLA optimization passes;
  `post_opt/` contains HLO after all passes
- Produced by: XLA compiler with `--xla_dump_to` and `--xla_dump_hlo_as_text`
- Consumed by: DeepResearch (commutativity analysis, fusion pattern scan),
  MaxKernel (scope test), GraphArchitect (barrier identification and payload verification)
- Note: Gitignored. Relevant fragments are copied to
  `experiments/<id>/hlo_pre.txt` and `hlo_post.txt` for persistence.

**ISA documentation** (`raw/isa/`)
- `v6e_instruction_set.pdf`: TPU v6e ISA reference manual
- `latency_tables/latency_tables.json`: Structured instruction latencies.
  Format:
  ```json
  {
    "vmatmul":  { "latency_cycles": <N>, "tile_shape": [128, 128], "unit": "MXU" },
    "vmatpush": { "latency_cycles": <N>, "tile_shape": [128, 128], "unit": "MXU" },
    "vfdot":    { "latency_cycles": <N>, "unit": "VPU" },
    "vrep":     { "latency_cycles": <N>, "unit": "VPU" },
    ...
  }
  ```
- Consumed by: DeepResearch (M3, M6, M7), MaxTile (M3, M6)
- Note: Human-provided before MaxPerf starts. Agents do not modify.
  If a needed instruction is missing, file a request to the human.

**Numeric-equivalence corpus** (`raw/numeric_ref/`)
- `prompts.jsonl`: 100 prompts, one JSON object per line:
  ```json
  {"prompt_id": 0, "text": "<prompt text>", "max_tokens": 256}
  ```
- `reference_outputs/prompt_NNN.json`: Per-prompt reference output:
  ```json
  {"prompt_id": 0, "tokens": [...], "logits_hash": "<hash>", "final_loss": <float>}
  ```
- Produced by: Phase 0 baseline run (before MaxPerf starts)
- Consumed by: GraphArchitect, MaxKernel (numeric-equiv gate)
- Tolerance: bf16 numeric tolerance per `program.md` binding 7

**HBM bandwidth counters**
- Source: Extracted from xprof trace by TPUDiagnoseAgent
  (not a separate artifact — embedded in the xprof trace data)
- TPUDiagnoseAgent reads these via xprof tool queries, same
  as roofline and per-op breakdown

**Diagnostic vector** (`experiments/<id>/diagnostic_vector.json`)
- Format:
  ```json
  {
    "experiment_id": "<YYYYMMDD>-<slug>",
    "roofline": {
      "side": "compute|memory",
      "distance_from_peak_pct": <float>,
      "binding_resource": "<resource>"
    },
    "headroom": [
      {"op": "<name>", "pct_step": <float>, "fused": <bool>,
       "shape": "<shape>", "category": "<category>"}
    ],
    "dma_idle_pct": <float>,
    "hbm_bw_utilization_pct": <float>,
    "vreg_spill_count": <int>,
    "collective_latency": [
      {"collective": "<type>", "count": <int>,
       "total_ms": <float>, "source_op": "<op>"}
    ]
  }
  ```

**Numeric-equiv result** (`experiments/<id>/numeric_equiv_result.json`)
- Format:
  ```json
  {
    "pass": true|false,
    "prompts_checked": 100,
    "max_deviation": <float>,
    "tolerance": <float>,
    "failed_prompts": [
      {"prompt_id": <int>, "deviation": <float>}
    ]
  }
  ```

**HLO scope test** (`experiments/<id>/hlo_scope_test.json`)
- Format:
  ```json
  {
    "target_pattern": "<name>",
    "xla_fused_op_found": true|false,
    "fused_op_name": "<name>"|null,
    "verdict": "in-scope|out-of-scope",
    "fusion_context": {
      "surrounding_fusions": ["<fusion_name>", ...],
      "estimated_fusion_tax": "<description>"
    }
  }
  ```

**Constraint log** (`experiments/<id>/constraint_log.md`)
- Format: Append-only markdown, one section per iteration:
  ```markdown
  ## Iteration N
  - Constraint identified: <specific hardware constraint>
  - Evidence: <error trace, corruption pattern>
  - Fix applied: <what was changed>
  - Result: PASS / FAIL
  ```

### Artifact inventory — READ-ONLY

Complete inventory of all artifacts in the system, by category.

#### Hardware artifacts (binary, from infrastructure)

| Artifact | Format | Path | Consumed by | Produced by |
|----------|--------|------|-------------|-------------|
| xprof trace | Binary xprof dir | `raw/profiles/<YYYYMMDD>-<slug>/` | TPUDiagnoseAgent | Measurement run |
| XLA compiler log | Plain text | `raw/xla_logs/<YYYYMMDD>-<slug>.log` | TPUDiagnoseAgent, MaxKernel | XLA compiler |
| HBM bandwidth counters | Embedded in xprof | (extracted from xprof trace) | TPUDiagnoseAgent | TPU hardware monitoring |

#### HLO dumps (text)

| Artifact | Format | Path | Consumed by | Produced by |
|----------|--------|------|-------------|-------------|
| Pre-XLA-optimization HLO | HLO IR text | `raw/hlo/<YYYYMMDD>-<slug>/pre_opt/` | DeepResearch, AutoRefactor | XLA compiler |
| Post-XLA-optimization HLO | HLO IR text | `raw/hlo/<YYYYMMDD>-<slug>/post_opt/` | DeepResearch, MaxKernel, AutoRefactor | XLA compiler |
| Pre-change HLO fragment | HLO IR text | `experiments/<id>/hlo_pre.txt` | Orchestrator (verdict) | MaxShard, AutoRefactor |
| Post-change HLO fragment | HLO IR text | `experiments/<id>/hlo_post.txt` | Orchestrator (verdict) | MaxShard, AutoRefactor |

#### ISA & hardware documentation (external, read-only)

| Artifact | Format | Path | Consumed by | Produced by |
|----------|--------|------|-------------|-------------|
| TPU v6e ISA reference | PDF | `raw/isa/v6e_instruction_set.pdf` | DeepResearch, MaxKernel | Human-provided |
| Instruction-latency tables | JSON | `raw/isa/latency_tables/latency_tables.json` | DeepResearch, MaxTile | Human-provided |

#### Numeric-equivalence corpus (fixed reference)

| Artifact | Format | Path | Consumed by | Produced by |
|----------|--------|------|-------------|-------------|
| 100-prompt corpus | JSONL | `raw/numeric_ref/prompts.jsonl` | MaxShard, MaxKernel, AutoRefactor | Human (fixed before MaxPerf) |
| Reference outputs | JSON (per prompt) | `raw/numeric_ref/reference_outputs/prompt_NNN.json` | MaxShard, MaxKernel, AutoRefactor | Phase 0 baseline run |
| HumanEval test suite | Standard benchmark | (external — standard install) | Orchestrator | OpenAI (fixed) |
| MBPP test suite | Standard benchmark | (external — standard install) | Orchestrator | Google (fixed) |

#### Code & config (from repository)

| Artifact | Format | Path | Consumed by | Produced by |
|----------|--------|------|-------------|-------------|
| Model source code | Python (JAX/Flax) | On experiment branch | MaxShard, MaxKernel, AutoRefactor | Repository |
| Phase 0 best config | Shell script | `runs/phase0-best.sh` | Orchestrator | Phase 0 output |

#### Inter-agent artifacts (produced within the loop)

| Artifact | Format | Path | Consumed by | Produced by |
|----------|--------|------|-------------|-------------|
| Diagnostic vector | JSON | `experiments/<id>/diagnostic_vector.json` | Orchestrator, all agents (prior vectors) | TPUDiagnoseAgent |
| Baseline diagnostic vector | JSON | Path in `program.md` Baseline section | TPUDiagnoseAgent | Phase 0 measurement |
| Derivation (proof or ISA model) | Markdown | `experiments/<id>/derivation.md` | MaxShard, MaxTile | DeepResearch |
| Numeric-equiv result | JSON | `experiments/<id>/numeric_equiv_result.json` | Orchestrator | MaxShard, MaxKernel, AutoRefactor |
| HLO scope test result | JSON | `experiments/<id>/hlo_scope_test.json` | Orchestrator | MaxKernel |
| Constraint log | Markdown | `experiments/<id>/constraint_log.md` | MaxKernel (next iter), MaxShard (next iter) | MaxKernel, MaxShard |
| Experiment metadata | JSON | `experiments/<id>/metadata.json` | Orchestrator | Orchestrator |

#### Governance artifacts

| Artifact | Format | Path | Consumed by | Produced by |
|----------|--------|------|-------------|-------------|
| Operating contract | Markdown | `program.md` | All agents (every turn) | Human + orchestrator |
| Agent prompts (7 files) | Markdown | `agents/*.md` | Gemini (role context) | Human |
| Cumulative ledger | TSV (18 columns) | `RESULTS.tsv` | Orchestrator | Orchestrator |
| Experiment page | Markdown | `experiments/<id>/experiment.md` | Orchestrator, agents (context) | Implementing agent |

### Prerequisites checklist — READ-ONLY

Before MaxPerf loop starts, the human must ensure these artifacts exist:

| Artifact | Path | Status |
|----------|------|--------|
| ISA reference PDF | `raw/isa/v6e_instruction_set.pdf` | `N/A (Replaced by Roofline)` |
| Instruction-latency tables | `raw/isa/latency_tables/latency_tables.json` | `N/A (Replaced by Roofline)` |
| Roofline analysis worksheet | `DeepSeekV3 Roofline Analysis Extended (Inference).xlsx` | `Done` |
| 100-prompt corpus | `raw/numeric_ref/prompts.jsonl` | Done (converted from `inputs/numeric_equiv_corpus_seed.txt`) |
| Reference outputs | `raw/numeric_ref/reference_outputs/` | `<TBD>` |
| Phase 0 best config | `runs/phase0-best.sh` | `<TBD>` |
| Phase 0 baseline diagnostic vector | (path TBD in Baseline section) | `<TBD>` |
| HumanEval + MBPP installed | (standard benchmark install) | `<TBD>` |

---

## Hypothesis Taxonomy — READ-ONLY

Every queue entry has a class. The class determines the authoring sub-agent,
accept/reject criteria, and what counts as evidence.

| Class | Sub-agent | Origination | Accept criteria | Reject criteria |
|-------|-----------|-------------|-----------------|-----------------|
| `config-tune` | (Phase 0 — rare in MaxPerf) | Profile | TPS up ≥X% within TPOT budget | TPS flat ±2% |
| `kernel-autotune` | MaxTile | Symbolic preferred, profile fallback | Bottleneck bucket down ≥Y%, no end-to-end regression | Default already optimal OR sweep variance exceeds gain |
| `graph-rewrite` | MaxShard | Symbolic | Numeric-equiv passes AND TPS up OR collective payload down ≥Z% | Numeric-equiv fails OR payload reduction doesn't translate to runtime |
| `code-refactor` | AutoRefactor | Profile | Numeric-equiv passes AND HLO barrier resolved AND TPS non-negative | Numeric-equiv fails OR HLO unchanged OR TPS down |
| `kernel-novel` | MaxKernel | Profile or iterative-debug | Numeric-equiv passes AND target bucket down ≥W% AND no VREG-spill regression | Numeric-equiv fails after 5 iterations OR bucket flat OR new VREG spill |

Thresholds `X`, `Y`, `Z`, `W` are set per experiment by the authoring agent
based on the diagnostic vector; defaults are 3% for kernel-level, 5% for
end-to-end gains.

---

## Theoretical Methods — READ-ONLY

These are the named reasoning methods the system uses. Each method maps
to one or more agents. Agents apply their assigned methods systematically;
they do not invent ad-hoc reasoning.

### Core methods (from the four pattern groups)

| # | Method | What it does | Primary agent(s) | Pattern group |
|---|--------|-------------|-------------------|---------------|
| M1 | **Bottleneck Analysis & Symbolic Graph Refactoring** | Recognize commutative execution-graph nodes; move reductions earlier to shrink collective payloads. Algebraic reasoning over graph structure, not pattern-matching. | DeepResearch (proof), MaxShard (implementation) | Group 1 |
| M2 | **Compiler Overhead Detection & Algorithmic Exploration** | Detect VREG spilling, DMA idling, and suboptimal scheduling from XLA compiler logs and xprof traces. Explore algorithmic workarounds (e.g., bin-packing) to balance uneven workloads. | TPUDiagnoseAgent (detection), MaxKernel (kernel fixes), DeepResearch (algorithm design) | Group 2 |
| M3 | **Empirical Roofline Latency Modeling** | Query Peak Compute (P) and Memory Bandwidth (B) parameters, calculate operational intensity (FLOPs/Byte), and derive absolute execution bounds for sharded operations. Replaces cycle-level ISA simulation with hardware-capacity limit equations. | DeepResearch (model), MaxTile (verification) | Group 3 |
| M4 | **Execution Graph Unblocking** | Identify dynamic runtime branches that halt compiler optimization (fusions, inlining, unrolling). Trace barriers to Python source and rewrite using compile-time-resolvable constructs. | AutoRefactor | Group 4 |

### Cross-cutting methods (applicable across all groups)

| # | Method | What it does | Primary agent(s) | When to apply |
|---|--------|-------------|-------------------|---------------|
| M5 | **Iterative Constraint Narrowing** | On numeric-equiv failure, systematically identify the specific hardware constraint violated (sublane width, alignment, bank conflict, data-packing rule), document it, fix it, and retry. Each iteration eliminates one constraint from the search space. The accumulated constraint set becomes reusable knowledge. | MaxKernel (5 iterations), MaxShard (3 iterations) | Every numeric-equiv failure on structural changes |
| M6 | **Pipeline Stage Balancing** | Given two or more functional units with different throughput (MXU vs VPU, matmul vs softmax), model their latencies algebraically and size work tiles so stage latencies equalize — minimizing pipeline bubbles. General form of hypothesis queue item 8. | DeepResearch (model), MaxTile (tile sizing) | Any kernel with heterogeneous functional units |
| M7 | **Payload Minimization via Algebraic Commutativity** | For any `collective(f(x))` where `f` is a reduction, systematically check whether `f` and the collective commute. If they do, rewrite as `f(collective(x))` or eliminate the collective entirely by passing metadata. Scan the full graph for all collective → reduction pairs, not just known instances. | DeepResearch (commutativity scan), MaxShard (implementation) | Whenever diagnostic vector shows collective latency as a top contributor |
| M8 | **Compile-Time Specialization Ladder** | Apply a priority-ordered resolution ladder to every barrier: (a) resolve at Python parse time → (b) resolve at JAX trace time → (c) resolve at XLA compile time → (d) accept as runtime. Each rung eliminates a class of HLO barriers. Work top-down: try the highest rung first. | AutoRefactor | Every code-refactor hypothesis |
| M9 | **Fusion Boundary Cost Accounting** | Before proposing any Pallas kernel replacement, account for the fusion context the target op sits in — not just the op in isolation. A custom-call boundary breaks surrounding XLA fusions (loop-fusion, conv-fusion). If `fusion_tax > kernel_gain`, the kernel regresses even when it's faster in isolation. Extends binding 5 scope test with a quantitative fusion-context check. | MaxKernel (pre-implementation), AutoRefactor (when evaluating refactors that change fusion boundaries) | Every kernel-novel hypothesis, and any code-refactor that may alter XLA fusion topology |

### How methods connect to the loop

- **Discovery phase** (loop step 6): TPUDiagnoseAgent applies M2 to produce
  the diagnostic vector. Profile-grounded hypotheses reference the method
  that would address the flagged inefficiency.
- **Planning phase** (loop step 3, DeepResearch): applies M1, M3, M6, M7
  to produce symbolic-grounded hypotheses with derivations.
- **Implementation phase** (loop step 3, implementers): MaxShard applies M1/M7,
  MaxKernel applies M2/M5/M9, MaxTile applies M3/M6, AutoRefactor applies M4/M8.
- **Debug phase** (iterative-debug path): MaxKernel and MaxShard apply M5.
- **Pre-implementation screening** (before loop step 3): MaxKernel applies M9
  alongside the binding 5 scope test.

### Method provenance — READ-ONLY

This is a clean-room optimization effort. No method is derived from
publications specific to Qwen3-Coder-480B (per binding 1). Each method
traces to independent, model-agnostic sources:

| Method | Origin | Independent sources |
|--------|--------|-------------------|
| M1 — Symbolic Graph Refactoring | Distributed computing algebra. Reducing before collecting is standard MapReduce optimization. | Dean & Ghemawat 2004 (MapReduce); MPI collective optimization literature; GShard (Lepikhin et al. 2020); Tutel (Hwang et al. 2023) |
| M2 — Compiler Overhead Detection | TPU compiler diagnostics. XLA compiler logs expose spill counts; xprof exposes DMA idle. Standard TPU profiling practice. | Google xprof documentation; XLA compiler source; TPU best practices guides |
| M3 — Algebraic ISA Modeling | Hardware performance modeling. Building throughput models from ISA latency tables predates GPUs — same method as CPU cache-line analysis. | Williams et al. 2009 (Roofline model); Ofenbeck et al. 2014 (roofline on accelerators); computer architecture textbooks |
| M4 — Execution Graph Unblocking | JAX/XLA compilation model. Trace-time resolution of Python control flow is documented in JAX's own guides. | JAX documentation ("Sharp Bits"); XLA fusion documentation; standard JIT compiler design |
| M5 — Iterative Constraint Narrowing | Hardware debugging methodology. Systematic constraint elimination is standard embedded/FPGA debug practice. | General engineering practice; no specific paper — standard debug methodology for hardware constraint violations |
| M6 — Pipeline Stage Balancing | Pipeline scheduling theory. Balancing heterogeneous stages to minimize bubbles is textbook computer architecture. | Hennessy & Patterson (Computer Architecture); Gpipe (Huang et al. 2019); pipeline parallelism literature |
| M7 — Payload Minimization via Commutativity | Generalized form of M1. Systematic scan for commutative collective-reduction pairs is the same principle as pushing selections below joins in databases. | Same as M1, plus relational algebra optimization (Ullman 1988); query optimization literature |
| M8 — Compile-Time Specialization Ladder | Partial evaluation theory. Resolving values at the earliest possible stage is Futamura projections / multi-stage programming. | Jones et al. 1993 (Partial Evaluation); Taha & Sheard 1997 (multi-stage programming); Scala staging literature |
| M9 — Fusion Boundary Cost Accounting | Discovered empirically in this project's Gemma4 campaign — exp33 (Pallas RMSNorm, -8.1%) and exp47 (Levanter CE, -5.6%) independently showed custom-call boundaries breaking XLA fusion. | This repo (Gemma4 experiments 33 and 47); XLA custom-call semantics documentation (custom calls are fusion barriers by design) |

---

## Hypothesis Queue — R/W (MaxPerf only)

Items listed in rough priority order. Each carries: class, origination path,
evidence pointer, expected gain, accept rule, reject rule, dependency,
authoring sub-agent.

The queue is pre-seeded with hypothesis classes from the four MaxPerf pattern
groups. Sub-agents append profile-grounded and symbolic-grounded items as the
loop runs and traces accumulate.

### Standing Queue

#### Group 1 — Graph & Communication

**1. Eliminate input all-to-all in EP forward path via reduction-collective swap.**

| Field | Value |
|-------|-------|
| Class | `graph-rewrite` |
| Origination | Symbolic |
| Author | GraphArchitect (load `agents/graph_architect.md`) |
| Evidence | DeepResearch derivation: reduction and collective commute under EP forward; payload shrinks from `(B*k, D)` to `(B, D)` |
| Expected | Collective-latency bucket down 30-60%; end-to-end TPS up 5-15% |
| Accept | Numeric-equiv passes AND collective bucket down ≥30% AND end-to-end TPS non-negative |
| Reject | Numeric-equiv fails (file iterative-debug back to GraphArchitect) OR collective bucket flat |
| Dependency | Phase 0 baseline complete, diagnostic vector captured |

**2. Fuse ragged unpermute and reduce in EP backward path.**

| Field | Value |
|-------|-------|
| Class | `graph-rewrite` + `kernel-novel` (composite) |
| Origination | Symbolic + profile |
| Author | GraphArchitect (graph change) + MaxKernel (fused kernel) |
| Evidence | DeepResearch: unpermute and reduce can fuse; TPUDiagnoseAgent: unfused form spends >5% step time on intermediate materialization |
| Expected | Ragged-bucket time down 15-25% |
| Accept/Reject | Per `kernel-novel` template |
| Dependency | Item 1 settled |

#### Group 2 — Kernel & Latency Hiding

**3. GMM dynamic tiling for routing-imbalance amortization.**

| Field | Value |
|-------|-------|
| Class | `kernel-novel` |
| Origination | Profile |
| Author | MaxKernel + MaxTile (kernel implementation; tile params from algebraic derivation) |
| Evidence | TPUDiagnoseAgent: DMA idle % spikes correlated with expert-routing imbalance across decode steps |
| Expected | GMM bucket time down 10-20% on imbalanced steps; end-to-end 3-8% |
| Dependency | Phase 0 best config (fixes batching and concurrency) |

**4. Pipelined quantized matmul (MXU) and dequantization (VPU).**

| Field | Value |
|-------|-------|
| Class | `kernel-novel` |
| Origination | Profile |
| Author | MaxKernel (load `agents/max_kernel.md`) |
| Evidence | Profile: MXU stalled waiting for VPU dequant; units are separately addressable per ISA |
| Expected | MXU utilization up 10-25% during quant matmul; end-to-end 3-7% |
| Dependency | None (parallelizable with item 3) |

**5. Attention sequence batching via greedy bin-packing.**

| Field | Value |
|-------|-------|
| Class | `graph-rewrite` + `kernel-novel` |
| Origination | Symbolic |
| Author | GraphArchitect (scheduling logic) + MaxKernel (packed-attention kernel if needed) |
| Evidence | DeepResearch: ragged sequence lengths shift attention from compute-bound to communication-bound; bin-packing rebalances |
| Expected | Attention bucket time down 10-20% in long-context regimes |
| Dependency | Workload regime is long-context (otherwise reject pre-experiment) |

**6. Custom ragged gather/scatter kernels for FP8 + 32-bit sublane data-packing.**

| Field | Value |
|-------|-------|
| Class | `kernel-novel` |
| Origination | Profile (initial) + iterative-debug (subsequent) |
| Author | MaxKernel (load `agents/max_kernel.md`) |
| Evidence | Profile: generic gather/scatter incurs uncoalesced reads; iterative-debug logs identify race conditions with FP8 writes on 32-bit sublanes |
| Expected | Gather/scatter bucket time down 30-50% |
| Iteration limit | 5 numeric-check failures terminates (per MaxKernel spec) |
| Dependency | Item 3 (GMM dynamic tiling) settled |

#### Group 3 — Tile Search & Hardware Thresholds

**7. GMM latch-bound threshold via algebraic ISA modeling.**

| Field | Value |
|-------|-------|
| Class | `kernel-autotune` |
| Origination | Symbolic |
| Author | MaxTile (load `agents/max_tile.md`) with DeepResearch supplying ISA model |
| Evidence | DeepResearch derivation of latch-bound threshold using vmatmul/vmatpush latency tables |
| Expected | GMM batch tile = 128 (verify); GMM bucket time within 2% of algebraic optimum |
| Accept | Measured GMM time matches algebraic prediction within 5% |
| Reject | Prediction off by >10% — ISA model incomplete; file follow-up to DeepResearch |
| Dependency | Items 3 and 4 settled |

**8. Matmul vs softmax latency balancing for attention kernel.**

| Field | Value |
|-------|-------|
| Class | `kernel-autotune` |
| Origination | Symbolic |
| Author | MaxTile (load `agents/max_tile.md`) |
| Evidence | ISA-modeled latency comparison of the two pipeline stages |
| Expected | 3-8% on attention bucket from rebalancing |
| Dependency | Items 5 and 7 settled |

#### Group 4 — Compiler Optimization

**9. Push static MoE-config logic into `@dataclass(frozen=True)`.**

| Field | Value |
|-------|-------|
| Class | `code-refactor` |
| Origination | Profile |
| Author | GraphArchitect (load `agents/graph_architect.md`) |
| Evidence | HLO dump: runtime branches on config values that are statically known at trace time |
| Expected | HLO size down, compile-time down 5-15%, runtime TPS flat-or-up |
| Accept | Per `code-refactor` template |
| Dependency | None |

**10. Replace Python `if` in routing path with `jnp.where` / `lax.cond`.**

| Field | Value |
|-------|-------|
| Class | `code-refactor` |
| Origination | Profile |
| Author | GraphArchitect (load `agents/graph_architect.md`) |
| Evidence | HLO: control-flow barriers in expert routing |
| Expected | Routing-bucket time down 5-15% |
| Dependency | Item 9 settled |

**11. Fuse GQA KV head repetition inside Pallas attention kernel.**

| Field | Value |
|-------|-------|
| Class | `kernel-novel` |
| Origination | Profile |
| Author | MaxKernel (load `agents/max_kernel.md`) |
| Evidence | TPUDiagnoseAgent: `broadcast_in_dim` takes 31.70% (decode) and 25.32% (prefill) of TPU time, while attention Pallas kernel takes only 6.05% (decode). GQA broadcasts KV heads from 2 to 16 heads. |
| Expected | `broadcast_in_dim` TPU time reduced by 50-80%; decode step latency improved by 15-25% |
| Dependency | Custom Pallas attention kernel path enabled |

**12. Fuse token/KV cache slice and concatenate operations using custom paged-cache copy kernel.**

| Field | Value |
|-------|-------|
| Class | `kernel-novel` |
| Origination | Profile |
| Author | MaxKernel (load `agents/max_kernel.md`) |
| Evidence | TPUDiagnoseAgent: `concatenate` takes 15.97% (prefill) / 20.20% (decode) and `slice` takes 8.42% (prefill) / 10.59% (decode) of TPU time. Fusing these into a single paged-cache update kernel avoids independent HBM copy-back passes. |
| Expected | `concatenate` and `slice` TPU overhead reduced by 60-90%; decode step latency improved by 15-30% |
| Dependency | Custom Pallas attention/cache kernel path enabled |

~~**13. Asynchronous logits sampling and host-device execution pipelining.**~~ (Resolved in [20260618-async-sampling-pipelining](experiments/20260618-async-sampling-pipelining/experiment.md))

| Field | Value |
|-------|-------|
| Class | `code-refactor` |
| Origination | Profile |
| Author | GraphArchitect (load `agents/graph_architect.md`) |
| Evidence | TPUDiagnoseAgent: TPU device idle time is 91.9% in prefill and 73.0% in decode, indicating host-device sync starvation. |
| Expected | TPU device duty cycle increases to >60%, resulting in 1.5x - 2.0x increase in serving throughput (TPS) |
| Dependency | None |

### Refuted — APPEND-ONLY

Items moved here when an experiment refutes them. Each entry: class,
short name, date refuted, link to experiment page, one-line reason,
generalizable lesson if any.

- **graph-rewrite** | Eliminate input all-to-all in EP forward path | 2026-06-03 | [20260603-eliminate-ep-all2all](experiments/20260603-eliminate-ep-all2all/experiment.md) | Target model Qwen2.5-Coder-3B-Instruct is a dense model and has no EP forward path or all-to-all collectives. | MoE/EP-specific communication and kernel optimization hypotheses do not apply to dense models.
- **graph-rewrite** | Fuse ragged unpermute and reduce in EP backward path | 2026-06-03 | [20260603-fuse-ep-backward](experiments/20260603-fuse-ep-backward/experiment.md) | Target model Qwen2.5-Coder-3B-Instruct is a dense model and has no EP backward path or unpermute operations. | MoE/EP-specific communication and kernel optimization hypotheses do not apply to dense models.
- **graph-rewrite** | Attention sequence batching via greedy bin-packing | 2026-06-03 | [20260603-attention-greedy-binpacking](experiments/20260603-attention-greedy-binpacking/experiment.md) | Workload sequence length is 1024 (not long-context) and engine runs on PyTorch/XLA fallback without Pallas ragged attention kernels. | Graph-level and kernel-level attention bin-packing requires long-context workload and custom Pallas kernel stack.
- **kernel-autotune** | GMM latch-bound threshold | 2026-06-03 | [20260603-gmm-latch-bound](experiments/20260603-gmm-latch-bound/experiment.md) | Target model Qwen2.5-Coder-3B-Instruct has no MoE/GMM layers, and dynamic-tiled GMM kernel dependency (Item 3) is unresolved. | MoE/GMM-specific autotuning is inapplicable to dense models.
- **kernel-autotune** | Matmul vs softmax latency balancing for attention kernel | 2026-06-03 | [20260603-attention-latency-balancing](experiments/20260603-attention-latency-balancing/experiment.md) | Target model runs on PyTorch/XLA engine fallback without custom Pallas attention kernels, and attention dependencies (Items 5 and 7) are unresolved. | Attention-kernel autotuning requires custom Pallas attention kernel stack.
- **code-refactor** | Push static MoE-config logic into @dataclass(frozen=True) | 2026-06-03 | [20260603-dataclass-moe-config](experiments/20260603-dataclass-moe-config/experiment.md) | Target model Qwen2.5-Coder-3B-Instruct has no MoE/GMM configuration logic. | MoE/GMM-specific config optimization does not apply to dense models.
- **code-refactor** | Replace Python if in routing path with jnp.where / lax.cond | 2026-06-03 | [20260603-replace-python-if-routing](experiments/20260603-replace-python-if-routing/experiment.md) | Target model Qwen2.5-Coder-3B-Instruct has no expert routing path, and routing dependency (Item 9) is unresolved. | MoE/GMM-specific control flow optimization does not apply to dense models.

### Wins So Far — APPEND-ONLY

Each accepted experiment appends a one-liner: class, short name,
cumulative TPS/chip after the change, primary diagnostic-vector delta.

- **code-refactor** | Asynchronous sampling pipelining | [20260618-async-sampling-pipelining](experiments/20260618-async-sampling-pipelining/experiment.md) | TPS: TBD (Warmup successful) | batch-queue-size: +1 (from 1 to 2)

---

## Open Questions — R/W (MaxPerf only)

1. **Numeric-equivalence corpus.** The 100-prompt corpus for
   structural-change verification — pin the prompt list and reference
   outputs before MaxPerf engages.

2. **ISA documentation access.** DeepResearch needs vmatmul, vmatpush,
   and other instruction-latency tables. Confirm read access.

3. **VREG-spill threshold for kernel-novel rejection.** Default proposed:
   any new spill in a hot kernel = reject. Confirm before kernel-novel
   work starts.

4. **Compile-time budget.** Proposed: compile time must not increase by
   >25% to accept. Confirm.

5. **Iteration count on iterative-debug hypotheses.** MaxKernel default
   is 5; proposed 3 for GraphArchitect (graph rewrites/refactors have smaller search
   space). Confirm.

---

## RESULTS.tsv Columns

```
date    class    branch    experiment_slug    agent    verdict    tps_per_chip    p50_tpot_ms    p99_tpot_ms    hbm_peak_gib    diagnostic_vector_delta    numeric_equiv_pass    vreg_spill_delta    compile_time_s    eval_humaneval    eval_mbpp    profile_gcs_path    notes
```

Tab-separated. `class` carries the hypothesis class. `agent` carries the
authoring sub-agent. `diagnostic_vector_delta` is a short structured string
(e.g., `headroom-collective:-32%,vreg-spill:0,hbm-bw:+8%`).

---

## Loop Protocol

Each loop iteration is orchestrated by MaxPerf (load
`agents/maxperf_orchestrator.md` for full orchestrator instructions):

| Step | Action | Agent |
|------|--------|-------|
| 1 | **Pull** — Read this file, pick top hypothesis with satisfied dependencies. For composite hypotheses, assemble sub-agent team per taxonomy. If the queue is empty, or all remaining hypotheses are blocked, refuted, or inapplicable to the target architecture (dense vs. MoE), trigger `TPUDiagnoseAgent` on the current best configuration to profile and generate new profile-grounded hypotheses. | MaxPerf |
| 2 | **Branch** — Create `qwen3coder-maxperf-YYYYMMDD-<slug>` off Phase 0 best. | MaxPerf |
| 3 | **Author** — Implement the hypothesis. For symbolic-grounded, DeepResearch provides derivation but does not author implementation. | Per taxonomy table |
| 4 | **Numeric-equiv** — (graph-rewrite, code-refactor, kernel-novel only) Run 100-prompt corpus, compare to reference. If fail → file iterative-debug hypothesis, reject. | Authoring agent |
| 5 | **Launch** — Standard run on v6e-16 with warmup ≥200 requests. | MaxPerf |
| 6 | **Diagnose** — Run profiling, produce full diagnostic vector. File new hypotheses if any by writing them as markdown files under `wiki/hypotheses/<slug>.md`. | TPUDiagnoseAgent (load `agents/tpu_diagnose.md`) |
| 7 | **Eval** — Run HumanEval + MBPP correctness gate. | MaxPerf |
| 8 | **Verdict** — Apply class-appropriate accept/reject template using diagnostic vector and eval result. | MaxPerf |
| 9 | **Author experiment page** — Write per SCHEMA.md. Include `Agent` field and diagnostic vector in profile section. | Authoring agent |
| 9b | **Extract observations** — If the experiment's profiling or analysis revealed a finding with implications beyond this specific experiment (e.g., a hardware constraint, a fusion pattern, a performance cliff), the authoring agent should extract it into a reusable observation page in `wiki/observations/` following the observation page format (frontmatter with `type: observation`, a `## What was observed` section, and links to the originating experiment). Not every experiment produces an observation — only extract when the finding is generalizable. For iterative-debug experiments (Method M5), this extraction is mandatory, not optional: every constraint identified in `constraint_log.md` must be filed as an observation page in `wiki/observations/` (or appended to an existing one if the constraint class already has a page), so that future experiments on unrelated hypotheses can avoid re-discovering the same hardware constraints. | Authoring agent |
| 10 | **Append RESULTS.tsv row** — Per column spec above. | MaxPerf |
| 11 | **Update queue** — Strike resolved item, append to Wins/Refuted, merge sub-agent proposals, linking to their `wiki/hypotheses/<slug>.md` pages in the queue. | MaxPerf |
| 12 | **Commit** — On experiment branch. Merge to main only on accepted results. | MaxPerf |

### Halt Rules

- After **three consecutive non-wins within a hypothesis class**: pause
  that class, rotate to next-priority class.
- After **three consecutive non-wins across classes**: stop and write
  a ceiling analysis.
- After **120 total experiments** (regardless of win rate): stop and
  write a ceiling analysis.

The ceiling analysis covers:
- Which axes are exhausted
- Which diagnostic-vector entries remain unimproved
- What would be required to break the ceiling (hardware change,
  out-of-scope quantization, new hypothesis class)
- The ceiling analysis is surfaced to the human. MaxPerf does not
  autonomously expand scope.
