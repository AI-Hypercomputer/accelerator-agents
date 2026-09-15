<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# How a Dispatch to MaxKernel Actually Works

A step-by-step walkthrough of what happens mechanically when the orchestrator routes a hypothesis to MaxKernel.

---

## The Trigger

The orchestrator is running the 12-step loop. In **Step 1 (Pull)**, it scans the hypothesis queue in `program.md` and finds the top-priority item with satisfied dependencies:

```
Hypothesis: "Fuse GeLU + Linear2 in MLP block via Pallas kernel"
Class:      kernel-novel
Priority:   1 (highest)
Dependencies: [diagnostic_vector exists, HLO dump available]
```

The taxonomy table says `class: kernel-novel → agent: MaxKernel → prompt: agents/max_kernel.md`.

---

## Step 2: Branch

The orchestrator creates a git branch:
```
git checkout -b qwen3coder-maxperf-20260510-gelu-linear-fusion
mkdir experiments/20260510-gelu-linear-fusion/
```

---

## Step 3: Assemble the HANDOFF

The orchestrator constructs a structured context block — everything MaxKernel needs to do its job without asking questions:

```
HANDOFF {
  experiment_id:    "20260510-gelu-linear-fusion"
  hypothesis:       "Fuse GeLU + Linear2 in MLP block via Pallas kernel"
  class:            "kernel-novel"
  origination:      "profile"     ← came from TPUDiagnoseAgent's boundary accounting
  branch:           "qwen3coder-maxperf-20260510-gelu-linear-fusion"
  artifact_dir:     "experiments/20260510-gelu-linear-fusion/"

  evidence: {
    hlo_dump_dir:       "raw/hlo/20260510-gelu-linear-fusion/"
    xprof_trace_dir:    "raw/profiles/20260510-gelu-linear-fusion/"
    isa_doc:            "raw/isa/v6e_instruction_set.pdf"
    latency_tables:     "raw/isa/latency_tables/latency_tables.json"
  }

  baseline: {
    tps_per_chip:   142.3
    p50_tpot_ms:    7.03
    diagnostic_vector_path: "experiments/20260509-baseline/diagnostic_vector.json"
  }

  iteration:        null         ← first attempt, not iterative-debug
  prior_constraints: []
  accept_rule:      "tps_per_chip > 142.3 AND numeric_equiv_pass = true AND vreg_spill_delta <= 0"
  reject_rule:      "numeric_equiv_pass = false OR vreg_spill_delta > 0 OR hlo_scope_verdict = out-of-scope"
}
```

---

## Step 3 (continued): Load MaxKernel's Role Context

The orchestrator loads `agents/max_kernel.md` as the system prompt. This gives MaxKernel:
- Its identity and scope (Pallas custom-kernel sub-agent)
- Methods it owns (M5: Iterative Constraint Narrowing, M9: Fusion Boundary Cost Accounting)
- Hard rules (HLO scope test first, 5-iteration limit, no VREG spill regression)
- Output format requirements

The combined context is: **system prompt** (max_kernel.md) + **user message** (the HANDOFF above).

---

## What MaxKernel Does (Steps 3-4 internally)

MaxKernel executes its protocol:

### 1. HLO Scope Test
Reads the HLO dump. Searches for whether XLA already produces a fused GeLU+Linear kernel:
```
grep "fusion.*gelu.*dot" raw/hlo/20260510-gelu-linear-fusion/module.hlo
```
If it finds one → **OUT OF SCOPE**, returns immediately with `hlo_scope_verdict: "out-of-scope"`.

If not found → IN SCOPE, proceeds.

### 2. Fusion Boundary Cost Check (M9)
Reads the diagnostic vector. Confirms the boundary costs enough to justify a kernel:
```
GeLU→Linear2 boundary: [4096, 32768] bf16 = 256 MB × 2 / 820 GB/s = 0.624 ms/layer × 96 layers = 60 ms/step
```
60 ms > threshold → worth pursuing.

### 3. Write the Pallas Kernel
Using ISA latency tables and the algebraic model, MaxKernel designs and writes:
- Tile size selection (from constraint narrowing)
- DMA pipeline schedule (double-buffered)
- Sublane-compatible layout
- The actual Pallas Python code

### 4. Numeric Equivalence Test
Runs the kernel on the 100-prompt reference corpus (`raw/numeric_ref/prompts.jsonl`), compares outputs against reference:
```
max |new - ref| = 2.1e-6  →  PASS (within bf16 tolerance)
```

### 5. VREG Spill Check
Compiles and checks register usage:
```
Before: 0 spills in hot loop
After:  0 spills in hot loop
Delta:  0  →  PASS
```

---

## Step 3 (continued): MaxKernel Returns

MaxKernel assembles its RETURN:

```
RETURN {
  experiment_id:    "20260510-gelu-linear-fusion"
  agent:            "MaxKernel"
  status:           "complete"

  experiment_page:  "experiment.md"
  artifacts: {
    hlo_scope_test:     "hlo_scope_test.json"
    numeric_equiv:      "numeric_equiv_result.json"
  }

  measurements: {
    tps_per_chip:       148.7
    p50_tpot_ms:        6.72
    vreg_spill_delta:   0
    numeric_equiv_pass: true
  }

  hlo_scope_verdict:    "in-scope"
  proposed_hypotheses: [
    {
      name: "Fuse LayerNorm + QKV projection similarly",
      class: "kernel-novel",
      origination: "pattern",
      expected_gain: "0.15 ms/layer",
      priority: 3
    }
  ]
}
```

---

## Steps 5-12: Orchestrator Resumes

| Step | What happens |
|------|-------------|
| 5 (Launch) | Orchestrator runs a full measurement on v6e-16 with the new kernel |
| 6 (Diagnose) | TPUDiagnoseAgent produces updated diagnostic vector |
| 7 (Eval) | HumanEval + MBPP correctness gate (ensures model still works) |
| 8 (Verdict) | Check accept_rule: `148.7 > 142.3 ✓`, `numeric_equiv = true ✓`, `spill_delta = 0 ✓` → **ACCEPT** |
| 9 (Experiment page) | MaxKernel writes the narrative page |
| 10 (RESULTS.tsv) | Append row: `20260510-gelu-linear-fusion | MaxKernel | kernel-novel | +4.5% TPS | ACCEPTED` |
| 11 (Update queue) | Strike hypothesis, add MaxKernel's proposed follow-up to queue |
| 12 (Commit) | Merge branch to main |

---

## What Doesn't Exist Yet

The above is the **designed protocol**. What's missing to make it run:

1. **No executor** — nothing actually calls an LLM API with the HANDOFF as input and max_kernel.md as system prompt
2. **No state machine** — nothing tracks which step the loop is on, or manages branch/commit lifecycle
3. **No measurement infrastructure** — Step 5 assumes a TPU cluster is running and accepting jobs
4. **No Phase 0 baseline** — the baseline measurements are still `<TBD>`

The protocol is fully specified. The runtime that executes it is not built.

---

## The Simplest Possible Implementation

If you wanted to make this work today with Claude Code:

```
Human (playing orchestrator):
  "You are MaxKernel. Here is your HANDOFF: {paste JSON}.
   Your role instructions are in agents/max_kernel.md.
   Execute the protocol and return a RETURN block."
```

That's it — a human pastes the HANDOFF into a chat session with the agent prompt loaded. The "dispatch" is a copy-paste into a new context window. The system is designed for this to be automated later, but works manually now.
