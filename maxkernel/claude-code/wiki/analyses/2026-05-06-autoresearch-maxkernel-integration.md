---
title: "Autoresearch–MaxKernel Integration Design"
type: analysis
tags: [integration, maxkernel, architecture, autoresearch]
created: 2026-05-06
updated: 2026-05-06
---

Design document for integrating the autoresearch optimization loop with [MaxKernel](https://github.com/AI-Hypercomputer/accelerator-agents/tree/main/MaxKernel) as the kernel-authoring backend.

**Companion document**: [MaxKernel integration requirements](2026-05-06-maxkernel-integration-requirements.md) (changes needed on the MaxKernel side).

---

## Problem statement

The autoresearch loop ([SCHEMA.md](../../SCHEMA.md)) can identify when a custom Pallas kernel would improve performance — the HLO pre-filter (FORMULATE-HYPOTHESIS step 2b) confirms XLA doesn't already fuse the target ops, and the hypothesis ranking says the expected gain justifies the effort. But the loop currently cannot *write* kernels. It either:

- Defers ("file as follow-up hypothesis, effort L")
- Attempts an inline implementation (limited quality, no profiling, no autotuning)
- Asks the human to write it

MaxKernel is purpose-built for exactly this: given a kernel specification, it plans, implements, tests, profiles, and autotunes a Pallas kernel. Connecting the two systems closes the gap.

---

## Architecture overview

```
┌──────────────────────────────────────────────────────────┐
│  AUTORESEARCH LOOP (Claude Code / SCHEMA.md)             │
│                                                          │
│  FORMULATE-HYPOTHESIS                                    │
│    → HLO pre-filter (step 2b)                           │
│    → hypothesis passes → kernel needed                   │
│         │                                                │
│         ▼                                                │
│  ┌─────────────────────────────┐                         │
│  │  KERNEL DISPATCH            │                         │
│  │  1. Build kernel_spec.json  │                         │
│  │  2. Invoke MaxKernel        │                         │
│  │  3. Collect kernel_result   │                         │
│  └────────────┬────────────────┘                         │
│               │                                          │
│               ▼                                          │
│  RUN-EXPERIMENT                                          │
│    → Wire kernel into training step                      │
│    → Run 20-step experiment                              │
│    → Profile, measure, verdict                           │
│                                                          │
└──────────────────────────────────────────────────────────┘
         │ kernel_spec.json          ▲ kernel_result.json
         ▼                          │
┌──────────────────────────────────────────────────────────┐
│  MAXKERNEL (Google ADK, headless mode)                   │
│                                                          │
│  PlanKernelAgent                                         │
│    → ImplementKernelAgent                                │
│      → TestGenerationAgent + RunTestsAgent               │
│        → ProfileAgent                                    │
│          → AutotuneAgent                                 │
│            → kernel_result.json                          │
│                                                          │
│  Runs on same TPU host. Scoped to experiment workdir.    │
└──────────────────────────────────────────────────────────┘
```

---

## When the autoresearch agent invokes MaxKernel

MaxKernel invocation is triggered by a specific conjunction of conditions in the optimization loop:

1. A kernel-replacement hypothesis has **passed the HLO pre-filter** (`hlo_prefilter: passed`) — confirmed that XLA does NOT already fuse the target ops.
2. The hypothesis is **ranked high enough** to justify the effort (expected gain justifies the build cost).
3. **No existing kernel** covers the need — the [Pallas kernel directory](2026-04-23-pallas-kernel-directory.md) and the [tokamax codebase](../codebases/tokamax.md) have been checked.
4. The human has **approved** the hypothesis for testing (or the loop is running autonomously and the hypothesis clears the ranking threshold).

If all four conditions are met, the autoresearch agent dispatches to MaxKernel instead of attempting an inline implementation.

### Decision tree

```
Kernel-replacement hypothesis filed
  │
  ├─ HLO pre-filter → XLA already fuses → RETIRE (xla-already-fuses)
  │
  ├─ HLO pre-filter → XLA does NOT fuse
  │    │
  │    ├─ Existing Pallas kernel covers it → USE existing kernel
  │    │   (tokamax, tpu-inference, axlearn, etc.)
  │    │
  │    ├─ No existing kernel → DISPATCH to MaxKernel
  │    │    │
  │    │    ├─ MaxKernel succeeds → wire kernel → RUN-EXPERIMENT
  │    │    │
  │    │    └─ MaxKernel fails → log failure → retire or defer hypothesis
  │    │
  │    └─ Build cost > threshold, expected gain < 5% → DEFER
  │
  └─ HLO pre-filter → pending (no dump available) → CAPTURE dump first
```

---

## Integration workflow step by step

### Step 1: Build the kernel specification

When the autoresearch agent decides to invoke MaxKernel, it constructs a `kernel_spec.json` from information already available in the wiki:

| Spec field | Source in the wiki |
|---|---|
| `kernel_description` | Hypothesis page — Statement + Rationale |
| `shapes_and_dtypes` | Model page — architecture table + baseline command |
| `target_hardware` | Program page — Fixed bindings table |
| `sharding` | Model page — mesh shape + `PartitionSpec` |
| `correctness_reference` | Written by the autoresearch agent — a pure-JAX reference function that computes the unfused version |
| `hlo_context` | HLO dump directory from the pre-filter step |

The agent writes this spec to the experiment's workdir:

```
wiki/experiments/<program>/<stack>/kernels/<kernel-slug>/kernel_spec.json
```

### Step 2: Write the reference implementation

The autoresearch agent writes a small Python file (`reference_impl.py`) containing the unfused version of the computation. This is the numerical ground truth MaxKernel's tests will validate against.

```python
# reference_impl.py — generated by autoresearch agent
import jax
import jax.numpy as jnp

def rmsnorm_matmul_reference(x, weight, projection):
    """Unfused RMSNorm + matmul (the current XLA path)."""
    variance = jnp.mean(x ** 2, axis=-1, keepdims=True)
    normed = x * jax.lax.rsqrt(variance + 1e-6) * weight
    return normed @ projection
```

### Step 3: Invoke MaxKernel

The autoresearch agent invokes MaxKernel in headless mode:

```bash
cd wiki/experiments/<program>/<stack>/kernels/<kernel-slug>

adk run hitl_agent --headless \
  --workdir $(pwd) \
  --spec kernel_spec.json \
  --output kernel_result.json
```

The agent waits for completion (or timeout).

### Step 4: Process the result

On return, the autoresearch agent reads `kernel_result.json`:

**If `status: success`:**

1. The kernel file (e.g., `fused_rmsnorm_matmul.py`) is in the workdir.
2. Tests passed with numerical equivalence confirmed.
3. Profiling and autotuning data are available.
4. The agent wires the kernel into the training step:
   - Imports the kernel function
   - Replaces the unfused computation path with a call to the kernel
   - Sets the autotuned config as the default
5. Proceeds to RUN-EXPERIMENT as normal (20-step TPU run, profile, verdict).

**If `status: failed`:**

1. Read `error.stage` and `error.type` to understand why.
2. If `correctness_failure` → the kernel can't match the reference. Log this in the hypothesis page as evidence, consider whether the spec was wrong or the problem is genuinely hard.
3. If `compilation_failure` → may be a VMEM budget issue or shape problem. Log and potentially retry with adjusted constraints.
4. Update the hypothesis page with the failure details. Either retire or defer.

**If `status: partial`:**

1. The kernel and tests are good, but profiling/autotuning failed.
2. Proceed to RUN-EXPERIMENT anyway — the full-training profile will provide the data MaxKernel's profiler couldn't get.

### Step 5: Record provenance

The experiment page records that MaxKernel authored the kernel:

```markdown
## Kernel provenance

- **Author**: MaxKernel (headless mode)
- **Spec**: `kernels/fused-rmsnorm-matmul/kernel_spec.json`
- **Result**: `kernels/fused-rmsnorm-matmul/kernel_result.json`
- **Kernel file**: `kernels/fused-rmsnorm-matmul/fused_rmsnorm_matmul.py`
- **Autotune config**: `block_q=1024, block_kv=1024, num_stages=2`
- **Test status**: 5/5 passed, max rtol 3.2e-3
```

---

## SCHEMA.md changes needed

### FORMULATE-HYPOTHESIS

No additional changes beyond the existing step 2b (HLO pre-filter). The dispatch-to-MaxKernel decision happens *after* hypothesis ranking, not during formulation.

### RUN-EXPERIMENT

Add a new optional step between the current step 1b (AOT screening) and step 2 (prepare the run):

> **Step 1c. (Optional) Kernel authoring via MaxKernel.** If the hypothesis requires a new Pallas kernel and no existing kernel covers the need, invoke MaxKernel in headless mode with a `kernel_spec.json`. Wait for completion. On success, wire the kernel into the training step before proceeding to step 2. On failure, update the hypothesis and decide whether to retry, defer, or retire. See [MaxKernel integration design](wiki/analyses/2026-05-06-autoresearch-maxkernel-integration.md).

### LINT

Add check:
- Experiment pages that reference a MaxKernel-authored kernel must have a `## Kernel provenance` section with the spec, result, and test status.

### sample-program.md

Add MaxKernel to the "Reference material at a glance" section:
```
- Kernel authoring: [MaxKernel](https://github.com/AI-Hypercomputer/accelerator-agents/tree/main/MaxKernel) — ADK agent for Pallas kernel generation, testing, profiling, autotuning.
```

---

## Environment and deployment

### Same-host deployment (recommended)

Both systems run on the same TPU host:

```
TPU v6e-8 host
├── Autoresearch loop (Claude Code)
│   └── Invokes MaxKernel via shell
├── MaxKernel (ADK agent, headless)
│   └── Uses TPU for kernel testing/profiling/autotuning
└── xprof server (shared)
```

**Resource sharing**: MaxKernel's test/profile/autotune stages use the TPU, but these are short-lived (seconds to minutes per kernel). The autoresearch loop's full 20-step experiments also use the TPU. They don't run simultaneously — MaxKernel finishes before the experiment begins.

### Prerequisites on the host

| Dependency | Purpose | Already present? |
|---|---|---|
| Python 3.10+ | Both systems | Yes (conda env) |
| JAX + libtpu | Kernel execution | Yes |
| Google ADK (`adk` CLI) | MaxKernel runtime | **No — install needed** |
| Vertex AI access | MaxKernel's LLM backend | **No — credentials needed** |
| MaxKernel repo | The agent code | **No — clone needed** |

### Setup steps

```bash
# 1. Clone MaxKernel
git clone https://github.com/AI-Hypercomputer/accelerator-agents.git
cd accelerator-agents/MaxKernel

# 2. Install in the existing conda env
pip install -r hitl_agent/dependency/main_requirements.txt
pip install -r hitl_agent/dependency/agent_requirements.txt
pip install -e .

# 3. Configure credentials
export GOOGLE_CLOUD_PROJECT=<project-id>
export GOOGLE_GENAI_API_KEY=<key>

# 4. Verify headless mode works (once implemented)
echo '{"kernel_description": {"name": "test", "summary": "identity kernel"}, ...}' > /tmp/test_spec.json
adk run hitl_agent --headless --spec /tmp/test_spec.json --output /tmp/test_result.json
```

---

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| MaxKernel produces a kernel that passes its own tests but fails in the full training loop | Experiment yields `invalid` verdict; wasted TPU time | The autoresearch loop's existing loss-sanity check catches this. Tighten tolerance in `kernel_spec.json` if it recurs. |
| MaxKernel's LLM generates low-quality kernels that lose to XLA | Hypothesis refuted despite the fusion opportunity being real | Log the failure. The autoresearch agent can file a follow-up hypothesis with tighter constraints or a different fusion strategy. Manual kernel authoring remains an option. |
| Vertex AI quota / credential issues stall the loop | MaxKernel can't run; loop blocks | Set a timeout on MaxKernel invocation (e.g., 15 minutes). On timeout, skip the kernel and defer the hypothesis. |
| MaxKernel's RAG corpus lacks knowledge of ops specific to this model | Plans miss TPU-specific optimization opportunities | Supplement the spec's `hlo_context` field with relevant HLO snippets. Long-term: feed the autoresearch wiki's concept pages into MaxKernel's RAG corpus. |
| Two LLM-based agents in series (autoresearch → MaxKernel) compound error rates | Lower end-to-end success rate | The test stage in MaxKernel catches implementation errors. The autoresearch loop's profile stage catches performance regressions. Both systems have validation gates. |

---

## Success criteria

The integration is successful when:

1. The autoresearch loop can dispatch a kernel-replacement hypothesis to MaxKernel and receive a tested, profiled, autotuned kernel **without human intervention**.
2. The kernel is wired into a training experiment and produces a valid verdict (supported, refuted, or inconclusive — but not a crash due to integration issues).
3. The end-to-end latency from "hypothesis passes HLO pre-filter" to "experiment verdict" is under 30 minutes (MaxKernel kernel authoring + 20-step experiment).
4. Provenance is fully tracked: the experiment page links to the spec, result, kernel file, and test status.

---

## Timeline and dependencies

```
Phase 1: MaxKernel changes (prerequisites)
  ├── Headless mode (P0)
  ├── Structured input spec (P0)
  ├── Structured output result (P0)
  └── Workdir passthrough (P1)

Phase 2: Autoresearch integration (after Phase 1)
  ├── Spec builder in the autoresearch agent
  ├── MaxKernel invocation wrapper (shell-out or ADK Python API)
  ├── Result parser + kernel wiring logic
  ├── SCHEMA.md step 1c addition
  └── Provenance template in experiment pages

Phase 3: Validation
  ├── End-to-end test: file a kernel-replacement hypothesis,
  │   watch the loop dispatch to MaxKernel, wire the kernel,
  │   run the experiment, produce a verdict
  └── Retrospective: was the kernel quality sufficient?
      Did the integration save time vs manual authoring?
```

---

## See also

- [MaxKernel integration requirements](2026-05-06-maxkernel-integration-requirements.md) — changes needed on the MaxKernel side
- [AOT Compilation](../concepts/aot-compilation.md) — the HLO pre-filter that gates kernel dispatch
- [HLO Dumping and Diffing](../concepts/hlo-dumping-and-diffing.md) — how HLO context is captured for the spec
- [Pallas kernel directory](2026-04-23-pallas-kernel-directory.md) — existing kernel catalog (check before dispatching)
- [SCHEMA.md](../../SCHEMA.md) — autoresearch loop operations

## Sources

- [MaxKernel README](https://github.com/AI-Hypercomputer/accelerator-agents/tree/main/MaxKernel)
- [MaxKernel agent hierarchy](https://github.com/AI-Hypercomputer/accelerator-agents/tree/main/MaxKernel/hitl_agent) — subagent structure
