---
title: "MaxKernel Integration — Requirements for MaxKernel Changes"
type: analysis
tags: [integration, maxkernel, pallas, kernel-authoring]
created: 2026-05-06
updated: 2026-05-06
---

Requirements document for changes needed to [MaxKernel](https://github.com/AI-Hypercomputer/accelerator-agents/tree/main/MaxKernel) to enable programmatic invocation by the autoresearch optimization loop.

**Companion document**: [Autoresearch–MaxKernel integration design](2026-05-06-autoresearch-maxkernel-integration.md).

**Reference branch**: [`max-perf-max-kernel-integration`](https://github.com/AI-Hypercomputer/accelerator-agents/tree/max-perf-max-kernel-integration/MaxKernel) — contains an `auto_agent/` directory with autonomous pipeline capability already implemented. This document is updated to reflect that branch's current state.

---

## Background

MaxKernel is an ADK-based (Google Agent Development Kit) multi-agent system for generating, optimizing, testing, and profiling TPU Pallas kernels. It ships with two agent variants:

1. **`hitl_agent/`** (main branch) — Human-in-the-loop: conversational, every stage requires human approval.
2. **`auto_agent/`** (`max-perf-max-kernel-integration` branch) — Autonomous pipeline: `AutonomousPipelineAgent` chains Plan → Implement → Validate → Test Gen → Test Run → Profile **without human approval gates**, with iterative self-improvement (up to `max_iterations=5`).

The autoresearch optimization loop ([SCHEMA.md](../../SCHEMA.md)) identifies kernel-replacement opportunities via HLO pre-filtering, but currently has no kernel-authoring backend. MaxKernel fills this gap. The `auto_agent/` already removes the human-gating problem — the remaining gaps are the I/O contract (structured spec in, structured result out) and autotuning.

---

## What's already done on `max-perf-max-kernel-integration`

The `auto_agent/` implementation covers the majority of what was originally scoped as Requirement 1 (headless mode). Specific capabilities already present:

| Capability | Implementation | Location |
|---|---|---|
| **Autonomous pipeline orchestration** | `AutonomousPipelineAgent` chains 6 subagents sequentially without approval gates | `auto_agent/subagents/pipeline_agent.py` |
| **Auto-retry on compilation failure** | `KernelCompilationValidationLoop` retries up to 6 times with `fix_kernel_compilation_agent`; after 2+ failures invokes `add_debug_statements_agent` | `auto_agent/subagents/kernel_writing/agent.py` |
| **Test validation loop** | 4-stage validation (syntax → import → structure → mock execution) with 6-retry loop before real TPU test execution | `auto_agent/subagents/testing/agent.py` |
| **Auto-test execution** | `TestRunner` dispatches pytest to eval server via HTTP POST; pass/fail by exit code | `auto_agent/subagents/testing/agent.py` |
| **Iterative improvement** | If profiling says `needs_improvement=True`, pipeline restarts from planning (up to `max_iterations=5`) | `auto_agent/subagents/pipeline_agent.py` |
| **Structured profiling decision** | `SummarizeProfileAgent` regex-parses output and injects `needs_improvement` boolean into session state | `auto_agent/subagents/profiling/agent.py` |
| **Workdir via env var** | `WORKDIR` read from env, `add_workdir_callback` creates session directories | `auto_agent/config.py`, `auto_agent/callbacks.py` |
| **Launch scripts** | `run_auto_agent.sh` / `prepare_auto_agent.sh` — parallel to HITL scripts | `MaxKernel/` root |
| **Internal state tracking** | Session state dict tracks `optimized_kernel_path`, `kernel_code`, `compilation_results`, `compilation_history`, latency, profiling summaries | `auto_agent/callbacks.py` |
| **Best-result selection** | Pipeline returns lowest-latency valid solution across iterations; LLM fallback when latency unavailable | `auto_agent/subagents/pipeline_agent.py` |

**Model**: `gemini-3.1-pro-preview` with `temperature=0.1`, `top_p=0.9`, `top_k=5`. Planner uses `thinking_level="high"`.

---

## Remaining requirements

### ~~Requirement 1: Headless execution mode~~ — DONE

> **Status: Implemented** on `max-perf-max-kernel-integration` as `auto_agent/`.

The `AutonomousPipelineAgent` already runs the full Plan → Implement → Validate → Test → Profile cycle without human input, with auto-retry and iterative improvement. Launched via `adk run auto_agent`.

**No further changes needed** for this requirement. The original acceptance criteria are met:
- HITL behavior preserved in `hitl_agent/` (separate agent, separate launch script).
- Autonomous mode completes without user input.
- Same subagent quality (planning, implementation, RAG) — shared `knowledge_base/` and similar prompt structure.

**One minor gap**: the pipeline currently lacks early-exit-on-success. If the first iteration produces a correct, well-performing kernel, it still runs remaining iterations. This is a nice-to-have optimization, not a blocker — the best-result selection handles it.

---

### Requirement 2: Structured input specification — OPEN

#### Problem

The `auto_agent` still takes free-form conversational input via `adk run auto_agent` (user types a kernel request in the chat). The autoresearch agent needs to pass machine-readable specifications with precise constraints — target ops, shapes, dtypes, shardings, HLO context — without a conversational prompt.

`custom_types.py` defines only `CustomLlmAgent` (a retry-aware LLM wrapper). No kernel spec type exists. Session state is a loosely-typed `dict` populated by callbacks.

#### Requirement

Add a `kernel_spec.json` input path that seeds the autonomous pipeline's session state before the first subagent runs.

#### Proposed approach

Add a thin adapter layer (not a new subagent) that:

1. Accepts a `--spec /path/to/kernel_spec.json` CLI flag (or env var `KERNEL_SPEC_PATH`).
2. Validates the spec against a JSON schema on startup.
3. Seeds `session.state` with the spec fields before `PlanKernelAgent` runs — converting the JSON into the same state keys the pipeline already uses.
4. Formats a synthetic "user message" from the spec for the planning prompt (so the existing planning prompt works unchanged).

This avoids modifying any subagent logic — the adapter just translates structured input into the conversational + state format the pipeline already expects.

#### Proposed schema: `kernel_spec.json`

```json
{
  "version": "1.0",
  "request_id": "llama3-rmsnorm-matmul-fusion-2026-05-06",

  "kernel_description": {
    "name": "fused_rmsnorm_matmul",
    "summary": "Fuse RMSNorm normalization with the subsequent linear projection (matmul) to eliminate the HBM round-trip for the normalized activations.",
    "target_ops": ["rsqrt", "multiply", "convolution"],
    "fusion_pattern": "norm(x) → matmul(norm(x), W) in a single Pallas kernel"
  },

  "shapes_and_dtypes": {
    "inputs": [
      {"name": "x", "shape": [4, 8192, 4096], "dtype": "bfloat16"},
      {"name": "weight", "shape": [4096], "dtype": "bfloat16"},
      {"name": "projection", "shape": [4096, 4096], "dtype": "bfloat16"}
    ],
    "outputs": [
      {"name": "result", "shape": [4, 8192, 4096], "dtype": "bfloat16"}
    ]
  },

  "target_hardware": {
    "tpu_generation": "v6e",
    "num_chips": 8,
    "vmem_mib": 96,
    "hbm_gib": 32
  },

  "sharding": {
    "mesh_shape": {"fsdp": 8},
    "input_shardings": {
      "x": ["fsdp", null, null],
      "weight": [null],
      "projection": ["fsdp", null]
    }
  },

  "correctness_reference": {
    "type": "python_function",
    "module": "reference_impl",
    "function": "rmsnorm_matmul_reference",
    "file": "reference_impl.py",
    "tolerance": {"rtol": 1e-2, "atol": 1e-2}
  },

  "hlo_context": {
    "baseline_hlo_dump_dir": "/tmp/hlo_dump",
    "module_id": "module_0262",
    "target_fusion": "fused_computation.47",
    "notes": "XLA currently does NOT fuse these ops (verified via HLO pre-filter). The RMSNorm output lands in HBM before the matmul reads it."
  },

  "constraints": {
    "numerical_equivalence": true,
    "max_autotune_configs": 200,
    "autotune_timeout_seconds": 600
  }
}
```

#### Required fields

| Field | Purpose |
|---|---|
| `kernel_description` | What to build — name, summary, target ops |
| `shapes_and_dtypes` | Concrete tensor shapes and dtypes for all inputs/outputs |
| `target_hardware` | TPU generation, chip count, VMEM/HBM budgets |
| `correctness_reference` | Reference implementation for numerical equivalence testing |

#### Optional fields

| Field | Purpose |
|---|---|
| `sharding` | Mesh shape and per-tensor shardings (if the kernel must be shard-aware) |
| `hlo_context` | Baseline HLO dump for MaxKernel to inspect (what XLA currently does) |
| `constraints` | Autotune budget, tolerance overrides, timeout |

#### Acceptance criteria

- `adk run auto_agent --spec kernel_spec.json` starts the pipeline with the spec as context, no conversational prompt needed.
- All spec fields are accessible in `session.state` by downstream subagents.
- If required fields are missing, the pipeline fails with a structured validation error before any subagent runs.
- Existing conversational mode (no `--spec` flag) still works unchanged.

---

### Requirement 3: Structured output result — OPEN

#### Problem

The `auto_agent` pipeline tracks results internally in `session.state` (kernel code, compilation results, test results, latency, profiling summary, `needs_improvement` boolean). But it does not write a consolidated result file. The calling agent would need to parse the ADK session JSON to extract results — fragile and undocumented.

#### Requirement

At pipeline completion (success or failure), write a `kernel_result.json` to the workdir summarizing what happened.

#### Proposed approach

Add a `finalize_result` callback (or a final `ResultWriterAgent`) that runs after the pipeline completes. It reads the relevant keys from `session.state` and writes `kernel_result.json`.

This is a ~50-line addition — no subagent logic changes. The data is already in session state; it just needs to be serialized to a file.

#### Proposed schema: `kernel_result.json`

```json
{
  "version": "1.0",
  "request_id": "llama3-rmsnorm-matmul-fusion-2026-05-06",
  "status": "success",

  "artifacts": {
    "kernel_file": "fused_rmsnorm_matmul.py",
    "test_file": "test_fused_rmsnorm_matmul.py",
    "plan_file": "plan_fused_rmsnorm_matmul.md",
    "profile_file": "profile_fused_rmsnorm_matmul.json"
  },

  "test_results": {
    "compilation": "pass",
    "correctness": "pass",
    "num_tests_run": 5,
    "num_tests_passed": 5,
    "max_rtol_observed": 3.2e-3,
    "max_atol_observed": 1.1e-3
  },

  "profiling": {
    "kernel_time_us": 142.3,
    "memory_bytes": 67108864,
    "flops": 2.74e12,
    "arithmetic_intensity": 40.8,
    "bottleneck": "compute",
    "mxu_utilization_percent": 72.1,
    "needs_improvement": false
  },

  "autotuning": null,

  "iterations_run": 2,
  "best_iteration": 1,

  "error": null
}
```

On failure:

```json
{
  "version": "1.0",
  "request_id": "llama3-rmsnorm-matmul-fusion-2026-05-06",
  "status": "failed",
  "artifacts": {},
  "test_results": null,
  "profiling": null,
  "autotuning": null,
  "iterations_run": 3,
  "best_iteration": null,
  "error": {
    "stage": "testing",
    "type": "correctness_failure",
    "message": "Max absolute error 0.15 exceeds tolerance 0.01",
    "retries_attempted": 6,
    "traceback": "..."
  }
}
```

#### Acceptance criteria

- Every pipeline run (success or failure) produces a `kernel_result.json` in the workdir.
- The `status` field is always present and is one of: `success`, `failed`, `partial`.
- All file paths in `artifacts` are relative to the workdir and point to files that exist.
- `iterations_run` and `best_iteration` reflect the iterative improvement loop state.

---

### Requirement 4: Autotuning subagent — OPEN (gap)

#### Problem

The `hitl_agent/` has a full `AutotuneAgent` (with `AutotunePlannerAgent`, `AutotuneRunner`, `AutotuneSummaryAgent`). The `auto_agent/` pipeline **does not include autotuning** — it has 4 subagents (planning, kernel writing, testing, profiling) but no autotuning stage.

This means the autonomous pipeline produces kernels with default block sizes / tiling parameters. For many kernels, autotuning yields 10-30% improvement (the Llama 3 splash attention autotune in this wiki found +30-32% kernel time improvement via a 171-config sweep).

#### Requirement

Port the `AutotuneAgent` from `hitl_agent/` into the `auto_agent/` pipeline, or make it available as an optional final stage.

#### Proposed approach

- Add `AutotuneAgent` (or a simplified version) as stage 7 in `AutonomousPipelineAgent`, after profiling.
- The autotune stage runs only if profiling succeeds and `needs_improvement` is true (or unconditionally if `constraints.max_autotune_configs > 0` in the spec).
- Autotune results (best config, speedup) are written to session state and included in `kernel_result.json`.

#### Acceptance criteria

- The autonomous pipeline can optionally run an autotune sweep after profiling.
- Autotuned kernel config is persisted in the output artifacts.
- Autotuning can be skipped via the spec (`max_autotune_configs: 0`) for quick iterations.

---

### Requirement 5: Workdir per-invocation passthrough — PARTIALLY DONE

#### Current state

`WORKDIR` is read from the environment in `config.py` (defaults to the `auto_agent/` directory). `add_workdir_callback` creates a session subdirectory. This works for single-invocation use.

#### Remaining gap

For integration with the autoresearch loop, `WORKDIR` needs to be settable per-invocation without modifying `.env`. The simplest approach: the calling agent sets the `WORKDIR` env var before invoking `adk run`. This already works with the current implementation — no code change needed, just document it.

```bash
WORKDIR=/path/to/experiment/kernels adk run auto_agent --spec kernel_spec.json
```

#### Status: **No code change needed.** Document the env-var approach.

---

## Summary: what's done vs what's needed

| Requirement | Status | Effort estimate |
|---|---|---|
| R1: Headless / autonomous execution | **DONE** (`auto_agent/` on branch) | — |
| R2: Structured input spec | **OPEN** | Small — adapter layer + JSON schema validation; no subagent changes |
| R3: Structured output result | **OPEN** | Small — finalize callback reading existing session state |
| R4: Autotuning in autonomous pipeline | **OPEN** | Medium — port from `hitl_agent/` subagents |
| R5: Workdir passthrough | **DONE** (env var already works) | — |

The total remaining work is significantly less than originally estimated. R2 and R3 are thin I/O adapters (~100-150 lines total). R4 is the only piece with real logic, and it's a port of existing code from `hitl_agent/`.

---

## Out of scope (no changes needed)

The following MaxKernel components should remain unchanged:

| Component | Rationale |
|---|---|
| `PlanKernelAgent` prompt/logic | Core planning quality is the value prop |
| `ImplementKernelAgent` code generation | Same |
| `KernelCompilationValidationLoop` | Already has 6-retry + debug-statement injection |
| Test generation and execution | Already autonomous with HTTP eval dispatch |
| Profiling pipeline + `needs_improvement` decision | Already produces structured boolean output |
| `AutonomousPipelineAgent` orchestration | Already iterates up to `max_iterations` |
| RAG over Pallas docs | Adequate for kernel authoring |
| GPU-to-JAX conversion | Orthogonal |
| `ExplanationAgent` | Not invoked in autonomous mode |
| Web UI mode | Not used by programmatic integration |
| HITL agent (`hitl_agent/`) | Unchanged; parallel implementation preserved |

---

## See also

- [Autoresearch–MaxKernel integration design](2026-05-06-autoresearch-maxkernel-integration.md)
- [AOT Compilation](../concepts/aot-compilation.md) — the HLO pre-filter that gates kernel-writing requests
- [SCHEMA.md](../../SCHEMA.md) — FORMULATE-HYPOTHESIS step 2b (HLO pre-filter)
- [Pallas kernel directory](2026-04-23-pallas-kernel-directory.md) — existing kernel catalog

## Sources

- [MaxKernel README (main)](https://github.com/AI-Hypercomputer/accelerator-agents/tree/main/MaxKernel)
- [MaxKernel `auto_agent/` (integration branch)](https://github.com/AI-Hypercomputer/accelerator-agents/tree/max-perf-max-kernel-integration/MaxKernel/auto_agent) — autonomous pipeline implementation
- [`pipeline_agent.py`](https://github.com/AI-Hypercomputer/accelerator-agents/blob/max-perf-max-kernel-integration/MaxKernel/auto_agent/subagents/pipeline_agent.py) — `AutonomousPipelineAgent` orchestration
- [`run_auto_agent.sh`](https://github.com/AI-Hypercomputer/accelerator-agents/blob/max-perf-max-kernel-integration/MaxKernel/run_auto_agent.sh) — autonomous launch script
