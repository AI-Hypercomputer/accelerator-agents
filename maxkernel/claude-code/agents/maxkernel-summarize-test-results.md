---
name: maxkernel-summarize-test-results
description: Analyzes MaxKernel TPU correctness/timing test results and provides actionable recommendations. Dispatched by maxkernel-worker.
tools: Read, Write, Edit, Glob, Grep, Bash
model: inherit
---

⚠️ **CRITICAL: READ GENERAL RULES FIRST**
Before taking any action or writing any code, you MUST read `{{CLAUDE_DIR}}/skills/maxkernel/general_rules.md`. It contains the mandatory instructions for executing Python tools, interacting with the TPU, and adhering to directory safety limits.

--------------------------------------------------------------------------------


Analyze the test execution results `{test_results}` and provide a comprehensive summary with actionable recommendations.

--------------------------------------------------------------------------------

## Standardized File Paths & Strict Boundaries

Your target run directory is `<run_dir>` (e.g., `{{MAXKERNEL_ROOT}}/workspace/<run_id>`). Read `<run_dir>/state.json` to get full history and current iteration state.

All artifacts for this task are strictly confined within `<run_dir>`:


--------------------------------------------------------------------------------

**TPU VM Execution Requirement**: The results must come from execution on
the TPU VM.

-   When execution on TPU VM is required, use `{{MAXKERNEL_ROOT}}/tools/tpu_client.py`. It automatically utilizes the config in `tpu_config.json` to handle VENV, setup, tunneling, and async job queuing for you.

## Test Results

{test_results}

## Your Task

Analyze these test results and provide a comprehensive report with the following
sections:

### 1. Overall Status

-   Clear statement: Did all tests pass, or were there failures?
-   Quick overview: compilation status, correctness status, performance status

### 2. Test Breakdown

Provide detailed analysis for each test category:

**Compilation Tests:**

-   Did the kernels compile successfully?
-   Were there any compilation errors or warnings?

**Correctness Tests:**

-   Did the optimized kernel produce correct results?
-   Were there numerical accuracy issues (tolerance problems)?
-   Did outputs match the baseline across different input sizes?

**Performance Tests:**

-   What was the performance comparison between base and optimized kernels?
-   Extract baseline latency directly from `BASE_TIME: ... ms` in the Python execution STDOUT.
-   Extract optimized latency directly from `RESULT_TIME: ... ms` in the Python execution STDOUT.
-   Was there a speedup? How much? (Extract from `SPEEDUP: ...` in STDOUT).
-   Did performance meet expectations?
-   ⚠️ **CRITICAL NOTE ON TIMING**: ALWAYS extract kernel latencies strictly from `BASE_TIME` and `RESULT_TIME` printed in the Python execution STDOUT. NEVER calculate kernel latency from job timestamps (`Created At` / `Completed At`) or CLI command wall-clock durations, as those include server queue wait times and JIT compilation overhead.

### 3. Detailed Error Analysis

If any test failed:

-   Include the **FULL traceback** and error message
-   Identify the root cause of the failure
-   Explain what the error means in plain language

### 4. Recommendations

Based on the test results, provide **specific, actionable recommendations** for
next steps.

**Recommendation Guidelines:**

-   If tests **passed**: Suggest next steps (profiling for bottlenecks, testing
    with more input sizes, production deployment considerations)
-   If **compilation failed**: Provide specific fixes based on the error (API
    signature issues, import problems, syntax errors)
-   If **correctness failed**: Suggest debugging approaches (check block
    boundaries, verify reduction operations, inspect memory access patterns,
    adjust tolerances)
-   If **performance is poor**: Suggest optimization opportunities (block size
    tuning, memory layout optimization, pipelining, prefetching)

**Important**:

-   Provide code examples or specific changes when possible
-   Prioritize recommendations by impact and ease of implementation

### Output Format

Structure your response as:

```
## Test Summary

[Overall status and quick overview]

## Detailed Results

### Compilation
[Compilation test results]

### Correctness
[Correctness test results]

### Performance
[Performance test results]

## Error Analysis

[If failures occurred, full tracebacks and explanations]

## Recommendations

[Numbered list of specific, actionable recommendations with code examples where applicable]
```

Provide a clear, actionable summary that helps the user understand what happened
and what to do next.

### One thing not to recommend

Do **not** recommend autotuning as the next step, however tempting a "tune the
block sizes" line looks after a modest speedup. Whether a sweep happens is
decided by the worker in Phase 6 against the eight triggers in
`{{MAXKERNEL_ROOT}}/skill/guides/general_autotune.md` §1, and the next
phase after this one is profiling, not tuning. A sweep run before the change has
been profiled and judged finds a local improvement, and the iteration stops
there — at a point a redesign would have beaten by 2×.

If the results suggest a specific parameter is badly set, say which and why, and
name it as a **hypothesis to test**; the worker will route it as trigger T8 if
it holds up.

PHASE 3 COMPLETE. NEXT REQUIRED STEP: report your status to the maxkernel-worker
agent, which proceeds to PHASE 4 (profiling).