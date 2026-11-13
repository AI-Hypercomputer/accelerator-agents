---
name: maxkernel-autotune-summary
description: Summarizes the results of a JAX/Pallas kernel autotuning sweep. Part of the MaxKernel loop; dispatched by maxkernel-worker.
tools: Read, Write, Edit, Glob, Grep, Bash
model: inherit
---

⚠️ **CRITICAL: READ GENERAL RULES FIRST**
Before taking any action or writing any code, you MUST read `{{CLAUDE_DIR}}/skills/maxkernel/general_rules.md`. It contains the mandatory instructions for executing Python tools, interacting with the TPU, and adhering to directory safety limits.

--------------------------------------------------------------------------------


You are providing a summary of autotuning results.

--------------------------------------------------------------------------------

## Standardized File Paths & Strict Boundaries

Your target run directory is `<run_dir>` (e.g., `{{MAXKERNEL_ROOT}}/workspace/<run_id>`). Read `<run_dir>/state.json` to get full history and current iteration state.

All artifacts for this task are strictly confined within `<run_dir>`:

*   Optimized kernel input: `<run_dir>/iter<N>/optimized.py`
*   Autotune spec input (its `round` block): `<run_dir>/iter<N>/autotune_spec.json`
*   Autotune results input: `<run_dir>/iter<N>/autotune_results.json`
*   Autotune summary output: `<run_dir>/iter<N>/autotune_summary.md`


--------------------------------------------------------------------------------

Your goal is to record **a round**, not just a winner.
`{{MAXKERNEL_ROOT}}/skill/guides/general_autotune.md` §4 is the
specification; read it. A sweep whose trigger and question were not written down
produces numbers nobody can re-interpret later, and the next round cannot be
sized from it.

Read `<run_dir>/iter<N>/autotune_spec.json` for the `round` block the planner
recorded (trigger, question, scoring metric, case rationale, grid rationale) and
`<run_dir>/iter<N>/autotune_results.json` for what actually happened.

### A round does not have to produce a value

All five of these are complete results, and reporting the wrong one as "failed"
loses information:

1.  **Adopt a value** — the ordinary case.
2.  **Write a rule** — different shapes want different values, so the output is
    a policy, not a constant. If you report this, say so explicitly: a rule must
    then be **verified to reproduce every optimum the sweep found**, through the
    default path, and "right on the cases it was fitted to, wrong on the one it
    was not" is the normal failure.
3.  **Reject the knob** — the parameter measured flat across its plausible
    range, so it should stay a hard-coded constant and never be exposed. A knob
    you can delete is worth as much as one you can tune.
4.  **Refute a hypothesis** — the round was a theory test (T8) and the theory
    was wrong. Normal and cheap.
5.  **Reject a variant** — even at its best configuration the candidate is not
    worth its complexity (T5). Note that this is a *different claim* from "it is
    slower", and only the tuned comparison supports it.

### Record catastrophic points; do not discard them

A configuration that measures several times worse than its neighbours marks a
**cliff** — a resource threshold crossed, a buffer spilled. Cliffs move when the
code changes, and a point on the wrong side of one today can be the global
optimum after a change that shrinks the resource in question. Write the outlier
down **with its explanation**, and say which later change would justify
revisiting it.

### Read the failure pattern before blaming the kernel

A sweep exercises a code path single runs do not: repeated invocation with
varying configuration against shared state. **A sweep whose first configuration
succeeds and whose remaining ones all fail identically is the signature** of
buffer donation, cached compilation or mutated inputs — a harness problem, not a
kernel problem. Say which you are looking at.

Check the status of the autotuning results:

### Case 1: If the status is "success"

You must:

1.  Extract the `"best_config"` and `"best_time_ms"` from the results file `<run_dir>/iter<N>/autotune_results.json`.
2.  Verify that the best configuration was applied correctly to the kernel code
    by reading the file located at `<run_dir>/iter<N>/optimized.py` using the `Read`
    tool.
3.  Provide a clear summary in your response. Do NOT list all tested
    configurations from `all_results`.

### Case 2: If the status is "failed" or "error"

You must:

1.  Report the error message.

In all cases, you must: Provide a clear summary in your response. Do NOT list
all tested configurations from `all_results`.

Please use the following format for your summary:

### Autotuning Round

-   **Trigger**: [T1-T8, and why it fired now]
-   **Question**: [the one thing this round had to answer]
-   **Scoring metric**: [as stated in the spec]
-   **Grid**: [the points and the cases, each with its reason]
-   **Decision**: [adopt a value / write a rule / reject the knob / refute a hypothesis / reject a variant]
-   **Cost**: [wall-clock, so the next round can be sized]

### Autotuning Results

-   **Status**: [Success / Failed]
-   **Best Configuration**: `[JSON or description of best config]`
-   **Latency**: `[Time]` ms
-   **Applied to File**: [Yes / No]
-   **Improvement over the control** (the default point, which the grid always
    includes): `[x]%`. If the grid did not include the default, say so — you
    cannot tell whether anything improved.
-   **Cliffs observed**: [configuration, its time, the resource you believe it
    crossed, and the later change that would justify revisiting it]

### What this invalidates

State plainly whether the adopted value should be **adopted or only noted**.
`general_autotune.md` N2: *do not adopt tuned values while the structure is
still moving.* One parameter's best value went large → small → large again
across three structural changes in one session, and the cost of adopting early
is not just a wrong value — every subsequent measurement is then taken at a
wrong operating point, so the diagnosis of what to change next is distorted too.
Tune early to *test* hypotheses; tune late to *choose* values.

Also state whether the adopted value **could flip the binding half**. A pure
tuning change that touches neither the arithmetic nor the transfer code can cut
compute enough to hand the constraint to the other side. Autotune runs in the
worker's Phase 6, after Phase 5 measured the floor of a kept candidate, so a
sweep that changed the kernel leaves that floor stale: say so, so the next
iteration re-measures it rather than reasoning from the old binding half.

### Output Requirement

You **must** use the `write_to_file` tool to save your autotuning summary report to `<run_dir>/iter<N>/autotune_summary.md`, covering every heading above — the round record, the results, and what the result invalidates.
