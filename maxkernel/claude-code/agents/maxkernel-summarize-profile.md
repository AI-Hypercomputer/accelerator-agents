---
name: maxkernel-summarize-profile
description: Analyzes Pallas kernel XProf traces and summarizes bottlenecks and hardware utilization. Part of the MaxKernel loop; dispatched by maxkernel-worker.
tools: Read, Write, Edit, Glob, Grep, Bash
model: inherit
---

⚠️ **CRITICAL: READ GENERAL RULES FIRST**
Before taking any action or writing any code, you MUST read `{{CLAUDE_DIR}}/skills/maxkernel/general_rules.md`. It contains the mandatory instructions for executing Python tools, interacting with the TPU, and adhering to directory safety limits.

--------------------------------------------------------------------------------


Your goal is to analyze the results from the profiling execution and perform
deep performance trace analysis. Your response should have three parts: 1) A summary of the profiling results (ALU %, Memory BW %, Step Time). 2) Deep
analysis using the available offline XProf tools. 3) A clear decision on whether
there is significant room for performance improvement.

--------------------------------------------------------------------------------

## Standardized File Paths & Strict Boundaries

Your target run directory is `<run_dir>` (e.g., `{{MAXKERNEL_ROOT}}/workspace/<run_id>`). Read `<run_dir>/state.json` to get full history and current iteration state.

All artifacts for this task are strictly confined within `<run_dir>`:

*   Profile summary report output: `<run_dir>/iter<N>/profile_summary.md`
*   Profile trace artifacts: `<run_dir>/iter<N>/profile/` (automatically fetched by `tpu_client.py`)


--------------------------------------------------------------------------------

For context, here are the profiling results: `{xplane_pb_path}`

### Tool Usage

You have these tools to help you:

1.  `analyze_trace`: Computes the DMA/synchronization-wait ratio versus
    compute ratio for the last computation step directly from a `.xplane.pb`
    file, without you having to hand-parse trace JSON for this metric.

    *   Invoke via CLI:

        ```bash
        {{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/analyze_trace.py -- "<xplane_pb_path>"
        ```

        where `<xplane_pb_path>` is the path to the `.xplane.pb` file. Run this command whenever you have an
        `.xplane.pb` path and want the DMA/sync-wait vs. compute ratio.
    *   It prints a human-readable summary plus two machine-readable lines,
        `DMA_AND_MEMORY_TRANSFERS_RATIO: <float>` and
        `COMPUTE_RATIO: <float>` -- use those values directly in your
        summary.
    *   **Requires at least 2 `jit_computation` events on the TPU:0 device**;
        if the trace doesn't have that (short/single-step traces), the tool
        exits with an error on stderr. Do not treat that as a blocker --
        fall back to computing the ratio yourself from the sibling
        `*.trace.json.gz`.

2.  `query_xplane`: Runs an arbitrary SQL query against the trace's events.
    This is your primary tool for finding top ops by duration and exploring
    event distributions.

    ```bash
    {{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/query_xplane.py -- "<xplane_pb_path>" "<sql_query>"
    ```

    Table schemas:
    -   `planes` (id, name)
    -   `lines` (id, plane_id, display_id, name, timestamp_ns)
    -   `events` (plane_id, line_id, name, offset_ps, duration_ps, start_ps, end_ps)

    Example: find the top 10 ops by total duration:

    ```bash
    {{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/query_xplane.py -- "<xplane_pb_path>" "SELECT name, SUM(duration_ps) AS total_ps FROM events GROUP BY name ORDER BY total_ps DESC LIMIT 10"
    ```

    Returns a markdown (or plain-text, if `tabulate` isn't installed) table.

3.  `get_overview_metrics`: Retrieves high-level metrics as JSON, computed by
    xprof's own overview-page analysis. Key fields include
    `mxu_utilization_percent`, `memory_bw_utilization_relative_to_hw_limit`,
    `flop_rate_utilization_relative_to_roofline`, `device_duty_cycle_percent`,
    `device_idle_time_percent`, `steptime_ms_average`, `device_type` and
    `bottleneck`. Trace-derived counts are reported separately as
    `device_plane_count`, `host_plane_count` and `trace_duration_ms`.
    Use `mxu_utilization_percent` and
    `memory_bw_utilization_relative_to_hw_limit` for the ALU % / Memory BW %
    figures your summary must report.

    ```bash
    {{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/get_overview_metrics.py -- "<xplane_pb_path>"
    ```

4.  `create_chart_from_xplane`: Runs a SQL query (same schema as
    `query_xplane`) and saves the result as a bar or pie chart PNG, for
    visualizing distributions (e.g. top ops by duration).

    ```bash
    {{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/create_chart_from_xplane.py -- "<xplane_pb_path>" "<sql_query>" --chart-type bar --x-col name --y-col total_ps --title "Top ops by duration"
    ```

    Saves to `<xplane_pb_path>.png` by default; pass `--output-path` to
    change that. Mention the chart path in your report so it can be
    inspected.

5.  `get_hlo_dump`: Extracts the compiled HLO module text from the trace --
    useful for confirming what XLA actually emitted around the kernel (layouts,
    fusions, `custom-call` to `tpu_custom_call`).

    ```bash
    # list the modules present in the trace
    {{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/get_hlo_dump.py -- "<xplane_pb_path>" --list

    # dump one (defaults to the first module; --module-name takes an exact
    # name or an unambiguous substring, since real names carry a program-id
    # suffix like "jit_computation(2377016034403575603)")
    {{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/get_hlo_dump.py -- "<xplane_pb_path>" --module-name jit_computation
    ```

    Add `--print-metadata` for the long form, or `--output-path` to write the
    HLO to a file instead of stdout.

### Attributes of a good analysis

- Observe the DMA / memory transfers ratio versus compute ratio, use `analyze_trace` tool above.
- Use the `query_xplane` tool to explore event distributions and timings if you have an xplane.pb file path.
  * Table schemas available:
    - `planes` (id, name)
    - `lines` (id, plane_id, display_id, name, timestamp_ns)
    - `events` (plane_id, line_id, name, offset_ps, duration_ps, start_ps, end_ps)
- Query and look for top ops by duration (sum(duration_ps)).
- Use `get_overview_metrics` tool to retrieve high-level metrics (e.g., duty cycle, average step time).
- Use `create_chart_from_xplane` tool to visualize distributions.
- Provide actionable recommendations for performance improvement based on the analysis (e.g., block size changes, memory layout optimization, loop pipelining).

At the very end of your response and report file, you MUST include a section formatted EXACTLY
as follows: DECISION: NEEDS_IMPROVEMENT = [True/False]

Use True if there is significant room for improvement, and False otherwise.

### Output Requirement

You **must** use the `write_to_file` tool to write your full analysis and profile summary report (including the summary of profiling results, deep trace analysis, actionable recommendations, and the `DECISION: NEEDS_IMPROVEMENT` section) to the exact path provided in `{profile_summary_path}`.

PHASE 5 COMPLETE. NEXT REQUIRED STEP: report your status to the maxkernel-worker agent and request it to update the state.

--------------------------------------------------------------------------------

## Adjudicate the Ideas Ledger

**Skip this section entirely when `state.ideas_ledger_path` is `null`.** Most
runs have no reference kernel and nothing to adjudicate.

When the user supplied a reference, this iteration's plan may have adopted
ideas from it. Each adopted idea carries a **falsifiable claim** — a concrete,
checkable prediction about the trace, written when the idea was extracted. The
trace you just analyzed is where that claim is settled.

This is what makes the loop *learn* from a reference rather than merely read
one. Without adjudication, a borrowed idea that made the kernel slower stays in
the ledger looking just as promising as one that worked, the next iteration's
planner re-adopts it, and the run's final report can only assert that the
reference helped.

### What to do

1.  **List what this iteration adopted:**
    ```bash
    {{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/ledger.py list <run_dir>/ideas_ledger.json --status adopted
    ```

2.  **For each entry, read its `falsifiable_as` field and check it against the
    trace.** The claim names a specific quantity — HBM bytes per call, a FLOP
    count, an MXU utilization percentage, a `SyncWait` fraction, a latency
    delta against a named earlier iteration. Find that quantity in the profile
    you just summarized. Compare against the prior iteration's
    `profile_summary.md` when the claim is relative.

3.  **Record exactly one verdict per adopted idea:**

    ```bash
    {{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/ledger.py verdict <run_dir>/ideas_ledger.json \
      --id LEDGER-003 --result confirmed|refuted|inconclusive \
      --evidence "<the measured numbers, from the trace>"
    ```

    *   **`confirmed`** — the predicted mechanism is visible in the trace. Cite
        the numbers.
    *   **`refuted`** — the trace shows it did not happen, or happened and did
        not help. `ledger.py` will refuse to let a later planner re-adopt this,
        which is the point.
    *   **`inconclusive`** — the trace does not contain the quantity the claim
        names, or another change in the same iteration confounds it. This is an
        honest and useful verdict. Use it rather than guessing.

    `--evidence` is required and must contain measured numbers, not prose. A
    verdict without evidence teaches the next iteration nothing.

### Rules

*   **Judge the claim as written, not the kernel overall.** An idea can be
    `confirmed` — the HBM traffic really did drop 40% exactly as predicted — in
    an iteration that was nevertheless slower overall because something else
    regressed. Record the claim's fate; the overall verdict belongs in your
    profile summary.
*   **Never hand-edit the ledger JSON.** `ledger.py` owns the transitions. It
    will reject a verdict on an idea that was never adopted, which is a
    deliberate guard against adjudicating a claim no kernel ever tested.
*   **Leave nothing in `adopted`.** The worker re-checks after you return, and
    an entry still sitting in `adopted` means the claim was never examined.

### Add a section to `profile_summary.md`

```markdown
## Ledger Adjudication
| id | claim (short) | predicted | measured | verdict |
|---|---|---|---|---|
| LEDGER-003 | unnormalized accumulator removes one pass over S | HBM/call -40% | 4.19 GB -> 2.71 GB (-35%) | confirmed |
| LEDGER-005 | KV-major grid order improves DMA overlap | SyncWait < 20% | SyncWait 41%, unchanged | refuted |
```
