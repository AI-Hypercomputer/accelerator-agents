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
deep performance trace analysis. Your response should have four parts:
1) A summary of the profiling results (ALU %, Memory BW %, Step Time).
2) A **per-XLA-op device-time breakdown**, per input case.
3) Deep analysis using the available offline XProf tools, including an HLO/LLO
   read of what the compiler actually emitted.
4) A clear decision on whether there is significant room for improvement.

--------------------------------------------------------------------------------

## Standardized File Paths & Strict Boundaries

Your target run directory is `<run_dir>` (e.g., `{{MAXKERNEL_ROOT}}/workspace/<run_id>`). Read `<run_dir>/state.json` to get full history and current iteration state.

Your dispatch prompt names a **target**, either `baseline` or `iter<N>` (a
prompt that says `iteration = <n>` means target `iter<n>`);
`<analysis_dir>` is `<run_dir>/baseline/` or `<run_dir>/iter<N>/` accordingly.
The kernel profiled was `<run_dir>/base.py` or
`<run_dir>/iter<N>/optimized.py` to match.

All artifacts for this task are strictly confined within `<run_dir>`:

*   Profile summary report output: `<analysis_dir>/profile_summary.md`
*   Machine-readable metrics output: `<analysis_dir>/profile_metrics.json`
*   HLO dump: `<analysis_dir>/hlo_dump.txt`
*   Profile trace artifacts: `<analysis_dir>/profile/` (automatically fetched by `tpu_client.py`)


--------------------------------------------------------------------------------

Find the trace yourself rather than waiting to be handed a path: `tpu_client.py`
fetches the artifacts into `<analysis_dir>/profile/`, and `<xplane_pb_path>`
below means the `.xplane.pb` under that directory. If there is more than one,
use the most recent. If there is none, stop and report that the profile job did
not produce a trace — do not analyze the previous iteration's.

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

--------------------------------------------------------------------------------

### The per-XLA-op breakdown (mandatory)

Before any prose, produce the op-level table. This is what tells the planner
whether the time is inside the Pallas kernel at all, or in unfused XLA ops
around it — the two lead to completely different plans, and the overview
metrics cannot distinguish them.

```bash
{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/query_xplane.py -- "<xplane_pb_path>" \
  "SELECT name, COUNT(*) AS n, SUM(duration_ps)/1e6 AS total_us
   FROM events GROUP BY name ORDER BY total_us DESC LIMIT 25"
```

Then split it per case. The profile script wraps each input case in a
`case_<idx>` `TraceAnnotation`, so the annotation's `start_ps`/`end_ps` bracket
that case's events:

```bash
{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/query_xplane.py -- "<xplane_pb_path>" \
  "SELECT name, start_ps, end_ps FROM events WHERE name LIKE 'case_%' ORDER BY start_ps"
```

Report, per case: op name, invocation count, total device µs, and share of that
case's device time. **Do not collapse to the geomean and stop.** Floors and
bottlenecks differ by regime; the small cases quietly dropping out of the
analysis is the single most common way a run stops early against a suite that
still had headroom.

Also name explicitly:

*   which row is the Pallas `custom-call` (`tpu_custom_call`), and what fraction
    of device time it holds. If it is well under 100%, the remainder is XLA
    around the kernel and belongs in the plan;
*   any op you did not expect to be there at all — a transpose, a copy, a
    convert. Those are usually layout fixups the compiler inserted, and they are
    cheap to remove once named.

### The HLO / LLO read (mandatory)

Dump the HLO to `<analysis_dir>/hlo_dump.txt` with `get_hlo_dump.py
--output-path`, then read it and report:

*   **what fused and what did not** around the `custom-call`. An unfused
    elementwise chain either side of the kernel is an HBM round-trip the kernel
    was supposed to avoid;
*   **the layouts** on the custom-call's operands and result. A layout that
    disagrees with the producer's forces a physical relayout;
*   whether XLA assigned the custom call a **zero cost** — it does by default
    unless `pl.CostEstimate` is supplied, which mis-schedules everything async
    around it. See `{{MAXKERNEL_ROOT}}/reference/pallas_xla_interaction.md`.

If the trace carries LLO artifacts (`*-packed-bundles-post-ra.txt` under
`<analysis_dir>/profile/`), also report register-pressure alarms, `vst`/`vld`
pairs occurring *during* compute blocks (register spills, as opposed to the
prefetch/store loops where they are expected), and `vxpose` counts (Mosaic
performing physical relayouts in VMEM). If those artifacts are absent, say so
rather than inferring their contents.

--------------------------------------------------------------------------------

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
- Route each recommendation by what binds, using the routing table in `{{MAXKERNEL_ROOT}}/reference/kernel_design_method.md` §12 (relayouts → layout or a unit trade; saturated VPU on the critical path → unit trading; DMA wait → buffering depth and async writes; spills → tile size; a ratio to the floor that *widens* as work is removed → fixed cost binds, not the modelled resource).
- Check the counters you quote against "Instruments that lie" in `{{MAXKERNEL_ROOT}}/reference/tpu_hardware_facts.md` before putting any of them in a denominator.
- When a recommendation hinges on a fixed cost or a bandwidth you have not measured on this TPU, measure it instead of assuming it: `{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/run_probe.py --pick dispatch_floor` (per-call and per-step cost), `--pick pallas_call_intercept_chained_clock`, or `--pick hbm_bandwidth_arms`, with `--run_dir <run_dir>`. Never pass `--allow-destructive`.

### A magnitude is not a hypothesis

Your report feeds `maxkernel-plan-kernel`, and the rule that agent works under is
in `{{MAXKERNEL_ROOT}}/skill/guides/general_profile_reasoning.md` §1:

> **The hypothesis lives in the discrepancy, not in the magnitude.**

*"This op costs 175 µs"* only suggests "do less of it", which is the wrong fix
when the volume is already minimal. So for the top one or two rows in your
table, do not stop at the magnitude — either state what the cost *ought* to be
(from `<analysis_dir>/roofline.json` or `<run_dir>/baseline/microbenchmarks.json`
if either exists, which is free) or **name the shape that should be priced in
isolation** so a microbenchmark can supply the "ought" later. A recommendation
without an "ought" behind it is a guess dressed as a finding.

Equally, if a metric shows the profiled thing is *not* where the time is, say
that plainly. The absence of a discrepancy is the more valuable outcome: it
kills work rather than creating it.

At the very end of your response and report file, you MUST include a section formatted EXACTLY
as follows: DECISION: NEEDS_IMPROVEMENT = [True/False]

Use True if there is significant room for improvement, and False otherwise.

### Output Requirement

1.  You **must** use the `write_to_file` tool to write your full analysis to
    `<analysis_dir>/profile_summary.md` — the summary of profiling results, the
    per-case per-op breakdown, the HLO/LLO read, deep trace analysis, actionable
    recommendations, the `## Ledger Adjudication` section when the section
    below applies, and the `DECISION: NEEDS_IMPROVEMENT` section.

2.  You **must** also write `<analysis_dir>/profile_metrics.json`, so the worker
    and the planner can branch on numbers instead of parsing prose:

    ```json
    {
      "target": "baseline",
      "kernel_version": "<absolute path to the kernel profiled>",
      "mxu_utilization_percent": 23.4,
      "memory_bw_utilization_percent": 61.0,
      "dma_and_memory_transfers_ratio": 0.68,
      "compute_ratio": 0.32,
      "steptime_ms_average": 2.231,
      "bottleneck": "memory",
      "pallas_custom_call_share_of_device_time": 0.82,
      "per_case": [
        {"case": 0, "device_us": 471.0,
         "top_ops": [{"name": "custom-call.1", "count": 3, "total_us": 386.0, "share": 0.82}]}
      ],
      "unexpected_ops": [],
      "register_spills_observed": null,
      "vxpose_count": null,
      "needs_improvement": true
    }
    ```

    Anything you could not obtain is `null` — never a guess. `null` on
    `register_spills_observed` means "no LLO artifacts in this trace", which is
    a different and honest claim from `false`.

Then — after the ledger adjudication below, if it applies — report a 2–3
sentence summary back to the maxkernel-worker agent, which uses this profile in
its Phase 5 keep/drop decision. Do not dispatch other agents and do not try to
advance the loop yourself.

--------------------------------------------------------------------------------

## Adjudicate the Ideas Ledger

**Skip this section entirely when `state.ideas_ledger_path` is `null`, or when
your target is `baseline`** (no plan has adopted anything yet). Most runs have
no reference kernel and nothing to adjudicate.

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
