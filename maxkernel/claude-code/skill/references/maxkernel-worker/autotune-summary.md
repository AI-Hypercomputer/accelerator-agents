# Autotune summary

`maxkernel-worker` reads this in Phase 4 step 4, after
`apply_best_config.py` has reduced the sweep and applied the winner. You write
the summary yourself, in this context, from the files the sweep left behind.

--------------------------------------------------------------------------------

## Inputs and outputs

`<run_dir>` and `<N>` are the worker's own (`<N>` is this iteration's `n`).
All artifacts for this step are strictly confined within `<run_dir>`:

*   Optimized kernel input: `<run_dir>/iter<N>/optimized.py`
*   Autotune results input: `<run_dir>/iter<N>/autotune_results.json`
*   Autotune summary output: `<run_dir>/iter<N>/autotune_summary.md`


--------------------------------------------------------------------------------

Your goal is to summarize the autotuning results provided in `<run_dir>/iter<N>/autotune_results.json`, report the best
configuration and latency, and verify if the best configuration was applied if
the status was success.

Check the status of the autotuning results:

### Case 1: If the status is "success"

You must:

1.  Extract the `"best_config"` and `"best_time_ms"` from the results file `<run_dir>/iter<N>/autotune_results.json`.
2.  Verify that the best configuration was applied correctly to the kernel code
    by reading the file located at `<run_dir>/iter<N>/optimized.py` using the `Read`
    tool.
3.  Write a clear summary into your report. Do NOT list all tested
    configurations from `all_results`.

### Case 2: If the status is "failed" or "error"

You must:

1.  Report the error message.

In all cases, you must: Write a clear summary into your report. Do NOT list
all tested configurations from `all_results`.

Please use the following format for your summary:

### Autotuning Results

-   **Status**: [Success / Failed]
-   **Best Configuration**: `[JSON or description of best config]`
-   **Latency**: `[Time]` ms
-   **Applied to File**: [Yes / No]

### Output Requirement

You **must** use the `Write` tool to save your autotuning summary report (including status, best configuration, latency, and verification of application) to `<run_dir>/iter<N>/autotune_summary.md`.

The `"success"` / `"failed"` values above belong to the `autotune_results`
file — they are the sweep's own record, written by `apply_best_config.py`. Do
not rename or reinterpret them.

When the sweep failed, report no latency anywhere, only an account of why.
There was no best time, and reporting one would be inventing a measurement.

Once `<run_dir>/iter<N>/autotune_summary.md` is written, continue to the
worker's Phase 5.
