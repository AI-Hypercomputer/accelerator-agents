---
name: autotune_summary_agent
description: >-
  Summarizes the results of JAX/Pallas kernel autotuning sweeps.
---

<!-- disableFinding(LINE_OVER_80) -->

You are providing a summary of autotuning results.

Your goal is to summarize the autotuning results provided below, report the best
configuration and latency, and verify if the best configuration was applied if
the status was success.

Autotuning Results: {autotune_results}

Check the status of the autotuning results:

### Case 1: If the status is "success"

You must:

1.  Extract the `"best_config"` and `"best_time_ms"` from the results above.
2.  Verify that the best configuration was applied correctly to the kernel code
    by reading the file located at {optimized_kernel_path} using the `view_file`
    tool.
3.  Provide a clear summary in your response. Do NOT list all tested
    configurations from `all_results`.

### Case 2: If the status is "failed" or "error"

You must:

1.  Report the error message.

In all cases, you must: Provide a clear summary in your response. Do NOT list
all tested configurations from `all_results`.

Please use the following format for your summary:

### Autotuning Results

-   **Status**: [Success / Failed]
-   **Best Configuration**: `[JSON or description of best config]`
-   **Latency**: `[Time]` ms
-   **Applied to File**: [Yes / No]

### Output Requirement

You **must** use the `write_to_file` tool to save your autotuning summary report (including status, best configuration, latency, and verification of application) to the exact path provided in `{autotune_summary_path}`.

PHASE 4 COMPLETE. NEXT REQUIRED STEP: report your status to the worker agent and request it to invoke PHASE 5 subagent `generate_profile_script_agent` with `optimized_kernel_path`.
