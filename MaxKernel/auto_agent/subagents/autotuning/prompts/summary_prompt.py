"""Prompt for AutotuneSummarizerAgent."""

PROMPT = """
You are providing a summary of autotuning results.

Your goal is to summarize the autotuning results provided below, report the best configuration and latency, and verify if the best configuration was applied if the status was success.

Autotuning Results:
{autotune_results?}


Check the status of the autotuning results:

### Case 1: If the status is "success"
You must:
1. Extract the `"best_config"` and `"best_time_ms"` from the results above.
2. Verify that the best configuration was applied correctly to the kernel code by reading the file located at {optimized_kernel_path?}.
3. Provide a clear summary in your response. Do NOT list all tested configurations from `all_results`.

### Case 2: If the status is "failed" or "error"
You must:
1. Report the error message.

In all cases, you must:
Provide a clear summary in your response. Do NOT list all tested configurations from `all_results`.

Please use the following format for your summary:
### Autotuning Results
- **Status**: [Success / Failed]
- **Best Configuration**: `[JSON or description of best config]`
- **Latency**: `[Time]` ms
- **Applied to File**: [Yes / No]
"""
