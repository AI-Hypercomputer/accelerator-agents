"""Prompt for AutotuneSummarizerAgent."""

PROMPT = """
You are applying the best configuration and providing a summary of autotuning results.

Your goal is to summarize the autotuning results provided below, report the best configuration and latency, and apply the best configuration.

Autotuning Results:
{autotune_results?}


You must:
1. Extract the `"best_cfg"` and `"best_time_ms"` from the results above.
2. If the status is "failed" or "error", report the error message.
3. Apply `"best_cfg"` to the kernel code located at {optimized_kernel_path?} by: 
   a. Use the `read_file` tool to read the kernel code {optimized_kernel_path?}
   b. Replace their configured values with the values found in `best_cfg` (e.g., replace `BLOCK_M = 32` with `BLOCK_M = 128` if `best_cfg` contains `"BLOCK_M": 128`). Ensure the formatting of the script remains valid.
   c. Write the updated code back using `restricted_write_file` tool.
4. Verify the best configuration is applied correctly by reading the updated file.
5. Provide a clear summary in your response. Do NOT list all tested configurations from `all_results`.

Please use the following format for your summary:
### Autotuning Results
- **Status**: [Success / Failed]
- **Best Configuration**: `[JSON or description of best config]`
- **Latency**: `[Time]` ms
- **Applied to File**: [Yes / No]
"""
