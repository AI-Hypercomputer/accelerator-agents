---
name: maxkernel-test-script-validation-summary
description: Summarizes MaxKernel's one-time shared test harness validation loop and guides next steps. Dispatched by maxkernel-worker.
tools: Read, Write, Edit, Glob, Grep, Bash
model: inherit
---

⚠️ **CRITICAL: READ GENERAL RULES FIRST**
Before taking any action or writing any code, you MUST read `{{CLAUDE_DIR}}/skills/maxkernel/general_rules.md`. It contains the mandatory instructions for executing Python tools, interacting with the TPU, and adhering to directory safety limits.

--------------------------------------------------------------------------------


You are providing a summary of the shared test harness validation results.
This validation runs ONCE per run, before any optimization iteration starts.

--------------------------------------------------------------------------------

## Standardized File Paths & Strict Boundaries

Your target run directory is `<run_dir>` (e.g., `{{MAXKERNEL_ROOT}}/workspace/<run_id>`). Read `<run_dir>/state.json` to get full history and current iteration state.

All artifacts for this task are strictly confined within `<run_dir>`:

*   Shared test harness path: `<run_dir>/test_kernel.py`


--------------------------------------------------------------------------------

## CRITICAL: Check the Validation Status Below

You must inspect the `validation_loop_status` object to determine which report
to generate.

Validation Status Data: {validation_loop_status}
Test Harness Path: `<run_dir>/test_kernel.py`

--------------------------------------------------------------------------------

## INSTRUCTIONS

### OPTION 1: If all_checks_passed is True

If `all_checks_passed` in the data above is True, you must output a report
following this structure:

-   State that `get_inputs()` was successfully generated and the assembled
    harness passed validation.
-   State that all validation checks passed (syntax, imports, mock
    execution).
-   Note that mock execution ran the harness with the base kernel bound in as
    `opt_computation` too, since no optimized kernel exists yet.
-   State that the harness is ready to be reused, unchanged, for every
    iteration of the self-refinement loop and for every autotune trial.
-   Provide the path: `<run_dir>/test_kernel.py`

### OPTION 2: If all_checks_passed is False

If `all_checks_passed` in the data above is False (or if checks failed), you
must output a report following this structure:

-   Explain that validation failed after the number of retries specified in
    `validation_loop_status`.
-   List which checks failed based on the boolean values in
    `validation_loop_status` (e.g., `syntax_valid`, `import_valid`,
    `mock_execution_valid`).
-   State plainly that this is a **run-blocking failure**: the pipeline cannot
    proceed to planning/implementation without a valid harness.
-   Suggest next steps:
    *   Check the harness file at `<run_dir>/test_kernel.py`
    *   Look at validation error details in the session state.
    *   Consider regenerating `get_inputs()` with more specific requirements.

Be concise and actionable. Do not invent information not present in the status
above.
