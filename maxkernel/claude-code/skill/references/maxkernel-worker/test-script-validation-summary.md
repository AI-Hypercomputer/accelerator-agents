# Test script validation summary

`maxkernel-worker` reads this at the end of Phase 0.9 to summarize the shared
test harness validation results. This validation runs ONCE per run, before any
optimization iteration starts. You write the report yourself, from the
validation status and history you already hold.

--------------------------------------------------------------------------------

## Inputs

`<run_dir>` is the worker's own.

*   Shared test harness path: `<run_dir>/test_kernel.py`


--------------------------------------------------------------------------------

## CRITICAL: Check the Validation Status Below

You must inspect the `validation_loop_status` object to determine which report
to generate.

Validation Status Data: {validation_loop_status}
Test Harness Path: `<run_dir>/test_kernel.py`

--------------------------------------------------------------------------------

## INSTRUCTIONS

See [Where the report goes](#where-the-report-goes) for what to do with the
report once it is written.

### OPTION 1: If all_checks_passed is True

If `all_checks_passed` in the data above is True, your report follows this
structure:

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

If `all_checks_passed` in the data above is False (or if checks failed), your
report follows this structure:

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

## Where the report goes

Append the report to `<run_dir>/maxkernel_debug_history.md`. Then:

-   when `all_checks_passed` is True, continue to Phase 1;
-   when it is False, this is a run-blocking failure: stop and report it to
    your caller with the report as the reason, and do not start Phase 1.

A failed validation still gets a full report. The loop above you can only
tell the user why the run stopped if your account reaches it intact.
