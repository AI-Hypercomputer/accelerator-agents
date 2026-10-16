# Fix the test script

`maxkernel-worker` reads this in Phase 0.9 step 5 when harness validation
failed and retries remain. You check the validation results and fix errors in
`<run_dir>/get_inputs.py`. This is the only file you can touch: the rest of the
shared harness at `<run_dir>/test_kernel.py` is assembled deterministically by
`{{MAXKERNEL_ROOT}}/tools/assemble_test_harness.py` from `<run_dir>/get_inputs.py` plus
`<run_dir>/base.py` plus the fixed `{{MAXKERNEL_ROOT}}/tools/test_harness_template.py`, so it
cannot itself contain a bug that a fix here would address -- if validation
keeps failing after `<run_dir>/get_inputs.py` looks correct, the problem is likely a
mismatch between what `get_inputs()` returns and what `<run_dir>/base.py`'s
`computation` actually expects.

--------------------------------------------------------------------------------

## Inputs and outputs

`<run_dir>` is the worker's own. All artifacts for this step are strictly
confined within `<run_dir>`:

*   Input generator to fix: `<run_dir>/get_inputs.py`
*   Base kernel reference: `<run_dir>/base.py`
*   Assembled test harness reference: `<run_dir>/test_kernel.py`


--------------------------------------------------------------------------------

**TPU VM Execution Requirement**: Mock validation runs the *assembled*
harness (produced fresh each retry by `assemble_test_harness.py` from your
latest `<run_dir>/get_inputs.py`) on the TPU VM -- not CPU. Neither `base.py` nor
the fixed harness template ever set `interpret=True` on any `pl.pallas_call`,
so running the assembled harness on a CPU-only backend fails outright with
`Only interpret mode is supported on CPU backend`, regardless of whether
`get_inputs()` and the base kernel are otherwise correct. Don't burn a retry
"fixing" `<run_dir>/get_inputs.py` in response to that error -- it means the
harness was run on the wrong backend, not that your file is wrong.

-   When execution on TPU VM is required, use `{{MAXKERNEL_ROOT}}/tools/tpu_client.py`. It automatically utilizes the config in `tpu_config.json` to handle VENV, setup, tunneling, and async job queuing for you.
-   You absolutely must activate the `maxkernel_venv` virtual environment on the
    TPU VM before execution: `source ~/maxkernel_venv/bin/activate` (the remote
    VM bootstraps its own venv at that fixed path; it is unrelated to the venv
    on the host machine).

## Context

`get_inputs()` file: `<run_dir>/get_inputs.py`
Assembled harness (read-only, do not edit): `<run_dir>/test_kernel.py`

**Validation Results** (from Phase 0.9 step 5):

-   Current validation error -- which check failed (syntax, import, or mock
    execution) and its output
-   Earlier errors and fixes from previous passes through this reference

## First: Check if the File Exists

**If `<run_dir>/get_inputs.py` does not exist or is empty:**

-   There is nothing to fix: Phase 0.9 step 3 did not produce it. Go back to
    step 3 and generate it first.
-   **STOP HERE**

## Second: Check for System/Connection Errors

**If the validation error contains the string `FATAL_CONNECTION_ERROR`:**

-   This is an unsolvable infrastructure error (e.g., SSH tunnel down, TPU
    unresponsive).
-   **DO NOT** attempt to write any code fixes.
-   **Immediately halt**: record `ESCALATE_SYSTEM_ERROR: <details of the
    error>` and follow step 3 of the worker's Strict Debuggability protocol,
    which stops the workflow.
-   **STOP HERE**.

## Third: Check if Fixes are Needed

1.  If there is no current validation error -- syntax, import and mock
    execution all passed:

    -   **All validations passed! No fixes needed.**
    -   Go back to Phase 0.9 step 5, which breaks the loop.
    -   **STOP HERE - do not modify the file**

2.  If any check failed → proceed to "Your Task" below.

## Tool Usage

1.  `Read`: To read `<run_dir>/get_inputs.py`, `<run_dir>/base.py`, and
    `<run_dir>/test_kernel.py` (for context on the error only -- never write to the
    latter).
2.  `Write`: To overwrite `<run_dir>/get_inputs.py` with the corrected
    version.

## Your Task (Only if Fixes are Needed)

### Step 1: Read the Current File and the Error Context

Use `Read` on `<run_dir>/get_inputs.py`. If the error trace references the
assembled harness (`<run_dir>/test_kernel.py`), read it too, but only to understand
*where* `get_inputs()`'s output diverges from what `base_computation` expects
-- not to edit it.

### Step 2: Identify and Fix the Error

-   **Syntax Errors**: fix Python syntax in `get_inputs()`.
-   **Import Errors**: fix/add imports `get_inputs()` needs.
-   **Mock Execution Errors**: usually a shape/arity mismatch between what
    `get_inputs()` returns and what `<run_dir>/base.py`'s `computation`
    expects — e.g. wrong number of `dynamic_args`/`static_args`, or a
    `(dynamic_args, static_args)` tuple malformed (must be exactly 2
    elements).

### Step 3: Write the Fixed File

Use `Write` to overwrite `<run_dir>/get_inputs.py` with the corrected
version.

**CRITICAL RULES:**

1.  **Only ever write to `<run_dir>/get_inputs.py`.** Never write to
    `<run_dir>/test_kernel.py` — Phase 0.9 step 4 regenerates it
    deterministically from this file after you're done.
2.  **DO NOT invent a new optimized kernel or `opt_computation` stub.**
3.  Keep the required return shape: a list of `(dynamic_args, static_args)`
    tuples.

**After writing:**

-   Append what you fixed to the validation history
-   Go back to Phase 0.9 step 4 to re-assemble the harness and validate
    again

## Important Notes

-   We are NOT fixing kernel bugs — only `get_inputs()`.
-   After your fix, Phase 0.9 re-runs `{{MAXKERNEL_ROOT}}/tools/assemble_test_harness.py` and
    validation runs again.
-   Unlike per-iteration kernel work, exhausting retries here is a
    **run-blocking failure**: nothing downstream (planning, implementation,
    testing, autotuning) can proceed without a valid harness. Say so plainly
    if you reach max retries.
