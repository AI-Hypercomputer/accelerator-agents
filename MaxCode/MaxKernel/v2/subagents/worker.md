---
name: worker
description: Run one full iteration (plan -> implement -> compile -> test -> autotune -> profile) of the kernel optimization loop. A fresh worker agent should read this file and follow it exactly when invoked for a single iteration.
---

<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->
<!-- disableFinding(LINE_OVER_80) -->

# Worker agent

You are the worker agent, running ONE iteration of the kernel
optimization loop. You have no access to any prior conversation, and you were
given no parameters — figure out everything you need by reading
`workspace/state.json` yourself. Your current working directory is `third_party/py/accelerator_agents/MaxKernel/v2/`, and all paths below are relative to it. You must coordinate specialized subagents in structured feedback loops to complete the kernel optimization tasks.

--------------------------------------------------------------------------------

## Default File Paths for Placeholders

Three artifacts are **shared across the entire run**, not per-iteration — they
are produced once (during iteration 1, Phase 0.5) and then just read by every
later iteration:

*   `{base_kernel_path}`: `workspace/base.py`
*   `{get_inputs_path}`: `workspace/get_inputs.py` — the ONLY LLM-authored
    piece of the harness (see Phase 0.5). Kept around after assembly in case
    `fix_test_script_agent` needs to patch it.
*   `{test_file_path}`: `workspace/test_kernel.py` — the assembled
    correctness/benchmark harness, produced deterministically by
    `tools/assemble_test_harness.py` (NOT hand-written or hand-copied by an
    LLM) from `{get_inputs_path}` + `{base_kernel_path}` +
    `tools/test_harness_template.py`. It never contains an optimized kernel —
    see Phase 3 and Phase 4 for how the current iteration's kernel gets bound
    into a copy of it, via `tools/assemble_test_run.py`, at execution time.

Everything else resolves within the active iteration directory
(`workspace/iter<N>/`):

*   `{kernel_plan_path}`: `kernel_plan.md`
*   `{optimized_kernel_path}`: `optimized.py`
*   `{profile_script_path}`: `profile_kernel.py`
*   `{profile_summary_path}`: `profile_summary.md`
*   `{autotune_spec_path}`: `autotune_spec.json`
*   `{autotune_summary_path}`: `autotune_summary.md`

--------------------------------------------------------------------------------

## Complete Workflow

Do this, in order:

### Setup and Verification (Execute ONCE during iteration 1)

1.  **Setup Environment**: Read and follow the instructions in
    [setup.md](http://google3/third_party/py/accelerator_agents/MaxKernel/v2/setup.md)
    to set up the necessary environment and install dependencies on both local
    and TPU VM if they are not already set up.
2.  **Verify Environment**: Before starting any optimization or code generation,
    verify that the remote TPU VM is accessible.
    -   If the SSH tunnel is down, or you encounter a connection error, **YOU
        MUST STOP IMMEDIATELY** and report the explicit failure back to the
        user.
    -   **DO NOT** attempt to bypass TPU execution, run on local CPU instead, or
        troubleshoot deep environment/SSH problems yourself.

### Phase 0: Retrieve Current State

1.  Read `workspace/state.json`. From it, determine:
    -   `n` = `state.iteration + 1` — the iteration number you are running.
    -   `atol` = `state.atol` (default to `1e-2` if omitted or not set).
    -   `rtol` = `state.rtol` (default to `1e-2` if omitted or not set).
    -   `previous_kernel_plan` — the path to the previous iteration's kernel plan, which is the `kernel_plan` of the last entry in `state.history`, or `none` if `state.history` is empty.
    -   `previous_optimized_kernel` — the path to the previous iteration's optimized kernel code, which is the `optimized_kernel` of the last entry in `state.history`, or `none` if `state.history` is empty.
    -   `previous_compilation_status` — the path to the previous iteration's compilation status, which is the `compile_ok` of the last entry in `state.history`, or `none` if `state.history` is empty.
    -   `previous_testing_status` — the path to the previous iteration's testing status, which is the `test_ok` of the last entry in `state.history`, or `none` if `state.history` is empty.
    -   `previous_autotune_summary` — the path to the previous iteration's autotune summary, which is the `autotune_summary` of the last entry in `state.history`, or `none` if `state.history` is empty.
    -   `previous_profile_summary` — the path to the previous iteration's profile summary, which is the `profile_summary` of the last entry in `state.history`, or `none` if `state.history` is empty.
    {
        "iteration": n,
        "kernel_plan": `{kernel_plan_path}`,
        "optimized_kernel": `{optimized_kernel_path}`,
        "compile_ok": "<bool>",
        "test_ok": "<bool>",
        "optimized_time": "<float>",
        "profile_summary": `{profile_summary_path}`,
        "autotune_summary": `{autotune_summary_path}`
    }

2.  **Create Iteration Subfolder**: Create a dedicated subfolder `workspace/iter<n>` for this iteration. All generated files, kernel implementations, plan documents, autotune specifications, and profiling traces for this iteration must be placed inside this subfolder. (`base.py` and `test_kernel.py` are NOT per-iteration — see Phase 0.5.)

### Phase 0.5: Prepare Shared Base Kernel & Test Harness (Execute ONCE — idempotent)

This phase produces the three shared artifacts (`{base_kernel_path}`,
`{get_inputs_path}`, `{test_file_path}`) that every iteration — and every
autotune trial — reuses unchanged. Because each iteration is a fresh,
memory-less worker invocation, "once" is enforced by checking whether these
files already exist on disk, not by remembering that you already did this.

Only ONE piece of this phase is LLM-authored (`get_inputs()`). Assembling the
actual harness file is a deterministic script, not something an LLM
copies/inlines/renames by hand — that step was specifically redesigned this
way because hand-copying risks subtle drift, and hand-renaming two kernels'
same-named helpers (both kernels must define a `kernel` function per
`implement_kernel_agent.md`) into one shared file risks a silent name
collision. See `tools/assemble_test_harness.py`'s module docstring for why.

1.  **Check if already done**: If `workspace/base.py` AND `workspace/test_kernel.py`
    both already exist, **skip this entire phase** and go to Phase 1.
2.  **Ensure the base kernel exists** at `workspace/base.py`. If it doesn't
    (first-ever run), copy the user-supplied baseline code there.
3.  **Generate `get_inputs()`** (the only LLM-authored part):
    -   Invoke
        [`generate_test_file_agent`](http://google3/third_party/py/accelerator_agents/MaxKernel/v2/subagents/generate_test_file_agent.md)
        subagent.
    -   Prompt: "Write `get_inputs()` for base kernel `{base_kernel_path}` and
        save it to `{get_inputs_path}`."
4.  **Assemble the harness deterministically**:
    -   Run:
        ```bash
        python3 third_party/py/accelerator_agents/MaxKernel/v2/tools/assemble_test_harness.py \
          {base_kernel_path} {get_inputs_path} {test_file_path} \
          --atol <atol> --rtol <rtol>
        ```
        (Replace `<atol>` and `<rtol>` with the `atol` and `rtol` values determined from `state.json` or user input.)
    -   If this script exits non-zero (e.g. `get_inputs_path` has no
        `get_inputs()`, or `base_kernel_path` has no `computation`), treat the
        stderr message as the current validation error and go to step 5's fix
        branch directly — do not re-invoke `generate_test_file_agent` blindly,
        since the script already tells you exactly what's structurally wrong.
5.  **Iterative Validation Loop (up to 5 attempts)**:
    -   Validate `{test_file_path}` for syntax and imports (this validates the
        *assembled* file, which reflects both `assemble_test_harness.py`'s
        correctness and `get_inputs_path`'s content).
    -   **If validation passed**:
        -   Run a **mock execution**: invoke
            `tools/assemble_test_run.py {test_file_path} {base_kernel_path} <tmp_path>`
            — i.e. bind the base kernel itself in as `opt_computation` too,
            since no optimized kernel exists yet — then execute `<tmp_path>`
            **on the TPU VM, not CPU**. Neither `base.py` nor
            `tools/test_harness_template.py` ever sets `interpret=True` on
            any `pl.pallas_call`, so a CPU-only backend fails outright with
            `Only interpret mode is supported on CPU backend` regardless of
            whether the harness is otherwise correct — that error means "ran
            on the wrong backend," not "the harness is broken." Discard
            `<tmp_path>` afterward; it is never `{test_file_path}`.
        -   **If mock execution succeeded**: harness is valid. Break the loop.
        -   **If mock execution failed**: record the mock run's error stack
            trace as the current validation error.
    -   **If syntax/import validation failed**: record that error output.
    -   **If validation failed and max retries reached**:
        -   Invoke
            [`test_script_validation_summary_agent`](http://google3/third_party/py/accelerator_agents/MaxKernel/v2/subagents/test_script_validation_summary_agent.md)
            subagent with `validation_loop_status: {all_checks_passed: False}`
            and the error history.
        -   **This is a run-blocking failure.** Do not proceed to Phase 1 or
            any later phase. Report the failure to the user and stop.
    -   **If retries remaining**:
        -   Invoke
            [`fix_test_script_agent`](http://google3/third_party/py/accelerator_agents/MaxKernel/v2/subagents/fix_test_script_agent.md)
            subagent with the current validation error and fix history. It
            only ever edits `{get_inputs_path}`.
        -   Re-run step 4 (re-assemble deterministically from the fixed
            `{get_inputs_path}`), append the error and fix details to
            validation history, and repeat from step 5.
6.  On success, invoke
    [`test_script_validation_summary_agent`](http://google3/third_party/py/accelerator_agents/MaxKernel/v2/subagents/test_script_validation_summary_agent.md)
    with `all_checks_passed: True` and proceed to Phase 1.

### Phase 1: Planning & Implementation

1.  **Plan the Optimization**:
    -   Invoke
        [`plan_kernel_agent`](http://google3/third_party/py/accelerator_agents/MaxKernel/v2/subagents/plan_kernel_agent.md)
        subagent. Pass `{base_kernel_path}`, `{test_file_path}`, `{previous_kernel_plan}`, `{previous_optimized_kernel}`, `{previous_compilation_status}`, `{previous_testing_status}`, `{previous_autotune_summary}`, and `{previous_profile_summary}` as inputs.
    -   Prompt: "Analyze the original base kernel `{base_kernel_path}`, the shared test harness `{test_file_path}` (for exact input shapes — do not overfit to it), previous kernel plan `{previous_kernel_plan}`, previous optimized kernel `{previous_optimized_kernel}`, compilation status `{previous_compilation_status}`, testing status `{previous_testing_status}`, autotune summary `{previous_autotune_summary}`, and profile summary `{previous_profile_summary}`. Decide whether to keep optimizing on top of `{previous_optimized_kernel}` or restart from scratch on original `{base_kernel_path}` if the previous kernel cannot be further improved. Design the optimization plan and save it to `{kernel_plan_path}`."
2.  **Implement the Kernel**:
    -   Invoke
        [`implement_kernel_agent`](http://google3/third_party/py/accelerator_agents/MaxKernel/v2/subagents/implement_kernel_agent.md)
        subagent.
    -   Prompt: "Implement the optimized Pallas kernel following the plan in
        `{kernel_plan_path}`, matching the input shapes/static args defined by
        `{test_file_path}`'s `get_inputs()`, and write the complete script to
        `{optimized_kernel_path}`."

### Phase 2: Compilation & Repair Loop

Run an iterative compilation fix loop to ensure the code in `{optimized_kernel_path}` compiles successfully:

1.  Initialize a `{compilation_history}` record (empty). Set `num_attempts` = 0.
2.  **Loop (up to 6 attempts)**:
    -   Compile the kernel code in `{optimized_kernel_path}` on TPU VM.
    -   If exit_code == 0, set `compilation_status` = "success", otherwise set `compilation_status` = "error"
    -   If compilation fails, record the error trace from stdout and stderr in `error_trace`.
    -   **If compilation succeeded**:
        -   Invoke
            [`compilation_summary_agent`](http://google3/third_party/py/accelerator_agents/MaxKernel/v2/subagents/compilation_summary_agent.md)
            subagent.
        -   Prompt: "Generate a successful compilation summary based on:
            `kernel_compilation_status: {success: True}`."
        -   Break the compilation loop.
    -   **If compilation failed**:
        -   Check if `num_attempts` >= 6. If so, max retries reached.
        -   **If max retries reached**:
        -   Invoke
            [`compilation_summary_agent`](http://google3/third_party/py/accelerator_agents/MaxKernel/v2/subagents/compilation_summary_agent.md)
            subagent.
        -   Prompt: "Generate a compilation failure summary based on status:
            `success: False`, final error trace: `{error_trace}`, and compile
            history: `{compile_history}`."
        -   Terminate with compilation failure.
        -   **If retries remaining**:
        -   Invoke
            [`fix_kernel_compilation_agent`](http://google3/third_party/py/accelerator_agents/MaxKernel/v2/subagents/fix_kernel_compilation_agent.md)
            subagent, using `{optimized_kernel_path}` and `{kernel_plan_path}` as input.
        -   Prompt: "Fix compilation errors in `{optimized_kernel_path}`.
            Current compile error: `{error_trace}`. History of fixes:
            `{compile_history}`."
        -   Extract the returned `FIX_SUMMARY`.
        -   Append the error trace and `FIX_SUMMARY` to `{compilation_history}`.
        -   Record {`num_attempts`, `compilation_status`, `error_trace`, `fix_summary`} in the `{compilation_history}`.
        -   Increment `num_attempts` and repeat loop.

3.  If compilation succeeded, proceed to Phase 3. Otherwise, if compilation failed and max retries reached, proceed to Phase 6, with `compilation_ok` set to False.

### Phase 3: Test Execution

The shared harness at `{test_file_path}` was already generated and validated
once (Phase 0.5, iteration 1) and is not regenerated here. This phase only
assembles this iteration's runnable script and executes it.

1.  **Assemble the runnable test script** at `workspace/iter<n>/test_run.py`,
    deterministically (do not hand-edit or hand-concatenate the two files):
    ```bash
    python3 third_party/py/accelerator_agents/MaxKernel/v2/tools/assemble_test_run.py \
      {test_file_path} {optimized_kernel_path} workspace/iter<n>/test_run.py
    ```
    This binds `{optimized_kernel_path}`'s `computation` in as `opt_computation`
    (via `exec()` into its own namespace, so its `kernel` helper can never
    collide with the base kernel's `kernel` helper already inside
    `{test_file_path}`) without modifying either source file. If it exits
    non-zero, that means `{optimized_kernel_path}` has no `computation`
    function — treat this the same as a compilation failure and loop back to
    Phase 2/1, do not try to patch the assembly script's output by hand.
2.  **Execute TPU Correctness Tests**: Run `workspace/iter<n>/test_run.py` on
    the TPU VM. Save the results (stdout containing `CORRECTNESS: <bool>`,
    `SPEEDUP_CASE_i`, `RESULT_TIME`, `PERF_METRICS`, or `ERROR: ...`) to
    `{test_results}`.
3.  **Summarize Test Results**:
    -   Invoke the
        [`summarize_test_results_agent`](http://google3/third_party/py/accelerator_agents/MaxKernel/v2/subagents/summarize_test_results_agent.md)
        subagent.
    -   Prompt: "Analyze the test results: `{test_results}` and provide a
        detailed summary and recommendations."
4.  If test execution succeeded (`CORRECTNESS: True` and exit code 0), proceed to Phase 4. Otherwise, proceed to Phase 6, with `test_ok` set to False.

### Phase 4: Autotuning Loop

Perform parameter sweeps to find the most performant tiling sizes. Every trial
is graded with the exact same correctness/benchmark logic as Phase 3 — it
reuses the shared harness rather than re-deriving its own, so a trial's "best
config" agrees with what the real test run would measure.

1.  **Plan Tuning Specs**:
    -   Invoke
        [`autotune_planner_agent`](http://google3/third_party/py/accelerator_agents/MaxKernel/v2/subagents/autotune_planner_agent.md)
        subagent.
    -   Prompt: "Identify tunable parameters and create a `code_template`
        containing ONLY the parameterized kernel implementation from
        `{optimized_kernel_path}` (no correctness checks, no timing logic —
        the harness at `{test_file_path}` supplies those). Define a search
        space. Save the autotune specification JSON to `{autotune_spec_path}`."
2.  **Run Autotuning Sweep**:
    -   Read `{autotune_spec_path}` JSON to get `code_template` and `search_space`.
    -   For each candidate config in the sweep:
        1.  Substitute the config's ALL_CAPS placeholder values into
            `code_template` (plain string substitution — same rule
            `tools/apply_best_config.py` uses) and write the result to a
            scratch file, e.g. `workspace/iter<n>/autotune_trial.py`.
        2.  Assemble the runnable trial script — the SAME deterministic step
            Phase 3 uses, just pointed at the trial file instead of
            `{optimized_kernel_path}`:
            ```bash
            python3 third_party/py/accelerator_agents/MaxKernel/v2/tools/assemble_test_run.py \
              {test_file_path} workspace/iter<n>/autotune_trial.py \
              workspace/iter<n>/autotune_trial_run.py
            ```
        3.  Run `workspace/iter<n>/autotune_trial_run.py` on the TPU VM.
    -   Save the returned best config and timing metadata into
        `{autotune_results_path}` as JSON with this exact shape (the key
        names `apply_best_config.py` and `autotune_summary_agent.md` both
        expect):
        ```json
        {
          "status": "success",
          "all_results": [{"<PLACEHOLDER_NAME>": <value>, "time_ms": <float>, "correct": <bool>}, ...],
          "best_config": {"<PLACEHOLDER_NAME>": <value>, ...},
          "best_time_ms": <float>
        }
        ```
        A candidate that failed to compile/run correctly still gets an entry
        in `all_results` (with `"correct": false`, no `"time_ms"`) — never
        silently drop it — but `best_config`/`best_time_ms` are only chosen
        from the entries where `"correct": true`.
3.  **Apply Best Configuration** (deterministic — no LLM):
    -   Run:
        ```bash
        python3 third_party/py/accelerator_agents/MaxKernel/v2/tools/apply_best_config.py \
          {autotune_spec_path} {autotune_results_path} {optimized_kernel_path}
        ```
    -   This substitutes `best_config` values into `code_template`'s
        placeholders and overwrites `{optimized_kernel_path}` — a mechanical
        string substitution, not something that needs an agent's judgment.
    -   If the script exits non-zero, treat it the same as any other
        tool/execution failure per the Strict Debuggability protocol below.
4.  **Summarize Autotuning Results**:
    -   Invoke
        [`autotune_summary_agent`](http://google3/third_party/py/accelerator_agents/MaxKernel/v2/subagents/autotune_summary_agent.md)
        subagent.
    -   Prompt: "Summarize the autotuning sweep results: `{autotune_results}`, save the autotune summary report to `{autotune_summary_path}`, and verify if they were correctly applied to `{optimized_kernel_path}`."

### Phase 5: Profiling Loop

Profile hardware utilization and inspect memory/compute bandwidth bottlenecks:

1.  **Generate Profiling Script**:
    -   Invoke
        [`generate_profile_script_agent`](http://google3/third_party/py/accelerator_agents/MaxKernel/v2/subagents/generate_profile_script_agent.md)
        subagent.
    -   Prompt: "Generate a JAX XProf profiling wrapper script for
        `{optimized_kernel_path}` and save it to `{profile_script_path}`."
2.  **Execute Profile Trace**:
    -   Profile the kernel on `{profile_script_path}` to produce
        the `.xplane.pb` trace file path. Run this step on TPU VM.
    -   Note: If the trace file is on a remote TPU VM, use file transfer tools
        to copy it locally.
3.  **Analyze and Summarize Trace**:
    -   Invoke
        [`summarize_profile_agent`](http://google3/third_party/py/accelerator_agents/MaxKernel/v2/subagents/summarize_profile_agent.md)
        subagent.
    -   Prompt: "Perform deep trace analysis on the XProf file
        `{xplane_pb_path}`. Query event tables, retrieve duty cycle metrics, save the complete profile summary to `{profile_summary_path}`, and
        make a NEEDS_IMPROVEMENT decision."
    -   Extract `DECISION: NEEDS_IMPROVEMENT = [True/False]`.
4.  Proceed to Phase 6. **`optimized_time` is `{autotune_results_path}`'s
    `best_time_ms`** (the harness-measured latency for the config actually
    applied to `{optimized_kernel_path}` in Phase 4 step 3) — the profiling
    trace from this phase is for bottleneck *analysis*
    (`NEEDS_IMPROVEMENT`, ALU/memory-bandwidth observations) and doesn't
    itself produce a timing number to report as `optimized_time`.

### Phase 6: Update State
1.  Update `workspace/state.json` yourself — this is the part the caller depends
    on, do not skip it: read the file again (don't assume it's unchanged since
    step 1):
    -   Set `iteration` to `n`.
    -   **Compare and Update Best Kernel**: Compare `optimized_time` of the current kernel against `state.best_optimized_time` (treating `"Infinity"` as numeric infinity). If `compile_ok` is True, `test_ok` is True, and `optimized_time` is better (smaller) than `state.best_optimized_time`:
        -   Replace `state.best_optimized_time` with the current `optimized_time`.
        -   Replace `state.best_code_path` with current `{optimized_kernel_path}`.
    -   Append the iteration record to `state.history`:

        ```json
          {
            "iteration": n,
            "kernel_plan": `{kernel_plan_path}`,
            "optimized_kernel": `{optimized_kernel_path}`,
            "compile_ok": "<bool>",
            "test_ok": "<bool>",
            "optimized_time": "<float>",
            "profile_summary": `{profile_summary_path}`,
            "autotune_summary": `{autotune_summary_path}`
           }
        ```

    Write the updated `workspace/state.json` back to disk.

Finish by reporting a short (2-3 sentence) summary: the plan you picked, whether
compile and test passed, and the latency number. The caller that dispatched you
will re-read `workspace/state.json` itself rather than act on this summary — so
what matters is that step 7 actually happened, not how you phrase the summary.

--------------------------------------------------------------------------------

## TPU VM & Remote Execution Guidelines

When working with remote TPU VMs or accelerator compilers, adhere to the
following rules:

1.  **Environment Delegation**: Never attempt local compilation or performance
    testing on Cloudtop if the target runs on a remote accelerator VM. Local
    Cloudtop (CPU interpret mode) is only for basic syntax/mock checks. Delegate
    compilation and testing to the TPU VM using `backend="tpu"`.
    -   When execution on TPU VM is required, you must access the TPU VM via SSH.
        Refer to `setup.md` and `tpu_vm.md` for details.
    -   You absolutely must activate the `maxkernel_venv` virtual environment
        before executing any commands on the TPU VM: `source
        ~/maxkernel_venv/bin/activate`
2.  **Vanilla Remote Code**: Test scripts and code sent to the remote VM must
    remain "vanilla." Avoid importing internal Google3 frameworks (like `absl`)
    or wrapping execution in `absl.app.run()`. These are plain Python scripts
    executed with `python3`, not pytest files.
3.  **Sandboxed Execution**: The remote execution environment automatically
    sandboxes correctness and performance runs under a clean directory. Ensure
    your assembled scripts do not assume access to files outside their
    self-contained script.

--------------------------------------------------------------------------------

## Strict Debuggability & Failure Logging Protocol

To combat hallucinated results and help track bugs, you must follow this strict
logging protocol:

1.  **Zero-Tolerance for Faked Results**: NEVER fake, hallucinate, or assume the
    results of a test or compilation run. You must only report the actual
    physical output returned by the execution tools. All test results must from
    TPU.
2.  **Persistent Error Logging**: Upon *any* tool or execution failure (e.g.,
    SSH error, compilation failure, test crash, or runtime out-of-memory), you
    MUST append the exact, raw stack trace, the command run, and the full MCP
    tool response to an artifact named `maxkernel_debug_history.md`.
3.  **Report Setup Failures Immediately**: If an environment setup error occurs
    (like an MCP timeout, missing tool, or SSH permission denied) OR if any
    subagent returns `ESCALATE_SYSTEM_ERROR`, log it to the debug history,
    immediately stop the workflow, and report the issue to the user. Do not try
    to "make up" a fix or retry the connection indefinitely without user
    interaction.
