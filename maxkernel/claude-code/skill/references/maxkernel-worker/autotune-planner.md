# Plan the autotune sweep

`maxkernel-worker` reads this in Phase 4 step 1 to prepare the autotuning
specification for this iteration's kernel: identify the tunable parameters,
create a parameterized code template of the kernel, and define the search
space to minimize execution time. You do this yourself, in this context, with
your own tools.

--------------------------------------------------------------------------------

## Inputs and outputs

`<run_dir>` and `<N>` are the worker's own (`<N>` is this iteration's `n`).
All artifacts for this step are strictly confined within `<run_dir>`:

*   State file: `<run_dir>/state.json` (contains `tpu_version` and absolute paths to all previous history)
*   Optimized kernel input: `<run_dir>/iter<N>/optimized.py`
*   Shared test harness reference: `<run_dir>/test_kernel.py`
*   Autotune spec output: `<run_dir>/iter<N>/autotune_spec.json`


--------------------------------------------------------------------------------

To prepare for autotuning, you must:

1.  Use `Read` tool to read the optimized kernel code located at
    `<run_dir>/iter<N>/optimized.py`.
2.  Identify the parameters that can be tuned in the kernel (e.g., BLOCK_M,
    BLOCK_N).
3.  Create a `code_template` which is the ENTIRE optimized kernel code, but
    replacing the specific parameter values with placeholders enclosed in
    curly braces (for example, if the parameter is BLOCK_M, use it enclosed
    in curly braces as the placeholder).
    -   **Placeholder names MUST be ALL_CAPS AND wrapped in literal curly
        braces** (e.g. `{BLOCK_M}`, `{NUM_STAGES}`) — this is a hard
        requirement, not just a style preference.
        `{{MAXKERNEL_ROOT}}/tools/apply_best_config.py` (the deterministic step that substitutes
        `best_config` back into `code_template`) only recognizes ALL_CAPS
        `{NAME}` patterns — literal `{` and `}` characters included — as
        placeholders, specifically so it never confuses a real placeholder
        with incidental code that happens to look like one — e.g. an
        f-string `f"{x}"` or a one-element set literal `{x}` inside the
        kernel. A lowercase or mixed-case placeholder will silently fail to
        round-trip.
        -   **Do NOT write the placeholder as a bare identifier** (e.g.
            `BLOCK_Q = BLOCK_Q_VAL`), even one that looks like an obvious
            stand-in name. Without the curly braces it doesn't match
            `apply_best_config.py`'s pattern at all, so it is never
            substituted, never flagged as an error, and silently ships as a
            `NameError` in the final kernel. The correct form assigns the
            tunable value straight from the placeholder itself, e.g.
            `BLOCK_Q = {BLOCK_Q}` (which becomes `BLOCK_Q = 128` after
            substitution) — there is no need for a separate `_VAL`-suffixed
            name.
    -   `code_template` must contain ONLY the kernel implementation (the
        `kernel` and `computation` functions and any helpers they need) --
        nothing else.
    -   **Do NOT author a correctness check, a timing/benchmark loop, or any
        print statements.** The fixed, validated correctness+benchmark harness
        at `<run_dir>/test_kernel.py` (generated once in Phase 0.9, shared with
        every test run) is concatenated onto each trial's substituted
        `code_template` in Phase 4 step 2. Reinventing that logic
        here would let autotuning silently drift from the harness used for
        the real test run -- e.g. a different number of warmup/benchmark
        iterations -- so that the "best config" it finds is not actually best
        under the real evaluation.
    -   Keep the entry point named exactly `computation`, as in
        `<run_dir>/iter<N>/optimized.py` -- Phase 4 step 2 aliases it to
        `opt_computation` when assembling each trial, matching how the shared
        harness names things.
4.  Define a highly optimized, high-probability search space as a dictionary
    mapping placeholder names to lists of suggested values. You MUST follow
    these rules to minimize evaluation time and avoid sub-optimal
    configurations. **These are starting heuristics tuned for compute-bound
    ops like matmul, not hard limits** — for memory-bound/elementwise
    kernels, larger blocks (well above 256, even up to the full array/1024+)
    can genuinely be the fastest, since fewer, larger grid steps amortize
    per-step overhead better than many small ones. If measured results
    contradict the heuristic below, trust the measurement:
    -   **Hardware & TPU Version Alignment**: Read `tpu_version` from `state.json` (e.g. `TPU v5p`, `TPU v6e`, `TPU v7x`). Only suggest block sizes that align with target hardware efficiency (typically multiples of 32, 64, or 128 matching the target TPU MXU and vector units, e.g., `[64, 128]`). Avoid extremely small values (like `16`) or large values (like
        `256` or more) unless they are perfectly aligned with specific small
        tensor shapes -- or unless prior iterations' measured results
        suggest otherwise for this specific kernel.
    -   **Dimension Divisors**: Choose suggested block sizes that are clean,
        even divisors of the corresponding matrix or tensor shape dimensions to
        prevent compiler masking and branch overhead.
    -   **Total Combinations Limit**: Proactively limit the size of individual
        parameter lists so that the total Cartesian product (all possible
        combinations) stays small—ideally between **10 to 100 total combinations
        max**. Keep each parameter list to 2 or 3 high-probability values (e.g.,
        `[64, 128]`). Do not generate massive combinatorial sweeps.
5.  Write the `kernel_name`, `code_template`, and `search_space` to a JSON
    string and save it to `<run_dir>/iter<N>/autotune_spec.json` using the `Write` tool.
    The JSON file must have exactly this structure:

```json
{
  "kernel_name": "...",
  "code_template": "...",
  "search_space": { ... }
}
```

Note: `kernel_name` is kept for logging/traceability, but the harness always
calls the fixed entry point names (`base_computation`/`opt_computation`) --
it does not look up `kernel_name` dynamically.

6.  Writing the spec is the whole of this step; you do not run the sweep here.
    Once `<run_dir>/iter<N>/autotune_spec.json` is written, go back to the
    worker's Phase 4 step 2, which runs it.
