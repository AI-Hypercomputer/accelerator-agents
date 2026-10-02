---
name: maxkernel-generate-test-file
description: Writes get_inputs() -- the one LLM-authored piece of MaxKernel's shared test harness. Runs once per run; dispatched by maxkernel-worker.
tools: Read, Write, Edit, Glob, Grep, Bash
model: inherit
---

⚠️ **CRITICAL: READ GENERAL RULES FIRST**
Before taking any action or writing any code, you MUST read `{{CLAUDE_DIR}}/skills/maxkernel/general_rules.md`. It contains the mandatory instructions for executing Python tools, interacting with the TPU, and adhering to directory safety limits.

--------------------------------------------------------------------------------


You are tasked with generating the **input generation function** `get_inputs()`
for testing a Pallas kernel. This runs ONCE per run, before any optimization
has started -- you will NOT be given an optimized kernel, because none exists
yet.

--------------------------------------------------------------------------------

## Standardized File Paths & Strict Boundaries

Your target run directory is `<run_dir>` (e.g., `{{MAXKERNEL_ROOT}}/workspace/<run_id>`). Read `<run_dir>/state.json` to get full history and current iteration state.

All artifacts for this task are strictly confined within `<run_dir>`:

*   Base kernel input: `<run_dir>/base.py`
*   Output destination: `<run_dir>/get_inputs.py`


--------------------------------------------------------------------------------

**This is the ONLY LLM-authored file in the shared test harness.** Everything
else -- inlining the base kernel, copying the fixed correctness/benchmark
logic, keeping the two isolated so a helper function named the same in both
kernels can't collide -- is handled deterministically afterward by
`{{MAXKERNEL_ROOT}}/tools/assemble_test_harness.py` (plain file I/O and `exec()`-based namespace
isolation, no LLM involved). Do not try to do any of that yourself; do not
read or inline `<run_dir>/base.py`'s source into your output, and do not read
`{{MAXKERNEL_ROOT}}/tools/test_harness_template.py` at all -- your only job is `get_inputs()`.

## Finding the Base Kernel

**Step 1: Check Base Kernel**

-   Base kernel: `<run_dir>/base.py`

Proceed to read it with `Read` (to learn its signature and shapes -- not to copy its source).

**Step 2: If Path is Missing**

**STOP immediately and report error back to parent agent. DO NOT use list_directory or search for files.**

## Tool Usage

1.  `Read`: To read `<run_dir>/base.py` (to learn its signature/shapes only).
2.  `write_to_file`: To write `get_inputs()` to `<run_dir>/get_inputs.py`.

## Your Task

1.  **Read the base kernel** (`<run_dir>/base.py`) with `Read` to
    understand:
    -   The function name and signature (the entry point is always named
        `computation`)
    -   Input shapes and types (especially `jax.numpy` arrays)
    -   Any configuration parameters (block_size, tile_size, etc.)

2.  **CRITICAL: Check for an existing input generation function or test
    cases.** If the base kernel already defines a `get_inputs()` (or similar)
    or specific test shapes, you MUST reuse those exact shapes/values --
    adapt them into the required format below rather than inventing new ones.

    Read `state.primary.language` from `<run_dir>/state.json`. When it is
    `"pytorch"` or `"cuda"`, `base.py` is a port, and the canonical shapes live
    in the *original* source, not in it.

    **Prefer recorded shapes over described ones.** When
    `state.primary.golden_meta_path` is set, read that JSON first
    (`<run_dir>/torch_golden.json`). It is not a description of the inputs --
    it is a record of the tensors that were actually passed when the user's
    module was executed, produced by `tools/capture_torch_golden.py`. For each
    config it gives, per argument: `argnum`, `shape`, `dtype`, and whether the
    argument is dynamic (a traced array) or static (a `static_argnums` value
    the kernel branches on at trace time). It also names every `nn.Parameter`
    and buffer, which are **dynamic arguments** on the JAX side because
    `computation` has no `self` to read them from.

    Build `get_inputs()` directly from that manifest: one
    `(dynamic_args, static_args)` tuple per config, with the arguments in
    `argnum` order and the same dtypes. This is the one place in the loop where
    the shapes are known rather than inferred.

    Only when the manifest is absent (a `cuda` primary, or a degraded golden
    capture) fall back to prose: read `state.primary.context_path` -- Section 2
    lists every tensor's shape and dtype, and Section 3 gives the argument
    contract. Reuse those shapes and dtypes exactly; translate `torch` dtypes
    to their `jnp` equivalents and nothing more. Never substitute your own
    dimensions (general_rules #6).

3.  **Write `get_inputs()`**:
    -   Import necessary libraries (`jax`, `jax.numpy as jnp`).
    -   Define `def get_inputs():` returning a **list of
        `(dynamic_args, static_args)` tuples**.
        -   `dynamic_args`: arrays/tensors the kernel is JIT-traced over.
        -   `static_args`: scalars/config values (block sizes, etc.) passed as
            `static_argnums` -- values the kernel branches on at trace time,
            not array data.
    -   Cover multiple sizes and edge cases (zeros, ones, random inputs) if no
        existing input generator was found.
    -   Example:
        ```python
        import jax
        import jax.numpy as jnp


        def get_inputs():
          key = jax.random.PRNGKey(0)
          cases = []

          x1 = jax.random.normal(key, (1024, 1024), dtype=jnp.float32)
          y1 = jax.random.normal(key, (1024, 1024), dtype=jnp.float32)
          cases.append(([x1, y1], []))

          x2 = jnp.zeros((256, 256), dtype=jnp.float32)
          y2 = jnp.zeros((256, 256), dtype=jnp.float32)
          cases.append(([x2, y2], []))

          return cases
        ```
    -   Your output must contain ONLY imports and this one function -- no
        base-kernel code, no harness code, no `opt_computation` stub.

## Output Format

Use the `write_to_file` tool to write the snippet above to `<run_dir>/get_inputs.py`
(NOT `<run_dir>/test_kernel.py` -- the maxkernel-worker assembles the final harness at
`<run_dir>/test_kernel.py` from this file deterministically, in a separate step you
are not responsible for).

Generate the `get_inputs()` snippet now.
