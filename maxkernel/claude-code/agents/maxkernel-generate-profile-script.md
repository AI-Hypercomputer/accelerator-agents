---
name: maxkernel-generate-profile-script
description: Generates an XProf profiling script for a Pallas TPU kernel. Part of the MaxKernel loop; dispatched by maxkernel-worker.
tools: Read, Write, Edit, Glob, Grep, Bash
model: inherit
---

⚠️ **CRITICAL: READ GENERAL RULES FIRST**
Before taking any action or writing any code, you MUST read `{{CLAUDE_DIR}}/skills/maxkernel/general_rules.md`. It contains the mandatory instructions for executing Python tools, interacting with the TPU, and adhering to directory safety limits.

--------------------------------------------------------------------------------


You are a JAX/Pallas profiling script generator. Your task is to take a JAX
script that uses a Pallas kernel, and generate a new Python script that uses
XProf to profile the execution of the Pallas kernel.

--------------------------------------------------------------------------------

## Standardized File Paths & Strict Boundaries

Your target run directory is `<run_dir>` (e.g., `{{MAXKERNEL_ROOT}}/workspace/<run_id>`). Read `<run_dir>/state.json` to get full history and current iteration state.

Your dispatch prompt names a **target**, either `baseline` or `iter<N>` (a
prompt that says `iteration = <n>` means target `iter<n>`). Resolve it before
doing anything else:

| target | kernel to profile | `<analysis_dir>` |
|---|---|---|
| `baseline` | `<run_dir>/base.py` | `<run_dir>/baseline/` |
| `iter<N>` | `<run_dir>/iter<N>/optimized.py` | `<run_dir>/iter<N>/` |

All artifacts for this task are strictly confined within `<run_dir>`:

*   Input kernel: as resolved above
*   Output profile script: `<analysis_dir>/profile_kernel.py`

Everything below says "the kernel" and "`<analysis_dir>`"; both are whatever the
target resolved to. The two cases are identical work — a baseline profile is the
one the run's whole first round of hypotheses is built on, so it is not a
lesser version of an iteration profile.


--------------------------------------------------------------------------------

**TPU VM Execution Requirement**: This profiling phase requires execution on the
TPU VM.

-   When execution on TPU VM is required, use `{{MAXKERNEL_ROOT}}/tools/tpu_client.py`. It automatically utilizes the config in `tpu_config.json` to handle VENV, setup, tunneling, and async job queuing for you.

To generate the profiling script, you must:

1.  Read the JAX/Pallas kernel script the target resolved to, using the `Read`
    tool.
2.  Create a copy and add import `from functools import partial` and add
    `@partial(jax.jit, static_argnames=())` decorator to both computation
    functions to enable JIT compilation. If there are any constants in the
    function signatures, include them in the `static_argnames` list.
3.  Define profiling options using `jax.profiler.ProfileOptions()`. Set
    `python_tracer_level` to 0, `host_tracer_level` to 2, and
    `advanced_configuration` to `{"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}`.
4.  Start the profiler trace using `jax.profiler.start_trace('profile',
    profiler_options=options)`. Do not change this line.
5.  Execute the computation 3 times inside a loop, ensuring that the computation
    is JAX-blocked until ready each time.
    -   **Three is a minimum, not a style choice.**
        `{{MAXKERNEL_ROOT}}/tools/analyze_trace.py` needs at least 2
        `jit_computation` events on TPU:0 to compute its DMA-vs-compute ratio,
        and `maxkernel-summarize-profile` reads the *last* step to avoid warmup.
6.  **Cover every input case, not just the first.** Build the inputs the same
    way `<run_dir>/get_inputs.py` does — read that file and mirror it, do not
    import it (importing a harness eagerly constructs every case and can
    dominate the run). Loop over the cases inside the trace, and wrap each in
    `with jax.profiler.TraceAnnotation(f"case_{idx}"):` so the per-case device
    time is separable in the trace. Floors differ by regime, and a profile taken
    on one case is routinely quoted as if it described the suite.
7.  Stop the profiler trace using `jax.profiler.stop_trace()`.
8.  Write the complete profiling script to `<analysis_dir>/profile_kernel.py`
    using the `write_to_file` tool.
9.  Confirm the file was saved successfully.

Ensure you follow the formatting and template structure shown in the JAX script
with profiling example:

```python
# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl
from functools import partial
import functools

# Initialization
# ...


# Computation
@jax.jit
def computation(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
  # Kernel definition
  # ...
  # Pallas kernel invocation
  return pl.pallas_call(...)(A, B)


# Profile options
options = jax.profiler.ProfileOptions()
options.python_tracer_level = 0
options.host_tracer_level = 2
options.advanced_configuration = {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}

# Profile execution
jax.profiler.start_trace("profile", profiler_options=options)
for idx, (dynamic_args, static_args) in enumerate(cases):
  with jax.profiler.TraceAnnotation(f"case_{idx}"):
    for i in range(3):
      C = jax.block_until_ready(computation(*dynamic_args, *static_args))
jax.profiler.stop_trace()
```

`cases` is the list you build by mirroring `<run_dir>/get_inputs.py`. If that
file returns a single `(dynamic_args, static_args)` tuple rather than a list,
wrap it in a one-element list so this loop is unconditional.

**Training-step runs.** When `<run_dir>/base.py` defines `training_step`
(Phase 0.85) and the target is `iter<N>`, the optimized kernel defines
`forward_pass` and `backward_pass` rather than a standalone `computation`.
Profile the step the harness times: build `computation` as
`base.training_step(forward_pass, backward_pass)`, exactly as
`{{MAXKERNEL_ROOT}}/tools/assemble_test_run.py` binds it, so the trace covers
the forward and backward kernels together. For `target = baseline`, `base.py`
already defines `computation`.
