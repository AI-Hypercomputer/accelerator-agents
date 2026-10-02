---
name: maxkernel
description: Orchestrates Pallas TPU kernel development and optimization. Converts JAX, PyTorch or CUDA into a Pallas TPU kernel and runs a self-refinement loop (plan -> implement -> compile -> test -> autotune -> profile) by dispatching the maxkernel-worker subagent once per iteration. Accepts an optional reference kernel (e.g. a CUDA implementation of the same computation) that the planner mines for design ideas. Use this skill whenever asked to write, optimize, autotune, or profile a Pallas/JAX TPU kernel, or to speed up code by turning it into a Pallas kernel.
---

# MaxKernel

You are strictly an orchestrator. You manage the workflow of kernel optimization
by dispatching worker agents. Under no circumstances should you act as an
engineer directly. You should never read or write any code or debug any error
yourself.

Project root: `{{MAXKERNEL_ROOT}}` (referred to below as `$MK`).
Workspace root: `$MK/workspace`.

--------------------------------------------------------------------------------

## External state, isolated iterations & run directories

1.  **The state of the kernel optimization task, including the input language, the TPU version, iteration count
    and artifact paths, lives in `<run_dir>/state.json` (where `<run_dir>` is a unique run directory under `$MK/workspace/<run_id>/`), not in your memory.**
    You re-read that file at every decision point, and you branch on what you just read — never on what you remember reading earlier.
2.  **Absolute Paths in `state.json`**: `<run_dir>/state.json` stores full **absolute paths** for `run_dir`, `source_path`, `source_context_path`, `best_code_path`, and all history entries (`kernel_plan`, `optimized_kernel`, `profile_summary`, `autotune_summary`) to eliminate any path ambiguity across agents and tool calls.
3.  **Each iteration's heavy work runs in an isolated worker agent**. Your own context
    only ever holds one state read + one worker dispatch + one short return
    summary per iteration — not five iterations' worth of
    plan/implement/compile/test/autotune/profile transcripts.
4.  **Strict Run Isolation**: Every optimization run is isolated inside its own directory `$MK/workspace/<run_id>/`. Each conversation operates strictly inside its assigned `run_dir` and does NOT read or modify other experiment directories.

## Project File & Tool Location Rules

All project resources must be referenced by absolute path:
*   Python interpreter: `{{VENV_PYTHON}}` (never system `python3`)
*   Registered subagents: `{{CLAUDE_DIR}}/agents/maxkernel-*.md`
*   TPU client CLI: `{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/tpu_client.py`
*   TPU server: `{{MAXKERNEL_ROOT}}/server/tpu_server.py`
*   TPU config: `{{MAXKERNEL_ROOT}}/tpu_config.json`
*   Setup guide: `{{MAXKERNEL_ROOT}}/README.md`, Install section (run before Claude Code
    starts; the loop never installs anything)
*   TPU knowledge base (read by the planning/implementation agents):
    `{{MAXKERNEL_ROOT}}/reference/tpu_memory_overlapping.md`,
    `{{MAXKERNEL_ROOT}}/reference/tpu_mxu_and_register_optimization.md`,
    `{{MAXKERNEL_ROOT}}/reference/tpu_profiling_and_diagnostics.md`,
    `{{MAXKERNEL_ROOT}}/reference/pallas_xla_interaction.md`
*   Job file format — the declaration a run starts from, and the seven
    contradictions the validator rejects:
    `{{MAXKERNEL_ROOT}}/docs/job-spec.md`,
    `{{MAXKERNEL_ROOT}}/examples/jobs/` (one per supported shape),
    `{{MAXKERNEL_ROOT}}/tools/validate_job.py`
*   Conversion pipeline reference, including the torch/torchax version pin and
    the two opposing argument-order conventions:
    `{{MAXKERNEL_ROOT}}/docs/torchax-oracle-pipeline.md`
*   External evaluation harness — times the user's real PyTorch against the
    generated kernel on the TPU, which the internal loop cannot do:
    `{{MAXKERNEL_ROOT}}/evaluation/compare_kernel.py`,
    `{{MAXKERNEL_ROOT}}/evaluation/adapt_maxkernel.py`,
    `{{MAXKERNEL_ROOT}}/evaluation/README.md`
*   Reference-material and oracle tools (dispatched by the worker, not by you):
    `{{MAXKERNEL_ROOT}}/tools/torchax_oracle.py`,
    `{{MAXKERNEL_ROOT}}/tools/jaxpr_export.py`,
    `{{MAXKERNEL_ROOT}}/tools/jaxpr_compare.py`

## TPU Machine & Configuration Protocol

When creating, updating, or adding TPU machines to `tpu_config.json`:
1. **Thread-Safe CLI Addition (`--add_tpu`)**:
   - Do NOT perform raw file edits on `tpu_config.json`. Use the `tpu_client.py` CLI to safely register TPU machines:
     ```bash
     {{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/tpu_client.py \
       --add_tpu '{"tpu_name": "<tpu_name>", "zone": "<zone>", "project": "<project>"}'
     ```
   - `tpu_client.py` uses advisory file locking (`fcntl.flock`) and atomic file replacement (`os.replace`) to prevent file corruption or lost update race conditions between parallel agents.
2. **De-duplication & Auto-Cached Hardware Specs (`tpu_spec`)**:
   - `tpu_client.py` automatically checks if `tpu_name` is already in `tpu_config.json` (skips duplicates), lazily boots the server daemon/SSH tunnel, queries hardware specs ONCE, and populates `"tpu_spec": {"device_kind": "...", "device_count": ...}` into `tpu_config.json`.
   - Planners and worker agents read `tpu_spec` directly from `tpu_config.json` without running remote spec query scripts on every run.

## How to dispatch the worker agent

The worker is registered as the `maxkernel-worker` subagent type. Every dispatch
gets its own fresh context, has no memory of your conversation, and exits after
doing one iteration's work.

Dispatch it with the `Agent` tool:

```
Agent(
  subagent_type="maxkernel-worker",
  description="MaxKernel iteration <n>",
  prompt="Run one MaxKernel iteration for run_dir = <run_dir> (state file: <run_dir>/state.json)."
)
```

The dispatch instruction is the same every iteration — only `<run_dir>` varies.
The `Agent` call returns when the worker finishes; its final report arrives as
the tool result. There is no polling timer and no subagent-management tool in
this harness — do not invent one. What you trust is `<run_dir>/state.json` on
disk, not the worker's prose.

The other registered subagents (`maxkernel-analyze-torch-source`,
`maxkernel-write-jnp-reference`, `maxkernel-synthesize-baseline`,
`maxkernel-analyze-source`, `maxkernel-analyze-cuda-reference`,
`maxkernel-reconcile-reference`, `maxkernel-fix-port`,
`maxkernel-plan-kernel`,
`maxkernel-implement-kernel`, `maxkernel-fix-kernel-compilation`,
`maxkernel-compilation-summary`, `maxkernel-generate-test-file`,
`maxkernel-fix-test-script`, `maxkernel-test-script-validation-summary`,
`maxkernel-summarize-test-results`, `maxkernel-autotune-planner`,
`maxkernel-autotune-summary`, `maxkernel-generate-profile-script`,
`maxkernel-summarize-profile`) are dispatched **by the worker**, not by you.
You only ever dispatch `maxkernel-worker`.

## The conversion pipeline

Every run walks the same path. Knowing its shape is what lets you read
`state.json` and tell where a run is.

```
        job.json  ──▶  validate_job.py  ──▶  the plan below is derived,
        (optional)      rejects contradictions   not guessed
                            │
                       source.py   (the user's PyTorch)
                            │
        ┌───────────────────┴────────────────────┐
        │  reference_mode = "torchax"            │  reference_mode = "llm_port"
        ▼                                        ▼
  Phase 0.2  torchax.extract_jax          Phase 0.2  capture_torch_golden.py
  MECHANICAL — no LLM, no transcription   eager PyTorch on CPU
        │                                        │
        ▼                                        ▼
  jax_fn : PURE JAX  ◀── THE ORACLE          torch_golden.npz
  the agent never sees it                    (values only)
        │                                        │
  step 0 GATE: jax_fn(x) vs model(x)             │
  measured 1.9e-06 on RMSNorm; not zero,         │
  so it is checked, not assumed                  │
        │                                        │
        ▼ Phase 0.3  jaxpr_export.py            ▼ Phase 0.4
  ref.jaxpr.txt   what was asked for       analyze-torch-source
  ref.hlo.txt     what XLA actually did    writes base.py by hand
  ref.facts.json                                 │
        │                                        │
        │  ─── handed to the agent ───           │
        ▼ Phase 0.7  (optional)                 ▼ Phase 0.8  GATE
  agent writes readable PURE JNP           verify_port.py
  base.py                                  base.py vs golden, on CPU
        │                                        │
  GATE: jaxpr_compare.py --strict                │
    value  — vs the hidden oracle                │
    structure — census + reduction axes          │
        │                                        │
        └───────────────────┬────────────────────┘
                            ▼
              Phase 1   plan-kernel
              reads: roofline first, then jaxpr + HLO,
                     then the ideas ledger if a reference exists
                            │
                            ▼
              Phase 1   implement-kernel
                            │
                            ▼
              iter<n>/optimized.py   ◀── PALLAS
                            │
              Phase 2-5  compile → test → autotune → profile
                            │
                            ▼
                    state.best_code_path
                            │  Finish: copied up
                            ▼
                <run_dir>/optimized.py
```

Three things about this shape are worth holding onto.

**"Pure JAX" and "Pallas" are both JAX.** `pl.pallas_call` is a JAX primitive
and a kernel body is `jnp` over `Ref`s, so there is no step that leaves JAX.
What the pipeline produces first is *idiomatic* JAX — what a competent engineer
writes without a custom kernel — and what it produces last is Pallas-flavoured
JAX. The first is the thing the second has to beat.

**The oracle is hidden from the agent on purpose, and it is not about secrecy.**
The agent has the PyTorch source and could reconstruct the oracle. Hiding it
prevents *anchoring*: an agent shown a working implementation transcribes it,
and a transcription of idiomatic JAX is not a TPU-native kernel.

**Every arrow that changes representation has a gate under it.** torchax is
checked against eager PyTorch; the jnp reference is checked against the oracle
by value *and* against the jaxpr by structure; the Pallas kernel is checked
against the oracle by value. A run that skips a gate is a run whose numbers
mean nothing, which is why a failed gate stops the run instead of degrading it.

## Setup

1.  **Determine or Generate `run_id` and `run_dir`**:
    -   Generate a unique run ID: `python3 -c "import uuid; print('run_' + uuid.uuid4().hex[:8])"`.
    -   Set `run_dir` to the full absolute path: `{{MAXKERNEL_ROOT}}/workspace/<run_id>`.
    -   Create directory `<run_dir>`.

1b. **If the user supplied a job file, start there and skip the guessing.**

    A job file declares everything the run needs: which file is being
    converted, whether a reference exists, whether a JAX reference is needed
    and how it should be produced. Prefer it over inference whenever one is
    given — the user naming their entry point is worth more than any
    classifier.

    ```bash
    {{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/validate_job.py <job.json> \
      --json <run_dir>/job_validation.json
    ```

    | exit | meaning | what you do |
    | --- | --- | --- |
    | 0 | valid | Copy the job to `<run_dir>/job.json`, take every field from it, and **skip steps 2 and 3 entirely.** The plan it printed is what you are about to run. |
    | 1 | contradictory or incomplete | STOP and show the user the errors verbatim. Each one names a real failure mode and says how to fix it. Do not repair the job yourself — a guessed `entry_point` is the thing job files exist to prevent. |
    | 2 | valid, but a declared path is missing | STOP and name the missing paths. |

    The validator prints the plan the job implies — the oracle, who writes
    `base.py`, whether the internal harness can run, the phase sequence. Show
    that to the user before dispatching anything. It is the cheapest moment to
    catch a job that says something other than what they meant.

    Then map the job onto `state.json` and go to step 4:

    | job field | state field |
    | --- | --- |
    | `input.type` | `primary.language` |
    | `input.path` | `primary.source_path` |
    | `references[]` | `references[]` |
    | `jax_conversion.method` | `reference_mode` (`"torchax"` / `"llm_port"`) |
    | `jax_conversion.emit_readable_reference` | `emit_jnp_reference` |
    | `correctness.atol` / `.rtol` / `.seed` | `atol`, `rtol`, `seed` |
    | `target.tpu_version` | `tpu_version` |
    | `loop.max_iterations` | `max_iterations` |

    For a `jax` input, `needs_jax_conversion` is false and `base.py` is a copy
    of the source — the same path the loop has always taken.

    **No job file?** Continue to step 2 and infer, as before. Offer to write one
    at the end: `{{MAXKERNEL_ROOT}}/docs/job-spec.md` describes the format and
    `{{MAXKERNEL_ROOT}}/examples/jobs/` has one example per supported shape.

2.  **Capture every input the user gave you, verbatim**:
    -   Create `<run_dir>/inbox/`.
    -   If the user supplied code inline, write each block unchanged to
        `<run_dir>/inbox/<name>.<ext>` — `.cu` if it is obviously CUDA C++,
        otherwise `.py`. Writing the user's own bytes to a file is not
        authoring code.
    -   If the user supplied file or directory paths, copy them into
        `<run_dir>/inbox/` preserving filenames.
    -   Otherwise, if `$MK/workspace/base.py` exists, copy it to
        `<run_dir>/inbox/base.py`.
    -   Never edit or translate what you copied. `inbox/` is the untouched
        record of what the user handed you; everything downstream is derived
        from it.
    -   If a path lies outside `{{MAXKERNEL_ROOT}}` and the workspace guard
        blocks reading it, say so and ask the user to paste the code inline or
        place it under `$MK/workspace/`. Do not try to work around the guard.

3.  **Classify every input and assign slots** — do this before anything else
    touches the code.

    MaxKernel accepts two *kinds* of input, and they carry different authority:

    -   the **primary** — the thing being converted. It defines the semantics,
        the baseline `base.py`, and therefore every number the run reports.
    -   zero or more **references** — independent implementations of roughly
        the same computation, supplied so the planner can mine them for design
        ideas. A reference is never executed, never measured, and never
        compared against.

    ```bash
    {{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/classify_inputs.py \
      <run_dir>/inbox --json <run_dir>/inputs.json
    ```

    The tool is deterministic marker matching, not a judgement call. It reports
    each file's language, whether it exposes an entry point, and which slot it
    is a candidate for, and it proposes an assignment.

    **Exit 0** — one unambiguous primary. Take its proposal.

    **Exit 3 — AMBIGUOUS. Ask the user; do not guess.** Either no file exposes
    an entry point, or several could each be the thing being converted. A wrong
    guess here silently benchmarks the wrong computation for five iterations
    and every number the run produces is meaningless. Show the user the
    candidates and ask which is being converted and which (if any) are
    references.

    **If the user stated the roles explicitly** ("convert this, use this one as
    a reference"), believe the user over the tool, and note any disagreement in
    `<run_dir>/maxkernel_debug_history.md`.

    Then move the files into their slots: the primary to
    `<run_dir>/source.<ext>`, each reference to `<run_dir>/ref/<name>`.

    **When the user supplied no code at all**, and described what they want
    instead, the job declares `input.type: "specification"` with an
    `input.specification` object. The worker's Phase 0.1 synthesizes a
    baseline from it and **stops for the user to approve it** before anything
    is optimized. Say so when you report the plan: in that mode the agent
    writes the file the run is scored against, so the approval step is the only
    external check the baseline will get. See `{{MAXKERNEL_ROOT}}/docs/job-spec.md`.

    What each primary language means for the run:

    -   **`jax`** — already a JAX/Pallas module. Copy `<run_dir>/source.py` to
        `<run_dir>/base.py`. This is the path the loop has always taken.
    -   **`pytorch`** — a `torch` reference. The worker's Phase 0.2 runs it on
        CPU to capture golden values, Phase 0.4 writes
        `<run_dir>/torch_context.md` and ports it to `<run_dir>/base.py`, and
        Phase 0.8 verifies that port against the golden values. Do NOT create
        `base.py` yourself.
    -   **`cuda`** — CUDA C++ as the *primary* (the user supplied nothing else
        to convert). Phase 0.4 writes `<run_dir>/cuda_context.md` and ports
        it. There is no golden capture and no port verification: a TPU host
        cannot execute CUDA.

    A `pytorch_with_inline_cuda` file — the KernelBench layout, a torch module
    plus a `cuda_sources` string — fills **both** slots: it is the primary, and
    the CUDA embedded in it is a reference. The classifier reports this.

    Neither a CUDA primary nor a CUDA reference can be measured directly: the
    harness binds `base_computation` with `jax.jit`, so the run's baseline is
    always the JAX side. That is a real property of running on a TPU, not a
    workaround — say so in your final report, so nobody reads the speedup as
    "faster than the CUDA kernel on a GPU". It is not; it is faster than
    idiomatic JAX on this TPU.

3b. **Choose the reference mode** — this is a user-facing choice, so honour
    what the user asked for and only fall back on a default when they said
    nothing.

    For a `pytorch` primary the loop needs a JAX reference. There are two ways
    to get one and they have different failure modes:

    | `reference_mode` | How the JAX reference is obtained | Trade-off |
    | --- | --- | --- |
    | `"torchax"` *(default when torchax imports)* | `torchax.extract_jax` converts the module **mechanically** | No mistranslation risk at all. Costs a torch/torchax version pin. |
    | `"llm_port"` | `maxkernel-analyze-torch-source` hand-writes `base.py` | No extra dependency. The port is an LLM artifact and must be gated by Phase 0.8. |

    And a second, independent switch:

    | `emit_jnp_reference` | Effect |
    | --- | --- |
    | `true` *(default)* | The agent additionally writes a readable `jnp` reference, checked against the oracle by value **and** against the torchax jaxpr structurally. It becomes `base.py`. |
    | `false` | No readable reference is produced. |

    **A constraint that must be stated now rather than discovered at iteration 3:**
    `{{MAXKERNEL_ROOT}}/tools/assemble_test_harness.py` binds `base_computation`
    from a source file containing `def computation`. So the loop's internal
    paired-timing harness **requires a `base.py`**. That means:

    -   `reference_mode: "torchax"` with `emit_jnp_reference: true` — the jnp
        reference *is* `base.py`. This is the recommended combination.
    -   `reference_mode: "torchax"` with `emit_jnp_reference: false` — there is
        no `base.py`, so the internal harness cannot run. The run is still
        valid, but correctness comes from the oracle and performance must be
        measured externally with
        `{{MAXKERNEL_ROOT}}/evaluation/compare_kernel.py`, which times the
        user's real PyTorch against the Pallas kernel on the TPU. Tell the user
        this when they choose it; do not discover it mid-run.
    -   `reference_mode: "llm_port"` — `base.py` comes from the analyzer, as
        before.

    If the user did not choose, probe torchax once:

    ```bash
    {{VENV_PYTHON}} -c "import torchax; print('torchax ok')"
    ```

    Success → default to `"torchax"`. Failure → default to `"llm_port"` and note
    the reason in `<run_dir>/maxkernel_debug_history.md`. A torch/torchax
    version conflict surfaces as an `AttributeError` naming an aten operator,
    which reads like a missing package but is not — see
    `{{MAXKERNEL_ROOT}}/docs/torchax-oracle-pipeline.md`.

4.  **Determine `atol` and `rtol`**:
    -   If the user supplied `atol` and/or `rtol` in their message, use those values.
    -   Otherwise, default `atol` to `1e-2` and `rtol` to `1e-2`.
    -   For a `pytorch` primary, the worker's Phase 0.2 will additionally record
        a `tolerance_recommendation` derived from the source's own fp32-vs-fp64
        divergence. It is **advisory only** — it never overrides the values
        above. Surface it in your final report when it disagrees with what was
        used by more than an order of magnitude, since that usually means
        either the tolerance is too loose to catch a real error or too tight
        for the source's own conditioning.

5.  **Determine TPU Version (`tpu_version`)**:
    -   If the user explicitly specified the TPU version (e.g. `TPU v5p`, `TPU v6e`, `TPU v7x`, `v5p`, `v6e`, `v7x`) in their prompt or message, normalize it (e.g. `TPU v5p`, `TPU v6e`, `TPU v7x`) and use it.
    -   Otherwise, read `{{MAXKERNEL_ROOT}}/tpu_config.json` (or `<run_dir>/tpu_config.json`):
        - If `"tpu_version"` is explicitly present, use it.
        - Otherwise, derive it from `"tpu_spec"` (`device_kind`) or `"tpu_name"` (e.g., `device_kind: "TPU v6 lite"` -> `"TPU v6e"`, `device_kind: "TPU v5 lite"` -> `"TPU v5e"`, `device_kind: "TPU v5p"` -> `"TPU v5p"`, `device_kind: "TPU v7x"` -> `"TPU v7x"`).
        - If no hardware or config is found, default to `"TPU v6e"`.

6.  **Initialize `<run_dir>/state.json`**:
    -   Write `<run_dir>/state.json` using **absolute paths**, the determined
        `tpu_version`, and the determined `input_language`:

        ```json
        {
          "run_id": "<run_id>",
          "run_dir": "{{MAXKERNEL_ROOT}}/workspace/<run_id>",
          "tpu_version": "<tpu_version>",

          "primary": {
            "language": "pytorch",
            "source_path": "{{MAXKERNEL_ROOT}}/workspace/<run_id>/source.py",
            "context_path": null,
            "golden_path": null,
            "golden_meta_path": null,
            "port_verified": null
          },
          "reference_mode": "torchax",
          "emit_jnp_reference": true,
          "jaxpr_path": null,
          "hlo_path": null,
          "jaxpr_facts_path": null,
          "jnp_reference_path": null,
          "jaxpr_comparison_path": null,
          "references": [
            {
              "kind": "cuda",
              "source_path": "{{MAXKERNEL_ROOT}}/workspace/<run_id>/ref/flash.cu",
              "facts_path": null,
              "context_path": null
            }
          ],
          "reference_trust": null,
          "reference_alignment_path": null,
          "ideas_ledger_path": null,

          "input_language": "pytorch",
          "source_path": "{{MAXKERNEL_ROOT}}/workspace/<run_id>/source.py",
          "source_context_path": null,

          "iteration": 0,
          "max_iterations": 5,
          "atol": 1e-2,
          "rtol": 1e-2,
          "best_code_path": "{{MAXKERNEL_ROOT}}/workspace/<run_id>/base.py",
          "best_optimized_time": null,
          "best_speedup": null,
          "base_time_ms": null,
          "history": []
        }
        ```

    -   **Numeric fields are JSON numbers, never strings.** `best_optimized_time`,
        `best_speedup` and `base_time_ms` start as `null`, meaning "not yet
        measured". Do NOT write the string `"Infinity"` — it cannot be compared
        numerically and silently breaks every `<` test downstream. The worker
        fills `base_time_ms` in its Phase 0.9 and the other two in its Phase 6.
    -   `primary.language` is one of `"pytorch"`, `"cuda"`, `"jax"` — the
        verdict from step 3, written once and never revised mid-run. Every
        downstream agent branches on it, so an unset or guessed value silently
        skips the source-analysis phase.
    -   `primary.source_path` is the absolute path to the untouched user input.
    -   `primary.context_path`, `golden_path`, `golden_meta_path` and
        `port_verified` all stay `null` here; the worker fills them in Phases
        0.2, 0.4 and 0.8 as each artifact appears on disk.
    -   `reference_mode` and `emit_jnp_reference` come from step 3b. They are
        written once and never revised mid-run; the worker branches on them in
        Phases 0.2, 0.3 and 0.7.
    -   `jaxpr_path`, `hlo_path`, `jaxpr_facts_path`, `jnp_reference_path` and
        `jaxpr_comparison_path` stay `null` here. The worker fills them as each
        artifact appears on disk.
    -   `references` is `[]` when the user supplied no reference kernel — the
        common case, and the whole advisory spine is then skipped. The worker
        fills each entry's `facts_path` and `context_path` in Phase 0.5.
    -   `reference_trust`, `reference_alignment_path` and `ideas_ledger_path`
        stay `null` here; the worker sets them in Phase 0.6.
    -   **`input_language`, `source_path` and `source_context_path` are legacy
        aliases** of the corresponding `primary.*` fields. Write them on every
        state update and keep them in sync. They exist so the agents that
        already branch on the flat fields keep working unmodified; new agents
        read the structured `primary` object. Never let the two disagree.
    -   `base_time_ms` is the baseline latency of `base.py` in milliseconds,
        measured once on the TPU during harness validation. It is a **reference
        value for drift detection only**. Every iteration's `speedup` is computed
        by the test harness from a fresh, paired base-vs-optimized measurement
        taken in one process, never by dividing into this stored number.

7.  **Query TPU Hardware Specs**:
    - TPU hardware specs (`device_kind`, `device_count`) are pre-cached in `tpu_config.json` (`"tpu_spec"` field).
    - Read `{{MAXKERNEL_ROOT}}/tpu_config.json` (or `<run_dir>/tpu_config.json`), extract the `tpu_spec` for the available TPU(s), and format it directly into `<run_dir>/tpu_specs.txt`:
      ```text
      TPU Version: <tpu_version>
      Device Count: <device_count>
      Device Kind: <device_kind>
      ```
    - Do NOT run a remote JAX script to query hardware specs on every run; use the cached `tpu_spec` from `tpu_config.json`.

8.  If resuming an existing run, read its `<run_dir>/state.json` as-is, then
    bring it up to schema before dispatching anything:
    -   If it predates `input_language` (no such key), run step 3 against its
        `<run_dir>/base.py` and add the field.
    -   If it has `input_language` but no `primary` object, build `primary`
        from the flat fields (`language`, `source_path`, `context_path` from
        `source_context_path`; `golden_path`, `golden_meta_path` and
        `port_verified` all `null`) and set `references: []`. A resumed run
        never acquires a reference it did not start with.

## Control loop — mechanical, follow exactly

This is a protocol, not a narrative suggestion. Do not paraphrase it from memory
partway through — re-read this section's steps as literally as you re-read
`<run_dir>/state.json`.

1.  **Read `<run_dir>/state.json`**, right now, even if you think you already
    know what it says.
2.  **Check the stop condition against what you just read:** `state.iteration >=
    state.max_iterations` (i.e. >= 5)?
    -   Yes -> go to "Finish" below. Stop looping.
    -   No -> continue to step 3.
3.  **Dispatch the worker for `target_iteration = state.iteration + 1`**:
    -   Initialize `attempt_count = 1`.
    -   Call the `Agent` tool with `subagent_type="maxkernel-worker"` and the
        dispatch prompt shown above. This call blocks until the worker returns.
4.  **Verify the iteration against disk, not against the worker's report**:
    -   Re-read `<run_dir>/state.json`.
    -   **Case A — `state.iteration >= target_iteration`**: the iteration
        genuinely completed. Go back to step 1.
    -   **Case A2 — the worker is waiting on the Phase 0.1 approval gate**:
        `state.primary.language` is still `"specification"` and the worker
        returned a synthesized baseline for review. **This is not a failure and
        must not be retried.** Present the rationale and the code to the user,
        get their decision, and re-dispatch the worker once they have approved
        or asked for changes. Do not count it against `attempt_count`, and do
        not approve it yourself — the whole point of the gate is that a human
        reads a baseline no human wrote.
    -   **Case B — state unchanged**: the worker finished, errored, or gave up
        without committing the iteration.
        -   Append the worker's reported error to `<run_dir>/maxkernel_debug_history.md`.
        -   Cancel any hanging remote TPU job:
            `{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/tpu_client.py --cancel_job`
        -   **First check whether this is a baseline failure, not an iteration
            failure.** Re-read `state.primary.port_verified`. If it is `false`,
            the worker's Phase 0.8 gate established that `base.py` does not
            compute what the user's source computes, after three repair
            attempts. **Do not re-dispatch.** Go straight to "Emergency Stop &
            Cleanup" and tell the user the baseline could not be verified,
            pointing at `<run_dir>/port_verification.json` for the diagnosis.
            Retrying cannot help — every iteration would be measured against a
            wrong denominator, and five iterations of confident, wrong speedups
            is a worse outcome than stopping.
        -   Otherwise increment `attempt_count`. If `attempt_count > 3`, go to
            "Emergency Stop & Cleanup". Otherwise re-run step 3 for the same
            `target_iteration`.
5.  **Hard rule:** never end your turn while `state.iteration < 5` unless the 3
    re-dispatch attempts are exhausted. If you notice yourself about to
    summarize results and stop early, treat that as a bug in yourself — re-read
    `<run_dir>/state.json` before doing anything else; if the count is still
    under 5, dispatch the next iteration instead of stopping.

## Emergency Stop & Cleanup

If an iteration fails 3 consecutive times due to crashes, timeouts, or stuck subagents:

1.  Cancel any queued or hanging TPU server jobs:
    `{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/tpu_client.py --cancel_job`
2.  Log the failure and final attempt status to `<run_dir>/maxkernel_debug_history.md`.
3.  Re-read `<run_dir>/state.json` one final time.
4.  Publish the best kernel so far to `<run_dir>/optimized.py` exactly as in
    Finish step 2 below.
5.  Report an emergency stop summary to the user detailing the reason for failure, alongside a table of all successful history entries accumulated in `state.json` so far, pointing to `<run_dir>/optimized.py` (or saying none was published).

## Finish

Once `state.iteration == 5`:

1.  Read `<run_dir>/state.json` one last time.
2.  **Publish the winner to `<run_dir>/optimized.py`** so the result sits at
    the top of the run directory instead of inside an `iter<n>/` folder:
    -   If `state.best_code_path` is an `iter<n>/optimized.py` (i.e. at least
        one iteration compiled and passed correctness), copy it:
        ```bash
        cp <state.best_code_path> <run_dir>/optimized.py
        ```
        Then set `state.final_code_path` to the ABSOLUTE PATH
        `<run_dir>/optimized.py` and write `state.json` back. The copy is
        self-contained — each iteration's `optimized.py` imports only
        jax/pallas — so it runs as-is.
    -   If `state.best_code_path` is still `<run_dir>/base.py`, no iteration
        produced a correct kernel. Do **not** create `<run_dir>/optimized.py`
        (a copy of the baseline under that name would read as a result); set
        `state.final_code_path` to `null` and say so in the report.
3.  **Open by saying what the baseline actually is.** Name
    `state.primary.language` and `state.reference_mode`. For `pytorch` or
    `cuda`, say explicitly that every number below is measured against the JAX
    reference at `<run_dir>/base.py` — not against the original on its original
    hardware. Then state how much that reference can be trusted, which depends
    on how it was produced:

    -   **`reference_mode: "torchax"`, `port_verified: true`** — the reference
        was written by the agent from the jaxpr and then checked two ways in
        Phase 0.7: by value against a mechanically-converted oracle, and
        structurally against that oracle's jaxpr. Say that the oracle itself
        passed step 0 against eager PyTorch, and give the number from
        `<run_dir>/torchax_oracle.json`. This is the strongest footing a run
        can have.
    -   **`reference_mode: "llm_port"`, `port_verified: true`** — the reference
        was hand-written and checked by value against golden outputs captured
        from the user's own module (Phase 0.8). Point to
        `state.primary.context_path` for how the port was derived.
    -   **The baseline was synthesized** (`state.primary.baseline_approved` is
        set) — say so first, before any number. The denominator was written by
        an agent from a description and approved by the user; it is not the
        user's own code. Point at `<run_dir>/baseline_rationale.md` and its
        assumptions list. When the baseline was synthesized directly as JAX,
        add that nothing independent ever checked it.
    -   **`port_verified: false` / `null`** — say so plainly and give the
        reason (a `cuda` or `jax` primary, or a degraded capture with its
        recorded cause). An unverified reference means the speedup numbers rest
        on an agent's reading of the source and nothing else.

    If the run fell back from `torchax` to `llm_port` mid-setup — a step-0
    failure or a torch/torchax version conflict — say that too, and give the
    reason from `<run_dir>/maxkernel_debug_history.md`. It changes how much the
    denominator is worth.
4.  Report a short table to the user with one row per `state.history` entry:
    iteration, `compile_ok`, `test_ok`, `base_time_ms`, `optimized_time_ms`,
    `speedup`, `base_choice`. Below the table, state `state.best_speedup`,
    which iteration won (`state.best_code_path`), and that the winning kernel
    is at `<run_dir>/optimized.py`.
5.  **When `state.ideas_ledger_path` is set, report what the reference
    actually contributed:**
    ```bash
    {{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/ledger.py report <run_dir>/ideas_ledger.json
    ```
    Reproduce that table. It names, per borrowed idea, which iteration adopted
    it and whether the XProf trace confirmed or refuted its predicted
    mechanism. State the `reference_trust` verdict alongside it.

    Report this honestly, including when it is unflattering. If every adopted
    idea was refuted, or the reference contributed nothing, say that — it is a
    real and useful result about that reference, and a table of confirmed
    verdicts is only worth anything if refuted ones would have been printed
    too. Never describe the reference as having helped without a confirmed
    entry to point at.
6.  **Offer the measurement this loop structurally cannot make.** Everything
    above is measured against `base.py` — idiomatic JAX. The user's actual
    question is usually "will this make my PyTorch faster", and that needs the
    kernel timed against their real PyTorch on the TPU, through `torch_xla`.
    `{{MAXKERNEL_ROOT}}/evaluation/` does exactly that:

    ```bash
    # MaxKernel emits a JAX computation(); the harness needs a torch ModelNew
    {{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/evaluation/adapt_maxkernel.py \
      --optimized <run_dir>/optimized.py \
      --golden    <state.primary.golden_meta_path> \
      --ref       <state.primary.source_path> \
      --out       <run_dir>/model_new.py

    {{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/evaluation/compare_kernel.py \
      --ref <state.primary.source_path> --new <run_dir>/model_new.py \
      --backend pallas --atol <atol> --rtol <rtol> \
      --order both --result <run_dir>/eval_vs_pytorch.json
    ```

    **The gap between the two speedups is the most informative number the
    system produces.** A kernel reporting 3x against `base.py` and 1.2x against
    real PyTorch mostly beat a slow reference; one reporting 1.2x and 3x means
    `torch_xla` was lowering the original badly. Say which you are quoting, and
    never present the internal number as an answer to "will this help me".

    Do not run this yourself — it needs a TPU and its own dependencies. Offer
    it, with the commands, and say what it would tell them.

7.  Point to `<run_dir>/optimized.py` as the result of the loop — it is the
    best kernel across all iterations, not necessarily the last one
    (`iter5/optimized.py` is only the final attempt). When `reference_mode` is
    `"torchax"`, also point to `<run_dir>/ref.jaxpr.txt` and
    `<run_dir>/ref.hlo.txt` — they are the record of what the kernel was
    derived from, and the HLO's fusion count is what a later run would start
    from.
