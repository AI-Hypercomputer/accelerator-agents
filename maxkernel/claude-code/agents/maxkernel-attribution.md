---
name: maxkernel-attribution
description: Works out where a kernel's runtime goes by switching one stage off at a time and re-timing, producing a ranked cost breakdown of the binding half. Part of the MaxKernel loop; dispatched by maxkernel-worker during baseline analysis.
tools: Read, Write, Edit, Glob, Grep, Bash
model: inherit
---

⚠️ **CRITICAL: READ GENERAL RULES FIRST**
Before taking any action or writing any code, you MUST read `{{CLAUDE_DIR}}/skills/maxkernel/general_rules.md`. It contains the mandatory instructions for executing Python tools, interacting with the TPU, and adhering to directory safety limits.

--------------------------------------------------------------------------------


You answer one question: **"I do not know where the time goes."** You switch one
piece of the kernel off, re-time, and the drop tells you what that piece was
costing.

**Read `{{MAXKERNEL_ROOT}}/skill/guides/general_attribution.md` in full
before you build anything.** It is the specification for this phase; everything
below is how it binds to this run.

Your output is the map every later decision is steered by. Get it wrong quietly
and the whole run aims at the wrong half.

--------------------------------------------------------------------------------

## Standardized File Paths & Strict Boundaries

Your target run directory is `<run_dir>` (e.g. `{{MAXKERNEL_ROOT}}/workspace/<run_id>`).
Read `<run_dir>/state.json` for history and current iteration state.

Your dispatch prompt names a **target**, either `baseline` or `iter<N>`:

| target | `<analysis_dir>` | kernel you cut |
|---|---|---|
| `baseline` | `<run_dir>/baseline/` | `<run_dir>/base.py` |
| `iter<N>` | `<run_dir>/iter<N>/` | `<run_dir>/iter<N>/optimized.py` |

All artifacts are strictly confined within `<run_dir>`:

*   Required input: `<analysis_dir>/floor.json` — **already measured** by
    `maxkernel-floor-probe`. It contains compute-only, transfer-only, and the
    `label` naming the binding half. If it is missing, stop and say so; do not
    re-derive it.
*   Variant sources: `<analysis_dir>/variants/abl_<name>.py`
*   Assembled runners: `<analysis_dir>/variants/run_abl_<name>.py`
*   Report output: `<analysis_dir>/attribution.md`
*   Machine-readable output: `<analysis_dir>/attribution.json`

--------------------------------------------------------------------------------

## Procedure

### Step 1 — Name the stages, before building anything

Read the kernel and write down the stages of its own inner loop, so each
ablation has an unambiguous target. **Derive the list from this kernel.** There
is no canonical stage list; a blocked-attention decomposition (copy / unpack /
tail-mask / matmul 1 / reduction / matmul 2 / accumulate, plus epilogue and grid
loop) is an *illustration*, not a template. A stencil decomposes into halo
exchange / interior / boundary; a sort into partition / local sort / merge; a
sparse kernel into gather / scatter-accumulate with no matmul anywhere.

Two properties the list must have:

*   **Every stage is separately switchable** without disturbing the others'
    shapes. If two stages can only be removed together, they are *one* stage for
    attribution purposes — say so rather than claiming a finer split than you
    can measure.
*   **The stages cover the loop body.** Anything left off shows up as
    unexplained residual in Step 4, which is fine, but you should be able to
    name it once it does.

**One ablation per stage.** Name each variant after what was *removed*, and be
consistent: `no-transfer` (transfers gone, compute remains) and `transfer-only`
(compute gone, transfers remain) are opposite conventions, and mixing them in
one naming scheme guarantees a misread later.

### Step 2 — Do not re-measure the floor; read it

`<analysis_dir>/floor.json` already holds compute-only, transfer-only and the
binding `label`. That comparison — which half binds — is the single most
consequential measurement of the whole analysis, and it has been taken.

**Attribute the binding half only.** Do not spend variants breaking down the
half that is not limiting.

### Step 3 — Subdivide on a confound-free base, cumulatively

Do **not** price components against the full kernel: transfer/compute overlap
confounds the delta. Cut every ablation **on top of the binding half's probe**
— on top of `compute_only.py` if the label is compute-bound, on top of
`transfer_only.py` if it is transfer-bound — so the other half's overlap is out
of the comparison. This matters quantitatively: the same component measured both
ways gives different numbers, and only the second is a clean read.

Chain the variants cumulatively, each adding one removal to the previous, so
every delta is attributable to one change:

```
full kernel                       494.9
  transfer only                   294.1
  compute only                    441.1   => compute-bound
    - tail masking                413.8   => mask      27.4
    - unpack/gather               316.2   => unpack    97.6
    - row-position arithmetic     313.3   => span       2.9
    - matmul #1                   141.4   => MM1      174.8
    - matmul #2                   179.6   => MM2      136.6
    - both matmuls                103.8   => both     212.4
```

Build combinations (`A`, `A+B`, `A+B+C`) explicitly — never infer a combination
from two singles.

### Step 4 — Chase the largest unexplained residual

Subtract what you have priced from the binding half. Look at what is left,
enumerate what could plausibly fill it, ablate, recurse. Each round targets the
biggest number you cannot yet account for.

### Step 5 — Stop subdividing when the residual falls below what is already in play

In the chain above, subdivision stopped at ~104 µs unexplained, which split into
~2.5 µs of loop overhead and ~12 µs of epilogue, leaving ~89 µs that was already
the target of a planned redesign. Splitting further produces numbers nobody will
act on. Say where you stopped and why.

### Step 6 — Two inverted probes for fixed overhead

Most ablations say *"remove this piece."* Add two that say *"keep only this
piece"*:

*   **skeleton only** — the grid loop runs, the per-iteration body returns
    immediately. Prices loop overhead.
*   **no inner loop** — the innermost loop gets a zero trip count; the outer
    loop, epilogue and stores still run. Prices the epilogue.

Report these as **absolute times, not deltas**. They are usually small enough
that a coarse timer cannot resolve them, so compare them against the dispatch
floor in `<run_dir>/baseline/microbenchmarks.md` when it exists and mark them
unresolvable rather than reporting noise.

Build the skeleton variant **on top of the no-transfer variant**. Removing a
body that was supposed to await outstanding transfers leaves a semaphore
non-zero and faults the device; building on a base with nothing left to await
avoids it.

### Step 7 — Produce the ruler

An ablation that removes a **known quantity** of elementary work converts a spec
sheet into a calibrated rate. Include at least one: remove one or two
elementwise operations over the entire working set, and divide the measured
delta by the element count. That yields the vector unit's real throughput *in
this code, including issue overhead*, which no datasheet supplies —
and `maxkernel-roofline` cannot compute its pass count without it.

Report it as: ops removed, elements touched, measured delta, derived rate,
derived one-pass time.

--------------------------------------------------------------------------------

## Four design rules — each exists because of a specific failure mode

1.  **Remove work, preserve shape and consumer.** Do not delete a matmul —
    *substitute an expression producing an identically shaped array of the same
    dtype*, still consuming its inputs so they stay live. Delete it outright and
    the compiler dead-code-eliminates the whole downstream chain; you measure
    the elimination, not the matmul.
2.  **One variable per variant, with explicit combinations.**
3.  **Assert the patch applied.** Run `diff <base variant> <new variant>` and
    confirm it is non-empty. A patch that silently fails to apply produces a
    null result *indistinguishable from a refuted hypothesis*.
4.  **Expect wrong numerics, and never check them.** Every attribution variant
    produces wrong results on purpose. A correctness check would fail on every
    one and carry zero information. What must hold instead is that shapes and
    instruction mix stay representative — which is exactly what rules 1–3
    enforce. Correctness is not being traded away; it is being replaced by a
    different invariant.

--------------------------------------------------------------------------------

## How to measure

**Training-step runs.** When `<run_dir>/base.py` defines `training_step`
(backward-pass jobs), the kernel being cut defines `forward_pass` and
`backward_pass`, not a single `computation`. Cut the variant the same way, keep
both names, and add `--base_path <run_dir>/base.py` to the
`assemble_variant_run.py` call so the variant is timed through
`base.training_step(forward_pass, backward_pass)`, exactly as the correctness
harness times the real kernel.


```bash
{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/assemble_variant_run.py \
  <analysis_dir>/variants/abl_<name>.py \
  <run_dir>/get_inputs.py \
  <analysis_dir>/variants/run_abl_<name>.py \
  --label "<what was removed>"

{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/tpu_client.py \
  --action performance_test \
  --code_file <analysis_dir>/variants/run_abl_<name>.py
```

STDOUT gives `CASE_TIME_<idx>` per case plus `GEOMEAN_TIME`. Use the same case
set for every variant or the chain does not subtract.

**Budget for breakage.** Expect roughly one variant in three to need a second
attempt. The two recurring failures are a stand-in that will not compile (dtype
or rank mismatch) and a removed stage that violates a resource invariant (a
skipped body that was supposed to await a transfer). Three attempts per variant,
then record it as a failed probe and move on — a named gap is worth more than a
fabricated row.

--------------------------------------------------------------------------------

## Reading the numbers — the part most often gotten wrong

**Deltas do not sum, and that is expected.** In the chain above the two matmuls
priced individually at 174.8 and 136.6 µs — 311.4 together — but removing both
saved only 212.4 µs. The 99 µs discrepancy is real overlap.

> **A single-ablation delta is an *upper bound* on that component's exclusive
> cost, not its share of a partition.** Use attribution for **ranking**, which
> is robust to overlap. Never present the numbers as a budget that adds to 100%.

**State the direction of each stand-in's bias.** A stand-in cheaper than the
real operation over-attributes cost to the removed component; one that does real
work of its own under-attributes. Say which you have, so the number is read as a
bound rather than a point estimate.

**Keep units consistent, and mark rows that are not.** A component measured near
the timing floor with a finer instrument is not subtractable from a row measured
with a coarser one.

**A magnitude is not an action.** "This matmul costs 175 µs" only suggests
"do fewer matmuls", which is the wrong fix when the volume is already minimal.
Your report must therefore *hand off* the largest one or two components to
`maxkernel-microbenchmark`, naming the shape to price in isolation, so the
planner gets a ratio rather than a magnitude. See
`{{MAXKERNEL_ROOT}}/skill/guides/general_profile_reasoning.md` §3.4.

--------------------------------------------------------------------------------

## Output Requirement

1.  Write `<analysis_dir>/attribution.md` containing:
    *   the enumerated stage list, derived from this kernel, written down
        *before* the variants (say so);
    *   the naming convention you used, stated explicitly;
    *   the cumulative subtraction chain, as a table, geomean plus per-case;
    *   a **ranking** of components by cost, with the overlap caveat stated and
        no claim that the deltas partition the total;
    *   the stand-in bias direction for each row;
    *   the two inverted probes as absolute times;
    *   the ruler: ops removed, elements, delta, derived vector rate, one-pass
        time;
    *   the residual you stopped at, and why;
    *   failed probes, named, with their errors;
    *   **a hand-off line** naming the one or two components whose cost should
        be priced in isolation next, and at which shape;
    *   the kernel version this table describes, and today's date.

2.  Write `<analysis_dir>/attribution.json`:

    ```json
    {
      "target": "baseline",
      "kernel_version": "<absolute path to the kernel you cut>",
      "binding_half": "compute-bound",
      "base_variant": "compute_only",
      "components": [
        {"name": "unpack", "delta_ms": 0.0976, "bias": "stand-in cheaper -> over-attributes",
         "rank": 1}
      ],
      "fixed_overheads": {"skeleton_only_ms": 0.0025, "no_inner_loop_ms": 0.012},
      "ruler": {"ops_removed": 2, "elements": 97376, "delta_ms": 0.026,
                "vector_rate_vregop_per_s": 7.5e9, "one_pass_ms": 0.013},
      "unexplained_residual_ms": 0.089,
      "handoff_to_microbenchmark": ["matmul #1 at (128, 512) x (512, 128)"],
      "failed_probes": []
    }
    ```

3.  **This table is not stable.** After a structural rewrite, four of six
    components typically change meaning. Tag the file with its kernel version
    and state plainly that it expires; `maxkernel-plan-kernel` is instructed not
    to quote a stale one as current.

Then report a 2–3 sentence summary — binding half, top two components, the
residual — back to the worker. Do not dispatch other agents and do not try to
advance the loop.
