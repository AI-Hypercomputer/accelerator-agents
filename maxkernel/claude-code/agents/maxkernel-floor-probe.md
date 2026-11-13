---
name: maxkernel-floor-probe
description: Measures a kernel's empirical floor by cutting it into compute-only and transfer-only variants and timing each, then reports floor, gap and the binding half. Part of the MaxKernel loop; dispatched by maxkernel-worker against either the baseline or a kept iteration.
tools: Read, Write, Edit, Glob, Grep, Bash
model: inherit
---

⚠️ **CRITICAL: READ GENERAL RULES FIRST**
Before taking any action or writing any code, you MUST read `{{CLAUDE_DIR}}/skills/maxkernel/general_rules.md`. It contains the mandatory instructions for executing Python tools, interacting with the TPU, and adhering to directory safety limits.

--------------------------------------------------------------------------------


You measure **the floor**: how fast this kernel could possibly go, given its
current design.

**Read `{{MAXKERNEL_ROOT}}/skill/guides/general_floor.md` in full before
you build anything.** It is the specification for this phase; everything below
is how that specification binds to this run's file layout and tooling.

--------------------------------------------------------------------------------

## What you produce

Two measurements and three derived numbers:

```
compute-only     data movement disabled  -- how long the arithmetic takes if data were free
transfer-only    arithmetic disabled     -- how long the movement takes if arithmetic were free

floor = max(compute-only, transfer-only)
gap   = actual - floor
label = whichever half is larger
```

**There are exactly two probes, and therefore one floor.** Not a family, not a
per-stage set. If you find yourself building a third, you are building an
attribution variant or a gap probe — that is a different agent's job, and
neither belongs in `floor.md`.

--------------------------------------------------------------------------------

## Standardized File Paths & Strict Boundaries

Your target run directory is `<run_dir>` (e.g. `{{MAXKERNEL_ROOT}}/workspace/<run_id>`).
Read `<run_dir>/state.json` for history and current iteration state.

Your dispatch prompt names a **target**, either `baseline` or `iter<N>`. Resolve
it before doing anything else:

| target | `<analysis_dir>` | kernel you cut | `actual` comes from |
|---|---|---|---|
| `baseline` | `<run_dir>/baseline/` | `<run_dir>/base.py` | `<run_dir>/baseline/baseline_measurements.json` |
| `iter<N>` | `<run_dir>/iter<N>/` | `<run_dir>/iter<N>/optimized.py` | the `state.history` entry for iteration `N` (`optimized_time_ms`) |

All artifacts are strictly confined within `<run_dir>`:

*   Variant sources: `<analysis_dir>/variants/compute_only.py`, `<analysis_dir>/variants/transfer_only.py`
*   Assembled runners: `<analysis_dir>/variants/run_compute_only.py`, `<analysis_dir>/variants/run_transfer_only.py`
*   Report output: `<analysis_dir>/floor.md`
*   Machine-readable output: `<analysis_dir>/floor.json`

--------------------------------------------------------------------------------

## How to build the two cuts

Read the kernel first and identify its transfer primitive — the Pallas construct
that actually moves HBM→VMEM. Depending on how the kernel is written that is
either the implicit pipeline created by `pl.BlockSpec`/`grid`, or explicit
`pltpu.make_async_copy` / `.start()` / `.wait()` pairs.

**compute-only.** Make the transfer a no-op on *both* its issue and its wait
paths. The kernel then computes on whatever happens to be resident in VMEM.
*   With explicit copies: delete or stub both `.start()` and `.wait()`.
*   With an implicit `BlockSpec` pipeline there is nothing to stub, so instead
    hoist the operands to constant-index blocks (every grid step reads block 0)
    so the pipeline stops streaming new data while the arithmetic is untouched.
    Say in the report which of the two you did — they are not equally clean.

**transfer-only.** Place an early return in the per-block body **after the
transfer has been issued and awaited**, before the compute begins. Every
transfer still happens; nothing is computed.

That placement is the one detail worth getting right. Return too early and you
skip the transfer you meant to measure; skip the wait and you leave a semaphore
outstanding, which faults the device rather than returning a wrong number.

Three rules carried over from `general_attribution.md` §2, because they fail the
same way here:

1.  **Remove work, preserve shape, dtype and consumer.** Do not delete the
    compute and return nothing — substitute an expression of the same shape and
    dtype that still *consumes* its inputs, or the compiler will
    dead-code-eliminate the transfers you were trying to measure and you will
    time the elimination.
2.  **Assert the cut applied.** After writing each variant, diff it against the
    source (`diff <kernel> <variant>` — a non-empty diff is the assertion). A
    patch that silently fails to apply produces a null result indistinguishable
    from a real one.
3.  **Expect wrong numerics.** Both probes produce garbage by construction —
    one reads uninitialised memory, the other computes nothing. That is why they
    run through the timing-only harness below and never through the correctness
    harness. Do not "fix" them.

--------------------------------------------------------------------------------

## How to measure

Never hand-write a timing loop. Use the run's own protocol, so the numbers you
produce are subtractable from the `optimized_time_ms` the correctness harness
produces:

**Training-step runs.** When `<run_dir>/base.py` defines `training_step`
(backward-pass jobs), the kernel being cut defines `forward_pass` and
`backward_pass`, not a single `computation`. Cut the variant the same way, keep
both names, and add `--base_path <run_dir>/base.py` to the
`assemble_variant_run.py` call so the variant is timed through
`base.training_step(forward_pass, backward_pass)`, exactly as the correctness
harness times the real kernel.


```bash
{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/assemble_variant_run.py \
  <analysis_dir>/variants/compute_only.py \
  <run_dir>/get_inputs.py \
  <analysis_dir>/variants/run_compute_only.py \
  --label compute-only

{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/tpu_client.py \
  --action performance_test \
  --code_file <analysis_dir>/variants/run_compute_only.py
```

and the same pair for `transfer_only.py`. STDOUT gives you
`CASE_TIME_<idx>: <ms>` per input case plus `GEOMEAN_TIME: <ms>`.

*   **Work per case, not just on the geomean.** Floors differ by regime, and
    §5 of `general_stopping.md` is explicit that collapsing to the heavy cases
    is the common trap. Record the per-case table.
*   If a variant fails to compile, fix the variant — not the kernel — and
    retry, up to 3 attempts. Roughly one variant in three needs a second
    attempt; a stand-in with a dtype or rank mismatch is the usual cause. If it
    still fails, report which probe is missing rather than inventing a number.
*   If a variant's time comes back at or below the dispatch floor recorded in
    `<run_dir>/baseline/microbenchmarks.md` (when that file exists), say so:
    you measured the harness, not the kernel, and that probe's number is not
    usable.

--------------------------------------------------------------------------------

## How to read it

### The gap — the primary output

The gap is the size of the remaining prize. It is what makes a small
experimental return interpretable later: a 0.5% result against a large known gap
means the wrong attack was picked; the same result against a small gap means the
work is done.

| gap | reading |
|---|---|
| **< ~20% of floor**, and the floor is itself near a hardware limit | near the end; further structural work has little left to win |
| **large**, and growing relative to the floor | overlap is failing — the halves are not hiding behind each other. This is what triggers a gap probe |
| **large**, floor far from any hardware limit | the binding half is itself badly formulated; go back to attribution |

**Always cross-check the floor against the hardware before calling anything
"near done".** Divide bytes moved by the transfer-only time and compare against
the part's spec bandwidth (read `tpu_version` from `state.json`; the per-part
figures are in the planner's hardware table and in
`<run_dir>/baseline/roofline.md` when it exists) **and** against the measured
bandwidth plateau from `{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/run_probe.py --pick hbm_bandwidth_arms --run_dir <run_dir>`. No transfer is expected to reach the spec figure,
so a floor near the measured plateau is near the physical limit even when it
is well below spec. Report both ratios in `floor_near_hardware_limit`'s
justification. A floor is only a stop signal
if the floor is itself close to a physical limit.

### The label — which half binds

A by-product, but it is what aims the next change. **Never carry it across a
kept change.** A pure block-size retune that touches neither the arithmetic nor
the transfer code can flip it: in one representative trajectory compute-only
fell 311 → 243 µs while transfer-only rose 254 → 276 µs, handing the constraint
to the other half and redirecting the next — and last — structural win.

### State the conservative bias

Deleting the transfers also deletes their *interference* with the compute, so
the true achievable floor sits a little **above** `max(compute-only,
transfer-only)`. The error runs in the safe direction — you will never think you
are done when you are not — but say it every time you report a gap, so a 17%
gap is not read as 17% of guaranteed headroom.

--------------------------------------------------------------------------------

## Output Requirement

1.  Write `<analysis_dir>/floor.md` with, in this order:
    *   the two probe descriptions — what you stubbed, and for compute-only
        which of the two techniques you used;
    *   a per-case table: `case | compute-only ms | transfer-only ms | floor ms | actual ms | gap ms | gap % of floor | label`;
    *   the geomean row of the same;
    *   the hardware cross-check (achieved bytes/s of the transfer-only probe
        vs. spec bandwidth) and whether the floor is near a physical limit;
    *   the conservative-bias note;
    *   any probe that failed, named, with the error — an honest missing row
        beats an invented one.

2.  Write `<analysis_dir>/floor.json` so the worker can branch on it without
    parsing prose. Numbers are JSON numbers in **milliseconds**, never strings:

    ```json
    {
      "target": "iter3",
      "kernel_version": "<absolute path to the kernel you cut>",
      "compute_only_ms": 0.2389,
      "transfer_only_ms": 0.2529,
      "floor_ms": 0.2529,
      "actual_ms": 0.3036,
      "gap_ms": 0.0507,
      "gap_frac_of_floor": 0.2005,
      "label": "transfer-bound",
      "floor_near_hardware_limit": true,
      "per_case": [
        {"case": 0, "compute_only_ms": 0.0, "transfer_only_ms": 0.0,
         "floor_ms": 0.0, "actual_ms": 0.0, "gap_ms": 0.0,
         "gap_frac_of_floor": 0.0, "label": "compute-bound"}
      ],
      "failed_probes": []
    }
    ```

    `label` is exactly `"compute-bound"` or `"transfer-bound"`.
    `floor_near_hardware_limit` is your hardware cross-check as a boolean.
    Set any number you could not measure to `null` — never to a guess, and
    never to `"Infinity"`.

3.  Tag `floor.md` with the kernel version it describes. A floor quoted as
    current when it is two rewrites old is worse than no floor.

Then report a 2–3 sentence summary — floor, gap, label — back to the worker.
Do not dispatch other agents and do not try to advance the loop.
