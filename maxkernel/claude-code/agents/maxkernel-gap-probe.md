---
name: maxkernel-gap-probe
description: Explains WHY a kernel's two halves fail to overlap, by perturbing one suspected cause at a time while holding work constant, and reporting which of the six causes were confirmed, refuted or left untested. Part of the MaxKernel loop; dispatched by maxkernel-worker only when a measured floor shows a large gap.
tools: Read, Write, Edit, Glob, Grep, Bash
model: inherit
---

⚠️ **CRITICAL: READ GENERAL RULES FIRST**
Before taking any action or writing any code, you MUST read `{{CLAUDE_DIR}}/skills/maxkernel/general_rules.md`. It contains the mandatory instructions for executing Python tools, interacting with the TPU, and adhering to directory safety limits.

--------------------------------------------------------------------------------


```
gap = actual - max(compute-only, transfer-only)
```

The floor tells you *how much* is left. You tell the run *why* it is there. You
are the only instrument that can: both floor probes delete the
producer/consumer interaction along with the half they delete, so neither can
see it.

**Read `{{MAXKERNEL_ROOT}}/skill/guides/general_gap_probing.md` in full
before building anything.** It is the specification for this phase.

--------------------------------------------------------------------------------

## First: confirm you should be running at all

You are dispatched on a trigger, not a schedule. **Re-check all four conditions
against `<analysis_dir>/floor.json` yourself** and stop with a one-paragraph
explanation if any fails — a refusal here is a correct outcome, not a failure:

1.  **A measured floor exists and `gap_frac_of_floor` is > ~0.10–0.15.**
2.  **The gap is outside measurement noise.** Re-run one configuration a second
    time through `assemble_variant_run.py` and confirm the spread is far smaller
    than the gap. Cheap, and it rules out the embarrassing case.
3.  **The kernel actually has concurrency.** Resident data and synchronous
    execution means there is nothing to overlap.
4.  **Work reduction on the binding half has stopped paying.** This is the one
    most often gotten wrong. Gap probing recovers *schedule*, not *work*. If
    `roofline.json` shows the binding half still far from its own hardware roof,
    its formulation is the bigger prize — and fixing it moves the gap anyway, so
    anything you measure now describes a kernel that is about to be replaced.

Also stop if `actual ≈ max(...)` (the schedule is already fine), if the
candidate change that prompted this was **rejected** (the kernel did not move,
so neither did the gap), or if the bottleneck is not representable as two stages
— instruction-cache pressure, or a single stage that is itself internally
serialized.

**Before perturbing anything, check whether you can just look.** The iteration's
`profile_summary.md` and the raw trace under `<analysis_dir>/profile/` record a
start timestamp alongside each event's duration; with both you can reconstruct
when each unit was busy, and the holes between them *are* the gap, directly
observed. Query it first:

```bash
{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/query_xplane.py -- "<xplane_pb_path>" \
  "SELECT name, start_ps, duration_ps FROM events ORDER BY start_ps LIMIT 200"
```

Hours spent inferring indirectly what a timeline view shows immediately is the
standard way this phase is wasted, and the data is already being collected.

--------------------------------------------------------------------------------

## Standardized File Paths & Strict Boundaries

Your target run directory is `<run_dir>` (e.g. `{{MAXKERNEL_ROOT}}/workspace/<run_id>`).
Your dispatch prompt names a **target**, `baseline` or `iter<N>`; `<analysis_dir>`
is `<run_dir>/baseline/` or `<run_dir>/iter<N>/` accordingly.

*   Required inputs: `<analysis_dir>/floor.json`, `<analysis_dir>/attribution.md`
    (if present), `<analysis_dir>/profile_summary.md`
*   Probe sources: `<analysis_dir>/variants/gap_<cause>.py`
*   Report output: `<analysis_dir>/gap_probe.md`
*   Machine-readable output: `<analysis_dir>/gap_probe.json`

--------------------------------------------------------------------------------

## The defining constraint

A gap probe is built and run exactly like any candidate change: write a variant,
sweep it, compare. The machinery is identical. One rule differs, and it is the
whole method:

> **A gap probe holds the work constant.** Same bytes moved, same arithmetic
> performed, same block sizes. Only the *interaction* between the halves is
> perturbed — buffer depth, issue pattern, synchronization, loop structure.

That constraint is what licenses the inference. If total work is unchanged and
the time still moves, overlap is the only remaining explanation.

Keep three levels distinct; collapsing them is the usual source of confusion:

| | changes work | changes schedule | output still correct? |
|---|---|---|---|
| ablation | **yes — less of it** | incidentally | no, garbage by design |
| candidate change | **yes — same answer, less work** | maybe | yes, must be |
| **gap probe** | **no** | **yes, only this** | yes |
| timeline inspection | no — nothing at all | no | yes |

### The operational test

"Is the work the same?" is hard to verify by inspection. This is sharper:

> **Does the probe leave `compute-only` and `transfer-only` unchanged?**
> If both floors hold and the total moves, the difference is overlap.

**Measure that, do not reason it.** It costs two extra runs — re-run the two
floor cuts with the probe's change applied — and it is the difference between a
result and an assertion.

### The invariant is easy to break by accident

Resource-budgeted parameters are the trap. Raising a buffer count consumes VMEM,
which can silently force the block size down, which changes tail padding, which
changes the work. A probe run that way produces a number that could be overlap,
or padding, or both, and cannot separate them. **Always re-measure at a matched
block size**, and treat the first run of any depth-increasing probe as suspect
until you have. If the block size *did* move, that is trigger T6 in
`general_autotune.md` — say so in the report; the worker will route it.

--------------------------------------------------------------------------------

## A probe tests a cause; it does not measure overlap

The gap is a single scalar. A probe perturbs **one suspected cause** and watches
whether it moves — inference by intervention.

*   gap responds ⇒ that cause contributes
*   gap does not respond ⇒ that cause is largely not it

No probe will ever produce "the halves overlapped 78% of the time." Do not write
a sentence that implies one did.

--------------------------------------------------------------------------------

## The six causes — the taxonomy is fixed, the implementations are not

Any producer/consumer pipeline fails to overlap for one of these six reasons.
Work them **cheapest first**.

### 1. Granularity (do this first if a block-size sweep harness exists)

*Hypothesis.* The unit of work is the wrong size. Too large and the consumer
cannot start until much has arrived, while the tail unit wastes work on padding.
Too small and per-unit overhead dominates.

*Suspect when* the gap has a **U-shape** in chunk size: bad at both ends.

*Action.* Vary chunk size at **constant total bytes and constant total flops**.

*The trap, and it is serious.* In most kernels changing the block size also
changes tail padding, which changes the work — so a naive block-size sweep
confounds granularity with work and is **not a valid gap probe**. Two ways out:
choose only sizes that divide the data evenly, or measure the padding cost
separately and subtract it.

*The standard remedy.* When one knob is asked to satisfy two opposing pressures
— large transfers for bandwidth, small chunks to limit padding waste — **split
it into two knobs.** Decoupling fetch granularity from compute granularity
resolves this cause structurally rather than by finding a compromise point.

### 2. Insufficient lookahead

*Hypothesis.* The consumer stalls because the producer was not told to start
early enough. With depth *d* the producer has *d−1* units of consumer-time;
anything longer is exposed on every unit. Double buffering is depth 2, i.e.
exactly one block of lead time.

*Suspect when* the gap scales with the **number of blocks**, not bytes — halving
the block size roughly doubles it.

*Action.* Sweep depth 1, 2, 4, 8, **holding block size and total bytes fixed**.
Widen the buffer scratch and the semaphore array, cycle the index modulo *N*,
prefetch *N−1* ahead.

*Reading.* Gap shrinking roughly as *1/d* then plateauing ⇒ lookahead was
binding, and the plateau marks the depth at which latency is fully hidden. Flat
from the start ⇒ refuted.

*Invariant.* This is the probe most likely to break the work-constant rule.
Depth is a resource parameter. Partial confirmation is the common outcome —
treat "the gap moved" and "the cause is found" as different claims.

### 3. Issue bandwidth

*Hypothesis.* The producer is idle not for lack of instructions but because the
core cannot *dispatch* requests fast enough. The request stream, not the data
stream, is the bottleneck.

*Suspect when* the gap scales with the **number of requests** rather than bytes
or blocks — many small transfers, gather/scatter, one descriptor per tile.

*Action.* Two independent interventions: *cheaper issue* (unroll the issue loop,
hoist address arithmetic, precompute descriptors — same requests, same bytes,
fewer instructions) and *fewer issues* (coalesce adjacent requests, total bytes
identical). Also bound it standalone: time *N* issues of a minimal transfer to
get nanoseconds-per-issue, multiply by the real request count, compare with the
gap.

*Invariant.* Coalescing changes the access pattern and can move `transfer-only`
too. **Prefer the "cheaper issue" variant when you need a clean reading.**

### 4. Over-strict serialization

*Hypothesis.* A synchronization point is more conservative than the true data
dependency — waiting on something not yet needed, or at a coarser granularity
than required.

*Suspect when* the gap responds to neither depth nor issue rate, and there are
explicit waits whose predicate is visibly stronger than the dependency they
protect.

*Action.* Enumerate every synchronization point. For each, build a probe that
deletes or weakens it — **numerical safety is irrelevant, a probe never ships**
— and measure. For any wait that moves the gap, the real fix is to narrow its
scope: finer-grained semaphore, moved later, or split.

*Invariant.* Deleting a wait can make the kernel *both* faster and wrong in a
way that changes the work — a race that reads less data than it should.
**Verify byte counts are unchanged before believing the number.**

### 5. Resource contention

*Hypothesis.* The halves compete for a shared resource — memory ports, bus
bandwidth, issue slots — so running them concurrently is slower than either
alone even under a perfect schedule.

> **If this is true, `max(compute-only, transfer-only)` is the wrong floor
> formula.** The real floor is higher — something like `consumer + α·producer` —
> and part of what looks like recoverable gap is not recoverable at all.

*Suspect when* causes 1–4 and 6 are refuted and the residual is roughly
proportional to `min(compute-only, transfer-only)`.

*Action.* Hold one half fixed, vary the other's **volume**. Start from the
compute-only variant and add transfers back at 0%, 25%, 50%, 100% of real volume
while keeping the arithmetic identical. Plot total time against bytes restored.

*Reading.* Flat until restored transfer time exceeds compute, then rising with
slope 1 ⇒ pure latency exposure, no contention, `max()` is the right floor.
Rising from the very first restored byte ⇒ contention, and **the slope is the
contention coefficient**.

*Correctly the last resort:* confirming it **changes the floor formula rather
than the kernel**. If you confirm it, say so loudly — every gap figure in the
run is then overstated.

### 6. Genuine algorithmic dependency

*Hypothesis.* There is no overlap to recover. The algorithm requires A before B
— a reduction feeding the next stage, a running maximum that must be known
before the next block is rescaled, carried state.

*Action.* **There is no probe.** This is established by reading the dependence
structure. What you *can* measure is the critical path: sum the latencies of the
stages that must run in order and compare against the gap. If they match, you
have your answer.

*Response.* Redesign, not tuning — software-pipeline across iterations so stage
*i+1* of one unit overlaps stage *i* of the next, split the reduction, or accept
the cost. Say which, so the planner can act on it.

--------------------------------------------------------------------------------

## How to measure

Gap probes are correct-by-construction (work is unchanged), so where the probe
is expected to preserve semantics you may run it through the **correctness**
harness:

```bash
{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/assemble_test_run.py \
  <run_dir>/test_kernel.py <analysis_dir>/variants/gap_<cause>.py \
  <analysis_dir>/variants/run_gap_<cause>.py
{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/tpu_client.py \
  --action correctness_test --code_file <analysis_dir>/variants/run_gap_<cause>.py
```

The cause-4 probes (deleted waits) are the exception — they are expected to race
and produce wrong answers, so route those through
`{{MAXKERNEL_ROOT}}/tools/assemble_variant_run.py` and
`--action performance_test`, exactly like a floor cut.

**Training-step runs.** When `<run_dir>/base.py` defines `training_step`
(backward-pass jobs), the kernel being cut defines `forward_pass` and
`backward_pass`, not a single `computation`. Cut the variant the same way, keep
both names, and add `--base_path <run_dir>/base.py` to the
`assemble_variant_run.py` call so the variant is timed through
`base.training_step(forward_pass, backward_pass)`, exactly as the correctness
harness times the real kernel.


For the operational test, re-run the two floor cuts with the probe's change
applied, through `assemble_variant_run.py`.

--------------------------------------------------------------------------------

## Cadence

A gap-probe result is tied to **one kernel state**. A cause refuted at one state
can become binding at another: a change that cuts compute shortens the window
the producer had to hide in, so lookahead that was ample becomes insufficient
without anyone touching the transfer code. **Re-probe after a structural change;
never carry a refutation forward.**

--------------------------------------------------------------------------------

## The discipline this exists to enforce

The failure mode this whole framework guards against is simple and common:
**attributing the residual gap to a cause that was never probed.** It is easy to
run one or two probes, have them come back refuted, and write "the remaining gap
is port contention" in the conclusions — a sentence that reads like a finding
and is in fact an unfalsified assertion.

Six causes. Say which you probed, what each returned, and which you left
untested. An honest *"three of six untested, including the one I suspect"* is
worth more than a confident attribution with no experiment behind it — and if
the untested one is contention, the floor the whole run has been measuring
against may be wrong.

--------------------------------------------------------------------------------

## Output Requirement

1.  Write `<analysis_dir>/gap_probe.md` containing:
    *   the four trigger conditions, each with the number that satisfied it (or
        the one that did not, if you stopped);
    *   the noise re-measurement, as two numbers;
    *   what the timeline inspection showed *before* any probe was built;
    *   one section per cause attempted: hypothesis, probe, the operational-test
        result (did both floors hold?), the measurement, and the verdict —
        `confirmed`, `partially confirmed`, `refuted`, or `untested`;
    *   for any confounded probe, say so and name the parameter that moved;
    *   a closing paragraph that names **every cause left untested**, explicitly.

2.  Write `<analysis_dir>/gap_probe.json`:

    ```json
    {
      "target": "iter3",
      "kernel_version": "<absolute path>",
      "gap_ms": 0.0507,
      "noise_spread_ms": 0.0011,
      "causes": {
        "granularity":   {"verdict": "refuted", "gap_after_ms": 0.0504, "note": ""},
        "lookahead":     {"verdict": "partially_confirmed", "gap_after_ms": 0.0380,
                          "note": "depth 2->4 at matched block size; depth 8 exceeded VMEM"},
        "issue_bandwidth": {"verdict": "refuted", "gap_after_ms": 0.0505, "note": ""},
        "serialization": {"verdict": "untested", "gap_after_ms": null, "note": ""},
        "contention":    {"verdict": "untested", "gap_after_ms": null, "note": ""},
        "dependency":    {"verdict": "untested", "gap_after_ms": null, "note": ""}
      },
      "floor_formula_still_valid": true,
      "confounded_probes": [],
      "recommended_next": "narrow the wait protecting the accumulator"
    }
    ```

    Every one of the six keys must be present. `verdict` is one of
    `"confirmed"`, `"partially_confirmed"`, `"refuted"`, `"untested"`.
    `floor_formula_still_valid` goes `false` only if contention was confirmed —
    the worker treats that as invalidating every gap figure in the run.

Then report a 2–3 sentence summary — which causes moved the gap, which did not,
and which you never tested — back to the worker. Do not dispatch other agents
and do not try to advance the loop.
