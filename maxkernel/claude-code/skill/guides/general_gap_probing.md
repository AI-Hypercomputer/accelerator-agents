# Gap probing — why the halves of a kernel fail to overlap

Applies to any asynchronous producer/consumer pipeline: data movement and
compute, two compute units, host and device, network and disk.

```
gap = actual − max(producer-only, consumer-only)
```

The floor tells you *how much* is left. A gap probe tells you *why* it is there.
It is the third kind of experiment, after attribution and floor measurement, and
it is the only one that can explain the gap — both floor probes delete the
producer/consumer interaction along with the half they delete, so neither can see
it.

---

## 0. When to apply it

Gap probing is triggered by a measurement, not run on a schedule. **All four
conditions should hold:**

1. **You have a measured floor, and the gap is large** — roughly >10–15% of the
   floor. Without a floor measurement you would be perturbing the schedule
   without knowing whether the schedule is the problem.
2. **The gap is outside measurement noise.** Re-measure one configuration twice
   and confirm the spread is far smaller than the gap. Cheap, and it rules out
   the embarrassing case.
3. **The kernel actually has concurrency.** Resident data and synchronous
   execution means there is nothing to overlap.
4. **Work reduction on the binding half has stopped paying.** This is the one
   most often gotten wrong. Gap probing recovers *schedule*, not *work*. If the
   binding half is still far from its own hardware roof, its formulation is the
   bigger prize — and fixing it moves the gap anyway, so any probe result you
   collect now describes a kernel you are about to replace.

**Do not apply it when:**

- `actual ≈ max(...)` — the schedule is already fine; further improvement must
  come from reducing work.
- A candidate change was just **rejected**. The kernel did not move, so neither
  did the gap.
- The bottleneck is not representable as two stages — instruction cache
  pressure, or a single stage that is itself internally serialized. See §5.

### Where it sits in the workflow

```
floor              which half binds, and how large the gap is
  └─ attribute     break down the binding half; reduce its work
       └─ gap probe   explain what overlap is not recovering
            └─ stop   or redesign
```

In practice this puts gap probing **late** — it is what remains once the
structural wins are gone — and it should run **before** you declare the kernel
done, because the gap is precisely the remaining prize.

### Cadence

Like the floor, a gap-probe result is tied to **one kernel state**. A cause
refuted at one state can become binding at another: a change that cuts compute
shortens the window the producer had to hide in, so lookahead that was ample
becomes insufficient without anyone touching the transfer code. **Re-probe after
a structural change; never carry a refutation forward.**

---

## 1. The defining constraint

A gap probe is built and run exactly like any candidate change: write a variant,
sweep it, compare. The machinery is identical. What differs is one rule:

> **A gap probe holds the work constant.** Same bytes moved, same arithmetic
> performed, same block sizes. Only the *interaction* between the halves is
> perturbed — buffer depth, issue pattern, synchronization, loop structure.

That constraint is what licenses the inference. If total work is unchanged and
the time still moves, overlap is the only remaining explanation.

### Three levels: code, work, schedule

Collapsing these is the usual source of confusion.

- **Code** — the source text you write.
- **Work** — what the hardware is asked to accomplish: bytes across the bus,
  multiply-accumulates, instructions issued. Countable, independent of how you
  wrote it.
- **Schedule** — *when* each piece of that work happens relative to the rest: how
  many transfers are in flight, what waits on what, in what order things issue.

All three experiment types change the **code**. They differ in the rest:

| | changes work | changes schedule | output still correct? |
|---|---|---|---|
| ablation | **yes — less of it** | incidentally | no, garbage by design |
| candidate change | **yes — same answer, less work** | maybe | yes, must be |
| **gap probe** | **no** | **yes, only this** | yes |
| timeline inspection | no — changes nothing at all | no | yes |

### The operational test

"Is the work the same?" is hard to verify by inspection. This is sharper:

> **Does the probe leave `producer-only` and `consumer-only` unchanged?**
> If both floors hold and the total moves, the difference is overlap.

Measure that, do not reason it. It costs two extra runs and it is the difference
between a result and an assertion.

### The invariant is easy to break by accident

Resource-budgeted parameters are the trap. Raising a buffer count consumes
on-chip memory, which can silently force the block size down, which changes tail
padding, which changes the work. A probe run that way produces a number that
could be overlap, or padding, or both, and cannot separate them. **Always
re-measure at a matched block size**, and treat the first run of any
depth-increasing probe as suspect until you have.

---

## 2. A probe tests a cause; it does not measure overlap

The gap is a single scalar. A probe perturbs **one suspected cause** of that
scalar and watches whether it moves — inference by intervention.

- gap responds ⇒ that cause contributes
- gap does not respond ⇒ that cause is largely not it

No probe will ever produce "the halves overlapped 78% of the time."

**Before perturbing anything, check whether you can just look.** Most profilers
record a start timestamp alongside each event's duration; with both you can
reconstruct when each unit was busy, and the holes between them *are* the gap,
directly observed. Aggregating tools routinely sum durations and discard
timestamps. If yours does, you will spend hours inferring indirectly what a
timeline view would have shown immediately — and the data was already being
collected.

---

## 3. The six causes

The **taxonomy is fixed** — any producer/consumer pipeline fails to overlap for
one of these six reasons. The **implementation of each probe is not**; it is
specific to your kernel's structure.

| # | cause | one-line probe |
|---|---|---|
| 1 | insufficient lookahead | increase buffer depth / prefetch distance |
| 2 | issue bandwidth | make each issue cheaper, or issue fewer |
| 3 | over-strict serialization | delete or weaken a wait |
| 4 | resource contention | hold one half fixed, vary the other's volume |
| 5 | genuine dependency | no probe — read the dependence structure |
| 6 | granularity | vary chunk size at constant total work |

### 3.1 Insufficient lookahead

**Hypothesis.** The consumer stalls because the producer was not told to start
early enough. With depth *d*, the producer has *d−1* units of consumer-time to
complete a transfer; anything longer is exposed on every unit.

**Suspect when** the gap scales with the **number of blocks**, not bytes — halving
the block size roughly doubles it — and the in-flight depth is small. Double
buffering is depth 2, i.e. exactly one block of lead time.

**Action.** Sweep depth 1, 2, 4, 8, **holding block size and total bytes fixed**.
Widen the buffer scratch and the semaphore array, cycle the index modulo *N*,
prefetch *N−1* ahead.

**Reading.** Gap shrinking roughly as *1/d* then plateauing ⇒ lookahead was
binding, and the plateau marks the depth at which latency is fully hidden. Flat
from the start ⇒ refuted.

**Invariant.** This is the probe most likely to break the work-constant rule (see
§1). Depth is a resource parameter.

*Representative result.* Depth 2 → 4 at matched block size: on one case the gap
fell by about a quarter — **partially confirmed**, but three quarters of that
case's gap remained unexplained. On another case it did not move at all —
**refuted**. Depth 8 exceeded the on-chip memory budget. Partial confirmation is
the common outcome; treat "the gap moved" and "the cause is found" as different
claims.

### 3.2 Issue bandwidth

**Hypothesis.** The producer is idle not for lack of instructions but because the
core cannot *dispatch* requests fast enough. The request stream, not the data
stream, is the bottleneck.

**Suspect when** the gap scales with the **number of requests** rather than bytes
or blocks — typical with many small transfers, gather/scatter patterns, one
descriptor per tile.

**Action.** Two independent interventions:
- *cheaper issue* — unroll the issue loop, hoist address arithmetic out,
  precompute descriptors. Same requests, same bytes, fewer instructions between
  them.
- *fewer issues* — coalesce adjacent requests, total bytes identical.

Also bound it standalone: time *N* issues of a minimal transfer in a
microbenchmark to get nanoseconds-per-issue, multiply by the real request count,
compare against the gap.

**Reading.** Gap shrinking in proportion to issue cost or count removed ⇒
confirmed. A 2× reduction in issue count moving nothing ⇒ refuted.

**Invariant.** Coalescing changes the access pattern and can change achieved
bandwidth, i.e. it moves `producer-only` too. **Prefer the "cheaper issue"
variant when you need a clean reading.**

*Representative result.* Unrolling a per-page issue loop into straight-line
guarded issues moved the total by 0.4% — **refuted**, cleanly and in a few
minutes. Independently corroborated by a standalone probe showing 8 transfers per
page achieved the same bandwidth as 1.

### 3.3 Over-strict serialization

**Hypothesis.** A synchronization point is more conservative than the true data
dependency — waiting on something not yet needed, or at a coarser granularity
than required (waiting for a whole block when the first chunk needs only the
first page).

**Suspect when** the gap responds to neither depth nor issue rate, and there are
explicit waits whose predicate is visibly stronger than the dependency they
protect.

**Action.** Enumerate every synchronization point. For each, build a probe that
deletes or weakens it — **numerical safety is irrelevant, a probe never ships** —
and measure. For any wait that moves the gap, the real fix is to narrow its
scope: finer-grained semaphore, moved later, or split.

**Reading.** Deleting a wait moves the gap ⇒ that wait is on the critical path
and its scope is the thing to attack. Nothing moves ⇒ the waits are already
tight.

**Invariant.** Deleting a wait can make the kernel *both* faster and wrong in a
way that changes the work — a race that causes it to read less data than it
should. **Verify byte counts are unchanged before believing the number.**

### 3.4 Resource contention

**Hypothesis.** The halves are not independent. They compete for a shared
resource — memory ports, bus bandwidth, issue slots — so running them
concurrently is slower than either alone even under a perfect schedule.

> **If this is true, `max(producer-only, consumer-only)` is the wrong floor
> formula.** The real floor is higher — something like `consumer + α·producer` —
> and part of what looks like recoverable gap is not recoverable at all.

**Suspect when** causes 1, 2, 3 and 6 have been probed and refuted, and the
residual is roughly proportional to `min(producer-only, consumer-only)` — the
time the two halves necessarily coexist.

**Action.** Hold one half fixed, vary the other's **volume**. Start from the
consumer-only variant and add the transfers back at 0%, 25%, 50%, 100% of real
volume (fetch every fourth page, every second, all) while keeping the arithmetic
identical. Plot total time against bytes restored.

**Reading.**
- Flat until restored transfer time exceeds compute, then rising with slope 1 ⇒
  pure latency exposure, no contention. `max()` is the right floor.
- Rising from the very first restored byte ⇒ contention. **The slope is the
  contention coefficient**, and it tells you how much of the gap is structural.

**Invariant.** Compute volume must stay identical as transfers are scaled — fetch
fewer pages, but still run the same number of chunks over whatever is in the
buffer.

### 3.5 Genuine algorithmic dependency

**Hypothesis.** There is no overlap to recover. The algorithm requires A before B
— a reduction feeding the next stage, a running maximum that must be known before
the next block is rescaled, carried state.

**Suspect when** the gap survives every probe above and is approximately the
length of the serial chain.

**Action.** **There is no probe.** This is established by reading the dependence
structure. What you *can* measure is the critical path: sum the latencies of the
stages that must run in order and compare against the gap. If they match, you
have your answer.

**Response.** Redesign, not tuning — software-pipeline across iterations so stage
*i+1* of one unit overlaps stage *i* of the next, split the reduction, or accept
the cost.

### 3.6 Granularity

**Hypothesis.** The unit of work is the wrong size. Too large and the consumer
cannot start until much has arrived, while the tail unit wastes work on padding.
Too small and per-unit overhead — issue, synchronization, loop control, pipeline
fill — dominates.

**Suspect when** the gap has a **U-shape** in chunk size: bad at both ends, best
in the middle.

**Action.** Vary chunk size at **constant total bytes and constant total flops**,
and plot the gap.

**The trap, and it is serious.** In most kernels, changing the block size also
changes tail padding, which changes the work — so a naive block-size sweep
confounds granularity with work and is not a valid gap probe. Two ways out:
choose only sizes that divide the data evenly, or measure the padding cost
separately and subtract it.

**The standard remedy.** When one knob is being asked to satisfy two opposing
pressures — large transfers for bandwidth, small chunks to limit padding waste —
**split it into two knobs.** Decoupling fetch granularity from compute
granularity resolves cause 6 structurally rather than by finding a compromise
point on a single axis.

---

## 4. Order, cheapest first

1. **Granularity** — if a block-size sweep harness already exists this is nearly
   free, and it is the most common cause in blocked kernels. Watch the padding
   confound.
2. **Lookahead** — usually a small structural change, and often the answer for
   double-buffered kernels.
3. **Issue bandwidth** — cheap to probe, and cheap to bound independently with a
   microbenchmark.
4. **Serialization** — requires enumerating the waits, but each probe is a
   one-line deletion.
5. **Contention** — most work to set up, and correctly the last resort: it is
   what remains when the others are refuted, and confirming it **changes the
   floor formula rather than the kernel**.
6. **Dependency** — reasoning, not measurement. Do it whenever the others come
   back empty.

---

## 5. Scope

**Applies when** the kernel has two or more resources that are *supposed* to run
concurrently, and you have measured a gap between achieved time and
`max(stage times)`. Causes 1, 2, 3, 4 and 6 are properties of any asynchronous
producer/consumer pipeline, not of any one vendor's hardware.

**Does not apply when:**

- there is no gap (`actual ≈ max(...)`) — the schedule is already fine, and any
  further improvement must come from reducing work;
- there is no concurrency — resident data and no asynchronous operations means
  nothing to overlap;
- the kernel is bound by something the two-stage model does not represent —
  instruction cache pressure, or a single stage that is itself internally
  serialized;
- the "gap" is measurement error. **Rule this out first** by re-measuring the
  same configuration twice and confirming the gap is well outside the spread.

**What does not transfer is the implementation.** Buffer scratch shapes,
semaphores and descriptor loops are platform specifics; the equivalents elsewhere
are streams, async-copy pipeline stages and barrier waits. The causes are
identical while every line of the probe differs.

---

## 6. The discipline this exists to enforce

The failure mode this whole framework guards against is simple and common:
**attributing the residual gap to a cause that was never probed.** It is easy to
run one or two probes, have them come back refuted, and then write "the remaining
gap is port contention" in the conclusions — a sentence that reads like a finding
and is in fact an unfalsified assertion.

Six causes. Say which you probed, what each returned, and which you left
untested. An honest "three of six untested, including the one I suspect" is worth
more to the next person than a confident attribution with no experiment behind
it — and, per §3.4, if the untested one is contention, the floor you have been
measuring against may be wrong.
