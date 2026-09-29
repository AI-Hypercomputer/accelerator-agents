# Microbenchmarking a kernel — an operating instruction

For an agent optimizing a compute kernel. Applies to any kernel on any
accelerator.

A **microbenchmark** is a standalone program that runs one component of your
kernel — or one candidate reformulation of it — outside the kernel, and reports
a rate or a per-iteration cost. It is not a smaller version of your kernel and
it is not a test. It exists to produce a number your kernel cannot produce about
itself.

---

## 1. Purpose

Three things, and only these three.

**P1 — Establish a reference price.** You know some component costs *N*. You do
not know whether *N* is reasonable. Every timing taken from inside your kernel
is self-referential: it can rank your kernel's parts against each other, but no
combination of them can tell you that a component is running at 0.43× the speed
the same work reaches when nothing else is around it. That judgement requires a
measurement taken *outside* the thing being judged, and a microbenchmark is the
only instrument that produces one.

This is the highest-value use, because it changes the *kind* of conclusion you
can draw. "Component X is 2/3 of compute" suggests doing less of X. "Component X
costs 2.3× what the identical shape costs standing alone" says the volume of X is
fine and the structure *around* X is broken. Those lead to opposite work.

**P2 — Search the design space cheaply.** When you have several candidate
formulations of the same computation and each would be hours of work to
implement for real, build all of them in a standalone harness and rank them
first. Seven candidates in twenty minutes is an ordinary result; five of the
seven being rejected is also an ordinary result, and that is the point — you
rejected them for the price of writing them, not the price of integrating them.

**P3 — Validate a hardware assumption before designing on it.** Any time you are
about to commit to a structure that rests on a belief about the machine — "many
small transfers cost the same as one large one", "this layout is legal", "the
narrow dtype halves the cost" — measure the belief directly first. These probes
are the cheapest in the set and the most likely to change a design.

**Do not write one** when the idea can be tested with a small local edit to the
real kernel and one measurement run. That path is both faster and free of the
modelling error described in §6.

---

## 2. The three archetypes

### 2.1 Price probe — "what should this cost?"

Produces a **rate**, not a time, for one primitive at the shapes your kernel
actually uses.

1. Write a standalone kernel that does nothing but the primitive, `REPS` times
   inside one grid step.
2. Sweep the shape parameter your real kernel varies. Pick the one your tiling
   or blocking controls — for a matmul in a blocked loop that is usually the
   narrow dimension; for a reduction, the reduction length.
3. Report in a unit tied to your **bottleneck hypothesis**, not in seconds. If
   you suspect the operand stream is the constraint, report elements/s or bytes/s
   of that operand. If you suspect arithmetic, report FLOP/s. The unit encodes
   the hypothesis, and choosing it forces you to state one.
4. Multiply the rate back out against the real kernel's workload to get an
   "ideal cost", and divide into the observed cost. **The ratio is the finding**,
   not either number alone.

A representative result. Sweeping the narrow dimension `n` of a blocked matmul,
reporting throughput of the *streamed* operand:

```
| operand dtype | n=4   | n=8   | n=32  | n=128 |
| wide          | 1816  | 1803  | 1857  | 1740   G-elem/s
| narrow        | 3730  | 3539  | 3357  | 1777
```

Both findings come from the *shape* of the table, not any single entry. The rate
is flat in `n` while `n` is small — quadrupling the rows costs nothing, so the
cost is set by streaming the operand, not by arithmetic. And the narrow dtype is
~2× faster at small `n` — so the port is **byte**-limited, not element-limited,
which means the narrow dtype buys a real 2× and is worth carrying through the
design. At `n=128` both collapse to the same rate: the constraint has switched
to arithmetic and the dtype advantage vanishes. One table, three structural
facts about the machine.

### 2.2 Bake-off — "which formulation is fastest?"

Ranks candidate implementations of the same semantic operation.

1. **Reproduce the whole inner loop, end to end** — every stage of one iteration
   of your innermost body, not one primitive from it. Partial reproduction is
   the single largest source of wrong predictions.
2. Implement every candidate against the same inputs and the same timing path.
3. **Always include two special arms beyond your candidates:**
   - the **status quo** — what the kernel does today, so every number is a ratio
     against something real;
   - a **control**: the idealized version in which the thing you are optimizing
     is *free* — inputs handed over pre-transformed, the step deleted outright.
     No candidate can ever beat the control, so it prices the entire
     optimization direction in one row.

   The control is what makes negative results possible, and negative results are
   usually the valuable ones. See §4.
4. Sweep at least two shapes spanning your real operating regimes. When they
   disagree, that disagreement is a finding — it usually becomes a runtime
   guard rather than a single static choice.
5. Report nanoseconds per inner-loop iteration, so the number multiplies cleanly
   by your kernel's iteration count and can be checked against reality.

### 2.3 Property probe — "is this assumption about the machine true?"

1. Strip to the single mechanism — a copy, a load, a barrier. No compute.
2. Measure the **ratio** between the two configurations you care about, not the
   absolute. Without overlap or double-buffering the absolute will be low; the
   ratio is still valid, and the ratio is the question.
3. **Record every error the probe throws.** These probes routinely fail at
   compile time in ways that reveal a hard constraint. A layout probe that dies
   with an alignment error on a particular dimension has just told you that
   dimension is inside the hardware tile and cannot be sliced — ruling out a
   whole design branch in seconds. That is a successful run, not a failed one.

---

## 3. Design rules

Each of these is a way a microbenchmark silently reports a wrong number. Treat
as pre-flight checks.

### 3.1 Measure the dispatch floor first

Time a null workload through the identical launch path — an identity or `x+1`
kernel with the **same argument count and donation pattern** as the real one:

```
jit(lambda x: x+1), small donated array, 200 iters   ->  24.5 µs/call
jit(lambda q,k,v: (q,k,v))                           ->  ~52 µs/call
```

Everything you subsequently measure is `floor + work`. The floor tells you which
effects your timing method can resolve at all, and lets you recognize a result
that is really just the instrument. It scales with arity and donation, so
measure it at yours, not from a remembered number.

### 3.2 Defeat dead-code elimination

Consume the **entire** result into a scratch buffer or accumulator. A version
that consumes only a slice of the output invites the compiler to delete the rest
of the work.

**Diagnostic:** if measured cost does not grow when you grow the work — identical
timing at two problem sizes — you are not measuring the work.

### 3.3 Report a slope, not an absolute

Measure at two problem sizes and report the difference:

```
rate = (work(m₂) − work(m₁)) / (time(m₂) − time(m₁))
```

This cancels every fixed cost — launch, grid setup, prologue, allocation —
without having to model any of them. Doubling the size (e.g. m=1024 vs m=2048)
is a good default.

### 3.4 Size the repetition count against the floor

Raise `REPS` until total time is at least ~10× the floor from §3.1. A few dozen
repetitions is usually not enough; several hundred to a couple of thousand
typically is.

**Diagnostic:** if every arm reports the same number *and* that number is near
the floor, you measured the harness, not the kernel.

### 3.5 Sweep one axis, and make it a real kernel parameter

The swept variable should be something the real kernel genuinely varies with its
input — block size, rows per block, reduction length. When the axis is a real
parameter, results transfer to the kernel without reinterpretation; when it is
an invented one, you will spend longer arguing about the mapping than you spent
measuring.

### 3.6 The benchmark owns its inputs

Never import the production benchmark harness for its input builder. Harnesses
routinely construct *every* test case eagerly at import, including the largest
ones, which can dominate the run:

```
minimal imports only                         ->  12.5 s startup
same script + one production harness import  -> 201.5 s startup
```

Roughly 12 s of that is framework and device initialization; the remaining ~190 s
is input construction no microbenchmark needed. Multiply by every iteration of
the edit-run loop and this becomes the largest single cost of an optimization
session — hours spent regenerating identical random tensors. Allocate minimal
inputs locally. If you must share a builder, make it lazy — build case *i* on
request, never all of them at import.

### 3.7 Handle donation and mutation

If the kernel donates or mutates a buffer, hand every measured call a **fresh
device copy**. Otherwise the second call fails outright ("donation requested for
invalid buffer"), or worse, a sweep silently corrupts later configurations by
consuming an array an earlier one already destroyed.

If you want to chain calls to amortize launch cost, first verify the operation is
**idempotent** — that repeating it writes the same values to the same places.
If it is not, chained timings drift and the numbers are meaningless. Also check
that chaining has not defeated donation: an intermediate value cannot be donated,
so the compiler may insert a full copy of a large buffer between calls. The tell
is a chained "device" time *larger* than the unchained wall time, which is
impossible; if you see it, discard that measurement path rather than trying to
patch it.

### 3.8 Cross-check the headline two ways

Convert the result into a second unit and check it against something
independently known — a datasheet peak, a bandwidth figure, a previously
measured constant.

Worth doing explicitly, because it catches factor-of-2 errors that survive
everything else. If a probe reports `1740e9 elem/s` at `n=128`, then the implied
arithmetic rate is `1740e9 × 2 FLOP × 128 ≈ 445 TFLOP/s`. If elsewhere you have
written that the same run "reached peak" on a machine whose peak is ~918 TFLOP/s,
one of the two numbers is wrong by 2× and you need to find out which *before*
building a cost model on it. Two lines of arithmetic at write-up time.

### 3.9 Annotate arms that are not comparable

It is easy to let one group of arms differ in a second respect — a dtype cast
applied to some and not others. That can make a subset unrealistically
pessimistic, so those columns are comparable within the group but not across it.
When it happens, **annotate the asymmetry in the table itself, adjacent to the
numbers**, not in prose three paragraphs away. Tables outlive the text around
them.

---

## 4. Interpreting the result

Read the *pattern across arms* before reading any single number.

| pattern | meaning | next action |
|---|---|---|
| all arms ≈ equal ≈ the floor | the benchmark measured nothing | fix DCE / `REPS`, re-run (§3.2, §3.4) |
| cost flat as the swept axis grows | that axis is not the constraint | rebuild the cost model around the operand or stage that is |
| **control ≈ status quo** | **the component you isolated is not the bottleneck** | **stop optimizing it; suspect the structure around it** |
| one arm wins by ≫ noise | a genuine design decision | implement the winner, re-measure in the real kernel |
| arms disagree across the shape sweep | the right choice is input-dependent | implement both behind a runtime guard |
| prediction good, real kernel regressed | your model omits a term | identify and quantify the term (§6) |

The third row is the one to hunt for, and the reason §2.2 makes a control arm
mandatory. A representative instance: an inner-loop bake-off whose control —
operands handed over pre-transformed, so the data-marshalling step cost
literally nothing — came in at 3013 ns against the existing implementation's
3177 ns. A 5% ceiling on an item that in-kernel accounting had priced at tens of
microseconds. Everything downstream of that reading changed: the work moved off
the data layout entirely and onto the loop *structure*, where collapsing eight
short serially-dependent chains into one stacked chain was worth **2.3×**. The
same table showed the win shrinking to 1.35× at the other end of the shape
sweep, which became a runtime guard in the shipped kernel rather than an
unconditional change.

A benchmark with no control arm cannot produce that row at all. It can only tell
you which of your candidates is least bad.

---

## 5. Reporting

The microbenchmark is disposable; its table is not. Write the table down
somewhere durable, because it is what carries the reasoning after the code is
gone.

A complete report is:

- **arms × shapes**, one unit, stated in the header;
- the **floor** from §3.1 recorded alongside;
- **caveats inline** for any non-comparable arms (§3.9);
- the **ratio** against the status quo, not just absolutes;
- a closing line naming **what decision this changes**. If you cannot write that
  line, you have a number, not a finding;
- the **date and kernel version** measured. These numbers go stale the moment
  the kernel is restructured, and a stale table quoted as current is worse than
  no table.

---

## 6. Validity limits

A microbenchmark measures the mechanism you simulated and nothing else. The
recurring omissions:

- **Per-request issue cost.** Throughput-per-byte and cost-per-request are
  different quantities. A probe showing that many small transfers achieve the
  same bandwidth as one large transfer says nothing about the cost of *issuing*
  them.
- **Resource pressure from neighbours** — register, scratchpad and occupancy
  effects caused by state that only exists in the real kernel.
- **Pipeline overlap.** In isolation nothing hides latency; in situ the
  surrounding work may hide it entirely, or may contend for the same port.
- **Per-launch overheads** that amortize differently at real iteration counts.

A representative failure: a layout change measured 1.6× faster on the isolated
step, and a companion probe confirmed the extra transfers it required were
bandwidth-free. Both results were correct. Built into the kernel, it regressed
the long-input cases by 1.56× and was reverted — bandwidth was free but *issue*
was not, at ~63 ns per request, and neither probe had simulated request count.
Cases that re-fetch their working set once per output block paid that overhead
on every pass.

**Operating rule:** a microbenchmark prediction is **necessary but not
sufficient**. It is strong evidence an idea is worth building and weak evidence
it will ship. Plan for a meaningful fraction of microbenchmark-backed ideas to
be built and reverted; that fraction is the price of not guessing, and it is
much cheaper than the alternative, but it is not zero.

When a prediction fails in the real kernel, **extend the benchmark rather than
discarding it**. Name the missing term and quantify it from the regression — a
figure like "63 ns per request issue" is reusable in every subsequent design
decision. The disappointment is not.

---

## 7. Procedure

1. **Floor.** Time a null workload at your kernel's arity and donation pattern.
   Record it. (§3.1)
2. **Purpose.** State which of P1/P2/P3 you are serving, as one sentence. If you
   cannot, you do not need a microbenchmark yet.
3. **Hypothesis and unit.** State what you believe the bottleneck is, and pick
   the reporting unit that would confirm or refute it. (§2.1 step 3)
4. **Minimal harness.** Own inputs, no production imports, fresh buffers per
   call. (§3.6, §3.7)
5. **Arms.** Candidates + status quo + control. (§2.2 step 3)
6. **Validate the instrument** before trusting any arm: does cost grow with work
   (§3.2), is total time ≫ floor (§3.4), does the slope method give a stable
   rate across two independent size pairs (§3.3)?
7. **Sweep** one real kernel parameter across ≥2 operating regimes. (§3.5)
8. **Cross-check** the headline in a second unit. (§3.8)
9. **Report** per §5.
10. **Implement the winner and re-measure in the real kernel.** Compare
    predicted against actual; if they disagree, go to §6.
11. **Delete the benchmark, keep the table.**

---

## 8. Checklist

```
[ ] floor measured at this arity, recorded next to the results
[ ] benchmark owns its inputs; no production harness import
[ ] full result consumed (cost grows with work)
[ ] total time >= ~10x floor
[ ] rate reported as a slope between two sizes
[ ] status-quo arm present
[ ] control / upper-bound arm present
[ ] >= 2 shapes spanning the real operating regimes
[ ] unit matches the stated bottleneck hypothesis
[ ] headline cross-checked in a second unit
[ ] non-comparable arms annotated inside the table
[ ] fresh buffers per measured call (donation/mutation safe)
[ ] chained timings verified idempotent, and not slower than unchained
[ ] table records date, kernel version, and what decision it changes
```
