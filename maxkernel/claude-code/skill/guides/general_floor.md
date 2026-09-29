# The floor — how fast could this kernel possibly go?

Applies to any kernel that both moves data and computes on it, on any
accelerator.

Attribution tells you *where to aim*. The floor tells you *whether you are done*.
This document is about the second.

---

## 1. What a floor is, concretely

A kernel does two things: move data from memory, and compute on it. Cripple each
in turn:

- **compute-only** — data movement disabled. *How long does the arithmetic take
  if data were free?*
- **transfer-only** — arithmetic disabled. *How long does the data movement take
  if arithmetic were free?*

The real kernel must do **both**, so it can never beat whichever half is slower
on its own:

```
floor = max(compute-only, transfer-only)
gap   = actual − floor
```

The floor is unreachable without a redesign. The gap is what is still on the
table in principle.

*Worked example.*

```
   compute-only    238.9 µs     arithmetic, if data were free
   transfer-only   252.9 µs     data movement, if arithmetic were free
                   --------
   floor           252.9 µs     <- unreachable without a redesign
   actual          303.6 µs
   gap              50.7 µs     <- 17 %, recoverable in principle
```

Three properties distinguish this from attribution:

| | floor |
|---|---|
| number of experiments | **exactly two**, always the same two |
| what they produce | **absolute times** for a crippled kernel, not deltas — so they do not suffer the overlap problem that makes attribution deltas non-additive |
| rebuilt when the kernel changes | **yes** — the same two cuts, re-applied to whatever kernel is current |

**There are exactly two probes, and therefore one floor.** Not a family, not a
per-stage set — two cuts, each removing one whole half of the kernel, and their
maximum. If you find yourself building a third, you are building an attribution
variant or a gap probe, not a floor.

Three unrelated things also get called "floor"; keep them separate:

| term | what it is | where it comes from |
|---|---|---|
| **floor** (this document) | `max(compute-only, transfer-only)` | **measured**, by crippling the current kernel |
| roofline bound | bytes ÷ bandwidth, FLOPs ÷ peak, etc. | **computed** from specs; a property of the algorithm and machine, not of your code |
| dispatch / launch floor | the fixed per-call cost of getting work onto the device | a property of your **measurement harness** |

Only the first moves when you change the kernel, and only the first is what §3's
cadence is about.

---

## 2. How to build the two probes

**compute-only.** Make the transfer primitive a no-op on *both* its issue and its
wait paths. The kernel then computes on whatever happens to be resident in
on-chip memory.

**transfer-only.** Place an early return in the per-block body **after the
transfer has been issued and awaited**, before the compute begins. Every transfer
still happens; nothing is computed.

That placement is the one detail worth getting right. Return too early and you
skip the transfer you meant to measure; skip the wait and you leave a resource
outstanding, which on strict hardware faults the device rather than returning a
wrong number.

Both probes produce garbage output by construction — one reads uninitialized
memory, the other computes nothing — so **run them with correctness checking
disabled**. A check would fail on both and carry no information.

Measure with an instrument that can actually resolve the kernel. If the kernel's
total time is comparable to your launch overhead, a wall-clock reading of either
probe will return the overhead and nothing else.

The pair is cheap — a few minutes end to end. That cheapness is what makes it
re-runnable, and re-running it is the whole point.

---

## 3. When to compute a floor

Once on the baseline, before optimizing, to choose the initial direction. After
that, **the floor is computed only when a hypothesis has been tested and kept.**

The cycle:

1. **Form a hypothesis and implement it.**
2. **A/B it** against the unmodified kernel — same cases, same measurement
   protocol, before and after.
3. **Regressed or negligible → drop it. Do not compute a floor.** The kernel did
   not move, so the previous floor still stands and re-measuring would tell you
   nothing.
4. **Improved → keep it, then compute the floor.** The thing the floor describes
   has changed, so the floor has changed with it.
5. **Read the gap and the label** (§4), and decide: continue, and on which half —
   or stop.

This pattern is clean in practice: every kept change earns a floor check, no
rejected change does. The rule follows from what a floor measures — *how much
room is left in the kernel I now have*. If you do not have a new kernel, you do
not have a new question.

A useful consequence: floor checks are rare. Over a full optimization session you
should expect a handful, one per structural change that survived, not one per
experiment.

---

## 4. How to read it — two outputs, in order of importance

### 4.1 The gap — the primary output

The gap is the size of the remaining prize. It is what makes a small experimental
return interpretable: a 0.5% result against a large known gap means you picked
the wrong attack; the same result against a small gap means you are done.

Rough reading:

| gap | reading |
|---|---|
| **< ~20% of floor**, and the floor is itself near a hardware limit | near the end; further structural work has little left to win |
| **large**, and growing relative to the floor | overlap is failing — the halves are not hiding behind each other |
| **large**, floor far from any hardware limit | the binding half is itself badly formulated; go back to attribution |

Always sanity-check the floor against the hardware. A floor is only a stop signal
if it is itself close to a physical limit — divide bytes moved by the
transfer-only time and compare to spec bandwidth.

### 4.2 The label — which half binds

A by-product, but the one that tells you where to aim next. **It is not stable,
and it must never be carried across a kept change.**

*Worked example — the label flipping.* Across three successive kernel states:

```
state A    compute 311.2   transfer 253.6   -> compute-bound
state B    compute 242.9   transfer 276.4   -> transfer-bound   <- flipped
state C    compute 238.9   transfer 252.9   -> transfer-bound
```

The flip at state B was caused by a **block-size retune**, not by any change to
the arithmetic: smaller blocks cut compute (311 → 243) while making the transfers
less efficient (254 → 276). The relabelling sent the next change at the *fetch*
path instead of the arithmetic, and that change was the last structural win of
the session. Trusting the stale "compute-bound" label would have meant optimizing
the wrong half.

Note what this implies: a change that touches neither the arithmetic nor the
transfer code — a pure tuning-parameter change — can still flip which half binds.
Re-measure after tuning changes, not just after rewrites.

---

## 5. What the floor cannot tell you

A natural idea is to measure the floor and then use attribution to explain the
gap. **Attribution cannot do that.** It explains what the *floor* is made of, not
what the *gap* is made of:

```
total           495
compute-only    441   <- attribution breaks THIS down:
transfer-only   294        mask · unpack · matmul #1 · matmul #2 ·
floor           441        reduction · epilogue · loop overhead
gap              54   <- nothing inside either probe accounts for this
```

The gap is by definition the part neither half accounts for on its own: time when
the transfer engine and the compute units are **both idle, each waiting on the
other**. Both probes delete that interaction along with the thing they delete, so
neither can see it. Explaining the gap requires a third kind of experiment —
probes aimed at specific causes of overlap failure (insufficient lookahead, issue
bandwidth, over-strict serialization, resource contention, genuine dependency,
granularity).

---

## 6. The floor is conservative

Deleting the transfers also deletes their **interference** with the compute —
they no longer compete for memory ports. So the true achievable floor probably
sits a little **above** `max(compute-only, transfer-only)`, meaning the kernel is
somewhat closer to optimal than the gap suggests.

The error runs in the safe direction: you will never think you are done when you
are not. It is rarely worth quantifying, but it should be stated whenever the gap
is reported, so a 17% gap is not mistaken for 17% of guaranteed headroom.

---

## 7. Ordering

1. **Floor first.** Two cheap probes. Tells you which half binds and how large
   the overlap loss is.
2. **Attribute the binding half only.** Do not spend time breaking down the half
   that is not limiting — cut every attribution variant on top of the
   compute-only or transfer-only base, whichever one binds, so the other half's
   overlap is out of the comparison.
3. **Gap experiments**, if the gap is large.

Then **repeat 1 and 2 after every kept change.** The common failure is doing this
once at the start and never again: the floor gets re-measured because it is
cheap, the attribution does not, and you end up choosing what to build from a
breakdown of a kernel that no longer exists.

---

## 8. Checklist

```
[ ] compute-only probe: transfer primitive is a no-op on BOTH issue and wait
[ ] transfer-only probe: early return placed AFTER issue and wait
[ ] correctness checking disabled on both probes
[ ] measured with an instrument that resolves the kernel, not launch overhead
[ ] floor = max(the two); gap = actual - floor; both reported
[ ] floor cross-checked against a hardware limit before being called "near done"
[ ] floor computed on the baseline, then after every KEPT change
[ ] no floor computed after a rejected change
[ ] label re-read every time, never carried forward
[ ] gap reported with the note that the true floor sits slightly above max(...)
[ ] attribution, if any, cut on the binding half's base
```
