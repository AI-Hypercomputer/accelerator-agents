# When to stop optimizing a kernel

Applies to any kernel on any accelerator. The rule has three parts; the third
governs how the first two are used.

---

## 0. The shape of the rule

**Stopping is marginal-return-driven, not threshold-driven.** You do not stop
because a percentage was reached. You stop because further experiments stopped
paying, and the floor measurement is what makes "stopped paying" a defensible
judgement instead of a guess.

Criteria A and B below are the two signals. Neither is sufficient alone.

---

## 1. First, measure the floor — do not compute it

An analytic roofline sets *direction*. The number you actually gate on is an
**empirical** floor, obtained by ablating the **current** kernel into two
variants and timing each:

- **compute-only** — every transfer becomes a no-op;
- **transfer-only** — the compute body returns immediately, every transfer still
  issues.

Then:

```
floor   = max(compute_only, transfer_only)     # what a perfectly overlapped
                                               # kernel of this design would hit
exposed = actual − floor                       # work that is not overlapping
                                               # and is recoverable in principle
```

**Re-measure the floor after every structural change.** The binding constraint
moves. In one representative trajectory, compute-only fell 311 → 243 µs across
two changes while transfer-only held at ~253–276 µs — the kernel **crossed over**
from compute-bound to memory-bound. That crossover is the signal that the next
change should target the fetch, not the arithmetic. A stale floor sends you on
optimizing the side that is no longer binding.

---

## 2. Criterion A — distance to the hardware floor is small

> **Stop considering structural work when the kernel is compute-bound at ~80% of
> its compute-only time, or memory-bound at ~80% of its transfer-only time.**

Equivalently: `actual ≤ ~1.25 × floor`, i.e. exposed time under ~20%.

Then **validate the floor against the hardware**, because a floor is only a stop
signal if it is itself near a physical limit:

- divide bytes moved by the transfer-only time and compare to the part's spec
  bandwidth;
- confirm there is no redundant traffic left to remove — each byte read the
  minimum number of times.

*Representative example.* A case finished at 303.8 µs against a 252.9 µs
transfer-only floor — **83% of floor**, 51 µs exposed. The floor itself worked
out to 1.58 TB/s on a ~1.6 TB/s part, with every byte read exactly once. Both
halves check out: the kernel is close to its design's floor, and the floor is
close to the machine. A second case in the same suite sat at 80% of its
compute-only floor. Nothing structural remained that would move either.

Caveat: ablation deletes work *and* its interference, so `max(compute, transfer)`
may understate the true overlapped floor. The error is conservative — you are
closer to optimal than the number claims — but it is unquantified.

---

## 3. Criterion B — experiments stop returning

> **Stop when multiple experiments return <1% or negative. Either your model of
> the bottleneck is right and exhausted, or it is wrong.**

Two or three consecutive null results is the signal. Before accepting it, ask
one question: **were the failures against *distinct* hypotheses?** Three attacks
on the same suspected cause refute one hypothesis; they do not exhaust the space.
Require independent hypotheses before concluding the queue is empty.

*Representative example — a net win that was still a stop.* A deeper prefetch
pipeline was implemented and swept. At its best configuration it improved the
suite total by 0.58% — a genuine win, not a regression. It was reverted anyway,
on two grounds:

1. **Cost/benefit.** A cross-sequence fetch cursor plus N-slot semaphore and
   update-id bookkeeping is a large permanent complexity increase for half a
   percent. A deeper variant of the same idea ran out of scratchpad memory and
   regressed another case outright.
2. **It refuted its own hypothesis.** The change was motivated by "the residual
   is insufficient prefetch depth." It moved exposed time from 51 µs to ~38 µs —
   nowhere near closing it. The model was wrong.

The second reason matters more than the first. A sub-1% result is not just a
small win; it is *evidence against the theory that produced it*.

*Counter-case.* An earlier experiment in the same session returned 0.4% and was
discarded in four minutes. Had it returned 15%, work would have continued — at
the same 83% of floor. The percentage of floor never triggered anything.

---

## 4. Criterion C — marginal return governs, threshold does not

The floor's job is **not** to supply a stop threshold. Its job is to **bound how
much is left**, so that a small experimental return can be judged against a known
remaining prize.

- 0.58% return with 51 µs known to be exposed ⇒ the remaining prize is small and
  this attack barely dents it ⇒ stop.
- 0.58% return with 300 µs known to be exposed ⇒ the prize is large and you have
  simply picked the wrong attack ⇒ keep going, change hypothesis.

Same experimental result, opposite decision. That is why the floor is measured
and why no fixed percentage is applied mechanically.

**If you want a testable rule instead of a judgement call**, declare it *before*
optimizing — e.g. "stop at ≥85% of `max(compute-only, transfer-only)` on every
case carrying >5% of suite time" — and record it. Then stopping is a test. Just
be aware you are trading away the ability to keep going when a large prize is
still visible.

---

## 5. What does *not* count as a stop signal

- Reaching a percentage of floor without having run experiments that failed.
- A single failed experiment.
- Multiple failures that all target the same suspected cause.
- Diminishing returns on the heaviest cases only. **This is the common trap:**
  diagnosis narrows to the two or three cases with the largest absolute times,
  and the small cases quietly drop out of the loop. If your scoring metric is a
  mean of per-case speedups rather than a total, a stop justified on the heavy
  cases can be badly premature — in one instance the four smallest cases held
  ~12% of suite time at 10.8× their bandwidth floor, and nothing had re-examined
  them for four hours.

---

## 6. Before you stop

1. **Gap-check every case carrying >5% of total time**, not just the heaviest.
   Floors differ by regime; latency-bound cases have floors far above their
   bandwidth floor, and you cannot know how much of their headroom is real
   without measuring.
2. **Re-verify correctness at the final configuration**, on every case, on every
   output the kernel produces.
3. **Name and quantify the residual in the deliverable** — exposed overlap time
   per case, plus any overhead outside the kernel proper. The next person should
   start from a map, not a dead end.
4. **List the hypotheses you never tested.** Diminishing returns is not a proof
   of optimality. If the leading explanation for the residual was never
   experimentally attacked, say so plainly rather than asserting the kernel is
   at its limit.
