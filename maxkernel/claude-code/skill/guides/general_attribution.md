# Attribution — working out where a kernel's time goes

> **Attribution** means working out which slice of a kernel's runtime belongs to
> which piece of its code: switch one piece off, re-time, and the drop tells you
> what that piece was costing.

Applies to any kernel on any accelerator.

**Trigger:** *"I do not know where the time goes."* Run it on the baseline before
changing anything. Its output is the map every later decision is steered by.

---

## 1. First, name the stages

Write down the stages of your own inner loop before building anything, so each
ablation has an unambiguous target. **Derive that list from your kernel — there
is no canonical stage list, and the one below is an illustration, not a
template.** It happens to be the decomposition of one blocked attention kernel:

1. **copy** a block from main memory into on-chip memory
2. **unpack** operands — deinterleave, split, or gather the layout the compute
   needs out of the layout memory holds
3. **neutralize the tail** — zero or mask the padding the last block carries
4. **matmul #1**
5. **mask + reduction** (softmax, normalization, row arithmetic)
6. **matmul #2**
7. **accumulate** into running totals

plus an **epilogue** (finalize and store) and the **grid loop** around everything.

A different kernel decomposes differently and the stage count is not meaningful:
a stencil might be halo exchange / interior update / boundary fixup; a sort,
partition / local sort / merge; a sparse kernel, index gather / scatter-accumulate
with no dense matmul anywhere. What generalizes is the *method* — enumerate
whatever stages your loop actually has, then remove them one at a time.

Two properties the list must have, whatever the kernel:

- **Every stage is separately switchable** without disturbing the others' shapes.
  If two stages can only be removed together, they are one stage for attribution
  purposes; say so rather than pretending to a finer split than you can measure.
- **The stages cover the loop body.** Anything not on the list shows up as
  unexplained residual in Step 4, which is fine — but you should be able to name
  it once it does.

**One ablation per stage.** Names should say what was removed; be consistent
about it, because "no-transfer" (transfers gone, compute remains) and
"transfer-only" (compute gone, transfers remain) are opposite conventions and
mixing them in one naming scheme guarantees a misread later.

---

## 2. Four design rules

Each exists because of a specific failure mode.

1. **Remove work, preserve shape and consumer.** Do not delete a matmul —
   *substitute an expression producing an identically shaped array of the same
   dtype*, and make it still consume its inputs so they stay live. Deleting it
   outright lets the compiler dead-code-eliminate the whole downstream chain, and
   you measure the elimination rather than the matmul.
2. **One variable per variant, with explicit combinations.** Build
   `A`, `A+B`, `A+B+C` as separate variants so every delta is attributable to one
   change.
3. **Assert the patch applied.** If you generate variants by source-text
   replacement, assert the text actually changed. A patch that silently fails to
   apply produces a null result *indistinguishable from a refuted hypothesis*.
4. **Expect wrong numerics.** See §5.

---

## 3. The procedure

### Step 1 — Split into two halves and classify

Build two variants of the *whole* kernel:

- **compute-only** — every transfer becomes a no-op; the kernel computes on
  whatever is already resident.
- **transfer-only** — an early return after the transfer issue and wait; every
  transfer happens, nothing is computed.

Compare. **This is usually the single most consequential measurement you will
take**, because it decides which half of the kernel is worth subdividing.

*Example.* Against a 495 µs kernel: transfer-only 294 µs, compute-only 441 µs.
441 > 294 ⇒ **compute-bound, not memory-bound** ⇒ leave the fetch path alone and
attack the inner loop. Every subsequent decision followed from that one
inequality.

### Step 2 — Subdivide the binding half, on a confound-free base

Do not price components against the full kernel: transfer/compute overlap
confounds the delta. Re-measure each component **on top of the compute-only
variant**, so overlap is out of the comparison.

This matters quantitatively. The same component measured against the full kernel
and against the no-transfer base gave different numbers; only the second is a
clean read.

### Step 3 — Chain cumulatively

Each variant adds one removal to the previous one, producing a subtraction chain:

```
full kernel                       494.9
  transfer only                   294.1
  compute only                    441.1   ⇒ compute-bound
    − tail masking                413.8   ⇒ mask      27.4
    − unpack/gather               316.2   ⇒ unpack    97.6
    − row-position arithmetic     313.3   ⇒ span       2.9
    − matmul #1                   141.4   ⇒ MM1      174.8
    − matmul #2                   179.6   ⇒ MM2      136.6
    − both matmuls                103.8   ⇒ both     212.4
```

### Step 4 — Keep subdividing the largest unexplained residual

That is the whole loop: measure, subtract, look at what is left, split it again.
Each round targets the biggest number you cannot yet account for.

### Step 5 — Stop when the residual is smaller than what you are already working on

In the chain above, subdivision stopped at ~104 µs unexplained, which split into
~2.5 µs of loop overhead and ~12 µs of epilogue, leaving ~89 µs that was already
the target of a planned redesign. Splitting further would have produced numbers
nobody would act on.

### Step 6 — Two inverted probes for fixed overhead

Most ablations say *"remove this piece."* Add two that say *"keep only this
piece"*:

- **skeleton only** — the grid loop runs, the per-iteration body returns
  immediately. Prices loop overhead.
- **no inner loop** — the innermost loop gets a zero trip count; the outer loop,
  epilogue and stores all still run. Prices the epilogue.

Report these as **absolute times, not deltas**, and be aware they are usually
small enough that a coarse timer cannot resolve them — measure them with your
finest instrument.

*Example.* Loop overhead over 65 iterations came to 2.5 µs — not a target, and
knowing that closed off a whole line of speculation. The epilogue came to ~12 µs,
small, but enough to motivate a change that shipped.

---

## 4. Reading the numbers

**Deltas do not sum, and that is expected.** In the chain above, the two matmuls
priced individually at 174.8 and 136.6 µs — 311.4 together — but removing both
saved only 212.4 µs. The 99 µs discrepancy is real overlap: the two are partly
pipelined against each other and against the reduction between them.

> **A single-ablation delta is an *upper bound* on that component's exclusive
> cost, not its share of a partition.** Use attribution for **ranking**, which is
> robust to overlap. Do not present the numbers as a budget that adds to 100%.

**Keep units consistent, and say when you cannot.** Components whose cost is near
your timing floor must be measured with an instrument that can resolve them.
Rows measured that way are not subtractable from rows measured with a coarser
one — mark them.

**State the direction of the stand-in's bias.** A stand-in cheaper than the real
operation over-attributes cost to the removed component; one that does real work
of its own under-attributes. Whichever you have, say which, so the number is
read as a bound rather than a point estimate.

---

## 5. Attribution variants skip correctness checks — by design

**Every attribution variant produces wrong results on purpose.** The compute-only
variant reads uninitialized memory; the unpack ablation hands the matmuls fake
operands; the matmul ablations return constants; the skeleton computes nothing.
A correctness check would fail on every one of them and carry **zero
information**. Run them with checking disabled.

**What must hold instead is that shapes and instruction mix stay
representative** — which is exactly what design rules 1–3 enforce. Correctness is
not being traded away; it is being replaced by a different invariant, and that
invariant is the one the rules protect.

**Correctness is checked on every change that can ship** — immediately after the
edit and before any timing of it, comparing *every* output the kernel produces
(not just the primary one — mutated caches and in-place buffers count) against a
reference at the task's tolerances. The two regimes are disjoint: throwaway
variants are never checked, shippable changes are always checked.

---

## 6. What attribution produces

Five kinds of output, all durable:

1. **A bound classification** — which half binds. Redirects everything.
2. **A ranking of components** by cost, robust to the non-summing problem.
3. **A ruler.** An ablation that removes a *known quantity* of elementary work
   converts a spec sheet into a calibrated rate. Removing two elementwise
   operations over the entire working set cost 26 µs; dividing by the element
   count yields the vector unit's real throughput *in your code, including issue
   overhead*, which no datasheet supplies. A roofline can then rest on it.
4. **Cheaply eliminated suspects.** Pricing a plausible culprit at 2.9 µs closes
   the question for the cost of one variant.
5. **Fixed-overhead floors** that bound how much any restructuring can possibly
   return.

**Budget for breakage.** An ablation is a code change and breaks like one. Two
recurring failures: the stand-in expression fails to compile (dtype or rank
mismatch in the substitute), and **removing a stage violates a resource
invariant** — skipping a body that was supposed to await outstanding transfers
can leave a semaphore non-zero and fault the device. Fix the second by building
the skeleton variant *on top of* the no-transfer variant, so there is nothing
left to await. Expect roughly one variant in three to need a second attempt.

---

## 7. The breakdown is not stable — re-attribute after structural change

The tempting assumption is that attribution is a one-time measurement: take the
breakdown once, reuse it forever. **It is false, and it fails in the direction
that hurts.**

After one structural rewrite of an inner loop, of six attributed components:

| component | still valid? |
|---|---|
| loop/grid overhead | yes |
| unpack | yes |
| tail masking | **no** — applied to fewer operands, different operator |
| both matmuls | **no** — batched across heads, entirely different cost profile |
| reduction and row arithmetic | **no** — one stacked computation instead of many |
| epilogue | **no** — sliced to the rows actually in use |

Four of six changed. A re-attribution would have produced a substantially
different table, and decisions made against the stale one were being made
against a kernel that no longer existed.

Two practical obstacles, worth planning around rather than discovering:

- **Cost.** A full attribution set is a significant fraction of an optimization
  session. Repeating it after every kept change is usually not affordable.
- **The patches rot.** Source-text replacements stop matching once the code they
  target is rewritten — and if you followed rule 3, they fail loudly rather than
  silently. Re-attributing is a partial *rewrite* of the ablation set, not a
  re-run of it.

**Practical rule:** re-attribute the stage you are still actively optimizing
after any rewrite that touches it; for the rest, mark the table stale with the
kernel version it was measured against. An attribution table quoted as current
when it is three rewrites old is worse than no table.

---

## 8. Checklist

```
[ ] inner-loop stages enumerated before any variant is built
[ ] naming convention states what was REMOVED, consistently
[ ] removed work replaced by a same-shape, same-dtype, input-consuming stand-in
[ ] one variable per variant; combinations built explicitly
[ ] patch application asserted
[ ] correctness checking disabled on ablation variants; enabled on every
    shippable change, covering every output
[ ] compute-only and transfer-only measured first; binding half identified
[ ] components re-priced on a confound-free base, not against the full kernel
[ ] subdivision follows the largest unexplained residual
[ ] subdivision stopped when the residual fell below what is already in progress
[ ] skeleton and no-inner-loop probes run, reported as absolute times
[ ] results presented as a ranking, with overlap stated; not as a 100% budget
[ ] stand-in bias direction stated
[ ] table tagged with the kernel version it describes
```
