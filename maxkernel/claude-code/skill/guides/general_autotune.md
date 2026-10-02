# Autotuning a kernel — when to sweep, what to sweep, how to sweep

Instructions for **hand-directed grid search**: a human or agent choosing a
handful of points from a cost model. §6 says when to reach for a real autotuner
instead.

Applies to any kernel with tunable block sizes, tile shapes, unroll factors,
buffer counts or similar.

---

## 1. When to autotune

> **The single most important negative finding: a measurement of the performance
> floor never motivates a tuning sweep.**

Tuning is not how you close a gap between achieved time and the floor. Every
genuine trigger below is about a **stale or invalidated optimum** — a parameter
whose correct value changed and nobody re-derived it — not about a gap.

### The eight triggers

**T1 — You inherited defaults tuned for different code.**
*Tell:* block sizes come from a lookup table, heuristic or hard-coded constant
that predates your change. Nothing fails; nothing even looks wrong.
*Why:* those defaults encode someone else's cost model, and your change
invalidated it silently.
*Sweep:* the parameters the table supplies, on shapes where they are free,
starting from the table's own values as the control.
*Note the signature:* no error, no regression — just a worse operating point.
This trigger must be checked deliberately, because it never announces itself.

**T2 — A change altered what a parameter *means*.**
*Tell:* ask of every change — *does this remove or add a cost that scales with a
tunable parameter?* If the parameter was balancing two opposing costs and you
eliminated one, it is now doing a different job.
*Why:* the previous optimum solved a trade-off that no longer exists, and the new
optimum can lie **outside the range you previously searched**.
*Sweep:* a line scan over that parameter alone, **including values you previously
ruled out**.

*Representative example — the strongest argument against reusing a grid.* One
configuration measured **1702.5 µs**, roughly 4.6× worse than its neighbours, and
was written off. A later change decoupled the two concerns that parameter had been
arbitrating. The *identical* configuration then measured **303.7 µs** — the best
point in its grid. A 5.6× swing on the same two numbers with no hardware change.
A grid that excluded it as "known bad" would have missed the global optimum.

**T3 — A change moved a resource budget.**
*Tell:* you altered consumption of a budgeted resource (scratch memory, shared
memory, registers, occupancy) and some *other* parameter is derived from that
budget.
*Why:* the feasible region moved, and a derived parameter may have been silently
re-clamped without you asking.
*Sweep:* re-establish the feasible boundary first (include one expected-failure
point), then re-sweep with anything budget-derived **pinned explicitly**.

**T4 — One shape regressed while the rest improved, and ablation found no
culprit.**
*Tell:* a global change helps most shapes and hurts at least one; splitting it
into sub-changes, no single part accounts for the regression.
*Why:* configuration is the only thing that varies per shape. A shape-specific
regression with no code-level cause is usually a configuration mismatch under the
new cost structure.
*Sweep:* the regressed shape and its nearest twin, over the parameter tied to
whatever makes that shape unusual.
*Test it as a configuration problem before debugging it as a code problem.* The
fastest discriminator is a two-point sweep.

**T5 — Before you accept or reject a candidate variant.**
*Tell:* you are about to judge a structural change on measured time, and the
incumbent has had tuning attention the challenger has not.
*Why:* the unfairness is as large as your typical tuning gain, and it is
invisible in the result.
*Sweep:* two or three points on the variant, on the shapes that drive your
metric, **before** the accept/reject call.

*Representative example.* At its own default a challenger measured **8% worse**
than the incumbent — an apparently clear reject. Tuned at a matched
configuration, and summed across all shapes at each one's best point, it measured
**0.58% better**. The decision was still to reject, but the reason changed from
"it is slower" to "it is marginally faster and not worth the complexity." Only
the second is a defensible engineering judgement, and only the tuned comparison
supports it.

**T6 — A schedule probe came back confounded.**
*Tell:* you perturbed something that should only affect *when* work happens, and
something affecting *how much* work happens moved with it.
*Why:* a schedule probe's inference requires work held constant. A confounded
result measures an unknown mixture.
*Sweep:* the probe repeated at several values of the parameter that moved, so a
matched-configuration comparison can be read off.
*Operational test:* does the probe leave your stage-isolated timings unchanged?
If both hold and the total moved, the difference is overlap.

**T7 — Before generalizing scattered results into a rule.**
*Tell:* you have optima from several sweeps run on different grids or different
shape subsets, and you want to write a policy.
*Why:* optima measured on inconsistent grids are not comparable; a rule fitted to
them is fitted partly to the differences between your experiments.
*Sweep:* one consistent grid across every affected shape, plus **at least one
shape whose structure opposes the trend**.

**T8 — As an early hypothesis test, before any code change.**
*Tell:* your cost model predicts a parameter is badly set, and two points would
confirm or refute it in minutes.
*Why:* a sweep is a cheap way to test a *theory about where time goes*, even with
no intention of adopting a value.
*Sweep:* two or three points that would move sharply if the theory held, ordered
so the most informative comparison runs first — a timeout that kills the run
partway should still leave you with the answer.

### The four anti-triggers

**N1 — Do not tune to close a gap against the floor.**
If your problem is `actual − max(stage-isolated times)`, that is overlap failure.
Probe the overlap — depth, issue pattern, synchronization, contention — not the
block sizes.

*Representative example.* Two shapes with identical geometry differing only in
operand precision. A full block-size grid moved one by **1%** and the other by
**27%**, while both carried gaps of 40–95 µs throughout. One was already near its
transfer floor so there was nothing for a configuration change to recover; the
other simply had a bad default. **In neither case did tuning address the gap** —
and you cannot predict from the shape which situation you are in, which argues
for probing cheaply rather than assuming tuning will pay.

**N2 — Do not adopt tuned values while the structure is still moving.**
Tune early to *test* hypotheses; tune late to *choose* values. One parameter's
best value went large → small → large again across three structural changes in a
single session. The cost of adopting early is not just a wrong value: every
subsequent measurement is taken at a wrong operating point, so your diagnosis of
what to change next is distorted too.

**N3 — Do not sweep a pinned parameter, or a variant you already rejected.**
A parameter clamped by the shape compiles the identical kernel at every grid
point. And a rejected variant did not change the kernel, so the previous optimum
still stands.

**N4 — Do not use tuning as a substitute for diagnosis.**
Tuning searches *within* a design; diagnosis changes the design. In one session
tuning was worth ~1.15× on the configuration-sensitive shapes, while structural
changes found by ablation and microbenchmarks were worth 1.74× overall and
2.7–2.8× on shapes whose parameters were pinned and which tuning could not touch
at all. If you do not yet know *why* the kernel is slow, a sweep will find a
local improvement and you will stop there — at a point a redesign would have
beaten by 2×.

### Checklist

Sweep when **any** holds:

```
[ ] defaults came from a table or heuristic written for different code
[ ] a change removed or added a cost that scales with a tunable parameter
[ ] a change moved a resource budget, so the feasible region shifted
[ ] one shape regressed and ablation found no single culprit
[ ] you are about to accept or reject a variant on its measured time
[ ] a schedule probe came back confounded by a parameter moving
[ ] you are about to fit a rule to optima gathered on inconsistent grids
[ ] a two-point sweep would test a specific directional theory
```

Do **not** sweep — or do not *adopt* the result — when:

```
[ ] you are trying to close a gap against the floor (probe the overlap instead)
[ ] the structure is still changing (test, but do not adopt)
[ ] every parameter is pinned for the shapes in question
[ ] you have not yet diagnosed why the kernel is slow
```

---

## 2. Which cases to sweep

> **Sweep the fewest cases that can distinguish the choices you are deciding
> between. Verify on all of them.**

Exploration and verification are different activities with different case sets.
Exploratory sweeps want 2–4 cases; full-suite runs happen before anything ships.

**2.1 Exclude a case if the parameter is pinned by its shape. Do this first.**
Evaluate the clamping expression the kernel actually applies, per case. If a
parameter can only take one value, sweeping it re-measures an identical compiled
kernel under different labels. Cost is usually *not* the reason to exclude —
duplication is.

**2.2 Sweep variants of the same shape together.** If two cases differ only in a
dtype, flag or precision, sweep both. The pair is what tells you whether the
optimum is shape-driven or dtype-driven; one alone silently assumes it
generalizes. This frequently overturns a prior that treated them as different.

**2.3 Include the cases that dominate your scoring metric — but state the metric
first.** If the score is a **sum** of times, sweep the cases carrying most of it.
If it is a **mean of per-case speedups**, small cases count equally and this rule
inverts. Decide before you pick cases, not after.

**2.4 Always include a case that could plausibly regress — the control.** The
least obvious rule and the highest-value one. A sweep over your biggest cases will
always find a value that helps them; without a case where that value plausibly
*hurts*, you will fit a rule to what you measured and regress what you did not.
Pick a shape whose structure opposes the expected trend.

*Representative example.* Four large shapes all wanted a bigger query block. One
short-sequence shape, swept over the same values, measured **36.5 / 38.7 / 42.6
µs** — monotonically **worse** in exactly the direction the others preferred,
because a query block is processed at its full static size and this shape's work
did not fill one. A rule fitted to the first four alone would have maximized the
parameter and regressed the fifth.

**2.5 If the parameter consumes a budgeted resource, include a case with a
different resource profile.** Compile-time resource failures **do not track
problem size monotonically** — the case that fails first is often not the biggest,
because what matters is the ratio of the resource estimate to the limit, not the
amount of data.

**Anti-patterns.** Sweeping every case by default (duplicates on pinned
parameters). Sweeping only the biggest case (no control, overfit rule). Changing
the case set between rounds of the same question (optima become incomparable and
you pay for a reconciliation round). Exploring and verifying with the same set.

---

## 3. Choosing the search space

> **Bound the space by the problem and the hardware, start from the existing
> default, and spend your points where the cost model says the curve bends.**

A grid is not a search; it is a set of hypotheses about where the optimum lies.
Each point should be there for a reason you can state.

### 3.0 First, inventory the knobs

Before choosing values, classify every parameter by **how you would reach it**.
The three classes have different sweep costs and different machinery.

| class | how it is swept | cost per point |
|---|---|---|
| **exposed through the public API** | pass the value per call; the harness overrides the default | one compilation per distinct combination |
| **source constants** | generate one variant module per value | one compilation, plus a file |
| **derived or structural** | **not swept** — see below | n/a |

Two consequences worth planning around:

- Every distinct combination of API knobs is usually a **separate compilation**.
  That, not execution, is what makes grids expensive.
- Source constants are cheap to *try* and expensive to *keep*. A constant that
  measures flat across its plausible range should stay a constant — see the
  "reject the knob" outcome in §4.

**Record what you deliberately do not tune, and why.** Typical exclusions:
parameters fixed by a data layout, parameters derived from another knob, and the
synchronization structure itself. Writing the exclusion list down prevents both
wasted sweeps and the later question "did anyone ever check this?"

### 3.0b State each knob's trade-off before sweeping it

For every knob you will sweep, write one line naming the **two opposing
pressures** it mediates:

> *large ⇒ longer transfers, better bandwidth, but the tail unit is processed at
> full width so padding is wasted; small ⇒ less waste, more issue overhead,
> shallower pipeline.*

This is not documentation. It is what tells you which direction to walk, where
the curve should bend, and — most importantly — **which later code change will
invalidate the optimum** (T2). A knob whose trade-off you cannot state is a knob
whose results you will not be able to interpret.

**3.1 Start at the current default and walk outward.** The existing value is
someone's prior measurement, and including it gives you a control measured under
identical conditions.

**3.2 Bound each parameter by the shape, not by round numbers.** Compute the
largest meaningful value from the problem itself. Anything beyond it is either
clamped — a duplicate measurement — or invalid.

**3.3 Let failures bound the space; they are free information.** Deliberately
include one point you expect to fail. A resource error is a hard upper bound
obtained in seconds, and more reliable than your estimate of the budget.
*Put the expected-failure point last* — per-configuration exceptions are usually
caught and reported, but a whole-run timeout is not.

**3.4 Match the design to the question.**

| question | design |
|---|---|
| which of two parameters matters, and do they interact? | **2×2 factorial** + the default |
| one parameter's meaning changed; where is its optimum now? | **line scan**, others held fixed |
| is the optimum on a boundary? | **edge probes**, after the interior settles |
| is a source constant worth exposing as a knob? | **one value per variant**, coarse |

A factorial separates the parameters; a line scan assumes they are separable and
is valid only once you have established that, or when only one has changed.

**3.5 Prefer values that divide the natural block structure.** Non-divisors
introduce padding, padding changes the *work*, and that confounds a schedule
measurement with a work measurement.

**3.6 Size the grid to your measurement budget.** Know two numbers before writing
the grid: the **fixed cost per process** and the **marginal cost per
measurement**. Then the largest grid that fits your timeout is arithmetic. Get
both by regressing total time against measurement count across two runs.

**3.7 Re-derive the space after any structural change — never reuse a grid.** An
optimum can move further than a previous grid's entire span (see T2).

**Anti-patterns.** A dense grid before a coarse one — go coarse, find the bend,
then refine. Omitting the default, so you cannot tell whether you improved
anything. Sweeping a parameter whose effect you have not reasoned about: if you
cannot say what a result would *mean*, do not measure it. Sweeping a resource
parameter without checking what else it moves.

---

## 4. Running a round

Tuning is not a phase at the end. It is a sequence of **rounds**, each one
triggered by something that happened — and the number of rounds you end up
running is simply the number of times your cost model moved. It is not a plan.

### Anatomy

Record all five parts of every round. The first two are what make the result
reusable; without them you have numbers nobody can re-interpret later.

| part | content |
|---|---|
| **trigger** | which condition from §1 fired, and why now |
| **question** | the one thing this round must answer, in a sentence |
| **grid** | the points, and the cases, with the reason for each |
| **decision** | adopt / write a rule / reject the knob / refute a hypothesis / reject a variant |
| **cost** | wall-clock, so the next round can be sized |

### Five outcomes, not one

A round does not have to produce a value. All of these are complete results:

1. **Adopt a value** — the ordinary case.
2. **Write a rule** — when different shapes want different values, the output is
   a policy, not a constant.
3. **Reject the knob** — the parameter measures flat across its plausible range,
   so it should stay a hard-coded constant and never be exposed. A knob you can
   delete is worth as much as one you can tune.
4. **Refute a hypothesis** — the round was a theory test (T8) and the theory was
   wrong. This is a normal and cheap outcome.
5. **Reject a variant** — tuning showed a candidate is not worth its complexity
   even at its best configuration (T5).

### Practical rules

**Order the grid so a killed run still answers the question.** Sweeps hit
timeouts. If the first two points are the ones that discriminate, a run that
completes a quarter of its grid has still done its job. Put the expected-failure
point last and the decisive comparison first.

**Batch a round into one process.** Startup cost — framework init, input
construction, compilation cache warmup — is typically a large fraction of a
sweep's wall time and is paid once per process, not once per measurement. Shell
loops that relaunch per configuration multiply it by the grid size.

**Treat the first run of any new sweep harness as a harness test, not a
measurement.** Sweeps exercise a code path that single runs do not: repeated
invocation with varying configuration against shared state. Buffer donation,
cached compilation, and mutated inputs all fail here and nowhere else — a sweep
whose first configuration succeeds and whose remaining ones all fail identically
is the signature. Look at the failure pattern before debugging the kernel.

**Record catastrophic points; do not discard them.** A configuration that
measures several times worse than its neighbours marks a **cliff** — a resource
threshold crossed, a buffer spilled. Cliffs move when the code changes, and a
point on the wrong side of one today can be the global optimum after a change
that shrinks the resource in question. Write the outlier down with its
explanation, and revisit it after any change that touches that resource.

**Verify that a rule reproduces the sweep's own optima.** When the output is a
policy rather than a value, implementing it is not the last step: re-run every
affected case through the default path and check the policy selects — and
achieves — the hand-picked optimum on each. A rule that is right on the cases you
fitted it to and wrong on the one you did not is the normal failure.

**Skipping a triggered round is not free.** The cost of not re-tuning after a
change that moved an optimum is the full distance between the stale value and the
new one, and it compounds: every subsequent measurement is taken at a wrong
operating point, so the diagnosis of what to change next is distorted too.

---

## 5. Procedure

0. **Name the trigger** (§1) and write the question this round must answer.
1. **State the scoring metric.** Sum of times, mean of speedups, worst case? This
   decides §2.3 before anything else does.
2. **Inventory the knobs** (§3.0): which are API-exposed, which are source
   constants, which you are deliberately not tuning. State each swept knob's
   trade-off in one line.
3. **Tabulate the free parameters per case.** Evaluate each clamping expression.
   Drop cases where everything is pinned; note cases where only one knob is free.
4. **Pick the case set:** dominant cases, in shape-matched variant pairs, plus at
   least one control that could regress, plus one with a different resource
   profile if a budgeted resource is involved.
5. **Measure the fixed and marginal cost** of one measurement; compute how many
   points fit your timeout.
6. **Build the grid:** default first, bounded by shape, factorial to separate
   parameters or line scan to refine one, decisive comparison early,
   expected-failure point last.
7. **Run, read, decide** — one of the five outcomes in §4. Record any
   catastrophic point with its explanation.
8. **Verify.** At the adopted values, on all cases including those you did not
   tune on. **If the output is a rule, verify the rule reproduces every optimum
   the sweep found**, through the default path.
9. **Record the round:** trigger, question, grid, decision, cost.
10. **After any structural change, return to step 0.** Free ranges, costs, the
    optimum and the position of any cliff may all have moved.

---

## 6. When to use a real autotuner instead

The above describes hand-picked grid search, 2–5 points per parameter per round,
aimed by a cost model. It suits slow compilation, a measurement loop of minutes,
and a model that predicts where the optimum should be.

**Use a real autotuner when:**

- the full cross product is affordable — check this explicitly rather than
  assuming, since a modest two-parameter product over a suite is often an hour;
- you have no cost model, so there is nowhere principled to aim;
- the space has more than two or three interacting dimensions, where factorial
  designs stop being cheap;
- you will retune often, so a compilation cache and search harness amortize.

**What hand-directed search predictably misses:** per-case optima that a single
global rule cannot express. A point found by an edge probe may beat the shipped
configuration on one shape while regressing or failing on others, and get rejected
for that reason — where a per-case policy would have captured it. If several
shapes each want a different value, that is a signal your output should be a
policy, not a constant.
