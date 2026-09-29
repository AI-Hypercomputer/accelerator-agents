# From measurement to hypothesis

How each measurement produces the next one. Applies to any kernel optimization.

Instruments are covered elsewhere — attribution, floors, microbenchmarks, gap
probes. This document is about the reasoning that connects them: given a number,
what is the next experiment, and why.

---

## 1. The core rule

> **The hypothesis lives in the discrepancy, not in the magnitude.**

A localized cost is not actionable on its own. Compare:

- *"This matmul costs 175 µs."* Suggests only "do fewer matmuls" — which, when
  the volume is already minimal, is the wrong fix.
- *"This matmul costs 175 µs where the identical shape costs 76 µs standing
  alone."* Now there is a ~100 µs discrepancy that must have a cause, and the
  search for that cause is a real experiment with a real outcome.

So: **every time a measurement localizes a cost, the next step is to establish
what that cost *ought* to be.** Attribution never closes a line of reasoning by
itself; it hands you a magnitude, and magnitudes do not imply actions.

Three sources of "ought", in increasing cost:

1. **An analytic bound** — bytes ÷ bandwidth, FLOPs ÷ peak, one pass over the
   data. Free.
2. **A microbenchmark of the same shape in isolation.** Minutes.
3. **An earlier measurement of the same quantity under different conditions.**
   Free if you kept it — which is a reason to keep it.

**The absence of a discrepancy is equally informative**, and is often the more
valuable outcome. See §3.5.

---

## 2. Match the metric to the question

Using the wrong instrument produces an answer that looks fine and is useless.

| question | instrument | unit |
|---|---|---|
| *what is this piece of code worth?* | delta between ablation variants | µs |
| *is this cost anomalous or intrinsic?* | the same primitive measured in isolation | a **rate** — elem/s, bytes/s, FLOP/s |
| *which of these formulations is fastest?* | a bake-off over candidate implementations | time per inner-loop iteration |
| *is this cost inside or outside the kernel?* | per-operation device time | µs per op |
| *how much room is left?* | the two floor probes | µs, and the gap |
| *why isn't the overlap working?* | a gap probe, work held constant | movement in the gap |

Note the second row is the only one that yields a **rate** rather than a time.
That is not cosmetic: a rate is what lets you multiply back out against a
different workload and get an expected cost, which is what makes discrepancy
detection possible at all.

---

## 3. The reasoning moves

### 3.1 Validate the instrument before reasoning from it

**Form.** *Can this metric resolve the effect I am about to chase?*

**How.** Time a null workload through the same path. Compare against the
magnitudes you expect to be discriminating between.

**Output.** Either confidence, or the knowledge that whole classes of case are
invisible to your current setup.

**Failure mode, and it is common:** discovering the metric is inadequate and
responding by *building* a replacement, without first checking whether the
environment already provides one. Inventory your measurement tooling before
writing any.

### 3.2 Bisect into halves and compare

**Form.** *Which half of this kernel binds?*

**How.** Two probes — one with all data movement disabled, one with all compute
disabled. Compare their absolute times.

**Output.** A direction, and an explicit *prohibition*. The prohibition is worth
as much as the direction: knowing the kernel is compute-bound is what justifies
not touching the fetch path for the next several hours, and the discipline to
honour that is where the time savings come from.

**Do this first.** It is two experiments and it decides what the next twenty are
about.

### 3.3 Chase the largest unexplained residual

**Form.** *N µs is still unaccounted for. What can occupy it?*

**How.** Subtract what you have priced from the half that binds. Look at what
remains, enumerate the candidates that could plausibly fill it, and ablate them.
Recurse.

**Output.** A progressively finer breakdown, each round targeting the biggest
number you cannot yet explain.

**Stop when** the residual falls below the size of things you are already working
on. Splitting further produces numbers nobody will act on.

### 3.4 Price the localized cost against a reference — the hinge

**Form.** *Is this cost anomalous or intrinsic?*

**How.** Take the largest attributed component and measure the same shape in
isolation (§1, source 2). Multiply the isolated rate back out against the real
workload. Divide.

**Output.** A ratio. The ratio, not the magnitude, is what you act on.

**Reading:**

| ratio | meaning | next |
|---|---|---|
| ≈ 1× | the cost is intrinsic | reduce the work, or accept it |
| 2–3× | the formulation is wrong | find what is different *in situ* — operand layout, dependency structure, surrounding code |
| ≫ 5× | something structural is badly broken | look for serialization, spills, or a pathological access pattern |

*Representative example.* An in-situ matmul priced at 175 µs; the identical shape
in isolation implied ~76 µs. A 2.3× ratio meant the volume was fine and the
formulation was not — the search that followed found many short, serially
dependent chains where one long one would do, and collapsing them was the largest
single win of the effort. The magnitude alone would have pointed at reducing
matmul count, which would have achieved nothing.

### 3.5 Read the absence of a discrepancy

**Form.** *The ideal case is barely better than what I have.*

**How.** Include an idealized control in any bake-off — the variant where the
thing you are considering optimizing costs literally nothing.

**Output.** An upper bound on an entire optimization direction, in one row.

**This is the highest-value move in the set**, because it kills work rather than
creating it. If the control is within a few percent of the status quo, the
component is not the bottleneck no matter how large its attributed cost was, and
the real cause is something common to every variant — usually the structure
around them.

*Representative example.* A data-unpacking step carried a large attributed cost.
A bake-off included a control where the operands arrived pre-split, costing
nothing at all: it beat the existing implementation by ~5%. That ruled out the
entire layout direction and redirected attention to the loop's dependency
structure, which is where the win actually was.

### 3.6 Compare the same quantity across configurations

**Form.** *This number moved, and the only thing I changed was a parameter.*

**How.** Keep your floor and attribution measurements, tagged with the
configuration they were taken at. When you re-measure, diff against the old
value rather than reading the new one in isolation.

**Output.** Detection of **coupled knobs** — parameters under two opposing
pressures.

*Representative example.* A transfer-only floor measured at two block sizes came
out worse at the smaller one, while compute came out better. One parameter was
being asked to satisfy two opposing constraints: large blocks transfer
efficiently but waste compute on padding; small blocks do the reverse. The
hypothesis that follows is not "pick a better value" but **"split the knob"** —
decouple the two granularities so each can be set independently. That
restructuring was the last structural win of the effort, and it came entirely
from diffing two measurements of the same probe.

### 3.7 When a prediction fails, name and quantify the missing term

**Form.** *The microbenchmark said this would win. It regressed.*

**How.** Do not discard the model. Work out what it failed to simulate, then
quantify that term from the regression itself: divide the unexplained time by
the count of whatever the change multiplied.

**Output.** A reusable constant — "≈63 ns per request issue" — that constrains
every subsequent design decision.

A prediction failure is an opportunity to extend the cost model. Treat the
disappointment as the least interesting part of the result.

---

## 4. Refutation is progress

Most links in a good reasoning chain are refutations. Four distinct kinds, each
with its own follow-up:

**A cheap hypothesis refuted cheaply.** A theory about padding waste, tested with
a parameter sweep in ten minutes, comes back backwards — smaller blocks are
*worse*. Follow-up: stop theorizing in that direction and go measure something.

**A search for a culprit that finds none.** A regression is split into its
component sub-changes, each is ablated, and each accounts for only a fraction.
The cause is diffuse. Follow-up: **change the class of explanation.** If no code
change explains it, suspect the *parameters* — a tuning table calibrated for the
old cost model is now stale. An ablation that fails to find a culprit is a useful
result precisely because it redirects.

**Eliminating the cheaper explanation first.** Before committing to an expensive
restructuring, test the cheap alternative explanation for the same symptom. A
few minutes spent refuting "the issue path is too slow" is what licenses spending
an hour on the structural fix.

**The accumulation itself.** Two or three consecutive experiments returning ≤1%
or negative is the stop signal. Check independence first: failures against one
hypothesis refute that hypothesis; failures against distinct hypotheses suggest
the queue is empty.

---

## 5. Never carry a classification forward

Any label derived from a measurement — *compute-bound*, *the bottleneck is the
unpack*, *this cause is refuted* — describes **one kernel state**. It expires the
moment the kernel changes.

The binding half in particular flips, and it flips for non-obvious reasons: a
pure tuning-parameter change that touches neither the arithmetic nor the transfer
code can reduce compute enough to hand the constraint to the other side. Trusting
a label taken before the previous change means aiming the next one at the wrong
half.

**Re-measure the cheap classifications after every kept change.** For the
expensive ones — a full attribution breakdown — either re-run them or tag the
result with the kernel version it describes, so it is not quoted as current.

---

## 6. Audit your own chain

Four weaknesses to look for, all of which are easier to see in someone else's
reasoning than in your own:

- **A conclusion resting on a single comparison.** If one number carried a
  redirect, it deserved a second measurement.
- **A conclusion resting on a model later shown to be incomplete.** When a
  microbenchmark is found to omit a first-order effect, every earlier conclusion
  drawn from it inherits the doubt. Go back and re-check them; the omission does
  not politely confine itself to the result that exposed it.
- **A residual attributed to a mechanism that was never probed.** The most common
  form of a sentence that reads like a finding and is actually an assertion. Name
  the mechanisms you tested and the ones you did not.
- **A breakdown that stopped being refreshed.** If every late decision used floors
  and A/Bs while the last attribution is several rewrites old, then the internal
  composition of the current kernel is simply unknown — and any statement about
  where its time goes is being read off a kernel that no longer exists.
