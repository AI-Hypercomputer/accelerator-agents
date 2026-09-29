# Roofline analysis for a kernel — an operating instruction

**It is not a plot.** It is four divisions and one inference, per case. It runs
on CPU in under a second and costs no accelerator time beyond two or three
ablation runs you will want anyway. Its output is a per-case floor table, a
bound classification, and a numeric target you can still use as a stopping rule
hours later.

Do it **early** — before optimizing, not after. Its job is to tell you which
direction to work in, and it is cheapest at the moment that question is open.

---

## 1. Inputs

### 1.A From the problem definition

Derive, per case, the quantity the algorithm *must* move. The one non-obvious
step:

> **Count re-reads introduced by blocking, not just the logical data size.**

If the kernel re-streams a tile of operand data once per output block, its
traffic is multiplied:

```
reads = Σ_seq  ceil(work_len / block_size) × data_len
```

For a long-input case with `block_size = 64` and `work_len = 512`, that is an
**8× multiplier** on that sequence's traffic. Getting this wrong understates the
memory roof by the same factor — and noticing it immediately identifies
`block_size` as a tuning lever, before any tuning has been done.

Also record bytes and elements per unit of data separately:

```
bytes_per_token    = fields × dim × itemsize     # dtype-dependent
elements_per_token = fields × dim                # dtype-INdependent
```

Step 3 needs both, and the difference between them is the whole fp8 question.

### 1.B Hardware constants — TPU v6e

| symbol | value | status |
|---|---|---|
| HBM bandwidth | **1.6 TB/s** | spec; confirm later against your own achieved rate |
| MXU peak, bf16 | **918 TFLOP/s** | spec |
| MXU operand throughput at small output width | **~3.6 TB/s** | **measure it** — see step 3 |
| VMEM per core | **~128 MB** | readable off a real OOM error message |
| clock | **~1.74 GHz** | assumed; only used for a plausibility check |
| MXU geometry | 4 MXUs × 128 elem/cycle weight port | gives an a-priori operand estimate of ~890 G elem/s — **2× pessimistic in practice** |

Treat the first two as assumptions until validated (§6). They usually hold
within a few percent.

### 1.C One measured input

A single ablation delta from the hot loop, used in step 4. Remove one known
elementwise operation, measure the time saved. This is the only input that
cannot come from a spec sheet, and it is the load-bearing one.

---

## 2. The four roofs

### Step 1 — Memory roof

```
bytes  = reads × bytes_per_token
t_HBM  = bytes / 1.6e12
```

*Example.* 97,376 token-reads × 2048 elem × 2 B = **398.9 MB** → `/1.6e12` =
**249 µs**. The same case in fp8 moves half the bytes → **125 µs**.

### Step 2 — FLOP roof

```
t_FLOP = total_FLOPs / 918e12
```

*Example.* 4.83 GFLOP → **5.3 µs**, against a measured 495 µs.

**If this comes out under ~5% of measured time, say so explicitly and stop
treating the FLOP roof as relevant.** Most memory-bound and all latency-bound
kernels land here. The textbook roofline has now told you everything it can, and
steps 3 and 4 are the reason you are still reading.

### Step 3 — Operand-streaming roof (usually the one that binds)

Every operand byte must cross into the matrix unit even when only a handful of
output rows consume it. This ceiling is independent of FLOPs and does not appear
on a textbook roofline. For decode-shaped work it is typically **~20× more
restrictive than the FLOP roof**.

```
a priori (element-limited):  t = elements / 890e9
measured (byte-limited):     t = bytes    / 3.6e12
```

**Measure this, do not estimate it.** The a-priori weight-port model says the
port is *element*-limited, which implies a narrow dtype buys nothing. Direct
measurement on v6e says it is **byte**-limited, which implies a narrow dtype buys
a full 2×. The estimate was 2× pessimistic, and the element-vs-byte distinction
decides whether fp8 is worth carrying through the design.

*Example.* 398.9 MB / 3.6e12 = **110 µs** bf16; the fp8 variant of the same
workload has identical element count but half the bytes → **55 µs**.

### Step 4 — Vector-unit rate, calibrated from your own ablation

No datasheet gives you the vector rate *in your code*, including issue overhead.
Infer it:

```
data as vregs   = bytes / vreg_size            # v6e vreg = 8 × 128 × 4 B = 4096 B
ops removed     = n_ops × vregs
rate            = ops_removed / measured_delta
one full pass   = vregs / rate
```

*Example.* 398.9 MB → 97.4 K vregs. Removing two elementwise ops over all of it
saved 26 µs ⇒ 194.8 K ops / 26 µs = **7.5 G vreg-op/s** ⇒ at 1.74 GHz, **4.3
ops/cycle** (plausible for a 4-ALU VPU — this is the check that validates the
assumed clock) ⇒ **one full pass over this data = 13.0 µs**.

That one-pass figure is the most useful constant the whole analysis produces.

---

## 3. The inference — pass count

```
passes = measured_compute_only_time / one_pass_time
```

This is a **software** quantity: how many times the kernel actually touches each
element. It is clock-independent, so it survives a wrong clock assumption.

| passes | reading | action |
|---|---|---|
| 2–4 | near the floor for this structure | tune block sizes |
| ~10 | structure is lossy | look for redundant traversals |
| ≥30 | **the structure is wrong** | tuning is a waste of time; reformulate the inner loop |

*Example.* 441 µs of compute-only against a 13 µs one-pass figure = **34
passes**, for an inner loop that needs about four elementwise steps plus two
matrix passes. Combined with `compute-only > transfer-only`, this produced the
one-sentence direction the rest of the work executed against: *the kernel is
nowhere near any hardware roof and the transfer path is not the problem; the
inner loop is losing to latency and instruction overhead.*

---

## 4. Outputs

Build one table, all cases:

| case | reads | MB | **t_HBM** | **t_operand** | GFLOP | t_FLOP | intensity | measured |
|---|---|---|---|---|---|---|---|---|

Then classify each case by `measured / max(roofs)`:

| ratio | diagnosis |
|---|---|
| **1–2×** | the named roof binds; you are close to it |
| **2–6×** | real headroom in the compute structure |
| **≥20×** | **no roof binds — the case is latency-bound.** Roofline is the wrong instrument here; the cost is fixed per-grid-step overhead, not data movement |

*Two representative cases from one suite.* A heavy case: 398.9 MB, HBM roof
249 µs, measured 471 µs → **1.9×**, memory-bound and worth attacking. A tiny
case: 1.8 MB, HBM roof 1.1 µs, measured 40.2 µs → **36×**. The second number is
not a headroom estimate — it means the roofline says nothing at all about that
case, which is itself the finding. It was 16 sequences paying ~2.5 µs of fixed
cost each for ~0.07 µs of work.

Finally, turn `max(roofs)` per case into a **target range**, with slack above the
largest roof for imperfect overlap. Keep it. Hours later, when every remaining
idea is a 1% gamble, it is the only principled stopping rule you have.

---

## 5. Why the classical roofline is not enough

Machine balance (ridge point) on v6e is `918e12 / 1.6e12` = **574 FLOP/byte**.
Real kernels of this kind run 4–512 FLOP/byte, i.e. entirely left of the ridge,
so a textbook roofline labels every case "memory-bound" and stops. That label is
*true* and *useless*: it predicts 1.1 µs for a case that ran at 40.2 µs, and it
offers no axis for the thing that was actually wrong.

The two ceilings that matter are not on the standard plot:

1. **Operand streaming** (step 3) — bytes through the matrix unit at small output
   width, independent of FLOPs.
2. **Pass count** (step 3 inference) — a software roof. The roof is one pass; the
   kernel was doing 34.

---

## 6. Expected accuracy, and caveats

Validated against final measured floors, this method lands close: HBM roof
predictions came within **1.4%** and **4%** of the empirically measured
transfer-only floors, assumed HBM peak was confirmed at 99% of spec, and the
target ranges proved **5–10% optimistic**. That is the accuracy to expect —
good enough to steer, not to certify.

- **Nothing here models overlap.** Each roof is computed as if the machine did
  one thing at a time. `max(roofs)` is achievable only under perfect overlap,
  which is why targets are stated as ranges with slack.
- **The vector rate rests on a single ablation delta.** If that measurement is
  30% off, "34 passes" becomes 24 or 48. Frame the conclusion as an
  order-of-magnitude gap, never as a per-op budget.
- **Peak figures are prior knowledge until you confirm them.** Confirm HBM from
  your own achieved bytes/time, and the matrix peak from a direct measurement at
  a wide output dimension.
- **Traffic depends on block size**, which you will change. Recompute the table
  after any blocking change, or state which block size it assumes.
- **Compute the arithmetic intensity, do not eyeball it.** It is easy to be wrong
  by orders of magnitude on a quantity that looks obvious.

---

## 7. Recipe

1. Count bytes the algorithm must move, **including re-read multipliers from
   blocking**. ÷ HBM bandwidth → memory roof.
2. Count FLOPs. ÷ peak. If <5% of measured, declare the FLOP roof irrelevant and
   move on.
3. Count bytes that must cross the matrix unit. Estimate from the weight-port
   model, then **measure** — the estimate runs ~2× pessimistic, and the
   element-vs-byte question decides whether narrow dtypes pay.
4. Calibrate the vector unit from your own ablation: remove one elementwise op,
   measure the delta, divide by element count. Derive the one-pass time.
5. Compute pass count = compute-only time ÷ one-pass time. 2–4 → tune. ≥30 →
   reformulate.
6. Tabulate all cases, classify by `measured / max(roofs)`, and flag anything
   ≥20× as latency-bound rather than as headroom.
7. Turn the largest roof per case into a target range and keep it.
