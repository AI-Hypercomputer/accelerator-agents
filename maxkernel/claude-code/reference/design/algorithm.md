---
name: algorithm
description: >-
  Whether the algorithm itself should change: exact restructurings versus approximations and the
  accuracy gate each approximation owes, reassociation, online softmax, accumulator precision,
  moving work from VPU to MXU, primitives that do not lower, and batching independent axes. Read
  alongside the layout, pipelining and unit-placement files, not after them.
---

# Whether the algorithm itself should change

## Questions and probes

| Question | Probe(s) (module `algorithm`) |
| --- | --- |
| Does splitting a reduction into partial sums change the answer, and by how much? What is the reassociation noise floor at this size and dtype? | `split_contraction_accuracy` |
| Does online softmax (flash) beat materializing the score matrix at this shape? | `flash_vs_materialized_attention` |
| What does accumulating in bf16 instead of f32 cost in time and error, and is it even allowed? | `accumulator_precision` |
| If `jnp.cumsum` does not lower, what does the triangular-matmul substitute cost, and does a hand-written scan beat it? | `cumsum_alternatives` |

Related probes in other modules: `mxu_peak_by_dtype`, `vpu_fma_rate`, `mxu_block_rows_cost`
(unit rates and the row-block cliff, [compute_units.md](compute_units.md)); `integer_ops_int32` and
`dot_dtype_matrix` (what lowers on integer blocks and which `jnp.dot` dtype pairs are accepted);
`xla_reference_vs_f64_oracle` (whether the reference is better than the kernel it judges);
`dispatch_floor` (per-call and per-grid-step fixed cost). Run with
`{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/run_probe.py --pick <probe_name> --run_dir <run_dir>`.

---

## The largest lever, not the last resort

Changing what is computed is usually the biggest win available, and it is consulted too late when
treated as a fallback. A constraint that cannot be satisfied by layout, pipelining or unit placement
is very often dissolved by computing something *else*, so this is where the other areas export their
costs (see [cross_area_moves.md](cross_area_moves.md)). Consider it **while** the layout, pipelining
and unit-placement decisions are still open.

Two framings decide whether this area gets used:

- **A different algorithm is a mutation of the current kernel, not a separate project.** It has a
  parent version and a motivating finding, and it is judged by the same benchmark as a tile-size
  change. Treated as a fresh start it gets tried once, late, with nothing to compare against.
- **More arithmetic is not worse arithmetic.** FLOP count is a proxy for time only when one unit is
  saturated and nothing waits on it. Adding work is right when it fills an idle unit, moves work
  from the slow unit to the fast one, shortens a sequential chain by paying in parallel width, or
  produces a value on the axis its consumer already wants. A roofline sees none of these; rejecting
  a restructuring on FLOP count alone rejects it on a number that was never the constraint.

---

## Exact restructurings vs approximations

| | exact | approximate |
| --- | --- | --- |
| what changes | schedule, memory traffic, which unit runs it | the numbers coming out |
| evidence needed | a perf number | a perf number **and** a named accuracy gate with a threshold, evaluated first |
| examples | online softmax, split reductions, prefix-sum-as-matmul (in exact arithmetic), batching an independent axis, recompute instead of save | narrowing an accumulator, dropping a correction term, low-rank substitution, static-threshold selection |

An exact restructuring can be adopted on a perf number alone. An approximation cannot, because a
kernel that skips work is fast and the benchmark cannot tell a speedup from a shortfall in what was
computed. "The user will decide" is not a gate; a gate is a metric, a reference, a threshold and a
number.

**Several restructurings exact in real arithmetic are approximations in floating point**: online
softmax, split reductions and matmul prefix sums all reassociate. Classify each move as `exact` or
`approximate` in the plan, and carry that label with the version.

---

## Reassociation can move an answer

Splitting a reduction into partial sums changes the order of additions and therefore the rounding.
Before judging any restructuring's error, measure the **reassociation noise floor** for the
reduction length and dtype in question with `split_contraction_accuracy` (one long f32 reduction
split into varying numbers of partial sums; it reports the largest deviation).

Use the floor as the calibration for every other error in the design:

- A restructuring exact in real arithmetic that moves the answer by about the floor has not broken
  anything — that is reassociation, and it is expected.
- One that moves it by orders of magnitude more has done something else; find out what before
  accepting it.
- hypothesis: the split count barely matters, so splitting a reduction to expose parallelism is
  numerically free in f32; verify with `split_contraction_accuracy` at the design's length.

For orientation without measurement: unit roundoff is `2^-24` for f32 and `2^-8` for bf16. Errors
from bf16 operands cluster near the bf16 decade, f32 paths near the f32 decade; a result landing
between them is worth understanding before accepting it.

Two parenthesizations with the same FLOPs can put the result on different axes, which makes the
choice a layout decision too (see [layout_precision.md](layout_precision.md)). They are not
bit-identical: compare the difference against the noise floor before calling them equal.

---

## Online softmax

The canonical restructuring: instead of materializing the full S×S score matrix, walk the key
blocks keeping a running max and running normalizer, rescaling the accumulator as the max moves.

hypothesis: a plain hand-written flash kernel may not beat XLA materializing the scores at moderate
sequence lengths, even where the score matrix is large — XLA's matmul rate is higher than a
hand-written Pallas matmul's, and that alone can cover the extra HBM traffic. Verify with
`flash_vs_materialized_attention` at the design's shape before assuming the win.

What follows for a design:

1. **The memory argument for online softmax is a memory argument first.** It stops the score matrix
   from exhausting HBM; it becomes a speed argument only past some sequence length. Name that
   boundary in the plan rather than assuming it.
2. A losing plain flash kernel does not show that a tuned one (causal block skipping, multi-head
   batching, good block shapes) loses. Causal masking, head batching and longer sequences all move
   the comparison.
3. **With bf16 operands the restructuring is an approximation.** The rescaling and bf16 inputs move
   the answer far past the reassociation floor; it needs a gate, not just a speedup.
4. Mask with a **finite** large negative applied before the exponential, never `-inf`: `-inf`
   produces NaN under the rescaling (`-inf - (-inf)`). Leave headroom so that adding a real logit to
   the mask value stays finite.

---

## Accumulator precision

Widen the accumulator, never narrow it. Accumulate in f32 for every non-integer operand dtype and in
int32 for all-integer operands; carry the accumulator across the K grid axis in an f32 VMEM scratch
regardless of input dtype.

- **The narrow option may not exist.** Mosaic may reject a bf16 accumulator outright
  (`VerificationError`); toolchain-dependent, verify with `accumulator_precision`.
- **It would not pay anyway.** hypothesis: bf16→f32 and bf16→bf16 matmul rates are within noise;
  verify with `mxu_peak_by_dtype`. There is no throughput reason to accumulate narrow.
- `accumulator_precision` also reports the error of f32 accumulation over a long contraction
  against an f32 `HIGHEST` reference; quote that when asked what f32 accumulation costs in accuracy.

---

## Moving work from the VPU to the MXU

The most reliable exact restructuring, because the units differ by a large factor (ratio of
`mxu_peak_by_dtype` to `vpu_fma_rate`). **Any reduction, mask, prefix or normalization expressible
as a matmul against a fixed structured matrix is a candidate.** Limits:

- **The MXU must be filled.** A mostly-zero matmul buys nothing: a few-tap convolution as a matmul
  is a narrow band in a full pass, so keep it on the VPU as straight-line FMAs. A structured matmul
  narrower than the row block pays a full pass ([compute_units.md](compute_units.md)).
- **It is a trade, not a free substitution.** A triangular-matmul prefix sum does O(n²) MACs to
  replace O(n) adds; it wins because the unit is faster, not because work shrank. At long enough
  scan axes it loses to a scan.
- **Precision.** A structured matmul at `DEFAULT` precision puts a bf16 rounding step inside what is
  supposed to be an exact sum. `HIGHEST` is honoured inside a Pallas `dot_general` and costs a
  multiple of `DEFAULT` (measure in-kernel). Decide per contraction: a rank computed by summing a
  0/1 mask can be bit-exact at `DEFAULT`, while contracting real values through a one-hot or
  triangular matrix at `DEFAULT` is not. Prove exactness with an integer checksum against `HIGHEST`
  when the result is used as an index.

Build structured matrices from a broadcast comparison of two `iota`s (e.g. `row_idx <= col_idx` for
a reverse prefix, the comparison flipped for a forward one), not with `jnp.tril`.

**Concatenate two narrow matmuls that share an operand.** Stacking the varying operands of two
matmuls against the same contracted operand along the row axis turns two passes that each fill part
of the row block into one, and stages the shared operand once. Split the result afterwards and apply
any per-branch scale on the VPU. It needs the two ops to be independent; it pays only while the
varying operand is below the row block. See [cross_area_moves.md](cross_area_moves.md).

### A prefix sum does not have to be a matmul

`jnp.cumsum` and `jax.lax.associative_scan` may not lower inside a Pallas TPU kernel
(toolchain-dependent; verify with `cumsum_alternatives`). That bounds which library call is
available, not what the hardware can compute. Three substitutes:

| Substitute | Cost shape | Accuracy |
| --- | --- | --- |
| lower-triangular matmul | O(n²) MACs on the MXU | bf16 rounding at `DEFAULT`; `HIGHEST` costs a multiple |
| hand-written log-step (doubling) scan on the VPU | `log2(n)` shift-and-add passes | f32-accurate |
| blocked scan: small in-block scan plus an exact cross-block offset | in-block cost at a small width | exact if offsets are exact |

hypothesis: at small blocks the log-step scan is faster *and* more accurate than the matmul, and at
wide blocks (thousands of lanes) the matmul wins; the crossover is unmeasured and depends only on
`n`. Verify with `cumsum_alternatives` at the design's width; do not interpolate. When a wide rank is
needed, a blocked scan bounds the *block* width at the cheap side of the crossover, not the problem
width.

The general lesson: **a refusal from `jnp.*` is a reason to write the recurrence out by hand and
measure it**, not a reason to adopt whichever substitute the nearest shipped kernel used.

### Three reasons to spend FLOPs

1. **Throughput** — the same work on a faster unit. Visible to a FLOPs model.
2. **Latency** — a sequential phase costs chain length × carried width × dependent-issue latency;
   operation count barely enters. Solving sequentially for a narrow **operator** and then applying
   it with one wide matmul is more arithmetic but shortens and narrows the chain, and the operator
   is reusable across every right-hand side. Blocking a sequential loop bounds how much is
   single-file but does not change what rides the chain; once blocked, attack the width.
3. **Layout** — a value produced on one axis and consumed on another needs a lane↔sublane move that
   no roofline term counts. Computing a wider quantity that can be contracted in place, or choosing
   `dot_general` dimension numbers from the layout you have, can beat the narrower transposing form.

A FLOPs-and-bytes model ranks reasons 2 and 3 last every time, because the only term it has is the
one that got worse. Mark such candidates `unranked`, not `rejected`, and give them a probe: a
dependent chain of depth `D` at width `W` against the equivalent independent matmul.

---

## Selection and ordering primitives may not lower; the algorithm absorbs it

Pallas TPU's TensorCore lowering may implement no primitive that reorders or ranks elements:
`jax.lax.top_k`, `jax.lax.sort`, `jnp.cumsum` and `jax.lax.associative_scan` have all raised
`NotImplementedError` / `MLIRError` naming the `tc` target. Toolchain-dependent; verify by compiling
the candidate form (`cumsum_alternatives` covers the scans; for the others, a one-kernel compile is
the whole experiment). A refusal naming the target is about what the target implements, not about
shapes, so reshaping does not fix it. The SparseCore lowering has a separate set
([sparsecore.md](sparsecore.md)).

**Why it matters more than a missing op usually does.** Ordering is reached for implicitly — "merge
the survivors into a running best-k", "keep the list ordered". Such a phrase in a plan is the whole
buildable-or-not question. Treat any ordering, ranking or selection step as an unlowered primitive
until a compile says otherwise.

**In-register gather refuses by width, not outright.** `jnp.take_along_axis` along the lane axis can
lower when every source lane lives in the same vreg as the destination and refuse when the gather
spans more than one source vreg ("Multiple source vregs along gather dimension"). The escape is to
re-granularize so each gather is confined to one vreg (independent groups, or a two-level scheme
whose coarse step is arithmetic). Prove confinement for the *worst* group: one spanning group fails
the compile, and a group that overflows its slots silently drops survivors.

### What lowers instead

Each substitute replaces data-dependent reordering with data-independent work over the whole tile —
asymptotically more work, winning or losing on tile size.

- **Rank by prefix sum, then scatter.** Each survivor's destination is the count of survivors before
  it. If the rank is a destination index it must be **exact integer arithmetic**: an off-by-one
  writes to the wrong slot silently. f32 represents integers exactly up to `2^24`; make that
  argument explicitly.
- **Permute by one-hot matmul** (`onehot(rank, S)ᵀ @ values`). Pays dense cost for a mostly-zero
  matrix, and materializing an `S × W` one-hot scales with input width `W`, so it can overflow VMEM
  at any `S`; block the input rather than narrowing the output. hypothesis: once grouped, cost tracks
  the group count more than the per-group width; tune groups first, then price the width at your
  geometry.
- **Threshold, then store under a mask.** Cheapest and not super-linear, but approximate at fixed
  cost or exact at variable cost, never both. It owes evidence of where the threshold comes from and
  how far over-selection can run, and an answer for buffer overrun. Before designing around a static
  threshold, print the spread of the true per-row k-th value across rows at the production shape: a
  wide spread kills the static-threshold family with no kernel written. A masked store still writes
  the full tile width — it suppresses values, not bytes.

**Separate rank from adjacency.** "Compaction" is two operations: the *rank* (each survivor's
destination, an arithmetic reduction, nothing moves) and the *adjacency* (physically packing
survivors into contiguous lanes). hypothesis: the rank is close to free and the adjacency costs
several times a masked store; ask whether the consumer can take a rank field alongside a sparse
buffer and never pay for adjacency. Isolated compaction costs are an **upper bound** inside an
MXU-busy kernel, where VPU and cross-lane work co-issue with matmuls; the reverse holds in a kernel
with no MXU work to hide behind. Say which neighbourhood a price came from.

**Grouped forms have a capacity ceiling.** With `g` groups of `S_sb` slots, recall against `r`
survivors is at most `min(1, g·S_sb / r)` — an upper bound exact only for uniform spread; real
inputs cluster. Size buffers from a measured occupancy histogram on real inputs, design for the
worst-case `r`, report `groups_overflowing` beside recall, and prefer `groups_overflowing == 0` as
the gate. Do not substitute one max/argmax per sub-block for a rank without measuring how many
survivors it drops against the true top-k.

---

## Batching an independent axis into the block

When the kernel processes one independent item per grid step — one request, sequence, expert,
image — that axis is not narrow, it is **absent**, and adding it is an algorithm change. It is exact:
no arithmetic or bytes change, only the trip count.

**Availability test, in einsum notation:** an axis can be batched iff it appears in the output and is
never contracted. `"hrd,hdm->hrm"` per item becomes `"bhrd,bhdm->bhrm"`; leading axes flatten into
one matmul batch dimension, so in practice it is a `reshape` and a leading dim on the scratch.

| Payoff | Mechanism | Wins when |
| --- | --- | --- |
| larger matmul | more rows per MXU pass | the per-item matmul is below the row block |
| fewer grid steps | fewer × the per-step fixed cost (`dispatch_floor`) | the grid is short and each step issues its own DMAs |

The second is the one that gets missed: "does packing fill the MXU?" can correctly answer **no**
while batching is still one of the largest wins. If every item owns a private state matrix, a
batched matvec is still `B` passes — occupancy unchanged — and the whole difference is the fixed cost
of the steps that disappeared. Model it as `T = F·steps + W·items`; when `F` dominates, the speedup
approaches the step-count ratio and no tuning inside the step body can reach it.

**Detect it:** if the trip count is proportional to item count and the block covers one item, the
axis is missing. **Cost:** VMEM grows linearly in `B` (search `B` with a footprint estimator,
descending candidates, taking the first that fits with headroom), plus per-item length and offset
metadata because a block of `B` may hold fewer valid items — shorten the DMA, keep the shape static
([precomputation.md](precomputation.md)). Choosing `B` is then [partitioning.md](partitioning.md).

Collapsing adjacent axes into the row axis (heads × query block, via `reshape`, not transpose) is the
same move from the MXU-fill side; it also cuts the trip count, which pays even when the matmul does
not get wider.

## The same axis one level down: the per-item loop in the body

An axis also goes missing *inside* the block: the block covers all `n` items and the body walks
them with a Python `for` over a static range. That loop is unrolled at trace time, so every matmul,
mask and exponential inside it is emitted `n` times. Keep the axis as a leading tensor axis and
declare it as a `dot_general` **batch** dimension:

```python
# [N, C, C] x [N, C, D] -> [N, C, D]: one matmul op for all N items
out = jax.lax.dot_general(a, b, (((2,), (1,)), ((0,), (0,))),
                          preferred_element_type=jnp.float32)
```

Same einsum test. Same FLOPs, bytes and per-item matmul shapes; only the number of ops issued
changes, so the row-block check correctly reports nothing and is answering the wrong question (see
the op-count cost in [compute_units.md](compute_units.md)). hypothesis: on a loop-heavy body this is
a multi-fold win; measure it.

- **VMEM is what bites.** `n` items are live where one was, and the failure is a compile-time
  `CompileTimeScopedVmemOom`, a wall, not a slowdown. A batched arm that compiles at a small shape
  says nothing about the large one (grid-extent-dependent scalar-prefetch operands grow with it).
  Budget the widened live set before measuring; algebra that deletes one live array may not clear
  the wall, and raising `vmem_limit_bytes` mid-comparison moves the ruler.
- **The store side can eat the win.** Recombine `[N, C, D]` by slicing on the untiled leading axis
  and lane-concatenating, not with `transpose(...).reshape(...)`, which is a real relayout.
- **Take it before other per-op moves, then re-price them**: batching already removes `(n−1)/n` of
  the ops they attack. Measure the composition; do not ship the product of separately measured
  factors.
- **Not the same as concatenating a shared operand.** Batching adds a new leading axis across
  independent items; concatenation widens an existing axis across two ops sharing an operand. They
  compose. A rank-1 expansion such as `q @ (S + kᵀv) = q @ S + (q·k)·v` can break a dependence and
  make a concatenation legal ([cross_area_moves.md](cross_area_moves.md)).

---

## Other restructurings worth checking for

- **Recompute in the backward instead of saving.** A fused linear + softmax-cross-entropy can keep
  logits in VMEM, return only the log-sum-exp, and recompute `x @ W` in the backward. Exact; trades
  FLOPs for HBM ([fusion_boundary.md](fusion_boundary.md)).
- **Exploit log-linearity of log-sum-exp** to accumulate over a blocked vocabulary or key axis
  without materializing the full row.
- **Solve for the inverse operator blockwise**: a per-block sequential substitution whose payload is
  the identity, coupling blocks with matmuls, then apply the operator once to a right-hand side wider
  than either output it serves.

---

## Writing the gate

For every approximation the plan names, before any perf number is believed:

1. **The metric** — report relative max error (catches a single broken tile) and relative mean error
   (catches systematic bias).
2. **The reference** — strictly better than the thing tested, *shown, not assumed*. An f32 XLA
   reference that does not pin `Precision.HIGHEST` multiplies in bf16 and can be less accurate than
   the kernel it judges. Pin `HIGHEST` in any f32 contraction in the reference, and check the
   reference against a float64 oracle (`xla_reference_vs_f64_oracle`; see
   [tpu_hardware_facts.md](../tpu_hardware_facts.md) §9).
3. **The threshold, in advance** — defensible against the reassociation floor from
   `split_contraction_accuracy`. A threshold far above the bf16 decade accepts anything; one below
   the f32 decade rejects f32 itself.
4. **The order** — the gate is evaluated first, not alongside the perf number.

---

## Deciding

1. **Classify**: does the answer change? If not, a perf number suffices. If so, write the gate first
   and label the move `approximate`.
2. **Check availability**: compile the restructuring's primitives. "Does not lower" bounds the call,
   not the computation — write the recurrence out before accepting an unavailability.
3. **If moving work to the MXU**, check it fills the row block, and say *why* it moves: throughput,
   latency or layout. The latter two survive failing the row-block test or adding FLOPs.
4. **Mark chain-length or layout claims `unranked`**, not `rejected`, and attach a probe.
5. **List the workload's independent axes** and check the block has each one; judge against step
   count as well as MXU fill.
6. **Check the body carries each axis as a tensor axis**, not a Python `for`. Price VMEM before the
   arm; re-price other per-op moves afterwards.
7. **Measure the error** against a strictly better reference and compare to the reassociation floor.
8. **Measure the speed**, and be prepared for the restructuring to lose.

## Gaps

- `flash_vs_materialized_attention` is non-causal, single-head and limited in sequence length; it can
  show a naive flash kernel losing, not that flash attention loses.
- The prefix-sum crossover between the log-step scan and the triangular matmul is unlocated, and the
  measurements on either side differ in more than width (f32 value scan vs integer rank over a mask).
- No probe has yet shown a gate rejecting a genuine approximation.
- The per-item-loop batching win has no scaling law in `n`; the VMEM cost that terminates it grows.
- Op count and matmul shape are confounded in every measurement of the per-op cost.
- Lowering verdicts (`cumsum`, `associative_scan`, `top_k`, `sort`, wide lane gathers) change
  between toolchains; re-run `cumsum_alternatives` and recompile after any upgrade.
- The triangular-matmul error in `cumsum_alternatives` may be measured at `DEFAULT`; the error at
  `HIGHEST` should be lower and should be measured if the design uses it.
