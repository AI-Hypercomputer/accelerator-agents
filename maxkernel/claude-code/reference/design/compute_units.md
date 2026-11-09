---
name: compute-units
description: >-
  Which unit (MXU, VPU, SparseCore) should run each piece of a kernel's work, and the cost models
  for each: per-dtype MXU rate, the row-block cliff, matmul op count, VPU and transcendental cost.
  Read while placing work on units and before writing the occupancy arithmetic for a design.
---

# Which unit does each piece of work

## Questions and probes

| Question | Probe(s) (module `compute_units`) |
| --- | --- |
| What matmul rate does this chip reach per operand/accumulator dtype? This is the denominator for every other rate. | `mxu_peak_by_dtype` |
| Does a matmul pass over fewer rows than the MXU block cost as much as a full pass? Where is the knee? | `mxu_block_rows_cost` |
| Do sub-block contraction depths (K) and column counts (N) also cost a full block? | `mxu_contraction_and_column_granularity` |
| What elementwise FMA rate does the VPU sustain on VMEM-resident data? | `vpu_fma_rate` |
| How much more do `exp`, `tanh`, `rsqrt`, `log`, `sigmoid`, divide cost than a multiply, per element? | `transcendental_relative_cost` |
| Is SparseCore reachable from Pallas here at all? | `sparsecore_surface` (then `sc_module_surface`, see [sparsecore.md](sparsecore.md)) |

Related probes in other modules: `tpu_info_reported_constants` and `tpu_info_scaling_denominator`
(what `get_tpu_info()` reports and whether it is per chip or per TensorCore),
`native_matmul_dtype_matrix` and `f32_default_is_one_bf16_pass` (which dtype pairs are native, what
`DEFAULT` precision does to f32), `arithmetic_intensity_crossover` (where a kernel stops being
bandwidth-bound). Run any of them with
`{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/run_probe.py --pick <probe_name> --run_dir <run_dir>`.

**Scope.** This file is about *placement and cost models*: which unit, what a pass costs, what the
design should compute before writing code. The mechanics of driving the MXU once work is placed on
it — `vmatpush`/`vmatmul`/`vpop`, queuing matmuls to hide latency, Mosaic vector layouts,
interleaving VALU and MXU work, unrolling, register spills — are in
[tpu_mxu_and_register_optimization.md](../tpu_mxu_and_register_optimization.md). Do not repeat that
material in a plan; cite it.

Work placed on the wrong unit is slow in a way no amount of tiling fixes.

---

## Read unit constants at run time

Never hardcode a lane count, sublane count, MXU dimension, capacity or peak rate in a kernel or a
plan. Read them from `pltpu.get_tpu_info()` on the target:

- **Structural fields** — `num_lanes`, `num_sublanes`, `num_mxus`, `mxu_column_size`,
  `vmem_capacity_bytes`, `smem_capacity_bytes`. These are the ones to reach for first. Tile-size
  floors in production kernels are written as multiples of `mxu_column_size` read from this call,
  so they adapt when the chip changes; a prose comment that hardcodes the number does not.
- **Rate fields** — `bf16_ops_per_second`, `fp8_ops_per_second`, `int8_ops_per_second`,
  `int4_ops_per_second`, `mem_bw_bytes_per_second`. These are *peaks*, not achievable rates.
- **The denominator.** Some fields are a chip-level constant divided by TensorCores per chip; others
  (VMEM, SMEM) are per core to begin with. Mixing the two is off by the TensorCore count in one
  direction or the other. See [tpu_hardware_facts.md](../tpu_hardware_facts.md) §1 and verify with
  `tpu_info_scaling_denominator`.
- **A zero rate field means "not supported natively"**, not "unknown". A chip whose
  `int8_ops_per_second` is 0 has no native integer MXU pair; an int8 `jnp.dot` is promoted and runs
  at roughly the bf16 rate. Quantizing to int8 on such a chip buys bytes (a bandwidth argument),
  not compute. Which narrow dtype buys compute is chip-dependent; check
  `native_matmul_dtype_matrix` and the `mxu_peak_by_dtype` table before planning a quantized design.
- **`sparse_core`** is a `SparseCoreInfo` or `None`. Its `vmem_capacity_bytes` is per vector
  subcore and not poolable; do not read it as a budget the way the TensorCore fields are read. See
  [sparsecore.md](sparsecore.md).

---

## MXU: the per-dtype rate is the denominator

Every rate in a design — "this kernel reaches X of peak", "the VPU is N times slower" — needs a
denominator, and the reported peak is the wrong one: it is a datasheet number no kernel reaches.
Use the **achieved** XLA matmul rate per dtype on the target, measured by `mxu_peak_by_dtype`
(a large square `jnp.dot` per operand → accumulator dtype and precision).

What to read off that table, as questions rather than numbers:

1. **Does f32 at `DEFAULT` precision run at near the bf16 rate?** If so, it is not f32 arithmetic:
   XLA is decomposing into bf16 passes. The honest f32 rate is the `HIGHEST` row. If the kernel
   needs genuine f32 matmul accuracy it pays that cliff; if it did not ask for `HIGHEST` it is not
   getting f32. Inside a Pallas kernel the `DEFAULT`/`HIGHEST` ratio differs from the XLA one —
   use the in-kernel figure for an in-kernel decision, and name which you used. Precision rungs
   available inside a kernel are in [layout_precision.md](layout_precision.md).
2. **Which narrow dtype is faster than bf16?** Only a dtype the MXU multiplies natively. On a chip
   with an fp8 path, fp8 is the one that buys throughput; a promoted dtype lands at the bf16 rate.
3. **Is an f32 accumulator slower than a bf16 one?** hypothesis: no, the difference is within noise;
   verify with `mxu_peak_by_dtype` (bf16→f32 vs bf16→bf16 rows). There is no throughput reason to
   accumulate narrow, and inside Pallas a bf16 accumulator may be rejected outright (probe
   `accumulator_precision`; see [algorithm.md](algorithm.md)).
4. **Is any row's spread much wider than the others?** A path with unstable timing (f16 has been a
   candidate) should not be used without measuring it at the kernel's own shape.

Caution on comparing a hand-written Pallas rate against this denominator: `mxu_peak_by_dtype` is
unbatched and at a large, MXU-favourable shape. When a decision turns on the ratio, run the XLA arm
and the Pallas arm in one probe at the same shape and batching.

---

## The row-block cliff, as a cost model

Hold total work fixed and vary only the number of rows in the matmul block. Below the MXU's row
block, wall time scales like `R_mxu / rows`: **a pass over fewer rows than the MXU block costs
approximately a full pass.** Feeding the MXU half a block throws away half the machine. Above the
block the curve is flat; enlarging that axis further buys little.

Cost model to use in a plan:

```
mxu_time(m, k, n) ≈ ceil(m / R_mxu) * R_mxu * k * n * 2 / rate_for_dtype
fraction_of_mxu_used_by_row_block = min(1, rows / R_mxu)
```

`R_mxu` is the knee measured by `mxu_block_rows_cost` on the target. The probe is phrased around
128 rows; on chips whose `mxu_column_size` is wider, check whether the knee sits at 128 or at
`mxu_column_size` rather than assuming either. The rule is a **threshold, not a gradient**: get the
row block to the knee, then stop optimizing that axis and spend effort elsewhere.

This is the MXU's systolic dimension, unrelated to the sublane tile (`num_sublanes`, and its
dtype-dependent widening) that governs block legality — that one is in
[layout_precision.md](layout_precision.md) and [tpu_hardware_facts.md](../tpu_hardware_facts.md) §4.

The levers that push rows over the knee: collapse adjacent leading axes into the row axis
(`reshape`, not transpose — e.g. heads × query block), batch an independent axis into the block,
concatenate the varying operands of two matmuls that share the contracted operand along the row
axis. All three are in [algorithm.md](algorithm.md) and [cross_area_moves.md](cross_area_moves.md).

---

## Contraction depth and column count behave differently

With rows held at a full block and work *not* held fixed, wall time can be nearly independent of K
and N across a wide range — a K or N several times larger costs almost nothing extra. Two readings
fit that shape and have not been separated:

- sub-block K and N extents are padded up to a full block (would give a step function), or
- at that row count the matmul is not compute-bound, and every arm is measuring operand movement
  or a fixed per-call cost (would give bumps and non-monotonicity).

What survives either reading, and is what a plan acts on: **at a full row block, widening K or N
toward a few times `mxu_column_size` is nearly free; take the larger extent.** It is also why
production kernels put a floor of a small multiple of `mxu_column_size` under the N tile "to avoid
stalling the MXU", and why shrinking a K or N tile below `mxu_column_size` to save VMEM is rarely
worth it. Verify at the kernel's shape with `mxu_contraction_and_column_granularity`; the probe
does not hold total work fixed, so do not read it as a K/N cliff measurement.

---

## The number of matmul ops is a separate cost

Everything above prices one MXU pass. A body that issues many small matmuls pays a second cost no
occupancy figure contains: each distinct matmul op carries its own operand staging, VPU glue and
scheduling around it. Two counters both get called "matmul issues", and they rank transformations
differently:

| Counter | What it is | Where to read it |
| --- | --- | --- |
| hardware MXU passes (`vmatmul`) | passes over the array, set by the operand shapes | LLO bundle counts in the profile summary / LLO dump |
| matmul **ops** (`tpu.matmul`) | distinct matmul operations in the program | the Mosaic MLIR in the LLO dump |

hypothesis: on a solve-heavy or loop-heavy body, wall time tracks the op count, not the pass count —
a transformation can raise pass count and MXU occupancy and still be faster because it removed ops,
and pass count can be anti-correlated with time. Verify by fitting `time = a + b · ops` across arms
of your own kernel; `b` is one kernel's slope, not a platform constant. No probe yet isolates op
count from matmul shape (the same matmul issued `n` times vs batched over `n`).

Two independent levers, then:

1. **make each matmul fill the array** — the row-block cliff above;
2. **emit fewer matmuls for the same arithmetic** — carry a per-item Python loop as a `dot_general`
   batch dimension, or concatenate operands of matmuls that share one
   ([algorithm.md](algorithm.md), [cross_area_moves.md](cross_area_moves.md)).

The second is invisible to every occupancy check because no matmul changes shape. Count ops before
concluding a body is done: count the static `for` loops and multiply.

---

## VPU rate and transcendental cost

The VPU sustains far less than the MXU on the same dtype. Measure the VPU FMA rate on the target
with `vpu_fma_rate` and divide the f32 MXU rate from `mxu_peak_by_dtype` by it: that ratio (large,
on the order of hundreds) is the whole reason the "which unit" question exists. Any reduction, mask,
prefix or normalization that can be restated as a matmul against a fixed structured matrix has that
headroom argument behind it ([algorithm.md](algorithm.md), "Moving work from the VPU to the MXU").

**The argument only pays if the MXU is filled.** A short convolution of a few taps written as a
matmul is a banded, mostly-zero matrix that fills almost none of a pass; production kernels keep it
on the VPU as a straight-line chain of FMAs on purpose. Same for any structured matmul narrower than
the row block.

Relative per-element cost against a multiply comes from `transcendental_relative_cost`. Decision
rules that follow from the direction it has shown (each a hypothesis to re-check on the target):

- hypothesis: divide costs about what a multiply costs. Then hoisting a reciprocal ("compute `1/x`
  once and multiply") saves nothing; a "fuse reciprocal" knob in an attention kernel is a
  fusion-boundary decision about where the normalizer lives, not about divide cost (see
  [fusion_boundary.md](fusion_boundary.md)).
- hypothesis: `exp`, `tanh`, `rsqrt`, `log` cost modestly more than a multiply. Then the softmax
  exponential is not what makes attention expensive; do not restructure to avoid it.
- hypothesis: `sigmoid` is the dear one, at roughly `exp` plus divide, already fused.

For the VPU floor of a design, multiply the op count by the relative cost of the dominant op.

---

## SparseCore

The SparseCore is a third unit with its own Pallas surface (`jax.experimental.pallas.tpu_sc`), its
own geometry (`get_tpu_info().sparse_core`) and its own lowering rules. It wins under two
conditions: read-only data-dependent addressing of narrow rows into a table that fits one subcore's
VMEM, or work independent of the TensorCore stage beside it, where it is free while being slower.
A serial SparseCore stage that is neither is a regression, and an offload must be measured as added
step time beside its TensorCore neighbour, never standalone. Before concluding it is unreachable,
note that `sparsecore_surface` enumerates candidate module names and a negative only means the
guessed names are absent; confirm with `sc_module_surface`. Everything else is in
[sparsecore.md](sparsecore.md).

---

## Occupancy arithmetic to emit with a design

Cheap, and owed before any kernel is written. Put these in the plan with the inputs that produced
them (constants from `get_tpu_info()`, rates from the named probes):

- **MXU-bound floor:** `2 * m * k * n / rate_for_dtype`, with `rate_for_dtype` from
  `mxu_peak_by_dtype`, not the reported peak. Scale by `R_mxu / rows` when the row block is below
  the knee.
- **VPU-bound floor:** `elementwise_ops * relative_cost_of_dominant_op / vpu_rate`, rates from
  `vpu_fma_rate` and `transcendental_relative_cost`.
- **HBM-bound floor:** `bytes_moved / achieved_bandwidth` — the roofline in
  [tpu_memory_overlapping.md](../tpu_memory_overlapping.md) §1.
- **Which one binds:** compare the three floors. The crossover between bandwidth-bound and
  compute-bound is measured by `arithmetic_intensity_crossover`
  ([pipelining.md](pipelining.md)). An elementwise kernel is never MXU-bound.
- **Row-block waste:** `min(1, rows / R_mxu)` for every matmul in the body.
- **Matmul op count:** static loop trip counts multiplied out, and how many of those ops are copies
  of each other that one batched `dot_general` would issue once.
- **Is there a SparseCore column?** Only if the work is (a) read-only narrow-row indexing into a
  table that fits one subcore's VMEM, or (b) independent of the TensorCore work beside it. For (b)
  the question is whether the SparseCore time fits inside the neighbour's, not whether it is
  smaller than the TensorCore's. Anything serial with the surrounding TensorCore work fails both.

---

## Gaps

- The two readings of K/N insensitivity (padding vs not compute-bound) have not been separated; a
  probe holding total work fixed while sweeping K and N would do it and does not exist.
- The per-op cost has no isolating probe: every measurement confounds op count with matmul shape.
- Only int8 has been checked against "a zero rate field means promoted"; int4 has only been shown
  to compile (`bitcast_and_subbyte`).
- The row-block knee on chips whose `mxu_column_size` exceeds 128 should be confirmed by
  `mxu_block_rows_cost` on that chip before a plan relies on 128 or on `mxu_column_size`.
