---
name: cross-area-moves
description: >-
  The costs paid between two design areas, which no single area file can see, and seven
  inversion questions (S1-S7) that find the restructurings collecting them. Read after a design
  has been worked area by area, before the plan is final.
---

# Cross-area moves

Some costs are paid **between** two design areas rather than inside either one. A design worked
one op, one value or one area at a time prices every row correctly and still misses them: a
relayout paid between two ops belongs to neither op's row; two ops that would be cheaper as one
wider op are two rows that each look fine; an exact rewrite that only changes where values land
improves no row's number. Nothing is wrong with per-row bookkeeping — this is what it hides. Read
the design **across** the rows once it exists.

Not an eighth area; each move found goes back to the area file that owns its mechanism.

| # | Area | File |
| --- | --- | --- |
| 1 | fusion boundary | [fusion_boundary.md](fusion_boundary.md) |
| 2 | partitioning | [partitioning.md](partitioning.md) |
| 3 | pipelining | [pipelining.md](pipelining.md) |
| 4 | layout / precision | [layout_precision.md](layout_precision.md) |
| 5 | compute units | [compute_units.md](compute_units.md) (SparseCore: [sparsecore.md](sparsecore.md)) |
| 6 | algorithm | [algorithm.md](algorithm.md) |
| 7 | precomputation | [precomputation.md](precomputation.md) |

**The seven seams below are a generator, not a checklist.** They are the cross-cutting moves that
have been written down; the list is open. What transfers is the *shape* of each — an
**inversion**, a question asked from the seam instead of from either side — so ask it on whatever
pair of ops, values or kernels your design actually puts next to each other. A move that fits none
of the seven is still a move. **"No instance, because —" is a complete answer for a seam; silence
is not.**

## The licence: three reasons to spend FLOPs

A FLOPs-and-bytes roofline (see the roofline in
[`tpu_memory_overlapping.md`](../tpu_memory_overlapping.md) §1) prices a restructuring with only
two terms. Three reasons to move or add work are invisible to it, so a candidate justified by any
of them looks worse than it is on the model. Mark the deciding quantity `unmodelled`, not zero,
and send it to a probe.

1. **Throughput.** The units are far apart in rate, so the same work costs less on the faster one.
   Measure the gap with `vpu_fma_rate` against `mxu_peak_by_dtype`. See
   [algorithm.md](algorithm.md) on moving VPU work onto the MXU.
2. **Latency.** A sequential phase costs chain depth × width carried, and operation count barely
   enters, so adding independent FLOPs to shorten the chain can win outright.
3. **Layout.** A value produced on one axis and consumed on another must be moved, and the move has
   no term in any floor. Computing a *wider* quantity you can contract in place can beat computing
   a narrower one you must transpose.

The third is most often missed: a relayout shows up in a profile folded into whatever op it was
fused with, and in a roofline not at all. Probe the primitive, not the variant — one timing loop of
"this relayout versus its absence" answers every candidate of that shape — and record a candidate
that spends FLOPs and loses: that is how the licence becomes a bound.

## The seven seams

Area (6) appears in three of the seven: **the algebra is where other areas' costs get exported
to**, so consider rewrites early, not after the other areas fail.

### S1 — (4) layout × (6) algorithm

**Inversion: what algebraic form produces every value already on the axis its consumer needs?**
Forms the answer takes (shapes to look for, not a list to walk):

1. **Get the layout from the load.** Split a fixed-stride interleaving (packed QKV, gate/value) by a
   strided `BlockSpec` / `index_map` instead of permuting in-kernel. An in-kernel strided read is
   toolchain-dependent and may not compile; try it first. A strided block is many small transfers
   ([pipelining.md](pipelining.md)): cost the two together.
2. **Absorb the transpose into the contraction.** Pick `dot_general` dimension numbers from the
   layout instead of moving the array. Probe: `transpose_cost`.
3. **Collapse a rank-1 structure before it materializes.** `(k⊗v)ᵀq = (k·q)·v`: the outer product
   is never formed and a 2-D move becomes a scalar multiply.
4. **Reassociate so the needed axis is already the contracting axis.** Same FLOPs, different
   output layout. Check numerical equivalence with `split_contraction_accuracy`.
5. **Broadcast and repeat last, and prefer the blocked form.** The cost of an expansion is set by
   its spelling more than its axis: hypothesis: a blocked expansion is near-free on any axis, and
   the interleaved form (e.g. `jnp.repeat` along a minor axis) is the one that costs. Probes:
   `expand_blocked_vs_interleaved`, `expand_axis_leading_vs_sublane_vs_lane`,
   `elementwise_before_vs_after_expand`, `expand_before_vs_after_matmul`. For a **roll**, tile
   alignment can matter. For a **split**, hypothesis: the split point's alignment is not a lever;
   verify with `lane_split_alignment` before restructuring to align one.
6. **Spend FLOPs to buy layout** — licence reason 3, applied.
7. **Do the move on the VPU**: iota mask × multiply × sum, when the XLU is the bottleneck and the
   VPU is not. Last resort. Probe: `degenerate_transpose_via_masked_sum`.

Test: for each value, write the axis it lands on, the axis its consumer needs, and what resolves
the mismatch. Every mismatch resolved by a real permute is an S1 candidate.

### S2 — (5) compute units × (6) algorithm

**Inversion: which operations share an operand or a contraction, and what do they cost as one wider
operation?**

Check the **row** axis. A matmul pass over fewer than a full MXU row-block may cost as much as a
full one (`mxu_block_rows_cost`), while widening columns is cheap by comparison
(`mxu_contraction_and_column_granularity`). Concatenating two sub-row-block matmuls along rows
turns two passes into one and loads the shared operand once. Probes for the merged form:
`concat_to_widen_narrow_matmul`, `weight_stationary_lhs_concat`.

```python
# Two branches contract against the same stationary operand w: [K, N].
# Stack the varying operands on the row axis, contract once, split after.
ab = jnp.concatenate([a, b], axis=0)  # [2*R, K]
ab_w = jnp.dot(ab, w, preferred_element_type=jnp.float32)  # [2*R, N]
a_w = ab_w[:R] * scale_a  # per-branch scaling on the VPU
b_w = ab_w[R:] * scale_b
```

The branches must share the *contracted* operand; concatenate on the non-contracted row axis, and
keep the split on a tile boundary so it is free.

- **The pair must be independent, and algebra often makes it so.** `q @ (S + kᵀv) = q @ S +
  (q·k)·v` puts both matmuls on operand `S` and turns a serial pair into a concatenable one.
- **`n` independent ops:** batch them on a leading axis as a `dot_general` batch dimension. Shapes
  are unchanged; the number of matmul *ops* drops, a cost of its own
  ([compute_units.md](compute_units.md)). Live VMEM of the batched form is what bounds it.
- **The inverse is also a seam.** A structured matmul that is mostly zeros buys nothing. A short-tap
  sliding window over a long axis is a banded matrix; keep it as a VPU multiply-add per tap. Probe:
  `sliding_window_compact_vs_sublane`.

Test: list pairs sharing an operand or contraction axis (row count now versus merged), pairs that
could share one after a rewrite, and every static `for` with the matmul ops it multiplies.

### S3 — (3) pipelining × (6) algorithm

**Inversion: what travels down each sequential chain, and is there a narrower object that
generates it?**

Solve sequentially for the **operator**, not the answer, then apply it with one matmul. More
arithmetic, often faster: the chain carries the operator's dimension instead of the payload width,
and one operator serves every right-hand side. Example: blocked forward substitution against an
identity right-hand side yields the triangular block's inverse; applying it is one matmul.
**Blocking is not a substitute**: it bounds how much of the chain is single-file without changing
*what rides through it*. Once blocked, the remaining variable is the width.

Test: per sequential phase, record chain depth, width carried, and the width of the narrowest
object generating the same result. No registered probe; time a dependent chain of depth `D` at
width `W` against the operator-plus-matmul form.

### S4 — (7) precomputation × (4) layout

**Inversion: which layout is produced per call that could be produced once, further out?**

The precomputation ladder ([precomputation.md](precomputation.md)) is usually applied to metadata.
This seam applies it to **operands**: a `swapaxes`, `reshape` or cast in the wrapper, applied to a
weight invariant since model load, executed every call — work that belongs at model load (level 2)
and is missed because the hoist lives in the caller. Name it even when the kernel cannot take it.

Test: list every operand reshaped, transposed or cast between caller and kernel, with the level it
happens at and the level its invariance allows. Price the per-call version with `transpose_cost`
or `dtype_conversion_cost`; check whether XLA already hoists it with
`grid_invariant_recompute_cost` (in-kernel) and `metadata_cse_across_pallas_calls` (across calls).

### S5 — (2) partitioning × (1) fusion boundary

**Inversion: which axis exists in the data but not in the loop?**

Test with einsum notation: an axis can be batched into the block iff it appears in the output and
is never contracted. Two payoffs, and **the second is available without the first**: a larger
matmul (only when the per-item matmul is under a full row-block — `mxu_block_rows_cost`) and fewer
grid steps (whenever the grid is long relative to the work — `dispatch_floor`). "Does packing fill
the MXU?" can correctly answer no while batching is still one of the largest wins available. See
[algorithm.md](algorithm.md) on batching an independent axis into the block.

Test: write the workload's independent axes next to the axes the grid actually walks.

### S6 — (5) compute units × (1) fusion boundary

**Inversion: which kernel edge exists only because the two sides run on different units, and what
would it cost to overlap them instead of serializing them?**

A TensorCore/SparseCore split feels like a fact, but the *schedule* is a decision. Two kernels on
two units joined by a data dependence run back to back, and the idle unit is invisible in a profile
of the busy one. Roofline both sides; one kernel's floor is not the block's floor.

- One Pallas kernel cannot host both a TensorCore and a SparseCore body (see
  [sparsecore.md](sparsecore.md), "Piping SparseCore with the TensorCore"); the handoff between
  units is never intra-kernel. Re-check with `sc_tc_one_kernel_two_meshes` after a toolchain
  upgrade before planning a cross-unit handshake.
- Whether two *independent* `pallas_call`s on different units overlap: `sc_tc_independent_overlap`,
  `sc_tc_overlap_vs_neighbour_kind`. Cost of handing a result across: `sc_tc_dependent_handoff`.
- Where independent calls overlap, S6 is a **scheduling** problem whose work is removing the
  dependence: chunk `k` on one unit beside chunk `k−1` on the other, or move the cut so both sides
  have independent work, or split the payload across the two paths.

Model the prize as `min(t_A, t_B) · (p−1)/p` for `p` chunks, minus per-launch overhead per unit
(`dispatch_floor`); imbalance is already inside the `min`. Cautions:

- **Check the loop is a loop.** A "partitioned" block may be one reshape feeding one launch; then
  the pipeline must be constructed, not reordered. Read the `p > 1` path, not the parameter name.
- **Degenerate regime.** With little work on one side the prize vanishes but `p` launches are
  still paid: a large regression. If what selects `p` is not the quantity in `min(t_A, t_B)`, gate
  on that quantity; occupancy alone is necessary, not sufficient — add a per-chunk work term.
- **Composition.** Once one side is hidden, savings there may be worth nothing; measure S6 first.
- **Measuring:** create dependences with `jax.lax.optimization_barrier`, never `x * 0` (folded).

### S7 — (1) fusion boundary × (4) layout

**Inversion: which fusion is blocked only by a layout conflict, and what upstream layout change
dissolves it?**

Read the usual "fusion does not work: layout conflict" the other way: the layout is a variable and
the boundary is what is bought — a value flattened, a head axis moved, a repeat deferred. These are
often changes *outside* the kernel; S1's forms are what they are made of.

The reverse is a real answer too: **a boundary can be worth keeping because it lets the two sides
hold different layouts.** Merging then costs an in-kernel relayout per step (`transpose_cost`)
against a saved HBM round trip (`intermediate_hbm_roundtrip`); do the arithmetic so the boundary is
chosen, not inherited ([fusion_boundary.md](fusion_boundary.md), Check E).

Test: for each fusion not taken, is layout the reason? For each boundary kept, what would merging
cost?

## Open questions

- bf16 relayout cost is less established than f32, and most S1 candidates are bf16: re-measure
  with `transpose_cost` at the kernel's dtype. S3 and S4 have no registered probe.
- Two seam moves that each win may not compose. Measure the combined candidate, not the sum.
