---
name: partitioning
description: >-
  How a Pallas TPU problem is cut into a grid and block shapes: legality rules, what fits in VMEM and
  SMEM, per-call and per-step fixed costs, block-size probing, traversal order, dimension_semantics,
  partitioning idioms, ragged work, and VMEM sizing heuristics. Read when choosing or revising a grid.
---

# How the problem is cut into tiles

## Questions and probes

| Question | Probe(s) |
| --- | --- |
| How large a VMEM scratch can a kernel actually allocate and use? | `vmem_scratch_bisect` |
| Does a narrower dtype buy proportionally more VMEM, or is there a floor? | `vmem_bytes_per_element_by_dtype` |
| How many int32 scalars fit in an SMEM scratch? | `smem_scratch_bisect` |
| What does one `pallas_call` cost before it does any work, and what does one extra grid step cost? | `dispatch_floor` |
| Is there a fixed per-`pallas_call` cost that survives batching, and is it device time? | `pallas_call_intercept_chained_clock` |
| Which block shape wins for a fixed matmul, and how much is between best and worst? | `matmul_block_size_sweep` |
| Does the order the grid is walked change wall time, holding the partition fixed? | `grid_traversal_order` |
| Does labelling grid axes `parallel` instead of `arbitrary` change anything measurable? | `dimension_semantics_effect` |
| What does a grid whose last block is partial cost, against one that divides evenly? | `ragged_tail_cost` |

Run any of them with:

```bash
tools/run_probe.py --pick <probe_name> --run_dir <run_dir>
```

Related probes owned elsewhere: `block_sublane_legality_by_dtype`, `vmem_limit_bytes_raises_ceiling`
(hardware); `operand_count_vmem_pressure` (fusion); `block_size_bandwidth_curve`,
`pipeline_depth_and_warmup` (pipelining); `mxu_block_rows_cost` (compute units).

---

## The API, briefly

Look up exact signatures and current spellings with `tools/search_api.py` rather than from memory;
several spellings have changed across toolchains (for example the compiler-params class name and
where `ANY` lives). Probes `compiler_params_spelling` and `pltpu_symbol_surface` report what exists
on the target. The shape of a partition:

```python
pl.pallas_call(
  body,  # body(*in_refs, *out_refs, *scratch_refs)
  grid=(gm, gn, gk),  # step counts; body runs once per grid point
  in_specs=[
    pl.BlockSpec(
      (bm, bk), lambda i, j, k: (i, k)
    ),  # block coords, not elements
    pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
  ],
  out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
  out_shape=jax.ShapeDtypeStruct(
    (M, N), out_dtype
  ),  # full array, not the block
  scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],  # persists across steps
  compiler_params=pltpu.CompilerParams(
    dimension_semantics=("parallel", "parallel", "arbitrary")
  ),
)
```

Load-bearing points: `index_map` returns coordinates **in units of blocks**; two grid points mapping
to the same block is legal and is how an operand is held constant across an axis;
`pl.program_id(axis)` / `pl.num_programs(axis)` give position and extent; `pl.when` expresses
first-step init and last-step writeback; a `BlockSpec(memory_space=pl.ANY)` operand is not pipelined
and the body gets the HBM ref to move itself (see [`pipelining.md`](pipelining.md)). Refs are not
arrays: `ref[...]` loads, `ref[...] = v` stores, `ref[pl.ds(start, size)]` is a dynamic slice.

---

## The block-shape legality rule

The last two dimensions of a block shape must be divisible by 8 and 128 respectively, **or equal the
corresponding dimensions of the full array**. Violations are rejected at trace time with a
`ValueError` saying so. Whether the sublane multiple changes for narrow dtypes is toolchain-dependent;
verify with `block_sublane_legality_by_dtype` and `sublane_tiling_by_dtype`.

The "or equal" clause is the useful half. A trailing extent smaller than the native tile is legal if
the block covers the whole axis, so a rank-3 array `(steps, sublanes, lanes)` blocked
`(1, sublanes, lanes)` accepts any `sublanes` and `lanes`. That is the standard way to handle (and
to test) small trailing extents; what such a block costs per useful element is
`native_tile_granularity` (see [`layout_precision.md`](layout_precision.md)).

---

## What fits

> Everything here is **TensorCore VMEM**, the memory a `pl.pallas_call` grid allocates into. A
> SparseCore kernel budgets against a different, pooled per-core ledger, fails with a different
> error, and is not moved by `vmem_limit_bytes`. See [`sparsecore.md`](sparsecore.md) and probe
> `sc_vmem_per_subcore`.

**The usable VMEM is not the capacity `get_tpu_info()` reports.** The compiler enforces a scoped
limit below the physical capacity, and an over-budget kernel fails at compile time with a
`CompileTimeScopedVmemOom` resource-exhausted error. Measure the actual ceiling with
`vmem_scratch_bisect`. The scoped limit is a compiler setting, not silicon:
`pltpu.CompilerParams(vmem_limit_bytes=...)` moves it in either direction, up to the physical
capacity (probe `vmem_limit_bytes_raises_ceiling`). Raising it does not create memory.

**Bytes per element should track itemsize.** Halving the dtype should double what fits, with no
floor for narrow types; confirm with `vmem_bytes_per_element_by_dtype` before planning around int8 or
bf16 headroom.

**SMEM** is small and carries a little fixed overhead; read its capacity from `pltpu.get_tpu_info()`. Measure the
int32 count with `smem_scratch_bisect`; there is rarely a reason to budget it finely.

**Other memory spaces:** read `cmem_capacity_bytes` and friends from `pltpu.get_tpu_info()` at run
time (probe `tpu_info_reported_constants`). Do not design around a memory space the target reports as
zero.

### How to measure a capacity: bisect, do not ladder, and read back elsewhere

- **Bisect** between a size that compiles and one that does not. A fixed ladder of doublings only
  brackets the answer within a factor of two.
- **Read back at a different offset than you wrote.** A probe that writes and reads the same location
  lets the compiler eliminate the buffer; it then measures a static shape check with the allocator
  never running, and "succeeds" far past physical capacity. A result above the physical capacity is
  the tell.

### The budget rule (TensorCore)

```
2 × sum(block_bytes over in_specs and out_specs) + scratch_bytes  <=  scoped VMEM limit
```

The factor 2 is default double-buffering: every pipelined operand is resident twice (deeper
`pl.Buffered` counts multiply accordingly). Operand count is therefore the other half of the
constraint; probe `operand_count_vmem_pressure` shows the block shrinking as operands are added (see
[`fusion_boundary.md`](fusion_boundary.md), check C). **Reach for the larger block first and the
`vmem_limit_bytes` knob second**: a design that only fits above the default limit has no headroom for
the next operand.

---

## Fixed costs: the dispatch floor and the per-step cost

### Host dispatch

One `pl.pallas_call` that does almost nothing still costs a host round trip when timed unbatched;
measure it with `dispatch_floor`, which also times a bare jitted XLA add. The two agreeing says it is
host cost, not a Pallas cost.

**Timing rule:** a single unbatched dispatch within a small multiple of that floor is measuring the
host, not the kernel. Batch the work (many invocations per timed region) and refuse to report a
number near the floor. This is a rule about *how you time*, not about how small a kernel may be.

### The per-`pallas_call` device intercept

Underneath host dispatch sits a fixed cost per `pallas_call` that batching does not remove, because
it is on the device. **The instrument is a chained clock**: time `M` invocations inside one `jit`,
chained by a data dependency XLA cannot fold, and divide by `M`. Host dispatch divides away; a
per-call device cost does not. Measure with `pallas_call_intercept_chained_clock`. Unchained, the two
are indistinguishable and you will attribute everything to Python.

Consequences:

1. A `pallas_call` whose modelled work is only a few dozen grid steps is intercept-dominated; the
   block-size lever is tuning a term that is not the largest.
2. In that regime, **fusing two calls into one beats any block-size tuning**: it deletes a whole
   intercept.
3. A predicted per-call time below the measured intercept is unreachable. A roofline (see
   [`../tpu_memory_overlapping.md`](../tpu_memory_overlapping.md) §1) that returns less has not
   modelled this floor.
4. **Say which clock you read.** Comparing a chained number on one side with an unchained one on the
   other measures the clock, not the kernel. Host dispatch and the device intercept do not cancel.

### The per-grid-step cost

A grid step costs a fixed term plus a bandwidth term:

```
T_step ≈ F_step + bytes_per_step / BW_step
```

`F_step` covers address arithmetic, DMA descriptors, the semaphore wait and SMEM metadata reads.
Fit `F_step` and `BW_step` on the target with `dispatch_floor` (its extra-step arms) and the
pipelining probes `block_size_bandwidth_curve` / `pipeline_depth_and_warmup`. Use the model with
care:

- **It has a validated range.** At very small steps the whole step is the non-DMA cost and an
  intercept fitted from larger steps overcharges. Hypothesis: the model is reliable above a few tens
  of KiB per step and pessimistic below; verify by fitting at your block size.
- **There is no single per-step constant.** The measured per-step slope varies with block size, so a
  "cost per grid step" quoted from one block shape is right only at that shape, and using it to rank
  block shapes is circular. Quote the model.
- Hypothesis: step cost does not scale measurably with operand count once bytes are held fixed; do not
  budget for operand count separately.
- **A step-count ladder that re-reads the same resident block does not price a step**: with no DMA and
  no address arithmetic per step it measures the loop counter. Price steps with a probe that moves a
  different block each iteration.

---

## Block size as something to probe

**Bigger blocks are usually better, up to what fits.** On a compute-bound matmul the block shape
alone can move throughput by a large factor; measure the surface on the target with
`matmul_block_size_sweep`. What to look for in its output:

1. **Is there an interior optimum?** If throughput rises monotonically to the largest block that
   fits, the practical rule is *the largest block that fits*, and the fitting is the work.
2. **Do the parallel axes trade?** Hypothesis: m and n are roughly interchangeable, shrinking one
   costs additively and shrinking two multiplicatively.
3. **Which axis is cheapest to shrink?** Hypothesis: the contraction axis k is cheapest, which is
   convenient because k is the axis you shrink to free VMEM for the other two. This conflicts with
   the deflation order in mature heuristics (below); see open questions.
4. **Time it batched.** An unbatched sweep carries the dispatch floor in every arm, compressing the
   spread and making absolute rates meaningless; only the ranking survives. Compare a Pallas matmul
   against `jnp.dot` in the same batched probe before concluding anything about the gap to XLA.

**First check the block has an axis to sweep.** If the kernel does one independent work item per
grid step, the axis is absent, not narrow, and no sweep will find it. Adding it is an algorithm
change (see [`algorithm.md`](algorithm.md)).

**Step count is a lever of its own.** Widening the block does the same arithmetic and moves the same
bytes in fewer, fatter steps, so the difference is the fixed cost of the removed steps plus a better
effective bandwidth per step. Where the fixed share dominates, the speedup approaches the ratio of
step counts. Model it before building: fit `T = F·steps + W·items`; if `F·steps` is the larger term,
the block is the lever. The prize grows when:

- every step issues its own DMAs (ragged or gathered kernels recompute addresses and wait on
  semaphores per step);
- the compute is too small to hide them (low arithmetic intensity);
- the grid is short, so nothing amortizes.

**Two independent questions:** does a bigger block improve MXU occupancy, and does it reduce step
count? A block can keep the per-item matmul the same shape (half the steps, each issuing twice the
identical passes), leaving occupancy unchanged while halving step count. Ask both.

A memory-bound step-cost curve (`block_size_bandwidth_curve`) and a compute-bound matmul sweep
(`matmul_block_size_sweep`) measure different things; do not carry one as evidence for the other.

---

## Traversal order

Holding the partition fixed, the grid order matters mainly through the contraction axis. With k
outermost, every k step re-reads and re-writes the whole output; with k innermost, the output tile
accumulates in place in scratch. **Put the contraction axis innermost.** Which parallel axis leads is
expected not to matter. Hypothesis: the penalty for k outermost is real but smaller than the extra
traffic suggests; measure with `grid_traversal_order` before contorting a design to avoid it.

---

## `dimension_semantics`

`pltpu.CompilerParams(dimension_semantics=(...))` labels each grid axis `"parallel"` or
`"arbitrary"`.

- **`parallel`** on any axis whose steps are independent.
- **`arbitrary`** on any axis that carries state across steps: an accumulator, a running max, a
  scratch read before it is written. Mislabelling an accumulating axis `parallel` is a **correctness
  bug**, not a performance one.
- `arbitrary` is also the conservative choice when a hand-written prefetch schedule assumes the
  traversal order. A backward kernel that reduces over heads, reduces over the query axis, and runs a
  hand-written kv prefetch schedule can legitimately label every axis `arbitrary`.

`parallel` exists so a multi-core chip can split an axis across cores. Read `num_cores` /
`is_megacore` from `pltpu.get_tpu_info()` (probe `tpu_info_reported_constants`); on a chip that
exposes one core per device there may be nothing to split. Whether the label has a measurable effect
on the target is `dimension_semantics_effect`. **Label correctly regardless**: it is a correctness
contract.

Under `emit_pipeline`, dimension semantics move onto the `emit_pipeline` call and take the
`pltpu.PARALLEL` / `pltpu.ARBITRARY` values rather than strings (check spelling with
`tools/search_api.py`).

---

## Partitioning idioms

Three idioms, chosen by how much of the schedule you need to control.

**1. Static grid with scalar prefetch.** `pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=n, ...)`
over a grid whose extent is known at trace time; the leading `n` arguments land in SMEM and the
`index_map`s read them to choose which block each step loads. Use when only the *indices* are dynamic
(for example ragged-group metadata choosing which group's weights a row tile uses). Probe
`scalar_prefetch_drives_index_map` (see [`precomputation.md`](precomputation.md)).

**2. Outer `pl.kernel`, inner `pltpu.emit_pipeline`: the default for anything non-trivial.** The
outer kernel has no grid and no `BlockSpec`s; it runs once per core over a mesh and receives whole,
unblocked HBM refs. Inside, `emit_pipeline(body, grid=..., in_specs=..., out_specs=...)` returns a
function; calling it with the HBM refs runs the block pipeline, and the inner body receives one set
of blocks per step. Outer scratch reaches the inner body only through an explicit `scratches=`
argument. Why prefer it:

- it has a place for **per-invocation setup and teardown** (fill a metadata buffer, issue zero-init
  DMAs before the pipeline, close them out after); a `pallas_call` grid has no code outside the grid;
- it is the path to buffer depths above 2, lookahead, traced block extents, per-phase profile scopes,
  and **a custom buffered-ref subclass, where you write the copies and the pipeline still owns the
  schedule**;
- hypothesis: it costs nothing in throughput against a `BlockSpec` grid on the same partition; verify
  with `emit_pipeline_vs_blockspec_grid`.

It is also the route when a ragged offset will not pass through a top-level `BlockSpec`: overriding
the copy-in/copy-out puts the offset in a runtime `pl.ds` the window prover never sees. A rejected
`index_map` does not mean the choice is now between padding and a hand-rolled pipeline; that is two
of the three routes. Mesh constructor spelling is toolchain-dependent; check with
`tools/search_api.py` and `pltpu_symbol_surface`. Mechanics are in [`pipelining.md`](pipelining.md).

**3. 1-D `arbitrary` grid, all operands unpipelined (`pl.ANY` or HBM), hand-rolled DMA.** Use only
when the *iteration* is irregular enough that no `index_map` describes it, not merely when offsets
are data-dependent (that is idiom 2 with a custom buffered ref). Choosing idiom 3 for data-dependent
offsets over a regular loop buys a semaphore protocol you did not need. Once here you write the
pipeline yourself; see [`pipelining.md`](pipelining.md).

---

## Ragged and uneven work

**Grouped work over tiles.** When rows belong to groups of uneven size, a group boundary inside a row
tile makes that tile visited once per group touching it. With `G` groups and `tiles_m = M / tm`, the
number of tile visits is bounded:

```
tiles_m  <=  visits  <=  tiles_m + G - 1
```

Size the grid to the upper bound, carry per-visit metadata (group offsets, group id per visit, tile
id per visit) in SMEM via scalar prefetch, and pad the metadata for unused trailing visits.

**Skip empty work at the earliest point you know it is empty**, three ways that are not
interchangeable:

- **compact it out of the grid** before launch, when emptiness is known at trace time;
- **a dynamic grid length**, when it is known at launch but not at trace time (probe
  `dynamic_grid_from_smem_skips_work`);
- **predicate inside the step** with `pl.when` / `lax.cond`, when it is known only per step.

**Boundary tiles: masked read-modify-write, not a narrowed block.** On the group's last write to a
tile, compute a row mask and `select` between the new value and the existing output. This works
because a partial tile costs most of a full tile anyway (probe `mxu_block_rows_cost`), so there is
little to save by narrowing it.

**A ragged tail.** When the array extent does not divide by the block, the last block is partial.
Measure its cost against even grids with `ragged_tail_cost`: derive the marginal cost of a full step
from two even arms, then compare the tail. Hypothesis: a partial tail costs most but not all of a
full step, plausibly because the DMA shrinks to the rows that exist while the MXU still does the full
block. That is negligible over a long grid and material over a short one. Narrowing the tail block
can recover at most the tail's cost and adds a second code path. For masking versus padding the
array, see `mask_versus_pad` in [`layout_precision.md`](layout_precision.md).

---

## VMEM heuristics: start big and deflate

Do not compute the right block analytically. Write an estimator for the block's VMEM footprint, start
from the largest candidate, and shrink until it fits the budget with headroom:

- **Estimator:** sum each input tile × its buffer count, plus a few output tiles' worth for the
  accumulator and the writeback buffer.
- **Budget:** a fraction of the scoped VMEM limit, not of reported capacity. Mature heuristics derive
  the scoped limit from `get_tpu_info().vmem_capacity_bytes` with a generation-dependent divisor and
  then apply a utilization factor below 1, deliberately leaving headroom. Calibrate yours against
  `vmem_scratch_bisect` on the target.
- **Deflation:** halve the tile, rounding down to a multiple of 128, with a floor of 128.
- **Order:** one mature heuristic deflates m first, then n, and the contraction axis k last. That
  disagrees with the hypothesis that k is the cheapest axis to shrink; resolve with
  `matmul_block_size_sweep` at the target shape, including the 128 floor.

An estimator plus a descending search beats a hard-coded tile, because the right block depends on
head count, dtype and operand count, which are the caller's. If block-size sweeps show no interior
optimum, deflate-until-it-fits is the correct strategy, not a shortcut.

---

## Deciding

1. **Enumerate every axis of the iteration space before choosing a block shape**, and write down the
   ones you are not tiling, with a reason. Derive them from the work, not the loop you picture: if two
   units of work can be independently sized, that is an axis. Two produce most misses: an axis whose
   extent is 1 in the regime you are looking at (still an axis, and where fixed per-step cost hurts
   most), and an axis that exists in how the data is grouped but nowhere in the current loop.
2. **Make the block as large as the budget allows**: `2 × sum(block_bytes) + scratch_bytes <= scoped
   VMEM limit` (`vmem_scratch_bisect`). On SparseCore use its own ledger
   ([`sparsecore.md`](sparsecore.md)).
3. **Check legality**: trailing two dims divisible by 8 and 128, or equal to the array extent.
4. **Contraction axis innermost**, labelled `arbitrary`; independent axes `parallel`.
5. **Count operands and recheck step 2**: each pipelined operand is resident twice.
6. **Probe block size against step count.** Fit `T_step ≈ F_step + bytes/BW_step` on the target; sweep
   the block up until VMEM stops you and measure (`matmul_block_size_sweep`,
   `block_size_bandwidth_curve`). On a short memory-bound grid this can beat anything inside the step
   body.
7. **Count the `pallas_call`s before tuning any of them.** Each carries a device intercept
   (`pallas_call_intercept_chained_clock`) no block choice removes. With only a few dozen steps of
   modelled work, fusing two calls beats every lever above.
8. **If the kernel is a matmul, compare it with `jnp.dot` in the same batched probe** before going
   further.

---

## Open questions

| Question | Probe |
| --- | --- |
| What moves the per-`pallas_call` intercept: operand count, scratch size, block shape, output count? Treat it as a range until swept; do not difference two kernels' intercepts. | `pallas_call_intercept_chained_clock` swept over those factors |
| What is the true block-shape spread when every arm is timed batched? | `matmul_block_size_sweep` (batched) |
| Where does VMEM turn the monotone bigger-is-better trend around, past the largest swept block? | `matmul_block_size_sweep` extended, with `vmem_scratch_bisect` |
| Is the contraction axis the cheapest to shrink, or should it deflate last? Down to the 128 floor? | `matmul_block_size_sweep` including 128 |
| Does `dimension_semantics` have any measurable effect on a single-core device? | `dimension_semantics_effect` (timed batched, above the dispatch floor) |
| Why does a ragged tail cost less than a full step: shrunken DMA, full MXU work, or something else? | `ragged_tail_cost` with a DMA-only and a compute-only arm |
| Below what bytes per step does the step-cost model stop holding on this target? | `block_size_bandwidth_curve`, `dispatch_floor` |
| Does a head-blocked (4-D, strided) kernel run above the step model because of strided transfers? | `block_size_bandwidth_curve` with a strided arm |
