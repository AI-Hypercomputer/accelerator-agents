---
name: precomputation
description: >-
  What a kernel computes once and where it keeps it: the five-level hoisting ladder for scalar and
  index metadata (trace time, model load, per call in XLA, once per invocation in SMEM scratch, per
  grid step), the four ways to get metadata into SMEM, how to spend it (scalar prefetch into
  index_map, dynamic grid length) and the SMEM budget. Read when a kernel needs tile counts, group
  offsets, block masks or index maps that are not its operands.
---

# What is computed once, and where it is kept

## Questions and probes

| Question | Probe(s) (module `precomputation`) |
| --- | --- |
| Does a scratch written at grid step 0 still hold its value at step N? | `scratch_persists_across_grid_steps` |
| Can a prefetched scalar array choose which block each grid step loads? | `scalar_prefetch_drives_index_map` |
| Which dtypes may metadata use in SMEM, as scratch and as an operand? | `smem_metadata_dtype_support` |
| Does Mosaic hoist grid-invariant work out of the grid, or is it paid every step? | `grid_invariant_recompute_cost` |
| Does XLA compute shared metadata once when several `pallas_call`s ask for it? | `metadata_cse_across_pallas_calls` |
| What does building metadata in-kernel cost per scalar written to SMEM? | `smem_scalar_fill_cost` |
| Is metadata cheaper built in XLA outside the kernel or by a scalar loop inside it? | `precompute_outside_versus_inside` |
| Does a grid length read from SMEM actually skip the excluded steps, and what fixed cost remains? | `dynamic_grid_from_smem_skips_work` |

Related probes in other modules: `smem_scratch_bisect` (module `partitioning`: how many int32
scalars fit in an SMEM scratch), `traced_dma_offset_by_dtype`, `traced_dma_length` and
`dynamic_index_lane_versus_sublane` (module `hardware`: which runtime values a DMA or index accepts),
`dispatch_floor` (per-call and per-step fixed cost). Run with
`{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/run_probe.py --pick <probe_name> --run_dir <run_dir>`.

**Scope.** Scalar and index metadata, and grid-invariant tables: tile counts, group offsets, block
masks, sequence-to-tile maps, index arithmetic derived from shapes. Whether to keep a *tensor*
intermediate is [fusion_boundary.md](fusion_boundary.md); recomputing an activation instead of
storing it is [algorithm.md](algorithm.md).

---

## The ladder

Every piece of metadata sits at exactly one of five levels, outermost first:

| Level | When it runs | Cost per model step | Can depend on |
| --- | --- | --- | --- |
| 1. trace time (Python/numpy) | once per compilation | zero | static shapes and config only |
| 2. model load | once per process | zero | weights, static config |
| 3. per call, in XLA, handed in via SMEM | once per layer, or once per step if CSE holds | an XLA op plus a transfer | anything traced |
| 4. once per kernel invocation, into SMEM scratch | once per `pallas_call` | a scalar-core loop | anything in SMEM |
| 5. per grid step | every step | the work × the grid | anything |

**The rule: place each quantity at the outermost level whose invariance it genuinely has.** Too far
in and you pay repeatedly for an answer that does not change. Too far out and it is stale: metadata
hoisted to trace time that actually depends on batch content is a correctness bug that reproduces
only on shapes you did not compile for — the worst failure in this file, because everything else is
merely slow.

### Level 1 — trace time

Build the metadata in numpy while tracing, from static shapes and config. If the same kernel is
constructed once per layer with the same inputs (an attention mask shared by every layer), memoize
the trace-time builder with `functools.lru_cache` so compile time does not grow with depth. No
Pallas API is involved. Two cautions: the cache helps only if the key really repeats (a per-layer
value thrashes it silently), and a batch of ragged sequences has a static padded *shape* but
non-static *content* — metadata nearly always depends on the content, which rules level 1 out.

### Level 2 — model load

A placement, not a mechanism. Weight reshapes, transposes, dtype casts and layout preprocessing
executed in the kernel wrapper on every call belong at model load. The hoisting code lives in the
serving framework rather than the kernel, so it is the level most often missed. Flag it in the plan
even when out of scope: "operand X is preprocessed per call and could be preprocessed at load".

### Level 3 — per call, in XLA, handed in through SMEM

The common case. Compute the metadata with ordinary jax before the `pallas_call` (counts with
`pl.cdiv`, offsets with a cumulative sum, a tile→item map with `jnp.repeat`) and pass it as an SMEM
operand.

- **Data-dependent metadata has a data-dependent length; jit needs a static one.** Pad to the static
  maximum (e.g. `jnp.repeat(..., total_repeat_length=MAX)`) and carry the true count as a separate
  scalar. Every consumer must respect the count, not the length.
- **Map a whole metadata pytree to SMEM in one line**: register the metadata container as a
  dataclass pytree and `jax.tree.map(lambda _: pl.BlockSpec(memory_space=pltpu.SMEM), meta)`. The
  cost is positional operand indexing (e.g. in `input_output_aliases`), which must account for the
  number of metadata leaves.
- **CSE across layers is what makes level 3 cost once per step instead of once per layer.** Built
  inside a per-layer helper from identical arguments, XLA can deduplicate it; verify with
  `metadata_cse_across_pallas_calls`, which counts surviving ops against a control with distinct
  metadata per call. CSE needs the *same* traced value, not an equal one, and a probe whose
  `pallas_call`s are identical may also collapse the kernels themselves — real layers pass different
  weights, so check that only the metadata subgraph is shared.

### Level 4 — once per kernel invocation, into SMEM scratch

The alternative fork: pass the raw inputs (e.g. per-group sizes) into SMEM, and have the kernel
build its own metadata into an SMEM **scratch** as the first thing it does, before any grid work.
The pattern, abstractly:

- Declare the scratch in `scratch_shapes` as `pltpu.SMEM((MAX,), jnp.int32)` per table, optionally
  grouped in a registered dataclass so it arrives as one argument. Size `MAX` from the worst case,
  derived from shapes: for tiles over groups, every group boundary can add at most one partial tile,
  so `MAX = num_groups + cdiv(total_rows, tile_rows) - 1`.
- Pin already-scalar inputs to SMEM at the call site with
  `pltpu.with_memory_space_constraint(x, pltpu.SMEM)`.
- Fill under the first grid step with a nested scalar loop (outer over groups, inner over that
  group's tiles), writing per-tile entries such as owning group and start offset, and return the
  true count.
- Store offsets as a **fencepost** table of length `MAX + 1`, writing entry `i` and `i + 1` on the
  same iteration, so a consumer reads a tile's start and end with two adjacent loads and no bounds
  logic.

Level 4 needs scratch to survive across grid steps (it is allocated per `pallas_call`, not per
step); confirm with `scratch_persists_across_grid_steps`. Had it been false, the pattern would read
stale-but-plausible values rather than fail.

**What level 4 buys:** no extra HLO op, no transfer, no reliance on CSE, metadata available from
grid step zero. **What it costs:** a sequential scalar loop at the head of every invocation.

### Level 5 — what not hoisting costs

Mosaic may hoist the *producer* of a grid-invariant value out of the grid loop (or fold it, if it is
a compile-time constant) and still not hoist the *reduction, search or index computation* that
turns it into the value consumed. So: **hoist the value you consume, not the intermediate that
produces it.** hypothesis: hoisting a grid-invariant table into VMEM scratch that is still reduced
every step recovers only a small part of the gap; hoisting the finished scalar into SMEM recovers
most of it. Verify with `grid_invariant_recompute_cost` (arms: recompute / VMEM-hoisted table /
SMEM-hoisted scalar). Note whether its invariant is iota-derived — a compile-time constant can be
folded, so the probe then underestimates the cost for metadata derived from runtime inputs.

---

## What the hoist costs

- **In-kernel fill (level 4)** has a per-scalar slope; measure it with `smem_scalar_fill_cost`, which
  sweeps the number of scalars written. Quote the slope, not the intercept (the launch floor varies
  between runs). A fill of a few thousand entries is typically lost in the launch floor; tens of
  thousands is a real term. Multiply the slope by the worst-case `MAX` from the formula above to
  know which side a given shape falls on.
- **Level 3 versus level 4** at a modest size: measure end to end, XLA op and kernel both inside the
  timed region, with `precompute_outside_versus_inside`. hypothesis: at around a thousand entries the
  two are indistinguishable and runs may disagree on direction. Then choose on other grounds:
  - level 3 is easier to write, test and read, composes with jax, and parallelizes prefix sums in
    XLA — prefer it once the entry count is large enough that the scalar-loop slope matters;
  - level 4 removes the dependence on CSE and on operand plumbing, and keeps metadata available to
    the body without extra prefetch operands.

---

## Getting metadata into SMEM — the four ways

| Way | Spelling | Use when |
| --- | --- | --- |
| operand, whole array in SMEM | `pl.BlockSpec(memory_space=pltpu.SMEM)` | built in XLA (level 3); the body reads it |
| operand, prefetched before the grid | `pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=N, ...)` | the `index_map`s need it, not just the body |
| scratch | `pltpu.SMEM((n,), dtype)` in `scratch_shapes` | built in-kernel (level 4) |
| constraint at the call site | `pltpu.with_memory_space_constraint(x, pltpu.SMEM)` | an already-scalar operand that should not go to VMEM |

`num_scalar_prefetch=N` makes the first `N` operands available to every `index_map` as trailing
arguments after the grid indices, and to the body. That is why the mechanism exists: metadata that
reaches an `index_map` moves the DMA rather than the compute, so the body never touches it.
Confirm the scalars are actually used (not ignored) with `scalar_prefetch_drives_index_map`, which
drives a permutation and checks the loaded block order.

**SMEM is effectively 1-D.** For N-D logical indexing, wrap the buffer in a small registered
dataclass holding the data and a static shape, and compute the flat position from
`pl.strides_from_shape(shape)`. Write it yourself; it is a few lines. Whether N-D index arithmetic
inside an `index_map` is as cheap as 1-D has not been measured.

**A metadata ref carried inside a registered pytree must be a static field.** It points at memory
allocated elsewhere; left dynamic, it becomes a tree leaf the tracer tries to move.

**Spending metadata through a custom pipeline buffer.** Subclassing `pltpu.BufferedRef` and having
`copy_in`/`copy_out` read offsets from a metadata ref lets an irregular access pattern keep
`emit_pipeline`'s schedule instead of a hand-rolled pipeline. The table's shape and the fetch
mechanism are one decision; see [pipelining.md](pipelining.md). Scalar prefetch into `index_map`s
and a plain SMEM ref read by the body are cheaper to write; use the subclass only when addresses are
dynamic *and* you want the pipeline's schedule.

---

## Spending it

In decreasing order of value:

1. **As the grid length.** Read the true tile count from SMEM and pass it as the grid of an inner
   `pltpu.emit_pipeline`. The top-level `pallas_call` grid holds the static allocation; the dynamic
   grid runs inside it. Allocation is the padded maximum; work is the true count.
2. **In an `index_map`, to move the DMA.** For a ragged grouped matmul, each grid step indexes the
   per-tile tables by its grid index to pick (a) which expert's weight block to load and (b) a row
   window whose start and length are both traced — derived from the fencepost offsets, rounded out
   to the sublane tile, and expressed as `pl.ds(start, size)`. The grid walks only the rows each
   group owns instead of masking a dense grid. Which traced values a DMA accepts (offset, length,
   lane vs sublane index) is toolchain-dependent; verify with `traced_dma_offset_by_dtype`,
   `traced_dma_length` and `dynamic_index_lane_versus_sublane`, and see
   [tpu_hardware_facts.md](../tpu_hardware_facts.md) §8.
3. **As a DMA offset or size** in a hand-issued copy.
4. **As a scalar the body reads.** Cheapest to write, least valuable.

**A dynamic grid skips work, but a fixed cost dilutes the win.** Measure with
`dynamic_grid_from_smem_skips_work` (one compiled kernel, two SMEM grid lengths) and fit
`T = fixed + per_step · steps`. Shortening the grid recovers the transfer and compute of the skipped
steps and nothing else, so the speedup at padding factor `P` (padded max over true count) is
`(fixed + per_step · max_steps) / (fixed + per_step · max_steps / P)`. hypothesis: the fixed term is
large enough that the win is real at high raggedness and small at low raggedness; compute it from
the fit before building the metadata. Whether the fixed term scales with the invocation or with the
padded allocation is open — if the latter, capping the allocation matters more than shortening the
grid.

That formula bounds **grid-shortening alone**. When the same metadata also narrows an intermediate
(a candidate array, a gather, rows launched), the win is not bounded by it, and at the same padding
factor it can differ by shape. Price the intermediate; do not let the grid-only ratio talk you out
of the change.

---

## The SMEM budget

Read `smem_capacity_bytes` from `pltpu.get_tpu_info()`; it is per core and not divided by anything
([tpu_hardware_facts.md](../tpu_hardware_facts.md) §1, §6). The usable scratch capacity is measured
by `smem_scratch_bisect` (bisects an int32 scratch); hypothesis: nearly all of it is addressable,
with a small reserved headroom. Most kernels never approach it. The exception is metadata that is
O(q_blocks × kv_blocks) — block-sparse attention masks — which grows quadratically in sequence
length.

The answer there is to **narrow the dtype to the smallest that holds the range**, with the range
derived from the *shape* (e.g. `q_blocks * kv_blocks` for a block index), not from the data, so the
dtype choice stays a trace-time constant and the build stays at level 1. Which dtypes SMEM accepts
as scratch and as operand (int32, int16, int8, uint32, bool, float32 are the candidates) is
toolchain-dependent; verify with `smem_metadata_dtype_support`. Accepted is not the same as
smaller: `smem_scratch_bisect` runs at int32, so that an int8 scratch of `4n` fits where int32 of
`n` did is an inference from itemsize. If close to the ceiling, bisect the dtype actually used.

Budget per table: `MAX_entries × itemsize`, plus one for a fencepost, plus the scalar count.

---

## Deciding

Work the list of quantities, not the list of levels. For each thing the kernel needs that is not an
operand:

1. **Name the axis it is invariant along** — grid, layer, decode step, process. This is the only
   step that can produce a correctness bug, so be pedantic: a quantity that depends on batch content
   is not invariant across steps however static its shape.
2. **Place it at the outermost level that axis allows.** Static shapes and config only → level 1,
   memoized if rebuilt per layer. Weights → level 2 (say so even if out of scope). This call's data →
   level 3 or 4.
3. **Between 3 and 4, do not decide on speed at modest sizes.** Decide on: does an `index_map` need
   it; is CSE across layers load-bearing (4 does not depend on it); is the entry count large enough
   that the per-scalar fill slope (`smem_scalar_fill_cost`) argues for XLA's parallel build (3).
4. **Hoist the value you consume, not the intermediate.** If the kernel consumes a scalar, hoist the
   scalar.
5. **Size the SMEM**: static maximum from the shape, true count carried separately, narrowest dtype
   whose range is known from the shape, against the capacity from `smem_scratch_bisect`.
6. **Decide how it is spent**, in order of value: grid length, `index_map`, DMA offset or size, body
   scalar.
7. **Check the padding factor before believing a dynamic-grid win** — then check what else the
   metadata shrinks, and price that separately.

The plan should carry one row per quantity: the invariance axis, the level, the memory space and
bytes, and for anything left at an inner level, the per-rebuild cost and rebuilds per step.

## Gaps

- `grid_invariant_recompute_cost` uses an iota-derived (compile-time constant) table, so it cannot
  distinguish constant folding from loop-invariant hoisting; an arm seeding the table from a kernel
  operand would.
- The origin of the dynamic grid's fixed cost (invocation vs padded allocation vs intermediate
  allocation) is unresolved; an arm holding intermediates at the padded maximum while shortening the
  grid would separate them. A two-point fit is not a line.
- `metadata_cse_across_pallas_calls` does not isolate the realistic case (distinct weights, shared
  metadata); give each call a distinct weight operand and recount.
- Narrow SMEM dtypes are accepted but not shown to save capacity; re-run `smem_scratch_bisect` at
  int8 and int16.
- Level 2 has no measurement: the per-call cost of a weight transpose that could move to load time
  has not been probed.
- N-D SMEM indexing via `pl.strides_from_shape` inside `index_map`s has not been exercised.
- No metadata cost has been measured at a realistic decode shape (paged attention, MoE routing);
  synthetic kernels only.
