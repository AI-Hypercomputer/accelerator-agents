---
name: kernel-design-method
description: >-
  The design workflow the MaxKernel planner follows to turn a job into a kernel_plan.md: system
  context, two floors, examples before API, dataflow, cost classes, affordances, unit trading, the
  sixteen contract questions, lowering, measurement rules, profile routing and re-planning. Read it
  at the start of every planning iteration; §9 lists the sections a plan must contain.
---

# Kernel design method

MaxKernel runs an autonomous loop, one iteration at a time: **plan → implement → compile → test →
autotune → profile**. There are no human gates. The planner (`maxkernel-plan-kernel`) writes
`<run_dir>/iter<N>/kernel_plan.md`; every later stage of the iteration reads it. This file is how
that plan is designed. It is a method, not a checklist of facts: the facts live in
[`tpu_hardware_facts.md`](tpu_hardware_facts.md), the design files under [`design/`](design/), and
the earned rules in [`strategies/INDEX.md`](strategies/INDEX.md). Questions that need hardware go to
the probes in [`probe_catalog.md`](probe_catalog.md).

The job's operating points are its **cases**: the input sets `get_inputs()` in
`<run_dir>/test_kernel.py` returns. Every floor, prediction and result in a plan is per case.

---

## §0 Principles

1. **Ask two questions, not one.** *Physics*: what is this computation and where must it run — the
   op DAG, the floors, the cost classes. *Affordances*: what does a competent kernel of this shape
   contain — buffering, async writes, tail paths, hoisted setup. The first finds the right design;
   only the second finds the blocking write that keeps it from reaching its floor.
2. **Elimination before tuning.** Before padding, retiling or re-buffering a structure, ask whether
   a native primitive, a fold into an existing matmul, or a different unit deletes it.
3. **Every case, every time.** Floors, predictions and results are written per case. A case that
   regresses is a regression, whatever the mean says.
4. **A correctness claim needs a fault the oracle catches.** A green test is evidence only once a
   deliberately broken variant has turned it red on the case class that matters.
5. **A performance claim needs a removal that moves the clock.** "X costs N" is a hypothesis until a
   variant without X measured about N less. Two instruments that agree are independent only if they
   reach the answer by different routes.
6. **Every number carries its origin.** Each number in a plan is one of: a probe output produced in
   this run (cite `<run_dir>/probes/<name>.txt` or the cache file), a harness or profile output of
   this run (cite the file), a source-read fact (cite the reference file or upstream path), or
   `[modelled]` (show the arithmetic). A roofline is always `[modelled]`. A number with no origin is
   `[assumed]`, and an assumed number may not rank a lever.
7. **A null holds only on its axis.** A lever that did nothing did nothing at that case, tile, dtype
   and toolchain. Record the axis and configuration, or the null hardens into a false constraint.
8. **Learn every API from upstream.** Pallas and Mosaic APIs move between toolchains, and a
   remembered signature is the most common source of wasted compiles. Find a working example with
   `{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/search_api.py -- "<api_name>"` and copy its shape.
9. **Upstream is an example, not an oracle.** jax and tokamax kernels target other generations,
   dtypes and shapes. Take the mechanism, the API usage and the test structure; re-derive the buffer
   arithmetic for this job.
10. **Zero hardcoded constants or dtypes.** Dtypes come from the inputs. Hardware constants come
    from `pltpu.get_tpu_info()` and `pltpu.get_sublane_tiling(dtype)` at run time, or from a probe
    on the target. A constant typed into the kernel is a bug on the next chip.

---

## §1 System context

Do this on paper before choosing a tile. Later sections make a decomposition fast; this one asks
whether it is the right decomposition.

1. **Write the op DAG.** Every op with shapes and dtypes, classified as exactly one of: dense
   contraction · elementwise · reduction · transcendental · data-dependent addressing · layout
   change. The class, not the op's name, is what maps to silicon. In torchax mode read the jaxpr
   for the computation and the optimized HLO for the fusion boundaries XLA already drew.
2. **Eliminate by unit capability.** Most ops have one legal unit. Check each unit's hard limits in
   [`tpu_hardware_facts.md`](tpu_hardware_facts.md) and [`design/compute_units.md`](design/compute_units.md)
   before costing anything: a unit without `dot_general` cannot hold a matmul, whatever its
   bandwidth. SparseCore placement questions go to [`design/sparsecore.md`](design/sparsecore.md).
3. **Price every unit boundary as a full HBM round trip**: the producer writes the intermediate and
   the consumer reads it back, `2 × intermediate_bytes`. A placement that is locally worse on one op
   but deletes a boundary usually wins. List which boundaries *could* be removed and what would have
   to lower for that; this list is the raw material for the fusion ladder in §2. The checks for
   whether a hand fusion beats XLA's are in [`design/fusion_boundary.md`](design/fusion_boundary.md).
4. **Record the denominator.** What fraction of the caller's time is this block? `measured` (cite
   the profile), `modelled` (show the arithmetic) or `UNKNOWN`. For a standalone job the
   denominator is the whole jitted call the harness times, including any XLA ops around the
   `pallas_call`.
5. **Concurrent units cost `max()`, not `sum()`.** An op already hidden under another is worth at
   most `max(0, T_hidden − T_cover)`. Deleting work that was overlapped buys nothing.
6. **Name the assumption behind the inherited decomposition** (the base kernel's op split, the
   reference's kernel boundaries) and check whether a newer primitive or toolchain has falsified it.
   The seven inversions in [`design/cross_area_moves.md`](design/cross_area_moves.md) are the usual
   places an inherited choice is wrong.

---

## §2 Two floors

Compute two floors per case with the roofline in [`tpu_memory_overlapping.md`](tpu_memory_overlapping.md)
§1, following [`../skill/guides/general_roofline.md`](../skill/guides/general_roofline.md) and
[`../skill/guides/general_floor.md`](../skill/guides/general_floor.md). Constants come from
`get_tpu_info()` or a probe of this run; never from memory.

| | **F1: today's decomposition** | **F2: full fusion** |
| --- | --- | --- |
| boundaries | today's op and kernel boundaries kept | none inside the block |
| HBM bytes | every intermediate written and read back | external inputs, weights and outputs only |
| time | Σ over ops of `max(bytes / BW, flops / rate)` | `max(total bytes / BW, total flops / rate)` |
| its gap | baseline → F1 is **tuning headroom** | F1 → F2 is **the fusion prize** |

Add the per-call fixed cost (measure with probe `dispatch_floor` or
`pallas_call_intercept_chained_clock`) on top of both.

**Mark F2's legality.** If §1 put two ops on different units with no handoff that lowers without
HBM, F2 is a `projection`: report it, and do not plan iterations toward it.

**Iteration targets.**

- **Early iterations aim at F1**: the per-op changes that reach the floor of today's decomposition,
  with the boundaries kept.
- **Later iterations climb a fusion ladder toward F2**, one boundary removal per iteration, ranked
  by bytes saved, each rung with its own modelled floor. Climb only if the fusion prize is above
  noise: `(F1 − F2)` against the harness's own run-to-run spread, and against the denominator.
- **Does the F1 design rule out the ladder?** Check every choice that fixes an interface (output
  layout and dtype, tiling axes, grid order). If the fused kernel could not consume it, change it now.
- **Bytes already at the minimum?** Then the gap is not traffic; the remaining levers are grid
  steps, descriptors and fixed cost. Rank them with cost class 6 (§5).

Both floors are `[modelled]`. §13 replaces their inputs with measured ones the moment a measurement
contradicts them.

---

## §3 Examples before API

Agents write plausible Pallas that does not compile, or compiles into something other than what was
meant. Design and write from working examples. Use `tools/search_api.py` for three things:

1. **Design exemplars**: the shipped kernels nearest this job in shape class (paged or ragged
   attention, grouped GEMM, splash attention, ragged gather/scatter, SparseCore kernels). Read them
   for mechanism: how they pipeline, tile, handle ragged tails, lay out scratch, split the grid. Then
   ask which of their choices were forced by a generation, dtype or shape this job does not share.
2. **One working example per API**: `pallas_call` with its grid spec, `BlockSpec` and index maps,
   `PrefetchScalarGridSpec`, `emit_pipeline`, `make_async_copy` and semaphores, memory spaces,
   `pl.when`, scratch shapes, `CompilerParams`, SparseCore primitives.
3. **Lowering questions**: "does this `jnp` op lower inside a TPU kernel", answered by the upstream
   test that covers it, or by a lowering probe from [`probe_catalog.md`](probe_catalog.md).

Record an **API ledger** in the plan: one row per API the design uses.

| API | upstream example (path and symbol, from `search_api.py`) | exact form copied (argument names, shapes, memory space) |
| --- | --- | --- |

An API with no row is written from memory and goes back to `search_api.py` before implement-kernel
is dispatched. Cite upstream by file path and symbol; do not paste upstream source into the plan.

---

## §4 Dataflow, working set, primitives

- **Pipeline diagram.** DMA in, compute and DMA out across grid steps T−1, T, T+1, with the buffer
  each stage uses. **Name the pipelining axis** and what overlaps on it.
  [`design/pipelining.md`](design/pipelining.md) covers which bandwidth applies, managed versus
  manual pipelines, and when to drop to raw DMA.
- **Working set, as a formula in the tile parameters**, per case, arithmetic shown:

  ```
  live_bytes = Σ operand blocks × buffer_count
             + Σ in-body intermediates          (temporaries one grid step holds live)
             + Σ scratch
             ≤ VMEM budget                      (vmem_limit_bytes, chosen or inherited)
  ```

  The in-body intermediates term is the one that gets skipped and the one that usually binds: it
  appears in no `BlockSpec`, no roofline term and no profile, and in a blocked algorithm it is often
  quadratic in the tile parameter the throughput curve wants to maximise. Write the formula first,
  then pick the tile from the intersection of capacity and throughput. Declared scratch that does
  not fit may fail to compile rather than spill, so treat this as a hard gate. Capacities: read
  `get_tpu_info()`, or measure with `vmem_scratch_bisect`. See
  [`design/partitioning.md`](design/partitioning.md).
- **Accumulators**: dtype and residency stated.
- **Primitive audit.** Before designing any data structure, list the native primitives the target
  unit offers (scans, cross-lane reductions, masked loads, indirect DMA, packing intrinsics) from
  [`tpu_hardware_facts.md`](tpu_hardware_facts.md) and the exemplars. Before copying an idiom from an
  exemplar, ask which primitive the original was avoiding and whether this unit has the same hole.
- **Structural deletion test.** Answer in the plan: which intermediate buffers, tables or scratch
  structures can be removed outright by a native primitive, rather than made cheaper?

---

## §5 Cost classes

Fill every row. Each row's verdict is one of:

```
examined: <work> → <critical-path contribution>  [by: ablation | overlap measurement | argued]
UNEXAMINED: <why>
```

A work count is not a verdict. Inside a pipelined kernel, work concurrent with a longer transfer
contributes `max(0, T_work − T_cover)`, often zero. An `argued` verdict is allowed at plan time and
is `[modelled]`; it must be backed by an ablation or an overlap measurement before it ranks a lever
in a later iteration or appears as an attribution (M8).

| # | cost class | instrument that reveals the work |
| --- | --- | --- |
| 1 | structural elimination | the primitive audit and deletion test (§4) |
| 2 | bulk memory traffic | F1/F2 bytes against measured time |
| 3 | per-element ALU in the innermost body | the body's op list written out by hand × trip count |
| 4 | cross-lane serializing ops | a per-chunk cost model, not the trip count |
| 5 | bank / port conflicts | latency that moves with input *distribution* at constant op count |
| 6 | launch / fixed overhead | the intercept of a call-count sweep (M2); a large intercept caps every work-deleting lever |
| 7 | boundary and layout handoff | shapes and pads at the wrapper; `2 ×` bytes per boundary |
| 8 | inter-unit concurrency | can the block overlap other work at all; `max()`, not `sum()` |

Rows 1, 3, 4 and 5 have no natural profiler view: a profile shows where time goes in the code that
was written, never the lever that was not. That is why the list is written down, and why **the
UNEXAMINED rows are a deliverable**: they tell the next iteration where nobody looked.

---

## §6 Implementation affordances

The cost classes ask where time could go. This table asks what a competent kernel of this shape
contains. Score every row `planned`, `NA: <reason>` or `absent: <reason>`. Rows may be added, never
removed. Cite the exemplar that shows each `planned` row in the API ledger.

| # | affordance | how §10 checks it in the code or artifact |
| --- | --- | --- |
| 1 | every critical-path HBM→VMEM transfer is double- or multi-buffered | BlockSpec / `emit_pipeline` buffer count ≥ 2, or the ring depth of a manual loop |
| 2 | every VMEM→HBM write is async, waited on as late as possible | no `.wait()` directly after its own `.start()` |
| 3 | tail and ragged tiles run a narrower body | a distinct path or masked shape for the last grid step |
| 4 | zero-trip and common-case fast paths | a branch ahead of loop setup |
| 5 | no host-computable work inside the jit | the region's XLA ops are only the custom call(s) plus declared ops |
| 6 | per-call setup (plans, index tables, constants) hoisted | built outside the jitted callable, or cached; see [`design/precomputation.md`](design/precomputation.md) |
| 7 | the epilogue is chunked or overlapped with the next load | it does not serialize after the last DMA |
| 8 | the accumulator stays resident across the contraction loop | no HBM round trip of partial sums |
| 9 | grid dimension semantics permit the core split | `parallel` on independent axes |
| 10 | data-dependent indices arrive by scalar prefetch | not loaded inside the kernel body |

An affordance scored `planned` and missing from the written code is a bug, caught in §10.

---

## §7 Algebra and unit trading

### Algebraic rewrites

Standing techniques, each proved against the test oracle before it is relied on:

- concatenating narrow contractions into one wide one so MXU columns do not idle;
- factoring shared terms (`Q·A + Q·(K·B) → Q·(A + K·B)`);
- online accumulation (Welford- or flash-style) instead of materialized intermediates.

[`design/algorithm.md`](design/algorithm.md) covers exact versus approximate restructurings and
accumulator precision.

### Unit trading: a search, not a list

Moving work between units, above all VPU work onto an idle MXU, is one of the largest and least
obvious levers, and no profile suggests it: a profile shows a busy VPU, never the matrix that would
empty it. Search for it on every kernel.

**Split each op into its linear and non-linear parts.** Any linear map is a matmul by some constant
matrix and can run on the MXU; the non-linear part stays. Few ops are purely non-linear: a softmax
is an `exp` plus a sum, a norm is a square, a sum and a scale. The question is "what is this op's
linear part, and what matrix expresses it".

**Try the folds first**: work absorbed into a matmul that already exists, such as a scale folded
into a weight or a transpose folded into `dot_general` dimension numbers. A fold deletes a pass
without adding MXU work.

**The decision rule: four costs, per case.**

1. **Padding.** A thin matmul (a reduction to width 1) is padded to full MXU width; price the
   padded FLOPs, not the useful ones ([`design/compute_units.md`](design/compute_units.md) has the
   row/contraction/column granularity model).
2. **Critical path and slack.** It pays only if the VPU pass is on the critical path (M8) *and* the
   MXU has idle capacity at that case. Small-row cases usually have MXU slack; large-row cases often
   bind on the MXU. Expect a trade to win at one case and lose at another.
3. **Precision.** An f32 reduction through a bf16 MXU pass loses bits; `Precision.HIGHEST` or a
   hi/lo split costs passes. The oracle decides.
4. **The constant matrix.** It needs VMEM and a source: built once and hoisted, or generated per
   tile, which is VPU work of its own.

The rule runs in reverse too (at tiny row counts a matmul can lose to the VPU) and covers the other
trades: a transpose into dimension numbers, a gather onto a SparseCore, a small matmul off the MXU.

**The unit-trading table.** Every VPU or XLU pass the cost-class verdicts put on the critical path,
and every op whose unit is still a choice after §1, gets a row:

| op | linear part | candidate formulation | padded cost per case | slack per case | precision plan | matrix source | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |

Every row ends in one of: a candidate, "linear part too small to pay" with the arithmetic, or "no
formulation found" with what was searched (`search_api.py` queries, design files, strategies). A
row may not be left out.

---

## §8 Contract questions

Sixteen questions that a design written against the math, rather than the chip, never asks. The
unit of answer is the individual value crossing an op boundary and the individual array crossing a
kernel edge, not the design as a whole. "The layout is fine" is not an answer.

**`unmodelled` is not an allowed answer.** Each row is answered from the layout, from arithmetic on
the tile parameters, or by naming a mechanism; or marked `NA: <reason>`; or answered with the name
of the probe that was run and its output file. A row whose answer would change a ranking gets its
probe in this plan, not a note for a later iteration.

`free` needs evidence exactly as much as `costs X` does. Several plausible layout rules turned out
to cost nothing when probed (see [`design/layout_precision.md`](design/layout_precision.md),
"Arranging so the relayout never exists"), so an inherited cost claim is labelled as inherited.

### Layout (per value)

| # | question | answer format | probe(s) |
| --- | --- | --- | --- |
| L1 | For every value: which logical axis is on lanes (last, 128 wide), which on sublanes, which are leading, at production and at consumption? | `(b,h)[sub=s, lane=d]` per value, both sides; never "as the compiler lays it out" | `lane_vs_sublane_reduction`, `native_tile_granularity` |
| L2 | For every sliding-window or shifted read: what is the extent of the windowed axis in registers, and what does one tap cost at that extent? Do not assume extent 1 is free. | `extent N, per-tap cost from <probe output>` · `extent N, indistinguishable from extent 1 at this k per <probe output>` · `NA: no windowed read` | `sliding_window_compact_vs_sublane` |
| L3 | For every broadcast, repeat or expand (head expansion, scalar-to-tile, batch tiling): along which axis, and is it **blocked** (each element repeated as a contiguous run, or the whole block tiled) or **interleaved** (elements alternated within the lane axis)? | `leading` · `sublane` · `lane, blocked` · `lane, interleaved, and the algebraic form that makes it blocked or moves it off lanes is <...>` | `expand_blocked_vs_interleaved`, `expand_axis_leading_vs_sublane_vs_lane` |
| L4 | For every concat or split: which axis? Where a contraction is narrower than the MXU wants, was it concatenated with a sibling that shares an operand? | `merged with <value>` · `not merged because <...>` · split point `at a 128 multiple` / `unaligned, cost from <probe output>` | `concat_to_widen_narrow_matmul`, `weight_stationary_lhs_concat` |
| L5 | For every transpose or axis swap: absorbed into a `dot_general`'s dimension numbers, expressible as a masked reduction over a degenerate axis, or a real permute? | `absorbed` · `masked-reduction form, cost <...>` · `real permute, cost <...>` | `transpose_cost`, `degenerate_transpose_via_masked_sum` |
| L6 | Does any op run at a larger multiplicity than its value is stored at (after an expand, a batch tile, a broadcast)? Ask especially of contractions on an expanded operand. | `none` · `elementwise only, compiler hoists per <probe output>` · `contraction at N×; moving the expansion after it saves <...>` | `elementwise_before_vs_after_expand`, `expand_before_vs_after_matmul` |
| L7 | What itemsize is every mask, iota and index? This is about the floor of what lowers, not cost: some narrow integer iotas and compares do not lower. | `int32` · `int16` · other with the lowering probe that cleared it | `mask_index_dtype_match`, `integer_ops_int32` |

L2, L3, L4 and L6 are identities in the math (a window is a window, `f(repeat(x)) = repeat(f(x))`)
and placement decisions on the chip. When one is expensive, the fix is usually upstream: ask what
algebraic form produces the value already on the axis its consumer wants (see
[`design/cross_area_moves.md`](design/cross_area_moves.md)) before pricing the relayout.

### Working set (per plan, as a formula)

| # | question | answer format | probe(s) |
| --- | --- | --- | --- |
| W1 | The live set as a formula in the tile parameters: operands, in-body intermediates, pipeline buffers × buffer count, scratch, each itemised. | `<term> = <formula> = <bytes at chosen tile>` per term | `operand_count_vmem_pressure`, `vmem_bytes_per_element_by_dtype` |
| W2 | The VMEM budget it is compared against, and whether that budget was chosen or inherited. | `chosen: CompilerParams(vmem_limit_bytes=<expr>)` with the capacity it is a fraction of · `inherited default, not raised because <...>` | `vmem_limit_bytes_raises_ceiling`, `vmem_scratch_bisect` |
| W3 | Which term binds, and what tile the formula selects, against what a throughput curve alone would select. | `binding term <name>; formula selects T; throughput curve alone picks T′; the gap is because <...>` (or "both pick T") | `matmul_block_size_sweep`, `block_size_bandwidth_curve` |

### Transfer (per array crossing HBM↔VMEM)

| # | question | answer format | probe(s) |
| --- | --- | --- | --- |
| T1 | Managed or manual pipeline (`BlockSpec` grid, or `emit_pipeline` / hand-written copies inside an outer call with HBM specs)? Is the grid extent static or dynamic? | `managed BlockSpec, static grid of <expr>` · `manual emit_pipeline, dynamic extent from <SMEM value>`, plus the reason | `emit_pipeline_vs_blockspec_grid`, `manual_dma_vs_managed`, `dynamic_grid_from_smem_skips_work` |
| T2 | Ragged or padded? Does the transfer move exactly the live rows or a padded worst case, and what fraction of the bytes is padding across the job's cases? | `ragged, per-step size from <metadata>` · `padded, padding fraction <expr> per case` | `ragged_tail_cost`, `mask_versus_pad` |
| T3 | Can the transfer be zero-length on grid steps that do not need it (a state read only on the first tile, a weight loaded once, a boundary value)? | `yes, predicate <expr>` · `no, every step needs it` · `no, and <fraction> of steps do not need it` | `traced_dma_length`; `zero_size_dma_needs_no_wait` is destructive and agents may not run it, so take its answer from [`design/pipelining.md`](design/pipelining.md) or [`tpu_hardware_facts.md`](tpu_hardware_facts.md) or predicate by branch |
| T4 | Is any value moved HBM→VMEM more than once across grid steps where a VMEM scratch could carry it? | `none` · `<value>, moved <count> times = <bytes>; carrying it costs <bytes> of scratch` | `scratch_persists_across_grid_steps` |
| T5 | For every array that is both read and written (state caches, in-place accumulators, an output written by two passes): is it `input_output_aliases` plus donation? If not, what does the copy cost? | `aliased, index map <...>, donated` · `not aliased; the copy is <bytes> per call because <...>` | manual: confirm `input_output_aliases` + donation in the lowered HLO |

T4 and T5 are invisible to a roofline, which counts the bytes an algorithm must move: T4 is the same
bytes moved several times because the grid lets the value fall out of VMEM between uses, T5 the
same bytes copied because nothing told XLA it may write in place. On a memory-bound block either can
exceed every in-kernel optimization combined. Padding that a ragged transfer could stop moving (T2)
is a kernel fix, not something to hand back to the caller.

### Partial tiles (correctness)

| # | question | answer format | probe(s) |
| --- | --- | --- | --- |
| R1 | For every tiled axis whose extent need not divide the tile: what does a partial tile compute? Per masked quantity, what value is it masked to, and why is that value correct (not merely finite)? | `masked to <v> because <identity>` per quantity · `the axis always divides, because <...>` | `ragged_tail_cost`, plus edge cases in the test (extents that do not divide the block, length 0 and 1) |

The correct mask value is often not zero: a multiplicative accumulator masks to 1, a max-reduction
to −∞, a decay to no-decay. A uniform-shape benchmark cannot catch this, which is why R1 is
answered in writing and backed by an edge case the oracle checks.

---

## §9 Required `kernel_plan.md` sections

Besides the plan's existing sections (strategy, tiling, expected performance), every
`kernel_plan.md` contains:

1. **Cost classes**: all 8 rows of §5, each `examined: …` or `UNEXAMINED: …`.
2. **Affordances**: all 10 rows of §6 (plus any added), each `planned`, `NA: …` or `absent: …`.
3. **Unit trading**: the §7 table, every row with a verdict.
4. **Contract questions**: all 16 rows of §8, answered per value / per array, or `NA: <reason>`, or
   the probe name and its output file.
5. **Facts needed**: every fact the design turns on, each answered by either a source-read fact
   (reference file and section, or upstream path and symbol) or a `run_probe.py` output this run
   produced, with its file path under `<run_dir>/probes/` or the probe cache. A fact that is
   neither is listed as `assumed`, and an assumed fact may not decide between two options.

A plan missing any of these sections is incomplete; implement-kernel is not dispatched on it. Also
carry the floors (§2), the API ledger (§3) and the working-set formula (§4); iteration N > 1 opens
with the modelled → measured table (§13).

Probing budget: at most four uncached probes per plan. Cache hits are free. Choose them with the
two-question rule in [`probe_catalog.md`](probe_catalog.md).

---

## §10 Lowering

For implement-kernel and fix-kernel-compilation.

1. **Translate the plan, copying each API's form from its ledger row.** Keep the wrapper (grid,
   block sizes, config) and the kernel body in separate functions, so the autotuner and the next
   iteration can reuse the wrapper. Style rules: [`pallas_code_style.md`](pallas_code_style.md).
2. **Interpreter dry-run on CPU.** `pl.pallas_call(..., interpret=pltpu.InterpretParams())` under
   `JAX_PLATFORMS=cpu` models TPU memory spaces and DMAs, and catches a wrong `index_map` arity, a
   bad `scratch_shapes` signature, a mismatched block shape, unaligned slices and a wrong algorithm.
   Shrink the grid to a step or two first; interpret mode walks the grid in Python. Its limits
   (toolchain-dependent; verify with the lowering probes):
   - plain `interpret=True` is not enough once the kernel names a TPU memory space;
   - `emit_pipeline` may refuse to run on a CPU device;
   - overlapping `pl.Element` windows may not be implemented in interpret mode;
   - SparseCore kernels may have no CPU tier (probe `sc_kernel_cannot_dry_run`);
   - interpret mode does not invoke Mosaic, so it cannot say whether an op lowers, what it costs,
     or what fits in VMEM, and an op with no TPU lowering runs happily on CPU. A dry run proves the
     script runs and the math is right; nothing more.
3. **A compile or lowering error goes to `tools/search_api.py` and a lowering probe before any
   variation.** Find the upstream test that exercises the failing API and diff the usage against it.
   If the question is whether a construct lowers at all, run the matching probe (`lowering`,
   `hardware` or `sparsecore` modules in [`probe_catalog.md`](probe_catalog.md)). A rejected
   construct is a fact about the usage until an upstream example shows it failing too. If an error
   names a compiler flag, retry with that flag before recording an absence.
4. **Affordance audit.** Check the written code, and the compiled artifact where needed, against
   every row of §6 with the check that row names. An affordance scored `planned` and absent in code
   is a bug; fix it before the test stage.

---

## §11 Measurement rules

These apply to the harness's timings, autotune sweeps, profiles and any probe or ad-hoc timing
script an agent writes.

**Is this number a number?**

- **M1. Build every callable before the timing loop.** A callable built inside the repeat loop
  recompiles every rep and times the compiler. JAX caches on the identity of the wrapped callable,
  and `functools.partial` has no value equality, so hoist the function object, not just the `jit`.
  Compile every (variant, case) pair first, then time by indexing. Hold the first rep out of the
  median; a late rep costing what the first cost means that variant is recompiling.
- **M2. Chained clocks barrier the loop carry, never the loop-invariant operands.** Otherwise XLA
  hoists the call out of the loop, M invocations run once, and the result is still correct, so no
  correctness check catches it. Defences: assert the HLO's `tpu_custom_call` count is the expected
  multiple of M; take per-call cost from the slope of a sweep over M, never total ÷ M (the intercept
  is cost class 6); treat `T(M_hi) / T(1)` near one, or any reading under the floor, as a hoist.
- **M3. Submit independent TPU jobs before harvesting any.** Use `tools/tpu_client.py` and poll;
  never sleep in the foreground. A dependent job waits: **a correctness gate that has not returned
  has not passed.** Fill the wait with work decided before submission.

**Does the number mean what the plan says?**

- **M4. Same artifact, same run, same surrounding work.** The baseline is the exact base callable,
  never a rebuild; base and candidate are timed in the same process (the harness does this). Both
  timed regions contain the same non-kernel work: compare their XLA op lists and account for any op
  in one and not the other.
- **M5. Write predictions before the run.** Signed, falsifiable, per lever and per case, in the
  plan's expected-performance section, with the condition that would make the result void rather
  than informative. A null is trustworthy only if the prediction is older than the data.
- **M6. A sweep must move its driver.** Before an autotune sweep or a probe sweep, compute the work
  driver (grid steps, tiles, bytes moved, DMA count) at every point from the same code that builds
  the kernel, and check it changes in the direction the prediction assumes. A flat driver voids the
  sweep before it costs a job.
- **M7. Correctness: show the oracle can fail.** A correctness claim names a fault the test rejects
  on the case class in question. A case class on which no injected fault is caught is reported as
  blind, not as passing.
- **M8. Performance: show the cost can be removed.** "X costs N" needs a variant without X (allowed
  to be wrong, labelled timing-only) whose delta is about N at that case. A gap split across parts
  is a sum of direct per-part measurements published against an independently measured whole; the
  difference is **unattributed**. One-at-a-time ablations do not sum to the whole when parts
  contend for the same slots. Two instruments reading the same quantity count once. Until backed
  this way an attribution is `[modelled]`: it may suggest a probe, never justify a redesign.
- **M9. Measure at the scope the work lives, and compose with a ladder.** A lever that deletes work
  between stages reads as zero on either stage's clock; decide on whole-call time per case against
  the base, never on a kernel-scope roofline fraction. Levers measured against a shared baseline
  cannot show interaction: confirm a stack with a forward ladder (each rung = previous + one lever)
  plus one leave-one-out. Disagreement beyond the harness's spread means the levers interact; say
  so rather than publishing per-lever numbers. No case may regress.

Probe output follows its own rules (verdicts, `INVALID`, ratios with named denominators): see
[`probe_catalog.md`](probe_catalog.md). Instruments that misreport are listed in
[`tpu_hardware_facts.md`](tpu_hardware_facts.md) §9.

---

## §12 Routing from the profile

The profile summary (and the LLO dump, when one was captured) says what binds. Route the next plan
on it:

| signal | route |
| --- | --- |
| XLU transposes / relayouts | §7 and §8 L1–L6: layout, scan order, or a unit trade into dimension numbers; [`design/layout_precision.md`](design/layout_precision.md) |
| VPU saturated, on the critical path | §7 unit trading |
| DMA wait | §6 rows 1, 2 and 7 (buffering depth, async writes, epilogue overlap) and the pipelining axis; [`design/pipelining.md`](design/pipelining.md) |
| spills | the working-set formula (§4, W1) and the tile knee, then autotune |
| ratio to floor **widens** as work is removed | the floor models the wrong resource; fixed cost binds. Go back to §1 or cost class 6 (measure with `dispatch_floor`, `pallas_call_intercept_chained_clock`), not another sweep |

Cap each question at two or three probes or iterations. Not converging means the plan's model is
wrong, not the parameters. Before re-planning, check [`strategies/INDEX.md`](strategies/INDEX.md)
for a rule about the signal in front of you.

---

## §13 Re-plan when a modelled number is refuted

**Trigger:** a measurement from the previous iteration (harness time, autotune summary, profile,
probe) lands outside the predicted range of a modelled number the lever ranking depended on: a
floor, a byte count, a regime split, an attribution.

**Action, in the next plan:**

1. Open `kernel_plan.md` with a **modelled → measured** table:

   | quantity | modelled (iter N−1) | measured (file) | consequence |
   | --- | --- | --- | --- |

2. Recompute the floors and the lever ranking from the measured values.
3. Re-answer every §5, §6 and §8 row the refutation touches.
4. Do not build on a lever the last iteration refuted. Predictions made in the previous plan are
   still scored against that plan.

A fusion-ladder rung (§2) always opens with this table, using the previous rung's measured numbers.
