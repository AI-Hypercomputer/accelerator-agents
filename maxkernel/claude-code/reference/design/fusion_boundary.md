---
name: fusion-boundary
description: >-
  Decision procedure for where a Pallas kernel's boundary falls: which ops to pull inside, which to
  leave in XLA, and how to price each choice. Read when proposing a new kernel over XLA code, when
  deciding whether an adjacent op comes inside an existing kernel, or when a kernel is too big to fit.
---

# Where the kernel boundary falls

## Questions and probes

| Question | Probe(s) |
| --- | --- |
| Does XLA already fuse a chain of elementwise ops as well as one Pallas kernel would? | `elementwise_chain_xla_vs_pallas` |
| Is a matmul with its bias-and-activation epilogue fused into one kernel faster than XLA's version? | `matmul_epilogue_fusion` |
| What does materializing an intermediate to HBM cost, at a known size? (check A) | `intermediate_hbm_roundtrip`, with `hbm_bandwidth_arms` for `BW_measured` |
| How does the largest workable block shrink as a kernel takes on more operands? (check C) | `operand_count_vmem_pressure`, `vmem_scratch_bisect` |
| What does fusing a matmul and its transposed-lhs twin into one grid cost? (check D) | `gmm_tgmm_grid_axis_conflict` |
| Does fusing the producer in let a hand-issued accumulator DMA hide behind compute? | `fusion_enables_dma_compute_overlap`, with `manual_dma_vs_managed` as the control |

Run any of them with:

```bash
{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/run_probe.py --pick <probe_name> --run_dir <run_dir>
```

The mechanics of fusion across the Pallas/XLA boundary (the fuser API, `allow_input_fusion`,
`pl.CostEstimate`, async kernels split across HLO) are in
[`../pallas_xla_interaction.md`](../pallas_xla_interaction.md). This file does not repeat them; it is
the procedure for deciding *what* to fuse, not *how*.

**This file does not reject fusions.** Every cost below has a fix route. Working through it produces
a list of **fusion options**: fuse everything at this VMEM setting; fuse everything at a smaller
block and pay the MXU efficiency; fuse only the subset that shares a contraction axis; keep the
boundary and pay the round trip. Each option carries a price, a compile precondition, and the next
thing to measure. A blocker is an option with a price, never a verdict. Rank the options together at
the end, holding every price at once.

---

## Which question are you asking

Three different questions show up under "fusion", and each has a different first move. Name yours
before reading on.

| Your situation | The question | Start at |
| --- | --- | --- |
| XLA-lowered code, no kernel yet | **Is a kernel worth writing at this boundary at all?** | check **A**, then the prior below |
| A kernel exists; an op sits just outside it | **Does this op come inside?** | check **B**, then **C** to **F** |
| A kernel exists and is too big: it spills or will not compile | **What comes out, and does anything have to?** | check **C**, then the profile read-back |

**The prior, for the first question.** XLA's fuser is the baseline, not the thing you rescue the code
from. Hypothesis: on a pure elementwise chain (one input, one output, no reduction, transpose, gather
or multi-consumer value) XLA and a single Pallas kernel are roughly equal; verify with
`elementwise_chain_xla_vs_pallas`. Hypothesis: on a matmul epilogue, a well-blocked Pallas kernel is
close to XLA but not ahead, and most of any gap is the block shape rather than the fusion; verify with
`matmul_epilogue_fusion`. So "we fuse the elementwise ops" is not, by itself, an advantage. **The
dominant risk in a fused kernel is that you block it worse than XLA would, not that fusing is
expensive.** Beating the prior is a result worth having; record it either way.

---

## The six checks

Run them in order. Each one produces an option with a price, or a dismissal with a reason. **None of
them ends the exercise.** The "decided from" column says what settles it; where it says "measure",
reasoning will not do.

| # | Check | The signal that it bites | Decided from |
| --- | --- | --- | --- |
| **A** | Is there a round trip to save, and how big? | an intermediate crosses HBM between the two ops | arithmetic: `2 × intermediate_bytes / BW_measured` |
| **B** | Is the intermediate already resident for another reason? | an accumulator the loop needs anyway | reading the kernel |
| **C** | Does the live set still fit, and at what block size? | operand count up, block rows down; or it does not compile | VMEM arithmetic, then VMEM used in the profile |
| **D** | Do the ops contract the same axis? | one accumulator goes from a tile to the whole output | `O(K·N)` arithmetic, then compile it |
| **E** | Do the two sides want different layouts? | a transpose or pack/unpack appears between them | relayout cost against the saved bytes |
| **F** | What happens to registers, bundles and units? | spills, NOP-filled bundles, one unit idle | **the profile summary / LLO dump: measure, do not reason** |

**A and B are the reasons to fuse; C to F are the prices.** A is the only one you can settle without
touching hardware, which is why it goes first.

Each check writes its result as an option entry of this shape:

```
check C (VMEM): <N> operands at <cols> columns caps the block at <R> rows (operand_count_vmem_pressure).
  option C1: fuse all, block R rows        -> compiles; MXU at full row efficiency
  option C2: fuse all, block R/2 rows      -> compiles; MXU cost at this shape: probe mxu_block_rows_cost
  option C3: split off the two widest operands -> 2 kernels; round trip of <bytes> back on (check A)
  next: read VMEM used for C1 from the LLO dump before committing
```

**A check you cannot price is written down as `unmodelled`, with the probe that would settle it.**
That is a complete answer and not a reason to drop the option.

---

## Check A: what the round trip costs

Two kernels joined by an HBM intermediate write it once and read it once. Fusing them removes both:

```
saving_seconds ≈ 2 × intermediate_bytes / BW_measured
```

`BW_measured` is the achievable read-write HBM bandwidth on the target, from probe
`hbm_bandwidth_arms`, **not** the bandwidth the platform advertises (which is higher than what a
kernel reaches). Confirm the model on the target with `intermediate_hbm_roundtrip`: it times two
kernels via HBM against one kernel at the same block, and the implied avoided-traffic rate should
land near `BW_measured`. A saving far above that ceiling means the arms differ in something besides
the round trip; far below means the round trip was partly hidden.

How to use the number:

- It is a bound for deciding whether to bother, not a prediction. If the avoided traffic would have
  moved in small steps, the real saving is larger, because small steps move bytes at a worse rate
  (see [`pipelining.md`](pipelining.md); probe `block_size_bandwidth_curve`).
- If it is small against the op's own runtime, fusing buys nothing **by this mechanism**. That is not
  the same as buying nothing: check B, the schedule argument below, and unit balance (check F) are
  separate reasons.
- The fraction saved is largest when the op is *nothing but* the round trip; real compute in the
  kernel dilutes it.

Do not carry the elementwise-chain prior into an option's rejection. It is scoped to a pure chain and
says nothing about one that contains a reduction, transpose, gather, multi-consumer value or matmul,
or about a chain that is one arm of a larger kernel where the fusion buys a hiding place for a
transfer.

---

## Check B: fusing onto something already resident

The cheapest fusion there is. If the intermediate is already in VMEM for a reason the fusion did not
invent (the canonical case is a matmul accumulator held in scratch across the k-loop), then applying
the activation, a group mask, or a read-modify-write of the output tile on the last k-step costs
only the VPU work on that tile. Price the activation with `transcendental_relative_cost`; most
elementwise functions cost a small multiple of a multiply per element.

A cache update that is an in-place write of an operand is the same shape of win: make it an
`input_output_aliases` entry rather than a separate op.

**Fuse onto an intermediate that is already resident. Restructuring a kernel to *make* one resident
is a different proposal** and goes through checks C to F like any other.

### The other reason to fuse: a schedule you could not otherwise write

Check A is about bytes not moved. There is a second argument, about bytes moved at a better time,
which the `saving_seconds` formula does not capture.

`BlockSpec`'s `index_map` is a pure function of the grid indices. Three things fall outside it, and
each is a reason to put an operand in `pl.ANY` / HBM and issue its copies yourself:

1. a DMA whose address depends on a value **computed inside the kernel** (scalar prefetch covers only
   values known before the kernel starts);
2. a traversal order that is not the grid's: reverse time, or a sequence derived arithmetically;
3. an accumulator too large for VMEM, which lives in HBM and is read-modify-written each step.

Once you take a schedule over, the compute inside the kernel is the budget you hide transfers
behind. A kernel with one op has nothing to overlap; fusing the producer in creates the hiding place.
Taking the schedule over is per operand, not per kernel: the other operands stay on ordinary
`BlockSpec`s. When the carried state is a recurrence (several outputs all reading state that would
otherwise be re-read from HBM every chunk), the fusion and the schedule are the same decision.

Rules that bound this argument:

- **Hand-roll only for an operand the `index_map` cannot address at all**, not to get a transfer
  issued earlier. Hypothesis: Mosaic already hoists a hand-issued `.start()` under a managed grid, so
  issuing early by hand buys nothing; verify with `fusion_enables_dma_compute_overlap` (early vs late
  arms).
- Hypothesis: a naive hand-roll that does `write.start(); write.wait()` in the same step exposes
  every write and can lose to not fusing at all; the same probe's `split` arm checks this. Defer the
  write wait into the next step's compute if you hand-roll.
- Hypothesis: on a pure copy with nothing to overlap, the managed pipeline beats a hand-rolled serial
  DMA; verify with `manual_dma_vs_managed` and `emit_pipeline_vs_blockspec_grid`.
- These rules are scoped to a TensorCore `pl.pallas_call` grid, where Mosaic does the hoisting. A
  kernel with no managed grid (for example a SparseCore `pl.kernel` on raw refs) has nothing hoisting
  for it, and hand-issued early prefetch may pay there. See [`sparsecore.md`](sparsecore.md).

Treat the schedule argument as a reason to look, not as a priced win.

---

## Check C: the VMEM squeeze

The blocker that produces the most options, and the one most often stated as a verdict when it is
really a block size.

Every operand needs a resident block, and pipelined blocks are double-buffered by default. So each
extra operand eats into the block you can afford: roughly, each doubling of operand count halves the
block rows. Measure the actual collapse on the target with `operand_count_vmem_pressure` (it ladders
block rows for N summed inputs; a roughly flat implied footprint across N is what identifies VMEM as
the binding constraint). The ceiling itself, and how `vmem_limit_bytes` moves it, are in
[`partitioning.md`](partitioning.md#what-fits).

### The cascade, and where to break it

```
more ops fused
  -> more live intermediates
    -> VMEM overflows, or the block must shrink
      -> smaller blocks move bytes less efficiently          (pipelining.md)
      -> on the MXU, a block below the native row count costs
         more per unit of work (probe mxu_block_rows_cost)     (compute_units.md)
      -> the space for double/triple buffering goes first      (pipelining.md)
        -> the transfer lands back on the critical path
```

**A fusion that only fits at a block too small to feed the MXU has not fused anything**, but that is
a statement about this block size, not about the fusion. Four routes out; carry the ones that apply
as options:

| Route | What it costs | Where it is priced |
| --- | --- | --- |
| narrow a dtype (storage, movement and accumulation are separate decisions) | numerics; needs an accuracy gate | [`layout_precision.md`](layout_precision.md) |
| drop a buffering level on the operand that does not need it | that operand's transfer is exposed | [`pipelining.md`](pipelining.md) |
| recompute an intermediate instead of holding it | FLOPs, possibly free on an idle unit | [`algorithm.md`](algorithm.md), [`cross_area_moves.md`](cross_area_moves.md) |
| split off the widest operand into its own kernel | its round trip | check A |

### Separate the block from the fusion before you blame either

A fused kernel that loses to XLA is usually blocked badly, not fused badly. **Run the unfused Pallas
arm (same kernel, same block, epilogue deleted) before the fused one.** `matmul_epilogue_fusion` does
exactly this: it reports `fused/xla` and `fused/plain` per block shape. Read them together:
`fused/xla` moving a lot across block shapes while `fused/plain` stays roughly constant means the
block is the cost and the epilogue is a small, stable tax. That one arm is what tells you whether a
later disappointment is the fusion or the schedule.

---

## Check D: contraction-axis conflict

**Two ops can only share a grid cheaply if they contract the same axis.** When they do not, fusing
turns one accumulator from a tile into whole-output residency, and the question becomes whether it
fits, not whether it is faster.

Example: `out[m,n] = Σ_k lhs[m,k]·rhs[k,n]` and its weight gradient `dw[k,n] = Σ_m lhs[m,k]·dout[m,n]`.
Apart, each gets a grid whose innermost axis is its own contraction and accumulates in a one-block
scratch. Fused, only one can have the innermost axis; the other's accumulator stays resident across
the whole grid, which for `dw` is all of `K×N`.

**Size the resident accumulator first**: `K × N × accumulator_itemsize`, against the VMEM ceiling.
It is one multiplication, and it is the difference between an option and a compile failure found
after the kernel is written. The trap is that the accumulator grows as `O(K·N)` while the tile it
replaces is constant, so the fused version can look like a small win right up to the size where it
stops compiling (`RESOURCE_EXHAUSTED ... vmem`). Measure the curve with
`gmm_tgmm_grid_axis_conflict`.

Routes out, if the fusion is worth wanting:

- reassociate so both ops contract the same axis (see [`cross_area_moves.md`](cross_area_moves.md));
- accept the whole-output accumulator where it fits, and gate the kernel to fall back above that size;
- keep the split. Mature grouped-matmul implementations keep the forward and transposed-lhs matmuls
  as separate kernels for this reason: it is what the VMEM budget leaves, not a style choice.

A traversal conflict is the same problem with no reassociation available: a forward and a backward
walking the same axis in opposite directions cannot share a grid at all.

---

## Check E: layout conflict across the boundary

Ops with incompatible axis expectations in one body force transposes and pack/unpack between
stages: work that exists only because of where the boundary was drawn. It is the blocker most often
reported as "the fusion does not work", and most often dissolved by changing something *outside* the
kernel.

The comparison is arithmetic: relayout cost (probe `transpose_cost`; see
[`layout_precision.md`](layout_precision.md) and [`../tpu_hardware_facts.md`](../tpu_hardware_facts.md))
against the saved bytes from check A. Subtract. Three properties of relayout cost that are easy to
get backwards:

- **Tile alignment, not the axis, is what costs.** A shift by a whole native tile (8 sublanes, 128
  lanes) is cheap; a shift by 1 costs on either axis. Probe `lane_split_alignment`.
- **A matmul operand's transpose is usually free**: `dot_general` dimension numbers absorb it.
- **Price it in the dtype you will use.** Values a kernel transposes are usually bf16 activations; if
  your relayout cost is only known for f32, tag the option `unmodelled` and run `transpose_cost` in
  the real dtype.

Fix routes are layout changes upstream of the kernel: flatten to one big array, move a head dimension
out, defer a repeat past the op that would have had to transpose it, reassociate so the needed axis
is already the contracting one, or spend FLOPs to buy the layout. See
[`cross_area_moves.md`](cross_area_moves.md). **A layout conflict is an option with one of those
routes attached, not a reason the fusion is off.**

---

## Check F: the unpriced blockers

Real, invisible in a roofline, and none produces a compile error. Say so in the fusion options and
measure them.

| Blocker | What it looks like | How to see it |
| --- | --- | --- |
| **Vector register spilling** | the body outgrows the register file long before VMEM is full; spills and fills land in the inner loop | spills + fills per bundle in the LLO dump |
| **VLIW scheduling breakdown** | the dependency graph is too tangled to fill issue slots; bundles come back sparse with NOPs; unrolling regresses | bundle timeline / bundle density in the LLO dump |
| **Collective straggler stalls** | a kernel fused with a collective is paced by the slowest participant, and the stall is attributed to the kernel | not visible from a single-chip run; needs a multi-chip profile |

**Register pressure surprises people because it has no ceiling you can compute.** VMEM has a number
and check C is arithmetic against it; the register file does not, so a body that fits VMEM can still
spill every bundle, and the only symptom is a kernel slower than its arithmetic says.

**VPU/MXU balance is the fourth thing a boundary silently decides** (priced in
[`compute_units.md`](compute_units.md); probes `vpu_fma_rate`, `mxu_peak_by_dtype`). Pulling ops into
one kernel fixes how much work sits on each unit; the kernel is paced by the saturated one. This cuts
both ways: **it is a reason to fuse *more*** as often as less, because more ops in one kernel gives
you something to move onto the idle unit.

---

## Residuals force the boundary

Training kernels only. "Residuals" here means the forward values saved for the backward pass (JAX
`custom_vjp` vocabulary: the forward returns `(out, res)`), **not** the transformer residual stream.

Why it is a fusion question: a saved residual is an extra array the kernel must write to HBM instead
of discarding in VMEM. It costs a round trip (check A) and an operand slot (check C), and, the part
that forces the boundary, the backward may need a value from *partway through* the forward, which
the forward therefore cannot fuse past.

Four patterns, in increasing order of what they give up:

1. **Return the pre-activation value and de-fuse the activation when residuals are requested.** The
   VJP differentiates the activation at the pre-activation value, so the kernel skips the activation
   when asked for residuals and XLA applies it afterwards.
2. **Make the fusion a named config knob** when the residual decision drives it. For example, an
   attention kernel can emit the softmax denominator instead of the log-sum-exp so the final divide
   leaves the kernel; the knob may only *unfuse* when residuals are saved, never force fusion off when
   they are not. This is a knob about where a value lives, not about the cost of the divide.
3. **Refuse the residual and recompute.** Save only a small statistic (for example the log-sum-exp)
   and let the backward recompute the large intermediate rather than materializing it.
4. **Write the recompute policy down as policy**: the forward explicitly drops every intermediate the
   backward will regenerate or never consume.

**Decide what the backward needs before deciding what the forward fuses.** An inference-only kernel
is free of this entirely and returns no residuals.

---

## What mature kernels deliberately leave outside

Read these as options someone already ranked, not as prohibitions:

- **A scale multiply and a layout transpose before attention**, left in XLA as a concession to the
  kernel's expected layout. That is check E unresolved, paid outside.
- **A cross-step reduction in a backward** (for example summing per-step partial `dq` after the call)
  to avoid read/write conflicts from in-kernel accumulation across steps.
- **Output zeroing for rows the kernel never writes**, as a `jnp.where` after the call. The fused
  alternative issues low-priority zero-fill DMAs in-kernel and waits on them at the end; both are
  legitimate.
- **The final batch reduction** (`sum` / `mean` of a per-row loss), even in an otherwise heavily
  fused kernel.
- **The whole kernel, below a size or hardware threshold.** A fallback to XLA when the hardware unit
  is absent, or when the working set is small relative to VMEM, is part of the design, and the
  threshold is a number you have to pick and measure.

---

## Reading it back from the profile

Checks C and F cannot be settled by reasoning. Read the profile summary / LLO dump on the incumbent
before designing, and on each variant after building (see
[`../tpu_profiling_and_diagnostics.md`](../tpu_profiling_and_diagnostics.md)). Four readings matter:

| Reading | What it decides | Which way it points |
| --- | --- | --- |
| **VMEM used vs limit** | check C, against the real body rather than an operand-count estimate | **headroom is room to fuse more**: the one signal that points that way |
| **spills + fills per bundle** | check F, register pressure | high: the body is too wide *at this block*; a smaller block or narrower dtype may fix it without splitting |
| **bundle occupancy / NOP density** | check F, VLIW scheduling | sparse: try the anti-fusion move below before splitting the kernel |
| **per-unit occupancy** | unit balance | one unit idle: a reason to fuse *more* |

There are no thresholds for these; compare variants against each other. Dump-side readings are
static compile-time analysis; only a sampled hardware capture is measured. Label which one you read.

### The anti-fusion move, for sparse bundles

Splitting the kernel is not the first response to a tangled dependency graph. **A conditional is an
optimization barrier**: the compiler will not move work across a branch boundary, so one branch whose
body handles several situations generically serializes everything inside it. Enumerate the
situations (first k-step, last k-step, both, neither, partial tile) as separate branches with
straight-line, statically specialized bodies, via `pl.when` cases or a `lax.cond` nest. More code and
more branches, in the service of denser bundles.

---

## Worked shape of the fusion options

For a proposal to pull a producer and an epilogue into an existing matmul kernel, with several
operands and a large intermediate:

| Option | What it does | Price | Gate / next action |
| --- | --- | --- | --- |
| **F0** | keep the boundary | round trip: `2 × intermediate_bytes / BW_measured` per call | the incumbent; everything is measured against it |
| **F1** | fuse both, block unchanged | live set must fit; operand-count estimate is not this body | **read VMEM used from the LLO dump first**; then epilogue tax vs F0's saving |
| **F2** | fuse both, smaller block | MXU row efficiency at the smaller block | `mxu_block_rows_cost`, or three block sizes on a skeleton, before building |
| **F3** | fuse the epilogue only | keeps the producer's round trip; buys the cheap half | run the unfused `plain` arm at this block first, either way |
| **F4** | fuse both, narrow the intermediate to bf16 | halves its live footprint; numerics | accuracy gate with a threshold, per [`algorithm.md`](algorithm.md) |
| **F5** | fuse both, relayout the producer's output upstream | `transpose_cost` in the real dtype, if check E found a mismatch | routes in [`cross_area_moves.md`](cross_area_moves.md) |

No rejections; every row carries a price or the probe that produces one. F1's "measure VMEM first" is
the point of the format: the operand-count estimate generated the candidate and cannot close it.

---

## Deciding

Nothing on this list terminates the exercise; every item produces an option.

0. **Start from the values, not the op list.** Checks A to F all begin from an intermediate that
   already exists, so on their own they can only merge seams the current code drew. List every value
   that crosses HBM and ask whether the boundary could fall somewhere the current code has no seam: a
   class of input, a phase, or a group of work sharing data. See
   [`../kernel_design_method.md`](../kernel_design_method.md).
1. **Check A.** `2 × intermediate_bytes / BW_measured` (`hbm_bandwidth_arms`), a first-order
   estimate. Small against the op's runtime? That mechanism buys nothing; go on to B, C and F.
2. **Check B.** Already resident for another reason? Then the fusion costs only the VPU work on the
   tile (`transcendental_relative_cost`).
3. **Training only:** what does the backward need saved? Residuals can invert the answer.
4. **Check C.** Count operands, estimate the block-row collapse (`operand_count_vmem_pressure`), size
   the live set against the VMEM ceiling (`vmem_scratch_bisect`). A matmul block pushed below the
   native row count is an option with a price (`mxu_block_rows_cost`). **Run the unfused Pallas arm at
   your chosen block before the fused one.**
5. **Check D.** Same contraction axis? If not, size the other accumulator (`K × N × itemsize`) first
   (`gmm_tgmm_grid_axis_conflict`).
6. **Check E.** Different layouts? Price the relayout (`transpose_cost`, in the real dtype) against
   the saved bytes.
7. **Check F.** Read spills, bundle density and per-unit occupancy from the LLO dump. An option
   blocked by one ships as `unmodelled` with the measurement that would settle it, never as a
   rejection.
8. **Separately:** does any operand need a schedule the `index_map` cannot express (an address
   computed in-kernel, a non-grid traversal, an HBM-resident accumulator)? That is the only case for
   taking the schedule over. Do not hand-roll to issue a transfer earlier under a managed TensorCore
   grid.

---

## Open questions

| Question | Probe |
| --- | --- |
| Is there a shape where a fused Pallas kernel beats XLA outright, rather than matching it? | `matmul_epilogue_fusion`, `elementwise_chain_xla_vs_pallas` at the target shape |
| Does the epilogue tax hold for a heavier epilogue, a smaller matmul, or a dtype-changing output? | `matmul_epilogue_fusion` with the real epilogue |
| Does XLA still match Pallas on a chain containing a reduction, transpose, gather or multi-consumer value? | extend `elementwise_chain_xla_vs_pallas` |
| What does relayout cost in bf16? | `transpose_cost` |
| What does MXU row efficiency cost between the large block sizes check C actually moves through? | `mxu_block_rows_cost`, `matmul_block_size_sweep` |
| Does the schedule argument hold when the write wait is deferred into the next step and there is a matmul per step? | `fusion_enables_dma_compute_overlap` (current version does not defer the wait) |
| Does Mosaic hoist a hand-issued `.start()` at other grid and tile sizes? | `fusion_enables_dma_compute_overlap` swept over shapes |
| What are the thresholds for spills, bundle density and collective stalls? | none; LLO dump comparisons only, multi-chip for collectives |
