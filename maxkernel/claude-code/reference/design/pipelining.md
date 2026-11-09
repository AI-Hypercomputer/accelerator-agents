---
name: pipelining
description: >-
  Design rules for keeping the HBM-to-VMEM pipeline full in a Pallas TPU kernel: which bandwidth to
  budget against, how large a block must be, the per-grid-step cost model, the memory/compute
  crossover, and when to move from the BlockSpec grid to emit_pipeline, a BufferedRef subclass or
  raw DMA. Read it when a kernel is memory-bound or its access pattern is not a plain BlockSpec.
---

# Keeping the pipeline full

This file covers decisions. The roofline arithmetic, the "is it memory-bound?" diagnosis, fetch
skipping, the `emit_pipeline` callbacks and pipeline bubbles are in
[`../tpu_memory_overlapping.md`](../tpu_memory_overlapping.md). Read that file first; this one does
not repeat it.

Every quantity below that depends on the chip or the toolchain is a probe, not a number. Run it with

```bash
{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/run_probe.py --pick <probe_name> --run_dir <run_dir>
```

## Questions and probes

| question | probe(s) (`pipelining`) |
| --- | --- |
| What HBM bandwidth is achievable, and does a Pallas grid copy reach XLA's? | `hbm_bandwidth_arms` |
| How large must a block be before a copy reaches full bandwidth? | `block_size_bandwidth_curve` |
| At how many FLOPs per byte does a kernel stop being bandwidth-bound? | `arithmetic_intensity_crossover` |
| What does the pipeline cost to start, and how many steps until steady state? | `pipeline_depth_and_warmup` |
| Does an inner `emit_pipeline` beat the BlockSpec grid at the same partition? | `emit_pipeline_vs_blockspec_grid` |
| What does the managed double-buffered pipeline buy over unoverlapped hand-rolled DMA? | `manual_dma_vs_managed` |

Probes from other modules that this file leans on: `dispatch_floor` and
`pallas_call_intercept_chained_clock` (partitioning) for the fixed per-call and per-step costs;
`buffer_count_cap_by_position` (hardware) for buffer depth; `tpu_info_reported_constants`
(environment) for the reported bandwidth and core count; `f32_default_is_one_bf16_pass`,
`xla_reference_vs_f64_oracle` (hardware) and `mxu_peak_by_dtype` (compute_units) for precision.

---

## Name which bandwidth

"HBM bandwidth" is at least four different numbers, and a roofline that does not say which one it
divided by cannot be checked:

| figure | where it comes from | use it for |
| --- | --- | --- |
| reported `mem_bw_bytes_per_second` | `pltpu.get_tpu_info()` (spec sheet) | nothing quantitative; it is optimistic against what a kernel achieves |
| XLA copy, read+write counted | `hbm_bandwidth_arms` | the achievable ceiling for a memory-bound kernel |
| Pallas grid copy, large blocks, read+write counted | `hbm_bandwidth_arms` | what a BlockSpec-managed kernel can reach |
| XLA read-only reduction, read counted | `hbm_bandwidth_arms` | checking whether reads are cheaper than writes |
| hand-rolled DMA, serial and managed | `manual_dma_vs_managed` | the bar a hand-written schedule must clear |

Rules that follow from the shape of these measurements (re-check the direction on the target with
`hbm_bandwidth_arms`):

- **Budget against the measured XLA copy, not the reported constant.** The reported figure is a
  spec; the measured one is what you can spend.
- **A Pallas grid copy with large blocks is expected to match XLA's copy.** Hypothesis: there is no
  bandwidth penalty for writing a memory-bound kernel in Pallas, unlike compute-bound comparisons;
  verify with `hbm_bandwidth_arms`. If your kernel is memory-bound and near that figure, Pallas is
  not what is costing you.
- **Do not model reads as cheaper than writes.** Hypothesis: the read-only arm does not beat the
  read+write arms; verify with `hbm_bandwidth_arms`.
- The roofline formula itself is in [`../tpu_memory_overlapping.md`](../tpu_memory_overlapping.md)
  §1; substitute the measured ceiling for the spec-sheet bandwidth there.

## How large a block must be

Bandwidth rises with block size and then flattens: a small block is dominated by a fixed per-block
cost, a large one by bytes. `block_size_bandwidth_curve` sweeps the block rows of a large f32 copy
and reports the fraction of the ceiling at each size. From it, read two values for the target:

- **the knee** — the bytes per operand block below which you lose more than half the bandwidth;
- **the plateau** — the size past which growth buys almost nothing.

The block-size target is therefore a range, not a maximum. Size every operand block above the knee,
stop at the plateau, and check the double-buffered total against the VMEM budget in
[`partitioning.md`](partitioning.md). Past the VMEM ceiling the call fails at run time rather than
slowing down.

The curve is measured on a pure copy. A kernel with compute in each step may reach the knee at a
different size, because the transfer has more to hide behind (see Gaps).

## What a grid step costs: `per_step + bytes / BW`

The curve ranks block sizes. A cost model prices a step, which is the form to design against:

```
t_step  = per_step + bytes_step / BW_marginal
t_floor = Σ_steps t_step  (+ the per-call fixed cost, below)
```

- `per_step` is the fixed cost of one grid step independent of bytes. Measure with
  `dispatch_floor` (its extra-grid-step arm), or fit it from `block_size_bandwidth_curve`.
- `BW_marginal` is what one more byte costs once the step is paid for. It sits *above* the
  whole-kernel achieved figure from `hbm_bandwidth_arms`, because the achieved figure has the fixed
  term folded in. Fit it from `block_size_bandwidth_curve`: two or more arms, same total bytes,
  differing only in block size.

Consequences:

- **A floor quoted without a step count is not a floor.** `bytes / BW` understates a kernel with
  small steps by an unbounded factor, and it moves by exactly zero when the same bytes are regrouped.
  A model with only a bandwidth term therefore cannot rank the "how many units of work per step?"
  lever at all; the option never enters the comparison, and nothing in the output looks blank.
- **Regrouping the same bytes into fewer, fatter steps is a real lever**, sometimes the largest one
  available.
- **Solve for the step that makes the fixed term small.** Setting
  `per_step = f × (per_step + bytes / BW_marginal)` gives the step size at which the fixed term is a
  fraction `f` of the step: `bytes = per_step × BW_marginal × (1 - f) / f`.
- **The model has a range.** Below some bytes per step a step is not bandwidth-bound: it pays
  something close to a no-DMA control and the `bytes / BW` half is noise. Fitting a bandwidth line
  through such points tilts both constants. Use the model inside the range where steps are
  bandwidth-bound; below it, measure a constant-`index_map` (no-DMA) control.
- **There is no single per-step constant.** The measured slope per step changes with block size, so
  quote the model, not one per-step figure.
- **Hypothesis: operand count is not a missing term.** Holding bytes per step and step count fixed
  while varying the number of buffers did not add a per-operand cost distinguishable from zero.
  Re-check with a variant of `block_size_bandwidth_curve` if a kernel with many operands sits well
  above the model.

### Price transfers, not just steps

The fit above had a fixed number of DMAs per step, so it cannot tell a cost per **step** from a cost
per **DMA**. In a kernel that issues many descriptors per step (a paged gather, one DMA per row),
they separate, and the per-DMA term can dominate:

```
t_step = per_step + n_dma × per_dma + bytes_step / BW_marginal
```

- Halving volume does nothing when descriptors dominate; halving descriptor count is the lever. The
  fix is usually a layout where the transfer is one strided copy instead of many.
- "Not bandwidth-bound" and "not gather-bound" are different claims. A block-size sweep that finds
  no bandwidth sensitivity is consistent with descriptors dominating.
- A fixed cost fitted at one transfers-per-step ratio does not carry to another ratio; carry the
  ratio with the constant.
- **A two-term fit from two arms has zero residual degrees of freedom.** It reproduces both inputs
  exactly by construction. Validate on a third arm or a refit on unseen data, and say which.

### A strided block is many small transfers

A block's transfer size is not its logical size when its rows are strided in HBM. A logically large
block laid out with a stride between rows can be issued as many small descriptors, which puts it at
the expensive end of the step-cost curve. Establish what granularity the step really moves before
applying the model. This is what prices "get the layout from the load"
([`cross_area_moves.md`](cross_area_moves.md)): that move trades a relayout for a worse per-step
transfer, and the two must be costed together. (An in-kernel strided *read* is a separate mechanism;
see [`../tpu_hardware_facts.md`](../tpu_hardware_facts.md).)

## Where the memory/compute crossover is

`arithmetic_intensity_crossover` runs a large f32 elementwise kernel with N dependent FMAs per
element and reports achieved bandwidth and achieved compute per N. Read off the target:

- **the ridge** — the flop/byte at which achieved bandwidth and achieved compute cross;
- **the compute asymptote** — it should recover `vpu_fma_rate`; if it does not, distrust the sweep.

Rules (hypothesis on direction, verify with the probe):

- The VPU ridge is low, on the order of one flop per byte, far below the MXU ridge in the roofline.
  **An elementwise kernel with more than a couple of ops per element is VPU-bound, not
  memory-bound.** The fix is moving work to the MXU or doing less of it
  ([`algorithm.md`](algorithm.md), [`compute_units.md`](compute_units.md)), not tuning transfers.
- The approach to the ridge is slow: even one FMA per element already costs bandwidth. There is no
  regime with full bandwidth *and* useful VPU compute.

## Pipeline warm-up and the per-call cost

`pipeline_depth_and_warmup` sweeps grid length at a fixed block, deliberately unbatched, and fits
`t = fixed + steps × per_step`. Two costs, not one:

- **`fixed`** — host round trip plus pipeline prologue and epilogue, paid once per call. It should
  agree with `dispatch_floor`; if it does not, one of the two is mismeasured.
- **`per_step`** — the step-cost model above evaluated at that block size. It is not a separate
  constant, and different probes report different per-step figures because they use different
  blocks.

**Long grids are cheap; short grids are pure overhead.** A kernel whose whole grid is a few steps
pays the fixed cost and little else. If the problem has only a handful of blocks, it may be too small
for its own kernel; fuse it with a neighbour instead ([`fusion_boundary.md`](fusion_boundary.md)).

Batching calls removes most of the fixed cost, but not all of it. A residue per `pallas_call`
survives when calls are chained inside one `jit` by a data dependency, and it is device time, not
dispatch. Measure it with `pallas_call_intercept_chained_clock`; express it as "equivalent grid
steps" at your block size. Below that many steps, fusing two calls beats any block-size tuning. Full
treatment in [`partitioning.md`](partitioning.md).

## What the managed pipeline buys

`manual_dma_vs_managed` compares the BlockSpec-managed grid with a hand-rolled *serial* DMA loop (no
overlap) over the same partition. The serial arm is a **control**, not a proposal: the ratio is what
default double-buffering buys over no overlap. Two consequences:

- A hand-rolled DMA schedule must beat the managed arm to be worth writing. The most it can win is
  the gap from the managed arm to the ceiling in `hbm_bandwidth_arms`, and it starts from the serial
  arm's deficit.
- At mid-sized blocks the managed arm sits below the ceiling by roughly what
  `block_size_bandwidth_curve` predicts at that size. Fix the block size before the scheduler.

`emit_pipeline_vs_blockspec_grid` compares the managed grid with an inner `pltpu.emit_pipeline` at
an identical, static partition (output checked exact). Hypothesis: they tie. **Read that for exactly
what it measures:** a pure copy at a regular partition, the one case both schedulers can express. It
is not a verdict on the cases below, where the managed grid cannot express the access pattern at all
and there is nothing to tie with. Choose `emit_pipeline` for capability, never for throughput.

### The two-level structure

`emit_pipeline` is not an alternative to `pallas_call`; it lives inside one. The outer call
(`pl.kernel`, or a `pallas_call` with no grid and `memory_space=pl.ANY` specs) hands the body whole
HBM refs. Inside, `emit_pipeline` configures a block pipeline and returns a function; calling it
runs the pipeline:

```python
pipeline = pltpu.emit_pipeline(inner_body, grid=..., in_specs=..., out_specs=...)
pipeline(x_hbm, w_hbm, o_hbm, scratches=(acc_vmem,))
```

What each piece does (source: `jax/_src/pallas/mosaic/pipeline.py:emit_pipeline`,
`_emit_pipeline`):

| argument | meaning |
| --- | --- |
| `grid` | may contain **traced** extents; a `pallas_call` grid may not |
| `in_specs` / `out_specs` | `BlockSpec`s; buffer depth, lookahead and `pl.BoundedSlice` live here |
| `tiling` | an enum (compact vs SparseCore), not a free tile shape; leave `None` on TensorCore |
| `core_axis` / `core_axis_name` | split the pipeline grid across TensorCores; pass one, not both. Inert when `get_tpu_info().num_cores` is 1 (check with `tpu_info_reported_constants`) |
| `dimension_semantics` | only tells the core partitioner which axis is safe to split; unlike on `pallas_call` it says nothing about revisiting or order, and is inert on one core |
| `trace_scopes` | wraps each phase in a named scope (`ep_copy_in`, `ep_wait_in`, `ep_run_kernel`, `ep_copy_out`, `ep_wait_out`, ...) |
| `no_pipelining` | debugging switch: all copies become synchronous. Use it when a kernel is *wrong* and you suspect a buffering race |

The returned function takes `*refs` plus three keywords:

- `scratches=` — forwarded to the body after the block refs. Allocate it on the outer kernel;
  `emit_pipeline` does not allocate it.
- `allocations=` — one `BufferedRef` per ref. `None` (normal) lets the pipeline allocate its own.
  Passing your own, held with `@pl.with_scoped(allocations=...)`, is how a `BufferedRef` subclass
  gets in (capability 5 below).
- `body_prologue=` — runs inside the grid environment after the input waits and before the body;
  the hook for touching a `pl.ANY` ref by hand while `pl.program_id` is available.

Toolchain-dependent: a config flag can route `emit_pipeline` to an alternative primitive
implementation whose returned function lacks `scratches` and `body_prologue`. Everything here
assumes the default path; if the `scratches=` idiom stops tracing, check that flag.

## What only `emit_pipeline` can do

The `pallas_call` grid is pipelined below Pallas, in the Mosaic lowering; `emit_pipeline`
re-implements the pipeline in Pallas. So they have different feature sets, and each item below is a
hard error on the `pallas_call` grid rather than a slow path. All are read from source and
toolchain-dependent.

1. **Input buffer depth above 2.** The managed grid accepts `pl.Buffered(1)` and `pl.Buffered(2)`
   only (verify with `buffer_count_cap_by_position`). `emit_pipeline`'s `BufferedRef` caps depth at
   2 **for outputs only**, so an input — typically a weight operand reused across steps, whose reload
   is long-latency — may be buffered deeper. The pipeline's stage count is the max depth over all
   specs (floored at 2); each ref skips the prologue rounds past its own depth, so a deep spec on one
   input gives that input alone a head start.
2. **Lookahead.** `pl.Buffered(n, use_lookahead=True)` fetches the next *changed* block as soon as a
   slot frees, however far ahead; useful when an `index_map` repeats a block for long runs. The
   managed grid rejects it; `emit_pipeline` rejects it only on outputs.
3. **Traced grid extents and `pl.BoundedSlice` block dims.** The managed grid accepts only
   `Blocked`, `Element` and `Squeezed` block dims; a `BoundedSlice` there is a trace-time error.
   `emit_pipeline` reads a `pl.ds` out of the `index_map` for a `BoundedSlice` dim, which is how a
   ragged final block is expressed without padding.
4. **Per-phase profile scopes.** With `trace_scopes`, the profile summary can split a slow kernel
   into `ep_wait_in` versus `ep_run_kernel` — the difference between "slow" and "DMA-starved". The
   managed grid's copies carry no such names.
5. **A `BufferedRef` subclass: you own the addresses, the pipeline owns the schedule.** Subclass
   `pltpu.BufferedRef` and override `copy_in` / `wait_in` / `copy_out` / `wait_out`. The copies are
   yours — run-time `pl.ds` offsets read from SMEM, several DMAs per step, per-operand conditions —
   while slot rotation, semaphores, prologue and drain stay the pipeline's. The offsets are ordinary
   values, not an `index_map`, so the window prover never sees them; the configured `BlockSpec` can
   be trivial.

Capability 5 is the important one. It removes the false choice between "a BlockSpec grid" and "raw
DMA": **when addresses are data-dependent but the iteration is regular**, which describes most ragged
workloads (packed variable-length sequences, page or slot tables, uneven groups, block-sparse masks,
run-time strides), use a subclass. A carried buffer — state read at the top of a step and written at
the bottom — is one class overriding all four hooks, not an input plus an output allocation. Mark
non-traced fields of the subclass (config objects, metadata refs) static, or tracing will treat
them as leaves ([`precomputation.md`](precomputation.md)).

### What it does not buy

- **Throughput** at a partition both schedulers can express (`emit_pipeline_vs_blockspec_grid`).
- **Deeper output buffering or output lookahead.** Refused in `emit_pipeline` too.
- **Anything from `core_axis_name` or `dimension_semantics` on a single-core chip.** Keep them for
  portability.
- **Simplicity.** For a static regular partition with no setup, the `pallas_call` grid is less code
  at the same speed.

## Overlapping windows: `pl.Element` is a toolchain-dependent pitfall

A stencil, convolution or sliding window needs a few rows of the previous block. `pl.Element` with
an element window looks purpose-built, but:

- **It is refused unless the windowed block extent satisfies the ordinary 8/128 tiling rule.** A
  halo width set by the algorithm is usually not a multiple of 8, and padding it changes the block
  you were sizing. Assume it is unavailable and check before designing around it.
- **Interpret mode may not implement element-window padding,** so a CPU dry run cannot catch the
  mistake; the first check is a TPU run.
- **Prefer carrying the halo in a VMEM scratch across grid steps.** Keep the last rows of each block
  in scratch and prepend them next step. Hypothesis: the carry costs about nothing against a no-halo
  arm, and a second offset `BlockSpec` over the same array costs measurably more because it
  re-fetches rows already in VMEM. No registered probe covers this; build a three-arm A/B (carry /
  second BlockSpec / no halo) at your halo width.

The general rule: **when consecutive steps overlap, move the overlap through scratch rather than
asking the pipeline to fetch it twice.** It holds whether or not `pl.Element` would have been
accepted.

## When to drop to raw DMA

Raw DMA sits below `emit_pipeline`, not beside it. Both clauses must hold:

1. **The iteration itself is irregular** — the number of copies per step varies, the operands a step
   touches are chosen at run time, or the body decides what to fetch after inspecting data. Not
   merely "no `index_map` the prover will accept"; that is capability 5.
2. **A `BufferedRef` subclass cannot express it.** If only the *addresses* are dynamic and the grid
   is still a loop over tiles, capability 5 supplies the schedule and hand-rolling only adds a
   semaphore protocol to get wrong.

Collapsing the clauses is the standard way to hand-write a pipeline `emit_pipeline` would have
supplied. A ragged offset rejected by the prover is a clause-2 case. Do not drop to raw DMA to go
faster on a regular pattern; the managed grid already reaches the copy ceiling.

**There are three routes, not two:** managed grid, `emit_pipeline` (with a subclass if addresses are
dynamic), raw DMA. A probe that only builds `pallas_call` arms grades those arms; it does not survey
the routes. Before writing "no route exists", name which of the three you rejected and why.

### Check that hand-rolled copies actually overlap

Read the emitted schedule in the profile summary / LLO dump: which DMA thread each transfer issues
on, and where issues sit relative to waits. A body that reads `issue ... issue | wait ... wait |
compute ... compute` has serialised; transfer cost is fully exposed. Whether that is the platform or
your code (a `pl.when` around the block, a scratch dependency, emission order) is not decidable from
the dump. Probe the cause before designing a multi-thread fix.

## Hand-rolled DMA idioms

The basic shape: `cp = pltpu.make_async_copy(src, dst, sem); cp.start(); ...; cp.wait()`. Issue the
next transfer before computing on the current one; wait only right before use.

- **Waiting without the descriptor.** The semaphore is the real object; the copy is a handle. To wait
  on a DMA whose descriptor you no longer hold, build a copy with the same ref as source and
  destination and call `.wait()` on it. **The extent must match** the DMA being waited on, because
  the wait is matched by size. This is the only spelling, not a hack.
- **Zero-length copy instead of a branch.** Every `pl.when` is an optimization barrier. Where a copy
  is conditionally needed, issuing one with zero length keeps the instruction stream uniform and lets
  the scheduler overlap work across what would have been branches. Whether a zero-size DMA still
  needs a wait is toolchain-dependent; verify with `zero_size_dma_needs_no_wait`.
- **Every started DMA must be waited.** Verify the consequence of leaving one undrained with
  `undrained_semaphore_halts_core` before relying on any shortcut.
- **Runtime offsets and lengths.** Whether a DMA offset or length may be traced, and per dtype, is
  toolchain-dependent: `traced_dma_offset_by_dtype`, `traced_dma_length`.
- **Scratch goes in `scratch_shapes`.** Prefer it over `pl.run_scoped` for kernel scratch.
  `@pl.with_scoped` is a different thing: it holds `emit_pipeline`'s `allocations` (the pipeline's
  own buffers), not kernel scratch, which still goes through `scratches=`.

## Precision is a pipelining decision

Dtype sets bytes per element, so it sets the bandwidth term directly.

- `DotAlgorithmPreset` is not available in a Pallas TPU kernel. The controls are
  `preferred_element_type` and `jax.lax.Precision`.
- **f32 operands at `Precision.DEFAULT` are one bf16 pass** — the same error as bf16 operands with an
  f32 accumulator. Verify with `f32_default_is_one_bf16_pass`. An f32 operand at DEFAULT pays four
  bytes per element to move two bytes of mantissa.
- **Inside a Pallas kernel the ladder has two rungs, `DEFAULT` and `HIGHEST`.** `Precision.HIGH`
  fails to compile in a Pallas `dot_general` (toolchain-dependent; check by compiling). A three-rung
  accuracy plan is an XLA plan and does not port inward. `HIGHEST` is honoured in-kernel; the cost
  of real f32 is a large multiple of DEFAULT. Measure the XLA-level ratio with `mxu_peak_by_dtype`;
  the in-kernel ratio at your shape needs its own A/B. Details in
  [`layout_precision.md`](layout_precision.md).
- **An f32 XLA reference may itself be bf16-accurate.** Check with `xla_reference_vs_f64_oracle`
  before using it to judge a kernel.
- **Keep HBM narrow, widen per tile in VMEM.** Store bf16, cast slices to f32 in the kernel. A
  full-tensor bf16→f32 cast in HBM doubles memory and DMA traffic. The in-kernel cast is cheap next
  to the bytes saved; measure with `dtype_conversion_cost`.
- `pltpu.CompilerParams` fields such as `shape_invariant_numerics` and
  `fuse_transposed_lhs_in_matmul` exist on some toolchains (check with `compiler_params_spelling`);
  their effect is unmeasured.

---

## Deciding

1. **Compute the stepped floor**, `Σ_steps (per_step + n_dma × per_dma + bytes_step / BW_marginal)`
   plus the per-call fixed cost, with constants from the probes above. `bytes / BW` alone is not a
   floor. If the kernel is near the stepped floor, stop; Pallas will not beat XLA on bandwidth.
2. **Compute flop/byte.** Above the VPU ridge from `arithmetic_intensity_crossover` (for
   elementwise work) you are compute-bound and transfer tuning will not help.
3. **Size blocks above the knee** from `block_size_bandwidth_curve`, within VMEM.
4. **Count grid steps.** A short grid is dominated by the per-call cost
   (`pipeline_depth_and_warmup`, `pallas_call_intercept_chained_clock`); fuse calls rather than tune
   blocks. Do not confuse the per-call cost with the per-step term.
5. **Choose the scheduler on capability; they tie on speed.** `pallas_call` grid for a static
   regular partition. `emit_pipeline` for a traced extent, a `BoundedSlice` dim, input depth above
   2, lookahead, work around the pipeline, per-phase profiling, or addresses the prover rejects (a
   `BufferedRef` subclass).
6. **Separate "addresses are dynamic" from "iteration is irregular".** The first keeps the schedule
   (capability 5); only the second is raw DMA, and only when both clauses hold.
7. **Count three routes** before writing that none exists, and name the one rejected.
8. **Choose dtype for bytes.** Narrow in HBM; f32 at DEFAULT buys no accuracy; `HIGH` is not an
   in-kernel option.

## Gaps (open questions with probes)

- **Does the block-size knee move when the step carries compute?** The curve is a pure copy. Probe: a
  compute-bearing variant of `block_size_bandwidth_curve`.
- **Does `emit_pipeline` win where the managed grid cannot follow** (traced extent, ragged
  addresses)? `emit_pipeline_vs_blockspec_grid` only covers the regular case. Probe to build: a
  `BufferedRef` subclass against a hand-rolled pipeline over the same ragged access pattern.
- **What does input depth above 2 buy?** Reachable under `emit_pipeline` from source, never priced.
  Probe to build: a depth sweep through an `emit_pipeline` `in_spec`; `buffer_count_cap_by_position`
  checks only legality.
- **Does lookahead pay?** Source-confirmed reachable under `emit_pipeline`, unmeasured.
- **Do the `ep_*` scopes appear in a captured profile as described?** Read from the emitter, not a
  trace. Check with the profile summary of any `emit_pipeline` kernel.
- **`no_pipelining`, `body_prologue`, caller-supplied `allocations`, non-`None` `tiling`** are
  source-read only, never exercised.
- **Is a 4-D head-blocked step above the model because it is strided?** A head-blocked kernel was
  observed above the step model with operand count ruled out; the strided-transfer hypothesis is
  untested. Probe to build: same bytes per step, contiguous versus head-strided block.
- **Halo carry at wider halos, multi-tile halos, or on a ragged grid** is unmeasured; no registered
  probe covers overlapping windows.
- **Multi-device bandwidth.** `hbm_bandwidth_arms` measures one device; nothing here covers
  collectives.
- **In-kernel `HIGHEST` cost versus shape** has no registered probe; only the XLA-level rate
  (`mxu_peak_by_dtype`) does.
