---
name: tpu-hardware-facts
description: >-
  Structural and legality rules for Pallas TPU kernels: what get_tpu_info() describes, block and
  tiling legality, what may be traced, VMEM limits, precision defaults, relayouts, profiler counters
  that mislead, and SparseCore architecture. Read whole before designing a kernel; every number is
  measured on the target with the named probe.
---

# TPU hardware facts and compiler limits

## How to use this file

- **Everything here is a structural or legality rule**: what a field means, what the compiler
  accepts or rejects, what may be a runtime value, which counter misleads. Most of these rules
  break a design without warning, so read the file whole before designing.
- **The file holds no measured numbers.** Any capacity, bandwidth, rate, latency or threshold
  comes from `pltpu.get_tpu_info()` / `plsc.get_sparse_core_info()` (`SparseCoreInfo`) read at run
  time, or from the named probe run on the target:
  `tools/run_probe.py --pick <probe_name> --run_dir <run_dir>`.
  [probe_catalog.md](probe_catalog.md) lists every probe.
- **Rules marked *toolchain-dependent*** must be re-verified by their probe on the current image
  (`software_versions` records the toolchain).
- `hypothesis:` marks a past direction without a number. Verify it before relying on it.

## 1. Device denominator and scaling

**`pltpu.get_tpu_info()` describes one TensorCore, not one chip.** On a chip whose two physical
TensorCores are addressed as separate devices, it reports `num_cores=1` and `is_split_chip=True`.
A Pallas kernel runs on one TensorCore, so for capacity and FLOPs the reported fields are already
the right denominator; use them as reported. Verify with `tpu_info_scaling_denominator`. Dump all
fields with `tpu_info_reported_constants`.

**The constructor divides only some fields.** `ChipVersion.num_physical_tensor_cores_per_chip` is 2
on some generations and 1 on others (source: `jax/_src/pallas/mosaic/tpu_info.py`). Where it is
2:

| field | treatment |
| --- | --- |
| `hbm_capacity_bytes`, `mem_bw_bytes_per_second`, `bf16_ops_per_second`, `fp8_ops_per_second` | chip constant ÷ TensorCores per chip |
| `vmem_capacity_bytes`, `smem_capacity_bytes` | **not divided**: declared per core |
| `cmem_capacity_bytes`, `int8_ops_per_second`, `int4_ops_per_second` | may be a literal 0 (absent / "Not Available", see §3) |
| `sparse_core.vmem_capacity_bytes` | **not divided**: per *vector subcore* (§10) |

The divided set differs by generation. Two opposite mistakes: comparing a divided field with a
chip datasheet (off by the core count), and multiplying an undivided field back up (there is no
"chip VMEM"; the budget compiles until it doesn't fit).

**Structural constants on the same object** (undivided, declared in source; read, do not
hard-code): `num_lanes` / `num_sublanes` (§4, §5), `num_mxus` (§9), `mxu_column_size`,
`num_accumulators`, `sparse_core.num_cores` × `num_subcores` (the wave, §10),
`sparse_core.num_lanes`, `sparse_core.dma_granule_size_bytes` (the gather granule, §10). Only the
cost models built on them need measuring.

## 2. Memory system and bandwidth ceilings

**A split chip has no chip-wide memory pool.** Each TensorCore device (a chiplet) has its own HBM
stack and its own SparseCores. That is why §1's fields are divided. A kernel's memory denominator
is its chiplet's, and it has no path to the other chiplet's HBM.

**Keep the bandwidth ceilings apart, and say which one you used:**

| ceiling | how to get it |
| --- | --- |
| theoretical peak, as reported | `get_tpu_info().mem_bw_bytes_per_second`: a rounded constant, not an achieved rate |
| practical peak (after protocol and on-chip network overhead) | plateau of `hbm_bandwidth_arms`, `block_size_bandwidth_curve` |
| SparseCore streaming *achieved* at your access width, one core and all cores | `sc_vs_tc_streaming_bandwidth`, `sc_subcore_scaling`, `sc_gather_row_width_scaling`: an achieved rate, not a ceiling |
| TensorCore and SparseCore reading concurrently | not established (§11) |

- **Build the roofline on the achieved plateau, not the reported constant.** The reported
  constant is a rounded theoretical peak. hypothesis: no transfer reaches it, and a modelled
  bandwidth between the measured plateau and the reported peak is already over the practical
  ceiling; verify with `hbm_bandwidth_arms` on your chip. The roofline method is in
  [tpu_memory_overlapping.md](tpu_memory_overlapping.md) §1.
- **SparseCore: keep the unit's ceiling and the achieved rate apart.** Unverified (vendor figures,
  no probe): a SparseCore's hardware ceiling is a per-unit command limit rather than a data rate,
  and sits far above typical achieved streaming. What the probes measure is the *achieved* rate,
  which depends on access width: rows spanning many whole DMA granules achieve more than narrow
  reads ([STRAT-MLA-06](strategies/STRAT-MLA-06-sparsecore-bandwidth-is-width-dependent.md)). Never
  treat an achieved figure measured at one width as a ceiling for another.
- **Read-modify-write**: unverified (vendor figure): a concurrent read+write stream may sustain
  less than twice the one-way rate. Price scatter-add against a measured read+write rate, not by
  doubling a one-way figure.
- **Mixing TC and SC traffic on one stack**: unverified (vendor guidance, direction only): it may
  cost more than either alone, because the units use different access sizes. Price an overlap with
  a probe (`sc_tc_independent_overlap`, `sc_tc_overlap_vs_neighbour_kind`), not by subtraction.
- **Check units before reconciling capacities** (vendor GiB vs decimal `hbm_capacity_bytes`);
  budget against what the runtime reports (`device_memory_stats`).

## 3. MXU native dtypes and precision

**The MXU multiplies natively only a fixed set of dtype pairs.** On the parts studied, these were
`(f32|bf16) × (f32|bf16)` and the pairs with an fp8 right operand. **There were zero native integer
pairs**, which is why `int8_ops_per_second = 0` is not an unfilled field. Source:
`tpu_info.py:is_matmul_supported`; it differs by generation. Verify with
`native_matmul_dtype_matrix` (MXU) and `dot_dtype_matrix` (in-kernel `jnp.dot`).
- A non-native dtype gets promoted. An int8 `jnp.dot` runs as bf16, and f16 looks emulated. On
  such a part, int8 quantization saves bandwidth, never compute; only a native narrow type (fp8)
  buys MXU throughput. Measure per-dtype rate with `mxu_peak_by_dtype`.

**f32 × f32 is "native", but the MXU does not multiply in f32.** It runs bf16 passes, and
`jax.lax.Precision` chooses how many.
- **`Precision.DEFAULT` on f32 operands is one bf16 pass.** Its error is *identical* to bf16
  operands with an f32 accumulator. If you name no precision, f32 storage buys bf16 accuracy at
  f32 bandwidth. Verify with `f32_default_is_one_bf16_pass`.
- **Inside Pallas there are two rungs, not three** (toolchain-dependent). `dot_general` with
  `Precision.HIGH` raises `NotImplementedError: Unsupported dot precision: HIGH` and does not fall
  back. Do not copy an XLA three-level precision ladder into a kernel.
- **`HIGHEST` costs a multiple of `DEFAULT` that differs between Pallas and XLA**; measure it at
  your layer.

**An f32 XLA reference is not automatically the accurate side.** An f32 `jnp.einsum` at default
precision multiplies in bf16 on the MXU. A Pallas kernel that contracts on the VPU
(`jnp.sum(a * b)`) stays in true f32 and is *more* accurate than its reference, yet the mismatch
is reported against the kernel. Loosening the tolerance then ships a false claim about the wrong
side. The tell: the kernel fails the XLA reference but matches an f64 oracle to many digits, which
a broken kernel does not do.
- Pin `Precision.HIGHEST` in any reference that contracts in f32, but do not assume XLA honours it
  (unverified; see §11): also check the reference against an f64 (numpy) oracle, so the wrong side
  is proven, not argued (`xla_reference_vs_f64_oracle`).
- **CPU interpret mode has no MXU and cannot catch this**: it passes locally and fails on
  hardware. Accumulator trade-offs: `accumulator_precision`,
  [design/layout_precision.md](design/layout_precision.md).

## 4. Sublane tiling and block legality

**Sublane tiling is a closed form in jax source, not a measurement of the chip.** For generation 7
(the case that was verified; on other generations confirm with `sublane_tiling_is_a_formula`),
`jax/_src/pallas/mosaic/tpu_info.py:get_sublane_tiling` returns
`num_sublanes * (32 // bitwidth)`. With `num_sublanes = 8` that gives f32 → 8, bf16/f16 → 16,
int8/fp8 → 32, int4 → 64.
- Bitwidth is `dtypes.itemsize_bits`; recomputing `itemsize * 8` gets sub-byte dtypes wrong.
- The function is pure Python and makes no device call. Its docstring calls it a heuristic that
  depends on XLA flags. It shows what the compiler is told, not what the silicon does
  (toolchain-dependent). Verify with `sublane_tiling_is_a_formula` and `sublane_tiling_by_dtype`.
  Whether XLA's chosen layout agrees with it is open (§11).

**Divisibility rule: the last two block dims must be divisible by 8 and 128 respectively, or equal
the full array extent. The rule is necessary, not sufficient.**
- f32 and bf16 blocks followed the (8, 128) rule. bf16 accepted 8-row blocks despite a tiling of
  16, so the rule is *not* "block rows ≥ tiling".
- **int8 blocks needed at least the int8 tiling (32) in rows.** 8- and 16-row int8 blocks that
  divided the array were rejected with `Failed to set window params`.
- Verify per dtype with `block_sublane_legality_by_dtype` before a sub-32-bit block
  ([design/partitioning.md](design/partitioning.md); sub-tile cost: `native_tile_granularity`).

## 5. Register layout and relayout costs

**A register value has two axes the hardware treats differently**: **lane** (last, `num_lanes`
wide) and **sublane** (second-to-last, tiled per §4). Moving data between them costs something
that a roofline scores at zero (neither FLOPs nor HBM bytes) and a profile folds into the fused op.
**A leading axis** (outside the last two) is a tile index: operations on it are bookkeeping, while
the same source line on a trailing axis goes through the XLU.

Cost classes (f32, value produced and consumed by arithmetic):

| move | cost class | probe |
| --- | --- | --- |
| add / drop / permute a leading axis | free | `expand_axis_leading_vs_sublane_vs_lane` |
| slice on a lane boundary (multiple of 128) | free | `lane_split_alignment` |
| roll by a whole tile, either axis | free | `sliding_window_compact_vs_sublane` |
| roll by 1, either axis | paid per element | `sliding_window_compact_vs_sublane` |
| swap lanes ↔ sublanes (`swapaxes(-1, -2)`) | paid; comparable to adjacent arithmetic | `transpose_cost` |
| interleaved `jnp.repeat` along sublanes | expensive | `expand_blocked_vs_interleaved` |
| interleaved `jnp.repeat` along lanes | did not compile (VMEM blow-up) | `expand_blocked_vs_interleaved` |
| strided read (`x[::2]`, `ref[::2]`) | **compile error**, not a slow path | — |

- **Cost depends on tile alignment, not the axis.** If you get to choose a sliding window's
  stride, a tile multiple beats any tuning around the roll.
- **Strided reads are refused**, with distinct messages for a value, an f32 ref and a bf16 ref
  (`strides to be confined`, `The last dim size is not 128 in original base memref`, `Strided load
  with non 32-bit data`). The second message names an untested gate, so other ref shapes may be
  legal. Do not design on that without a probe (toolchain-dependent).
- **Mosaic defers relayouts, and forcing one early to cut VREGs can lose.** Reshaping into the
  consumer's layout before a run of elementwise ops was correct as a VREG count and still a measured
  regression, because the compiler had been placing the relayout somewhere cheaper. **A static tally
  (VREGs, ops, bytes) is not a cost model** where the compiler chooses when to do the work; it is
  a reason to measure (`elementwise_before_vs_after_expand`).
- **A transpose passed straight to `dot_general` is absorbed** into the dimension numbers for
  free. If no layout removes a mismatch, restructure the algebra
  ([design/algorithm.md](design/algorithm.md)).
- Scope: these classes are established for f32 only; bf16 is open (§11). Conversion cost:
  `dtype_conversion_cost`. Layout design: [design/layout_precision.md](design/layout_precision.md).

## 6. Memory capacities and allocations

**The default scoped VMEM limit is a knob, `vmem_limit_bytes`, distinct from and below physical
VMEM.** A scratch above the default fails with
`RESOURCE_EXHAUSTED: E1001: CompileTimeScopedVmemOom` under `None`, and compiles with
`pltpu.CompilerParams(vmem_limit_bytes=...)`. The knob moves the *scoped* limit, not the silicon.
- Physical: `get_tpu_info().vmem_capacity_bytes`. Largest scratch under the default:
  `vmem_scratch_bisect`. That the knob moves the limit: `vmem_limit_bytes_raises_ceiling`. The
  class name is toolchain-dependent (`compiler_params_spelling`).
- SMEM: `get_tpu_info().smem_capacity_bytes`; usable int32 count: `smem_scratch_bisect`.

**VMEM bytes per element tracked itemsize with no floor**: int8 used one byte, so halving the
dtype doubled what fits (`vmem_bytes_per_element_by_dtype`). Block shrinkage as operands are added:
`operand_count_vmem_pressure`, [design/fusion_boundary.md](design/fusion_boundary.md).

**A capacity probe must read back at a different offset than it wrote.** If it reads the offset
it wrote, the buffer is eliminated, the allocator never runs, and the "capacity" can exceed
physical VMEM. Trust a cross-dtype ratio only if every arm lands below physical capacity.

## 7. Dispatch and call overheads

**Two floors sit under every kernel timing**: the host round trip (batching calls in one `jit`
removes it) and the per-`pallas_call` intercept (on the **device**; batching does not remove it).

- **The per-call intercept is a fixed device cost that grid tuning does not remove.** Read it
  from the *intercept* of a sweep (grid extent at constant live work) and the per-step cost from
  the *slope*, never from one timing. Probes: `dispatch_floor`,
  `pallas_call_intercept_chained_clock`. Recipe:
  [STRAT-MLA-11](strategies/STRAT-MLA-11-localize-fixed-cost-before-building-a-grid-lever.md).
- **Use a chained clock to separate the two floors.** Run `M` invocations in one `jit`, chained by
  a data dependency XLA cannot fold, and divide by `M`. Host dispatch divides away and device cost
  does not. Say which clock you read; the two costs add.
- **The intercept is a range, not a constant.** It varies with the unit (TensorCore vs SparseCore)
  and the instrument (chained clock vs in-program profile). An isolated chained intercept is not
  paid in full per call inside a larger program. Never use one value as a hard gate on a roofline.
  If the floor decides the design, measure it for your unit and instrument.
- **This holds whatever the value:** a `pallas_call` with few grid steps is intercept-dominated, and
  fusing two calls there beats any block-size tuning
  ([design/fusion_boundary.md](design/fusion_boundary.md)).

## 8. Dynamic indexing and pipelining rules

These fail late, as a compile error or a core halt after the kernel is written.

**Allocate statically, transfer dynamically.** A DMA's transferred extent may be traced (a row
count from SMEM into a fixed VMEM scratch); its destination shape may not (`traced_dma_length`).

**A traced DMA source offset may need `pl.multiple_of`** (toolchain-dependent;
`traced_dma_offset_by_dtype`). An HBM DMA at an SMEM-supplied offset was accepted bare for f32;
for bf16 it was **rejected even at offset 0** (`Offsets along tiled dimensions must be aligned to
tiles`) until wrapped in `pl.multiple_of(offset, rows)`. Provability matters, not the value. Use
`pl.multiple_of` by default.

**A traced index into a VMEM ref was rejected on both axes.** `ref[pl.ds(i, 8), :]` and
`ref[:, pl.ds(i, 128)]` both raise `E2003: CompileTimeMosaicUnprovenMemoryAccessAlignment`. The
mechanism decides, not the axis. **Express data-dependent addressing as a DMA (`make_async_copy`
out of HBM), not as an index** (`dynamic_index_lane_versus_sublane`). Whether `pl.multiple_of`
rescues the VMEM case is open (§11).

**Buffer depth is not a free parameter** (toolchain-dependent; `buffer_count_cap_by_position`).
`pl.Buffered(buffer_count=n)` on a top-level grid `BlockSpec` accepted 1 and 2. For 3 and 4 it
raised `Only single (1) and double (2) buffering are supported`; "deepen the pipeline" needs
another lever ([design/pipelining.md](design/pipelining.md)).

**Every started DMA must be waited before the kernel exits.**
- A truly zero-size copy started and never waited completed correctly, so a ragged `end - start`
  copy needs no `pl.when` guard (`zero_size_dma_needs_no_wait`).
- A *real* DMA left unwaited **halts the core**:
  `E0200: RuntimeUnexpectedCoreHalt ... Semaphore ... has a nonzero value upon exit`. Every later
  launch in the same process then fails with errors that look like infrastructure noise.
  `undrained_semaphore_halts_core` is destructive and runs alone. **If `E0200` follows passing
  probes, suspect an undrained semaphore in the previous probe, not the hardware.**

Persistent scratch and scalar-prefetch index maps: `scratch_persists_across_grid_steps`,
`scalar_prefetch_drives_index_map`, [design/precomputation.md](design/precomputation.md).

## 9. Instruments that lie

These are the xprof/LLO hardware counter planes (`..._TC_...` on `/device:TPU:N`). The facts below
hold for the capture path and toolchain they were found on; re-check on a new one (§11).
Cross-linked from [tpu_profiling_and_diagnostics.md](tpu_profiling_and_diagnostics.md).

**`COUNT_CYCLES` is a free-running counter over an enable window, not a measure of execution.**
The window is decoupled from the kernel, module and capture. It can run far longer than the
program, and its implied clock rate varies wildly between captures. **Never put it in a
denominator**: `X / COUNT_CYCLES` divides a real quantity by a meaningless one.

**`COUNT_MXU_BUSY_<k>` counts MXU *idle*, not busy.** It equals
`COUNT_CYCLES − COUNT_MATPUSH_CYCLES_MXU_<k>` exactly, and the `MXU_BUSY_*` counters sum exactly to
`COUNT_CYCLES`. They partition the enable window and inherit its non-determinism. `XLU_BUSY_*` reads
the same way. With `num_mxus` MXUs receiving identical push counts, one MXU's counter is only its
share of the traffic, not a per-unit breakdown.

**Pinning a different `jax`/`jaxlib` wheel changes neither compiler**: Mosaic TPU and XLA:TPU
both live in `libtpu.so`. **An A/B across jax versions that holds libtpu fixed is null by
construction**, and its near-zero delta does not clear the compiler.

**Run a determinism control before reading any counter.** Run the same variant and shape in two
jobs, or two replicate captures in one, and discard every counter that moves.
- `COUNT_CYCLES` and `COUNT_MXU_BUSY_*` moved between clean replicates, and moved much more when
  the host stalled.
- These stayed bit-identical: `COUNT_MATPUSH_MATRICES_MXU_*`, `COUNT_MATPUSH_CYCLES_MXU_*`,
  `COUNT_MATMUL_VREG_F32_MXU_*`, `COUNT_VECTOR_ISSUE`, `COUNT_SCALAR_ISSUE`,
  `COUNT_BUNDLES_INTO_VIF`, `COUNT_VST_INSTRUCTION`, `COUNT_VLD_INSTRUCTION_*`,
  `COUNT_ROTATE_PERMUTE_INSTRUCTION_XLU_*`. **They are sound as raw counts and as same-job deltas
  between arms.** They are not sound as fractions. They are whole-chip aggregates that include
  non-kernel ops, so a per-time rate built from one is not an occupancy.

**A percentage looks normalised by something real.** Sanity-check the implied physical constant
(cycles over wall time is a clock speed; that check exposed `COUNT_CYCLES`). Prefer a raw count
with its denominator stated inline: a "percent of cycles" once turned out to be
`COUNT_VECTOR_ISSUE / COUNT_SCALAR_ISSUE`. **A resource at exactly 100.0% is a quantity divided by
itself, never a finding.**

## 10. SparseCore architecture

Read parameters from `SparseCoreInfo` at run time or measure them with the named probe. Design:
[design/sparsecore.md](design/sparsecore.md). Reachability: `sparsecore_surface`,
`sc_module_surface`; no CPU dry-run (`sc_kernel_cannot_dry_run`).

**Meshes and memory spaces** (source: `jax/_src/pallas/mosaic/sc_core.py`):

| mesh | shape | `supported_memory_spaces` |
| --- | --- | --- |
| `plsc.ScalarSubcoreMesh(axis_name, num_cores)` | one sequencer per SparseCore | `VMEM_SHARED`, `SMEM`, `SEMAPHORE` |
| `plsc.VectorSubcoreMesh(core_axis_name, subcore_axis_name, num_cores, num_subcores)` | cores × vector subcores | `VMEM`, `VMEM_SHARED`, `SMEM`, `SEMAPHORE` |

- **A scalar-subcore kernel cannot allocate private `pltpu.VMEM`** (a scratch-type error;
  `sc_scalar_kernel_runs`, `sc_vector_kernel_runs`).
- **No mesh touches an HBM ref directly** (`Get does not support loading from HBM`): DMA into
  core-local VMEM/SMEM, compute, DMA back. **`VMEM_SHARED` is a DMA target, not an operand.**
  **Scalar stores are SMEM-only**, even `obuf[0] = x` into a VMEM output buffer.
- **A two-mesh (MPMD) kernel needs scoped scratch**, e.g. `(pltpu.SMEM @ mesh)(...)`, and both
  subcore meshes must share the core axis name and core count. Mesh combinations are
  toolchain-dependent: `sc_two_sparsecore_meshes_one_kernel`, `sc_tc_one_kernel_two_meshes`,
  `sc_mpmd_vmem_scratch`, `sc_mpmd_dma_core_id`.
- The native vector shape is `(sparse_core.num_lanes,)`. Which shapes, dtypes and primitives lower
  is toolchain-dependent: `sc_dtype_support`, `sc_wide_get_layout_passes`, `sc_primitive_lowering`,
  `sc_pl_ops_inside_kernel`, `sc_bf16_widen_and_pack`.

**Capacities are per subcore, pooled per core, and smaller than declared.**
- `sparse_core.vmem_capacity_bytes` is **per vector subcore and does not pool into one subcore's
  working set**. A working set sized against `capacity × subcores` will not fit on one subcore.
- The allocator's ledger is a **pooled per-core budget** (the refusal message names it). What one
  subcore gets depends on what the others on that core already took. A scratch that compiled alone
  can be refused once another is added, with `E3000: CompileTimeSparseCoreAllocationFailure`. Even
  with the core otherwise empty, the largest single scratch is below the declared capacity
  (`sc_vmem_per_subcore`).
- **SparseCore SMEM is far smaller than TensorCore SMEM and may accept only 32-bit types**
  (`sc_smem_capacity`). Metadata that fits TensorCore SMEM may not fit. Accumulators built on
  `fetch_and_add` live in SMEM, so this budget binds them.

**VMEM is banked, and conflicts serialize linearly in ways.** When a lane-private table's per-lane
stride shares a factor with the bank count, lanes fold onto fewer banks (onto a single bank when the
stride is a multiple of the bank count), and cost grows linearly
with the number of colliding ways. `SparseCoreInfo` does not declare the bank count; it was
inferred from a conflict ladder. To check a kernel, time it with its lane-private stride and with
stride + 1 at fixed work. Do not confuse the bank count with the subcore count. Rule:
[STRAT-SC-06](strategies/STRAT-SC-06-pad-scatter-stride-off-bank-count.md).

**A wave is all vector subcores in lockstep, and it costs as much as its longest row.** Wave width
`W = sparse_core.num_cores × sparse_core.num_subcores` per chiplet. If row `w` maps to
`w * W + flat`, a per-row variable extent is governed by `E[max over W]`, not the mean; for
`rl ~ U(0, n)` that is `W/(W+1) · n`. The count comes from source; the cost model needs measuring
(`sc_subcore_scaling`). Rule: [STRAT-SC-05](strategies/STRAT-SC-05-wave-longest-row.md).

**Achieved SC streaming bandwidth is an all-subcore aggregate per chiplet, not a ceiling.**
Dividing it by the subcore count justifies no transfer lever, and it compares only with the
all-cores ceiling. The gap comes from the DMA granule and access width, not a silicon limit
(`sc_vs_tc_streaming_bandwidth`). Indirect rows:
[STRAT-MLA-06](strategies/STRAT-MLA-06-sparsecore-bandwidth-is-width-dependent.md),
`sc_gather_row_width_scaling`.

**"Good at data-dependent addressing" splits: reads tend to win, read-modify-write tends to lose.**
- hypothesis: SC gather beats XLA's at narrow rows; the advantage shrinks with row width until a
  wide row is an XLA win, saturating once a row fills `dma_granule_size_bytes`. Price every gather
  against its row width. Verify: `sc_gather_vs_xla_gather`, `sc_gather_row_width_scaling`,
  `sc_gather_width_curve_fixed_chunk`, `sc_gather_index_count_scaling`, `sc_xla_gather_cliff_bracket`.
- hypothesis: SC scatter-add loses to XLA's; fanning across all subcores narrows but does not close
  the gap. Verify: `sc_scatter_add_vs_xla`, `sc_scatter_add_fanout`, `sc_addupdate_variants`.

More SparseCore rules: data-dependent control flow (`sc_data_dependent_control_flow`,
[STRAT-MLA-08](strategies/STRAT-MLA-08-sparsecore-data-dependent-skip-lowers.md)); rank-0 SMEM
access ([STRAT-MLA-09](strategies/STRAT-MLA-09-sparsecore-rank0-smem-xor.md)); traced-extent DMA
([STRAT-MLA-12](strategies/STRAT-MLA-12-sparsecore-traced-dma-needs-a-1d-ring.md)); loop carries
(`sc_pl_loop_init_carry`, [STRAT-SC-03](strategies/STRAT-SC-03-barrier-the-loop-carry.md));
`subcore_barrier`'s subcore-count ceiling (`sc_subcore_barrier_minimal`,
`sc_multi_subcore_reduction`); placing work across TensorCore and SparseCore
([design/cross_area_moves.md](design/cross_area_moves.md); `sc_tc_independent_overlap`,
`sc_tc_dependent_handoff`, `sc_vs_tc_transcendental`).

## 11. Not established

Open questions, each with the probe that settles it. Do not fill a gap with an assumed value.

- **Practical HBM peak vs reported constant**: `hbm_bandwidth_arms`, `block_size_bandwidth_curve`.
- **SparseCore ceilings** (one-way, read+write, one core vs all): `sc_vs_tc_streaming_bandwidth`,
  `sc_subcore_scaling`.
- **TC/SC concurrent-access penalty** (direction only): `sc_tc_independent_overlap`,
  `sc_tc_overlap_vs_neighbour_kind`.
- **Per-call intercept as one quantity** (never varied unit alone or clock alone):
  `pallas_call_intercept_chained_clock` and `dispatch_floor` on a TC and an SC kernel in one job,
  then both timed in-program. Until then, never reject a design for landing under an intercept.
- **`HIGHEST` honoured at XLA level?** (seen not to be, unrecorded): `xla_reference_vs_f64_oracle`.
- **bf16 passes per `Precision` level**: `mxu_peak_by_dtype` plus `accumulator_precision` per level.
- **XLA's chosen layout vs `get_sublane_tiling`** (seen to disagree, unrecorded):
  `sublane_tiling_by_dtype`.
- **Per-operand VMEM budgets**: `operand_count_vmem_pressure`, `vmem_scratch_bisect`.
- **bf16 relayout costs**, transpose especially (the bf16 control misbehaved): `transpose_cost` and
  the `layout_arrange` probes at bf16, after validating the control.
- **Strided reads at other ref shapes** ("last dim size is not 128" gate): no probe yet.
- **int4 promotion and block minimum** (`8 * (32 // 4)` rows?): extend
  `native_matmul_dtype_matrix`, `block_sublane_legality_by_dtype`, `bitcast_and_subbyte` to int4.
- **float8 block legality** (the tiling-rows rule rests on int8 alone):
  `block_sublane_legality_by_dtype` at float8.
- **`buffer_count` under `pltpu.emit_pipeline`**: `buffer_count_cap_by_position`.
- **`pl.multiple_of` rescuing a traced VMEM index (`E2003`)**: `dynamic_index_lane_versus_sublane`
  with an annotated arm.
- **Interpret mode vs traced-index VMEM scratch** (e.g. `lax.rem(p, 2)` double buffers): judged
  unchecked, not demonstrated. Settle with a deliberately wrong index under interpret and hardware.
- **Counters on another capture path or chip**: repeat the §9 determinism control there.
- **SparseCore bank count and cost per way**: not declared by `SparseCoreInfo`; check per kernel by timing the stride and stride + 1.
- **SparseCore usable VMEM / SMEM vs declared**: `sc_vmem_per_subcore`, `sc_smem_capacity`.
- **Multi-device**: everything here is one TensorCore device; no registry probe covers two chiplets.
