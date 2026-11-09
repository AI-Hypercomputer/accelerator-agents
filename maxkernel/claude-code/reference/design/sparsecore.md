---
name: sparsecore
description: >-
  Designing and debugging SparseCore Pallas kernels: what the unit is, the tpu_sc programming
  surface, every compile error it is known to produce and the fix for each, which primitives lower,
  how to pipe SparseCore work with the TensorCore, and when an offload pays. Read before putting any
  part of a design on the SparseCore, and when a SparseCore kernel fails to compile.
---

# SparseCore

## Questions and probes

Run any probe with
`{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/run_probe.py --pick <probe_name> --run_dir <run_dir>`.
Every answer below is toolchain-dependent. Re-run the probe on the current image before relying on
an answer that came from another image. [../probe_catalog.md](../probe_catalog.md) lists every probe.

**Reachability and surface** (module `sparsecore`, except where noted)

| Question | Probe |
| --- | --- |
| Is SparseCore reachable from Pallas here at all? | `sparsecore_surface` (module `compute_units`) |
| Which `tpu_sc` surface does this jax expose, at what module path and with what geometry? | `sc_module_surface` |
| Can a SparseCore kernel be dry-run on CPU? | `sc_kernel_cannot_dry_run` |
| Does a minimal `VectorSubcoreMesh` kernel compile, run and give the right answer? | `sc_vector_kernel_runs` |
| Does a `ScalarSubcoreMesh` kernel run, and which memory spaces does it accept? | `sc_scalar_kernel_runs` |
| Which element dtypes survive a load-compute-store round trip? | `sc_dtype_support` |
| How much VMEM can one vector subcore allocate before the compiler refuses? | `sc_vmem_per_subcore` |
| How much SMEM can a SparseCore kernel allocate, and in which dtypes? | `sc_smem_capacity` |
| Which `plsc` primitives lower and compute the right answer? | `sc_primitive_lowering` |
| Which ordinary `jnp`/`lax` ops survive inside a kernel body? | `sc_pl_ops_inside_kernel` |
| Does `pl.loop` carry a value, or must a running scalar live in SMEM? | `sc_pl_loop_init_carry` |
| How is bf16 consumed and produced on a unit whose arithmetic is f32? Which `pack` order is right? | `sc_bf16_widen_and_pack` |
| Can a kernel run a data-dependent `while` loop? | `sc_data_dependent_control_flow` |
| Is the `(num_lanes,)` shape rule a property of the hardware or of `needs_layout_passes=False`? | `sc_wide_get_layout_passes` (module `sparsecore_gaps`) |
| With layout passes on, does a wide `Get` lower in a copy or elementwise body? | `sc_wide_get_without_reduction` (`sparsecore_gaps`) |
| Is the wide spelling faster than the lane-group loop, or only shorter? | `sc_wide_get_vs_lane_loop` (`sparsecore_gaps`) |
| Do `addupdate` and `addupdate_compressed` lower, and under which flag setting? | `sc_addupdate_variants` (`sparsecore_gaps`) |

**Multiple meshes** (`sparsecore`, `sparsecore_gaps`)

| Question | Probe |
| --- | --- |
| Can one `pl.kernel` host a TensorCore mesh and a SparseCore mesh? | `sc_tc_one_kernel_two_meshes` |
| Can one `pl.kernel` host the scalar-subcore and vector-subcore meshes together (MPMD)? | `sc_two_sparsecore_meshes_one_kernel` |
| Can an MPMD SparseCore kernel allocate VMEM scratch? | `sc_mpmd_vmem_scratch` (`sparsecore_gaps`) |
| How does a DMA inside a two-mesh kernel name its core? | `sc_mpmd_dma_core_id` (`sparsecore_gaps`) |

**Scaling and coordination** (`sparsecore`, `sparsecore_gaps`)

| Question | Probe |
| --- | --- |
| How does uncoordinated work scale with subcore and core count? | `sc_subcore_scaling` |
| Does a bare `subcore_barrier()` complete, and at what subcore count does it stop? **Destructive: may hang the device.** | `sc_subcore_barrier_minimal` (`sparsecore_gaps`) |
| Does a cross-subcore reduction built from `subcore_barrier` and `fetch_and_add` work, and how does it scale? **Contains a barrier.** | `sc_multi_subcore_reduction` |

**Cost against the TensorCore** (`sparsecore`, `sparsecore_gaps`)

| Question | Probe |
| --- | --- |
| On a purely memory-bound stream, what fraction of the TensorCore's HBM bandwidth does SparseCore reach? | `sc_vs_tc_streaming_bandwidth` |
| On VPU-shaped elementwise math, how does SparseCore compare with the TensorCore VPU? | `sc_vs_tc_transcendental` |
| Is a SparseCore gather faster than the XLA gather it replaces? | `sc_gather_vs_xla_gather` |
| How does the gather advantage depend on row width? | `sc_gather_row_width_scaling`, `sc_gather_width_curve_fixed_chunk` |
| How does it depend on the number of indices? | `sc_gather_index_count_scaling` |
| Where does XLA's gather switch implementation, and does the switch move with table size? | `sc_xla_gather_cliff_bracket` |
| Is a SparseCore scatter-add faster than XLA's? Does fanning out across subcores rescue it? | `sc_scatter_add_vs_xla`, `sc_scatter_add_fanout` |
| Does independent SC work hide behind TC work in one jit? | `sc_tc_independent_overlap` |
| Does it still hide when the TC neighbour is memory-bound? | `sc_tc_overlap_vs_neighbour_kind` |
| What does handing an SC result to the TC cost? | `sc_tc_dependent_handoff` |

**Whole algorithms** (module `sparsecore_kernels`)

| Question | Probe |
| --- | --- |
| Does SC radix-select top-k return the same set as `jax.lax.top_k`? | `sc_topk_correctness` |
| How does it compare with `lax.top_k`, standalone? Beside a matmul? | `sc_topk_vs_lax_topk`, `sc_topk_frees_the_mxu` |
| Does SC MoE routing give the same counts and stable permutation as `argsort`? | `sc_moe_route_correctness` |
| How does it compare with XLA routing, standalone? Beside a matmul? Across shapes and neighbours? | `sc_moe_route_vs_xla`, `sc_moe_route_frees_the_mxu`, `sc_moe_route_offload_sweep` |

Architecture facts (meshes, memory spaces, capacities, banking, wave width, bandwidth scope) live
in [../tpu_hardware_facts.md](../tpu_hardware_facts.md) §10. This file does not repeat them; where
the two touch, §10 states the fact and this file states what to do about it. Placement against the
MXU and VPU: [compute_units.md](compute_units.md). Moves that cross the TensorCore/SparseCore
boundary: [cross_area_moves.md](cross_area_moves.md). SparseCore strategies:
`STRAT-SC-*` and the `STRAT-MLA-*` SparseCore entries in
[../strategies/INDEX.md](../strategies/INDEX.md).

---

## What it is, and why it matters

The SparseCore is a third compute unit beside the MXU and the VPU, and it is **the only one the
TensorCore program does not contend with**. Work moved onto it is not faster or slower so much as
*somewhere else*: the MXU it vacates can run a matmul at the same time. Every decision in this file
comes down to whether a given block is worth moving there.

Each SparseCore has one **scalar subcore** (the sequencer: control flow, scalar arithmetic,
dynamic indexing, DMA issue) and several **vector subcores**, each a narrow SIMD datapath over its
own private VMEM. There is no matrix unit. What it has instead is address generation: each subcore
can read and write HBM at data-dependent addresses, at a fine DMA granule, with many accesses in
flight.

**Read the geometry at run time; do not hard-code it.** Core count, subcore count, lane count,
per-subcore VMEM and DMA granule differ by chip generation. Read them from
`plsc.get_sparse_core_info()` (`SparseCoreInfo`) or `pltpu.get_tpu_info().sparse_core`. Source:
`jax/_src/pallas/mosaic/tpu_info.py`. Three readings are easy to get wrong:

- **`num_cores` is per TensorCore, not per chip.** `TpuInfo` is per-TensorCore throughout; multiply
  by the TensorCore count for a chip total.
- **`sparse_core is None` means "Pallas cannot see one", not "the chip has none".** Some chips
  that have SparseCores are simply not populated in this table. For a kernel, "can Pallas see one"
  is the right question.
- **A kernel that hard-codes the lane count is a single-generation kernel.** Lane count changes
  across generations, and narrower dtypes pack more elements per vector. Always use
  `num_lanes` from `SparseCoreInfo`.

**Budget against the TensorCore, structurally:**

- The native vector is `(num_lanes,)`, against the TensorCore's `(8, 128)` tile, so dense arithmetic
  is far narrower per instruction.
- The addressing granule is `dma_granule_size_bytes`, far finer than a TensorCore DMA block, so
  scattered reads are where it can win.
- There is no matmul, and of the transcendentals only `exp` lowers (see "What lowers").
- VMEM is per vector subcore and does not pool into one subcore's working set. SMEM is much smaller
  than the TensorCore's and may hold only 32-bit types. Capacities:
  [../tpu_hardware_facts.md](../tpu_hardware_facts.md) §10, `sc_vmem_per_subcore`,
  `sc_smem_capacity`.

hypothesis: SparseCore streams bulk data at a small fraction of the TensorCore's HBM bandwidth;
verify with `sc_vs_tc_streaming_bandwidth`. If that holds on the target, it is the first thing to
internalise: the reason to use the unit is that it is a *different* unit, not a better one.

---

## The programming surface

```python
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
```

The module path is `jax.experimental.pallas.tpu_sc`. It re-exports from
`jax/_src/pallas/mosaic/sc_core.py` and `sc_primitives.py`: two meshes, a parallel loop, and the
primitives that exist because the TensorCore cannot express them. Confirm what this image exports
with `sc_module_surface`. A probe that looks for SparseCore under any other path will report it
unreachable when it is not.

### Meshes

```python
scalar_mesh = plsc.ScalarSubcoreMesh(axis_name="core", num_cores=n_cores)
vector_mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore",
                                     num_cores=n_cores, num_subcores=n_subcores)
```

`VectorSubcoreMesh` is two-dimensional: `core` across SparseCores, `subcore` across each core's
vector subcores. Split work with `jax.lax.axis_index` on either axis. For a flat worker count over
both axes, use `jax.lax.axis_size(("core", "subcore"))` rather than computing
`core * num_subcores + subcore` by hand. A natural stripe is `num_lanes * num_workers`.

The two meshes accept different memory spaces. A scalar-subcore kernel cannot allocate private
`pltpu.VMEM` (table and rule: [../tpu_hardware_facts.md](../tpu_hardware_facts.md) §10;
`sc_scalar_kernel_runs`, `sc_vector_kernel_runs`).

### Memory spaces, in one rule

- **HBM refs are never operands.** Inputs and outputs arrive as HBM refs. DMA a chunk into
  core-local VMEM or SMEM, compute, then DMA the result back.
- **`VMEM_SHARED` is a DMA endpoint, not an operand.** Copy into local VMEM before reading it.
- **Scalar stores are SMEM-only**, including `obuf[0] = x` into a VMEM output staging buffer.
- **`fetch_and_add` targets SMEM**, so cross-subcore accumulators are bounded by the SMEM budget,
  not the VMEM one.

### Shapes are `(num_lanes,)`, and that is the whole layout story

The native shapes follow one rule, in `sc_core.supported_shapes(dtype)`
(`jax/_src/pallas/mosaic/sc_core.py`). Paraphrased: with `packing = 4 // itemsize`, a 4-byte dtype
has exactly one native shape, `(num_lanes,)`. A narrower dtype has `(num_lanes * packing,)` or
`(packing, num_lanes)`. There is no `(8, 128)` tile and no second-minor dimension, so **the
SparseCore has no TensorCore-style layout problem**: the concerns of
[layout_precision.md](layout_precision.md) do not apply. In their place, every vector op is
`num_lanes` wide, a kernel is a loop over lane groups, and loop overhead is what to watch.

Three properties of that rule matter downstream:

- **It is private and cannot be queried.** It lives in `jax._src` and is not re-exported, so
  `plsc.supported_shapes` raises `AttributeError` (`sc_module_surface`). You cannot assert a shape
  before compiling. The exception text is the rule's only public channel.
- **It writes the error message.** `_check_aval_is_supported` in
  `jax/_src/pallas/mosaic/sc_lowering.py` raises `<caller> only supports <dtype> arrays of shapes
  [<supported_shapes>], got <shape>`. It is called from `Get` and `Swap`, which means every load
  and every store.
- **It is enforced only when `needs_layout_passes=False`.** Both call sites are guarded by
  `if not ctx.lowering_context.needs_layout_passes`. So the flag that makes `scan_count`,
  `load_gather`, `store_scatter` and the reductions lower (see the error catalogue) is also the
  flag that turns on strict lane-group shape checking. The flag is per kernel: one op that needs it
  puts the whole body under the `(num_lanes,)` rule.

With the flag left at its default `True`, a whole-buffer copy or elementwise map (`obuf[...] =
buf[...] * 2` over many lane groups) lowers and computes correctly (`sc_wide_get_without_reduction`).
So the lane-group loop works around a compiler check, not the vector unit. **Keep the lane-group
loop as the default spelling anyway**:

- Nearly every primitive that justifies being on this unit needs the flag off, and so does
  `jnp.sum`.
- hypothesis: the lane-group loop is no slower than the wide spelling, and possibly faster; verify
  with `sc_wide_get_vs_lane_loop`.

**Control-cell rule for this surface.** `sc_wide_get_layout_passes` used `jnp.sum(buf[...])` as its
wide read. With the flag on, it failed at both widths, including the native one, because
`jnp.sum` lowers to `tpu.scan`, which the layout pass rejects. That table seems to say "wide Get
does not work". What it actually says is "`jnp.sum` does not work with the flag on". **When a probe
on this surface returns all failures, check that the arm meant as the control passed before
believing the arm you care about.** If the native-width control failed, the variable under test was
never exercised.

**Dtypes** (`sc_dtype_support`): f32, int32, uint32, bf16, f16 and int16 are the candidates to
check. int8 VMEM scratch is the known casualty (error below); widen to int16 or pack by hand with
`plsc.pack`.

`Tiling.SPARSE_CORE` and `infer_tiling` set the SparseCore tiling (`tpu_info.py`).
`pltpu.CompilerParams(use_tc_tiling_on_sc=...)` chooses between that and TensorCore tiling. It
applies only to SparseCore subcore kernels and defaults to True (docstring in
`jax/_src/pallas/mosaic/core.py`).

---

## Errors this surface produces, and what fixed each

**Toolchain-dependent; re-verify with the named probe.** Every row below was hit on one toolchain.
A newer toolchain may stop producing a message, reword it, or fix the gap. Treat a message as a key
to look up here, and re-run the probe before treating a row as a current limitation. With no other
documentation for this surface, the error text is the only thing that separates "this op does not
exist" (the design ends) from "you called it the wrong way" (a one-line fix).

**Before recording any op as absent:** if the error names a compiler flag, retry once with that
flag. An error that names a flag is a claim about your invocation, not about the op. Several ops
were once recorded absent this way and all lowered on retry.

| Error (verbatim, may be truncated) | What it means | Fix | Probe |
| --- | --- | --- | --- |
| `Get does not support loading from HBM.` / `Swap does not support storing to HBM.` | You wrote the TensorCore idiom `o_ref[...] = v` (or read `x_ref[...]`) on an HBM ref. Says **nothing** about the primitive under test. | Stage through core-local scratch: `obuf[...] = v; pltpu.sync_copy(obuf, o_ref)` | `sc_vector_kernel_runs` |
| `INTERNAL: Operation not supported in the Mosaic-SC infer-vector-layout pass. Set needs_layout_passes=False in the TPU custom call.` | The op exists and lowers; a pass you did not ask for rejects it. Seen on `tpu.scan_count`, `tpu.vector_load_idx` (behind `load_gather`), `store_scatter`, `addupdate_scatter`, and `tpu.scan`, which is what `jnp.sum` / `jnp.max` / `jnp.cumsum` / `jnp.sort` become, at every width including native. | `compiler_params=pltpu.CompilerParams(needs_layout_passes=False)`. Then budget for lane-group loops (next row). | `sc_primitive_lowering`, `sc_pl_ops_inside_kernel` |
| `NotImplementedError: Get only supports int32 arrays of shapes [(N,)], got (M,)` (N is `num_lanes`) | A whole-buffer read with `needs_layout_passes=False`. The bracketed list is `supported_shapes(dtype)` printed. | Loop over lane groups: `@pl.loop(0, chunk, step=num_lanes)` and work on `buf[pl.ds(i, num_lanes)]`. Or, if nothing in the body needs the flag off, leave the flag on and read the whole buffer. | `sc_wide_get_layout_passes`, `sc_wide_get_without_reduction` |
| `Swap only supports scalars in SMEM. Trying to swap a scalar in VMEM.` | A scalar store, which is a swap, into VMEM. Also fires on `obuf[0] = x` into a **VMEM output staging buffer**. This is a separate rank-0 guard from the shape check and fires regardless of the flag. | Scalar scratch goes in `pltpu.SMEM((n,), jnp.int32)`. To emit a scalar result, widen it: `obuf[...] = jnp.full((num_lanes,), x, jnp.int32)` with a `(num_lanes,)` `out_type`. | `sc_primitive_lowering`, `sc_pl_loop_init_carry` |
| `Integer indexing of refs that follows a non-trivial slice is not supported on SC` | `ref[pl.ds(i, n), col]` or `ref.at[:, col]`: an integer index after a slice. | Transpose so the integer index comes first: store `(width, rows)` and write `ref[col, pl.ds(i, n)]`, `ref.at[col]`. | `sc_gather_row_width_scaling` |
| `Cannot confirm that the memref is memory tile-aligned since its producer op tpu.memref_squeeze is not supported by this check` | `ref.at[col]` rank-reduces a 2-D gather table, and `tpu.vector_load_idx` cannot verify the result's tiling. No indexing spelling avoids it. | Flatten the table to 1-D and fold the column into the index: `plsc.load_gather(flat, [row * width + col])`. | `sc_gather_row_width_scaling` |
| `TypeError: ref must be an AbstractRef or TransformedRef, got JitTracer(Ref<vmem>{...})` | A bare scratch ref passed where a `TransformedRef` is wanted (`store_compressed`, `load_expanded` and similar). | Pass `ref.at[pl.ds(0, n)]`, not `ref`. | `sc_primitive_lowering` |
| `INTERNAL: Not implemented: Mask is not supported` | Depends on the call shape, not on the primitive. `store_compressed` lowers into a **VMEM scratch** sliced `.at[pl.ds(off, num_lanes)]` with `needs_layout_passes=False`. Writing straight to the kernel output ref raised this. `addupdate_compressed` raises it only with layout passes on. | Use a VMEM-scratch destination, sliced `.at[pl.ds(off, num_lanes)]`, with `needs_layout_passes=False`. Only if that refuses, hand-roll with `plsc.cumsum` over the mask, then `plsc.store_scatter`. | `sc_primitive_lowering`, `sc_addupdate_variants` |
| `INTERNAL: Offsets along tiled dimensions must be aligned to tiles.` | An int8 VMEM scratch. | Use int16 or wider, or pack by hand. | `sc_dtype_support` |
| `'memref.alloca' op E3000` / `E3000: CompileTimeSparseCoreAllocationFailure` | Scratch exceeds the allocator's pooled per-core budget. A scratch that compiled alone can be refused once another is added. | Shrink or window the scratch; budget across all subcores in use. | `sc_vmem_per_subcore`, `sc_smem_capacity` |
| `NotImplementedError: MPMD map with more than one mesh requires scratch_type to have a core_type specified, but scratch_type=MemoryRef(..., memory_space=SMEM) is missing it` | An unscoped scratch in a two-mesh kernel. An MPMD kernel must be told which core owns each buffer. | Scope it: `(pltpu.SMEM @ mesh)(shape, dtype)`. `pltpu.SemaphoreType.DMA @ mesh` returns the ref itself, not a callable. | `sc_two_sparsecore_meshes_one_kernel`, `sc_mpmd_vmem_scratch` |
| `VerificationError: ... 'tpu.enqueue_dma' op Core id must be specified` | A DMA inside a two-mesh kernel whose scratch is correctly scoped. The copy must also name a core. | **Open.** Try remote-copy spellings that carry a `device_id`, with the shared core axis name on every arm. | `sc_mpmd_dma_core_id` |
| `ValueError: ScalarSubcoreMesh(axis_name=..., num_cores=...) should have the same core axis name and number of cores as the VectorSubcoreMesh ...` | An MPMD mesh pair built with independently chosen axis names. The two meshes are two views of the same cores. | Give both the same core axis name and `num_cores`. | `sc_mpmd_dma_core_id` |
| `NotImplementedError: Get does not support loading from VMEM_SHARED. Copy the data to a core-local memory space, e.g. VMEM, via pltpu.async_copy.` | `VMEM_SHARED` read as an operand. | `pltpu.async_copy(shared_ref, local_vmem, sem).wait()`, then read the local copy. | `sc_mpmd_vmem_scratch` |
| `ValueError: Mesh self=TensorCoreMesh(...) is not compatible with other_mesh=VectorSubcoreMesh(...)` | A TensorCore mesh and a SparseCore mesh in one `pl.kernel`. Behind it, `pallas_call_registration.py` also refuses TC in an MPMD map (`mpmd_map does not support TC kernels yet.`). | No intra-kernel fix. Compose at the XLA level: put both calls in one `jax.jit` (see "Piping"). | `sc_tc_one_kernel_two_meshes` |
| `NotImplementedError: Unimplemented primitive in Pallas TPU lowering for sc_vector_subcore: <op>` | Genuinely absent. Seen on `log`, `rsqrt`, `tanh`, `dot_general`. | Redesign: split the block, or leave it on the TensorCore. `exp` is the transcendental that lowers. | `sc_pl_ops_inside_kernel` |
| `VerificationError: Pallas encountered an internal verification error.` on `iota` | f32 iota only. | Build the index as `jnp.arange(n, dtype=jnp.int32)` and cast. | `sc_pl_ops_inside_kernel` |
| `ValueError: Unsupported TPU device kind: cpu` / `Only interpret mode is supported on CPU backend.` / `MLIR translation rule for primitive 'parallel_loop' not found for platform cpu` | A dry-run on CPU. | None. See "It cannot be dry-run". | `sc_kernel_cannot_dry_run` |
| **No error: the job hangs and its log stays empty** | `plsc.subcore_barrier()` above some subcore count. | Run barrier code alone, under a wall-clock timeout. See "Pitfalls". | `sc_subcore_barrier_minimal` |

### Four errors that point away from the bug

**1. The HBM error masks the op you are testing.** It is raised at the store. So a kernel testing
`plsc.cumsum` that stores straight to an HBM output reports the HBM error and says nothing about
`cumsum`. Stage the output through scratch first, then believe the next error.

**2. `needs_layout_passes=False` is the default posture, and it has a cost.** Set it at the kernel:

```python
@pl.kernel(..., compiler_params=pltpu.CompilerParams(needs_layout_passes=False))
```

`CompilerParams.needs_layout_passes` defaults to True, and its docstring calls the flag temporary,
so check on each toolchain whether it is still needed (`sc_primitive_lowering`,
`sc_pl_ops_inside_kernel`). The cost is the strict `(num_lanes,)` check on every `Get` and `Swap`:
expect to write lane-group loops around whatever primitive you turned the flag off for.

**3. Refs are effectively column-major.** An integer index may not follow a slice. A natural
row-major embedding gather such as `obuf[pl.ds(i, L), col] = plsc.load_gather(tbl.at[:, col], ...)`
fails on both sides. Store multi-column buffers `(width, rows)` and plan the transposes at the
kernel boundary, where they are free; inside the kernel they are impossible. For a **gather
source**, transposing is not enough, because rank-reducing a 2-D ref fails the tile-alignment
check. **A gather source should be one 1-D ref, and any structure in it belongs in the index
arithmetic** (`row * width + col`). That is the table's real memory layout anyway.

**4. `Swap only supports scalars in SMEM` fires on output buffers.** `obuf[0] = value` into a VMEM
output staging buffer looks nothing like scalar scratch, yet it raises this error. A VMEM store is
a lane group, so every emit must be a full `(num_lanes,)` vector. This one sat latent in several
probes, and each attributed the failure to the primitive under test. **When every arm of a
differential probe fails identically, the defect is in the shared scaffolding, not in what the arms
differ on.** A probe that reports a verdict without checking that at least one arm ran reports
green over an empty measurement.

### Ops that return a wrong answer instead of an error

- **`plsc.scan_count` returns a 1-based count, not a 0-based rank.** Scattering `run + 1`
  over-counts every duplicate run by one, with no diagnostic. The error appears only in buckets
  that had a duplicate inside one vector, so a uniform test input looks almost right. When `run`
  is used to place an element, the placement is `base + run - 1` (`sc_primitive_lowering`,
  `sc_moe_route_correctness`).
- **`plsc.pack(first, second)` puts `first` in the low 16 bits of the word.** Getting the order
  backwards yields a plausible tensor with every pair transposed, and no diagnostic
  (`sc_bf16_widen_and_pack`).
- **A candidate buffer sized for the typical case** overflows on skewed input and returns a wrong
  answer, not an error (see "Patterns that work").

On this surface, be as suspicious of ops that return an answer as of ops that refuse to compile.

---

## Writing a kernel

Every SparseCore kernel has the same shape: inputs arrive as HBM refs, you DMA a chunk into VMEM or
SMEM, work on it one lane group at a time, and DMA the result out. An illustrative skeleton (API
shape only):

```python
info = plsc.get_sparse_core_info()
L = info.num_lanes

@pl.kernel(
    out_type=jax.ShapeDtypeStruct((n,), jnp.int32),
    mesh=plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore",
                                num_cores=info.num_cores, num_subcores=info.num_subcores),
    scratch_types=[pltpu.VMEM((chunk,), jnp.int32),   # in/out staging
                   pltpu.SMEM((8,), jnp.int32),        # named scalar slots
                   pltpu.SemaphoreType.DMA],
    compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
)
def kernel(x_hbm, o_hbm, vbuf, smem, sem):
    off = worker_offset()                               # from axis_index on both axes
    pltpu.sync_copy(x_hbm.at[pl.ds(off, chunk)], vbuf)  # in

    @pl.loop(0, chunk, step=L)                          # work, one lane group at a time
    def _(i):
        vbuf[pl.ds(i, L)] = vbuf[pl.ds(i, L)] * 2 + 1

    pltpu.sync_copy(vbuf, o_hbm.at[pl.ds(off, chunk)])  # out
```

Deviating from this skeleton, for example by reading or writing `x_hbm[...]` directly, is what
produces the HBM errors.

**Three loops, and picking the wrong one is a correctness bug, not a slow kernel:**

- **`pl.loop(lower, upper, step=...)`** runs sequentially, so iterations may depend on each other.
  It accepts `init_carry=` and returns the final carry, whether the carry is a scalar or a vector
  (`sc_pl_loop_init_carry`). Bounds may be traced values.
- **`plsc.parallel_loop(lower, upper, step, *, unroll, carry)`** lets the compiler reorder and
  overlap iterations. Its docstring says an iteration may not depend on another iteration's side
  effects, especially ref writes. A step smaller than the width written leaves the overlap region
  **undefined**, not merely racy. It takes a pytree `carry`, but refs may not be carried.
- **`jax.lax.fori_loop` / `while_loop`** lower through the registered `scan` / `while` rules.
  Data-dependent trip counts: `sc_data_dependent_control_flow`,
  [STRAT-MLA-08](../strategies/STRAT-MLA-08-sparsecore-data-dependent-skip-lowers.md).

**DMA.** `pltpu.sync_copy` issues and waits one copy at a time. To overlap several copies, collect
`pltpu.make_async_copy(...)` objects, start them all, then wait on them all. hypothesis: a handful
of chunks per batch helps, and a deep split loses to per-DMA issue and wait overhead. No registry
probe measures this yet, so time it in the kernel. `pltpu.emit_pipeline` also lowers inside a
SparseCore kernel. Import it from `pltpu`, not `plsc`, where nobody finds it.
See [STRAT-SC-04](../strategies/STRAT-SC-04-emit-pipeline-scratches-noop.md). DMAs with a traced
extent: [STRAT-MLA-12](../strategies/STRAT-MLA-12-sparsecore-traced-dma-needs-a-1d-ring.md).

**Gather straight from HBM.** Indexing an HBM ref with an indices ref inside a copy,
`pltpu.sync_copy(table_hbm.at[idx_ref], vmem_buf)`, is the embedding-lookup path. `load_gather`
and `store_scatter` work on VMEM.

---

## The primitives, and what each is for

These are the reason to be on this unit. Each one does something the VPU either cannot do or can
only do by falling back to a sort. Whether each lowers is toolchain-dependent: check with
`sc_primitive_lowering` and `sc_addupdate_variants`.

**Data-dependent memory.**

| Primitive | Does |
| --- | --- |
| `load_gather(ref, indices, *, mask)` | `ref[indices]` from VMEM, one index array per dimension |
| `store_scatter(ref, indices, x, *, mask)` | `ref[indices] = x` |
| `addupdate_scatter(ref, indices, x, *, mask)` | `ref[indices] += x`, atomically; one index array per dimension, so a 2-D accumulator takes `(row_idx, col_idx)` |
| `addupdate(ref, x)` | atomic add over the whole ref |
| `pltpu.sync_copy(hbm.at[idx_ref], vmem)` | gather straight out of HBM |

**Compaction:** turning a mask into a dense array without a sort.

| Primitive | Does | Lowering (toolchain-dependent) |
| --- | --- | --- |
| `store_compressed(ref, x, *, mask)` | writes the true lanes of `x` to `ref[0], ref[1], ...` in order | lowers into a VMEM scratch at `.at[pl.ds(off, num_lanes)]` with the flag off; other ref forms (a direct output-ref write) refused |
| `load_expanded(ref, *, mask)` | the inverse | refused once, in the same call shape that refused `store_compressed`; not separately retested |
| `addupdate_compressed(ref, x, *, mask)` | compressed accumulate | lowers with the flag off; refused with it on |

The append-in-a-loop pattern this exists for:

```python
plsc.store_compressed(buf.at[pl.ds(woff, L)], vals, mask=sel)
woff = woff + plsc.all_reduce_population_count(sel)[0]
```

One refusal is evidence about one call shape, not about the primitive. An earlier reading of a
single `Mask is not supported` as a blanket gap would have cost a kernel its cheapest inner loop.
Two primitives can produce identical mask error text for different causes (a lowering gap versus a
layout-pass rejection). The flag separates them.

**Scans and reductions.** All are masked and all run along the innermost axis.

| Primitive | Does |
| --- | --- |
| `cumsum(x, *, mask)` | prefix sum over eligible lanes |
| `cummax(x, *, mask)` | prefix max |
| `scan_count(x, mask) -> (run, last)` | per-lane running duplicate **count** (1-based), plus a mask marking the last lane of each duplicate run |
| `sort_key_val(keys, values, *, mask, descending)` | sorts one vector; shape must be exactly `(num_lanes,)` |
| `all_reduce_ffs(x, *, reduce)` | index of the first true lane, or `x.size` |
| `all_reduce_population_count(x, *, reduce)` | number of true lanes |

`scan_count` is what makes a histogram correct. A scatter-add of ones silently loses duplicates
that land in the same vector, because the lanes race. `scan_count` returns each lane's count within
its duplicate run plus a mask selecting the last lane of each run. One masked `addupdate_scatter`
of `run` then adds the whole run.

Prefer `plsc.cumsum` and `plsc.sort_key_val` over `jnp.cumsum` and `jnp.sort` even where the `jnp`
forms lower. The `plsc` forms take the mask, and `sort_key_val` carries the payload.

**Cross-subcore.**

| Primitive | Does |
| --- | --- |
| `subcore_barrier()` | blocks until every subcore of the core arrives; `VectorSubcoreMesh` only. **Can hang the device above some subcore count**; see "Pitfalls" |
| `fetch_and_add(smem_ref, value, *, subcore_id)` | atomic add into another subcore's SMEM, returning the old value |

Those two plus `VMEM_SHARED` are the whole cross-subcore toolkit. There is no shuffle and no
cross-subcore all-reduce, so a cross-subcore prefix sum must be built by hand from `fetch_and_add`
and a barrier. Whether that works, and how it scales, is open (`sc_multi_subcore_reduction`). This
is where a SparseCore kernel stops being easy. Loop carries across a barrier:
[STRAT-SC-03](../strategies/STRAT-SC-03-barrier-the-loop-carry.md).

**Bit manipulation.** `bitcast(x, dtype)` keeps the rank and resizes the minormost dimension,
unlike `lax.bitcast_convert_type`. `pack` / `unpack` take `PackFormat.COMPRESSED` or `INTERLEAVED`.
Both carry a docstring warning that they are temporary until the compiler packs automatically.
Rank-0 SMEM tricks: [STRAT-MLA-09](../strategies/STRAT-MLA-09-sparsecore-rank0-smem-xor.md).

---

## What lowers, and what does not

**Probe-verified, toolchain-dependent; re-run `sc_primitive_lowering` and
`sc_pl_ops_inside_kernel` on the current image.** A registered lowering rule is not a working op;
these lists come from real kernels run on hardware. Registered rules live in
`jax/_src/pallas/mosaic/lowering.py` and `sc_lowering.py`. The vector-subcore kernel type registers
every scalar-subcore rule plus `iota`, `reduce_max`, `reduce_min`, `reduce_sum`, `rev` and `sort`.

**Registered for SparseCore and worth knowing:** `add sub mul div rem neg abs sign min max clamp`,
`and or xor not shift_left shift_right_{arithmetic,logical}`, `eq ne lt le gt ge select_n`, `exp`,
`cumsum`, `sort`, `unique`, `gather`, `scatter`, `iota`, `reduce_{sum,max,min,and,or}`,
`concatenate reshape slice split squeeze stack unstack rev`, `cond scan while jit run_scoped`,
`dma_start dma_wait semaphore_{read,signal,wait} barrier`,
`bitcast bitcast_convert_type convert_element_type broadcast_in_dim`, `debug_print`.

**Verified working inside a kernel body** (`sc_pl_ops_inside_kernel`; `*` = only with
`needs_layout_passes=False`): `add`, `mul`, `div`, `exp`, `select`, bf16 cast, `jnp.max`\*,
`jnp.sum`\*, `jnp.cumsum`\*, `jnp.sort`\*. Integer `jnp.arange`.

**Verified primitives** (`sc_primitive_lowering`, `sc_addupdate_variants`): `load_gather`,
`store_scatter`, `addupdate_scatter`, `scan_count`, `cumsum` (flag off); `addupdate` (either flag
setting, the one primitive seen lowering with layout passes on); `addupdate_compressed` (flag off);
`store_compressed` in the VMEM-scratch slice form (flag off).

**Absent, and these end an offload:**

- `dot_general`: no matmul at all, and no `matmul_acc_lhs` / `matmul_push_rhs` / `matmul_pop`.
- Every transcendental except `exp`: no `log`, `tanh`, `logistic`, `rsqrt`, `sqrt`, `sin`, `cos`,
  `tan`, `atan2`, `erf`, `pow`, `exp2`, `log1p`, `reciprocal`, `square`, `integer_pow`. `log`,
  `rsqrt` and `tanh` were each put through a real kernel and refused.
- `argmax` / `argmin`. Use `reduce_max`, then recover the index with `all_reduce_ffs` over an
  equality mask.
- `transpose`, `pad`, `roll`, `tile`, `broadcast_to`.
- `floor`, `ceil`, `round`, `is_finite`, `population_count`, `clz`.
- f32 `iota` (use int32 and cast).
- The whole PRNG family (`random_bits`, `threefry2x32`, `prng_seed`, ...). **No RNG on the
  SparseCore**: generate on the TensorCore and hand the numbers over.

The pattern is consistent. The SparseCore does addressing, comparison, integer arithmetic and
control flow; it does not do maths. A block that is mostly gather with a little `rsqrt` cannot move
as a whole. Split it, or leave it on the TensorCore.

### It cannot be dry-run

`JAX_PLATFORMS=cpu` works for TensorCore Pallas kernels and fails here at three levels
(`sc_kernel_cannot_dry_run`):

1. `VectorSubcoreMesh.__post_init__` calls `get_tpu_info()`, which raises
   `ValueError: Unsupported TPU device kind: cpu`.
2. With `SparseCoreInfo` faked: `ValueError: Only interpret mode is supported on CPU backend.`
3. With `interpret=pltpu.InterpretParams()`:
   `NotImplementedError: MLIR translation rule for primitive 'parallel_loop' not found for platform cpu`.

**Tracing does work off-device.** Monkeypatch `get_sparse_core_info` to return a realistic
`SparseCoreInfo` and call `jax.eval_shape`. That runs the kernel body to a jaxpr and catches shape
mismatches, wrong primitive signatures and bad ref slices, which is most of what a first draft gets
wrong. Do this before spending a TPU run. It does not check lowering, and lowering is where the
remaining failures are.

---

## Piping SparseCore with the TensorCore

**One Pallas kernel cannot contain both.** A TensorCore mesh beside a SparseCore mesh is rejected
by the mesh-compatibility check, and again by the MPMD path
(`jax/_src/pallas/mosaic/pallas_call_registration.py`; probe `sc_tc_one_kernel_two_meshes`). You
cannot write a fused kernel whose prologue runs on the SparseCore and whose matmul runs on the MXU.
The SC-to-TC handoff is never intra-kernel.

Two *SparseCore* meshes in one kernel (scalar sequencer plus vector subcores) get past that check.
They need core-scoped scratch and a shared core axis name. Whether DMAs inside such a kernel can
name their core is open (`sc_two_sparsecore_meshes_one_kernel`, `sc_mpmd_vmem_scratch`,
`sc_mpmd_dma_core_id`). Treat scalar+vector MPMD as unproven, so the "scalar subcore prepares
indices while vector subcores gather" pattern remains a design on paper until those probes pass.
`pallas_call_registration.py` also raises `Invalid core type for VMEM` for non-TC core types, so a
mesh's listed memory spaces and what lowering accepts may disagree.

**Composition is at the XLA level.** Put both calls in one `jax.jit`, and XLA schedules them
concurrently. There is no async-launch API, no stream, and nothing to wait on. The only way to ask
for overlap is to put the SparseCore call and the TensorCore work in the same jit **with no data
dependence between them**. Public guide: https://docs.jax.dev/en/latest/pallas/tpu/sparsecore.html

```python
@jax.jit
def step(a, b, ids):
    return a @ b, sc_route(ids)          # independent: can overlap

@jax.jit
def step_serial(a, b, ids):
    counts, perm = sc_route(ids)         # dependent: the matmul waits
    return a[perm] @ b, counts
```

Both compile and run; only the first can hide anything. The second is how an MoE layer looks when
written in the obvious order, because routing genuinely precedes the matmul it feeds. So **an
offload's independent neighbour has to come from a different layer or the previous step**, not
from the op next to it.

### Independent overlap

Measure with `sc_tc_independent_overlap`: SC alone, TC alone, both in one jit. Score it as
(definitions, not measurements):

```
hidden        = sc + tc - both
max_hideable  = min(sc, tc)
overlap_score = hidden / max_hideable      # 1.0 = perfect concurrency, 0.0 = fully serial
added_by_sc   = both - tc                  # the budgetable quantity
```

The denominator is `min(sc, tc)`, not `sc`, because you can never hide more than the shorter
kernel. "Score = fraction of SC work hidden" is true only when the SC kernel is the shorter of the
two. **Plan with `added_by_sc`, not the score**: when the neighbour is short, a moderate score can
still mean most of the SC work lands on the step.

Two things govern it:

1. **The duration ratio.** No kernel hides more than its neighbour is long. Design against
   "neighbour longer than the SC kernel", not "neighbour enormous". hypothesis: once the neighbour
   is comfortably longer, making it longer still buys nothing; verify with
   `sc_moe_route_offload_sweep`.
2. **What the neighbour is bound by.** hypothesis: SC work hides well behind a compute-bound
   matmul and poorly behind a memory-bound neighbour, even when the neighbour's HBM traffic is
   reduced and the combined step is far from the bandwidth roofline. The mechanism is not
   established. Verify with `sc_tc_overlap_vs_neighbour_kind`. **The offload case is a
   compute-bound-neighbour case** until that probe says otherwise on the target.

### Dependent handoff

Measure with `sc_tc_dependent_handoff`: the SC stage alone, the TC stage alone, and a chain where
the TC consumes the SC output in one jit. `handoff_ratio = chained / (sc + tc)`, where 1.0 means
the stages simply add and crossing units is free. hypothesis: the ratio is about 1.0, so the
crossing itself costs nothing measurable; verify with `sc_tc_dependent_handoff`.

Read what that does not say. The serial case pays the SC stage's full duration. Independent and
serial use the same unit and the same cheap handoff, yet the cost can differ by an order of
magnitude. A free handoff is therefore no argument for offloading. **The question that decides an
offload is whether the TensorCore has something else to do.** Placement across units:
[cross_area_moves.md](cross_area_moves.md).

---

## What it costs against the TensorCore

Every past measurement here is a direction. Price each on the target with its probe before
designing around it.

| Question | Hypothesis (direction only) | Probe(s) |
| --- | --- | --- |
| Bulk streaming bandwidth, SC vs TC | hypothesis: SC reaches a small fraction of TC bandwidth, and its spread is low once uncontended; anything that touches every element of a big tensor once belongs on the TC | `sc_vs_tc_streaming_bandwidth` |
| Elementwise maths, SC vs TC VPU | hypothesis: the TC wins by roughly the bandwidth ratio, not an arithmetic ratio; chained `exp`s cost little more than one multiply on SC, so once data is on a subcore, extra arithmetic is nearly free | `sc_vs_tc_transcendental` |
| Uncoordinated subcore scaling | hypothesis: near-linear at low subcore counts, flattening as the SC path to HBM saturates; the second core is free at low occupancy and not at high; a single-subcore kernel leaves most of the unit idle | `sc_subcore_scaling` |
| Gather vs XLA gather | hypothesis: SC gather can beat the XLA gather by a large factor at narrow rows with a VMEM-resident table and enough indices | `sc_gather_vs_xla_gather` |
| Gather vs row width | hypothesis: the advantage shrinks with row width and inverts to an XLA win at wide rows; SC useful throughput saturates once a row fills `dma_granule_size_bytes`, and narrower rows waste granule bytes | `sc_gather_row_width_scaling`, `sc_gather_width_curve_fixed_chunk` |
| Gather vs index count | hypothesis: SC gather is mostly fixed cost at small index counts, where XLA wins; XLA has a fast path it falls off sharply at some index count, and SC wins above that | `sc_gather_index_count_scaling` |
| Where XLA's gather switches path | hypothesis: the switch is a sharp discontinuity, not a capacity threshold; a small table can be on the slow path at every index count; the threshold moves outward with table size; the selection rule is unknown | `sc_xla_gather_cliff_bracket` |
| Scatter-add vs XLA | hypothesis: SC scatter-add may lose to XLA; fanning out to private per-subcore histograms narrows but does not close the gap, and the fan-out saturates far below the streaming scaling | `sc_scatter_add_vs_xla`, `sc_scatter_add_fanout` |
| Overlap behind compute- vs memory-bound neighbour | see "Independent overlap" | `sc_tc_independent_overlap`, `sc_tc_overlap_vs_neighbour_kind` |
| SC to TC handoff | see "Dependent handoff" | `sc_tc_dependent_handoff` |
| Top-k, SC radix select vs `lax.top_k` | hypothesis: SC loses standalone, with the gap growing in `n` (the TC baseline is launch-bound, the SC kernel does O(n) work per pass); beside a matmul, SC hides a larger fraction of itself but can still add more time | `sc_topk_vs_lax_topk`, `sc_topk_frees_the_mxu` |
| MoE routing, SC vs `bincount` + stable `argsort` | hypothesis: roughly parity standalone on one subcore; beside a matmul, SC adds less step time than XLA, across token/expert shapes and neighbour sizes | `sc_moe_route_vs_xla`, `sc_moe_route_frees_the_mxu`, `sc_moe_route_offload_sweep` |

**Mechanisms behind the directions** (these hold whatever the numbers turn out to be):

- **Gather is a read; scatter-add is a read-modify-write.** A gather's lanes issue independent
  loads with no coordination. A scatter-add into shared buckets needs `scan_count` to serialise
  duplicates within each vector, plus a fold of private tables when fanned out, and that fold
  grows with subcore count. "Data-dependent addressing" is not one workload, so **do not carry the
  gather result over to a scatter.**
- **Do not project a subcore-scaling factor measured on one workload onto another.** The streaming
  curve parallelises work that needs no coordination; coordinated work scales differently.
- **Per-primitive wins do not compose.** An algorithm built from gather-shaped primitives can lose
  if it makes several passes over the data. On a bandwidth-poor unit, passes are the currency.
- **XLA's gather has a fast path that is easy to fall off.** If a gather in a real model is
  unexpectedly slow, check this first. Time it at two index counts and at your table size, in
  seconds, before anyone writes a SparseCore kernel. Never look the threshold up; measure it at
  your shape.
- **A launch-bound TensorCore baseline is almost unbeatable standalone**, because the offload pays
  its own dispatch on top. It is not unbeatable deployed: when the TC version's overhead contends
  with the matmul and the SC version's does not, the SC version can win in place.

Achieved SC bandwidth is an all-subcore aggregate, not a per-subcore ceiling, and it depends on
access width ([../tpu_hardware_facts.md](../tpu_hardware_facts.md) §10;
[STRAT-MLA-06](../strategies/STRAT-MLA-06-sparsecore-bandwidth-is-width-dependent.md)).

---

## When to reach for it

Reach for the SparseCore when the block is **addressing-bound, not arithmetic-bound**: what makes it
slow on the TensorCore is a data-dependent access pattern. The public guide lists gather/scatter,
sorting, unique, counts, histograms, ragged operations and small vector arithmetic
(https://docs.jax.dev/en/latest/pallas/tpu/sparsecore.html). Typical inference-stack candidates:

- **MoE routing**: count tokens per expert, prefix into offsets, emit the permutation that groups
  tokens by expert. On the TensorCore this is an `argsort`, even though the ordering is already
  implicit in the ids.
- **Top-k over a vocabulary.**
- **Ragged gather, scatter and gather-reduce.**
- **Block-table and page-index construction** for paged attention.
- **Embedding lookups** with narrow rows.

**Do not** reach for it for anything with a matmul, anything whose inner loop is a transcendental
other than `exp`, anything that needs an RNG, or dense elementwise work over a large tensor.

**The prior question is the roofline.** A block that is a small share of device time cannot repay a
SparseCore kernel, and the profile cannot see inside the custom call afterwards. Model the floor
first (the roofline in [../tpu_memory_overlapping.md](../tpu_memory_overlapping.md) §1). See
[STRAT-SC-08](../strategies/STRAT-SC-08-prove-the-load-binds-first.md) and
[STRAT-SC-09](../strategies/STRAT-SC-09-unit-choice-follows-what-binds.md).

### Three conditions an offload must clear

1. **The TensorCore must have independent work while the SparseCore runs**, ideally compute-bound
   (`sc_tc_independent_overlap`, `sc_tc_overlap_vs_neighbour_kind`). Without an independent
   neighbour, the SC version is just the slower version.
2. **Compare added time beside that neighbour, not standalone time.** Standalone comparisons
   mislead in both directions. Use three arms, always: `tc_only`, `tc_plus_baseline`, `tc_plus_sc`.
   Compare each arm on what it **adds** to `tc_only`. The `*_frees_the_mxu` probes have this shape.
3. **The SparseCore work must fit in the shadow available.** Estimate its duration from the
   streaming ratio measured on the target (`sc_vs_tc_streaming_bandwidth`), and size the offloaded
   block against the neighbour's duration. Count **passes over the data**, not operations.

The corollary inverts the usual advice: **do not offload a block because the SparseCore is faster
at it.** Usually it is not. Offload because the SparseCore is idle and the TensorCore is not, and
treat the SC's slowness as the budget you spend.

**The one outright-win candidate:** narrow-row gather from a VMEM-resident table at enough indices
to clear the SC's fixed cost and push XLA off its fast path. Here the SparseCore can be the first
choice rather than the spare-capacity choice. All three qualifiers bear load: table fits and
staging amortises, row is narrow, index count is high. Missing any one of them can turn it into an
XLA win. Verify each on the target with the gather probes above.

### Reading SparseCore time out of a profile

Two trace lines look as if they measure the same thing, and they do not:

- **`TEC 0` is busy time.** Use it for occupancy, overlap fractions, and any claim about what the
  SparseCore was doing.
- **`Sparse Core Modules` is a launch-to-completion envelope that includes idle wait.** On a block
  whose two units are strictly serial, it shows as one event spanning almost the whole block.

Substituting the envelope for busy time gives the **opposite** answer, in the flattering direction:
an overlap analysis built on the envelope can report full overlap for units that were in fact
disjoint. Whenever an overlap fraction comes back near 1.0 for an arm you expected to be serial,
suspect the envelope. Merge busy intervals with a tolerance, because zero-tolerance merging splits
contiguous runs at scheduling-noise gaps. Sanity-check the interval count against what the kernel
launches. See [../tpu_profiling_and_diagnostics.md](../tpu_profiling_and_diagnostics.md) and
[STRAT-SC-12](../strategies/STRAT-SC-12-measure-at-the-scope-the-work-lives.md).

---

## Patterns that work

**Free the MXU by co-scheduling, not by "launching".** Put the SC call and independent TC work in
one jit with no data dependence. Source the independent work from another layer or the previous
step. Measure it with the three-arm pattern above.

**Histogram / count-by-key.** This is the most useful idiom on the unit, and it is wrong by default:

```python
run, last = plsc.scan_count(keys)                        # run is 1-based
plsc.addupdate_scatter(hist, (keys,), run, mask=last)    # run, NOT run + 1
```

**Stable placement uses the same primitive to position rather than count.** The first duplicate
goes at `base + 0`:

```python
base = plsc.load_gather(offsets, (keys,)) + plsc.load_gather(fill, (keys,))
plsc.store_scatter(perm, (base + run - 1,), token_index)
plsc.addupdate_scatter(fill, (keys,), run, mask=last)
```

**Compaction: try `store_compressed` first.** Write into a VMEM scratch sliced
`.at[pl.ds(off, num_lanes)]`, with the flag off, appending at a running offset advanced by
`all_reduce_population_count`. Only if that refuses in your kernel, hand-roll it: a prefix sum over
the mask is a compaction, and the same mask discards lanes with meaningless destinations:

```python
rank = plsc.cumsum(jnp.where(mask, 1, 0))
dest = jnp.minimum(base + rank - 1, limit - 1)   # clamp keeps an overrun in bounds
plsc.store_scatter(buf, (dest,), value, mask=mask)
```

For a `+=` into the compacted destination, `addupdate_compressed` lowers with the flag off and
needs no hand-roll. Pass cost: [STRAT-SC-01](../strategies/STRAT-SC-01-compaction-pass-cost.md).

**Running scalars: a `pl.loop` carry or SMEM.** `init_carry=` works for scalar and vector carries
(`sc_pl_loop_init_carry`). Use SMEM when the value must survive several loops or an enclosing
window. It must be SMEM, not VMEM, because a scalar store is a swap. Name the slots, since a
SparseCore kernel accumulates many of them:

```python
_R, _LO, _HI, _RUN, _FOUND, _COUNT, _OFF, _USED = range(8)   # indices into pltpu.SMEM((8,), jnp.int32)
```

**bf16 travels as packed uint32 and is widened in the bit pattern.** SparseCore arithmetic is
f32/int32, and bf16 is the top 16 bits of an f32 of the same value. To widen, shift the wanted
half into the high 16 bits, mask with `0xFFFF0000`, and `plsc.bitcast` to f32. To narrow,
`plsc.pack(first, second, format=plsc.PackFormat.INTERLEAVED)` and bitcast to uint32, with `first`
in the low half. Accumulate in f32 and pack once at the end. Verify the round trip and the order
with `sc_bf16_widen_and_pack`.

**Multi-column buffers are column-major,** `(width, rows)`. **Gather sources are 1-D,** with
structure in the index arithmetic.

**Fuse aggressively; minimise passes, not operations.** Once data is on a subcore, arithmetic is
cheap and another trip over the array is not (hypothesis; `sc_vs_tc_transcendental`). Given a
choice between more work per pass and more passes, take the work.

**Bound resident scratch with a window, not with the input size.** Stage one window of the input
(for example, of a sort permutation) at a time under a `pl.loop` with a traced bound, so scratch
stops growing with the problem. With per-subcore VMEM this small, that is the difference between a
kernel with a shape limit and one without.

**Prefer O(1) state to data-dependent state.** A radix select that copies survivors into a
candidate buffer must size it for the worst case, every key in one bucket, which is `n` and does
not fit. Sized optimistically, it returns a wrong answer, not an error. Carrying an `[lo, hi]`
range in two SMEM scalars and re-scanning costs extra passes and cannot overflow. With small fixed
scratch and no dynamic allocation, take the constant-factor cost.

**A radix select's cost is set by the input distribution, and synthetic inputs flatter it.** The
first level of an MSB radix select on monotone-mapped f32 keys buckets on sign plus exponent, so
each bucket spans whole binades. On a strictly positive, narrow-dynamic-range distribution (scores,
norms, anything after a `relu` or `exp`), nearly everything lands in one bucket and the first level
separates almost nothing. Generators that are easy to write (`jax.random.normal`) sit at the easy
end. **Measure the real distribution before designing the selection**: report the occupancy of the
top first-level bucket and the high quantiles from one real input, a cheap host-side pass. If the
quantiles are close to shape-invariant across the serving range, a static threshold becomes
defensible ([algorithm.md](algorithm.md)). See
[STRAT-SC-07](../strategies/STRAT-SC-07-distribution-sensitivity-is-a-conflict.md).

**Mind the wave and the banks.** A wave of vector subcores costs as much as its longest row, and
lane-private tables whose stride shares a factor with the bank count serialise. See
[STRAT-SC-05](../strategies/STRAT-SC-05-wave-longest-row.md),
[STRAT-SC-06](../strategies/STRAT-SC-06-pad-scatter-stride-off-bank-count.md).

**Check the answer on an all-identical input.** The `scan_count` off-by-one and the candidate-buffer
overflow are invisible on random data and immediate on an input where every element ties, where the
right histogram is `n` in one bucket and nothing else. Give the correctness oracle a positive
control as well ([STRAT-SC-11](../strategies/STRAT-SC-11-oracle-needs-a-positive-control.md)).

---

## Pitfalls

- **Never store to an HBM ref the TensorCore way.** `o_ref[...] = v` raises `Swap does not support
  storing to HBM` and hides whatever else is wrong. Stage through scratch and `sync_copy` out.
- **Some primitives need `pltpu.CompilerParams(needs_layout_passes=False)`.** If an error names a
  flag, retry with the flag before recording an absence. Then expect the strict `(num_lanes,)` check
  on every load and store.
- **`subcore_barrier` can deadlock at higher subcore counts.** With no error and an empty log, the
  device hangs until the job's deadline. Run anything containing `subcore_barrier`
  (`sc_subcore_barrier_minimal`, which is **destructive**, and `sc_multi_subcore_reduction`)
  **alone, never beside work whose results you want, under a wall-clock timeout** (for example
  `timeout -k 30 <seconds>`), with unbuffered output (`python -u`) and a `> starting <arm>` line
  printed before each call so a hang names its arm. hypothesis: the barrier completes at small
  subcore counts and hangs above some threshold, and the threshold is not bisected; until
  `sc_subcore_barrier_minimal` says otherwise on the target, assume a coordinated kernel is capped
  at a few subcores and that the uncoordinated scaling curve cannot be spent on it.
- **Per-subcore VMEM does not pool,** and the allocator budget is shared across the core
  ([../tpu_hardware_facts.md](../tpu_hardware_facts.md) §10).
- **SMEM is tiny and 32-bit-only.** Metadata that fits TensorCore SMEM may not fit. Pack fields
  into int32 bit fields to halve the number of index arrays.
- **Hard-coded lane counts** make a single-generation kernel. Read `num_lanes`.
- **A differential probe where every arm fails identically** has a scaffolding bug, not a result.

---

## Open questions

Each line is an open question with the probe that answers it.

- **Coordinated fan-out.** Does a cross-subcore prefix sum or reduction from `fetch_and_add` and
  `subcore_barrier` work, and how far does it scale? Has `fetch_and_add` run on hardware at all?
  `sc_multi_subcore_reduction` (run alone, under a timeout).
- **The barrier ceiling.** At what subcore count does `subcore_barrier` stop completing, and why?
  `sc_subcore_barrier_minimal` (destructive; bisect the counts between the last success and the
  first hang).
- **`store_compressed` destination rule.** Is the destination memory space (VMEM scratch versus
  kernel output ref) the variable that separates lowering from `Mask is not supported`? Does
  `load_expanded` follow the same rule? Extend `sc_primitive_lowering` with an arm that varies only
  the destination.
- **Memory-bound overlap collapse.** Why does SC work hide poorly behind a memory-bound neighbour
  when the combined step is far from the bandwidth roofline? Repeat
  `sc_tc_overlap_vs_neighbour_kind` with a memory-bound neighbour *longer* than the SC kernel to
  remove the duration-ratio confound.
- **XLA gather path selection.** Which operand property selects XLA's two gather implementations?
  `sc_xla_gather_cliff_bracket` at more table sizes and row widths, plus a read of the emitted HLO.
- **Scalar+vector MPMD.** How does a DMA inside a two-mesh kernel name its core? Can such a kernel
  hold VMEM scratch? `sc_mpmd_dma_core_id` (with the shared core axis name on every arm, including
  remote-copy spellings that take a `device_id`), `sc_mpmd_vmem_scratch`.
- **Fanned-out algorithms.** Do top-k and MoE routing gain from fanning out across subcores once
  coordination works? `sc_topk_vs_lax_topk`, `sc_moe_route_vs_xla` at more than one subcore.
- **Routing on a real layer.** Does the routing offload hold against a real MoE layer's routing
  prologue rather than a stand-in? `sc_moe_route_offload_sweep` with production shapes.
- **Managed versus hand-rolled pipelining on SC.** Does `pltpu.emit_pipeline` beat a hand-rolled
  double buffer inside a SparseCore kernel? No registry probe yet.
- **`core_map` versus `pl.kernel` dispatch.** Does calling `core_map` directly instead of going
  through `pl.kernel`'s `mpmd_map` dispatch change anything? No registry probe yet.
- **Bank count.** `SparseCoreInfo` does not declare the VMEM bank count. The planned
  `sc_bank_stride_sweep` will measure it ([../tpu_hardware_facts.md](../tpu_hardware_facts.md) §10).
- **Other generations.** The per-generation `SparseCoreInfo` table is jax's model of the hardware,
  not a datasheet. Every probe answer is per chip, so re-run on each generation targeted.
