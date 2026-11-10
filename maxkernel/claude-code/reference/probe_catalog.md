---
name: probe-catalog
description: >-
  When to run a hardware probe, how to run one with tools/run_probe.py, how to read its verdict, and
  the catalog of every registered probe by module with the question it answers and the design file
  that uses the answer. Read it when a plan needs a fact that no reference file states.
---

# Probe catalog

A probe answers one question about the hardware or the toolchain actually in front of you: does
this lower, what does this primitive cost, what does this API do on this chip. Almost everything
worth knowing here is true of a *toolchain and chip*, not of TPUs in general, so the answer comes
from the target, not from memory. Probes live in `probes/checks/<module>.py`; agents run them
through `tools/run_probe.py`. How the planner uses the answers: [`kernel_design_method.md`](kernel_design_method.md)
§9 ("Facts needed").

A probe is not the benchmark. Its numbers answer a design question; they are never quoted as the
kernel's result.

---

## When to probe

Two questions, in order:

1. **Is it a source-read or structural fact in [`design/`](design/) or
   [`tpu_hardware_facts.md`](tpu_hardware_facts.md)?** A legality rule read from source, a formula,
   a mechanism: use it, cite the file and section, and move on. Re-establishing it buys nothing.
   These files contain no measured numbers; where they need one, they name the probe.
2. **Otherwise, would being wrong change a decision?** If yes, run the probe, or reuse its cached
   result (a cache hit is free). If no, write the fact down as `assumed` and carry on. Not every
   unknown is worth a TPU job.

Both halves matter and fail in opposite directions. Skip the first and the loop re-measures what is
already known until probing feels expensive and gets skipped. Skip the second and a design turns on
a guess one job would have settled. A probe earns its slot when you can name the decision it
settles and what you would do under each answer; the "Gaps" lists at the end of the design files
are a backlog, not a checklist. Stop when the remaining uncertainty would not change which option
wins.

Budget: at most four uncached probes per plan.

---

## How to run

List what exists (local, no TPU; imports the registry and prints it):

```bash
{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/run_probe.py --list
{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/run_probe.py --list --group partitioning
```

Run one or more probes on the TPU (submitted through `tools/tpu_client.py`):

```bash
{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/run_probe.py \
    --pick vmem_limit_bytes_raises_ceiling --pick dispatch_floor --run_dir <run_dir>
```

- `--pick` is an **exact** name match. Check `--list` first; an unknown name is an error.
- Every run also includes `software_versions` and `device_inventory`, so each output is stamped
  with the live TPU type and toolchain.
- Output is saved to `<run_dir>/probes/<name>.txt` and cached under
  `workspace/_probe_cache/<tpu_type>/<versions>/<name>.txt`, keyed by the TPU type and toolchain
  stamp read from that same output. A different chip or toolchain is a cache miss and the probe
  re-runs. Cached results are returned without touching the TPU.
- **Destructive probes** (marked below) can halt the Mosaic core and break later jobs on the same TPU
  VM. `run_probe.py` refuses them unless `--allow-destructive` is passed. **Agents never pass
  `--allow-destructive`.** Take those answers from the design files, or design around them.

---

## Reading the output

**Read the `probe suite: X of N` line first.** `0 of N` means nothing ran; the run measured
nothing, whatever colour it came back.

Each probe prints three lines: group and name, the question, and the verdict with elapsed time and
the detail string.

| verdict | meaning |
| --- | --- |
| `PASS` | the probe ran; its answer is in the detail string |
| `FAIL` | the probe ran, and the thing it checks is not true here |
| `UNSUPPORTED` | the feature does not exist on this toolchain or chip; a different finding from `FAIL` |
| `INVALID` | the probe does not trust its own measurement (wrong answer, under the dispatch floor, spread too wide) and has **withheld the number** |
| `SKIPPED` | destructive, and not opted into |

- **The detail string is the answer.** It carries the numbers and the arm each ratio is against.
  The verdict alone says only whether the probe ran.
- **`INVALID` has no number to quote.** Do not reconstruct one from the log.
- **`UNSUPPORTED` is not `FAIL`.** "This toolchain does not have it" and "it exists and misbehaves"
  lead to different designs.
- **`PASS` means it ran, not that it discriminated.** Compare the reported spread with the gap
  between arms before believing an ordering.
- **A number from a probe is quotable only with its output file**: cite
  `<run_dir>/probes/<name>.txt` or the cache path beside it. A probe output from another chip or
  toolchain is not a fact about this one.
- If an error in the detail string names a compiler flag, the absence is not established until the
  flag has been tried.

---

## Catalog

Rows: probe name | question | design file it informs. Paths are relative to `reference/`.

### `environment` → [`tpu_hardware_facts.md`](tpu_hardware_facts.md)

| probe | question | informs |
| --- | --- | --- |
| `software_versions` | Which jax, jaxlib and libtpu is this image on? | [`tpu_hardware_facts.md`](tpu_hardware_facts.md) |
| `device_inventory` | What devices does jax see, and what does it call this chip? | [`tpu_hardware_facts.md`](tpu_hardware_facts.md) |
| `device_memory_stats` | How much HBM does one device report, and how much is already in use? | [`tpu_hardware_facts.md`](tpu_hardware_facts.md) |
| `pallas_hello_world` | Does the simplest possible `pallas_call` compile and produce the right answer? | [`tpu_hardware_facts.md`](tpu_hardware_facts.md) |
| `pltpu_symbol_surface` | Which `pltpu` / `pl` symbols that the reference files name actually exist here? | [`tpu_hardware_facts.md`](tpu_hardware_facts.md) |
| `tpu_info_reported_constants` | What per-core hardware constants does `pltpu.get_tpu_info()` report for this chip? | [`tpu_hardware_facts.md`](tpu_hardware_facts.md) |
| `pltpu_full_surface` | What is the complete public symbol list of `jax.experimental.pallas.tpu` here? | [`tpu_hardware_facts.md`](tpu_hardware_facts.md) |
| `compiler_params_spelling` | Is the compiler-params argument spelled `TPUCompilerParams` or `CompilerParams` here? | [`tpu_hardware_facts.md`](tpu_hardware_facts.md) |

### `hardware` → [`tpu_hardware_facts.md`](tpu_hardware_facts.md)

| probe | question | informs |
| --- | --- | --- |
| `tpu_info_scaling_denominator` | Are `get_tpu_info`'s capacities and rates per chip or per TensorCore? | [`tpu_hardware_facts.md`](tpu_hardware_facts.md) |
| `native_matmul_dtype_matrix` | Which operand dtype pairs does the MXU multiply natively, without a cast? | [`tpu_hardware_facts.md`](tpu_hardware_facts.md) |
| `sublane_tiling_is_a_formula` | Is `get_sublane_tiling` a measurement of the chip or a computed heuristic? | [`tpu_hardware_facts.md`](tpu_hardware_facts.md) |
| `f32_default_is_one_bf16_pass` | What precision does `precision=DEFAULT` on f32 operands actually compute in? | [`tpu_hardware_facts.md`](tpu_hardware_facts.md) |
| `vmem_limit_bytes_raises_ceiling` | Does `CompilerParams(vmem_limit_bytes=...)` actually move the scoped limit? | [`tpu_hardware_facts.md`](tpu_hardware_facts.md) |
| `buffer_count_cap_by_position` | How deep can `pl.Buffered` go, and does the answer depend on where it sits? | [`tpu_hardware_facts.md`](tpu_hardware_facts.md) |
| `block_sublane_legality_by_dtype` | Is the divisible-by-8 block rule the same for f32, bf16 and int8? | [`tpu_hardware_facts.md`](tpu_hardware_facts.md) |
| `traced_dma_offset_by_dtype` | May a DMA offset be a runtime value, and does the dtype change the answer? | [`tpu_hardware_facts.md`](tpu_hardware_facts.md) |
| `traced_dma_length` | May the number of rows a DMA transfers be a runtime value? | [`tpu_hardware_facts.md`](tpu_hardware_facts.md) |
| `dynamic_index_lane_versus_sublane` | Is a runtime index allowed on the lane axis, the sublane axis, or both? | [`tpu_hardware_facts.md`](tpu_hardware_facts.md) |
| `zero_size_dma_needs_no_wait` **(destructive)** | Is a zero-size DMA a true no-op, or does it still owe a wait? | [`tpu_hardware_facts.md`](tpu_hardware_facts.md) |
| `undrained_semaphore_halts_core` **(destructive)** | What does leaving a real DMA un-waited actually do? | [`tpu_hardware_facts.md`](tpu_hardware_facts.md) |
| `xla_reference_vs_f64_oracle` | Is an f32 XLA reference more accurate than the Pallas kernel it is judging? | [`tpu_hardware_facts.md`](tpu_hardware_facts.md) |
| `input_output_alias_in_place` | When a pallas_call aliases an input to its output, does the caller pay a copy of the whole operand, with and without buffer donation? | contract row T5 in [`kernel_design_method.md`](kernel_design_method.md) §8 |

### `lowering` → [`tpu_hardware_facts.md`](tpu_hardware_facts.md)

| probe | question | informs |
| --- | --- | --- |
| `integer_ops_int32` | Which integer operations lower on an int32 block? | [`tpu_hardware_facts.md`](tpu_hardware_facts.md) |
| `dot_dtype_matrix` | Which operand dtype combinations does an in-kernel `jnp.dot` accept? | [`tpu_hardware_facts.md`](tpu_hardware_facts.md) |

### `fusion` → [`design/fusion_boundary.md`](design/fusion_boundary.md)

| probe | question | informs |
| --- | --- | --- |
| `elementwise_chain_xla_vs_pallas` | Does XLA already fuse a chain of elementwise ops as well as one Pallas kernel would? | [`design/fusion_boundary.md`](design/fusion_boundary.md) |
| `matmul_epilogue_fusion` | Is a matmul with its bias-and-activation epilogue fused into one kernel faster than XLA's version? | [`design/fusion_boundary.md`](design/fusion_boundary.md) |
| `intermediate_hbm_roundtrip` | What does materializing an intermediate to HBM actually cost, at a known size? | [`design/fusion_boundary.md`](design/fusion_boundary.md) |
| `operand_count_vmem_pressure` | How does the largest workable block shrink as a kernel takes on more operands? | [`design/fusion_boundary.md`](design/fusion_boundary.md) |
| `fusion_enables_dma_compute_overlap` | Does fusing the producer in let a hand-issued accumulator DMA hide behind compute? | [`design/fusion_boundary.md`](design/fusion_boundary.md) |
| `gmm_tgmm_grid_axis_conflict` | What does fusing a matmul and its transposed-lhs twin into one grid cost? | [`design/fusion_boundary.md`](design/fusion_boundary.md) |

### `partitioning` → [`design/partitioning.md`](design/partitioning.md)

| probe | question | informs |
| --- | --- | --- |
| `vmem_scratch_bisect` | How large a VMEM scratch can a kernel actually allocate and use? | [`design/partitioning.md`](design/partitioning.md) |
| `vmem_bytes_per_element_by_dtype` | Does a narrower dtype buy proportionally more VMEM, or is there a floor? | [`design/partitioning.md`](design/partitioning.md) |
| `smem_scratch_bisect` | How many int32 scalars fit in an SMEM scratch? | [`design/partitioning.md`](design/partitioning.md) |
| `dispatch_floor` | What does one `pallas_call` cost before it does any work, and what does one extra grid step cost? | [`design/partitioning.md`](design/partitioning.md) |
| `matmul_block_size_sweep` | Which block shape wins for a fixed matmul, and how much is between best and worst? | [`design/partitioning.md`](design/partitioning.md) |
| `grid_traversal_order` | Does the order the grid is walked change wall time, holding the partition fixed? | [`design/partitioning.md`](design/partitioning.md) |
| `dimension_semantics_effect` | Does labelling grid axes `parallel` instead of `arbitrary` change anything measurable? | [`design/partitioning.md`](design/partitioning.md) |
| `ragged_tail_cost` | What does a grid whose last block is partial cost, against one that divides evenly? | [`design/partitioning.md`](design/partitioning.md) |
| `pallas_call_intercept_chained_clock` | Is there a fixed per-`pallas_call` cost that survives batching, and is it device time? | [`design/partitioning.md`](design/partitioning.md) |

### `pipelining` → [`design/pipelining.md`](design/pipelining.md)

| probe | question | informs |
| --- | --- | --- |
| `hbm_bandwidth_arms` | What HBM bandwidth is achievable, measured three different ways? | [`design/pipelining.md`](design/pipelining.md) |
| `block_size_bandwidth_curve` | How large does a block have to be before a copy reaches full bandwidth? | [`design/pipelining.md`](design/pipelining.md) |
| `arithmetic_intensity_crossover` | At how many FLOPs per byte does a kernel stop being bandwidth-bound? | [`design/pipelining.md`](design/pipelining.md) |
| `pipeline_depth_and_warmup` | How many grid steps does a pipeline need before it reaches steady state? | [`design/pipelining.md`](design/pipelining.md) |
| `emit_pipeline_vs_blockspec_grid` | Does an inner `pltpu.emit_pipeline` beat the BlockSpec-managed grid at the same partition? | [`design/pipelining.md`](design/pipelining.md) |
| `manual_dma_vs_managed` | Does a hand-rolled `make_async_copy` beat the BlockSpec-managed transfer? | [`design/pipelining.md`](design/pipelining.md) |

### `layout` → [`design/layout_precision.md`](design/layout_precision.md)

| probe | question | informs |
| --- | --- | --- |
| `native_tile_granularity` | What does a block smaller than one native tile cost, per useful element? | [`design/layout_precision.md`](design/layout_precision.md) |
| `lane_vs_sublane_reduction` | Is reducing over the last axis cheaper or dearer than reducing over the second-minor? | [`design/layout_precision.md`](design/layout_precision.md) |
| `transpose_cost` | What does transposing an operand inside the kernel cost, against transposing it outside? | [`design/layout_precision.md`](design/layout_precision.md) |
| `mask_versus_pad` | For a shape that does not divide, is masking the tail cheaper than padding the array? | [`design/layout_precision.md`](design/layout_precision.md) |
| `dtype_conversion_cost` | What does converting between dtypes inside a kernel cost, per element? | [`design/layout_precision.md`](design/layout_precision.md) |
| `bitcast_and_subbyte` | Do `pltpu.bitcast` and sub-byte dtypes work in a kernel here? | [`design/layout_precision.md`](design/layout_precision.md) |
| `sublane_tiling_by_dtype` | Does the native sublane tile widen for narrow dtypes, and does it match what `get_tpu_info` reports? | [`design/layout_precision.md`](design/layout_precision.md) |

### `layout_arrange` → [`design/layout_precision.md`](design/layout_precision.md)

| probe | question | informs |
| --- | --- | --- |
| `expand_axis_leading_vs_sublane_vs_lane` | What does expanding a value cost along a leading, a sublane and a lane axis? (Its lane arm also changes spelling; pair it with `expand_blocked_vs_interleaved`.) | [`design/layout_precision.md`](design/layout_precision.md) |
| `lane_split_alignment` | Does the cost of a lane-axis split depend on whether it lands on a 128 multiple? | [`design/layout_precision.md`](design/layout_precision.md) |
| `concat_to_widen_narrow_matmul` | Do n narrow matmuls cost more than one matmul over their lane-concatenated operand? | [`design/layout_precision.md`](design/layout_precision.md) |
| `weight_stationary_lhs_concat` | Does concatenating the varying operands of matmuls that share a stationary one pay, and at which row counts? | [`design/layout_precision.md`](design/layout_precision.md) |
| `sliding_window_compact_vs_sublane` | Does a sliding window run differently with its windowed axis leading (sublane extent 1) than on sublanes? | [`design/layout_precision.md`](design/layout_precision.md) |
| `degenerate_transpose_via_masked_sum` | How does a masked reduction compare with a real transpose for a degenerate axis, across widths? | [`design/layout_precision.md`](design/layout_precision.md) |
| `elementwise_before_vs_after_expand` | Does the compiler already hoist elementwise work out of an expansion, or must the design do it? | [`design/layout_precision.md`](design/layout_precision.md) |
| `expand_before_vs_after_matmul` | Does expanding a matmul operand before the contraction cost in proportion to the extra MACs? | [`design/layout_precision.md`](design/layout_precision.md) |
| `mask_index_dtype_match` | Does a select cost more when its index/iota width does not match the data it selects, and which widths lower? | [`design/layout_precision.md`](design/layout_precision.md) |
| `expand_blocked_vs_interleaved` | Is the cost of an expansion set by the axis it runs along, or by whether it interleaves? | [`design/layout_precision.md`](design/layout_precision.md) |

### `compute_units` → [`design/compute_units.md`](design/compute_units.md)

| probe | question | informs |
| --- | --- | --- |
| `mxu_peak_by_dtype` | What matmul rate does XLA achieve per dtype? This is the denominator for every other rate. | [`design/compute_units.md`](design/compute_units.md) |
| `mxu_block_rows_cost` | Does a matmul pass over fewer than 128 rows cost less than a full 128-row pass? | [`design/compute_units.md`](design/compute_units.md) |
| `mxu_contraction_and_column_granularity` | Do sub-128 contraction depths and sub-128 column counts cost a full 128 as well? | [`design/compute_units.md`](design/compute_units.md) |
| `vpu_fma_rate` | What elementwise FMA rate does the VPU sustain on VMEM-resident data? | [`design/compute_units.md`](design/compute_units.md) |
| `transcendental_relative_cost` | How much more does exp / tanh / rsqrt cost than a multiply, per element? | [`design/compute_units.md`](design/compute_units.md) |
| `sparsecore_surface` | Is SparseCore reachable from Pallas on this stack at all? | [`design/compute_units.md`](design/compute_units.md) |

### `algorithm` → [`design/algorithm.md`](design/algorithm.md)

| probe | question | informs |
| --- | --- | --- |
| `flash_vs_materialized_attention` | What does the online-softmax restructuring of attention buy over materializing the score matrix? | [`design/algorithm.md`](design/algorithm.md) |
| `accumulator_precision` | What does accumulating a long reduction in bf16 instead of f32 cost, in time and in error? | [`design/algorithm.md`](design/algorithm.md) |
| `cumsum_alternatives` | Does `jnp.cumsum` lower here, and what does a triangular-matmul substitute cost? | [`design/algorithm.md`](design/algorithm.md) |
| `split_contraction_accuracy` | Does splitting a reduction into partial sums change the answer, and by how much? | [`design/algorithm.md`](design/algorithm.md) |

### `precomputation` → [`design/precomputation.md`](design/precomputation.md)

| probe | question | informs |
| --- | --- | --- |
| `scratch_persists_across_grid_steps` | Does a scratch written at grid step 0 still hold its value at step N? | [`design/precomputation.md`](design/precomputation.md) |
| `scalar_prefetch_drives_index_map` | Can a prefetched scalar array choose which block each grid step loads? | [`design/precomputation.md`](design/precomputation.md) |
| `smem_metadata_dtype_support` | Which dtypes may metadata use in SMEM, as a scratch and as an operand? | [`design/precomputation.md`](design/precomputation.md) |
| `grid_invariant_recompute_cost` | Does Mosaic hoist grid-invariant work out of the grid, or do you pay it every step? | [`design/precomputation.md`](design/precomputation.md) |
| `metadata_cse_across_pallas_calls` | Does XLA compute shared metadata once when several `pallas_call`s ask for it? | [`design/precomputation.md`](design/precomputation.md) |
| `smem_scalar_fill_cost` | What does building metadata in-kernel cost, per scalar written to SMEM? | [`design/precomputation.md`](design/precomputation.md) |
| `precompute_outside_versus_inside` | Is metadata cheaper built in XLA outside the kernel or by a scalar loop inside it? | [`design/precomputation.md`](design/precomputation.md) |
| `dynamic_grid_from_smem_skips_work` | Does a grid length read from SMEM actually skip the steps it excludes? | [`design/precomputation.md`](design/precomputation.md) |

### `sparsecore` → [`design/sparsecore.md`](design/sparsecore.md)

| probe | question | informs |
| --- | --- | --- |
| `sc_module_surface` | Which SparseCore Pallas surface does this jax expose, at what module path, and with what geometry? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_kernel_cannot_dry_run` | Can a SparseCore kernel be dry-run on CPU the way a TensorCore Pallas kernel can? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_vector_kernel_runs` | Does a minimal `VectorSubcoreMesh` kernel compile, run and compute the right answer? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_scalar_kernel_runs` | Does a `ScalarSubcoreMesh` kernel compile and run, and which memory spaces does it accept? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_dtype_support` | Which element dtypes survive a load-compute-store round trip in a SparseCore vector kernel? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_vmem_per_subcore` | How much VMEM can one vector subcore allocate before the compiler refuses, and does it match what `SparseCoreInfo` reports? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_primitive_lowering` | Which of the `tpu_sc` primitives actually lower to Mosaic and compute the right answer? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_pl_ops_inside_kernel` | Which ordinary `jnp`/`lax` constructs survive inside a SparseCore kernel body? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_subcore_scaling` | Does SparseCore work scale with subcore and core count, and by how much? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_gather_vs_xla_gather` | Is a SparseCore gather faster than the XLA gather it would replace, and by how much? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_tc_one_kernel_two_meshes` | Can one `pl.kernel` host a TensorCore mesh and a SparseCore mesh at once? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_two_sparsecore_meshes_one_kernel` | Can one `pl.kernel` host both SparseCore meshes (scalar subcore and vector subcore) at once? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_tc_independent_overlap` | When independent SparseCore and TensorCore work is issued in one jit, does the SC time hide behind the TC time? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_tc_dependent_handoff` | What does a SparseCore result cost to hand to the TensorCore in the same jit? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_vs_tc_streaming_bandwidth` | On a purely memory-bound stream, what fraction of HBM bandwidth does SparseCore reach against the TensorCore? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_vs_tc_transcendental` | On VPU-shaped elementwise math, how does SparseCore compare with the TensorCore VPU? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_scatter_add_vs_xla` | Is a SparseCore scatter-add faster than the XLA scatter-add it would replace? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_data_dependent_control_flow` | Can a SparseCore kernel do data-dependent work, such as a while loop whose trip count comes from the data? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_gather_row_width_scaling` | How does SparseCore gather throughput depend on the width of the row gathered, relative to the DMA granule `SparseCoreInfo` reports? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_smem_capacity` | How much SMEM can a SparseCore kernel allocate, and what dtypes does it hold? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_multi_subcore_reduction` **(destructive)** | Does a cross-subcore reduction built from `subcore_barrier` and `fetch_and_add` work, and how does it scale? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_pl_loop_init_carry` | Does `pl.loop` carry a value between iterations on the SparseCore, or does a running scalar have to live in SMEM? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_bf16_widen_and_pack` | How does a SparseCore kernel consume and produce bf16, on a unit whose arithmetic is f32? | [`design/sparsecore.md`](design/sparsecore.md) |

### `sparsecore_gaps` → [`design/sparsecore.md`](design/sparsecore.md)

| probe | question | informs |
| --- | --- | --- |
| `sc_wide_get_layout_passes` | Is the `(num_lanes,)` shape rule a property of the SparseCore, or only of `needs_layout_passes=False`? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_wide_get_without_reduction` | With `needs_layout_passes=True`, does a wide Get lower when the kernel is a copy or an elementwise map rather than a reduction? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_gather_index_count_scaling` | At a fixed row width, how does the SparseCore-versus-XLA gather comparison vary with the number of indices gathered? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_gather_width_curve_fixed_chunk` | Does the gather width curve keep its shape when chunk is held constant instead of shrinking with the row? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_tc_overlap_vs_neighbour_kind` | Does SparseCore work still hide behind the TensorCore when the TensorCore neighbour is memory-bound rather than compute-bound? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_scatter_add_fanout` | Does fanning a SparseCore scatter-add histogram across all vector subcores change how it compares with XLA's scatter-add? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_addupdate_variants` | Do `plsc.addupdate` and `plsc.addupdate_compressed` lower, and do they share any masking restriction with `store_compressed`? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_mpmd_vmem_scratch` | Can a scalar+vector MPMD SparseCore kernel allocate VMEM scratch, or is it limited to the SMEM the scalar subcore can also reach? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_subcore_barrier_minimal` **(destructive)** | Does a bare `subcore_barrier()` complete, and does that depend on the subcore count? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_wide_get_vs_lane_loop` | Where a wide Get lowers, is it faster than the lane-group loop, or only shorter? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_xla_gather_cliff_bracket` | At what index count does the XLA gather's cost step sharply, and does that threshold move when the table size changes? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_mpmd_dma_core_id` | How does a DMA inside a two-mesh MPMD SparseCore kernel name its core? | [`design/sparsecore.md`](design/sparsecore.md) |

### `sparsecore_kernels` → [`design/sparsecore.md`](design/sparsecore.md)

| probe | question | informs |
| --- | --- | --- |
| `sc_topk_correctness` | Does a SparseCore radix-select top-k return the same set as `jax.lax.top_k`? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_topk_vs_lax_topk` | How does SparseCore radix-select top-k compare with `jax.lax.top_k` on the TensorCore? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_topk_frees_the_mxu` | Does moving top-k to the SparseCore actually free the MXU, or just move the time? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_moe_route_correctness` | Does the SparseCore routing kernel produce the same counts and stable permutation as argsort? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_moe_route_vs_xla` | How does SparseCore MoE routing compare with argsort on the TensorCore? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_moe_route_frees_the_mxu` | Does moving MoE routing to the SparseCore free the MXU for the expert matmul? | [`design/sparsecore.md`](design/sparsecore.md) |
| `sc_moe_route_offload_sweep` | Does the routing-offload comparison hold across a different routing shape and a different neighbour? | [`design/sparsecore.md`](design/sparsecore.md) |

---

## Destructive probes

`zero_size_dma_needs_no_wait`, `undrained_semaphore_halts_core`, `sc_multi_subcore_reduction` and
`sc_subcore_barrier_minimal` are registered with `destructive=True`: they can halt or deadlock the
core, and a halted core poisons every later job on the same TPU VM. `run_probe.py` refuses them
without `--allow-destructive`, and agents never pass it. When a design question depends on one
(for example, contract row T3 on zero-length transfers), take the answer from the design file, or
choose the design that does not depend on it (predicate by branch rather than by size; keep
cross-subcore coordination out of the kernel).

---

## Checks that are answered without a probe

Some rules in `strategies/` and `design/` are about whether a construct lowers, or about a parameter
that only matters inside one kernel. They have no probe on purpose: the real kernel answers them
more directly than a generic probe would.

| question | how it is answered | where |
| --- | --- | --- |
| Is `emit_pipeline(scratches=)` honoured? | Not needed: always pass scratch buffers by closure capture, which works either way | [STRAT-SC-04](strategies/STRAT-SC-04-emit-pipeline-scratches-noop.md) |
| Do `pl.when` around `emit_pipeline` and a traced grid length lower on SparseCore? | The compilation test of the kernel itself; on failure, the error table in [`design/sparsecore.md`](design/sparsecore.md) | [STRAT-MLA-08](strategies/STRAT-MLA-08-sparsecore-data-dependent-skip-lowers.md) |
| Which memory spaces allow a rank-0 load on SparseCore? | Same: the compilation test, then the error table | [STRAT-MLA-09](strategies/STRAT-MLA-09-sparsecore-rank0-smem-xor.md) |
| Does a traced-extent SparseCore DMA need a 1-D buffer? | Same: the compilation test, then the error table | [STRAT-MLA-12](strategies/STRAT-MLA-12-sparsecore-traced-dma-needs-a-1d-ring.md) |
| What is the SparseCore bank count? | Not needed to apply the rule: time your own kernel with its lane-private stride and with stride + 1 at fixed work | [STRAT-SC-06](strategies/STRAT-SC-06-pad-scatter-stride-off-bank-count.md) |
