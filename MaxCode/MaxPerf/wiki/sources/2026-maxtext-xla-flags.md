<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# MaxText XLA Flags Library Reference

- **Source File**: `raw/sources/xla_flags_library.py` (extracted from `google/maxtext` repository)
- **Status**: Curated set of performance-relevant XLA/TPU compiler flags for LLM optimization on TPU v5e/v6e/v7x.

---

## 1. VMEM Allocation Limits
* **Dense Models**: `--xla_tpu_scoped_vmem_limit_kib=98304` (96 MiB)
* **MoE Models**: `--xla_tpu_scoped_vmem_limit_kib=81920` (80 MiB)
* **Rationale**: Sets the VMEM limit for the current HLO instruction, leaving the remaining VMEM space available for prefetching downstream operands.

---

## 2. Continuation Fusion (CF) for Collectives
Enables overlapping/parallelizing compute workloads with collective communication operations (All-Gather and All-Reduce) on TPU TensorCore:

* **All-Gather CF (`CF_FOR_ALL_GATHER`)**:
  ```bash
  --xla_tpu_enable_async_collective_fusion=true
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true
  --xla_tpu_enable_async_collective_fusion_multiple_steps=true
  --xla_tpu_overlap_compute_collective_tc=true
  --xla_enable_async_all_gather=true
  ```
* **All-Reduce CF (`CF_FOR_ALL_REDUCE`)**:
  ```bash
  --xla_tpu_enable_async_collective_fusion=true
  --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true
  --xla_tpu_enable_async_collective_fusion_multiple_steps=true
  --xla_tpu_overlap_compute_collective_tc=true
  --xla_enable_async_all_reduce=true
  ```
* **Both combined (`CF_FOR_ALL_REDUCE_AND_ALL_GATHER`)**:
  Combines both sets of async flags and overlap options.

---

## 3. SparseCore Collective Offloading
Offloads irregular sharding operations (Reduce-Scatter, All-Gather, All-Reduce) from TensorCore to SparseCore to minimize TensorCore stalls.

* **Base SparseCore Flags**:
  ```bash
  --xla_tpu_use_tc_device_shape_on_sc=true
  --xla_sc_enable_instruction_fusion=false
  --xla_sc_disjoint_spmem=false
  --xla_sc_disable_megacore_partitioning=true
  --2a886c8_chip_config_name=megachip_tccontrol
  ```
* **Offload RS, AG, and AR**:
  ```bash
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false
  --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=false
  --xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true
  --xla_tpu_enable_all_gather_offload_tracing=true
  --xla_tpu_enable_reduce_scatter_offload_tracing=true
  --xla_tpu_enable_all_reduce_offload_tracing=true
  ```
* **Selective Offloading**:
  Can offload individual collectives by setting the corresponding TC-fusion flag to `false` and the SC-offload flag to `true` (e.g., `ENABLE_SPARSECORE_OFFLOADING_FOR_REDUCE_SCATTER` or `ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_REDUCE`).

---

## 4. Collective Layout & Fusion Optimizations
* **Memory Layout for All-Reduce / Reduce-Scatter**:
  ```bash
  --xla_tpu_use_minor_sharding_for_major_trivial_input=true
  --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1
  --xla_tpu_assign_all_reduce_scatter_layout=true
  ```
* **Reduce-Scatter Fusion**:
  Fuses All-Reduce and Dynamic-Slice operations to create implicit Reduce-Scatter calls:
  ```bash
  --xla_tpu_use_minor_sharding_for_major_trivial_input=true
  --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1
  ```
* **Data Parallel Pipelining & DCN Overlap**:
  ```bash
  --xla_tpu_enable_data_parallel_all_reduce_opt=true
  --xla_tpu_data_parallel_opt_different_sized_ops=true
  ```

---

## 5. Host Offloading & Scheduling Optimization
Optimizes pipeline execution when parameters/activations are offloaded to host memory:
```bash
--xla_tpu_enable_all_experimental_scheduler_features=true
--xla_tpu_enable_scheduler_memory_pressure_tracking=true
--xla_tpu_host_transfer_overlap_limit=24
--xla_tpu_aggressive_opt_barrier_removal=ENABLED
--xla_lhs_prioritize_async_depth_over_stall=ENABLED
--xla_tpu_enable_ag_backward_pipelining=true
--xla_should_allow_loop_variant_parameter_in_chain=ENABLED
--xla_should_add_loop_invariant_op_in_chain=ENABLED
--xla_max_concurrent_host_send_recv=100
--xla_tpu_scheduler_percent_shared_memory_limit=100
--xla_latency_hiding_scheduler_rerun=2
```
* **Large Host Offload Chunking**:
  `--xla_tpu_iova_dma_chunk_size_bytes=16777216` (Breaks DMA transfers to/from host into 16 MiB chunks).

---

## 6. Miscellaneous Compiler Controls
* **Disable Bundle-Aware Cost Model**:
  `--xla_tpu_use_bundle_aware_cost_model_for_fusions=false`
  *Note: Disabling cost model prevents backward pass fusions from slowing down (up to 3x speedup on affected fusions).*
* **Enhanced Launch Barrier**:
  `--xla_tpu_use_enhanced_launch_barrier=true` (For error propagation and out-of-order execution detection on Pathways).
* **Disable Windowed Einsum (Collective Matmul)**:
  `--xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000`
* **Disable Megacore Fusion for AGs**:
  `--xla_tpu_megacore_fusion_allow_ags=false`
* **Enable Async Collective Permute**:
  `--xla_enable_async_collective_permute=true`
