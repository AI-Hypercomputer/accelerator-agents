<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# MaxText Benchmarking & Tuning Guide

- **Source URL**: `https://maxtext.readthedocs.io/en/latest/guides/optimization/benchmark_and_performance.html`
- **Status**: Comprehensive guide on establishing benchmarks, metrics calculation, and performance tuning configurations (remat policy, precision, custom kernels, and communication offloading).

---

## 1. Benchmark Setup
* **Synthetic Data & Repeated Batches**: To isolate training speed from data input pipeline bottlenecks, configure:
  * `dataset_type="synthetic"`
  * `reuse_example_batch=1`
* **Arithmetic Intensity (AI) Analysis**:
  * Measured as `Arithmetic Intensity = FLOPs / Bytes` (ratio of compute FLOPs to memory or communication transfer in bytes).
  * Helps choose the optimal sharding configuration (MXU-bound vs. memory/communication-bound).

---

## 2. Core Metrics
* **Model FLOPs Utilization (MFU)**:
  $$\text{MFU} = \frac{\text{flops\_train\_step}}{\text{step\_time} \times \text{peak HW FLOPS}}$$
* **Throughput**:
  $$\text{Throughput} = \frac{\text{global tokens}}{\text{step\_time} \times \text{number of devices}}$$

---

## 3. Rematerialization (Remat) Policy Tuning
* **Goal**: Balances memory savings (HBM footprint) against computational overhead (extra FLOPs for recompute during the backward pass).
* **Presets**: Range from `minimal_with_context` (highest HBM usage, fastest) to `full` (lowest HBM footprint, slowest due to full recomputation).
  * Key options: `save_dot_except_mlp`, `save_qkv_proj`, `qkv_proj_offloaded`, `minimal_offloaded`.
* **Granular Custom Policy**: Users can set `remat_policy="custom"` and define layer settings:
  * `offload`: Offloads to CPU host memory.
  * `device`: Retains in TPU device memory.
  * `remat`: Performs recomputation.

---

## 4. Low Precision Training & Quantization
* **QWIX Quantization**: Enabled via `use_qwix_quantization=true`.
* **Recipes**:
  * **TPU v6e and earlier**: `"int8"`
  * **TPU v7x and later**: `"fp8_full"`
  * **NVIDIA GPUs**: `"fp8_gpu"`
  * **AMD GPUs**: `"nanoo_fp8"`

---

## 5. Pallas Kernel Tuning
* Custom Pallas kernels can be optimized using the `tune-jax` library.
* **Usage**: Annotate kernel functions with `@tune(hyperparams={'block_q': [256, 512, 1024], 'block_k': [8, 16]})` to sweep the search space and choose the fastest block configurations automatically.

---

## 6. Asynchronous Collective Offloading & Overlaps
* **Offloading to SparseCore (v7x)**: Offloads collective communication off the TensorCores. Enabled via:
  * `ENABLE_SPARSECORE_OFFLOADING_FOR_RS_AG_AR`
  * `ENABLE_SPARSECORE_OFFLOADING_FOR_REDUCE_SCATTER`
  * `ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_GATHER`
  * `ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_REDUCE`
* **Continuation Fusion (v5p & v6e)**: Overlaps collectives with compute on TensorCore. Enabled via:
  * `CF_FOR_ALL_GATHER`
  * `CF_FOR_ALL_REDUCE`
* **Scoped VMEM Limit**: Configured using `xla_tpu_scoped_vmem_limit_kib` via `LIBTPU_INIT_ARGS` (up to 64M on v5e, 128M on v6e, and 64M on v7x).
