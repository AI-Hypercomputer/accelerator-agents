---
name: tpu-memory-overlapping
description: >-
  Detailed reference for diagnosing and resolving memory bottlenecks by overlapping
  communication and compute in JAX/Pallas. Includes roofline calculations, access efficiency under tiled layouts, cost estimation,
  pipelining, bubbles, and distributed ICI configurations.
---

# Overlapping Communication and Compute

This skill covers how to diagnose when a JAX/Pallas TPU kernel is memory-bound and provides detailed strategies to overlap memory transfers with compute core execution.

---

## 1. Diagnosing Memory Bounds

### A. Roofline Analysis (Theoretical)
Before writing or modifying a kernel, compute its arithmetic intensity (FLOPs / bytes transferred) and compare it against the target TPU hardware limitations.

#### Hardware Limits Reference
Retrieve peak performance metrics from the target specifications or [tpu_info.py](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/tpu_info.py).

*   **TPU v5e (Viperlite)**:
    *   Peak BF16 performance: 197 TFLOP/s
    *   HBM Bandwidth: 819 GB/s
    *   $\text{Compute-Bound Threshold} = 197 \text{ TFLOP/s} / 819 \text{ GB/s} \approx 240 \text{ FLOPs/byte}$
    *   *Interpretation*: If the kernel's arithmetic intensity is less than 240, it is theoretically memory-bound on TPU v5e.
*   **TPU v5p (Megacore)**:
    *   Peak BF16 performance: 459 TFLOP/s
    *   HBM Bandwidth: 2765 GB/s (HBM3)
    *   $\text{Compute-Bound Threshold} = 459 \text{ TFLOP/s} / 2765 \text{ GB/s} \approx 166 \text{ FLOPs/byte}$
    *   *Interpretation*: If the kernel's arithmetic intensity is less than 166, it is theoretically memory-bound on TPU v5p.
*   **TPU v6e (Trillium)**:
    *   Peak BF16 performance: 918 TFLOP/s
    *   HBM Bandwidth: 1638 GB/s
    *   $\text{Compute-Bound Threshold} = 918 \text{ TFLOP/s} / 1638 \text{ GB/s} \approx 560 \text{ FLOPs/byte}$
    *   *Interpretation*: If the kernel's arithmetic intensity is less than 560, it is theoretically memory-bound on TPU v6e.
*   **TPU v7x (Ironwood / Ghostfish)**:
    *   Peak BF16 performance: 2307 TFLOP/s per chip (FP8: 4614 TFLOP/s)
    *   HBM Bandwidth: 7380 GB/s per chip (192 GiB capacity)
    *   $\text{Compute-Bound Threshold} = 2307 \text{ TFLOP/s} / 7380 \text{ GB/s} \approx 312 \text{ FLOPs/byte}$
    *   *Interpretation*: Because compute grows significantly faster than interconnect, leverage FP8 precision and favor larger batch sizes/sequence lengths to keep arithmetic intensity high. Manage dual-chiplet collective operations across the 1200 GB/s 3D Torus ICI.

### B. Automated Cost Estimation
Use JAX's internal Pallas cost estimator to calculate theoretical FLOPs and bytes accessed without manual calculation:
```python
from jax.experimental.pallas import cost_estimate as pl_cost


# Define your kernel reference function
def matmul_ref(a, b):
  return a @ b


cost = pl_cost.estimate_cost(
  matmul_ref,
  jax.ShapeDtypeStruct((4000, 8000), jnp.bfloat16),
  jax.ShapeDtypeStruct((8000, 9000), jnp.bfloat16),
)
print("FLOPs:", cost.flops)
print("Memory Access (Bytes):", cost.bytes_accessed)
```

### C. Empirical Trace Analysis
Open an **Xprof** trace and inspect the **`Tensor Core Sync Flag`** track. If the execution timeline is dominated by **`SyncWait`** blocks, the vector and matrix cores are idling while waiting for DMAs to complete, confirming the kernel is memory-bound.

### D. Access Efficiency Under Tiled Layouts
TPU memory (HBM and VMEM) stores the last two dims of an array in tiles of 8 rows × 128 columns for 32-bit data (narrower dtypes may use more rows; check the HLO layout, e.g. `T(8,128)(2,1)`, or `tpu_info.get_sublane_tiling`). Data moves most efficiently in whole tiles, so an access that uses only part of each tile wastes bandwidth.

*   **Check**: for each array, compare the bytes the kernel needs with the bytes in the tiles it touches. Dense, contiguous blocks are ~1×, so nothing to do. If touched is much larger, use it in the roofline (Section 1.A) and consider a layout change.
*   **Common causes of waste**:
    *   *Rows picked by an index* (gather, embedding lookup, page table, permutation): uses 1 row of each multi-row tile.
    *   *Last dim smaller than 128*: each row is padded to 128 columns.
    *   *Column or strided access*: uses 1 column, or a few rows, of each tile.
    *   *Partial tiles*: ragged edges, or a block height that is not a multiple of the tile height.
*   **Fixes**:
    *   *Make each access cover whole tiles* by keeping the indexed dim out of the last two. E.g. store a row-indexed `(N, H)` f32 array as `(N, 8, H // 8)` (needs `H // 8` divisible by 128).
    *   *Have the producer write that layout*: converting later costs an extra HBM pass.
    *   *Gather inside the kernel*: scalar-prefetch the indices and DMA the rows directly, rather than letting XLA build a reordered copy first.
    *   *Sort once* if the data is read many times.
    *   *Pack narrow rows* into the 128 columns, e.g. `(N, 64)` → `(N // 2, 128)`.
*   **Many small DMAs are slow** because each has a fixed issue cost: achieved bandwidth ≈ bytes per DMA ÷ issue time (capped at peak). Measure that cost on the target instead of assuming it.

---

## 2. Pipelining & HBM-VMEM Overlap Strategies

If the kernel is memory-bound, use these programming techniques to force memory transfers to happen concurrently with compute.

### A. Scaling Block Sizes
*   **Concept**: For operations like matrix multiplication, the compute requirements scale faster than the memory requirements ($O(N^3)$ vs $O(N^2)$). Increasing the block sizes (tile dimensions) increases arithmetic intensity.
*   **Action**: Gradually increase the `BlockSpec` shapes for inputs/outputs.
*   **Limitation**: Eventually you will hit VMEM limits (triggering OOM compiler errors) or register pressure limits (triggering register spills).

### B. Lower Precision
*   **Action**: Use lower bitwidth types (e.g., `bf16` or `float8` instead of `f32`).
*   **Benefit**: This reduces the total bytes transferred by half (or more), easing memory bandwidth congestion.
*   **Limitation**: Be extremely careful with potential drops in model output quality and numerical accuracy. Low-precision casting (especially quantization to INT8/FP8) can lead to output divergence; correctness and tolerance thresholds must be monitored closely.

### C. VMEM Fusion
*   **Action**: Perform all elementwise operations (activation functions, log, exp, masks) and reductions (softmax normalizations) directly on VMEM buffers while the data is in flight, rather than saving to HBM and reading it back.
*   *Example*: **FlashAttention** performs online softmax scaling iteratively to compute the full attention block entirely inside VMEM.

### D. Exploiting Consecutive Iteration Fetch Skipping
Pallas' pipeline emitter will automatically skip a DMA read/write instruction if the indices returned by the `BlockSpec`'s `index_map` are identical to the previous iteration.
*   **Action**: Structure your loops so that the output block index remains static throughout the inner loop.
*   *Example*: In matrix multiplication ($C = A \times B$), map the contracting dimension (K-axis) as the inner loop grid dimension. This keeps the output tile $C[i, j]$ constant, meaning the emitter only writes the output tile to HBM *once* after the inner loop finishes, rather than every step.

### E. Manual Pipelining with `pltpu.emit_pipeline`
If the default automated pipelining is insufficient, implement manual prefetching:
*   Use `pltpu.emit_pipeline` with the following callbacks:
    *   `prefetch`: Executed at the end of a pipeline step to begin prefetching inputs for the *next* block.
    *   `postyeet`: Executed at the beginning of a pipeline step to wait for DMAs triggered in the previous iteration to finish.

### F. Optimizing for Sparse Kernels
*   Sparse layouts cannot be easily prefetched because the next physical page index is data-dependent.
*   **Action**: Perform index pre-processing on the host or in a separate scalar kernel to generate the page lists, allowing the main kernel to prefetch blocks smoothly.

---

## 3. Resolving Pipeline Bubbles

In any pipelined kernel, there is a "bubble" at the start (waiting for the first block to load) and at the end (waiting for the final block to write).

*   **Hazard**: If the total input size is small or the pipeline has too many stages, these bubbles will constitute a large percentage of the total runtime, reducing performance.
*   **Remediation**:
    *   Ensure the total input volume is large enough to distribute the bubble overhead over many cycles.
    *   Avoid setting the pipeline stages count too high (on TPUs, double-buffering—2 stages—is usually optimal because DMA copy latency is typically lower than compute tile latency).

---

## 4. Distributed Pipelining (ICI / Multi-Chip)

Pallas does not automatically manage pipeline generations across the Inter-Chip Interconnect (ICI). You must implement and tune this yourself:

1.  **Use Bi-directional Communication**: ICI connections are bi-directional. Wasting one direction effectively halves the available bandwidth. Ensure you are sending and receiving data simultaneously.
2.  **Reduce Synchronization Points**: Increase the number of buffer slots and pipeline stages to allow communication to run multiple steps ahead of the active compute stage.
