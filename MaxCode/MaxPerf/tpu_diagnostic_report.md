<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# TPU Diagnostic Report: Qwen2.5-Coder-3B-Instruct Baseline

This report presents the TPU performance diagnostic vector derived from the Cloud TPU xprof traces for the **`Qwen/Qwen2.5-Coder-3B-Instruct`** model serving baseline.

The benchmark was executed on a **Cloud TPU v6e-8** slice (8 physical chips) with the optimized Phase 0 parameters:
* **Block Size**: 256
* **KV Cache Precision**: FP8
* **Tensor Parallel Size**: 8 (single-host execution)
* **Concurrency Limit**: 128 sequences (request mix: 1024 input, 1024 output)

---

## 📊 Phase-Level Performance Vectors

### 1. Prefill Phase (Step graph: `prefill_heavy/2026_06_21_17_55_51`)

* **Step Latency per Core**: **3.76 ms**
* **Roofline Regime**: Compute-Bound (Binding Resource: TPU Vector/Matrix Processing Unit)
* **Distance from Peak FLOP Rate**: **99.40%**
* **TPU MXU (Matrix Co-Processor) Utilization**: **0.7%**
* **DMA Unit Idle Time**: **0.0%** (fully overlapped or busy)
* **HBM Bandwidth Utilization**: **0.36%** (5.51 GB/s vs 1,525.5 GB/s theoretical peak)

#### Prefill Headroom Analysis (Ops >5% of Step Time)

| Operation | Self Time (%) | Category | Measured Speed (GFLOPs/s) | Notes / Bottleneck |
| :--- | :---: | :---: | :---: | :--- |
| `RowParallelLinear/dot_general` | **48.32%** | MatMul (dot_general) | **71,081.07** | **Down Projection / Attention Output Projection**. Low arithmetic efficiency (7.5% of peak). |
| `ragged_paged_attention/pallas_call` | **16.56%** | Custom Pallas Attention | **0.00** | Custom memory-intensive attention execution. |
| `MergedColumnParallelLinear/dot_general` | **7.08%** | MatMul (dot_general) | **817,794.73** | **Gate/Up Projection & QKV Projection**. High arithmetic efficiency (86.4% of peak). |

---

### 2. Decode Phase (Step graph: `decode_heavy/2026_06_21_17_55_59`)

* **Step Latency per Core**: **0.211 ms** (211.2 microseconds)
* **Roofline Regime**: Compute-Bound (Binding Resource: TPU Vector/Matrix Processing Unit)
* **Distance from Peak FLOP Rate**: **89.43%**
* **TPU MXU (Matrix Co-Processor) Utilization**: **4.0%**
* **DMA Unit Idle Time**: **0.0%**
* **HBM Bandwidth Utilization**: **6.38%** (97.26 GB/s vs 1,525.5 GB/s theoretical peak)

#### Decode Headroom Analysis (Ops >5% of Step Time)

| Operation | Self Time (%) | Category | Measured Speed (GFLOPs/s) | Notes / Bottleneck |
| :--- | :---: | :---: | :---: | :--- |
| `RowParallelLinear/dot_general` | **48.97%** | MatMul (dot_general) | **71,132.14** | **Down Projection / Attention Output Projection**. Low arithmetic efficiency (7.5% of peak). |
| `ragged_paged_attention/pallas_call` | **15.56%** | Custom Pallas Attention | **0.00** | Custom memory-intensive attention execution. |
| `MergedColumnParallelLinear/dot_general` | **7.18%** | MatMul (dot_general) | **817,414.10** | **Gate/Up Projection & QKV Projection**. High arithmetic efficiency (86.3% of peak). |

---

## 🔍 Critical Diagnostic Findings & Headroom Hypotheses

### 🚨 Finding 1: RowParallelLinear MatMul Efficiency Drop (Down Projection)
The Row-Parallel matrix multiplication operations (`RowParallelLinear/dot_general`) consume **nearly 50% of the entire execution time** in both prefill and decode phases.
While the Column-Parallel projection (`MergedColumnParallelLinear`) runs at a near-hardware-optimal speed of **817 TFLOPS/sec (86.4% peak)**, the Row-Parallel counterpart drops to only **71 TFLOPS/sec (7.5% peak)**.

* **Root Cause Analysis**:
  In the 3B parameter model shape on TP=8 tensor parallel, the row projection dimension is split across the 8 TPU devices. For the FFN Down-Projection, this leads to an unaligned contraction size of **1376** (which is $11008 / 8$). Because 1376 is not a multiple of the MXU hardware tiling dimension (128), XLA is forced to insert aggressive padding/masking overlays, dramatically reducing hardware pipeline efficiency.
* **Overhead Verification**:
  The subsequent `all-reduce` communication steps only take **10.4 microseconds** (0.27% of step time), verifying that collective communications are not the source of this performance degradation.

> [!TIP]
> **Hypothesis H-001 (Graph-Rewrite / Sharding Layout Refactor) - RESOLVED / ACCEPTED**:
> * **Action Taken**: Aligned the MLP intermediate dimension (`config.intermediate_size`) from `11008` to `11264` (a multiple of `1024` for `TP=8`, resulting in a perfect `1408` contracting dimension on each device, which is a multiple of the `128` TPU MXU hardware tile size).
> * **Performance Impact**:
>   * **Total Token Throughput**: Increased from `7,370 tok/s` to **`12,867.42 tok/s`** (**+$74.6\%$ throughput increase**).
>   * **Throughput per Chip**: Increased from `921.25 tok/s` to **`1,608.43 tok/s`**.
>   * **Median Time to First Token (TTFT)**: Reduced from `3,492.56 ms` to **`45.49 ms`** (**$76\times$ reduction**).
>   * **Median Time per Output Token (TPOT)**: Reduced from `12.04 ms` to **`9.32 ms`** (**$22.5\%$ reduction**).
>   * **P99 Latency**: P99 TTFT dropped from `8,877.21 ms` to **`676.47 ms`**.
>   * **Numeric Correctness**: **100% Pass** on numeric equivalence verify (0 mismatches across all 100 prompts).


---

### 🚨 Finding 2: Custom Attention Kernel (`ragged_paged_attention`)
The custom attention Pallas kernel (`ragged_paged_attention/pallas_call`) consumes **~16% of step time**.

> [!NOTE]
> **Hypothesis H-002 (Kernel Autotuning / Tile-Size Sweep)**:
> Since this is a custom-written kernel, running a micro-benchmark and applying Roofline-aligned tile size overrides can reduce memory bandwidth stalls and improve SRAM cache utilization inside the VPU (Vector Processing Unit) local memory.
