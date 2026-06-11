# Hypothesis D: Query Block Sizing Derivation & Analysis

This document details the VMEM constraints, padding overhead, and TensorCore occupancy derivation for the custom Pallas attention kernel (`ragged_paged_attention`) on TPU v6e (`cathygao-v6e`, `v6e-8` accelerator with `TP=4`).

---

## 1. VMEM Allocation & Constraints

The GQA Pallas attention kernel allocates dynamic VMEM buffers during compilation. The size of these buffers scales with the query block size `bq_sz`.

### Parameters per Chip (TP=4):
*   `actual_num_kv_heads` = $8 / 4 = 2$ heads
*   `num_q_heads_per_kv_head` = $40 / 8 = 5$ (aligned to $6$ due to packing)
*   `head_dim` = $128$
*   `page_size` = $128$
*   `q_dtype` = `bf16` ($2$ bytes)
*   `kv_dtype` = `fp8` ($1$ byte)

### VMEM Estimation Equation:
The estimated VMEM size in bytes is:
$$\text{VMEM Estimate} = \frac{\text{Total Bits}}{8} \times 2.4$$

Where:
$$\begin{aligned}
\text{Total Bits} &= (2 \times \text{bkv\_sz} \times \text{num\_kv\_heads\_x2} \times \text{head\_dim}) \times (32 // \text{kv\_packing}) \\
&+ 2 \times (2 \times \text{actual\_num\_kv\_heads} \times \text{bq\_sz} \times \text{num\_q\_heads\_per\_kv\_head} \times \text{head\_dim}) \times (32 // \text{q\_packing}) \\
&+ 2 \times (\text{actual\_num\_kv\_heads} \times \text{bq\_sz} \times \text{num\_q\_heads\_per\_kv\_head} \times 128) \times 32 \\
&+ (\text{actual\_num\_kv\_heads} \times \text{bq\_sz} \times \text{num\_q\_heads\_per\_kv\_head} \times \text{head\_dim}) \times 32
\end{aligned}$$

Using `bkv_p = 16` (`bkv_sz = 2048` tokens), we evaluate the VMEM usage for different query block sizes (`bq_sz`):

| `bq_sz` | Total Bits | Raw Bytes | Estimated VMEM (with 2.4x multiplier) | Fits in 16MB VMEM Limit? |
|---|---|---|---|---|
| **16** (Current) | $20,709,376$ | $2.59 \text{ MB}$ | **5.9 MB** | **Yes** (Huge Headroom) |
| **32** | $24,641,536$ | $3.08 \text{ MB}$ | **7.0 MB** | **Yes** |
| **64** | $32,505,856$ | $4.06 \text{ MB}$ | **9.3 MB** | **Yes** |
| **128** | $48,234,496$ | $6.03 \text{ MB}$ | **14.4 MB** | **Yes** (Safe margin) |

**Conclusion:** VMEM size is **not** the bottleneck preventing the kernel from using a larger query block size.

---

## 2. Systolic Occupancy vs. Tiling Padding Trade-off

The query block size `bq_sz` dictates systolic array alignment and query padding. Since the TPU v6e MXU systolic array tile size is **256×256**:

$$\text{MXU Occupancy} = \frac{\text{bq\_sz}}{256}$$

During the decode phase, the query sequence length per sequence is $1$. For a batch size $B$, the total number of query tokens is $B$.

$$\text{Padding Overhead} = \frac{\text{bq\_sz} - B}{\text{bq\_sz}}$$

### Trade-off Evaluation for Batch Size = 8:

*   **Case A: `bq_sz = 8`**
    *   Padding: $\frac{8 - 8}{8} =$ **0%**
    *   MXU Occupancy: $\frac{8}{256} =$ **3.1%** (Severe hardware under-utilization)
*   **Case B: `bq_sz = 16`** (Current Auto-tuned Default)
    *   Padding: $\frac{16 - 8}{16} =$ **50%**
    *   MXU Occupancy: $\frac{16}{256} =$ **6.2%**
*   **Case C: `bq_sz = 32`**
    *   Padding: $\frac{32 - 8}{32} =$ **75%**
    *   MXU Occupancy: $\frac{32}{256} =$ **12.5%**

**Conclusion:** With small batch sizes (e.g., $8$), increasing the block size to improve MXU occupancy results in prohibitive padding overhead.

---

## 3. Recommended Optimization Path

To achieve optimal kernel performance, we must scale the serving batch size to eliminate padding overhead while maximizing systolic occupancy:

*   **Scaling to Batch Size = 64:** Compile with `bq_sz = 64`.
    *   Padding: **0%**
    *   MXU Occupancy: **25%**
*   **Scaling to Batch Size = 256:** Compile with `bq_sz = 256`.
    *   Padding: **0%**
    *   MXU Occupancy: **100%** (Max hardware throughput)
