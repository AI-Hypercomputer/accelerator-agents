<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Ragged Paged Attention: A High-Performance and Flexible LLM Inference Kernel for TPU

- **Title**: Ragged Paged Attention: A High-Performance and Flexible LLM Inference Kernel for TPU
- **Authors**: JAX/XLA TPU Inference team
- **Source**: [arXiv:2604.15464](https://arxiv.org/abs/2604.15464) / [HTML Version](https://arxiv.org/html/2604.15464)
- **Status**: Production-grade attention kernel integrated into vLLM (TPU backend) and SGLang.

---

## 1. Key Architectural Challenges on TPU
1. **Static-First Compiler Stack**: JAX/XLA compilation is heavily optimized for static shapes and regular access patterns. Today's LLM serving workloads (with dynamic mixed batching of decode/prefill and varying sequence lengths) run counter to this model.
2. **Tiled Memory Layout**: Data in HBM, VMEM, and registers is organized in tiles (e.g., `T(8, 128)`). Slicing or scattering/gathering along dynamic or unaligned boundaries causes severe overhead (VPU unpacking, blending, and repacking).
3. **Immutable Semantics**: JAX's immutable array updates make standard in-place KV cache modifications expensive.

---

## 2. Key Techniques & Solutions in RPA

### A. Fine-Grained Tiling for Raggedness
* **Problem**: If the sequence length dimension or head sharding is placed on a minor (tiling) dimension, XLA adds implicit padding (e.g., padding `12` to `16` to fit a `T(8,128)` tile) and forces slow VPU slice calculations.
* **Solution**: RPA introduces a **packing dimension** in the second minor dimension of Q, K, and V (e.g., `(s, ⌈h/p⌉, p, d_k)` where $p$ is the packing factor, e.g., 2 for BF16). This forces XLA to select the minimum hardware tile size (`T(packing, 128)`), allowing arbitrary dynamic slicing along the leading non-tiled dimension without VPU blending overhead.

### B. KV Cache Update & Transpose Fusion
* **Transpose Fusion**: Strided vector loads fetch $K$ and $V$ from VMEM and interpret them on-the-fly, avoiding explicit transposition. To prevent VMEM bank conflicts from strided access, a small offset is introduced in the block layout to make the effective stride odd.
* **Merged KV Layout**: $K$ and $V$ are merged along the head dimension into a single representation `(s_kv, ⌈2*h_kv/p_kv⌉, p_kv, d_k)`. This halves the number of load/store operations and increases the DMA transfer size, amortizing DMA latency and matching hardware efficiency thresholds.
* **Cache Update Fusion**: KV cache update (scatter) is fused directly inside the FlashAttention compute loop. Newly projected KV tokens are written back to HBM asynchronously, overlapping the write latency with the compute stages of attention.

### C. Distribution-Aware Compilation
* **Workload Segmentation**: RPA segments the dynamic input batch into three distinct regions: decode-only (token length = 1), prefill-only (fixed-size chunk), and mixed-batch.
* **Specialized Dispatches**: Case enums are passed to the kernel, allowing it to dispatch to specialized pre-compiled variants. In decode-only and prefill-only regions, block size $b_q$ is statically known at compile time, eliminating VREG allocation overhead and allowing aggressive loop unrolling.

### D. Custom Software Pipeline
* RPA departs from Pallas's high-level automatic multi-buffering. It uses primitive APIs (`async_copy`, `semaphore_signal`) to explicitly schedule HBM-to-VMEM gathers, in-place updates, and VMEM-to-HBM stores.
* It manages four overlapping asynchronous stages:
  1. **Fetch $B_q$**: Transfers only the effective token range.
  2. **Fetch $B_{kv}$**: Gathers non-contiguous KV pages from HBM according to the page table and concatenates them in VMEM.
  3. **Update $U_{kv}$**: Asynchronously writes back new tokens.
  4. **Send $B_o$**: Returns results to HBM.

---

## 3. Kernel Tuning Parameters
RPA exposes four primary block-size parameters for offline tuning (stored in a lookup table and loaded at server startup):
- $b_q$, $b_{kv}$: Tuned to optimize HBM-to-VMEM DMA transfer sizes and overlap.
- $c_q$, $c_{kv}$: Tuned to fit compute tiles into register constraints, preventing VREG spilling.

---

## 4. Performance Results
* **Decode**: Up to **86% Memory Bandwidth Utilization (MBU)** on TPU v7x (Ironwood) at context lengths above 8K (or 4K for $d_k=256$). KV cache update overhead is fully hidden.
* **Prefill**: Up to **73% Model FLOPs Utilization (MFU)** without causal masking, and **63% MFU** with causal masking. Reaches compute saturation at sequences $\ge$ 8K.
* **Impact**: Enabled a **2x to 5x increase in token throughput** when integrated as the primary TPU backend in vLLM and SGLang.

---

## 5. System Integration & Layout Caveats
* **XLA Layout Override**: JAX/XLA's Layout Assignment pass frequently reorders intermediate layouts (performing implicit transpositions) to optimize global compute, ignoring user-specified constraints on intermediate tensors.
* **Workaround**: RPA defers reshaping and merging to the kernel preprocessing stage rather than doing it offline in weight preparation.
* **Recompilation Guard**: To prevent compilation latency on the serving critical path, inputs must be padded to predefined upper bounds for the maximum tokens ($s$) and maximum sequences ($n$) established during server boot.
