# Hypothesis 2 Exploration: Fusing Ragged Unpermute and Reduce in EP Backward Path

This artifact presents a detailed algebraic derivation and HBM memory-bandwidth analysis for **Hypothesis 2 (Fuse ragged unpermute and reduce in EP backward path)**.

---

## 📋 Problem Statement & Observation

In the Expert Parallel (EP) backward pass of Mixture-of-Experts (MoE) models:
1.  Gradients from the expert outputs are scattered back to their original sequence order (**Ragged Unpermute**).
2.  A reduction (e.g. summing gradients along the sequence/expert dimension) is performed on the unpermuted gradients (**Reduce**).

In a non-fused (sequential) baseline:
*   The `unpermute` kernel materializes a large intermediate tensor of shape $(T, D)$ in HBM.
*   The `reduce` kernel then reads this intermediate tensor from HBM to perform the reduction.
This round-trip to HBM wastes valuable memory bandwidth and creates a significant latency bottleneck.

---

## 💡 HBM Memory-Traffic Model

Let:
*   $T$ be the total active tokens across all experts ($T = \sum_{e} C_e$, where $C_e$ is the capacity of expert $e$).
*   $D$ be the hidden dimension.
*   $R$ be the dimension of the reduced output ($R \le T$).

Assume a standard `bfloat16` layout (2 bytes per element).

### 1. Non-Fused (Sequential) Memory Traffic
The sequential execution flow incurs the following HBM transactions:
1.  **Ragged Unpermute**:
    *   Read expert gradients from HBM: $2 \times T \times D$ bytes.
    *   Write unpermuted intermediate to HBM: $2 \times T \times D$ bytes.
2.  **Reduce**:
    *   Read unpermuted intermediate from HBM: $2 \times T \times D$ bytes.
    *   Write reduced gradients to HBM: $2 \times R \times D$ bytes.

Total HBM traffic:
$$\text{Traffic}_{\text{seq}} = 6 \times T \times D + 2 \times R \times D \text{ bytes}$$

---

### 2. Fused Custom Kernel Memory Traffic
The fused kernel performs the unpermute scatter indexing mapping and accumulates the reduction on the fly in TPU VMEM/registers, avoiding intermediate HBM materialization:
1.  **Fused Unpermute + Reduce**:
    *   Read expert gradients from HBM: $2 \times T \times D$ bytes.
    *   Resolve scatter-reduction mappings dynamically in VMEM.
    *   Write final reduced gradients directly to HBM: $2 \times R \times D$ bytes.

Total HBM traffic:
$$\text{Traffic}_{\text{fused}} = 2 \times T \times D + 2 \times R \times D \text{ bytes}$$

---

### 3. Quantitative HBM Savings
The memory bandwidth traffic reduction ($\Delta \text{Traffic}$) is:
$$\Delta \text{Traffic} = 4 \times T \times D \text{ bytes}$$

Assuming a high reduction factor ($R \ll T$):
$$\text{Traffic Reduction Ratio} = \frac{\text{Traffic}_{\text{seq}}}{\text{Traffic}_{\text{fused}}} \approx \frac{6 \times T \times D}{2 \times T \times D} = 3\text{x}$$

This represents a theoretical **3x reduction in memory traffic** for this segment of the backward pass.

---

## 📋 Hypothesis Filing (deep_research Output)

- **Class:** `graph-rewrite` + `kernel-novel` (Composite)
- **Origination:** Symbolic + Profile
- **Author:** MaxShard (graph change routing) + MaxKernel (fused Pallas custom kernel)
- **Evidence:** TPUDiagnoseAgent logs reporting $>5\%$ of step time spent on intermediate materialization during EP backward pass.
- **Expected Gain:** **15% to 25%** reduction in ragged-bucket execution time.
- **Accept/Reject Criteria:** Numeric-equivalence checks pass (no deviation from baseline), and HBM bandwidth consumption for the target segment drops by $\ge 60\%$.
