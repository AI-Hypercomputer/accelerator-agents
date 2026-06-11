# Hypothesis 3 Exploration: GMM Dynamic Tiling for MoE Routing-Imbalance Amortization

This artifact presents a detailed algebraic derivation and performance analysis of **Hypothesis 3 (GMM dynamic tiling for routing-imbalance amortization)** in Mixture-of-Experts (MoE) models.

---

## 📋 Problem Statement & Observation

In Mixture-of-Experts (MoE) layers, token routing is dynamic: different experts receive highly unbalanced numbers of tokens depending on the input sequence context.

Grouped Matrix Multiply (GMM) computes the MLP projections for all experts in a single execution grid. 
*   If we use a **fixed tile size** (e.g., $B_{\text{fixed}} = 64$ tokens), experts with very few active tokens (e.g. 8 tokens) suffer from massive padding overhead.
*   Because different experts have unbalanced workloads, TPU cores assigned to low-workload experts finish early and stall, resulting in severe hardware under-utilization.

---

## 💡 Algebraic Model & Padding Overhead

Let:
*   $E$ be the number of active experts.
*   $T_e$ be the number of tokens routed to expert $e \in \{0, \dots, E-1\}$.
*   $B_{\text{fixed}}$ be the fixed row tile size (block size along the token dimension).

### 1. Fixed-Tiling Padding Cost
The number of blocks allocated to expert $e$ is:
$$N_e = \left\lceil \frac{T_e}{B_{\text{fixed}}} \right\rceil$$
The total computed tokens (including padding) is:
$$\text{Tokens}_{\text{padded}} = \sum_{e=0}^{E-1} \left\lceil \frac{T_e}{B_{\text{fixed}}} \right\rceil \times B_{\text{fixed}}$$
The wasted compute overhead ratio is:
$$\text{Ratio}_{\text{padding}} = \frac{\sum_e \lceil T_e / B_{\text{fixed}} \rceil \times B_{\text{fixed}} - \sum_e T_e}{\sum_e T_e}$$

**Example (Imbalanced Routing):**
Suppose $E=4$ experts, and tokens are routed as: $T_0 = 128$, $T_1 = 8$, $T_2 = 16$, $T_3 = 0$.
With $B_{\text{fixed}} = 64$:
*   $N_0 = 2$ blocks. Computed tokens = 128. Padding = 0.
*   $N_1 = 1$ block. Computed tokens = 64. Padding = 56.
*   $N_2 = 1$ block. Computed tokens = 64. Padding = 48.
*   $N_3 = 0$ blocks. Computed tokens = 0. Padding = 0.
*   Total Computed: $256$. Total Real Tokens: $152$.
*   **Padding Overhead:**
    $$\text{Ratio}_{\text{padding}} = \frac{256 - 152}{152} \approx 68.4\%$$
    *Here, 40.6% of the compute FLOPs are wasted on padding zeros.*

---

## 🧪 GMM Dynamic Tiling Mitigations

With dynamic tiling, we define a set of supported tile sizes $\mathcal{S} = \{8, 16, 32, 64, 128\}$.
The row tile size $B_e$ for expert $e$ is determined dynamically:
$$B_e = \min\left(\{B \in \mathcal{S} \mid B \ge T_e\} \cup \{B_{\max}\}\right)$$

### 2. Dynamic-Tiling Padding Cost
*   For Expert 0 ($T_0 = 128$): Select $B_0 = 128$. Computed tokens = 128. Padding = 0.
*   For Expert 1 ($T_1 = 8$): Select $B_1 = 8$. Computed tokens = 8. Padding = 0.
*   For Expert 2 ($T_2 = 16$): Select $B_2 = 16$. Computed tokens = 16. Padding = 0.
*   **Total Computed:** $152$. **Padding Overhead:** **$0\%$**.

### 3. Implementation Strategies in Pallas
Since Pallas VMEM buffers require static shape definitions, we utilize two implementation paths:
1.  **Heterogeneous Multi-Tile Dispatches**: Compile multiple specialized GMM kernels (e.g. a small-tile kernel for $B_m=16$ and a large-tile kernel for $B_m=64$). The host-side scheduler bins experts by workload and launches the appropriate kernel.
2.  **Dynamic Masking with Pointer Offsets**: Implement a single kernel with a small block size (e.g. $B_m=16$) and combine multiple steps inside a loop for larger workloads, avoiding compile-time padding.

---

## 📋 Hypothesis Filing (deep_research Output)

- **Class:** `kernel-novel` + `kernel-autotune`
- **Origination:** Profile
- **Author:** MaxKernel (multi-tile Pallas kernels) + MaxTile (tile parameter selection)
- **Evidence:** Spike in DMA idle % and low arithmetic intensity during low-token expert steps.
- **Expected Gain:** **10% to 20%** reduction in GMM bucket execution time on imbalanced steps; **3% to 8%** end-to-end speedup.
- **Accept/Reject Criteria:** Padding overhead ratio drops below $5\%$, and GMM latency scales linearly with active tokens rather than active experts.
