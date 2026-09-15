<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Customizing Model Configs for TPUs

- **Source URL**: `https://maxtext.readthedocs.io/en/latest/guides/optimization/custom_model.html`
- **Status**: Curated guide on hardware alignment, dimensions, sharding configurations, and architectural best practices for MaxText on Cloud TPUs.

---

## 1. TPU Hardware Alignment Best Practices
To keep the TPU Matrix Multiply Unit (MXU) fully utilized:
* **Systolic Dimension Alignment**:
  * **Trillium (v6e) and Ironwood (v7x)**: Optimized for **256x256** matrix multiplications. emb_dim, mlp_dim, and head_dim must be multiples of 256.
  * **v4, v5e, and v5p**: Optimized for **128x128** matrix multiplications. emb_dim, mlp_dim, and head_dim must be multiples of 128.
  * **General Fallback**: If standard sizes are not multiples of 256/128, ensure they are multiples of at least 8 to enable basic compiler optimizations.
* **Batch Size Alignment**:
  * For peak performance and memory padding minimization, the batch size should be a multiple of **128** (corresponding to internal 8x128 TPU core vector registers). Fallback to multiples of 8.

---

## 2. Platform Specific Guidelines

### Ironwood (TPU v7x)
* **FP8 Precision**: Delivers 2x throughput compared to BF16. Mixed-precision training should use FP8 for weights and activations where possible.
* **SparseCore Offloading**: Collective communication operations (All-Reduce, All-Gather, Reduce-Scatter) should be offloaded to SparseCore by default to overlap with TensorCore computations.
* **Dual-Chiplet Architecture**: Utilizes two TensorCores per chip connected via a fast die-to-die link (6x faster than ICI links).

---

## 3. Sharding Strategies & Arithmetic Intensity (AI)
Mesh dimension sizes and sharding configuration must be designed in tandem with model dimensions.

### Hardware ICI Arithmetic Intensity Limits
* **v5p**: 2550 for 1D-ICI
* **Trillium**: 5100 for 1D-ICI (or 2D without wraparound); 2550 for 2D-ICI (with wraparound on both dimensions, e.g. v6e-256)
* **Ironwood**: 12800 for 1D-ICI

### Fully Sharded Data Parallelism (FSDP)
* **Pure FSDP**: Requires global batch size / sparsity > hardware AI (e.g. global batch size > 40k tokens for v5p and Trillium under sparsity=16).
* **Layer Size Constraint**: Single layer of weights must fit into a single chip's memory (e.g. Trillium's 32GiB HBM limit restricts pure FSDP to layers with <= 5B params).
* **Mix FSDP**: Combined with TP, EP, or PP for larger models or when scaling. Condition: `per_device_batch / sparsity * TP * EP * PP > hardware AI`.

### Expert Parallelism (EP)
* Overcomes layer size limits in sparse models.
* **EP AI**: `4 * mlp_dim / EP`. Requires large MLP dimensions to exceed the hardware AI (e.g. `mlp_dim` > 5k for Trillium with EP=4).

### Tensor Parallelism (TP)
* Best for large dense models or super-large sparse models under small per-device batch sizes.
* **TP AI**: `mlp_dim / TP`.
* Requires large MLP dimensions (e.g. `mlp_dim` > 80k for TP=16 on Trillium to maintain efficiency). Used for custom 900B model.

### Pipeline Parallelism (PP)
* Used when global batch size limits per-device batch size, making data parallelism inefficient. Low communication cost (permutes layer inputs).
* **PP AI**: `1.5 * layers_per_stage * mlp_dim * num_experts_per_tok`.

---

## 4. Empirical Configurations Case Studies

### 900B Dense Model (Trillium)
* **Configuration**: emb_dim = 16384, mlp_dim = 131072, head_dim = 256, 128 layers.
* **Sharding**: TP=16.
* **Results**: 39.8% MFU (Trillium pod) with 4k batch size; stays at 37% MFU at 1k tokens per device.

### 700B Mixtral-like MoE (Trillium)
* **Configuration**: emb_dim = 8192, mlp_dim = 32768, 16 experts, 56 layers.
* **Results**: ~50% MFU at capacity factor 1.0; 38.1% MFU at capacity factor 1.5.

### 10T MoE Model (Trillium)
* **Configuration**: emb_dim = 10240, mlp_dim = 40960, 64 experts, 128 layers, sparsity = 32.
* **Sharding**: PP=16 (over DCN) combined with EP=64, TP=4 (over ICI).
* **Results**: 26.2% MFU on 16 pods with a low batch size of 2k tokens per device.
