<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# MaxText Optimized Models Tiering

- **Source URL**: `https://maxtext.readthedocs.io/en/latest/reference/models/tiering.html`
- **Status**: Reference list of optimized models and recipes for Cloud TPU platforms (Trillium v6e and v5p).

---

## 1. Definition of Tiers
* **Gold Tier**: Fully Optimized Models certified to run with maximum efficiency on Cloud TPUs. They are thoroughly refined for the highest possible performance, making them ideal for production-critical workloads requiring peak throughput.
* **Silver Tier**: High Performance Models that are well-optimized to deliver high, reliable performance on Cloud TPUs. They are effective for most use cases but may offer opportunities for expert tuning to achieve peak (Gold Tier) performance.

---

## 2. Trillium (v6e) Models

### Gold Tier
* **Llama 2 70B**:
  * **Recipe**: [training/trillium/Llama2-70B-MaxText](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Llama2-70B-MaxText)
  * **Benchmark Config**: 256 Chips, BF16, Sequence Length (SL) = 4096
  * **Model FLOPs Utilization (MFU)**: 43.8%
  * **Performance**: ~900 tokens/sec/device
* **Llama 3.1 8B**:
  * **Recipe**: [training/trillium/Llama3.1-8B-MaxText/v6e-256](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Llama3.1-8B-MaxText/v6e-256)
  * **Benchmark Config**: 256 Chips, BF16, SL = 8192
  * **MFU**: 45.46%
  * **Performance**: ~7,207 tokens/sec/device
* **Llama 3.1 70B**:
  * **Recipe**: [benchmarks/maxtext_trillium_model_configs.py](https://github.com/AI-Hypercomputer/maxtext/blob/92e59fdf547421f647590087f50fea5729da42d8/benchmarks/maxtext_trillium_model_configs.py#L959)
  * **Benchmark Config**: 256 Chips, BF16, SL = 8192
  * **MFU**: 50.33%
  * **Performance**: ~960 tokens/sec/device

### Silver Tier
* **Llama 3.1 405B**:
  * **Recipe**: [benchmarks/maxtext_trillium_model_configs.py](https://github.com/AI-Hypercomputer/maxtext/blob/5e6a7caff904f67fa654fc0ae983a16156bc21f8/benchmarks/maxtext_trillium_model_configs.py#L723)
  * **Benchmark Config**: 256 Chips, BF16, SL = 8192
  * **MFU**: 38.55%
  * **Performance**: ~123 tokens/sec/device
* **Mixtral 8x7B**:
  * **Recipe**: [training/trillium/Mixtral-8x7B-MaxText](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Mixtral-8x7B-MaxText)
  * **Benchmark Config**: 256 Chips, BF16, SL = 4096
  * **MFU**: 35.23%
  * **Performance**: ~3,899 tokens/sec/device
* **Mixtral 8x22B**:
  * **Recipe**: [training/trillium/Mixtral-8x22B-MaxText](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Mixtral-8x22B-MaxText)
  * **Benchmark Config**: 256 Chips, BF16, SL = 4096
  * **MFU**: 36.2%
  * **Performance**: ~1,326 tokens/sec/device

---

## 3. v5p Models

### Gold Tier
* **Llama 2 70B**:
  * **Recipe**: [benchmarks/maxtext_v5p_model_configs.py](https://github.com/AI-Hypercomputer/maxtext/blob/92e59fdf547421f647590087f50fea5729da42d8/benchmarks/maxtext_v5p_model_configs.py#L156)
  * **Benchmark Config**: 512 Chips, BF16, SL = 4096
  * **MFU**: 65.4%
  * **Performance**: ~692 tokens/sec/device

### Silver Tier
* **Mixtral 8x7B**:
  * **Recipe**: [training/v5p/Mixtral-8X7B-Maxtext](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/v5p/Mixtral-8X7B-Maxtext)
  * **Benchmark Config**: 256 Chips (8x4x4 topology), BF16, SL = 4096
  * **MFU**: 52.56%
  * **Performance**: ~2,909 tokens/sec/device

---

## 4. Performance Notes
* MFU calculations incorporate halving of causal attention FLOPs as per MaxText PR #1988 (which impacts sequence lengths with causal masking).
