---
title: "Qwen3 jax — offload decoder-layer input (MaxText's 5th tensor)"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: open
expected_gain: "extra HBM headroom at seq8192 → bs4 reachable if scan+offload (v025) fits bs3 with margin"
confidence: medium
effort: S
origin: 2026-06-12-v024-offload-bs3
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — offload decoder-layer input

*Stub — expand when v025 (scan + offload_attn at bs3) lands.*

*Hypothesis*: Adding the decoder-layer input to the offloaded-names set
(`checkpoint_name` tag on the layer's hidden-state input, name in the
`save_and_offload_only_these_names` list) — matching MaxText's
`decoder_layer_input=offload` — frees a further ~`B×T×4096×2` bytes/layer of
HBM, measurable as lower peak HBM at bs3 @ seq 8192 and, if v025 fits with
< 2 GiB margin, the difference that lets **bs4 @ seq 8192** compile.

*Rationale*: MaxText's recipe offloads 5 tensors (layer input + q/k/v/out
proj); our v024/v025 port offloads only the 4 attention tensors.
[v024](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v024-offload-bs3.md)
proved the offload pass routes our tagged tensors to host correctly.

## See also

- [qwen3-jax-host-offload-remat](qwen3-jax-host-offload-remat.md)
