---
title: recurrentgemma — overview
type: overview
updated: 2026-07-03
---
# recurrentgemma — what it is and how it fits together

## In one paragraph

RecurrentGemma (Griffin/Hawk) is a hybrid recurrent/attention language model: most layers use a
gated linear recurrence (the RG-LRU) with a short causal convolution, and every third layer uses
windowed local multi-query attention instead — the mix is literally a tuple,
[`GriffinConfig.block_types`](concepts/recurrentgemma-common.md), read off once at model-construction
time. The repo ships the *same* architecture twice: a JAX/Flax implementation with a hand-written
TPU Pallas kernel for the recurrence, and a PyTorch implementation with a plain Python scan loop —
kept numerically equivalent by a shared config module and a cross-framework test suite. The central
design idea worth internalizing is that the RG-LRU's per-step decay is *input-dependent* (gated, not
fixed), computed in log-space for numerical safety, and that its sequential computation is the one
piece of the model deliberately isolated behind a swappable backend (native JAX loop, associative
scan, or Pallas kernel) via `ScanType`.

## Core architecture

```mermaid
flowchart TD
  CFG["GriffinConfig / Preset\n(recurrentgemma-common)"] --> GRIFFIN["Griffin model\n(jax-griffin / torch-modules)"]
  GRIFFIN --> BLOCKS["ResidualBlock × num_layers\n(jax-modules / torch-modules)"]
  BLOCKS -->|RECURRENT| REC["RecurrentBlock:\nConv1D + RGLRU\n(jax-layers / torch-layers)"]
  BLOCKS -->|ATTENTION| ATT["LocalAttentionBlock:\nRoPE + windowed MQA"]
  REC --> SCANDISPATCH["ScanType dispatch"]
  SCANDISPATCH -->|TPU, AUTO| PALLAS["Pallas kernel\n(jax-pallas)"]
  SCANDISPATCH -->|native / CPU| NATIVE["jax.lax.scan or\ntorch Python loop"]
  REC -.->|bf16 / Pallas| COMPLEX["Complex wrapper\n(jax-complex_lib)"]
  GRIFFIN --> SAMPLER["Sampler: prefill + decode\n(jax-sampler / torch-sampler)"]
  TYPING["array_typing: shape aliases\n+ typed decorator"] -.->|checked (JAX) /\ndocumentation-only (torch)| GRIFFIN
```

## Main concepts

**Configuration as data** — [`GriffinConfig`](concepts/recurrentgemma-common.md) is a hashable
`NamedTuple` whose `block_types` field literally *is* the architecture (a tuple of
`RECURRENT`/`ATTENTION` per layer), and whose `scan_type` decides the recurrence backend without
touching model code. Three independent constructors (`from_preset`, `from_flax_params_or_variables`,
`from_torch_params`) all funnel into one consistency-checked path, so a config can be recovered
purely from a checkpoint's parameter-tree shape. See
[recurrentgemma-common](concepts/recurrentgemma-common.md).

**The gated linear recurrence (RG-LRU)** — the model's namesake contribution: an input-dependent
decay rate computed in log-space (`log_a = -8 * sigmoid(gate) * softplus(param)`), normalized with a
gradient-clipped `sqrt` for bf16 stability, optionally represented as a hand-rolled
[`Complex`](concepts/recurrentgemma-jax-complex_lib.md) number pair when running at reduced precision
or inside Pallas. See [recurrentgemma-jax-layers](concepts/recurrentgemma-jax-layers.md) (JAX) /
[recurrentgemma-torch-layers](concepts/recurrentgemma-torch-layers.md) (torch).

**The TPU Pallas scan kernel** — the JAX lane's performance-critical core: a 3-D-gridded Pallas
kernel computing the recurrence with a `fori_loop` per grid cell, a hand-implemented forward/backward
via `jax.custom_vjp` (the same kernel body runs both directions), and a post-hoc
all-gather-based correction for sequence-sharded (context-parallel) execution. The torch lane has no
equivalent — it uses a plain Python loop instead. See
[recurrentgemma-jax-pallas](concepts/recurrentgemma-jax-pallas.md).

**The residual block's temporal-mixing dispatch** — `ResidualBlock` builds *either* a
`RecurrentBlock` *or* a `LocalAttentionBlock` at construction time based on one config-tuple entry,
exposing both through a single `temporal_block` property so the surrounding pre-norm/MLP/residual
logic is agnostic to which one is present; local attention itself is windowed, single-KV-head
(multi-query), and RoPE'd. See [recurrentgemma-jax-modules](concepts/recurrentgemma-jax-modules.md) /
[recurrentgemma-torch-modules](concepts/recurrentgemma-torch-modules.md).

**The top-level model and gradient checkpointing** — `Griffin` stacks one `ResidualBlock` per
`block_types` entry (optionally `nn.remat`-wrapped uniformly across every layer), scales each block's
final-layer init by `1/num_layers`, and returns a `(logits, cache)` pair whose components are
independently optional (`return_logits`/`return_cache`) so the same `__call__` serves both scoring
and cache-only advancement. See [recurrentgemma-jax-griffin](concepts/recurrentgemma-jax-griffin.md).

**Two-phase autoregressive sampling** — `Sampler` splits generation into a one-shot prefill
(processing the whole prompt, building the initial cache) and a token-at-a-time decode loop; the JAX
lane traces the entire decode loop as one `jax.lax.while_loop` inside a `jax.jit`, while the torch
lane runs a plain eager Python `while` with no compilation and no explicit RNG-key threading. See
[recurrentgemma-jax-sampler](concepts/recurrentgemma-jax-sampler.md) /
[recurrentgemma-torch-sampler](concepts/recurrentgemma-torch-sampler.md).

**Shape-typed everything, enforced in one lane only** — nearly every function across both lanes
carries shape-annotated type aliases (`Activations`, `SegmentPos`, `ExpandedActivations`, etc.) via an
`@at.typed` decorator; in JAX this performs a real `jaxtyping`+`typeguard` runtime check, while in
torch the same decorator is a deliberate no-op ("breaks torch.compile"), leaving the annotations as
documentation only. See [recurrentgemma-jax-array_typing](concepts/recurrentgemma-jax-array_typing.md) /
[recurrentgemma-torch-array_typing](concepts/recurrentgemma-torch-array_typing.md).

## How a request flows

Config resolution (`Preset` → `GriffinConfig`) → model construction (`Griffin.setup`/`__init__`
builds one `ResidualBlock` per `block_types` entry, each internally choosing `RecurrentBlock` or
`LocalAttentionBlock`) → `Sampler` tokenizes the prompt → prefill (`_prompt_processing_fn`, one full
forward pass building the initial cache) → decode loop (`_sample_step` repeated, one token at a time,
each step running every layer's `RGLRU`/`Conv1D` recurrence — dispatched through `ScanType` to either
the Pallas kernel or a native loop — and every attention layer's windowed KV-cache update) →
detokenization into `SamplerOutput`.

## Map of the wiki

- Model configuration, presets, and the `block_types`/`scan_type` switches → [recurrentgemma-common](concepts/recurrentgemma-common.md).
- "How does the RG-LRU actually compute its recurrence?" → [recurrentgemma-jax-layers](concepts/recurrentgemma-jax-layers.md)
  (JAX) / [recurrentgemma-torch-layers](concepts/recurrentgemma-torch-layers.md) (torch).
- "Where's the TPU-specific performance code?" → [recurrentgemma-jax-pallas](concepts/recurrentgemma-jax-pallas.md).
- "How does bf16/complex-number handling work?" → [recurrentgemma-jax-complex_lib](concepts/recurrentgemma-jax-complex_lib.md).
- "How do recurrent and attention layers combine into one block?" → [recurrentgemma-jax-modules](concepts/recurrentgemma-jax-modules.md)
  / [recurrentgemma-torch-modules](concepts/recurrentgemma-torch-modules.md).
- "How does the whole model forward pass work?" → [recurrentgemma-jax-griffin](concepts/recurrentgemma-jax-griffin.md).
- "How does text generation/sampling work?" → [recurrentgemma-jax-sampler](concepts/recurrentgemma-jax-sampler.md)
  / [recurrentgemma-torch-sampler](concepts/recurrentgemma-torch-sampler.md).
- "What's the shape-checking convention?" → [recurrentgemma-jax-array_typing](concepts/recurrentgemma-jax-array_typing.md)
  / [recurrentgemma-torch-array_typing](concepts/recurrentgemma-torch-array_typing.md).
- For the exhaustive per-symbol index (every function/class/field with its def site), see `catalog/`.
