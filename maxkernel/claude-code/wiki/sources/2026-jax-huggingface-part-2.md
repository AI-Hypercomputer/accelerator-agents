---
title: "How to Run a Hugging Face Model in JAX (Part 2): 8-way tensor parallelism"
type: source
tags: [blog, torchax, jax, huggingface, llama, tensor-parallelism, gspmd, sharding, tpu-v6e]
author: Han Qi (qihqi)
upstream: https://github.com/qihqi/learning_machine/blob/main/jax-huggingface/02-run-huggingface-model-distributed.md
companion_script: jax_hg_02.py
created: 2026-04-22
updated: 2026-04-22
---

Blog post #2 of a four-part series. Extends Part 1's single-chip forward pass to 8-way tensor parallelism (NeMo-Megatron sharding) on an 8-chip TPU v6e host using JAX's gSPMD programming model. Headline measured result: **~4.3× end-to-end speedup** on 8 chips vs 1 chip for Llama-2-7B forward — i.e. **3.4–3.8 ms cached** per call vs the 13 ms single-chip baseline. The post explicitly notes the speedup is sub-linear and points at `jax.profiler.trace` + xprof as the tool to understand why.

## Overview

The post takes the Part-1 script and adds exactly two things: (1) a `jax.make_mesh((jax.device_count(),), ('axis',))` mesh and (2) a per-weight sharding assignment function `shard_weights_llama(mesh, weights)`. Under JAX's gSPMD model, this is sufficient — the XLA compiler inserts the collectives (one all-reduce per attention block, one per MLP block, per the column/row matmul pairing). The post also teaches the canonical NeMo-Megatron column-then-row decomposition of a transformer block so the sharding assignment is not magic.

## Key claims

1. **Tensor parallelism (NeMo-Megatron style) requires only one all-reduce per matmul pair** when you shard the first matmul by column and the second by row. For a Llama attention block, this is Q/K/V (column) → attention → O (row). For an MLP, this is gate+up (column) → SiLU+multiply → down (row).
2. **Attention itself is data-parallel across heads** — no collective needed inside the attention math — because each head's Q/K/V/output slice is fully local when sharded on the head dimension.
3. **JAX gSPMD is one-process-per-host, not one-process-per-device.** The programmer specifies mesh + shardings; the compiler inserts collectives. (Contrast with PyTorch DDP / TP which is typically one-process-per-device with explicit collective calls.)
4. **The Llama sharding recipe is mechanical** — `q/k/v/gate/up = P('axis', None)`, `o/down/lm_head/embed = P(None, 'axis')`, everything else replicated. The post shows this works without modifying the HF model code.
5. **Inputs must be replicated, not sharded,** for this TP scheme (every device needs the full `input_ids`).
6. **Actual 8-device utilization must be verified by profile**, not inferred from wall-clock speedup alone. The post recommends `jax.profiler.trace` + the xprof TensorBoard plugin.

## Key data points

### Llama-2-7B bfloat16, forward pass, 8-chip TPU v6e (tensor-parallel)

| Iteration | Wall time | Notes |
|---|---|---|
| 0 (first jit call) | 5.062 s | includes compilation |
| 1 | 3.80 ms | cached |
| 2 | 3.43 ms | cached |

### Comparison vs Part 1 single-chip baseline

| Metric | 1-chip (Part 1) | 8-chip (Part 2) | Ratio |
|---|---|---|---|
| First call (cold) | 4.365 s | 5.062 s | 1.16× slower (more to compile) |
| Cached call | 13.0 ms | 3.43 ms | **~3.8× faster** (post rounds to "4.3×") |

Sub-linear scaling (3.8× on 8 chips, not 8×) is flagged but not quantified further. The post defers root-cause analysis to a profile screenshot (`image.png`) without numerical decomposition.

### Llama-2-7B weight-shape-to-sharding map (verbatim, 32 layers omitted for brevity)

| Tensor name pattern | Shape | Sharding | Rationale |
|---|---|---|---|
| `self_attn.q_proj.weight` | `(4096, 4096)` | `P('axis', None)` | column: first matmul of attn block |
| `self_attn.k_proj.weight` | `(4096, 4096)` | `P('axis', None)` | column |
| `self_attn.v_proj.weight` | `(4096, 4096)` | `P('axis', None)` | column |
| `self_attn.o_proj.weight` | `(4096, 4096)` | `P(None, 'axis')` | row: second matmul of attn block |
| `mlp.gate_proj.weight` | `(11008, 4096)` | `P('axis', None)` | column |
| `mlp.up_proj.weight` | `(11008, 4096)` | `P('axis', None)` | column |
| `mlp.down_proj.weight` | `(4096, 11008)` | `P(None, 'axis')` | row |
| `input_layernorm.weight`, `post_attention_layernorm.weight` | `(4096,)` | `P()` (replicated) | pointwise, no shardable axis |
| `embed_tokens.weight` | `(32000, 4096)` | `P(None, 'axis')` | post calls this "flexible" — row chosen |
| `lm_head.weight` | `(32000, 4096)` | `P(None, 'axis')` | same |
| `model.rotary_emb.inv_freq` | `(64,)` | `P()` | replicated |

## Techniques referenced

- **gSPMD (Generalized Single Program Multiple Data)** — JAX's primary parallelism model; programmer declares sharding, compiler emits collectives. See also the linked [JAX distributed-arrays doc](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html).
- **NeMo-Megatron tensor parallelism** — column/row matmul pairing to achieve one all-reduce per block. Linked [Lightning AI TP doc](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/tp.html).
- **`jax.make_mesh` + `NamedSharding` + `PartitionSpec`** — the three primitives used to declare a mesh and assign axes to tensors.
- **`jax.device_put(x, NamedSharding(mesh, spec))`** — the "apply sharding to a host-resident array" operation.
- **`jax.profiler.trace` + xprof** — the profile-capture idiom. See [xprof](../codebases/xprof.md).

## Gaps & caveats

- **Sub-linear scaling is unexplained.** 3.8× on 8 chips is a common TP ceiling (all-reduce + imperfect-overlap overhead), but the post does not break down the gap — no "X% of time in compute, Y% in comm" numbers, just a profile screenshot.
- **No per-collective measurement.** Which all-reduce is the bottleneck (attention-O or MLP-down) is not identified.
- **Input is always replicated.** Sequence parallelism is not considered. For long-context workloads, replicating the full `input_ids` on every chip is wasteful.
- **No comparison to FSDP or data parallelism.** Llama-2-7B at batch=1 is clearly TP territory, but the post does not justify the choice.
- **Hardware is v6e only.** The 3.8× scaling may not transfer to v5p (different interconnect topology) or GPU (NVLink mesh).
- **No MFU / tokens-per-second** — same gap as Part 1.

## Connections

Updates / informs:
- [codebases/jax-huggingface](../codebases/jax-huggingface.md) — this is the reference for the Llama TP sharding recipe.
- [codebases/torchax](../codebases/torchax.md) — the `extract_jax` output pytree is what `shard_weights_llama` keys into.
- [codebases/xprof](../codebases/xprof.md) — profile-capture recipe demonstrated here (`/tmp/jax-trace` with the xprof plugin).

Future hypothesis anchors (not filed — no `model/` page yet):
- **Sequence parallelism on top of TP** for long-context Llama inference.
- **Async-collective overlap tuning** (XLA flag) to close the 3.8×→8× gap.
- **tokamax splash-attention swap** inside the attention block — the row/column sharding is unchanged but the attention kernel is not HF SDPA. See [tokamax](../codebases/tokamax.md).

## See also

- [tensor-parallelism](../concepts/tensor-parallelism.md) — the NeMo-Megatron scheme this post applies.
- [sharding](../concepts/sharding.md) — gSPMD mesh + `NamedSharding` primitives.
- [all-reduce](../concepts/all-reduce.md) — the single collective per attention/MLP block under this scheme.
- [collective-communication](../concepts/collective-communication.md) — broader context for gSPMD-inserted collectives.
- [sequence-parallelism](../concepts/sequence-parallelism.md) — complementary TP extension (not used here).
- [profile-capture](../concepts/profile-capture.md) — `jax.profiler.trace` usage demonstrated.
- [Part 1](2026-jax-huggingface-part-1.md) — single-device baseline this post extends.
- [Part 3](2026-jax-huggingface-part-3.md) — decode-side counterpart with `StaticCache`.

## Sources

- `raw/code/learning-machine/jax-huggingface/02-run-huggingface-model-distributed.md`
- `raw/code/learning-machine/jax-huggingface/jax_hg_02.py`
- `raw/code/learning-machine/jax-huggingface/tensor-parallelism.png` (diagram)
- `raw/code/learning-machine/jax-huggingface/image.png` (xprof screenshot, 8-device trace)
- Upstream: <https://github.com/qihqi/learning_machine/blob/main/jax-huggingface/02-run-huggingface-model-distributed.md>
