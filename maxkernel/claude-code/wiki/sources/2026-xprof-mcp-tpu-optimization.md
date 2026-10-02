---
title: "TPU Performance Optimization Guide (xprof-mcp docs)"
type: source
tags: [docs, performance, optimization, tpu, roofline, rematerialization, fusion, kv-cache, quantization, sharding, xla]
created: 2026-04-22
updated: 2026-06-01
---

Practical optimization guide shipped inside the `xprof-mcp` repository. Synthesizes TPU hardware fundamentals, the roofline diagnostic framework, a catalogue of common gotchas (dimension alignment, dtype, materialized broadcasts, dynamic shapes, remat, TP all-reduce, KV cache decode), training and inference playbooks, XLA compiler flags, and a decision tree for profile-driven debugging. This is the highest-density practical "what to check and why" document ingested so far.

## Overview

The guide treats TPU performance as a roofline problem on top of a tiled-matmul hardware substrate: the MXU has fixed tile sizes (128×128 pre-v6e, 256×256 on v6e), HBM has finite bandwidth, and every kernel is either compute-bound or memory-bandwidth-bound relative to a per-generation critical arithmetic intensity. It then layers a **gotcha catalogue** — dimension alignment, dtype, materialized broadcasts, PyTorch/XLA deferred execution, dynamic shapes, rematerialization, tensor-parallel all-reduce, KV cache decode — with measured impact numbers where available. Separate sections cover training-side levers (selective activation checkpointing, `scan` over layers, mesh design) and inference-side levers (splash/flash attention, int8 quantization, batching strategy, static cache). It closes with XLA flags, a top-down "what's slow?" decision tree, and quick-reference tables for dimension rules and training memory budgets.

The document is prescriptive and opinionated: it names specific thresholds (240 FLOPs/byte on v5e, TP ≤ 8, batch multiple of 1024), names specific tools (xprof, xprof-mcp, `jax.checkpoint_policies.dots_with_no_batch_dims_saveable`, AQT, StaticCache), and provides before/after numbers for two specific optimizations (StaticCache+jit → 8.8× on Llama-2-7B decode; bf16 fix → 17% device-time improvement).

## Key claims

### Hardware and roofline

- TPU v5e peak bf16 throughput is 1.97e14 FLOPs/s; HBM bandwidth 820 GB/s; **critical arithmetic intensity = 240 FLOPs/byte**.
- TPU v5p peak bf16 ≈ 5e14 FLOPs/s; HBM ≈ 2.8 TB/s; critical intensity ≈ 180.
- TPU v6e (Trillium) peak bf16 = 9.1e14 FLOPs/s; HBM 1.6 TB/s; critical intensity ≈ 570.
- The MXU systolic array is 128×128 on v5e and earlier, 256×256 on v6e and later. Dimensions below these tile sizes waste a significant fraction of MXU capacity.
- VMEM / scratchpad is on-chip, MB-scale, and much faster than HBM; the memory hierarchy from fastest to slowest is Registers → VMEM → HBM → Host RAM (PCIe).
- A `[B, D] × [D, F]` bf16 matmul on v5e becomes compute-bound only when `B > 240`; below that the chip waits on HBM.
- Arithmetic intensity = FLOPs ÷ bytes. A kernel is compute-bound iff intensity > peak_FLOPs / peak_bandwidth; otherwise memory-bandwidth-bound.
- On the xprof Roofline tool, dots above the diagonal are compute-bound; far below the diagonal = severely under-utilizing HBM (padding, small shapes, control overhead).
- **ICI roofline:** when tensors are sharded, the crossover shifts. For tensor parallelism across `N` chips on TPU v5e ICI, compute-bound only when the sharded dim > ~8755. The batch dim is **not** what matters for ICI — the sharded dim is.

### Dimension alignment

- Batch size should be multiple of 64 (8 per core) minimum; multiple of 1024 for best efficiency.
- Global hidden/feature dims: multiple of 128; multiple of 256 preferred for v6e.
- **For sharded dims, check the per-chip size, not the global size** — this is the most commonly missed constraint.
- Example: hidden dim 8512 sharded across 16 chips = 532 per chip, which is neither a multiple of 128 nor 256; resharding to 8 chips = 1064 per chip (≈ 1024) yields a **16% throughput improvement**.
- Diagnosis: Memory Viewer in xprof shows padding percentage per tensor; Op Stats shows low FLOP utilization.

### Dtype

- Using fp32 weights for inference is "the most common unforced performance error" — the MXU natively consumes bf16, so fp32 weights trigger an implicit cast on every matmul.
- Converting weights to bf16 typically yields a **17% device-time improvement** on production workloads.
- Quality impact on inference is negligible (weights are constant; no gradient-accumulation noise).
- Diagnosis: HLO Graph Viewer → inspect matmul weight operands → look for `f32` type.

### Fusion and materialized broadcasts

- Some broadcast patterns cannot fuse with their consumer and land in HBM as full-size temporaries — pure bandwidth waste.
- Explicit example: `tf.argmax(tf.add(vector, zero_matrix), axis=0)` — the broadcast cannot fuse with `argmax` and materializes.
- Diagnosis: look for large `bitcast`, `copy`, or `broadcast` ops at the top of HLO Op Stats; use `get_hlo_neighborhood` to isolate.
- Fix: restructure the computation so the consumer appears inside the same fusion as the broadcast.

### PyTorch/XLA deferred execution

- PyTorch/XLA lazy execution accumulates ops until a "materialization trigger" forces a sync and potentially a new compile.
- Triggers: `.item()`, `.cpu()`, `print(tensor)`, data-dependent branches.
- Shape-dependent branches on static shapes (`tensor.shape[0] == 1`) are fine; value-dependent branches (`tensor.sum() > 0`) break fusion.
- Forcing eager / one-op-at-a-time dispatch (where the backend supports it) helps debugging — errors then point at the offending op, not the materialization site.

### Dynamic shapes and StaticCache

- XLA compiles a separate executable per unique input shape. Variable-length batches, growing KV caches, or sequence-length-varying inputs each trigger a recompile.
- Llama-2-7B, 50 tokens, TPU v6e: DynamicCache eager = 130.9 s; StaticCache eager = 88.4 s (1.5×); **StaticCache + `jax.jit` = 14.8 s (8.8×)**.
- Inference-serving rule: pre-compile all (batch_size, seq_len) shapes at startup; warm them all up before accepting requests.

### Rematerialization / activation checkpointing

- Rematerialization saves HBM by recomputing tensors during backward. Too much remat wastes compute.
- If HBM headroom > ~20%, reduce remat.
- Selective activation checkpointing is the production default: checkpoint cheap-to-recompute ops (norms, elementwise), keep expensive ones (attention matmul outputs).
- Full AC: +30–40% compute, ~100% of activations saved.
- **Selective AC: +~2.7% compute, ~70% of activations saved** — the canonical production trade.
- Diagnosis: HLO Op Stats tab → "Time spent on rematerialization"; Memory Profile → HBM headroom.
- JAX knob: `jax.checkpoint_policies.dots_with_no_batch_dims_saveable`. MaxText exposes `remat_policy`. PyTorch/XLA via `torch.distributed.algorithms.checkpoint_wrapper`.

### Tensor parallelism / collectives

- TP within an ICI island is efficient; TP across slices (DCN) degrades severely.
- **Rule: keep TP ≤ 8 within an ICI island.** Going TP=8 (intra-node) → TP=16 (inter-node) caused ~43% degradation in a measured case.
- Diagnosis: `aggregate_xplane_events(..., event_regex="all-reduce")` — count, avg duration, variance.

### KV cache / decode

- Autoregressive decode streams the entire KV cache through HBM every step. With small batch, token/byte ratio is tiny → deeply memory-bandwidth-bound.
- Fixes: increase decode batch size; continuous batching / paged attention; splash/flash attention to keep attention compute in VMEM.
- Profile signature of decode bottleneck: high `all-reduce` time + high `dynamic-slice`/`update-slice` time; overview shows device duty cycle < 60%.

### Scan over layers

- For N identical transformer layers, `jax.lax.scan` (or torchprime's `scan_layers`) reduces XLA compile time from O(N) to O(1).
- Gotcha: scan complicates 2D sharding propagation in the backward pass. OOM during backward with scan enabled → add explicit `shard_as` annotations.

### Attention kernels (inference)

- Standard attention materializes an N×N matrix in HBM; at N=4096 that's 4096² × 2 bytes = **32 MB per layer per request** of pure bandwidth waste.
- Splash Attention is the recommended TPU kernel; FlashAttention-2/3 on GPU; tokamax for GPU-first `dot_product_attention`.

### Quantization

- Lower precision shifts the roofline toward compute-bound.
- bf16: critical batch on v5e is 240.
- int8 weights + bf16 compute: critical batch ~120; 2× memory bandwidth benefit.
- int8 weights + int8 compute: critical batch ~240; FLOPs and bytes both halve.
- Recommended recipe: bf16 first → AQT int8 on weight matrices → Profile-Guided Quantization (PGQ) for per-layer config → validate quality on held-out eval.

### Batching for serving

- Measure device time at batch ∈ {1, 2, 4, 8, 16, 32, 64}.
- Set `max_batch_size` so latency is 30–50% of P50 target.
- Set `batch_timeout_micros = (1 / throughput) × batch_size`.
- Warm up all shapes before accepting traffic — one compilation per shape.
- Once compute-bound, latency scales linearly with batch; larger batches no longer improve per-chip throughput, only increase latency.

### Mesh / sharding design

- Place high-bandwidth collectives (FSDP AllGather / ReduceScatter) on ICI axes.
- Place low-bandwidth collectives (DP gradient sync) on DCN axes.
- Per-chip hidden dim multiple of 128 (256 on v6e).
- Pipeline parallelism only across pod boundaries (DCN).

### Profiling discipline

- Always skip the first 1–3 steps (JIT / XLA compile distorts results); profile after steady state.
- Investigation order: `get_overview` → `get_top_hlo_ops` → `get_op_profile` (fallback when `hlo_stats` is empty, common for inference) → `list_hlo_modules` / `get_hlo_module_content` → `get_hlo_neighborhood` → `get_memory_profile` → `aggregate_xplane_events`.

## Key data points

### Per-generation roofline constants

| Generation | Peak bf16 FLOPs/s | HBM bandwidth | Critical intensity (FLOPs/byte) | MXU tile |
|---|---|---|---|---|
| TPU v5e | 1.97e14 | 820 GB/s | 240 | 128×128 |
| TPU v5p | ~5e14 | ~2.8 TB/s | ~180 | 128×128 |
| TPU v6e (Trillium) | 9.1e14 | 1.6 TB/s | ~570 | 256×256 |

### Dimension alignment rules

| Dimension | Minimum | Preferred |
|---|---|---|
| Batch per core | 8 | 128 |
| Total batch | multiple of 64 | multiple of 1024 |
| Hidden / feature dim (global) | multiple of 128 | multiple of 256 |
| Sharded dim (per chip) | multiple of 128 | multiple of 256 |

### Activation checkpointing trade

| Variant | Compute overhead | Memory saved |
|---|---|---|
| Full AC | +30–40% | ~100% of activations |
| Selective AC | +~2.7% | ~70% of activations |
| None | 0% | 0% |

### Quantization critical batch (v5e)

| Precision | Critical batch | Notes |
|---|---|---|
| bf16 | > 240 | Native TPU format |
| int8 weights, bf16 compute | > ~120 | 2× memory bandwidth benefit |
| int8 weights + int8 compute | > ~240 | FLOPs and bytes both halve |

### Training memory budget per parameter (bf16 mixed, Adam)

| Component | Bytes/param |
|---|---|
| bf16 weights | 2 |
| bf16 gradients | 2 |
| fp32 master weights | 4 |
| Adam m | 4 |
| Adam v | 4 |
| **Total** | **~16** |

### StaticCache impact (Llama-2-7B, 50 tokens, TPU v6e)

| Setup | Wall time | Speedup |
|---|---|---|
| DynamicCache, eager | 130.9 s | 1.0× |
| StaticCache, eager | 88.4 s | 1.5× |
| StaticCache + `jax.jit` | 14.8 s | 8.8× |

### XLA compiler flags worth trying

| Flag | Effect |
|---|---|
| `--xla_tpu_enable_async_collective_fusion=true` | Overlap all-reduce with compute |
| `--xla_enable_async_all_gather=true` | Pipelined weight all-gather (FSDP) |
| `--xla_tpu_megacore_fusion_allow_ags=true` | More aggressive fusion |
| `--xla_tpu_enable_latency_hiding_scheduler=true` | Better op scheduling |

### HLO dumping

| Env var | Effect |
|---|---|
| `XLA_FLAGS="--xla_dump_to=/tmp/hlo_dumps --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*"` | Dump HLO at every pass — prereq for `diff_hlo_stages` |
| `TPU_PREMAPPED_BUFFER_SIZE=8589934592` | 8 GB pre-mapped DMA buffer (raise if RESOURCE_EXHAUSTED on DMA) |
| `unset LD_PRELOAD` | Disable tcmalloc for DLRM-style large-embedding workloads |

### Named rules-of-thumb

| Name | Rule |
|---|---|
| 240 rule (v5e) | bf16 matmul compute-bound when batch > 240 |
| TP ≤ 8 rule | Keep tensor parallelism within one ICI island (measured 43% degradation at TP=16 inter-node) |
| 20% headroom rule | If HBM headroom > 20%, reduce remat |
| 1024 batch rule | Total batch multiple of 1024 for best MXU efficiency |
| 30–50% latency rule | Choose `max_batch_size` so batch latency is 30–50% of P50 target |
| 32 MB attention rule | Standard attention wastes 32 MB per layer per request at N=4096 |

### Diagnosis → cause table (from the decision tree)

| Profile signature | Likely cause | First fix |
|---|---|---|
| Device idle > 30%, host-device gap | Input pipeline bound | Prefetch, parallel reads (grain/tf.data) |
| Device idle > 30%, collective waits | All-reduce / all-gather bound | Async collectives, reduce TP, switch DP→FSDP |
| Matmul FLOP utilization < 40% | Dimension misalignment | Per-chip hidden dim multiple of 128/256 |
| Large bitcast/copy/broadcast at top of Op Stats | Layout mismatch or materialized broadcast | `get_hlo_neighborhood`, refactor fusion boundary |
| High rematerialization time | Over-remat | Reduce remat policy if HBM headroom exists |
| `dynamic-slice` / `update-slice` dominant | KV cache decode (memory-bandwidth) | Bigger batch, splash attention |
| Training OOM | Memory pressure | Selective AC, scan layers, FSDP optimizer-state sharding |

### Investigation tool order (xprof-mcp)

1. `list_runs()`
2. `get_overview(run)` — step time, host vs device split, bottleneck category
3. `get_top_hlo_ops(run)` — hottest ops by time/FLOPs/memory
4. `get_op_profile(run)` — per-program breakdown (fallback when hlo_stats empty)
5. `list_hlo_modules(run)` → `get_hlo_module_content(run, module)`
6. `get_hlo_neighborhood(run, instruction_name)` — BFS around a slow op
7. `get_memory_profile(run)` — peak HBM, fragmentation
8. `aggregate_xplane_events(run, plane_regex="/device:TPU:0")` — kernel stats

## Techniques referenced

- **Roofline analysis** (arithmetic intensity vs peak FLOPs / peak bandwidth; xprof Roofline tool)
- **ICI roofline** (roofline extended to sharded, multi-chip collectives)
- **Rematerialization / activation checkpointing** (full vs selective; `jax.checkpoint`, `jax.checkpoint_policies.dots_with_no_batch_dims_saveable`; MaxText `remat_policy`; PyTorch `checkpoint_wrapper`)
- **Fusion** (XLA op fusion; materialized-broadcast anti-pattern; "fusion boundary" as a debug concept)
- **Dimension alignment** (MXU tile size 128/256; padding; per-chip shape rule)
- **bf16 dtype conversion** (avoiding fp32 → bf16 cast on every matmul)
- **Int8 quantization** (AQT weight-only int8; full int8 compute; PGQ — Profile-Guided Quantization)
- **StaticCache** (`transformers.StaticCache` for fixed-shape KV cache) + `jax.jit`
- **Continuous batching / paged attention** (inference throughput)
- **Flash / Splash Attention** (avoid N×N HBM materialization; splash_attention on TPU; FlashAttention-2/3 on GPU; tokamax `dot_product_attention`)
- **Scan over layers** (`jax.lax.scan`, torchprime `scan_layers`, torchax `ScannedModule`, Flax `nn.scan`)
- **Sharding mesh design** (ICI vs DCN axis placement; FSDP AllGather/ReduceScatter on ICI; DP on DCN; PP across pods)
- **Tensor parallelism within ICI island** (TP ≤ 8)
- **FSDP** (shard optimizer states + params)
- **Async collectives** (`--xla_tpu_enable_async_collective_fusion`, `--xla_enable_async_all_gather`)
- **Latency-hiding scheduler** (`--xla_tpu_enable_latency_hiding_scheduler`)
- **Megacore fusion** (`--xla_tpu_megacore_fusion_allow_ags`)
- **HLO dumping + diffing** (`XLA_FLAGS`, `diff_hlo_stages`, `get_hlo_dump_neighborhood`)
- **Deferred execution** (PyTorch/XLA lazy-tensor materialization triggers)
- **Pre-mapped DMA buffer tuning** (`TPU_PREMAPPED_BUFFER_SIZE`)
- **tcmalloc on/off** for DLRM-style embeddings
- **Warmup / shape pre-compilation** for serving (one compile per shape)
- **Memory Viewer, HLO Graph Viewer, Op Stats, Memory Profile** xprof tabs

## Gaps & caveats

- **Generation coverage is v5e-centric.** v5p and v6e numbers appear only in the roofline table; the 240 rule, ICI 8755 threshold, and TP=16 degradation are explicitly v5e. The guide does not restate these for v6e.
- **"~" everywhere.** v5p (~5e14, ~2.8 TB/s, ~180), v6e critical intensity (~570), int8 critical batches (~120, ~240), TP degradation (~43%), selective AC overhead (+~2.7%) — rough numbers, not measured bounds.
- **Single-workload measurements.** The 17% bf16 win and the 16% reshard win are each one data point from "production workloads" without model, sequence length, or batch specified.
- **Llama-2-7B 8.8× is a specific combo** (eager DynamicCache → StaticCache+jit). How much is StaticCache alone (1.5×) vs how much is jit is given; how it scales with batch, seq len, or model size is not.
- **ICI roofline crossover 8755** is stated without derivation; the guide calls out "not the batch dim" but does not give the general formula.
- **Splash Attention** is recommended for TPU but block sizes, head-dim constraints, seq-len limits, and whether it supports causal masks / sliding windows / GQA are not discussed.
- **AQT / PGQ** gets a code snippet but no quality numbers, no guidance on which layers to quantize first, no failure modes.
- **Remat decision** uses a "20% HBM headroom" trigger but no relation to model size, sequence length, or microbatch.
- **Scan backward + sharding OOM** is flagged with `shard_as` as the fix but no worked example.
- **No profile examples.** The decision tree references xprof tabs ("HLO Op Stats", "Memory Viewer") without screenshots or example numeric signatures.
- **No negative space on XLA flags.** The four recommended flags are not benchmarked; unclear when `--xla_tpu_enable_async_collective_fusion` regresses, or how it interacts with the latency-hiding scheduler.
- **FSDP AllGather on ICI** is the recipe, but how to actually place axes with `jax.sharding.Mesh` vs `torch_xla.distributed.spmd` is omitted.
- **No discussion of pipeline parallelism depth / microbatch trade-offs** beyond "PP across pod boundaries only".
- **No coverage of embedding tables, MoE routing/all-to-all, or speculative decoding** — all increasingly relevant.

## Connections

Concept pages this source informs (to be stubbed / created later):

- `roofline-model` — arithmetic intensity, compute-bound vs memory-bound, critical intensity per generation
- `arithmetic-intensity` — FLOPs ÷ bytes; general definition and TPU thresholds
- `compute-vs-memory-bound` — diagnostic dichotomy central to every optimization decision
- `memory-hierarchy` — Registers / VMEM / HBM / Host; bandwidths and why VMEM matters
- `mxu-tile-sizes` — 128×128 (v5e) vs 256×256 (v6e) and the alignment consequences
- `dimension-alignment` — batch / hidden / sharded dim rules, per-chip constraint
- `dtype-strategy` — bf16 as the default; fp32→bf16 cast anti-pattern
- `rematerialization` — full vs selective AC; policies; HBM-headroom heuristic
- `fusion` — XLA op fusion; materialized-broadcast anti-pattern
- `kv-cache` — decode memory-bandwidth problem; fixes
- `static-cache` — fixed-shape KV cache for JIT-friendly decode
- `flash-attention` — FlashAttention-2/3 family, GPU-primary
- `splash-attention` — TPU-native flash-style attention
- `scan-over-layers` — O(1) compile time for stacked identical layers
- `tensor-parallelism` — TP ≤ 8 ICI rule; DCN degradation
- `fsdp` — mesh placement; optimizer-state sharding; AllGather/ReduceScatter on ICI
- `sharding-mesh-design` — ICI vs DCN axis assignment, PP-across-pods
- `int8-quantization` — AQT, PGQ, weight-only vs full int8
- `profile-guided-quantization` — per-layer precision selection
- `continuous-batching` / `paged-attention` — inference throughput
- `async-collectives` — overlapping all-reduce / all-gather with compute via XLA flags
- `latency-hiding-scheduler` — XLA scheduling pass
- `deferred-execution-xla` — lazy-tensor materialization triggers
- `dynamic-shape-recompiles` — why XLA recompiles per shape; warmup discipline
- `xla-flags-reference` — canonical list of perf-relevant flags
- `hlo-dumping-and-diffing` — XLA_FLAGS dump + diff_hlo_stages workflow
- `xprof-investigation-order` — canonical `get_overview → top_hlo_ops → …` sequence
- `decode-profile-signature` — dynamic-slice/update-slice + duty cycle < 60%
- `serving-warmup` — pre-compile all shapes before traffic
- `training-memory-budget` — ~16 bytes/param for bf16+Adam mixed

## See also

- [xprof-mcp](../codebases/xprof-mcp.md)

## Sources

- `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md`
