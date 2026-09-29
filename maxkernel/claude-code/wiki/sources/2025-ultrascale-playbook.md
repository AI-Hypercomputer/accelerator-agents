---
title: "The Ultra-Scale Playbook: Training LLMs on GPU Clusters"
type: source
tags: [blog, distributed-training, parallelism, zero, fsdp, tensor-parallel, pipeline-parallel, context-parallel, expert-parallel, flash-attention, mixed-precision, fp8, gpu, pytorch, reference]
created: 2026-04-22
updated: 2026-04-22
authors: [Nouamane Tazi, Ferdinand Mom, Haojun Zhao, Phuc Nguyen, Mohamed Mekkouri, Leandro Werra, Thomas Wolf]
affiliation: Hugging Face
published: 2025-02-19
source_url: https://huggingface.co/spaces/nanotron/ultrascale-playbook
raw: raw/sources/2025-ultrascale-playbook.html
assets_dir: raw/assets/ultrascale-playbook/
---

Tazi et al.'s *Ultra-Scale Playbook* is an open long-form guide from Hugging Face that walks LLM training from one GPU to thousands, section by section across the five parallelism axes (data, tensor, pipeline, context, expert) plus kernel-level tricks (FlashAttention, fused kernels, mixed precision, FP8). Claims are backed by 4,000+ benchmark configurations run on GPU clusters with the Nanotron framework. **For this wiki the value is the technique taxonomy and the quantitative claims about compute/memory/communication tradeoffs; the delivery vehicle (CUDA/NCCL/PyTorch-FSDP) is not the TPU reality and must be translated.** This page records the playbook's claims alongside, for each technique, how the TPU + JAX + XLA implementation diverges and which of our ingested codebases is the correct surface to target.

## Overview

The playbook's thesis: at LLM scale there are four bottlenecks — fitting a training step in memory, hitting the target global batch size, maximising throughput, and keeping cost tractable — and each parallelism dimension is a distinct lever on exactly which bottleneck. The authors structure the document as a single training-step memory/compute model (Section 2), followed by each parallelism axis as a transformation of that model (Sections 3–7), followed by a methodology for composing them (Sections 8–9) and then a descent into GPU kernel engineering (Section 10). Appendices cover the collective-operation primer, PyTorch-profiler-based distributed-training profiling, and the closed-form math for compute/communication overlap under each parallelism axis.

![Cheatsheet summarising the full playbook](../../raw/assets/ultrascale-playbook/what_we_learnt_heatmap.svg)
*Figure: the playbook's "what we learnt" heatmap (from the final summary). A config-space sweep of the five parallelism axes × model size × cluster size, colored by throughput. Intended as a decision tool; valid only for the specific GPU/interconnect stack benchmarked.*

## Key claims

Claims are quoted or paraphrased verbatim; each row notes whether the claim transfers to TPU and, if not, what changes.

| # | Claim (GPU/PyTorch context) | Transfers to TPU? | TPU delta / caveat |
|---|---|---|---|
| 1 | Activation memory dominates for long sequence × large batch; selective recomputation of attention activations gives **70% activation memory reduction at 2.7% compute cost** (Korthikanti et al., GPT-3 175B). | **Yes** | Same identity; TPU implements via `jax.checkpoint` / `jax.remat` with a policy argument. The compiler's free-choice rematerializer can also do this without user code changes — a separate optimization surface. |
| 2 | Gradient accumulation is free composition with any parallelism axis but costs linear wall time; decouples `global_batch` from `micro_batch`. | **Yes** | Identity. JAX expresses it via a `scan` over micro-batches inside the pjit'd step. |
| 3 | Overlap grad all-reduce with backward pass via DDP hooks + **bucketed gradients** (25 MB default) yields measurable DP speedup. | **Partial** | The *concept* (overlap collectives with compute) transfers. The *mechanism* does not — on TPU, XLA's async collective scheduler + latency hiding scheduler do overlap automatically given good sharding; there is no user-facing bucket knob. The equivalent tuning surface is `xla_tpu_enable_async_collective_permute`-family flags, mesh axis placement, and `shard_map` layout. |
| 4 | ZeRO-1/2/3 progressively shards optimizer states, gradients, and parameters across DP rank. ZeRO-3 (FSDP) communicates a full all-gather per layer forward + per layer backward. | **Yes, reframed** | On TPU this is SPMD partitioning via `jax.sharding.PartitionSpec` over a device mesh. ZeRO-3 ≈ `PartitionSpec('fsdp', ...)` on the parameter axis with per-layer `all_gather` scheduled by XLA. No separate "ZeRO stage" toggle — it is emergent from how parameters are sharded. |
| 5 | TP column-parallel + row-parallel matmul pair reduces activation memory by `tp_size` but introduces an `all_reduce` inside the transformer block (GEMM-AR pattern). Scales until `tp_size = 8` (intra-node NVLink), collapses beyond. | **Yes, reframed** | On TPU the bandwidth cliff is not "leaves the node" but "leaves the ICI ring." TP usually placed on the fastest ICI axis (typically 8 or 16 chips on v5p, up to 256 along a torus ring on v5p full pod). Scaling limit is topology-dependent, not `tp=8`. |
| 6 | Sequence parallelism replaces the `all_reduce` at LayerNorm/Dropout boundaries with `reduce_scatter + all_gather`, halving the activation memory of the non-matmul ops without extra comm. | **Yes** | Same identity. In JAX it falls out of a different `PartitionSpec` on the sequence axis. |
| 7 | Ring Attention splits Q/K/V along sequence and streams K/V chunks around a ring, overlapping with compute, enabling seq lengths > per-device memory. | **Yes** | Kernel-level TPU analogue is [tokamax's `ring_attention_kernel`](../codebases/tokamax.md) (currently **not reachable** from `tokamax.dot_product_attention` — noted in [log.md](../log.md)). Splash attention covers the non-ring case. |
| 8 | Zig-Zag Ring Attention rebalances causal-mask compute so all ring steps have equal load; naive Ring Attention wastes half the compute on the causal triangle. | **Yes** | Identity, not yet implemented in any TPU kernel we ingested — candidate hypothesis. |
| 9 | Pipeline schedules: AFAB → 1F1B → interleaved 1F1B → Zero-Bubble → DualPipe each cut the bubble size. DualPipe (DeepSeek-V3) achieves near-zero bubble by splitting the backward into `B_input` and `B_weight`. | **Yes, mostly unused on TPU today** | Schedule theory is HW-agnostic. In practice TPU jobs rarely use PP because ICI bandwidth lets TP+FSDP scale further than NVLink does on GPU. PP becomes relevant only across DCN boundaries (multi-pod). |
| 10 | Expert parallelism (MoE) is orthogonal to other axes and dominated by `all_to_all`. DeepSeek-V3 reports "near-zero all-to-all communication overhead" under their setup. | **Yes** | TPU ICI `all_to_all` is bandwidth-rich; DCN `all_to_all` is the cliff. EP axis placement on the ICI mesh matters more than on NVLink clusters. |
| 11 | FlashAttention: tile Q/K/V, keep softmax state in SRAM, recompute on backward; 2–4× speedup, linear memory in seq len. | **Yes, under a different kernel family** | TPU analogue is **splash attention** (Pallas Mosaic-TPU). Tokamax exposes `block_q`, `block_kv`, and backward-pass `block_q_dkv`, `block_kv_dkv`, `block_kv_dkv_compute`, `block_q_dq`, `block_kv_dq`. Backward block sizes are **not surfaced to the autotuner** (log.md finding) — hidden tuning surface. |
| 12 | Mixed precision: keep FP32 master weights; forward/backward in BF16 or FP16 with a loss scaler. BF16 needs no scaler; FP16 does. | **Simplified on TPU** | BF16 is the TPU-native low-precision format; there is no FP16 path, no loss scaler, no scaler-NaN debugging. Master weights are typically FP32 in optimizer state. |
| 13 | FP8 pretraining (DeepSeek-V3 style): **per-tile** quantization (1×128 activations, 128×128 weights) with FP8 MMA, FP32 accumulate. Beats naive tensor-wide quantization. | **Yes, hardware-dependent** | Trillium (v6e) has FP8 MXU support. Tile-granularity matches the MXU's native tile size, so the DeepSeek recipe translates structurally — but the JAX/XLA FP8 API surface (`jax.numpy.float8_e4m3fn`, `jax.lax.dot_general` with FP8 operands) is newer and less mature than H100's TransformerEngine. |
| 14 | Fused kernels (layer_norm, softmax, SwiGLU) pay off only for **memory-bound** ops. Compute-bound ops (matmul) do not benefit. | **Yes** | Identity. On TPU the rule applies to Pallas kernels: `layer_norm`, `gated_linear_unit`, `softmax` are the candidates. **Tokamax's `gated_linear_unit` and `layer_norm` have NO TPU-specific kernel today** (log.md finding) — they fall back to the XLA reference lowering. Direct hypothesis target. |
| 15 | CUDA kernel engineering (memory coalescing, tiling, thread coarsening, control divergence). | **No (different programming model)** | TPU has no thread model, no shared memory in the CUDA sense. Pallas programs on TPU are tile-at-a-time over VMEM with explicit HBM↔VMEM DMAs. The coalescing/divergence sections of the playbook do not transfer; Pallas-Mosaic-TPU has its own tiling/pipelining vocabulary. |

## Key data points

Numbers worth extracting verbatim (all from GPU benchmarks; treat as priors to re-measure on TPU).

### Selective recomputation (attention only), Korthikanti et al. as cited
- 70% activation memory reduction
- 2.7% compute cost
- Basis: GPT-3 175B

### ZeRO-3 / FSDP communication cost (Appendix A3)
Per the playbook's overlap math for ZeRO-3: each transformer block incurs **2 × param_count_per_layer all-gather bytes + 1 × reduce-scatter on the backward**. Overlap is possible iff compute time per layer ≥ comm time per layer, else a bubble appears.

### Pipeline bubble fractions
| Schedule | Bubble fraction (p = pp_size, m = num_microbatches) |
|---|---|
| AFAB | (p − 1) / m |
| 1F1B | (p − 1) / m (same, but lower activation memory) |
| Interleaved 1F1B (v chunks) | (p − 1) / (m · v) |
| Zero-Bubble | ~0 given balanced chunks |
| DualPipe | near-0 by splitting B into B_input and B_weight |

### Mixed precision formats (Fig. `mixedprecision.png`, `sign-mantissa-exponent.svg`)
| Format | Bits (S/E/M) | Dynamic range | Use |
|---|---|---|---|
| FP32 | 1/8/23 | ~1e−38..1e+38 | Master weights, loss |
| BF16 | 1/8/7 | Same range as FP32 | TPU-native training format |
| FP16 | 1/5/10 | ~1e−5..6.5e4 | GPU; needs loss scaler |
| FP8 E4M3 | 1/4/3 | ~1e−3..448 | Forward (activations, weights) |
| FP8 E5M2 | 1/5/2 | ~1e−5..5e4 | Backward (gradients) |

### DeepSeek-V3 FP8 tile sizes
- Activations / inputs: **1×128**
- Weights + scale elements: **128×128**
- Reason: outlier insensitivity at the tile level.

## Techniques referenced

### Single-GPU / single-device memory
- Activation recomputation: full and selective (Korthikanti et al. 2022)
- Gradient accumulation
- Memory profiling via `torch.cuda.memory._record_memory_history()` and its `memory_timeline.html` export. **TPU analogue:** xprof's Memory Profile tool (see [xprof codebase](../codebases/xprof.md), [xprof-mcp](../codebases/xprof-mcp.md) `get_memory_profile`).

### Data parallelism
- PyTorch DDP with backward hooks for overlap
- Gradient bucketing (25 MB default)
- ZeRO-1: optimizer state partitioning (Rajbhandari et al. 2020)
- ZeRO-2: + gradient partitioning
- ZeRO-3 / FSDP: + parameter partitioning

### Tensor parallelism
- Column + row GEMM pair inside transformer block (Megatron-LM style)
- Sequence parallelism over LayerNorm / Dropout boundaries
- Domino overlap (Wang et al. 2024, cited as further reading)

### Context parallelism
- Ring Attention (Liu et al. 2023)
- Zig-Zag Ring Attention (Brandon et al. 2023)
- All-gather vs. all-to-all CP variants

### Pipeline parallelism
- GPipe / AFAB
- 1F1B
- Interleaved 1F1B (virtual pipeline stages)
- Zero-Bubble (Qi et al. 2023)
- DualPipe (DeepSeek-V3)
- Llama-3.1-style schedule

### Expert parallelism
- Top-k MoE routing with all-to-all dispatch (Switch, GShard, Mixtral, DeepSeek-V3)

### Kernel engineering (GPU-specific)
- Memory coalescing, tiling, thread coarsening, control-divergence minimization
- Fused kernels: layer-norm, dropout, SwiGLU
- Triton + `torch.compile` flow
- FlashAttention v1/v2 (Dao et al. 2022)
- Mixed precision training (Micikevicius et al. 2018) — BF16, FP16, FP8
- DeepSeek-V3 FP8 tile scheme

### Appendix A0 — collectives primer
Broadcast, Reduce, AllReduce, Gather, AllGather, Scatter, ReduceScatter, Ring-AllReduce, Barrier, NCCL semantics. **On TPU the same primitives exist via `jax.lax.p*`/`jax.lax.all_gather` over a mesh; the substrate is ICI, not NCCL/NVLink/IB.**

### Appendix A3 — compute/comm overlap math
Closed-form for DP, ZeRO-3, TP, PP. **Formulas transfer; the bandwidth/latency constants must be re-measured for ICI and DCN.**

## GPU/PyTorch ↔ TPU/JAX translation matrix

The centrepiece of this page. For every mechanism the playbook names, the table below records **what changes** when the target is TPU + JAX + XLA. "Same concept" is not "same implementation."

| Axis | GPU/PyTorch (playbook) | TPU / JAX / XLA (this wiki) | Tuning surface that actually matters on TPU |
|---|---|---|---|
| **Device memory hierarchy** | HBM → L2 → SM shared mem → registers | HBM → VMEM → SREG (Pallas view) | HBM sharding; tile sizes in Pallas kernels; layout annotations (`with_sharding_constraint`). |
| **Programming model for kernels** | CUDA / Triton (thread blocks, shared memory, warp-level prims) | Pallas Mosaic-TPU (tile-at-a-time, explicit DMAs, no threads) | Pallas tile/pipeline config; [tokamax](../codebases/tokamax.md) `Config` dataclasses. |
| **Intra-host interconnect** | NVLink (~900 GB/s H100) | ICI (varies by generation; v5p ~4.8 TB/s bisection per chip axis) | Device mesh axis placement; which parallelism rides which ICI axis. |
| **Inter-host interconnect** | IB / RoCE (~25 GB/s H100 node) | DCN (slower than ICI; cross-pod) | Hierarchical mesh (ICI-local + DCN-global); placement of PP/DP across the DCN boundary. |
| **Collective runtime** | NCCL; user-visible buckets, hooks, streams | XLA collective scheduler; async-collective flags; pjit + shard_map | `xla_tpu_enable_async_collective_permute`, latency-hiding scheduler flags; no user-side bucketing knob. |
| **Memory profiling** | `torch.cuda.memory._record_memory_history` → `memory_timeline.html` | xprof Memory Profile tool; HLO dump analysis | [xprof-mcp](../codebases/xprof-mcp.md) `get_memory_profile`, `get_hlo_dump`, `list_hlo_modules`. |
| **Trace profiling** | Nsight Systems / `torch.profiler` → Chrome trace | xprof traces (`.xplane.pb`) → TensorBoard plugin or xprof-mcp | `get_overview`, `get_op_profile`, `get_top_hlo_ops`, `list_xplane_events`. |
| **Kernel profiling** | Nsight Compute (`ncu`), PMU counters | xprof op profile; Mosaic-TPU kernel-level traces | `get_op_profile`, `aggregate_xplane_events`. |
| **Activation recomputation** | `torch.utils.checkpoint` at nn.Module boundaries | `jax.checkpoint` / `jax.remat` with `policy=`; XLA free-choice rematerializer (compiler-level) | `jax.checkpoint_policies.nothing_saveable`, `save_only_these`, `dots_with_no_batch_dims_saveable`; `--xla_tpu_rwb_fusion`, remat-related XLA flags. |
| **Gradient accumulation** | Python loop + `zero_grad` gating | `jax.lax.scan` over micro-batches inside pjit'd step | Micro-batch count vs. memory tradeoff; scan-unrolled vs. rolled. |
| **DP overlap** | DDP hooks + 25 MB buckets | XLA async collective scheduler — implicit | Sharding annotations; mesh axis order. |
| **ZeRO-1/2/3 / FSDP** | `torch.distributed.fsdp`, wrap policies, `sharding_strategy` | SPMD via `PartitionSpec` over `fsdp` axis; emergent from param sharding choice | Which axes are in the mesh; which tensor dims are sharded over which axis. |
| **Tensor parallelism** | Megatron-style column/row split; `tp=8` cliff at NVLink edge | `PartitionSpec('tp', None)` on weights; placed on the fastest ICI ring | Which ICI axis TP rides; mesh shape; `with_sharding_constraint` at activation boundaries. |
| **Sequence parallelism** | Seq-axis sharding at LN/Dropout with RS+AG replacing AR | `PartitionSpec(..., 'sp')` on activations; XLA lowers to ICI RS+AG | Same mesh concerns; interaction with TP axis. |
| **Context parallelism / Ring Attention** | Custom FlashAttention-based ring kernel; SDPA override | Pallas splash attention (non-ring) **or** [tokamax `ring_attention_kernel`](../codebases/tokamax.md) (exists, not reachable from public API) | Block sizes (Q/KV); which ring direction over which ICI axis; Zig-Zag balancing is an open implementation hole on TPU. |
| **Pipeline parallelism** | NeMo / Megatron PP with 1F1B, interleaved, zero-bubble | Rarely used on single TPU pod; multi-pod PP across DCN possible | PP chunk size; bubble math (`(p−1)/m`); DCN bandwidth vs. per-layer compute. |
| **Expert parallelism / MoE** | Megatron-MoE / Tutel all-to-all dispatch | `jax.lax.all_to_all` over EP mesh axis; Pallas ragged_dot for expert compute | EP axis on ICI (not DCN); [tokamax `ragged_dot`](../codebases/tokamax.md) as the compute kernel. |
| **Attention kernel** | FlashAttention v2/v3 (CUTLASS/CUDA) | Splash attention (Pallas Mosaic-TPU) | `block_q`, `block_kv`, **backward block sizes not autotuned** (hidden surface per log.md). |
| **Mixed precision** | BF16 or FP16 + loss scaler; autocast regions | BF16 everywhere, no scaler; optimizer state FP32 | Numerics accumulators; `jax.default_matmul_precision`. |
| **FP8** | TransformerEngine; per-tensor or per-block scaling; H100 MMA | `jax.numpy.float8_e4m3fn` / `float8_e5m2`; v6e MXU FP8; DeepSeek-V3 1×128 / 128×128 tile scheme applicable | Scale granularity; which ops opt in; calibration cadence. |
| **Fused kernels** | Triton + `torch.compile` codegen | Pallas kernel library; XLA fusion passes | `--xla_tpu_enable_experimental_fusion` flag family; hand-written Pallas (tokamax path) vs. XLA fusion. |
| **Collectives API** | `torch.distributed.*` + NCCL | `jax.lax.p*`, `jax.lax.all_gather`, `jax.lax.psum`, `shard_map` | Async-collective XLA flags; mesh-axis-to-collective mapping. |

## Section-by-section notes with images

Each subsection below quotes what the playbook says, then states the TPU delta. Images referenced from `../../raw/assets/ultrascale-playbook/`.

### First steps: memory on one device

![Memory breakdown for a training step](../../raw/assets/ultrascale-playbook/predict_memory_tool.png)

Claim: training-step memory = weights + grads + optimizer states + activations + transient buffers. At Adam + FP32 master weights in BF16 training: `4 × P` weights, `4 × P` grads, `8 × P` optimizer state = 16 bytes × parameter count.

**TPU delta:** identical identity. The tool the playbook links is GPU-specific. On TPU, the same decomposition is read off the HLO + xprof Memory Profile.

### Activation recomputation

![Gradient accumulation diagram](../../raw/assets/ultrascale-playbook/gradaccumulation_diag.png)

Claim: full recomputation (recompute every activation) has ~30% compute overhead; **selective** (recompute attention only) gives 70% activation-memory reduction at 2.7% compute per Korthikanti et al.

**TPU delta:** `jax.remat` with a policy is the direct analogue. Additionally, XLA has a **free-choice rematerializer** that can do this without user code. These two compete/complement and the choice is a hypothesis surface.

### Data parallelism — overlap, buckets, ZeRO

![DP all-reduce schedule — no overlap](../../raw/assets/ultrascale-playbook/dp_overlap1.svg)
![DP with overlap via backward hooks](../../raw/assets/ultrascale-playbook/dp_overlap2.svg)
![DP with bucketed overlap](../../raw/assets/ultrascale-playbook/dp_overlap3.svg)

Claim: overlap brings DP cost from serial all-reduce to ~0 given comm < compute per layer; bucketing (25 MB default) reduces NCCL call overhead.

**TPU delta:** bucketing is a NCCL-specific concern. XLA schedules collectives asynchronously at the HLO level; the user-visible surface is sharding annotations + mesh shape, not bucket size. Overlap is emergent.

![ZeRO memory shrinkage per stage](../../raw/assets/ultrascale-playbook/zero_memory.svg)
![ZeRO-3 forward all-gather](../../raw/assets/ultrascale-playbook/dp_zero3_fwd.svg)
![ZeRO-3 backward reduce-scatter](../../raw/assets/ultrascale-playbook/dp_zero3_bwd.svg)

Claim: ZeRO-3 pays 1 all-gather per layer forward + 1 all-gather + 1 reduce-scatter per layer backward. Overlappable when compute per layer ≥ comm per layer.

**TPU delta:** this is the standard FSDP-style sharding emergent from `PartitionSpec` over an fsdp axis. XLA schedules the per-layer gathers. No "ZeRO stage" dial — the shape of the PartitionSpec *is* the stage.

### Tensor parallelism + sequence parallelism

![TP inside a full transformer block](../../raw/assets/ultrascale-playbook/tp_full_diagram.png)
![TP + SP region transitions](../../raw/assets/ultrascale-playbook/tp_sp_diagram.png)
![TP + SP detailed op view](../../raw/assets/ultrascale-playbook/tp_sp_diagram_zoomed.png)
![TP scaling, limited by NVLink](../../raw/assets/ultrascale-playbook/tp_scaling.svg)

Claim: TP caps at `tp=8` (intra-node NVLink); beyond that the all-reduce spills to IB and collapses throughput.

**TPU delta:** the cap is **topology-dependent**, not a fixed number. On a v5p torus, TP placed along a 16-chip ring stays fast; placing TP across a slice boundary is the cliff. Mesh design, not `tp` value.

### Context parallelism — Ring & Zig-Zag

![Ring attention causal mask imbalance](../../raw/assets/ultrascale-playbook/cp_attnmask.svg)
![Zig-Zag mask — balanced compute](../../raw/assets/ultrascale-playbook/cp_zigzagmask.svg)
![Ring attention animation](../../raw/assets/ultrascale-playbook/ring-attention.gif)
![CP with all-gather overlap](../../raw/assets/ultrascale-playbook/cp_overlap_allgather.svg)
![CP with all-to-all overlap](../../raw/assets/ultrascale-playbook/cp_overlap_all2all.svg)

Claim: naive Ring Attention with causal masking wastes ~50% of compute on the lower triangle; Zig-Zag rebalances.

**TPU delta:** splash attention (Pallas Mosaic-TPU, [tokamax](../codebases/tokamax.md)) is the default TPU attention kernel — non-ring. Ring Attention exists as `tokamax/_src/ops/experimental/...ring_attention_kernel` but is not wired into the public `dot_product_attention` dispatch (log.md finding). **Zig-Zag Ring Attention on TPU is a hole — no ingested codebase implements it.**

### Pipeline parallelism — schedules

![AFAB schedule](../../raw/assets/ultrascale-playbook/pp_afab.svg)
![1F1B schedule](../../raw/assets/ultrascale-playbook/pp_1f1b.svg)
![Interleaved 1F1B](../../raw/assets/ultrascale-playbook/pp_1f1b_interleaved.svg)
![Zero-Bubble schedule](../../raw/assets/ultrascale-playbook/pp_zerobubble_ppschedule.png)
![DualPipe (DeepSeek-V3)](../../raw/assets/ultrascale-playbook/pp_zerobubble_dualpipe.png)

**TPU delta:** schedule theory is HW-agnostic. In practice, TPU pods don't hit the cross-node bandwidth cliff that forces PP on GPU clusters — FSDP+TP over ICI scales further than NVLink. PP becomes relevant only across DCN (multi-pod). Bubble formulas transfer unchanged.

### Expert parallelism

![MoE layer structure](../../raw/assets/ultrascale-playbook/ep_moe.png)
![EP schema](../../raw/assets/ultrascale-playbook/ep_schema.png)

**TPU delta:** the compute kernel on TPU is [tokamax `ragged_dot`](../codebases/tokamax.md); all-to-all dispatch/combine is `jax.lax.all_to_all`. Bandwidth-rich over ICI, cliff at DCN. EP should ride an ICI axis.

### 5D composition

![5D composition full diagram](../../raw/assets/ultrascale-playbook/5d_full.svg)
![5D parallelism 8B memory usage](../../raw/assets/ultrascale-playbook/5Dparallelism_8Bmemoryusage.svg)

Claim: DP × TP × PP × CP × EP is the full space; composition rules are constrained by mesh shape and collective overlap.

**TPU delta:** on TPU the mesh is literally the device grid (e.g. a v5p 4×4×4 slice is a 3-axis mesh). The five parallelism axes are partitioned over a subset of physical axes, and the axis assignment is the core design choice. `jax.experimental.mesh_utils.create_device_mesh` + `PartitionSpec` is the full interface.

### Finding the best training configuration

Claim: the search procedure is (1) fit in memory, (2) hit target GBS, (3) maximise throughput. Benchmark thousands of configs; record heatmaps like the lead figure.

**TPU delta:** the procedure is identical — it **is** the autoresearch loop this wiki is built around. See [autoresearch codebase](../codebases/autoresearch.md) for the methodology substrate. The constants in step (3) are ICI/DCN bandwidths, VMEM capacity, and XLA fusion boundaries — not NVLink/IB.

### GPU kernel primer (Section 10)

![GPU memory hierarchy](../../raw/assets/ultrascale-playbook/diving_primergpu.svg)
![GPU SM architecture](../../raw/assets/ultrascale-playbook/diving_primergpu2.svg)
![Memory coalescing](../../raw/assets/ultrascale-playbook/memorycoalescing.png)
![Tiling](../../raw/assets/ultrascale-playbook/tiling.png)
![Thread coarsening](../../raw/assets/ultrascale-playbook/threadcoarsening.png)

**TPU delta: this entire section does not transfer.** TPU has no thread/warp model. Pallas Mosaic-TPU programs operate on tiles in VMEM with explicit DMAs; the analogous tuning vocabulary is block_q/block_kv (per kernel), pipeline depth, and VMEM layout — documented per-kernel in tokamax.

### FlashAttention

![FlashAttention tiling](../../raw/assets/ultrascale-playbook/flashattn.png)
![FlashAttention backward](../../raw/assets/ultrascale-playbook/flashattn2.png)

**TPU delta:** splash attention plays the same role. The autotuning surface on TPU is the splash `Config` with explicit block sizes. Forward blocks are autotuned in tokamax; backward blocks are **not** (log.md finding).

### Mixed precision / FP8

![Mixed precision regions](../../raw/assets/ultrascale-playbook/mixedprecision.png)
![Format bit layouts](../../raw/assets/ultrascale-playbook/sign-mantissa-exponent.svg)
![FP8 flow diagram](../../raw/assets/ultrascale-playbook/fp8_diagram.png)

**TPU delta:** BF16 is the default — no loss scaler. FP8 on v6e / Trillium MXU is available; the DeepSeek-V3 1×128 / 128×128 tile scheme matches the MXU's native tile and is a natural starting point.

### Appendix A0 — collectives primer

![Broadcast](../../raw/assets/ultrascale-playbook/a0_broadcast.png)
![AllReduce](../../raw/assets/ultrascale-playbook/a0_reduce_allreduce.png)
![AllGather](../../raw/assets/ultrascale-playbook/a0_all_gather.gif)
![ReduceScatter](../../raw/assets/ultrascale-playbook/a0_reduce_scatter.gif)

Same primitives, different substrate. ICI replaces NCCL/NVLink.

## Gaps & caveats

- **Every throughput number in the playbook is from a GPU cluster.** Treat as qualitative priors, not quantitative benchmarks, when transplanting to TPU.
- **Nanotron, DDP hooks, NCCL buckets, `torch.compile`+Triton** are the delivery mechanisms throughout — none apply on TPU. Extract the algorithmic claim, drop the mechanism.
- **GPU kernel engineering section (Section 10 "Diving into the GPUs")** does not transfer — different programming model. Pallas Mosaic-TPU is the analogue, but its tuning vocabulary is different.
- **Nothing in the playbook addresses:** ICI torus topology, VMEM sizing, splash attention backward block tuning, XLA compiler flags, HLO-level rematerialization policies, device-mesh shape optimization. These are the TPU-side surfaces that drive actual throughput.
- **Hypothesis candidates** surfaced by this source but **not yet filed as hypothesis pages** (no `model:` page exists yet — schema requires one):
  1. Selective activation recomputation (attention-only) via `jax.checkpoint_policies` — expected 10–30% memory reduction, small compute cost, on the first model we ingest.
  2. Wire tokamax `ring_attention_kernel` through `dot_product_attention` dispatch — kernel exists, API gap only. **PARTIAL RETRACTION (2026-04-23)** per [analyses/2026-04-23-pallas-kernel-directory.md](../analyses/2026-04-23-pallas-kernel-directory.md): **three public reference impls exist** — [maxdiffusion](../codebases/maxdiffusion.md)'s splash-integrated `ring_attention_kernel.py` (2026-04-16, recommended pattern), [ringattention](../codebases/ringattention.md)'s canonical paper companion (haoliuhl/ringattention, from-scratch flash inner kernel), and [ejkernel](../codebases/ejkernel.md)'s splash-wrapper. Reduces this hypothesis from "open research" to "port + adapt; pick among three patterns".
  3. Zig-Zag Ring Attention on TPU — no implementation found; algorithmic port from Brandon et al. 2023. **RETRACTION STANDS (2026-04-23)** — confirmed absent from every repo surveyed in the pallas-kernel directory, including haoliuhl canonical (straight unidirectional, `below_or_on_diag` causal check only). Remains open.
  4. Implement a TPU-native Pallas kernel for `gated_linear_unit` and `layer_norm` in tokamax — currently fall back to XLA reference. **PARTIAL RETRACTION (2026-04-23)**:
     - **RMSNorm/LayerNorm**: external absence from maxtext + tpu-inference + axlearn + upstream is evidence that **XLA-fusion is sufficient**. [Gemma 4 exp 33](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp33-pallas-rmsnorm-rejected.md)'s −8.1% empirical result agrees with the ecosystem. **Don't build.**
     - **Fused GLU**: [alphafold3 @ v3.0.1](../codebases/alphafold3.md) provides a production Pallas fused-GLU reference (GPU). Porting to Mosaic-TPU is feasible; needs HLO-level validation that XLA isn't already fusing the equivalent pattern (same lesson as norm).
  5. Tile-scaled FP8 (DeepSeek-V3 1×128 / 128×128) via `jax.numpy.float8_*` on v6e.
  6. Expose tokamax splash-attention **backward** block sizes to the autotuner.

  These will be filed as [hypotheses/*.md](../hypotheses/) when the first model page exists.

## Connections

Pages this source informs and which have been updated to list it in their `## Sources`:

- [codebases/tokamax.md](../codebases/tokamax.md) — ring attention kernel, splash attention block autotuning, missing TPU kernels for GLU/LayerNorm, ragged_dot for MoE
- [codebases/torchax.md](../codebases/torchax.md) — FSDP / SPMD partitioning via the JAX-backend path; `gradient_checkpoint` interop
- [codebases/scaling-book.md](../codebases/scaling-book.md) — parallel reference; overlapping claims to be cross-checked chapter by chapter in Wave 3
- [codebases/autoresearch.md](../codebases/autoresearch.md) — playbook Section 8 ("Finding the best training configuration") is a direct match for the autoresearch-style search procedure

Existing concepts pages updated to cite this source (see [concepts/](../concepts/)):

- [concepts/rematerialization.md](../concepts/rematerialization.md) — activation recomputation
- [concepts/fsdp.md](../concepts/fsdp.md) — ZeRO-3 / FSDP sharding
- [concepts/sharding.md](../concepts/sharding.md) — GSPMD partitioning
- [concepts/tensor-parallelism.md](../concepts/tensor-parallelism.md)
- [concepts/flash-attention.md](../concepts/flash-attention.md)
- [concepts/splash-attention.md](../concepts/splash-attention.md) — TPU analogue
- [concepts/dtype-strategy.md](../concepts/dtype-strategy.md) — bf16 / fp8 mixed precision
- [concepts/async-collectives.md](../concepts/async-collectives.md) — XLA collective/compute overlap

New concept stubs created from this source:

- [concepts/ring-attention.md](../concepts/ring-attention.md)
- [concepts/context-parallelism.md](../concepts/context-parallelism.md)
- [concepts/sequence-parallelism.md](../concepts/sequence-parallelism.md)
- [concepts/pipeline-parallelism.md](../concepts/pipeline-parallelism.md)
- [concepts/expert-parallelism.md](../concepts/expert-parallelism.md)

## Full image index

All 90 figures referenced on the playbook page, saved to `raw/assets/ultrascale-playbook/`.

| Section | File | Note |
|---|---|---|
| Hero / summary | `what_we_learnt_heatmap.svg` | Final cheatsheet heatmap |
| Hero / summary | `conclusion_llama3_parallelism.png` | Llama-3 parallelism choice |
| Hero / summary | `predict_memory_tool.png` | HF memory predictor tool |
| First steps | `gradaccumulation_diag.png` | Grad accumulation diagram |
| First steps | `profile_trace_annotated.png` | PyTorch profiler trace |
| DP | `dp_diagram.png` | Baseline DP diagram |
| DP | `dp_overlap1.svg`, `dp_overlap2.svg`, `dp_overlap3.svg` | Overlap progression |
| DP | `dp_scaling.svg` | DP scaling |
| DP | `dp_ourjourney_memoryusage.svg` | DP memory journey |
| ZeRO | `zero_memory.svg` | Stage-by-stage memory shrink |
| ZeRO | `dp_zero1.gif`, `dp_zero1_overlap.svg` | ZeRO-1 |
| ZeRO | `dp_zero2.gif`, `dp_zero2_overlap.svg` | ZeRO-2 |
| ZeRO | `dp_zero3_fwd.svg`, `dp_zero3_bwd.svg`, `dp_zero3_overlap.svg` | ZeRO-3 |
| ZeRO | `zero3_memoryusage.svg` | ZeRO-3 memory |
| TP | `tp_diagram.svg`, `tp_diagram2.png`, `tp_diagram3.png`, `tp_diagram4.png`, `tp_full_diagram.png` | TP progression |
| TP | `tp_overlap.svg`, `tp_scaling.svg`, `tp_memoryusage.svg` | TP overlap + scaling |
| TP+SP | `tp_sp_diagram.png`, `tp_sp_diagram_zoomed.png`, `tp_sp_overlap.svg`, `tp_sp_scaling.svg`, `tp_sp_memoryusage.svg` | TP + SP |
| CP | `cp_attnmask.svg`, `cp_zigzagmask.svg` | Masks |
| CP | `cp_overlap_allgather.svg`, `cp_overlap_all2all.svg` | Overlap variants |
| CP | `ring-attention.gif` | Ring attention animation |
| CP | `cp_memoryusage.svg`, `cp_8Bmemoryusage.svg` | CP memory |
| PP | `pp_afab.svg`, `pp_afab2.svg` | AFAB |
| PP | `pp_1f1b.svg`, `pp_1f1b_interleaved.svg`, `pp_1f1b_scaling.png` | 1F1B family |
| PP | `pp_bubblesize.png`, `pp_comm_bandwidth.svg`, `pp_memoryusage.svg` | PP costs |
| PP | `pp_llama3.1_schedule.png` | Llama 3.1 schedule |
| PP | `pp_zerobubble_ppschedule.png`, `pp_zerobubble_compgraph.png`, `pp_zerobubble_dualpipe.png` | Zero-bubble + DualPipe |
| EP / MoE | `ep_moe.png`, `ep_schema.png` | MoE + EP |
| 5D | `5d_full.svg`, `5Dparallelism_8Bmemoryusage.svg`, `5d_nutshell_tp_sp.svg`, `5d_nutshell_cp.svg`, `5d_nutshell_ep.svg` | 5D composition |
| GPU primer | `diving_primergpu.svg`, `diving_primergpu2.svg` | GPU memory hierarchy — GPU-only |
| GPU primer | `memorycoalescing.png` … `memorycoalescing5.png` (5 files) | Coalescing — GPU-only |
| GPU primer | `tiling.png`, `threadcoarsening.png` | Tiling / thread coarsening — GPU-only |
| GPU primer | `fused_kernels1.png`, `fused_kernels2.png` | Fused kernels |
| GPU primer | `torch-compile-triton.png`, `torch-compile-triton-kernel.png` | `torch.compile` + Triton — GPU-only |
| Attention | `flashattn.png`, `flashattn2.png` | FlashAttention |
| Mixed prec | `mixedprecision.png`, `mixedprecision_2.png`, `sign-mantissa-exponent.svg` | BF16/FP16 |
| Mixed prec | `fp8_diagram.png` | FP8 |
| Appendix A0 | `a0_general.png`, `a0_broadcast.png`, `a0_reduce_allreduce.png`, `a0_gather_allgather.png`, `a0_scatter_reducescatter.png`, `a0_barrier.png` | Collectives diagrams |
| Appendix A0 | `a0_all_gather.gif`, `a0_reduce_scatter.gif` | Collectives animations |
| Appendix A1 | `a1_profile_trace.png`, `a1_kernels.png`, `a1_ncu.png` | PyTorch profiler / ncu — GPU-only |
| Misc | `placeholder.png` | Placeholder asset |

## See also

- [codebases/tokamax.md](../codebases/tokamax.md)
- [codebases/torchax.md](../codebases/torchax.md)
- [codebases/scaling-book.md](../codebases/scaling-book.md)
- [codebases/autoresearch.md](../codebases/autoresearch.md)
- [codebases/xprof.md](../codebases/xprof.md)
- [codebases/xprof-mcp.md](../codebases/xprof-mcp.md)

## Sources

- [Ultra-Scale Playbook (live page)](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- [Raw HTML capture](../../raw/sources/2025-ultrascale-playbook.html) @ 2026-04-22
- [Figure assets (90 files)](../../raw/assets/ultrascale-playbook/)
- Citation: Tazi, Mom, Zhao, Nguyen, Mekkouri, Werra, Wolf. "The Ultra-Scale Playbook: Training LLMs on GPU Clusters." Hugging Face, 2025-02-19.
