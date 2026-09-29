---
title: "jax (jax-ml/jax)"
type: codebase
tags: [jax, xla, pallas, mosaic, tpu, jit, sharding, profiler, roofline, scan, rematerialization, compilation-cache]
repo: jax-ml/jax
commit: feb5ba05855a9e1fdf6d3095b6f5b87614bdbbc7
created: 2026-04-23
updated: 2026-04-23
---

> **Grounded front door → [jax overview](jax/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

The JAX library itself — the ground-truth for every transformation, kernel, and TPU surface the rest of this wiki builds on. Every other ingested codebase ([torchax](torchax.md), [tokamax](tokamax.md), [pallas-forge](pallas-forge.md), [xprof](xprof.md), [xprof-mcp](xprof-mcp.md), [jax-huggingface](jax-huggingface.md), [scaling-book](scaling-book.md)) imports or lowers through this repo. It also ships **first-party reference TPU Pallas kernels** (`flash_attention`, `splash_attention`, `paged_attention`, `ragged_paged_attention`, `megablox`, `matmul`, `all_gather`, `threefry`) that the wiki's Pallas-kernel analysis treats as canonical. **Commit `feb5ba05` was `HEAD` on 2026-04-23** — a bleeding-edge pin; rerun `git submodule update --remote` to bump.

## Overview

JAX is far too large to fully-ingest; this page indexes the surfaces this wiki actually uses for TPU performance work. Four broad buckets:

1. **Transformations & compilation** — `jax.jit`, `jax.pjit`, `jax.vmap`, `jax.grad`, `jax.lax.scan`, `jax.ad_checkpoint` (remat), `jax.custom_vjp`, `jax.custom_jvp`. The "program transformations" half of JAX's pitch.
2. **Parallelism & sharding** — `jax.sharding.{Mesh, NamedSharding, PartitionSpec}`, `jax.experimental.mesh_utils`, `jax.experimental.shard_map`, `jax.experimental.pjit`, `jax.experimental.custom_partitioning`, `jax.experimental.layout`. The "at scale" half.
3. **Kernels & lowering** — `jax.experimental.pallas` (DSL), `jax._src.pallas.mosaic` (TPU lowering), `jax.experimental.mosaic` (Mosaic dialects), plus the **reference TPU Pallas ops tree** at `jax.experimental.pallas.ops.tpu.*`.
4. **Profiling & analysis** — `jax.profiler` (trace API), `jax.experimental.roofline` (built-in roofline analyzer), `jax.collect_profile`, `jax.experimental.compilation_cache` (persistent compile cache), `jax.experimental.serialize_executable`, `jax.experimental.source_mapper`.

For TPU performance hypotheses, the most-touched surfaces are: **the reference TPU Pallas ops tree** (splash, ragged-paged, megablox), **sharding + mesh + shard_map**, **`jax.profiler`**, and **`jax.experimental.roofline`**. The compilation cache and source mapper get cited for compile-time mitigation hypotheses.

## Architecture

```
jax/
  jax/                             ← Python public API (re-exports from _src)
    jit.py  pjit.py  vmap.py       ← transformations
    profiler.py                    ← jax.profiler public surface
    sharding.py                    ← Mesh, NamedSharding, PartitionSpec re-exports
    ad_checkpoint.py               ← remat (`jax.checkpoint`, `policy`)
    collect_profile.py             ← CLI-level profile collection entry
    lax/                           ← primitive ops (scan, while_loop, reduce, dot_general, ...)
    numpy/  scipy/  nn/  random/   ← array-API surfaces
    experimental/
      pallas/                      ← Pallas DSL + ops
        __init__.py                ← top-level re-exports (BlockSpec, GridSpec, pallas_call, ...)
        fuser.py                   ← kernel-fusion helpers
        mosaic_gpu.py              ← GPU backend hook
        tpu.py                     ← TPU backend hook
        tpu_sc.py                  ← TPU SparseCore backend hook
        triton.py                  ← Triton (GPU) lowering hook
        ops/                       ← reference kernel implementations
          tpu/                     ← the crown jewel for this wiki
            flash_attention.py
            splash_attention/{splash_attention_kernel,_mask,_mask_info}.py
            paged_attention/{paged_attention_kernel,util,quantization_utils}.py
            ragged_paged_attention/{kernel,tuned_block_sizes}.py
            megablox/{gmm,ops,common}.py              ← MoE grouped matmul
            matmul.py              ← reference tiled matmul
            all_gather.py          ← Pallas-implemented collective
            example_kernel.py      ← teaching kernel
            random/{threefry,philox}.py                ← PRNG kernels
          gpu/                     ← GPU counterparts (Triton / Mosaic-GPU)
      mosaic/                      ← Mosaic dialect front-end
        dialects.py                ← MLIR dialect registration
        gpu/                       ← Mosaic-GPU internals (dialect_lowering, fragmented_array, ...)
      roofline/
        __init__.py                ← roofline, register_roofline, roofline_and_grad, ...
        roofline.py                ← core implementation
        rooflines.py               ← per-op roofline rules
      mesh_utils.py                ← create_device_mesh, create_hybrid_device_mesh
      shard_map.py                 ← shard_map transform
      pjit.py                      ← legacy pjit surface
      custom_partitioning.py       ← user-defined SPMD rules
      layout.py                    ← logical→physical array layout control
      compilation_cache.py         ← persistent XLA compile cache
      serialize_executable.py      ← AOT-compiled executable save/load
      source_mapper/               ← HLO↔Python source mapping
      ...
    _src/                          ← private implementation (BUILD-exported by Bazel)
      pallas/
        mosaic/                    ← TPU Pallas lowering (core.py, lowering.py, pipeline.py, ...)
        mosaic_gpu/                ← GPU Pallas lowering
        triton/                    ← Triton lowering
        cost_estimate.py           ← cost-model hook for Pallas kernels
        primitives.py              ← Pallas primitive set
      ...
  jaxlib/                          ← C++ bindings (shipped as a separate wheel)
  tests/                           ← integration tests
  docs/                            ← Sphinx docs (ignored for this wiki)
  benchmarks/                      ← micro-benchmarks
  third_party/                     ← vendored deps
  cloud_tpu_colabs/                ← TPU setup notebooks
```

The **public/private split** is load-bearing: anything under `jax/` is a re-export surface with stability guarantees (up to version), everything under `jax/_src/` is implementation. When a hypothesis wants to hook into a specific lowering pass, the line it is editing is under `_src/`; when it wants to *call* the surface, the import lives at `jax.*` or `jax.experimental.*`.

## Key abstractions

| Abstraction | Public path | What it is | Notes |
|---|---|---|---|
| `jax.jit(fn, *, static_argnums, static_argnames, donate_argnums, in_shardings, out_shardings)` | `jax.jit` | Just-in-time compile a pure function through XLA | `out_shardings` / `in_shardings` are the way to pin layouts without `shard_map`. `donate_argnums` is the free-weights knob. |
| `jax.lax.scan` | `jax.lax.scan` | Loop-carry scan; compile-time folded | The **scan-over-layers** hypothesis lever. One compile, N-step unroll at runtime. |
| `jax.ad_checkpoint.checkpoint` (alias `jax.checkpoint`) | `jax.ad_checkpoint` | Rematerialization | `policy=` accepts names of activations to save (vs recompute); critical for memory-bound training. |
| `jax.profiler.{start_trace, stop_trace, trace, StepTraceAnnotation, TraceAnnotation, ProfileOptions, ProfileData}` | `jax.profiler` | Capture XProf traces | See [jax-trace](../concepts/jax-trace.md) for wiki-level treatment. |
| `jax.experimental.roofline.{roofline, roofline_and_grad, register_roofline}` | `jax.experimental.roofline` | **Built-in roofline analyzer** — static FLOPs + bytes accounting per jaxpr | First-party alternative to `pallas-forge`'s `roofline_chart`; works on *any* JAX function, not just Pallas kernels. |
| `jax.sharding.{Mesh, NamedSharding, PartitionSpec}` | `jax.sharding` | gSPMD primitives | `Mesh` shape + axis-names, `NamedSharding(mesh, PartitionSpec('data', 'model'))`. |
| `jax.experimental.mesh_utils.{create_device_mesh, create_hybrid_device_mesh}` | `jax.experimental.mesh_utils` | Helpers that pick good ICI-aware mesh shapes | `create_device_mesh((4,2))` on v6e-8 returns the right physical layout. |
| `jax.experimental.shard_map.shard_map` | `jax.experimental.shard_map` | Per-device explicit SPMD | Inverse of implicit-gSPMD `pjit`; used when you need to write per-device collectives by hand. |
| `jax.experimental.custom_partitioning` | `jax.experimental.custom_partitioning` | User-defined SPMD lowering for custom ops | Used when default partitioner gets a sharding wrong. |
| `jax.experimental.layout.Layout` / `DeviceLocalLayout` | `jax.experimental.layout` | Logical→physical layout declarations on inputs/outputs of `jax.jit` | Maps to XLA's `Layout` / `TileAssignment`. Lever for e.g. last-minor-dim choice. |
| `jax.experimental.pallas.{pallas_call, BlockSpec, GridSpec, Blocked, ...}` | `jax.experimental.pallas` | Pallas DSL entrypoints | `pallas_call` is the kernel launch; `BlockSpec` describes input-to-kernel block-windowing. |
| `jax.experimental.pallas.ops.tpu.*` | ↑ | Reference TPU kernels (see next section) | First-party reference implementations for flash, splash, paged, ragged-paged, megablox, matmul, all_gather, threefry. |
| `jax.experimental.mosaic` | `jax.experimental.mosaic` | Mosaic dialect front-end | The MLIR dialect that Pallas lowers to on TPU (via `jax._src.pallas.mosaic`). |
| `jax.experimental.compilation_cache` | `jax.experimental.compilation_cache` | On-disk XLA compile cache | Persistent-cache hypothesis lever (explicitly flagged in the optimization-ceiling analysis as a cold-start mitigation). |
| `jax.experimental.serialize_executable` | `jax.experimental.serialize_executable` | AOT-compile + persist executable | Deployment lever. |
| `jax.experimental.source_mapper` | `jax.experimental.source_mapper` | HLO-op → Python-source mapping | Paired with the [xprof graph viewer](../sources/2026-xprof-graph-viewer.md) for "which line of Python generated this fusion?". |
| `jax.custom_vjp` / `jax.custom_jvp` | `jax` | User-provided backward / JVP | Required to make a forward-only Pallas kernel usable in training — the missing piece in [pallas-forge](pallas-forge.md). |
| `jax.collect_profile` | `jax.collect_profile` | Programmatic profile-collection entry used by `jax.profiler` CLI | Called by `xprof`'s gRPC capture path. |

## Entry points

```bash
# install (TPU runtime)
pip install "jax[tpu]"                     # canonical TPU install
pip install jax jaxlib                     # CPU
pip install "jax[cuda12]"                  # GPU

# check TPU visibility
python -c "import jax; print(jax.devices(), jax.device_count())"

# smallest end-to-end TPU program
python -c "import jax, jax.numpy as jnp; print(jax.jit(jnp.sin)(jnp.arange(8.0)))"

# capture a profile
python -c "
import jax, jax.numpy as jnp
@jax.jit
def f(x): return jnp.sin(x) ** 2 + jnp.cos(x) ** 2
with jax.profiler.trace('/tmp/jax-trace'):
    f(jnp.arange(1024.0)).block_until_ready()
"
```

Canonical programmatic hooks:

```python
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.experimental.roofline import roofline
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel import make_splash_mha

mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), ('axis',))
```

## Dependencies

JAX depends on:
- `jaxlib` (C++ bindings, compiled from this repo's `jaxlib/` via Bazel — distributed as a separate wheel).
- `libtpu` (via `jax[tpu]`) for TPU execution; the wiki's gemma4 program runs `libtpu==0.0.40`.
- `numpy`, `scipy`, `opt_einsum`, `ml_dtypes`, `etils`.
- `absl-py`.

Every other ingested codebase in this wiki depends on JAX:
- [torchax](torchax.md) imports `jax`, `jax.export`, `jax.sharding`, `jax.tree_util`, `jax.numpy`.
- [tokamax](tokamax.md) imports `jax.experimental.pallas` + `jax.experimental.mosaic.gpu`.
- [pallas-forge](pallas-forge.md) imports `jax.experimental.pallas` + `jax.profiler`.
- [xprof](xprof.md) reads profiles produced by `jax.profiler.trace`.
- [xprof-mcp](xprof-mcp.md) calls xprof which reads `jax.profiler` output.

## Notable files

### `jax/experimental/pallas/__init__.py`
Top-level Pallas DSL re-exports: `pallas_call`, `BlockSpec`, `GridSpec`, `Blocked`, `emit_pipeline`, `run_scoped`, `program_id`, `when`, `store`, `load`. **The user-facing Pallas API lives here.** Hardware-specific entrypoints at `pallas.tpu`, `pallas.tpu_sc`, `pallas.mosaic_gpu`, `pallas.triton`.

### `jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py`
First-party canonical **Splash Attention** implementation — the TPU flash-attention variant. Exports `make_splash_mha`, `make_splash_mqa`, `make_splash_mha_single_device`, plus `SplashConfig`. This is the **upstream** that `tokamax`'s `_src/ops/experimental/tpu/splash_attention/` mirrors, and it is the one the gemma4 program currently uses (see [exp 15](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp15-splash-bf16ce-batch3-accepted.md), [exp 24](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp24-splash-seq-minor-accepted.md), [exp 25](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp25-splash-block1024-accepted.md)). Authoritative for splash-attention tunables.

Sibling files:
- `splash_attention_mask.py` — mask constructors (causal, local, sliding-window).
- `splash_attention_mask_info.py` — mask-metadata (block-sparse structure the kernel exploits).

### `jax/experimental/pallas/ops/tpu/flash_attention.py`
Predecessor to splash — simpler flash-attention-style TPU kernel. Splash is generally preferred; flash remains for simpler use cases / as the reference the wiki's [flash-attention](../concepts/flash-attention.md) concept page discusses.

### `jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py` (+ sibling files)
Paged attention kernel — used by inference engines (vLLM-class). Signature supports page-aligned KV caches. `quantization_utils.py` in the same directory adds int8 KV-cache support. Out of scope for the gemma4 training program but relevant to any future decode-serving hypothesis.

### `jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py` + `tuned_block_sizes.py`
**Ragged-paged attention** — the kernel that combines ragged batching with paged KV cache, used by the vLLM TPU backend. `tuned_block_sizes.py` is a baked-in table of (head-dim, seq-len) → block-size tuning results. Worth reading as a template for "how to ship tuned block sizes in-tree".

### `jax/experimental/pallas/ops/tpu/megablox/gmm.py` + `ops.py` + `common.py`
**MegaBlox grouped matmul** — the MoE `ragged_dot` / `gmm` reference implementation. Upstream of the gemma4-relevant MoE story; the 2026-04-23 Pallas-kernel-source-survey analysis flags this directory as first-party authoritative.

### `jax/experimental/pallas/ops/tpu/matmul.py` + `example_kernel.py`
Teaching-quality reference matmul (same canonical 3D-grid / fp32-accumulator / K-axis-iterated pattern as `pallas-forge/pallas_forge/kernels/matmul.py`). `example_kernel.py` is the "hello world".

### `jax/experimental/pallas/ops/tpu/all_gather.py`
A Pallas-implemented `all_gather` — i.e. a collective op written as a Pallas kernel rather than a compiler intrinsic. Relevant for kernel-level collective-overlap hypotheses.

### `jax/experimental/pallas/ops/tpu/random/{threefry,philox}.py`
PRNG kernels in Pallas — relevant to any kernel-level dropout / masked-attention hypothesis.

### `jax/_src/pallas/mosaic/` (lowering)
`core.py`, `lowering.py`, `pipeline.py`, `primitives.py`, `pallas_call_registration.py`, `tpu_info.py`, `sc_core.py` (SparseCore), `sc_lowering.py`. **The Pallas→Mosaic-TPU lowering internals.** When a kernel fails at lower-time with an esoteric error, this is where the translation rules live. Not a modification target for optimization hypotheses, but essential reading when debugging.

### `jax/_src/pallas/cost_estimate.py`
The cost-model hook that Pallas uses. Relevant for hypotheses about cost-model-vs-measured divergence (the XProf [roofline](../sources/2026-xprof-roofline-model.md) page discusses exactly this gap).

### `jax/experimental/roofline/{roofline,rooflines}.py`
**The built-in JAX roofline analyzer.** Accepts any JAX function (not just Pallas), traces it, and returns `RooflineResult` with FLOPs + bytes counts per op via `RooflineRuleContext`. `register_roofline` / `register_standard_roofline` let you add rules for custom primitives. `roofline_and_grad` adds autodiff counts. First-party alternative to [pallas-forge](pallas-forge.md)'s matplotlib `roofline_chart`.

### `jax/profiler.py`
The public `jax.profiler` surface: `trace(logdir)` ctx-manager, `start_trace` / `stop_trace`, `StepTraceAnnotation` (step boundaries), `TraceAnnotation` (custom regions), `ProfileData` / `ProfileEvent` / `ProfilePlane` (programmatic profile reading), `ProfileOptions` (trace-mode knob for TPU), `annotate_function` (function decorator), `device_memory_profile` / `save_device_memory_profile` (static HBM snapshot). See [jax-trace](../concepts/jax-trace.md) for the wiki-level protocol.

### `jax/experimental/compilation_cache.py`
Persistent XLA compile cache. `set_cache_dir(path)` + reading-back at the same path avoids recompilation on cold-start. **Explicitly named in the gemma4 optimization-ceiling analysis** as one of the three remaining viable levers (the other two: scan-over-layers Option B, hardware scale-up).

### `jax/experimental/layout.py`
`Layout`, `DeviceLocalLayout`. Declares physical minor-major axis orders on `jax.jit` inputs/outputs. **This is the lever behind the gemma4 `SEQ_MINOR` choice** ([exp 24](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp24-splash-seq-minor-accepted.md)) — confirming a layout recommendation surfaces from splash-attention's contiguity requirements.

### `jax/experimental/custom_partitioning.py`
`custom_partitioning(sharding_rule, partition, infer_sharding_from_operands, propagate_user_sharding, ...)`. Used when an op's default partitioning is wrong. The "escape hatch" for SPMD issues; referenced in the 2026-04-23 pallas-kernel-source-survey as how several production libraries route around partitioner gaps.

### `jax/ad_checkpoint.py`
`checkpoint` (also `jax.checkpoint`), `checkpoint_policies.*`, `checkpoint_name`. Rematerialization control. The gemma4 program's [exp 3](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp3-full-remat-accepted.md) / [exp 5](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp5-selective-remat-accepted.md) hypotheses live here.

### `jax/lax/` (focus: `scan`)
`jax.lax.scan` is the scan-over-layers lever — collapses N transformer blocks into O(1) compile time instead of O(N). The gemma4 ceiling analysis names scan-over-layers Option B as one of three viable paths. Source: `jax/_src/lax/control_flow/loops.py`.

### `jax/collect_profile.py`
The module the `jax.profiler` CLI uses under the hood for gRPC-triggered profile collection (the "on-demand" capture mode in the [xprof capturing-profiles](../sources/2026-xprof-capturing-profiles.md) page).

### `jax/experimental/source_mapper/`
HLO-op → Python-source mapping. Complements the [xprof graph viewer](../sources/2026-xprof-graph-viewer.md) — lets you trace a specific HLO op back to the Python line that emitted it. Underused by the wiki so far; candidate hypothesis-support tool.

## Performance-relevant surfaces

Concrete levers for TPU performance hypotheses, grounded to file paths. These are the JAX-side surfaces that the gemma4 program (and any future optimization loop) will touch.

### 1. Reference TPU Pallas kernel tree — first-party, autodiff-capable, authoritative
`jax/experimental/pallas/ops/tpu/` contains the canonical first-party TPU Pallas kernels. Unlike [pallas-forge](pallas-forge.md), these have **proper `custom_vjp` definitions** where applicable and are used in production by JAX-based trainers. Hypothesis-writers should **start here** before considering pallas-forge:
- `splash_attention/` — current gemma4 attention (exp 15/24/25).
- `ragged_paged_attention/` — inference-time canonical.
- `paged_attention/` — simpler paged variant; int8 KV supported.
- `megablox/` — MoE grouped matmul (`gmm`).
- `matmul.py` — canonical 3D-grid matmul template.
- `all_gather.py` — Pallas-written collective; relevant for collective-overlap work.

### 2. `jax.experimental.roofline` — first-party, JAX-native roofline
The wiki's other two roofline treatments ([xprof-roofline-model](../sources/2026-xprof-roofline-model.md) and [pallas-forge](pallas-forge.md)'s `roofline_chart`) can both be checked against this first-party tool. Works on any JAX function. Use when you want a **pre-run** roofline estimate rather than post-run profile analysis.

### 3. `jax.experimental.compilation_cache` — persistent cold-start cache
The optimization-ceiling analysis from 2026-04-23 names this lever: *"persistent compile cache (infrastructure only, no TPS)"*. A hypothesis that wires this in would reduce first-step compile cost (currently ~150s per the [exp 2](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp2-pin-out-shardings-rejected.md) notes) but would not change steady-state TPS.

### 4. `jax.experimental.layout` — minor-axis control
The `SEQ_MINOR` choice that bought [exp 24](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp24-splash-seq-minor-accepted.md) its win is fundamentally a layout decision surfaced to `jax.jit` through `jax.experimental.layout.Layout`. Any future "what if we change the minor axis" hypothesis edits inputs/outputs through this surface.

### 5. `jax.lax.scan` — scan-over-layers (Option B in the ceiling analysis)
Collapses O(N) compile → O(1). The ceiling analysis flags this as one of three remaining viable levers; its path-of-least-resistance implementation is ~300–500 LOC per the analysis. Requires refactoring the training loop so the N transformer blocks become a single `scan`-ed step function.

### 6. `jax.ad_checkpoint.checkpoint` — selective vs full remat
Already exercised in gemma4 exp 3 (full remat, accepted) and exp 5 (selective remat, accepted and kept in the trunk). The `policy=` argument accepts names of activations to save vs recompute — the tuning surface.

### 7. `jax.custom_vjp` — the gate for any new training-use kernel
Any forward-only Pallas kernel (including all three in [pallas-forge](pallas-forge.md)) must be wrapped in `jax.custom_vjp` to be usable in training. The `ops/tpu/splash_attention/` kernels demonstrate the pattern and can be copied as a template.

### 8. `jax.experimental.custom_partitioning` — SPMD escape hatch
For ops whose default partitioning is wrong. The 2026-04-23 pallas-kernel survey flags this as the mechanism several production libraries use to route around partitioner gaps in ragged/paged attention and MoE kernels.

### 9. `jax.profiler.ProfileOptions` TPU trace-mode knob
The [jax-profiling source page](../sources/2026-xprof-jax-profiling.md) covers the user-facing API, but the underlying option object at `jax.profiler.ProfileOptions` is what controls TPU trace modes — the knob that determines whether you get op-level or HLO-level events, and the CMEM/SMEM/HBM counters that feed the [xprof utilization viewer](../sources/2026-xprof-utilization-viewer.md).

### 10. `jax.experimental.source_mapper` — HLO↔Python back-reference
Every optimization hypothesis that reads an HLO op name from a profile ("this `convolution.123` is hot") needs a bridge back to the Python callsite. `source_mapper` provides it. Underused in the wiki so far.

### 11. `jax.experimental.shard_map` vs implicit `pjit`/`jax.jit(in_shardings=...)`
Two SPMD programming models live side-by-side: implicit (declare shardings on inputs/outputs, let the partitioner fill in collectives) and explicit (`shard_map`: write per-device collectives by hand). Any hypothesis about collective placement needs to choose; exp 2's `out_shardings` path is the implicit model; the ceiling analysis flags 2D-mesh experiments (exp 32) that used the implicit model too.

### 12. `jax.experimental.mesh_utils.create_device_mesh` — ICI-aware mesh topology
Creates the right physical device mesh for a given logical shape on TPU v6e (respecting the ICI torus). Any multi-axis mesh hypothesis (exp 32 was 2D) uses this. `create_hybrid_device_mesh` handles multi-slice (DCN).

## Connections

Every other codebase in this wiki:
- [torchax](torchax.md) — imports and extends JAX; its `Tensor` wraps a `jax.Array`.
- [tokamax](tokamax.md) — Pallas kernel library built on `jax.experimental.pallas` and `jax.experimental.mosaic.gpu`. The `_src/ops/experimental/tpu/splash_attention/` mirror of this repo's `ops/tpu/splash_attention/` is the direct link.
- [pallas-forge](pallas-forge.md) — imports `jax.experimental.pallas` + `jax.profiler`; reimplements a matplotlib roofline that duplicates `jax.experimental.roofline`.
- [xprof](xprof.md) / [xprof-mcp](xprof-mcp.md) — read profiles produced by `jax.profiler`.
- [jax-huggingface](jax-huggingface.md) — uses `jax.jit`, `jax.make_mesh`, `jax.sharding.NamedSharding`, `jax.profiler.trace`, `jax.tree_util.register_pytree_node`.
- [scaling-book](scaling-book.md) — documents the scaling primitives this repo implements.
- [stablehlo](stablehlo.md) — the IR JAX lowers to before Mosaic/XLA backends.
- [autoresearch](autoresearch.md) — not a JAX consumer but benchmark baseline; included here for completeness.

Cross-referenced analyses and experiments:
- [Pallas kernel source survey (2026-04-23)](../analyses/2026-04-23-pallas-kernel-source-survey.md) — `jax.experimental.pallas.ops.tpu.*` is Tier-1 first-party in that survey.
- [Gemma 4 v6e-4 optimization ceiling (2026-04-23)](../analyses/2026-04-23-gemma4-v6e4-optimization-ceiling.md) — names `jax.lax.scan`, `jax.experimental.compilation_cache`, `jax.experimental.layout` as remaining levers.
- gemma4 experiments that directly touch JAX internals: [exp 3 (full remat)](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp3-full-remat-accepted.md), [exp 5 (selective remat)](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp5-selective-remat-accepted.md), [exp 15 (splash)](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp15-splash-bf16ce-batch3-accepted.md), [exp 24 (SEQ_MINOR layout)](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp24-splash-seq-minor-accepted.md), [exp 25 (splash block=1024)](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp25-splash-block1024-accepted.md), [exp 26 (scan-over-layers — blocked)](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp26-scan-over-layers-potential.md), [exp 32 (2D mesh)](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp32-2d-mesh-tp2-rejected.md).

## See also

- [pallas-kernel](../concepts/pallas-kernel.md) — the DSL this repo ships.
- [mosaic-kernel](../concepts/mosaic-kernel.md) — the TPU lowering target.
- [splash-attention](../concepts/splash-attention.md) — the attention kernel currently authoritative upstream is in this repo.
- [flash-attention](../concepts/flash-attention.md) — the ancestor pattern; both TPU Pallas implementations live here.
- [rematerialization](../concepts/rematerialization.md) — `jax.ad_checkpoint` is this concept's primary Python surface.
- [scan-over-layers](../concepts/scan-over-layers.md) — `jax.lax.scan` is the primitive.
- [sharding](../concepts/sharding.md) — `jax.sharding.*` are the primitives.
- [all-gather](../concepts/all-gather.md) / [all-reduce](../concepts/all-reduce.md) / [reduce-scatter](../concepts/reduce-scatter.md) — collective concepts whose TPU lowerings can be inspected through `jax._src/` and whose Pallas-level reimplementations live under `ops/tpu/all_gather.py`.
- [jax-trace](../concepts/jax-trace.md) — `jax.profiler` capture protocol.
- [profile-capture](../concepts/profile-capture.md) — umbrella concept.
- [roofline-model](../concepts/roofline-model.md) — `jax.experimental.roofline` is the first-party implementation.
- [xla-fusion](../concepts/xla-fusion.md) — the compiler pass JAX lowers through.
- [xla-flags](../concepts/xla-flags.md) — the flag surface exposed via `LIBTPU_INIT_ARGS`.
- [custom-call](../concepts/custom-call.md) — the mechanism by which Pallas kernels appear in HLO.
- [hlo-dumping-and-diffing](../concepts/hlo-dumping-and-diffing.md) — `source_mapper` helps here.

## Sources

- `raw/code/jax/README.md`
- `raw/code/jax/jax/__init__.py`
- `raw/code/jax/jax/profiler.py`
- `raw/code/jax/jax/sharding.py`
- `raw/code/jax/jax/ad_checkpoint.py`
- `raw/code/jax/jax/experimental/pallas/__init__.py`
- `raw/code/jax/jax/experimental/pallas/ops/tpu/` (directory; 8 kernel packages)
- `raw/code/jax/jax/experimental/roofline/__init__.py`
- `raw/code/jax/jax/experimental/mesh_utils.py`
- `raw/code/jax/jax/experimental/shard_map.py`
- `raw/code/jax/jax/experimental/layout.py`
- `raw/code/jax/jax/experimental/compilation_cache.py`
- `raw/code/jax/jax/experimental/custom_partitioning.py`
- `raw/code/jax/jax/experimental/source_mapper/`
- `raw/code/jax/jax/_src/pallas/mosaic/` (directory; lowering internals)
- Upstream: <https://github.com/jax-ml/jax>
