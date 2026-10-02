---
title: "torchax"
type: codebase
tags: [framework, pytorch, jax-interop, tpu-backend]
commit: 8f957d11128ee273a14414750b3998870f933d15
created: 2026-04-22
updated: 2026-04-22
---

> **Grounded front door → [torchax overview](torchax/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

torchax is a PyTorch backend that runs PyTorch programs on TPU by dispatching every PyTorch op to a JAX equivalent. It is the torch-to-TPU bridge for any torch-origin model under optimization in this wiki — it supplies the `Tensor` subclass, the op lowering tables, the `jax.jit` entry point, and the interop for moving values between `torch.Tensor` and `jax.Array` pytrees. Upstream version at this commit is `0.0.12`.

## Overview

torchax uses a `torch.Tensor` subclass (`torchax.tensor.Tensor`) that stores a `jax.Array` in `_elem` and presents itself with `device='meta'` to PyTorch. Two torch "modes" — `XLAFunctionMode` (torch-function level) and `XLADispatchMode` (torch-dispatch level) — intercept ops and route them through an `Environment`, which looks up an implementation in an op registry and calls the corresponding JAX function. Compilation is opt-in via `jax.jit` wrappers (`torchax.interop.jax_jit`, `JittableModule`, `torchax.compile`), so the same code can run eagerly (one-op-at-a-time into XLA) or as a single compiled graph.

This makes torchax the graph-compilation boundary for torch-origin models on TPU: anywhere a PyTorch op hits a torchax tensor, the compiled XLA graph is whatever JAX produces from the registered lowerings.

## Architecture

Three layers, bottom-up:

1. **Tensor + Environment** (`torchax/tensor.py`). `Tensor` wraps `jax.Array`; `Environment` owns the op registry, the PRNG key, mesh/autocast runtime properties, and the `dispatch` function that every torch op ultimately calls.
2. **Op lowering tables** (`torchax/ops/`). Registered via `@op(...)` / `register_torch_function_op`. Separate tables for aten ops (`all_aten_ops`) and higher-level `torch.*` functions (`all_torch_functions`). Fallback is torch's decomposition database, re-registered through `torchax/decompositions.py`.
3. **Compilation / interop** (`torchax/interop.py`, `torchax/train.py`, `torchax/export.py`). `wrap_jax_jit` is the single pattern that turns a torch-callable into a JAX-callable, jits it, and wraps it back. Same machinery powers `jax_jit`, `jax_shard_map`, `jax_value_and_grad`, `gradient_checkpoint`.

Entering torchax is via `torchax.enable_globally()` which pushes the function mode + dispatch mode onto the torch mode stacks; `torchax.default_env()` is the singleton `Environment`.

## Key abstractions

- **`torchax.tensor.Tensor`** — `torch.Tensor` subclass holding a `jax.Array`. `.jax()` extracts, `.torch()` copies to CPU `torch.Tensor`. Reports `device='jax:0'` but internally `device='meta'`.
- **`View`** (`torchax/view.py`) — lazy view representation (`NARROW`, `PERMUTE`, `RESHAPE`, `SELECT`, `AS_STRIDED`, `DIAGONAL`, …) used to defer materialization of strided/mutating ops like `copy_`. Materialized on op boundaries that need a real tensor or when `config.force_materialize_views=True`.
- **`Environment`** (`torchax/tensor.py:354`) — owns op registry (`_ops`, `_decomps`), PRNG key rotation (`get_and_rotate_prng_key`), conversions (`t2j_iso`, `j2t_iso`, `t2j_copy`, `j2t_copy`), autocast policy application, and the torch mode lifecycle.
- **`Operator`** (`torchax/ops/ops_registry.py`) — dataclass holding the torch op, the JAX (or torch) impl, and flags `is_jax_function`, `needs_env`, `is_view_op`, `is_user_defined`.
- **`JittableModule`** (`torchax/interop.py:75`) — wraps a `torch.nn.Module`, extracts params/buffers, deduplicates shared params, and per-method caches the `jax.jit`-compiled functional form. This is the standard compile-a-model path.
- **`ScannedModule`** (`torchax/train.py`) — stacks a list of identical submodules and runs them with `jax.lax.scan` + per-layer `jax.checkpoint`, so a transformer stack becomes a single scan + remat instead of N unrolled layers.

## Entry points

- `torchax.enable_globally()` / `disable_globally()` — turn dispatch on/off globally (`torchax/__init__.py:115`).
- `torchax.default_env()` — get the singleton `Environment`.
- `torchax.enable_accuracy_mode()` / `enable_performance_mode()` — toggle `jax_enable_x64` and `jax_default_matmul_precision` (see Performance-relevant surfaces).
- `torchax.compile(fn, CompileOptions)` — wrap a module or function with `jax.jit`; only `mode="jax"` is implemented, `dynamo` and `export` modes raise.
- `torchax.interop.jax_jit(torch_fn, kwargs_for_jax_jit=...)` — compile a function that takes/returns `torchax.Tensor`s.
- `torchax.interop.JittableModule(m, extra_jit_args=...)` — compile an `nn.Module`.
- `torchax.extract_jax(mod)` — get a pure `(states, jax_fn)` pair for use with arbitrary `jax.*` transforms.
- `torchax.train.make_train_step(model_fn, loss_fn, optax_optimizer, remat_policy)` — canonical training step with gradient + optax update.
- `torchax.save_checkpoint` / `load_checkpoint` (`torchax/checkpoint.py`).

## Dependencies

- `jax` (`>=0.8.0` path uses top-level `jax.shard_map`; older path falls back to `jax.experimental.shard_map`).
- `torch` CPU build — torch is only used for its mode/dispatch machinery and pytree; the actual arrays live in JAX.
- `optax` — used by `train.py` and example training loops.
- `torchvision` — NMS lowering only (`torchax/ops/jtorchvision_nms.py`).
- Optional: `flax` bridge (`torchax/flax.py`), `torchtitan` in the llama3 example.

## Notable files

- [`torchax/__init__.py`](../../raw/code/torchax/torchax/__init__.py) — package root, enable/disable, `compile`, accuracy/performance mode switches.
- [`torchax/tensor.py`](../../raw/code/torchax/torchax/tensor.py) — `Tensor`, `XLAFunctionMode`, `XLADispatchMode`, `Environment`, `dispatch`.
- [`torchax/config.py`](../../raw/code/torchax/torchax/config.py) — `Configuration` dataclass (all runtime knobs).
- [`torchax/interop.py`](../../raw/code/torchax/torchax/interop.py) — `jax_view`/`torch_view`, `wrap_jax_jit`, `jax_jit`, `jax_shard_map`, `gradient_checkpoint`, `JittableModule`, `j2t_autograd`.
- [`torchax/train.py`](../../raw/code/torchax/torchax/train.py) — `make_train_step`, `ScannedModule` (scan+remat over transformer layers).
- [`torchax/decompositions.py`](../../raw/code/torchax/torchax/decompositions.py) — extra torch decomps used when no direct JAX lowering exists.
- [`torchax/view.py`](../../raw/code/torchax/torchax/view.py) — lazy view IR.
- [`torchax/amp.py`](../../raw/code/torchax/torchax/amp.py) — autocast policy execution (bf16 casting, `fp32_set_opt_dtype`, etc.).
- [`torchax/mesh_util.py`](../../raw/code/torchax/torchax/mesh_util.py) — mesh construction + auto-sharding helpers (`SingleAxisSharder`).
- [`torchax/export.py`](../../raw/code/torchax/torchax/export.py) — `torch.export` → stablehlo path via `JaxInterpreter` over the FX graph.
- [`torchax/ops/ops_registry.py`](../../raw/code/torchax/torchax/ops/ops_registry.py) — `Operator` dataclass and registration functions.
- [`torchax/ops/jaten.py`](../../raw/code/torchax/torchax/ops/jaten.py) — ~416 `@op` registrations covering `torch.ops.aten.*`.
- [`torchax/ops/jtorch.py`](../../raw/code/torchax/torchax/ops/jtorch.py) — ~50 `@register_function` entries for top-level `torch.*` APIs.
- [`torchax/ops/jc10d.py`](../../raw/code/torchax/torchax/ops/jc10d.py) — `_c10d_functional` collectives mapped to `jax.lax.all_gather`/`psum`/`pmean`/`pmin`/`pmax` on axis `"torch_dist"`.
- [`torchax/ops/jimage.py`](../../raw/code/torchax/torchax/ops/jimage.py) — image interpolation ops.
- [`torchax/ops/jtorchvision_nms.py`](../../raw/code/torchax/torchax/ops/jtorchvision_nms.py) — torchvision NMS lowering.
- [`torchax/ops/jlibrary.py`](../../raw/code/torchax/torchax/ops/jlibrary.py) — register a JAX or torch implementation as a StableHLO composite for export.
- [`torchax/ops/mappings.py`](../../raw/code/torchax/torchax/ops/mappings.py) — `t2j`/`j2t` array conversion + dtype table; DLPack is optional.
- [`torchax/ops/op_base.py`](../../raw/code/torchax/torchax/ops/op_base.py) — `InplaceOp`, `OutVariant`, `convert_dtype`, `promote_int_input` helpers used by `jaten`.
- [`torchax/ops/jax_reimplement.py`](../../raw/code/torchax/torchax/ops/jax_reimplement.py) — JAX functions reimplemented for torch semantics mismatches.
- [`docs/docs/user_guide/how-it-works.md`](../../raw/code/torchax/docs/docs/user_guide/how-it-works.md) — design narrative for the tensor subclass + dispatch mechanism (to be ingested as a source page).
- [`docs/docs/user_guide/working-with-impure-functions.md`](../../raw/code/torchax/docs/docs/user_guide/working-with-impure-functions.md) — how `jax.jit` interacts with globals/side effects and the pattern `JittableModule` implements (to be ingested as a source page).
- [`examples/eager_mode.py`](../../raw/code/torchax/examples/eager_mode.py), [`examples/train_gpt/`](../../raw/code/torchax/examples/train_gpt/), [`examples/train_llama_torchtitan/train_llama.py`](../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py) — end-to-end examples, including splash-attention + FSDP+TP sharding for Llama3.

## Performance-relevant surfaces

These are the knobs and code paths optimization hypotheses will touch.

### 1. Op lowering tables (the torch → JAX boundary)

- **Aten lowerings**: `torchax/ops/jaten.py:1–5830`. `@op(torch.ops.aten.X)` registers a JAX function keyed by torch op. ~416 registrations. Missing ops fall through to `decompositions.py` and then torch core decomps; unfound ops raise `OperatorNotFound` and the dispatch mode re-runs the original torch op (see `tensor.py:249–255`).
- **Top-level torch function lowerings**: `torchax/ops/jtorch.py:1–603`, ~50 entries.
- **Collectives**: `torchax/ops/jc10d.py:31–64` — all distributed collectives go through axis name `"torch_dist"`.
- **Registration API**: `torchax/ops/ops_registry.py:35–74`.
- **Decomposition fallback**: `torchax/tensor.py:436–455` (`Environment.load_ops`) and `torchax/decompositions.py`. Decomposed ops execute through the dispatch mode a second time, which is the common way to hit a long chain of small JAX calls — a candidate site for performance pathologies.
- **Operator precedence**: `torchax/tensor.py:457–477` (`_get_op_or_decomp`). Direct op > OpOverload default > OpOverloadPacket > decomp table.

### 2. Graph-compilation boundary

- **Single jit entry**: `torchax/interop.py:314–344`. `wrap_jax_jit(torch_fn, jax_jit_func=jax.jit, kwargs_for_jax=...)` unwraps torch-land pytrees to JAX-land, jits, and re-wraps. Same wrapper is reused with `shard_map`, `jax.value_and_grad`, and `jax.checkpoint`.
- **Per-method compile cache**: `torchax/interop.py:131–163` (`JittableModule.jittable_call`, `make_jitted`). Cache key is the method name; `jax.jit` then caches on input shape/dtype as usual.
- **Forwarding `jax.jit` kwargs** (static_argnums, donate_argnums, in/out shardings): passed via `extra_jit_args` / `kwargs_for_jax_jit` / `CompileOptions.jax_jit_kwargs` (`torchax/__init__.py:156–175`).
- **Autograd via `jax.vjp`**: `torchax/interop.py:218–308` (`j2t_autograd`) — wraps a JAX function in a `torch.autograd.Function` using `jax.vjp`, saves residuals into the ctx. Cache key is the `fn` object id, hence the nested `_jax_forward` per call to avoid collisions (see the comment at 226–229).
- **Scan + remat over transformer layers**: `torchax/train.py:76–131` (`ScannedModule`) uses `jax.lax.scan` + `jax.checkpoint(..., policy=self.checkpoint_policy)`; `remat_policy` in `make_train_step` forwards to `jax.ad_checkpoint.checkpoint_policies`.

### 3. Configuration knobs affecting compilation & performance

All live on `torchax.Configuration` (`torchax/config.py:18–41`) accessed via `default_env().config`:

| Knob | Default | Perf effect |
|---|---|---|
| `use_dlpack_for_data_conversion` | `False` | Zero-copy `t2j`/`j2t`. Off by default; enabling avoids a host numpy round-trip in `mappings.t2j`/`j2t`. |
| `use_int32_for_index` | `False` | Narrow indices, reduces HBM pressure on large gather/scatter. |
| `force_materialize_views` | `False` | If `True`, every `View` is materialized eagerly — trades extra copies for simpler graphs. |
| `internal_respect_torch_return_dtypes` | `False` | When `True` (accuracy mode), operators preserve torch dtype semantics; can prevent implicit bf16 downcasts that speed things up. |
| `treat_cuda_as_jax_device` | `True` | Routes `cuda` tensors through torchax too. |
| `debug_print_each_op`, `debug_accuracy_for_each_op`, `debug_print_each_op_operands`, `debug_mixed_tensor` | `False` | All are heavy; must be off for real runs. |

Globals toggled via `torchax.enable_accuracy_mode()` / `enable_performance_mode()` (`torchax/__init__.py:144–153`):

| Mode | `jax_enable_x64` | `jax_default_matmul_precision` | `internal_respect_torch_return_dtypes` |
|---|---|---|---|
| accuracy | `True` | `highest` | `True` |
| performance | `False` | `default` | `False` |

On TPU, `jax_default_matmul_precision` is the primary compile-time switch between bf16 and f32 matmul; `highest` forces f32 accumulation and is typically much slower.

### 4. Autocast

- `torchax/amp.py` — implements the four C++ cast policies (`LOWER_PRECISION_FP`, `FP32`, `FP32_SET_OPT_DTYPE`, `FP32_APPEND_DTYPE`, `PROMOTE`). Policy is looked up per op at dispatch time (`tensor.py:596–602`) when `env.param.autocast_dtype` is set. Target dtype is supplied via `Environment.override_property(autocast_dtype=...)`.

### 5. Sharding / mesh

- `torchax/mesh_util.py` — helpers to build a `jax.sharding.Mesh` and auto-generate `PartitionSpec`s (`SingleAxisSharder` shards the first dimension divisible by axis size). Sharding is actually applied at tensor level via `Tensor.shard_(sharding)` which calls `jax.lax.with_sharding_constraint`.
- `torchax.train.mark_sharding = torch_view(jax.lax.with_sharding_constraint)` (`torchax/train.py:26`) — the torch-side alias for inserting sharding constraints inside a compiled function.
- `jax_shard_map` (`torchax/interop.py:327–330`) — the way to compile with `shard_map` instead of `jit`.

### 6. Eager-mode overhead

The dispatch path in `Environment.dispatch` (`torchax/tensor.py:555–631`) re-enters `_dispatch_mode` to walk async collectives, applies autocast, and for decomposed ops enters a second dispatch pass. Eager-mode model runs therefore incur per-op Python overhead proportional to decomposition depth. This is the reason `JittableModule` / `jax.jit` is the recommended path for any real workload.

### 7. Export / StableHLO

- `torchax/export.py` turns a `torch.export` ExportedProgram into a JAX function by walking the FX graph with `JaxInterpreter` and calling registered aten lowerings directly — bypassing the torch dispatch modes entirely. Relevant when capturing a whole-graph for ahead-of-time compilation.
- `torchax/ops/jlibrary.py` lets a specific aten/custom op be preserved as a `stablehlo.composite` in the exported module rather than being inlined — affects downstream compiler pattern-matching (flash-attention, fused kernels) on the exported graph.

## Connections

- Relevant for every torch-origin model on TPU: torchax is the boundary where PyTorch semantics turn into HLO. Optimization hypotheses about op choice, dtype, precision, sharding, rematerialization, or kernel substitution (e.g. swapping an attention lowering for pallas splash-attention) all land in one of the surfaces above.
- `JittableModule` + `jax.jit` kwargs are the natural place to set `donate_argnums` and in/out shardings for FSDP-style training.
- `ScannedModule` is the scan+remat path for transformer stacks and is a direct analogue of torch `torch.utils.checkpoint` applied uniformly across layers.

## See also

*No related pages exist yet — this is the first codebase ingested.*

## Sources

- [`raw/code/torchax/README.md`](../../raw/code/torchax/README.md)
- [`raw/code/torchax/torchax/`](../../raw/code/torchax/torchax/) (package root at commit `8f957d1`)
- [`raw/code/torchax/docs/docs/user_guide/how-it-works.md`](../../raw/code/torchax/docs/docs/user_guide/how-it-works.md)
- [`raw/code/torchax/docs/docs/user_guide/working-with-impure-functions.md`](../../raw/code/torchax/docs/docs/user_guide/working-with-impure-functions.md)
- [`raw/code/torchax/examples/`](../../raw/code/torchax/examples/)
- [Ultra-Scale Playbook (2025)](../sources/2025-ultrascale-playbook.md) — GPU-side reference for DDP/FSDP/ZeRO/TP/SP; on TPU these are emergent from `PartitionSpec` over a mesh, not torch.distributed wrappers — the GPU↔TPU translation table in that source page maps each torch.distributed mechanism to its JAX/XLA analogue relevant to `torchax.compile` / `jax_jit` / `jax_shard_map` / `gradient_checkpoint`.
