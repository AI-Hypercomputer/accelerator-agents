---
title: "AOT Compilation"
type: concept
tags: [compiler, workflow, analysis, jax]
created: 2026-05-06
updated: 2026-05-06
sources: 3
---

Ahead-of-time (AOT) compilation in JAX lets you lower, inspect, and compile a function **without executing it on an accelerator**. The pattern — `jax.jit(fn).lower(*args).compile()` — produces the optimized HLO, cost-model estimates, and even the compiled executable, all on CPU. This is the foundation for two capabilities central to the autoresearch loop: **HLO pre-filtering** (rejecting kernel-replacement hypotheses that XLA already handles) and **pre-experiment screening** (catching compilation errors and predicting metric movement before burning TPU time).

## Definition

The JAX AOT pipeline is a four-stage chain:

```
jax.jit(fn)          # (1) trace: capture a Jaxpr
  .lower(*args)      # (2) lower: produce StableHLO / MHLO
  .as_text()         # (2a) inspect: read the pre-optimization HLO text
  .compile()         # (3) compile: run XLA's full optimization pipeline
  .cost_analysis()   # (4) analyze: query the compiler's static cost model
```

Each stage is independently useful:

| Stage | Object returned | What you learn |
|---|---|---|
| `.lower(*args)` | `jax.stages.Lowered` | Shape/dtype/sharding of every intermediate; pre-optimization HLO text via `.as_text()` |
| `.compile()` | `jax.stages.Compiled` | Post-optimization HLO (what XLA actually runs); compilation success/failure; memory requirements |
| `.cost_analysis()` | `list[dict]` | Per-device FLOPs, bytes accessed, optimal seconds (cost-model, not wall-clock) |

The `*args` passed to `.lower()` determine shapes, dtypes, and shardings — they are not consumed, only traced. On a CPU-only host the compilation still runs XLA's full pass pipeline; the resulting HLO is the same HLO the TPU would execute (modulo target-specific backend differences flagged below).

## Why it matters for TPU perf

1. **HLO pre-filter for kernel-replacement hypotheses.** Before writing a Pallas kernel that fuses two HLO ops, AOT-compile and grep the post-optimization HLO for whether XLA already fuses them into a `kind=kOutput` Mosaic kernel. This refuted the [Pallas RMSNorm+matmul](../hypotheses/llama3-jax-rmsnorm-matmul-prologue-fusion.md) and [Pallas SwiGLU+down_proj](../hypotheses/llama3-jax-pallas-swiglu-downproj-fusion.md) hypotheses in 30 minutes, saving weeks of kernel-authoring effort. **Mandatory** for kernel-replacement hypotheses per [SCHEMA.md](../../SCHEMA.md) FORMULATE-HYPOTHESIS step 2b.

2. **Cost-model prediction.** `cost_analysis()` returns `flops`, `bytes accessed`, and `optimal_seconds` per device. Comparing baseline vs modified gives a first-order prediction of whether the change will move the metric — before any TPU-hours are spent.

3. **Compile-time error detection.** Many OOM failures (`CompileTimeHbmOom`, shape mismatches, sharding incompatibilities) surface at `compile()` time. AOT screening catches these on CPU, avoiding a wasted TPU launch.

4. **Offline HLO analysis.** When a customer sends HLO dumps but no TPU access, the xprof-mcp dump tools (`list_hlo_dump_modules`, `get_hlo_dump`, `diff_hlo_stages`, `get_hlo_dump_neighborhood`) work entirely on CPU — no xprof server or TPU required. Diagnose sharding, fusion, and collective patterns from the dump files alone.

5. **Program-size estimation.** `.lower(*args).as_text()` reports the HLO program size in characters; a 50M-char program compiles orders of magnitude slower than a 5M one. `scan-over-layers` is the lever that collapses program size.

## Mechanism

### The AOT pipeline walkthrough

```python
import jax
from jax.sharding import NamedSharding, PartitionSpec as P

# 1. Define the function and create shaped inputs
fn = train_step  # your training step function
mesh = jax.sharding.Mesh(jax.devices(), ('fsdp',))
args = (weights, buffers, opt_state, data, labels)  # real or dummy tensors

# 2. Lower — produces StableHLO
lowered = jax.jit(
    fn,
    donate_argnums=(0, 2),
    out_shardings=(NamedSharding(mesh, P()), w_shardings, o_shardings),
).lower(*args)

# 2a. Inspect pre-optimization HLO
print("Program size:", len(lowered.as_text()) / 1e6, "M chars")
# Optionally write to disk for diffing:
# Path("pre_opt.hlo").write_text(lowered.as_text())

# 3. Compile — runs XLA's full optimization pipeline
compiled = lowered.compile()

# 4. Query cost model
for device_cost in compiled.cost_analysis():
    print(f"FLOPs: {device_cost['flops']:.2e}")
    print(f"Bytes accessed: {device_cost['bytes accessed'] / 1e9:.2f} GB")
```

### Production reference: `helper.py`

The `compile_step_func` function in `wiki/experiments/llama3_8B_autoresearch_optimization/torchax/helper.py` is the production instantiation of this pattern. It:

1. Converts torch tensors to JAX views via `torchax.interop.jax_view`
2. Extracts per-tensor shardings from the JAX arrays
3. Calls `jax.jit(...).lower(...)` with `donate_argnums` and explicit `out_shardings`
4. Logs program size (`len(lowered.as_text()) / 1e6`)
5. Calls `.compile()` and prints `cost_analysis()` per device
6. Returns the compiled callable wrapped back to torch via `interop.torch_view`

### Inline utility: `aot_screen()`

For hypothesis pre-screening, a lighter-weight utility suffices:

```python
def aot_screen(fn, *args, donate_argnums=(), out_shardings=None, **jit_kwargs):
    """AOT compile fn and return (compiled, cost_analysis, hlo_text).

    Use before a full experiment to:
    - Catch compile errors (OOM, shape mismatch) on CPU
    - Read post-optimization HLO (grep for fusion patterns)
    - Compare cost_analysis between baseline and modified versions
    """
    jitted = jax.jit(fn, donate_argnums=donate_argnums,
                     out_shardings=out_shardings, **jit_kwargs)
    lowered = jitted.lower(*args)
    hlo_text = lowered.as_text()
    compiled = lowered.compile()
    costs = compiled.cost_analysis()
    return compiled, costs, hlo_text
```

### HLO pre-filter recipe

The specific workflow for kernel-replacement hypothesis screening:

1. AOT-compile the baseline function: `_, _, hlo = aot_screen(baseline_step, *args)`
2. Write HLO to disk or search in-memory for the target fusion pattern
3. Grep for `kind=kOutput` fusions containing your target ops (e.g., `fused_computation` entries that inline RMSNorm + matmul)
4. If XLA already fuses the target pattern, retire the hypothesis with reason `xla-already-fuses`
5. If XLA does *not* fuse it, the hypothesis passes the pre-filter — proceed to full experiment

Alternatively, use HLO dump files with xprof-mcp tools:
```bash
XLA_FLAGS="--xla_dump_to=/tmp/hlo --xla_dump_hlo_as_text" python -m train --steps 1
# Then query via MCP:
#   list_hlo_dump_modules(dump_dir="/tmp/hlo")
#   get_hlo_dump(dump_dir="/tmp/hlo", module_id="...", stage="after_optimizations")
```

## When it applies / when it doesn't

| Scenario | AOT useful? | Notes |
|---|---|---|
| **Kernel-replacement hypothesis** (Pallas fuse X+Y) | **Mandatory** | SCHEMA step 2b — pre-filter before ranking |
| **Flag/config experiment** | Optional, recommended | `cost_analysis()` delta predicts metric direction |
| **Sharding change** | Optional | Catches shape/shard mismatches early |
| **Offline HLO diagnosis** (customer dump) | Yes | xprof-mcp dump tools work on CPU |
| **Runtime phenomena** (collective stalls, scheduling) | **No** | These only manifest during execution with real inter-chip traffic |
| **Kernel autotuning** | **No** | Autotune needs actual hardware timing; cost model is too coarse |
| **Numerical validation** (loss sanity) | **No** | Must run on real data with real compute |

## Known results

| Date | Model | What AOT revealed | Outcome | Reference |
|---|---|---|---|---|
| 2026-04-27 | Llama 3 8B (JAX, v6e-8) | RMSNorm+matmul already fused by XLA into `kind=kOutput` Mosaic kernel (`fused_computation.47/.48/.31/.32`) | Hypothesis refuted — `xla-already-fuses` | [HLO inspection log](../hypotheses/llama3-jax-rmsnorm-matmul-prologue-fusion.md) |
| 2026-04-27 | Llama 3 8B (JAX, v6e-8) | SwiGLU+down_proj+residual already fused into single `kind=kOutput` kernel (`fused_computation.40`) | Hypothesis refuted — `xla-already-fuses` | [HLO inspection log](../hypotheses/llama3-jax-pallas-swiglu-downproj-fusion.md) |
| 2026-04-24 | Gemma 4 E4B (torchax, v6e-4) | exp 33 Pallas RMSNorm kernel −8.1 % vs XLA fusion | Confirmed XLA RMSNorm fusion is sufficient on TPU | [Gemma 4 ceiling analysis](../analyses/2026-04-23-gemma4-v6e4-optimization-ceiling.md) |
| Production | Llama 3 8B (torchax) | `helper.py` `compile_step_func` uses full AOT pipeline with `cost_analysis()` | Program size + cost logged every compile | [helper.py](../experiments/llama3_8B_autoresearch_optimization/torchax/helper.py) |

## Caveats

- **CPU-compiled HLO vs TPU-compiled HLO**: when compiling on a CPU-only host, XLA uses the CPU backend by default. To get TPU-targeted HLO, you need either: (a) a TPU host (even a single chip), or (b) HLO dump files from a prior TPU run. The xprof-mcp dump tools work on either.
- **Cost model is approximate**: `cost_analysis()` uses XLA's static cost model, which can disagree with hardware performance counters (see [xla-cost-model](xla-cost-model.md)). Use it for directional signals, not precise predictions.
- **Compile time itself can be expensive**: for large models without [scan-over-layers](scan-over-layers.md), a single `compile()` call can take 2-5 minutes. Factor this into screening workflows.

## See also

- [HLO Dumping and Diffing](hlo-dumping-and-diffing.md) — how to capture and diff HLO dumps
- [HLO (High Level Optimizer IR)](hlo.md) — the IR that AOT compilation produces
- [XLA Fusion](xla-fusion.md) — the optimization pass that AOT inspection often targets
- [XLA Cost Model](xla-cost-model.md) — what `cost_analysis()` reports
- [Pallas Kernel](pallas-kernel.md) — the kernel type whose hypotheses require AOT pre-filtering
- [Scan Over Layers](scan-over-layers.md) — key lever for reducing AOT compile time
- [Custom Call](custom-call.md) — how Pallas/Mosaic kernels appear in HLO

## Sources

- `wiki/experiments/llama3_8B_autoresearch_optimization/torchax/helper.py` — production `compile_step_func` using the full AOT pipeline
- [xprof-mcp TPU optimization guide](../sources/2026-xprof-mcp-tpu-optimization.md) — HLO dump workflows and optimization patterns
- [xprof-mcp codebase](../codebases/xprof-mcp.md) — dump tools (`list_hlo_dump_modules`, `get_hlo_dump`, `diff_hlo_stages`, `get_hlo_dump_neighborhood`)
