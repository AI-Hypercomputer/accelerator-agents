---
title: "tokamax docs — autotuning framework"
type: source
tags: [docs, kernels, pallas, autotune, tuning, cache, stub]
created: 2026-04-22
updated: 2026-04-22
---

The autotuning doc in the tokamax repo is a placeholder at the time of ingest — the prose deep-dive is "coming soon". What we know about the autotuning framework comes from the basic-usage doc, the codebase, and the `AutotuningResult` API. This source page captures that, flags what is still unknown about autotuner internals, and points downstream `autotuning` concept work at the code rather than this doc.

## Overview

Tokamax's autotuner is a per-`Op`, per-`(op, shape, dtype, config-key)` search over a backend-defined configuration space. A single top-level call — `tokamax.autotune(f, *args)` — walks the lowered HLO of `f`, finds every tokamax op, dedupes by a cache key, sweeps each op's `_get_autotuning_configs` list, and returns an `AutotuningResult` that can be applied as a context manager, combined with another result via `|`, or serialized to JSON for reuse.

The doc file itself (`docs/autotuning.md`) currently contains only a one-line "coming soon" notice. The framework is real and usable today; the reference material is elsewhere. For the TPU performance loop this is the practical surface for kernel tuning — a hypothesis that says "tile the splash kernel differently" is really a hypothesis that says "override the autotuner's winning config".

## Key claims

Claims explicitly in the doc:

- The autotuning framework exists in tokamax; **the detailed write-up is pending**.

Claims from the basic-usage doc that pertain to autotuning:

- **`tokamax.autotune(f, *args)` returns an `AutotuningResult`.**
- **`AutotuningResult` is a context manager** — `with autotune_result: out = f_grad(x, scale)` applies the tuned configs to any tokamax op reached inside the block.
- **`AutotuningResult.dumps()` / `AutotuningResult.loads(...)`** serialize the result to/from JSON so that expensive searches are reusable.
- **Autotuning is non-deterministic.** Measuring kernel execution time is noisy, and different configs may produce different numerics. Serializing a fixed result is the recommended way to lock numerics across sessions.
- **Per-kernel tuning is user-extensible.** A user can subclass `tokamax.Op`, override `_get_autotuning_configs`, and their kernel participates in `tokamax.autotune` on equal footing.

Claims from the codebase (see [tokamax](../codebases/tokamax.md) — `_src/autotuning/`):

- **`tokamax.autotune(f, *args, ignore_cache=False, all_implementations=False)`** — the public signature:
  - `ignore_cache=True` forces a fresh search.
  - `all_implementations=True` sweeps every backend, not just the one that would be selected under `implementation=None`.
- **On-disk cache** — `_src/autotuning/cache.py` stores tuned results persistently. The repo ships a pre-populated cache under `tokamax/data/` — some shapes are already tuned out of the box.
- **`tokamax_autotuning_cache_miss_fallback`** — a global config flag in `tokamax/config.py` controlling what happens when a shape is not in the cache (fall back to heuristic defaults vs. run a fresh search vs. error). Exact semantics documented only in the code.
- **`AutotuningData`** — per-op per-key timings, the low-level data structure the tuner builds before it materialises an `AutotuningResult`.
- **`autotuning_cache_key`** on `BoundArguments` — the dedup key. Two calls that share a key are tuned once.
- **Op-level hooks** on `tokamax.Op`:
  - `_get_autotuning_configs(self, ba: BoundArguments) -> list[Config]` — enumerate the search space.
  - `_get_heuristics_config(self, ba: BoundArguments) -> Config` — the fallback when the cache misses and the fallback policy says "don't run fresh".
  - `supported_on(...)` — gate which backends are eligible (e.g., TPU-generation checks).
- **Progress bar** — `tqdm` is a dependency, which tells us the tuner emits a progress bar per op / per config.

## Key data points

### Autotuner API surface (as reachable from code)

| Call | Returns | Notes |
|---|---|---|
| `tokamax.autotune(f, *args, ignore_cache=False, all_implementations=False)` | `AutotuningResult` | Walks HLO; returns tuned configs for every tokamax op in `f`. |
| `tokamax.get_bound_args(f, *args, **kwargs)` | list of `BoundArguments` | Inspect ops without tuning — useful for listing what *would* be tuned. |
| `result1 \| result2` | `AutotuningResult` | Combine results across calls / models. |
| `with result: ...` | context manager | Applies tuned configs. |
| `result.dumps()` / `.dump(path)` | `str` / file | Serialize. |
| `AutotuningResult.loads(s)` / `.load(path)` | `AutotuningResult` | Deserialize. |
| Subclass `tokamax.Op`, override `_get_autotuning_configs` | — | Register a user kernel with the framework. |

### Op-level autotuning contract

```python
class MyOp(tokamax.Op[P, T, R, Config, Key]):
    def _get_heuristics_config(self, ba): ...        # default when cache misses
    def _get_autotuning_configs(self, ba): ...       # list of candidate configs
    def _fwd(self, config, ...): ...                 # the kernel itself
```

The tuner calls `_get_autotuning_configs` once per unique `autotuning_cache_key`, times each candidate, and records the winner in `AutotuningData`. The winner survives in the `AutotuningResult`.

### Autotune search space — per TPU kernel (from the codebase)

| Op | Config knobs in the search | Key pruning |
|---|---|---|
| Attention (splash) | `block_q`, `block_kv`, `block_kv_compute` ∈ `{128, 256, 512, 1024, 2048, 4096}`; `{q,k,v}_layout ∈ {HEAD_DIM_MINOR, SEQ_MINOR}`; `use_experimental_scheduler ∈ {True, False}` | `seq_len ≥ 1024 ⇒ block ≥ 1024`; `bkv_c ≤ 1024`; `8192` excluded (code TODO) |
| Ragged-dot | `tile_m ∈ {64, 128, 256, 512, 1024}`, `tile_k`/`tile_n` ∈ powers of two ≤ dim (+ full-axis); `input_buffer_count ∈ {2, 3, 4}` | `tile_m` capped at 1024 for "reasonable compilation time" |
| CE loss (fwd / bwd) | `b_block_size`, `h_block_size`, `v_block_size` = powers of two / divisors of each dim | Defaults differ by TPU generation to avoid VMEM OOM |

### Backward-pass knobs NOT in the search

Splash attention's `SplashConfig` exposes `block_q_dkv`, `block_kv_dkv`, `block_kv_dkv_compute`, `block_q_dq`, `block_kv_dq`, but **`pallas_mosaic_tpu.py` hard-wires them to 128** and does not sweep them. That is a concrete gap in the autotune coverage for training workloads.

### Pre-tuned cache

- Directory: `tokamax/data/`.
- Contents: pre-serialized `AutotuningResult` entries for common shapes, shipped with the repo.
- Implication: calling `tokamax.autotune` on an already-cached shape is cheap and deterministic; on a fresh shape it is expensive.

### Non-determinism sources

| Source | Why it varies run-to-run |
|---|---|
| Kernel-time measurement noise | Thermal, interrupt, clock jitter |
| Candidate ranking | Several configs are within noise; a tie can flip |
| Numerics across configs | Different tile shapes → different partial-sum accumulation order |

Lock-in strategy: run `tokamax.autotune` once on a rep machine, `result.dumps()`, commit the JSON, load in every subsequent session. This is the doc's explicit recommendation.

## Techniques referenced

- **HLO-walking autotuner** — the tuner scans the lowered JAX program to discover every tokamax op, rather than requiring the user to register ops individually. This is what makes a single `tokamax.autotune(f, *args)` call sufficient.
- **Per-key deduplication** — `autotuning_cache_key` on `BoundArguments` ensures that two calls sharing a key are tuned once. The key is a function of shape + dtype + static metadata; it is *not* a function of array *values*.
- **On-disk cache with fallback policy** — a tuning result is memoized across processes. The `tokamax_autotuning_cache_miss_fallback` flag governs what a cache miss does (heuristic default vs. fresh search vs. error).
- **Context-manager override** — `with AutotuningResult(...)` is a thread-local override that replaces the heuristic/cache lookup for the duration of the block. This is the user-visible primitive.
- **Combinable results** — `result1 | result2` lets a user build up a single `AutotuningResult` from per-model tunes and apply the union.
- **Serialization to JSON** — `dumps/loads` make tuning reproducible across sessions and machines of the same generation.
- **User-extensible `Op`** — subclassing `tokamax.Op` and overriding `_get_autotuning_configs` registers a user kernel with the same tuner.

## Gaps & caveats

- **The doc is a stub.** One sentence; no API reference, no recipe, no cache semantics, no "what to do when tuning takes hours", no per-op tuning advice. Every concrete claim on this page is extrapolated from the code or from the basic-usage doc, not from `autotuning.md` itself.
- **No discussion of tuning *cost*.** Autotuning a splash-attention op has `6³ × 2³ × 2 = 3456` candidates before pruning — that is minutes to hours of measurement per unique shape. For a model with several distinct attention shapes this compounds. The doc does not acknowledge this.
- **Cache-miss fallback policy is undocumented at the doc level.** Known from code that `tokamax_autotuning_cache_miss_fallback` exists; users cannot discover the exact semantics without reading `tokamax/config.py`.
- **`all_implementations=True` is not described in the doc.** This flag makes the tuner evaluate every backend (not just the `None` pick) — load-bearing for hypotheses of the form "is the XLA fallback actually faster than the Pallas kernel for this shape?".
- **No guidance on autotuning during training vs. ahead-of-time.** The non-determinism caveat says "serialize the result" — but the *when* is not stated. Typical practice: tune once pre-training, load JSON in the training job. Not spelled out.
- **No interaction with JIT recompilation caveats.** Applying `AutotuningResult` as a context manager changes the selected config, which changes the traced HLO, which can trigger JAX recompilation. Not discussed.
- **No TPU-vs-GPU differences.** Both are handled by the same framework, but e.g., compile-time cost for TPU Pallas is different from Triton compile time for GPU; affects the "reasonable tuning time" discussion.
- **Backward-pass tile knobs for splash attention are not exposed to the tuner.** This is a real coverage gap and is not called out anywhere in the docs.

## Connections

Concept slugs this source informs:

- `autotuning` — this doc anchors the concept page; the concept page should lean on the code.
- `tokamax-api-dispatch` — autotune overrides the `implementation=None` default selection.
- `autotune-cache` — on-disk cache + `tokamax_autotuning_cache_miss_fallback`.
- `autotuning-non-determinism` — repeat-run config flips + numerical jitter.
- `attention-block-sizes` — splash attention's block-q/kv/kv-compute sweep.
- `ragged-dot-tiling` — grouped-matmul tile sweep.
- `autotune-bwd-coverage-gap` — splash backward-pass knobs not autotuned.
- `autotuning-all-implementations` — the `all_implementations=True` path.

### Open questions for the autotuning concept page

When the doc stub is filled in (or when we want to lean on the code instead), capture:

1. **Tuning time budget** per op × per shape on each TPU generation.
2. **Cache schema**: what is the exact key? (Shape / dtype / static args / quant / mask pattern?)
3. **`tokamax_autotuning_cache_miss_fallback` values and their behavior** — transcribe from `tokamax/config.py`.
4. **JIT recompilation cost of entering/exiting `with AutotuningResult(...)`** — is it once per context, once per shape, something else?
5. **Forward/backward config coherence** — when the tuner picks different configs for fwd and bwd, are the tiles forced to be compatible, or can the bwd pick something weird?
6. **`all_implementations=True` — real-world outcomes**: for which TPU shapes does XLA actually beat splash attention?
7. **How much is on the table** from autotuning a model that today uses heuristic defaults? Headline number (e.g., X% step-time on Llama-3-8B) once we measure.

## See also

- [tokamax](../codebases/tokamax.md) — `_src/autotuning/` files and per-op `Config` dataclasses.
- [tokamax basic usage](2026-tokamax-basic-usage.md) — the canonical `tokamax.autotune(...)` + context-manager recipe.
- [tokamax splash attention](2026-tokamax-splash-attention.md) — the TPU attention op with the richest tuning surface.
- [tokamax supported ops](2026-tokamax-supported-ops.md) — which kernels have tuners at all.
- [tokamax benchmarking](2026-tokamax-benchmarking.md) — how to measure a tuned vs. untuned run without JAX Python overhead polluting the comparison.

## Sources

- [`raw/code/tokamax/docs/autotuning.md`](../../raw/code/tokamax/docs/autotuning.md)
