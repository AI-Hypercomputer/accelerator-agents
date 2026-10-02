---
title: "jaxpr-hash cache keys for Pallas autotune"
type: concept
tags: [autotuning, jaxpr, cache, marin, levanter, stub]
created: 2026-04-23
updated: 2026-04-23
---

Pin an autotune cache entry to the stringified jaxpr of the function being tuned, so silent shape/dtype/soft-cap/flag changes invalidate the cache. First-party reference: marin/levanter `_autotune_jaxpr_hash`. *Stub — expand when more sources are available.*

## Definition

A Pallas kernel's optimal block sizes depend on the full computational graph around it, not just nominal argument shapes. **jaxpr-hash cache keys** SHA-256 the stringified jaxpr (`str(jax.make_jaxpr(fn)(*args).jaxpr)[:16]`) and use that hash as a cache-key suffix. Any shape / dtype / flag change that shows up in the jaxpr invalidates.

## Why it matters for TPU perf

Naive cache keys (e.g., `(dtype, shape)` tuples) miss:
- Static-arg flag changes (`return_argmax=True` vs `False` produces a different graph).
- Soft-cap / bias / mask_fn changes that don't change argument shapes.
- Any refactor that adds/removes an op in the hot loop.

Stale-cache-hit on a jaxpr-changed graph produces measurably bad block sizes. The hash-based key prevents that.

## Mechanism

```python
def _autotune_jaxpr_hash(fn, *example_args) -> str:
    return sha256(str(jax.make_jaxpr(fn)(*example_args).jaxpr).encode()).hexdigest()[:16]
cache_key = f"{impl}|{backend}|{device_kind}|{...shapes...}|{jaxpr_hash}"
```

Cache entry written under `<jax_compilation_cache_dir>/<kernel_name>/<cache_key>` — same bucket as PJRT compile cache, so every training job shares tunings transparently.

## When it applies / when it doesn't

- **Applies** to any autotune harness serving multiple shape / flag / config combinations across training jobs.
- **Does not apply** when the tuner runs per-invocation (no cross-job cache) or when the graph is trivially parameterized.

## Known results

| Reference | Repo | Notes |
|---|---|---|
| `_autotune_jaxpr_hash` in `fused_cross_entropy_loss/api.py` | [marin](../codebases/marin.md) | Canonical impl; GCS-persistent; shared with PJRT compile cache |

## Connections

- [autotuning](autotuning.md)
- [compile-time-aware-autotune-filtering](compile-time-aware-autotune-filtering.md)
- [vmem-oom-fallthrough](vmem-oom-fallthrough.md)

## Sources

- [marin codebase](../codebases/marin.md) "Performance-relevant surfaces §1".
- [Pallas kernel directory §5.8](../analyses/pallas-kernel-directory/05-frameworks-quant.md#58-marin-communitymarin-vendors-levanter).
