---
title: "VMEM-OOM-aware autotune fallthrough"
type: concept
tags: [autotuning, vmem, tpu, oom, marin, levanter, stub]
created: 2026-04-23
updated: 2026-04-23
---

Catch `resource_exhausted ... vmem` errors during autotune candidate compilation and **demote** the candidate (move to next) instead of raising. First-party reference: marin/levanter `_is_tpu_vmem_compile_error`. *Stub — expand when more sources are available.*

## Definition

A Pallas TPU kernel's block-size candidates sometimes request more VMEM than the hardware has (96 MiB on v6, 48 MiB on v7). Lowering fails with `resource_exhausted` + `vmem` substring. A naive tuner surfaces this as a crash. The VMEM-OOM fallthrough catches it, issues a one-time warning, and moves to the next candidate.

## Why it matters for TPU perf

The alternative is pre-filtering candidates by static VMEM estimation — fragile because actual usage depends on compiler decisions. Let the compiler try, fall through when it doesn't fit. Keeps the candidate space wide without risking tuner abort.

## Mechanism

```python
def _is_tpu_vmem_compile_error(e: Exception) -> bool:
    message = str(e).lower()
    return "resource_exhausted" in message and "vmem" in message

try:
    candidate.compile()
except XlaRuntimeError as e:
    if _is_tpu_vmem_compile_error(e):
        _warn_vmem_compile_fallback_once(candidate)
        continue   # next candidate
    raise
```

Paired with `_warn_vmem_compile_fallback_once` to avoid log spam — warns per-candidate at most once.

## When it applies / when it doesn't

- **Applies** to any TPU Pallas autotune over block sizes that might exceed VMEM.
- **Does not apply** on GPU (SRAM fits differently) or when candidates are pre-filtered for VMEM.

## Known results

| Reference | Repo | Notes |
|---|---|---|
| `_is_tpu_vmem_compile_error` + `_warn_vmem_compile_fallback_once` in `fused_cross_entropy_loss/api.py` | [marin](../codebases/marin.md) | Canonical impl |

## Connections

- [vmem](vmem.md)
- [vmem-budget](vmem-budget.md)
- [autotuning](autotuning.md)
- [compile-time-aware-autotune-filtering](compile-time-aware-autotune-filtering.md)

## Sources

- [marin codebase](../codebases/marin.md) "Performance-relevant surfaces §1".
- [Pallas kernel directory §5.8](../analyses/pallas-kernel-directory/05-frameworks-quant.md#58-marin-communitymarin-vendors-levanter).
