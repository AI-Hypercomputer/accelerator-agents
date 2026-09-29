---
title: axlearn.common.flash_attention.common — BaseFlashAttention's capability-negotiation contract
type: concept
provenance: mixed
concept: axlearn-common-flash_attention-common
updated: 2026-07-03
status: fresh
---
# axlearn.common.flash_attention.common — BaseFlashAttention's capability-negotiation contract

<!-- connect:up:begin -->
> **Cross-repo concept:** part of [flash-attention](../../../concepts/flash-attention.md) across this wiki's repos.
<!-- connect:up:end -->
## Overview

`BaseFlashAttention` defines a shared capability-check contract every Flash Attention backend
implementation (TPU Pallas, GPU cuDNN, ...) satisfies:
[`is_supported`](../catalog/axlearn/common/flash_attention/common.md#BaseFlashAttention.is_supported)
("Returns whether the attention kernel supports the given configuration") lets a caller ask, for a
given input batch shape and KV-cache type, whether a specific kernel backend can actually handle it —
rather than attempting the call and catching a runtime error. `BaseSingleStepDecoding.is_supported`
("See `BaseFlashAttention.is_supported`") extends this same check for the single-token decode case.
[`_log_unsupported`](../catalog/axlearn/common/flash_attention/common.md#BaseFlashAttention._log_unsupported)
("Logs this class is unsupported with `reason`") is the shared diagnostic every negative
`is_supported` result routes through.

## Diagram

```mermaid
flowchart TD
  CALLER["caller (e.g. attention layer selecting a kernel backend)"]
  CALLER -->|is_supported(input_batch, kv_cache_type)| CHECK["BaseFlashAttention.is_supported"]
  CHECK -->|_validate_input_batch| VALIDATE["shape/dtype/bias-type checks"]
  VALIDATE -->|fails| LOG["_log_unsupported(reason) -> False"]
  VALIDATE -->|passes| TRUE["True"]
  DECODE["BaseSingleStepDecoding.is_supported"] -->|delegates to| CHECK
  BACKEND["CuDNNGPUFlashAttention.__call__ / TPU Pallas backend"] -.->|only called after is_supported() == True| CHECK
```

## Design rationale (why it's built this way)

**`is_supported` is checked *before* calling the kernel, not wrapped in a try/except around the actual
call — capability negotiation is explicit, not exception-driven.**
[`BaseFlashAttention.is_supported`](../catalog/axlearn/common/flash_attention/common.md#BaseFlashAttention.is_supported)'s
doc — "Returns whether the attention kernel supports the given configuration" — takes exactly the
`input_batch`/`kv_cache_type` a real call would take, letting a dispatcher iterate over candidate
backends and pick the first one that returns `True`, rather than trying each backend and catching
failures.

**Every negative support result is logged with a specific reason via one shared method
(`_log_unsupported`), not silently returning `False`.**
[`_log_unsupported`](../catalog/axlearn/common/flash_attention/common.md#BaseFlashAttention._log_unsupported)
returns `Literal[False]` — its return type itself documents the calling convention: `return
self._log_unsupported(reason)` doubles as both the logging side effect and the boolean `False` result,
so every unsupported-configuration code path is self-documenting about *why* in the logs.

**`BaseSingleStepDecoding.is_supported` explicitly delegates to (or mirrors) `BaseFlashAttention`'s own
check rather than reimplementing validation, per its own doc pointing back — "See
`BaseFlashAttention.is_supported`"** — decode-time support-checking is a specialization of the same
general contract, not an independent one.

## Entry points

- [`BaseFlashAttention.is_supported`](../catalog/axlearn/common/flash_attention/common.md#BaseFlashAttention.is_supported) —
  the capability check every concrete backend (e.g.
  [`CuDNNGPUFlashAttention`](../catalog/axlearn/common/flash_attention/gpu_attention.md#CuDNNGPUFlashAttention.__call__))
  implements/inherits.
- [`BaseSingleStepDecoding.is_supported`](../catalog/axlearn/common/flash_attention/common.md#BaseSingleStepDecoding.is_supported) —
  the decode-specific variant of the same check.

## Mechanism (step-by-step)

1. **A caller (typically an attention layer selecting among available kernel backends) calls
   [`is_supported`](../catalog/axlearn/common/flash_attention/common.md#BaseFlashAttention.is_supported)`(input_batch, kv_cache_type)`
   on a candidate backend.**
2. **[`is_supported`](../catalog/axlearn/common/flash_attention/common.md#BaseFlashAttention.is_supported)
   validates the input batch's shapes/dtypes/bias types** via
   `_validate_input_batch`, checking compatibility with the given `kv_cache_type` (e.g.
   `KVCache`/`PagedKVCache`).
3. **On any validation failure,
   [`_log_unsupported`](../catalog/axlearn/common/flash_attention/common.md#BaseFlashAttention._log_unsupported)`(reason)`
   logs the specific reason and returns
   `False`** — the caller can try the next candidate backend.
4. **Only after `is_supported` returns `True` does the caller invoke the backend's actual `__call__`**
   (e.g.
   [`CuDNNGPUFlashAttention.__call__`](../catalog/axlearn/common/flash_attention/gpu_attention.md#CuDNNGPUFlashAttention.__call__),
   itself `jax.jit`-compiled with `static_argnames=["self"]`).

## Key data structures

- **`BaseFlashAttention`** — holds [`cfg`](../catalog/axlearn/common/flash_attention/common.md#BaseFlashAttention.cfg)
  (a `Config`); every concrete backend subclasses this to gain the shared `is_supported`/
  `_log_unsupported` contract.

## Dynamics (design intent)
Not addressable beyond the capability-check-before-call pattern described above from this packet's
subgraph.

## Edge cases
None directly visible in this packet's subgraph.

## Open questions
- The full enumeration of concrete `BaseFlashAttention` subclasses (TPU Pallas, GPU cuDNN, and any
  others) and their respective supported-configuration boundaries isn't fully resolved by the symbols
  in this packet's subgraph — only the GPU cuDNN backend's `__call__` is directly cited here.

## See also
- [axlearn-common-flash_attention-layer](axlearn-common-flash_attention-layer.md) — `FlashAttention`,
  the `MultiheadAttention`-integrating layer that presumably selects among `BaseFlashAttention`
  backends via this `is_supported` contract.
- [axlearn-common-attention_bias](axlearn-common-attention_bias.md) — `BaseAttentionBias`, part of the
  `input_batch` shape `is_supported` validates.
