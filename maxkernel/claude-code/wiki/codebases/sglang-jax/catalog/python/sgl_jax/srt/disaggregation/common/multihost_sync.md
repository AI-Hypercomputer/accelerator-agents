---
title: 'Module: python/sgl_jax/srt/disaggregation/common/multihost_sync.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/disaggregation/common/multihost_sync.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.disaggregation.common.multihost_sync`/
symbols:
  synced_terminal_rooms: synced_terminal_rooms().
  _SYNC_MAX_INFLIGHT: _SYNC_MAX_INFLIGHT.
---
# Module: [`python/sgl_jax/srt/disaggregation/common/multihost_sync.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/multihost_sync.py)

## Functions
- `synced_terminal_rooms(entries: Iterable, poll_fn: Callable[[object], KVPoll], room_fn: Callable[[object], int | None])` — [`L21`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/multihost_sync.py#L21) — Return ``(success_rooms, failed_rooms)`` agreed across all processes.

## Module values
- `_SYNC_MAX_INFLIGHT` — [`L18`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/multihost_sync.py#L18)

