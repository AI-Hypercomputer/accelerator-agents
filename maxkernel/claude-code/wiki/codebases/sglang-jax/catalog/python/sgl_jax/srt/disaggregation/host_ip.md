---
title: 'Module: python/sgl_jax/srt/disaggregation/host_ip.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/disaggregation/host_ip.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.disaggregation.host_ip`/
symbols:
  resolve_host_ip: resolve_host_ip().
  _validate: _validate().
  logger: logger.
  _REJECTED_HOSTNAMES: _REJECTED_HOSTNAMES.
---
# Module: [`python/sgl_jax/srt/disaggregation/host_ip.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/host_ip.py)

## Functions
- `_validate(ip: str, *, source: str)` — [`L67`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/host_ip.py#L67) — Reject bind/loopback addresses regardless of textual form.
- `resolve_host_ip(explicit: str | None = None, *, env_name: str = "HOSTNAME")` — [`L18`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/host_ip.py#L18) — Return the per-host IP a remote PD peer can dial.

## Module values
- `_REJECTED_HOSTNAMES` — [`L15`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/host_ip.py#L15)
- `logger` — [`L10`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/host_ip.py#L10)

