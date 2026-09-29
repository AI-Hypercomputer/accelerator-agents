---
title: 'Module: python/sgl_jax/srt/disaggregation/pd_auth.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/disaggregation/pd_auth.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.disaggregation.pd_auth`/
symbols:
  resolve_secret: resolve_secret().
  verify_tag: verify_tag().
  compute_tag: compute_tag().
  bearer_header: bearer_header().
  verify_bearer: verify_bearer().
  _ENV_VAR: _ENV_VAR.
---
# Module: [`python/sgl_jax/srt/disaggregation/pd_auth.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/pd_auth.py)

## Functions
- `bearer_header(secret: str | None)` — [`L45`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/pd_auth.py#L45)
- `compute_tag(secret: str, payload: bytes)` — [`L32`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/pd_auth.py#L32)
- `resolve_secret(server_args_value: str | None)` — [`L25`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/pd_auth.py#L25)
- `verify_bearer(secret: str | None, header_value: str | None)` — [`L51`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/pd_auth.py#L51)
- `verify_tag(secret: str | None, payload: bytes, candidate: bytes | None)` — [`L36`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/pd_auth.py#L36)

## Module values
- `_ENV_VAR` — [`L22`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/pd_auth.py#L22)

