---
title: 'Module: easydel/inference/esurge/distributed/protocol.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/distributed/protocol.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.distributed.protocol`/
symbols:
  _canonicalize: _canonicalize().
  compute_sampled_digest: compute_sampled_digest().
  STATUS_OK: STATUS_OK.
  make_config_fingerprint: make_config_fingerprint().
  CMD_SHUTDOWN: CMD_SHUTDOWN.
  STATUS_ERROR: STATUS_ERROR.
  _hash_payload: _hash_payload().
  CMD_HELLO: CMD_HELLO.
  CMD_HEALTH: CMD_HEALTH.
  CMD_STEP: CMD_STEP.
---
# Module: [`easydel/inference/esurge/distributed/protocol.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/protocol.py)

## Functions
- `_canonicalize(value: tp.Any)` — [`L52`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/protocol.py#L52) — Convert an arbitrary Python value into a stable, JSON-serializable structure.
- `_hash_payload(payload: tp.Any)` — [`L109`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/protocol.py#L109) — Return the SHA-256 hex digest of *payload* after canonicalization.
- `compute_sampled_digest(req_ids: list[str], sampled_token_ids: list[list[int]])` — [`L149`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/protocol.py#L149) — Hash request IDs and sampled token IDs to verify lockstep sampling consistency.
- `make_config_fingerprint(config: Mapping[str, tp.Any])` — [`L131`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/protocol.py#L131) — Build a stable SHA-256 fingerprint of an engine configuration mapping.

## Module values
- `CMD_HEALTH` — [`L44`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/protocol.py#L44)
- `CMD_HELLO` — [`L43`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/protocol.py#L43)
- `CMD_SHUTDOWN` — [`L46`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/protocol.py#L46)
- `CMD_STEP` — [`L45`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/protocol.py#L45)
- `STATUS_ERROR` — [`L49`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/protocol.py#L49)
- `STATUS_OK` — [`L48`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/protocol.py#L48)

