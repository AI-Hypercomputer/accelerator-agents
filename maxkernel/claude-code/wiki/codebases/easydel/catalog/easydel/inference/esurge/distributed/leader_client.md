---
title: 'Module: easydel/inference/esurge/distributed/leader_client.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/distributed/leader_client.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.distributed.leader_client`/WorkerRpcClient#
symbols:
  WorkerRpcClient.begin_step: begin_step().
  WorkerRpcClient._create_socket: _create_socket().
  WorkerRpcClient._socket: _socket.
  WorkerRpcClient.finish_step: finish_step().
  WorkerRpcClient.reset_connection: reset_connection().
  WorkerRpcClient.hello: hello().
  WorkerRpcClient.shutdown: shutdown().
  WorkerRpcClient._request: _request().
  WorkerRpcClient.health: health().
  WorkerRpcClient.close: close().
  WorkerRpcClient._inflight_step_id: _inflight_step_id.
  WorkerRpcClient: ''
  WorkerRpcClient.endpoint: endpoint.
  WorkerRpcClient._auth_token: _auth_token.
  WorkerRpcClient.has_inflight_step: has_inflight_step().
  WorkerRpcClient._step_timeout_ms: _step_timeout_ms.
  WorkerRpcClient._connect_timeout_ms: _connect_timeout_ms.
  WorkerRpcClient._context: _context.
  WorkerRpcClient.__init__: __init__().
---
# Module: [`easydel/inference/esurge/distributed/leader_client.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/leader_client.py)

## Classes
### `WorkerRpcClient`
- def: [`easydel/inference/esurge/distributed/leader_client.py:37`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/leader_client.py#L37)
- doc: Leader-side ZeroMQ REQ client for a single worker control-plane endpoint.
- signature: `class WorkerRpcClient:`
- members:
  - `__init__(self, *, endpoint: str, auth_token: str, connect_timeout_s: float, step_timeout_s: float)` — [`L63`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/leader_client.py#L63) — Initialize the RPC client and connect to a worker endpoint.
  - `_create_socket(self)` — [`L92`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/leader_client.py#L92) — Create and connect a configured REQ socket for this worker.
  - `_request(self, payload: dict[str, tp.Any])` — [`L106`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/leader_client.py#L106) — Send *payload* and block until a dict response is received.
  - `begin_step(self, *, step_id: int, scheduler_output: tp.Any)` — [`L137`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/leader_client.py#L137) — Send a step command to the worker without waiting for the response.
  - `close(self)` — [`L210`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/leader_client.py#L210) — Close the underlying ZeroMQ socket immediately.
  - `finish_step(self)` — [`L166`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/leader_client.py#L166) — Block until the worker completes the in-flight step and return its result.
  - `has_inflight_step(self)` — [`L102`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/leader_client.py#L102) — Whether a step request has been sent and not yet collected.
  - `health(self)` — [`L128`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/leader_client.py#L128) — Query the worker's health status.
  - `hello(self)` — [`L119`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/leader_client.py#L119) — Perform the initial handshake with the worker.
  - `reset_connection(self)` — [`L198`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/leader_client.py#L198) — Reset the underlying socket and clear in-flight state.
  - `shutdown(self)` — [`L188`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/leader_client.py#L188) — Send a graceful shutdown command to the worker.
  - `endpoint` — [`L82`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/leader_client.py#L82)
- protocol/private: `_auth_token`[`L83`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/leader_client.py#L83), `_connect_timeout_ms`[`L85`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/leader_client.py#L85), `_context`[`L87`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/leader_client.py#L87), `_inflight_step_id`[`L90`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/leader_client.py#L90), `_socket`[`L88`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/leader_client.py#L88), `_step_timeout_ms`[`L84`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/leader_client.py#L84)
- uses (calls/refs, reference-scoped): [`CMD_SHUTDOWN`](protocol.md#CMD_SHUTDOWN), [`CMD_HEALTH`](protocol.md#CMD_HEALTH), [`CMD_HELLO`](protocol.md#CMD_HELLO), [`CMD_STEP`](protocol.md#CMD_STEP)
- used by: [`start`](controller.md#DistributedController.start), [`dispatch_step`](controller.md#DistributedController.dispatch_step), [`shutdown`](controller.md#DistributedController.shutdown), [`verify_step`](controller.md#DistributedController.verify_step), [`_worker_clients`](controller.md#DistributedController._worker_clients)

