---
title: 'Module: easydel/inference/esurge/distributed/worker_server.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/distributed/worker_server.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.distributed.worker_server`/WorkerControlServer#
symbols:
  WorkerControlServer._serve_loop: _serve_loop().
  WorkerControlServer.start: start().
  WorkerControlServer.stop: stop().
  WorkerControlServer.endpoint: endpoint().
  WorkerControlServer._thread: _thread.
  WorkerControlServer.is_running: is_running().
  WorkerControlServer._is_authorized: _is_authorized().
  WorkerControlServer: ''
  WorkerControlServer._rank: _rank.
  WorkerControlServer._stop_event: _stop_event.
  WorkerControlServer._ready_event: _ready_event.
  WorkerControlServer._last_step_id: _last_step_id.
  WorkerControlServer._last_error: _last_error.
  WorkerControlServer._auth_token: _auth_token.
  WorkerControlServer._world_size: _world_size.
  WorkerControlServer._steps_executed: _steps_executed.
  WorkerControlServer._bind_host: _bind_host.
  WorkerControlServer._port: _port.
  WorkerControlServer._config_fingerprint: _config_fingerprint.
  WorkerControlServer._execute_step: _execute_step.
  WorkerControlServer._extract_runner_output: _extract_runner_output().
  WorkerControlServer.__init__: __init__().
---
# Module: [`easydel/inference/esurge/distributed/worker_server.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/worker_server.py)

## Classes
### `WorkerControlServer`
- def: [`easydel/inference/esurge/distributed/worker_server.py:44`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/worker_server.py#L44)
- doc: ZeroMQ REP server running on a worker rank for lockstep step execution.
- signature: `class WorkerControlServer:`
- members:
  - `__init__(self, *, bind_host: str, port: int, auth_token: str, rank: int, world_size: int, config_fingerprint: str, execute_step: tp.Callable[[tp.Any], tp.Any])` — [`L72`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/worker_server.py#L72) — Initialize the worker control server with binding and auth configuration.
  - `_extract_runner_output(model_output: tp.Any)` — [`L176`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/worker_server.py#L176) — Extract ``(req_ids, sampled_token_ids)`` from a model output.
  - `_is_authorized(self, message: dict[str, tp.Any])` — [`L171`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/worker_server.py#L171) — Check whether *message* carries the expected auth token.
  - `_serve_loop(self)` — [`L198`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/worker_server.py#L198) — Main server loop executed in the daemon thread.
  - `endpoint(self)` — [`L114`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/worker_server.py#L114) — The ``tcp://<host>:<port>`` address this server is bound to.
  - `is_running(self)` — [`L119`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/worker_server.py#L119) — Whether the server thread is currently alive.
  - `start(self)` — [`L123`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/worker_server.py#L123) — Start the server thread and block until the socket is ready.
  - `stop(self)` — [`L141`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/worker_server.py#L141) — Gracefully stop the server by sending a shutdown command to itself.
- protocol/private: `_auth_token`[`L99`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/worker_server.py#L99), `_bind_host`[`L97`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/worker_server.py#L97), `_config_fingerprint`[`L102`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/worker_server.py#L102), `_execute_step`[`L103`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/worker_server.py#L103), `_last_error`[`L111`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/worker_server.py#L111), `_last_step_id`[`L110`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/worker_server.py#L110), `_port`[`L98`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/worker_server.py#L98), `_rank`[`L100`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/worker_server.py#L100), `_ready_event`[`L107`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/worker_server.py#L107), `_steps_executed`[`L109`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/worker_server.py#L109), `_stop_event`[`L106`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/worker_server.py#L106), `_thread`[`L105`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/worker_server.py#L105), `_world_size`[`L101`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/distributed/worker_server.py#L101)
- uses (calls/refs, reference-scoped): [`compute_sampled_digest`](protocol.md#compute_sampled_digest), [`STATUS_OK`](protocol.md#STATUS_OK), [`CMD_SHUTDOWN`](protocol.md#CMD_SHUTDOWN), [`STATUS_ERROR`](protocol.md#STATUS_ERROR), [`CMD_HEALTH`](protocol.md#CMD_HEALTH), [`CMD_HELLO`](protocol.md#CMD_HELLO), [`CMD_STEP`](protocol.md#CMD_STEP)
- used by: [`start`](controller.md#DistributedController.start), [`shutdown`](controller.md#DistributedController.shutdown), [`_worker_server`](controller.md#DistributedController._worker_server)

