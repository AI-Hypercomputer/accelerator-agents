---
title: 'Module: easydel/workers/esurge/auth/worker_manager.py'
type: catalog
provenance: extracted
module: easydel/workers/esurge/auth/worker_manager.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.workers.esurge.auth.worker_manager`/
symbols:
  AuthWorkerManager.start: AuthWorkerManager#start().
  AuthWorkerManager._spawn_auth_worker: AuthWorkerManager#_spawn_auth_worker().
  AuthWorkerManager.shutdown: AuthWorkerManager#shutdown().
  AuthWorkerManager._auth_client: AuthWorkerManager#_auth_client.
  logger: logger.
  AuthWorkerManager._auth_process: AuthWorkerManager#_auth_process.
  AuthWorkerManager._terminate_process: AuthWorkerManager#_terminate_process().
  AuthWorkerManager._wait_for_endpoint: AuthWorkerManager#_wait_for_endpoint().
  AuthWorkerManager._auth_endpoint: AuthWorkerManager#_auth_endpoint.
  AuthWorkerManager.auth_client: AuthWorkerManager#auth_client().
  AuthWorkerManager._cleanup_ipc_file: AuthWorkerManager#_cleanup_ipc_file().
  AuthWorkerManager._make_ipc_endpoint: AuthWorkerManager#_make_ipc_endpoint().
  AuthWorkerManager._auth_owned: AuthWorkerManager#_auth_owned.
  AuthWorkerManager.auth_endpoint: AuthWorkerManager#auth_endpoint().
  AuthWorkerManager._admin_key: AuthWorkerManager#_admin_key.
  AuthWorkerManager._max_audit_entries: AuthWorkerManager#_max_audit_entries.
  AuthWorkerManager._storage_dir: AuthWorkerManager#_storage_dir.
  AuthWorkerManager._auto_save_interval: AuthWorkerManager#_auto_save_interval.
  AuthWorkerManager._ipc_dir: AuthWorkerManager#_ipc_dir.
  AuthWorkerManager: AuthWorkerManager#
  AuthWorkerManager._require_api_key: AuthWorkerManager#_require_api_key.
  AuthWorkerManager._enable_audit_logging: AuthWorkerManager#_enable_audit_logging.
  AuthWorkerManager._enable_persistence: AuthWorkerManager#_enable_persistence.
  AuthWorkerManager._startup_timeout: AuthWorkerManager#_startup_timeout.
  AuthWorkerManager.__init__: AuthWorkerManager#__init__().
---
# Module: [`easydel/workers/esurge/auth/worker_manager.py`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py)

## Classes
### `AuthWorkerManager`
- def: [`easydel/workers/esurge/auth/worker_manager.py:41`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L41)
- doc: Spawns and manages auth worker process and client.
- signature: `class AuthWorkerManager:`
- members:
  - `__init__(self, *, require_api_key: bool = False, admin_key: str | None = None, enable_audit_logging: bool = True, max_audit_entries: int = 10000, storage_dir: str | None = None, enable_persistence: bool = True, auto_save_interval: float = 60, startup_timeout: float = 30, ipc_dir: str | None = None)` — [`L49`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L49) — Initialize the auth worker manager.
  - `_cleanup_ipc_file(self, endpoint: str | None)` — [`L254`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L254) — Clean up IPC socket file.
  - `_make_ipc_endpoint(self, prefix: str)` — [`L227`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L227) — Create a unique IPC endpoint.
  - `_spawn_auth_worker(self, endpoint: str)` — [`L154`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L154) — Spawn auth worker process.
  - `_terminate_process(self)` — [`L240`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L240) — Terminate the auth worker process.
  - `_wait_for_endpoint(self, endpoint: str, process: subprocess.Popen | None)` — [`L200`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L200) — Wait for the worker to bind to the endpoint.
  - `auth_client(self)` — [`L96`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L96) — Get the auth worker client.
  - `auth_endpoint(self)` — [`L91`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L91) — Get the auth worker endpoint.
  - `shutdown(self)` — [`L132`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L132) — Shutdown the auth worker and clean up resources.
  - `start(self, *, auth_endpoint: str | None = None)` — [`L100`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L100) — Start auth worker process.
- protocol/private: `_admin_key`[`L76`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L76), `_auth_client`[`L85`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L85), `_auth_endpoint`[`L88`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L88), `_auth_owned`[`L87`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L87), `_auth_process`[`L86`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L86), `_auto_save_interval`[`L81`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L81), `_enable_audit_logging`[`L77`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L77), `_enable_persistence`[`L80`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L80), `_ipc_dir`[`L83`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L83), `_max_audit_entries`[`L78`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L78), `_require_api_key`[`L75`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L75), `_startup_timeout`[`L82`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L82), `_storage_dir`[`L79`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L79)
- uses (calls/refs, reference-scoped): [`logger`](worker_manager.md#logger), [`AuthWorkerClient`](zmq_client.md#AuthWorkerClient), [`shutdown`](zmq_client.md#AuthWorkerClient.shutdown), [`close`](zmq_client.md#AuthWorkerClient.close)

## Module values
- `logger` — [`L38`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/worker_manager.py#L38)

