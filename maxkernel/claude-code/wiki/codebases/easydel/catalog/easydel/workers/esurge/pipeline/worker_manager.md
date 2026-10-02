---
title: 'Module: easydel/workers/esurge/pipeline/worker_manager.py'
type: catalog
provenance: extracted
module: easydel/workers/esurge/pipeline/worker_manager.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.workers.esurge.pipeline.worker_manager`/
symbols:
  WorkerManager.start: WorkerManager#start().
  WorkerManager.drain_workers: WorkerManager#drain_workers().
  WorkerManager.shutdown: WorkerManager#shutdown().
  WorkerManager._resolve_startup_timeout: WorkerManager#_resolve_startup_timeout().
  WorkerManager._spawn_worker: WorkerManager#_spawn_worker().
  WorkerManager._shutdown_client: WorkerManager#_shutdown_client().
  WorkerManager._tokenizer_client: WorkerManager#_tokenizer_client.
  WorkerManager._detokenizer_client: WorkerManager#_detokenizer_client.
  WorkerManager._startup_timeout: WorkerManager#_startup_timeout.
  logger: logger.
  WorkerManager._tokenizer_endpoint: WorkerManager#_tokenizer_endpoint.
  WorkerManager._detokenizer_endpoint: WorkerManager#_detokenizer_endpoint.
  WorkerManager._wait_for_endpoint: WorkerManager#_wait_for_endpoint().
  WorkerManager._make_ipc_endpoint: WorkerManager#_make_ipc_endpoint().
  WorkerManager.tokenizer_endpoint: WorkerManager#tokenizer_endpoint().
  WorkerManager.detokenizer_endpoint: WorkerManager#detokenizer_endpoint().
  WorkerManager: WorkerManager#
  WorkerManager._terminate_process: WorkerManager#_terminate_process().
  WorkerManager._cleanup_ipc_file: WorkerManager#_cleanup_ipc_file().
  WorkerManager._tokenizer_source: WorkerManager#_tokenizer_source.
  WorkerManager._ipc_dir: WorkerManager#_ipc_dir.
  WorkerManager._tokenizer_process: WorkerManager#_tokenizer_process.
  WorkerManager._detokenizer_process: WorkerManager#_detokenizer_process.
  DEFAULT_WORKER_STARTUP_TIMEOUT: DEFAULT_WORKER_STARTUP_TIMEOUT.
  _WORKER_STARTUP_TIMEOUT_ENV_VARS: _WORKER_STARTUP_TIMEOUT_ENV_VARS.
  WorkerManager._tokenizer_kwargs: WorkerManager#_tokenizer_kwargs.
  WorkerManager._tokenizer_owned: WorkerManager#_tokenizer_owned.
  WorkerManager._detokenizer_owned: WorkerManager#_detokenizer_owned.
  WorkerManager.__init__: WorkerManager#__init__().
---
# Module: [`easydel/workers/esurge/pipeline/worker_manager.py`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py)

## Classes
### `WorkerManager`
- def: [`easydel/workers/esurge/pipeline/worker_manager.py:49`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L49)
- doc: Spawns and manages tokenizer/detokenizer worker processes and clients.
- signature: `class WorkerManager:`
- members:
  - `__init__(self, tokenizer_source: str | None, *, tokenizer_kwargs: dict[str, Any] | None = None, startup_timeout: float | None = None, ipc_dir: str | None = None)` — [`L57`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L57) — Initialize the worker manager.
  - `detokenizer_endpoint(self)` — [`L99`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L99)
  - `drain_workers(self)` — [`L187`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L187) — Flush in-flight tokenizer/detokenizer state.
  - `shutdown(self)` — [`L174`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L174) — Shut down all managed workers and release resources.
  - `start(self, *, detokenizer_max_states: int, tokenizer_endpoint: str | None, detokenizer_endpoint: str | None)` — [`L102`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L102) — Start or connect to tokenizer and detokenizer workers.
  - `tokenizer_endpoint(self)` — [`L95`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L95)
- protocol/private: `_cleanup_ipc_file`[`L323`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L323), `_detokenizer_client`[`L83`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L83), `_detokenizer_endpoint`[`L92`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L92), `_detokenizer_owned`[`L89`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L89), `_detokenizer_process`[`L86`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L86), `_ipc_dir`[`L80`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L80), `_make_ipc_endpoint`[`L282`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L282), `_resolve_startup_timeout`[`L200`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L200), `_shutdown_client`[`L287`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L287), `_spawn_worker`[`L231`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L231), `_startup_timeout`[`L79`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L79), `_terminate_process`[`L311`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L311), `_tokenizer_client`[`L82`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L82), `_tokenizer_endpoint`[`L91`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L91), `_tokenizer_kwargs`[`L78`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L78), `_tokenizer_owned`[`L88`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L88), `_tokenizer_process`[`L85`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L85), `_tokenizer_source`[`L77`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L77), `_wait_for_endpoint`[`L261`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L261)
- uses (calls/refs, reference-scoped): [`DetokenizerWorkerClient`](zmq_workers.md#DetokenizerWorkerClient), [`TokenizerWorkerClient`](zmq_workers.md#TokenizerWorkerClient), [`logger`](worker_manager.md#logger), [`drain`](zmq_workers.md#TokenizerWorkerClient.drain), [`DEFAULT_WORKER_STARTUP_TIMEOUT`](worker_manager.md#DEFAULT_WORKER_STARTUP_TIMEOUT), [`_WORKER_STARTUP_TIMEOUT_ENV_VARS`](worker_manager.md#_WORKER_STARTUP_TIMEOUT_ENV_VARS)
- used by: [`_worker_startup_timeout`](../../../inference/esurge/esurge_engine.md#eSurge._worker_startup_timeout), [`__del__`](../../../inference/esurge/esurge_engine.md#eSurge.__del__), [`_worker_manager`](../../../inference/esurge/esurge_engine.md#eSurge._worker_manager), [`_detokenizer_client`](../../../inference/esurge/esurge_engine.md#eSurge._detokenizer_client), [`_detokenizer_endpoint`](../../../inference/esurge/esurge_engine.md#eSurge._detokenizer_endpoint), [`_tokenizer_endpoint`](../../../inference/esurge/esurge_engine.md#eSurge._tokenizer_endpoint)

## Module values
- `DEFAULT_WORKER_STARTUP_TIMEOUT` — [`L45`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L45)
- `_WORKER_STARTUP_TIMEOUT_ENV_VARS` — [`L46`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L46)
- `logger` — [`L43`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/worker_manager.py#L43)

