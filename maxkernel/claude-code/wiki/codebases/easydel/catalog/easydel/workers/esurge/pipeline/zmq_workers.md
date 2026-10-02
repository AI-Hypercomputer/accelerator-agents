---
title: 'Module: easydel/workers/esurge/pipeline/zmq_workers.py'
type: catalog
provenance: extracted
module: easydel/workers/esurge/pipeline/zmq_workers.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.workers.esurge.pipeline.zmq_workers`/
symbols:
  _BaseWorkerClient._request: _BaseWorkerClient#_request().
  _BaseWorkerClient._socket: _BaseWorkerClient#_socket.
  TokenizerWorkerClient: TokenizerWorkerClient#
  DetokenizerWorkerClient: DetokenizerWorkerClient#
  _BaseWorkerClient: _BaseWorkerClient#
  DetokenizerResult: DetokenizerResult#
  TokenizerWorkerClient.shutdown: TokenizerWorkerClient#shutdown().
  DetokenizerWorkerClient.decode: DetokenizerWorkerClient#decode().
  DetokenizerWorkerClient.shutdown: DetokenizerWorkerClient#shutdown().
  _BaseWorkerClient.close: _BaseWorkerClient#close().
  TokenizerWorkerClient.drain: TokenizerWorkerClient#drain().
  TokenizerWorkerClient.__init__: TokenizerWorkerClient#__init__().
  TokenizerWorkerClient.tokenize: TokenizerWorkerClient#tokenize().
  DetokenizerWorkerClient.__init__: DetokenizerWorkerClient#__init__().
  DetokenizerWorkerClient.reset: DetokenizerWorkerClient#reset().
  DetokenizerWorkerClient.drain: DetokenizerWorkerClient#drain().
  _BaseWorkerClient.__init__: _BaseWorkerClient#__init__().
  _BaseWorkerClient._context: _BaseWorkerClient#_context.
  _BaseWorkerClient._lock: _BaseWorkerClient#_lock.
  DetokenizerResult.accumulated_text: DetokenizerResult#accumulated_text.
  DetokenizerResult.delta_text: DetokenizerResult#delta_text.
  DetokenizerResult.last_decoded_index: DetokenizerResult#last_decoded_index.
  DetokenizerResult.finished: DetokenizerResult#finished.
  DetokenizerResult.detoktook: DetokenizerResult#detoktook.
---
# Module: [`easydel/workers/esurge/pipeline/zmq_workers.py`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py)

## Classes
### `DetokenizerResult`
- def: [`easydel/workers/esurge/pipeline/zmq_workers.py:36`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L36)
- doc: Result from a detokenization operation.
- signature: `class DetokenizerResult:`
- members:
  - `accumulated_text` — [`L46`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L46)
  - `delta_text` — [`L47`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L47)
  - `detoktook` — [`L50`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L50)
  - `finished` — [`L49`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L49)
  - `last_decoded_index` — [`L48`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L48)
- used by: [`_decode_with_pipeline`](../../../inference/esurge/mixins/utils.md#EngineUtilsMixin._decode_with_pipeline), [`decode`](zmq_workers.md#DetokenizerWorkerClient.decode)

### `DetokenizerWorkerClient`  ·  implements/extends _BaseWorkerClient
- def: [`easydel/workers/esurge/pipeline/zmq_workers.py:133`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L133)
- doc: Client for communicating with a detokenizer worker process.
- signature: `class DetokenizerWorkerClient(_BaseWorkerClient):`
- members:
  - `decode(self, request_id: str, generated_tokens: list[int], *, finished: bool, skip_special_tokens: bool, spaces_between_special_tokens: bool = True, prompt_context: list[int] | None = None)` — [`L148`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L148) — Decode tokens incrementally.
  - `drain(self)` — [`L201`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L201) — Flush all detokenizer state (used during pause/resume).
  - `reset(self, request_id: str)` — [`L193`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L193) — Reset the decoding state for a request.
  - `shutdown(self)` — [`L205`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L205) — Shutdown the detokenizer worker and close the connection.
- protocol/private: `__init__`[`L143`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L143)
- uses (calls/refs, reference-scoped): [`_request`](zmq_workers.md#_BaseWorkerClient._request), [`_BaseWorkerClient`](zmq_workers.md#_BaseWorkerClient), [`DetokenizerResult`](zmq_workers.md#DetokenizerResult), [`close`](zmq_workers.md#_BaseWorkerClient.close), [`__init__`](zmq_workers.md#_BaseWorkerClient.__init__)
- used by: [`start`](worker_manager.md#WorkerManager.start), [`_BaseWorkerClient`](zmq_workers.md#_BaseWorkerClient), [`_detokenizer_client`](worker_manager.md#WorkerManager._detokenizer_client)

### `TokenizerWorkerClient`  ·  implements/extends _BaseWorkerClient
- def: [`easydel/workers/esurge/pipeline/zmq_workers.py:86`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L86)
- doc: Client for communicating with a tokenizer worker process.
- signature: `class TokenizerWorkerClient(_BaseWorkerClient):`
- members:
  - `drain(self)` — [`L119`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L119) — Ensure all tokenizer-side buffers are flushed.
  - `shutdown(self)` — [`L123`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L123) — Shutdown the tokenizer worker and close the connection.
  - `tokenize(self, request_id: str, prompt: str)` — [`L101`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L101) — Tokenize a text prompt.
- protocol/private: `__init__`[`L96`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L96)
- uses (calls/refs, reference-scoped): [`_request`](zmq_workers.md#_BaseWorkerClient._request), [`_BaseWorkerClient`](zmq_workers.md#_BaseWorkerClient), [`close`](zmq_workers.md#_BaseWorkerClient.close), [`__init__`](zmq_workers.md#_BaseWorkerClient.__init__)
- used by: [`start`](worker_manager.md#WorkerManager.start), [`drain_workers`](worker_manager.md#WorkerManager.drain_workers), [`_BaseWorkerClient`](zmq_workers.md#_BaseWorkerClient), [`_tokenizer_client`](worker_manager.md#WorkerManager._tokenizer_client)

### `_BaseWorkerClient`
- def: [`easydel/workers/esurge/pipeline/zmq_workers.py:53`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L53)
- doc: Base class for ZeroMQ worker clients.
- signature: `class _BaseWorkerClient:`
- members:
  - `_request(self, payload: dict)` — [`L68`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L68) — Send a request to the worker and return the response.
  - `close(self)` — [`L81`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L81) — Close the ZeroMQ socket.
- protocol/private: `__init__`[`L62`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L62), `_context`[`L63`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L63), `_lock`[`L66`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L66), `_socket`[`L64`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/pipeline/zmq_workers.py#L64)
- uses (calls/refs, reference-scoped): [`DetokenizerWorkerClient`](zmq_workers.md#DetokenizerWorkerClient), [`TokenizerWorkerClient`](zmq_workers.md#TokenizerWorkerClient)
- used by: [`DetokenizerWorkerClient`](zmq_workers.md#DetokenizerWorkerClient), [`TokenizerWorkerClient`](zmq_workers.md#TokenizerWorkerClient), [`decode`](zmq_workers.md#DetokenizerWorkerClient.decode), [`shutdown`](zmq_workers.md#DetokenizerWorkerClient.shutdown), [`shutdown`](zmq_workers.md#TokenizerWorkerClient.shutdown), [`drain`](zmq_workers.md#TokenizerWorkerClient.drain), [`__init__`](zmq_workers.md#DetokenizerWorkerClient.__init__), [`__init__`](zmq_workers.md#TokenizerWorkerClient.__init__), [`drain`](zmq_workers.md#DetokenizerWorkerClient.drain), [`reset`](zmq_workers.md#DetokenizerWorkerClient.reset), [`tokenize`](zmq_workers.md#TokenizerWorkerClient.tokenize)

