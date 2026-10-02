---
title: 'Module: simply/serving/page_server.py'
type: catalog
provenance: extracted
module: simply/serving/page_server.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.serving.page_server`/
symbols:
  main: main().
  SimplyService.Run: SimplyService#Run().
  SimplyService.batcher_thread: SimplyService#batcher_thread().
  SimplyService.batcher: SimplyService#batcher.
  set_notes: set_notes().
  SimplyServiceResponse: SimplyServiceResponse.
  SimplyService.error_message_queue: SimplyService#error_message_queue().
  PyTree: PyTree.
  SimplyService.stop_event: SimplyService#stop_event().
  _SIMPLY_PORT: _SIMPLY_PORT.
  SimplyService: SimplyService#
  SimplyService._done_callback: SimplyService#_done_callback().
---
# Module: [`simply/serving/page_server.py`](../../../../../../raw/code/simply/simply/serving/page_server.py)

## Classes
### `SimplyService`
- def: [`simply/serving/page_server.py:79`](../../../../../../raw/code/simply/simply/serving/page_server.py#L79)
- doc: The Simple service with batching.
- signature: `class SimplyService(server_pb2_grpc.SimplyService):`
- members:
  - `Run(self, request: struct_pb2.Value, context: grpc.aio.ServicerContext)` — [`L96`](../../../../../../raw/code/simply/simply/serving/page_server.py#L96)
  - `batcher_thread(self)` — [`L93`](../../../../../../raw/code/simply/simply/serving/page_server.py#L93)
  - `error_message_queue(self)` — [`L89`](../../../../../../raw/code/simply/simply/serving/page_server.py#L89)
  - `stop_event(self)` — [`L85`](../../../../../../raw/code/simply/simply/serving/page_server.py#L85)
  - `batcher` — [`L82`](../../../../../../raw/code/simply/simply/serving/page_server.py#L82)
- protocol/private: `_done_callback`[`L117`](../../../../../../raw/code/simply/simply/serving/page_server.py#L117)
- uses (calls/refs, reference-scoped): [`dump`](../utils/pytree.md#dump), [`enqueue`](page_batcher.md#Batcher.enqueue), [`py_to_struct_pb`](common.md#py_to_struct_pb), [`struct_pb_to_py`](common.md#struct_pb_to_py), [`code`](common.md#SimplyServiceResponse.code), [`SimplyServiceResponse`](page_server.md#SimplyServiceResponse), [`thread`](page_batcher.md#Batcher.thread), [`result`](common.md#SimplyServiceResponse.result), [`Batcher`](page_batcher.md#Batcher), [`details`](common.md#SimplyServiceResponse.details)
- used by: [`main`](page_server.md#main)

## Functions
- `main(argv: Sequence[str])` — [`L134`](../../../../../../raw/code/simply/simply/serving/page_server.py#L134)
- `set_notes(notes: str)` — [`L69`](../../../../../../raw/code/simply/simply/serving/page_server.py#L69)

## Module values
- `PyTree` — [`L66`](../../../../../../raw/code/simply/simply/serving/page_server.py#L66)
- `SimplyServiceResponse` — [`L75`](../../../../../../raw/code/simply/simply/serving/page_server.py#L75)
- `_SIMPLY_PORT` — [`L62`](../../../../../../raw/code/simply/simply/serving/page_server.py#L62)

