---
title: 'Module: simply/serving/common.py'
type: catalog
provenance: extracted
module: simply/serving/common.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.serving.common`/
symbols:
  json_to_struct_pb: json_to_struct_pb().
  struct_pb_to_json: struct_pb_to_json().
  py_to_struct_pb: py_to_struct_pb().
  struct_pb_to_py: struct_pb_to_py().
  SimplyServiceResponse.code: SimplyServiceResponse#code.
  SimplyServiceResponse.result: SimplyServiceResponse#result.
  SimplyServiceResponse: SimplyServiceResponse#
  SimplyServiceResponse.details: SimplyServiceResponse#details.
---
# Module: [`simply/serving/common.py`](../../../../../../raw/code/simply/simply/serving/common.py)

## Classes
### `SimplyServiceResponse`  ·  implements/extends NamedTuple
- def: [`simply/serving/common.py:24`](../../../../../../raw/code/simply/simply/serving/common.py#L24)
- signature: `class SimplyServiceResponse(NamedTuple):`
- members:
  - `code` — [`L25`](../../../../../../raw/code/simply/simply/serving/common.py#L25)
  - `details` — [`L26`](../../../../../../raw/code/simply/simply/serving/common.py#L26)
  - `result` — [`L27`](../../../../../../raw/code/simply/simply/serving/common.py#L27)
- used by: [`loop`](page_batcher.md#Batcher.loop), [`Run`](page_server.md#SimplyService.Run), [`_try_get_request`](page_batcher.md#Batcher._try_get_request), [`query_and_evaluate`](../eval/page_decode_eval.md#main.query_and_evaluate), [`SimplyServiceResponse`](page_batcher.md#SimplyServiceResponse), [`SimplyServiceResponse`](page_server.md#SimplyServiceResponse)

## Functions
- `json_to_struct_pb(jtree: common.PyTree)` — [`L30`](../../../../../../raw/code/simply/simply/serving/common.py#L30) — Converts a json-like tree to a struct_pb2.Value.
- `py_to_struct_pb(py: Any)` — [`L79`](../../../../../../raw/code/simply/simply/serving/common.py#L79) — Converts a python object to a struct_pb2.Value.
- `struct_pb_to_json(struct: struct_pb2.Value)` — [`L56`](../../../../../../raw/code/simply/simply/serving/common.py#L56) — Converts a struct_pb2.Value to a json-like tree.
- `struct_pb_to_py(struct: struct_pb2.Value)` — [`L84`](../../../../../../raw/code/simply/simply/serving/common.py#L84) — Converts a struct_pb2.Value to a python object.

