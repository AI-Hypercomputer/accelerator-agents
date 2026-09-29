---
title: 'Module: src/maxtext/common/vertex_tensorboard.py'
type: catalog
provenance: extracted
module: src/maxtext/common/vertex_tensorboard.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.common.vertex_tensorboard`/VertexTensorboardManager#
symbols:
  VertexTensorboardManager.configure_vertex_tensorboard: configure_vertex_tensorboard().
  VertexTensorboardManager.upload_data: upload_data().
  VertexTensorboardManager.uploader_flag: uploader_flag.
  VertexTensorboardManager.__del__: __del__().
  VertexTensorboardManager.setup: setup().
  VertexTensorboardManager: ''
  VertexTensorboardManager.__init__: __init__().
---
# Module: [`src/maxtext/common/vertex_tensorboard.py`](../../../../../../../raw/code/maxtext/src/maxtext/common/vertex_tensorboard.py)

## Classes
### `VertexTensorboardManager`
- def: [`src/maxtext/common/vertex_tensorboard.py:29`](../../../../../../../raw/code/maxtext/src/maxtext/common/vertex_tensorboard.py#L29)
- doc: Class to create Vertex AI Tensorboard and upload logs to that instance.
- signature: `class VertexTensorboardManager:`
- members:
  - `__del__(self)` — [`L35`](../../../../../../../raw/code/maxtext/src/maxtext/common/vertex_tensorboard.py#L35) — Stop the Tensorboard uploader thread.
  - `configure_vertex_tensorboard(self, config)` — [`L101`](../../../../../../../raw/code/maxtext/src/maxtext/common/vertex_tensorboard.py#L101) — Creates Vertex Tensorboard and start thread to upload data to Vertex Tensorboard.
  - `setup(self)` — [`L40`](../../../../../../../raw/code/maxtext/src/maxtext/common/vertex_tensorboard.py#L40) — Creates Tensorboard instance and Experiment in Vertex AI.
  - `upload_data(self, tensorboard_dir)` — [`L73`](../../../../../../../raw/code/maxtext/src/maxtext/common/vertex_tensorboard.py#L73) — Starts an uploader to continuously monitor and upload data to Vertex Tensorboard.
  - `uploader_flag` — [`L33`](../../../../../../../raw/code/maxtext/src/maxtext/common/vertex_tensorboard.py#L33)
- protocol/private: `__init__`[`L32`](../../../../../../../raw/code/maxtext/src/maxtext/common/vertex_tensorboard.py#L32)

