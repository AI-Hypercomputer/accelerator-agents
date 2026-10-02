---
title: 'Module: tpu_inference/tpu_info.py'
type: catalog
provenance: extracted
module: tpu_inference/tpu_info.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.tpu_info`/
symbols:
  get_tpu_metadata: get_tpu_metadata().
  get_tpu_type: get_tpu_type().
  logger: logger.
  get_node_name: get_node_name().
  get_node_worker_id: get_node_worker_id().
  get_tpu_vmem_size_bytes: get_tpu_vmem_size_bytes().
  get_num_cores_per_chip: get_num_cores_per_chip().
  get_num_chips: get_num_chips().
  GCE_TPU_ACCELERATOR_ENDPOINT: GCE_TPU_ACCELERATOR_ENDPOINT.
  GCE_TPU_HEADERS: GCE_TPU_HEADERS.
---
# Module: [`tpu_inference/tpu_info.py`](../../../../../raw/code/tpu-inference/tpu_inference/tpu_info.py)

## Functions
- `get_node_name()` — [`L56`](../../../../../raw/code/tpu-inference/tpu_inference/tpu_info.py#L56)
- `get_node_worker_id()` — [`L63`](../../../../../raw/code/tpu-inference/tpu_inference/tpu_info.py#L63) — For multi-host TPU VM, this returns the worker id for the current node.
- `get_num_chips()` — [`L80`](../../../../../raw/code/tpu-inference/tpu_inference/tpu_info.py#L80)
- `get_num_cores_per_chip()` — [`L73`](../../../../../raw/code/tpu-inference/tpu_inference/tpu_info.py#L73)
- `get_tpu_metadata(key: str = "")` — [`L30`](../../../../../raw/code/tpu-inference/tpu_inference/tpu_info.py#L30)
- `get_tpu_type()` — [`L49`](../../../../../raw/code/tpu-inference/tpu_inference/tpu_info.py#L49)
- `get_tpu_vmem_size_bytes()` — [`L95`](../../../../../raw/code/tpu-inference/tpu_inference/tpu_info.py#L95) — Returns the TPU Vector Memory (VMEM) size in Bytes per Tensor Core.

## Module values
- `GCE_TPU_ACCELERATOR_ENDPOINT` — [`L25`](../../../../../raw/code/tpu-inference/tpu_inference/tpu_info.py#L25)
- `GCE_TPU_HEADERS` — [`L27`](../../../../../raw/code/tpu-inference/tpu_inference/tpu_info.py#L27)
- `logger` — [`L23`](../../../../../raw/code/tpu-inference/tpu_inference/tpu_info.py#L23)

