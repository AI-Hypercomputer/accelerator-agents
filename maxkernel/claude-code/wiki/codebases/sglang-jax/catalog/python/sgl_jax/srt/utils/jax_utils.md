---
title: 'Module: python/sgl_jax/srt/utils/jax_utils.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/utils/jax_utils.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.utils.jax_utils`/
symbols:
  device_array: device_array().
  get_num_kv_heads_by_tp: get_num_kv_heads_by_tp().
  get_available_device_memory: get_available_device_memory().
  is_tpu_runtime: is_tpu_runtime().
  get_device_name: get_device_name().
  get_device_hbm_limit: get_device_hbm_limit().
  _IS_TPU_RUNTIME_CACHED._IS_TPU_RUNTIME_CACHED: _IS_TPU_RUNTIME_CACHED._IS_TPU_RUNTIME_CACHED.
  get_original_kv_head_id: get_original_kv_head_id().
  get_available_device_memory.filter_devices: get_available_device_memory().filter_devices().
  GBYTES: GBYTES.
  pathways_hbm_usage_gb: pathways_hbm_usage_gb().
  device_array._to_device: device_array()._to_device().
  get_device_id_offset: get_device_id_offset().
  print_memory: print_memory().
  get_available_device_memory._get_available_memory_distributed: get_available_device_memory()._get_available_memory_distributed().
  device_array._to_device.fn: device_array()._to_device().fn().
  get_memory_usage: get_memory_usage().
  TPU_HEAD_SIZE_ALIGNMENT: TPU_HEAD_SIZE_ALIGNMENT.
  TPU_SECOND_LAST_MINOR: TPU_SECOND_LAST_MINOR.
---
# Module: [`python/sgl_jax/srt/utils/jax_utils.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/jax_utils.py)

## Functions
- `_get_available_memory_distributed(a)` — [`L182`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/jax_utils.py#L182)
- `_to_device(arr)` — [`L202`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/jax_utils.py#L202)
- `device_array(data, sharding=None, **kwargs)` — [`L198`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/jax_utils.py#L198)
- `filter_devices(device_list, device_indexes)` — [`L122`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/jax_utils.py#L122)
- `fn(idx, a=arr)` — [`L205`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/jax_utils.py#L205)
- `get_available_device_memory(device, distributed=False, empty_cache=True, device_indexes: list[int] | None = None)` — [`L114`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/jax_utils.py#L114) — Get available memory for device:device_id.
- `get_device_hbm_limit()` — [`L49`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/jax_utils.py#L49)
- `get_device_id_offset(devices)` — [`L17`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/jax_utils.py#L17)
- `get_device_name(num_devices: int | None = None)` — [`L26`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/jax_utils.py#L26)
- `get_memory_usage()` — [`L241`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/jax_utils.py#L241) — Get actual memory usage if available
- `get_num_kv_heads_by_tp(total_num_kv_heads: int, tp_size: int)` — [`L74`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/jax_utils.py#L74) — Calculate the number of KV heads per device for tensor parallelism.
- `get_original_kv_head_id(tp_rank: int, total_num_kv_heads: int, tp_size: int)` — [`L92`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/jax_utils.py#L92) — Determine which original KV head this device should replicate.
- `is_tpu_runtime()` — [`L216`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/jax_utils.py#L216) — Return True if the current JAX runtime is on TPU devices.
- `pathways_hbm_usage_gb(live_arrays, devices: Any)` — [`L65`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/jax_utils.py#L65)
- `print_memory(stage_name)` — [`L232`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/jax_utils.py#L232) — Print current memory usage

## Module values
- `GBYTES` — [`L11`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/jax_utils.py#L11)
- `TPU_HEAD_SIZE_ALIGNMENT` — [`L12`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/jax_utils.py#L12)
- `TPU_SECOND_LAST_MINOR` — [`L13`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/jax_utils.py#L13)
- `_IS_TPU_RUNTIME_CACHED` — [`L213`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/jax_utils.py#L213)

