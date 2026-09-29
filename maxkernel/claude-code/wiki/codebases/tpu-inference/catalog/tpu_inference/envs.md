---
title: 'Module: tpu_inference/envs.py'
type: catalog
provenance: extracted
module: tpu_inference/envs.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.envs`/
symbols:
  env_bool: env_bool().
  environment_variables.environment_variables: environment_variables.environment_variables.
  env_with_choices: env_with_choices().
  MOE_REQUANTIZE_WEIGHT_DTYPE: MOE_REQUANTIZE_WEIGHT_DTYPE.
  __getattr__: __getattr__().
  MOE_REQUANTIZE_BLOCK_SIZE: MOE_REQUANTIZE_BLOCK_SIZE.
  enable_envs_cache: enable_envs_cache().
  TPU_MULTIPROCESS_DP: TPU_MULTIPROCESS_DP.
  NEW_MODEL_DESIGN: NEW_MODEL_DESIGN.
  env_str_list: env_str_list().
  env_int_list: env_int_list().
  PREFILL_SLICES: PREFILL_SLICES.
  __dir__: __dir__().
  TPU_MULTIHOST_BACKEND: TPU_MULTIHOST_BACKEND.
  DECODE_SLICES: DECODE_SLICES.
  VLLM_XLA_CHECK_RECOMPILATION: VLLM_XLA_CHECK_RECOMPILATION.
  MODEL_IMPL_TYPE: MODEL_IMPL_TYPE.
  DRAFT_MODEL_IMPL_TYPE: DRAFT_MODEL_IMPL_TYPE.
  PHASED_PROFILING_DIR: PHASED_PROFILING_DIR.
  AGGREGATED_STATS_DIR: AGGREGATED_STATS_DIR.
  USE_MOE_EP_KERNEL: USE_MOE_EP_KERNEL.
  NUM_SLICES: NUM_SLICES.
  ATTN_CUSTOM_NUM_REQS_BUCKETS: ATTN_CUSTOM_NUM_REQS_BUCKETS.
  LAYOUT_Q_PROJ_AS_NDH: LAYOUT_Q_PROJ_AS_NDH.
  RAGGED_GATHER_VERSION: RAGGED_GATHER_VERSION.
  RAGGED_GATHER_REDUCE_VERSION: RAGGED_GATHER_REDUCE_VERSION.
  env_with_choices._get_validated_env: env_with_choices()._get_validated_env().
  env_bool._get_bool_env: env_bool()._get_bool_env().
  env_str_list._get_str_list_env: env_str_list()._get_str_list_env().
  env_int_list._get_int_list_env: env_int_list()._get_int_list_env().
  JAX_PLATFORMS: JAX_PLATFORMS.
  TPU_ACCELERATOR_TYPE: TPU_ACCELERATOR_TYPE.
  TPU_NAME: TPU_NAME.
  TPU_WORKER_ID: TPU_WORKER_ID.
  SKIP_JAX_PRECOMPILE: SKIP_JAX_PRECOMPILE.
  PYTHON_TRACER_LEVEL: PYTHON_TRACER_LEVEL.
  USE_UNFUSED_MEGABLOCKS: USE_UNFUSED_MEGABLOCKS.
  USE_DENSE_MOE: USE_DENSE_MOE.
  ENABLE_QUANTIZED_MATMUL_KERNEL: ENABLE_QUANTIZED_MATMUL_KERNEL.
  REQUANTIZE_BLOCK_SIZE: REQUANTIZE_BLOCK_SIZE.
  REQUANTIZE_WEIGHT_DTYPE: REQUANTIZE_WEIGHT_DTYPE.
  MOE_REQUANTIZE_CLIP_PERCENTILE: MOE_REQUANTIZE_CLIP_PERCENTILE.
  ATTN_BUCKETIZED_NUM_REQS: ATTN_BUCKETIZED_NUM_REQS.
  USE_BATCHED_RPA_KERNEL: USE_BATCHED_RPA_KERNEL.
  FORCE_MOE_RANDOM_ROUTING: FORCE_MOE_RANDOM_ROUTING.
  JITTED_MM_MODULE_KEYS: JITTED_MM_MODULE_KEYS.
  REGISTER_MM_MODULE_CUSTOM_PYTREE_CLASSES: REGISTER_MM_MODULE_CUSTOM_PYTREE_CLASSES.
  MOE_ALL_GATHER_ACTIVATION_DTYPE: MOE_ALL_GATHER_ACTIVATION_DTYPE.
  MOE_APPROX_TOPK: MOE_APPROX_TOPK.
  MOE_APPROX_TOPK_RECALL_TARGET: MOE_APPROX_TOPK_RECALL_TARGET.
  VLLM_TPU_PATCH_MM_EMBEDDINGS: VLLM_TPU_PATCH_MM_EMBEDDINGS.
  ENABLE_RS_KERNEL: ENABLE_RS_KERNEL.
  NUM_PRECOMPILE_WORKERS: NUM_PRECOMPILE_WORKERS.
  VLLM_MOE_CHUNK_SIZE: VLLM_MOE_CHUNK_SIZE.
  ONEHOT_MOE_PERMUTE_THRESHOLD: ONEHOT_MOE_PERMUTE_THRESHOLD.
  PROFILE_SINGLE_DEVICE: PROFILE_SINGLE_DEVICE.
  LORA_MODULE_PATH: LORA_MODULE_PATH.
  SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES: SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES.
  RAY_USAGE_STATS_ENABLED: RAY_USAGE_STATS_ENABLED.
  VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE.
  USE_JAX_PROFILER_SERVER: USE_JAX_PROFILER_SERVER.
  JAX_PROFILER_SERVER_PORT: JAX_PROFILER_SERVER_PORT.
  TPU_OFFLOAD_SKIP_JAX_PRECOMPILE: TPU_OFFLOAD_SKIP_JAX_PRECOMPILE.
  TPU_OFFLOAD_DECODE_SAVE: TPU_OFFLOAD_DECODE_SAVE.
  TPU_OFFLOAD_NUM_CPU_CHUNKS: TPU_OFFLOAD_NUM_CPU_CHUNKS.
  TPU_OFFLOAD_NUM_STAGING_BLOCKS: TPU_OFFLOAD_NUM_STAGING_BLOCKS.
  TPU_OFFLOAD_SAVE_THREADS: TPU_OFFLOAD_SAVE_THREADS.
  TPU_OFFLOAD_BATCHED_SAVE: TPU_OFFLOAD_BATCHED_SAVE.
  TPU_OFFLOAD_METRICS_LOG_INTERVAL: TPU_OFFLOAD_METRICS_LOG_INTERVAL.
  TPU_OFFLOAD_USE_UNPINNED_HOST: TPU_OFFLOAD_USE_UNPINNED_HOST.
  TPU_OFFLOAD_BLOCK_SIZE_BUCKETS: TPU_OFFLOAD_BLOCK_SIZE_BUCKETS.
  DP_SCHED_BATCH_PREFILL: DP_SCHED_BATCH_PREFILL.
  DP_SCHED_BATCH_PREFILL_FLUSH_TIMEOUT_MS: DP_SCHED_BATCH_PREFILL_FLUSH_TIMEOUT_MS.
---
# Module: [`tpu_inference/envs.py`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py)

## Functions
- `__dir__()` — [`L454`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L454)
- `__getattr__(name: str)` — [`L423`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L423) — Gets environment variables lazily.
- `_get_bool_env()` — [`L139`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L139)
- `_get_int_list_env()` — [`L194`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L194)
- `_get_str_list_env()` — [`L175`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L175)
- `_get_validated_env()` — [`L100`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L100)
- `enable_envs_cache()` — [`L435`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L435) — Enables caching of environment variables by wrapping the module's __getattr__
- `env_bool(env_name: str, default: bool | None = False, requires: list[str] | None = None)` — [`L125`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L125) — Accepts both numeric strings ("0", "1") and boolean strings
- `env_int_list(env_name: str)` — [`L185`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L185) — Accepts a comma-separated string and returns a list of strings.
- `env_str_list(env_name: str)` — [`L166`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L166) — Accepts a comma-separated string and returns a list of strings.
- `env_with_choices(env_name: str, default: str | None, choices: list[str] | Callable[[], list[str]], case_sensitive: bool = True, allow_csv: bool = False)` — [`L78`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L78) — Create a lambda that validates environment variable against allowed choices

## Module values
- `AGGREGATED_STATS_DIR` — [`L24`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L24)
- `ATTN_BUCKETIZED_NUM_REQS` — [`L38`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L38)
- `ATTN_CUSTOM_NUM_REQS_BUCKETS` — [`L39`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L39)
- `DECODE_SLICES` — [`L17`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L17)
- `DP_SCHED_BATCH_PREFILL` — [`L69`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L69)
- `DP_SCHED_BATCH_PREFILL_FLUSH_TIMEOUT_MS` — [`L70`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L70)
- `DRAFT_MODEL_IMPL_TYPE` — [`L21`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L21)
- `ENABLE_QUANTIZED_MATMUL_KERNEL` — [`L32`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L32)
- `ENABLE_RS_KERNEL` — [`L67`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L67)
- `FORCE_MOE_RANDOM_ROUTING` — [`L44`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L44)
- `JAX_PLATFORMS` — [`L10`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L10)
- `JAX_PROFILER_SERVER_PORT` — [`L42`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L42)
- `JITTED_MM_MODULE_KEYS` — [`L45`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L45)
- `LAYOUT_Q_PROJ_AS_NDH` — [`L40`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L40)
- `LORA_MODULE_PATH` — [`L74`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L74)
- `MODEL_IMPL_TYPE` — [`L20`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L20)
- `MOE_ALL_GATHER_ACTIVATION_DTYPE` — [`L54`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L54)
- `MOE_APPROX_TOPK` — [`L64`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L64)
- `MOE_APPROX_TOPK_RECALL_TARGET` — [`L65`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L65)
- `MOE_REQUANTIZE_BLOCK_SIZE` — [`L35`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L35)
- `MOE_REQUANTIZE_CLIP_PERCENTILE` — [`L37`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L37)
- `MOE_REQUANTIZE_WEIGHT_DTYPE` — [`L36`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L36)
- `NEW_MODEL_DESIGN` — [`L22`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L22)
- `NUM_PRECOMPILE_WORKERS` — [`L68`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L68)
- `NUM_SLICES` — [`L29`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L29)
- `ONEHOT_MOE_PERMUTE_THRESHOLD` — [`L72`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L72)
- `PHASED_PROFILING_DIR` — [`L23`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L23)
- `PREFILL_SLICES` — [`L16`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L16)
- `PROFILE_SINGLE_DEVICE` — [`L73`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L73)
- `PYTHON_TRACER_LEVEL` — [`L25`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L25)
- `RAGGED_GATHER_REDUCE_VERSION` — [`L53`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L53)
- `RAGGED_GATHER_VERSION` — [`L49`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L49)
- `RAY_USAGE_STATS_ENABLED` — [`L30`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L30)
- `REGISTER_MM_MODULE_CUSTOM_PYTREE_CLASSES` — [`L46`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L46)
- `REQUANTIZE_BLOCK_SIZE` — [`L33`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L33)
- `REQUANTIZE_WEIGHT_DTYPE` — [`L34`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L34)
- `SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES` — [`L75`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L75)
- `SKIP_JAX_PRECOMPILE` — [`L18`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L18)
- `TPU_ACCELERATOR_TYPE` — [`L11`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L11)
- `TPU_MULTIHOST_BACKEND` — [`L14`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L14)
- `TPU_MULTIPROCESS_DP` — [`L15`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L15)
- `TPU_NAME` — [`L12`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L12)
- `TPU_OFFLOAD_BATCHED_SAVE` — [`L60`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L60)
- `TPU_OFFLOAD_BLOCK_SIZE_BUCKETS` — [`L63`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L63)
- `TPU_OFFLOAD_DECODE_SAVE` — [`L56`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L56)
- `TPU_OFFLOAD_METRICS_LOG_INTERVAL` — [`L61`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L61)
- `TPU_OFFLOAD_NUM_CPU_CHUNKS` — [`L57`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L57)
- `TPU_OFFLOAD_NUM_STAGING_BLOCKS` — [`L58`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L58)
- `TPU_OFFLOAD_SAVE_THREADS` — [`L59`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L59)
- `TPU_OFFLOAD_SKIP_JAX_PRECOMPILE` — [`L55`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L55)
- `TPU_OFFLOAD_USE_UNPINNED_HOST` — [`L62`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L62)
- `TPU_WORKER_ID` — [`L13`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L13)
- `USE_BATCHED_RPA_KERNEL` — [`L43`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L43)
- `USE_DENSE_MOE` — [`L28`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L28)
- `USE_JAX_PROFILER_SERVER` — [`L41`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L41)
- `USE_MOE_EP_KERNEL` — [`L26`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L26)
- `USE_UNFUSED_MEGABLOCKS` — [`L27`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L27)
- `VLLM_MOE_CHUNK_SIZE` — [`L71`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L71)
- `VLLM_TPU_PATCH_MM_EMBEDDINGS` — [`L66`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L66)
- `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE` — [`L31`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L31)
- `VLLM_XLA_CHECK_RECOMPILATION` — [`L19`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L19)
- `environment_variables` — [`L204`](../../../../../raw/code/tpu-inference/tpu_inference/envs.py#L204)

