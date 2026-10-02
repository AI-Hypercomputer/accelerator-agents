---
title: 'Module: python/sgl_jax/srt/eplb/expert_location.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/eplb/expert_location.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.eplb.expert_location`/
symbols:
  compute_initial_expert_location_metadata: compute_initial_expert_location_metadata().
  ExpertLocationMetadata: ExpertLocationMetadata#
  ExpertLocationMetadata.init_by_mapping: ExpertLocationMetadata#init_by_mapping().
  ExpertLocationMetadata.init_by_eplb: ExpertLocationMetadata#init_by_eplb().
  ExpertLocationMetadata._init_common: ExpertLocationMetadata#_init_common().
  ExpertLocationMetadata.tree_flatten: ExpertLocationMetadata#tree_flatten().
  ExpertLocationMetadata.tree_unflatten: ExpertLocationMetadata#tree_unflatten().
  ExpertLocationMetadata.init_trivial: ExpertLocationMetadata#init_trivial().
  get_global_expert_location_metadata: get_global_expert_location_metadata().
  topk_ids_logical_to_physical: topk_ids_logical_to_physical().
  ExpertLocationMetadata._init_raw: ExpertLocationMetadata#_init_raw().
  _topk_ids_logical_to_physical_dynamic: _topk_ids_logical_to_physical_dynamic().
  logger: logger.
  init_expert_location_metadata: init_expert_location_metadata().
  _compute_logical_to_all_physical_map: _compute_logical_to_all_physical_map().
  _topk_ids_logical_to_physical_static: _topk_ids_logical_to_physical_static().
  ExpertLocationMetadata.logical_to_rank_dispatch_physical_map: ExpertLocationMetadata#logical_to_rank_dispatch_physical_map.
  ExpertLocationMetadata.logical_to_all_physical_map: ExpertLocationMetadata#logical_to_all_physical_map.
  ExpertLocationMetadata.logical_to_all_physical_map_num_valid: ExpertLocationMetadata#logical_to_all_physical_map_num_valid.
  ExpertLocationMetadata.ep_dispatch_algorithm: ExpertLocationMetadata#ep_dispatch_algorithm.
  set_global_server_args: set_global_server_args().
  ExpertLocationMetadata.physical_to_logical_map: ExpertLocationMetadata#physical_to_logical_map.
  set_global_expert_location_metadata: set_global_expert_location_metadata().
  _GLOBAL_SERVER_ARGS: _GLOBAL_SERVER_ARGS.
  _GLOBAL_EXPERT_LOCATION_METADATA: _GLOBAL_EXPERT_LOCATION_METADATA.
  get_global_server_args: get_global_server_args().
  ExpertLocationMetadata.num_physical_experts: ExpertLocationMetadata#num_physical_experts.
  _pad_nested_array: _pad_nested_array().
  ExpertLocationMetadata.__init__: ExpertLocationMetadata#__init__().
---
# Module: [`python/sgl_jax/srt/eplb/expert_location.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py)

## Classes
### `ExpertLocationMetadata`
- def: [`python/sgl_jax/srt/eplb/expert_location.py:42`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L42)
- doc: Stores global expert mapping metadata.
- signature: `class ExpertLocationMetadata:`
- members:
  - `init_by_eplb(server_args: ServerArgs, model_config: ModelConfig, logical_count)` — [`L144`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L144)
  - `init_by_mapping(server_args: ServerArgs, model_config: ModelConfig, physical_to_logical_map)` — [`L117`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L117)
  - `init_trivial(server_args: ServerArgs, model_config: ModelConfig)` — [`L95`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L95) — Trivial location - logical expert i corresponds to physical expert i
  - `tree_flatten(self)` — [`L69`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L69)
  - `tree_unflatten(cls, aux_data, children)` — [`L84`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L84)
  - `ep_dispatch_algorithm` — [`L56`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L56)
  - `logical_to_all_physical_map` — [`L60`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L60)
  - `logical_to_all_physical_map_num_valid` — [`L63`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L63)
  - `logical_to_rank_dispatch_physical_map` — [`L57`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L57)
  - `num_physical_experts` — [`L67`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L67)
  - `physical_to_logical_map` — [`L66`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L66)
- protocol/private: `__init__`[`L47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L47), `_init_common`[`L181`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L181), `_init_raw`[`L201`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L201)
- uses (calls/refs, reference-scoped): [`ServerArgs`](../server_args.md#ServerArgs), [`hf_config`](../configs/model_config.md#ModelConfig.hf_config), [`ModelConfig`](../configs/model_config.md#ModelConfig), [`device_array`](../utils/jax_utils.md#device_array), [`ep_size`](../server_args.md#ServerArgs.ep_size), [`_compute_logical_to_all_physical_map`](expert_location.md#_compute_logical_to_all_physical_map), [`nnodes`](../server_args.md#ServerArgs.nnodes), [`ep_num_redundant_experts`](../server_args.md#ServerArgs.ep_num_redundant_experts), [`ep_dispatch_algorithm`](../server_args.md#ServerArgs.ep_dispatch_algorithm)
- used by: [`__call__`](../layers/gate.md#TopK.__call__), [`compute_initial_expert_location_metadata`](expert_location.md#compute_initial_expert_location_metadata), [`topk_ids_logical_to_physical`](expert_location.md#topk_ids_logical_to_physical), [`_topk_ids_logical_to_physical_dynamic`](expert_location.md#_topk_ids_logical_to_physical_dynamic), [`_topk_ids_logical_to_physical_static`](expert_location.md#_topk_ids_logical_to_physical_static), [`expert_location_metadata`](../model_executor/forward_batch_info.md#ForwardBatch.expert_location_metadata)

## Functions
- `_compute_logical_to_all_physical_map(server_args: ServerArgs, physical_to_logical_map: np.ndarray, num_logical_experts: int)` — [`L311`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L311)
- `_pad_nested_array(arr, pad_value)` — [`L332`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L332)
- `_topk_ids_logical_to_physical_dynamic(topk_ids: jax.Array, info: ExpertLocationMetadata, layer_id: int = 0)` — [`L365`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L365)
- `_topk_ids_logical_to_physical_static(topk_ids: jax.Array, info: ExpertLocationMetadata, layer_id: int = 0)` — [`L357`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L357)
- `compute_initial_expert_location_metadata(server_args: ServerArgs, model_config: ModelConfig)` — [`L244`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L244)
- `get_global_expert_location_metadata()` — [`L32`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L32)
- `get_global_server_args()` — [`L23`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L23)
- `init_expert_location_metadata(server_args, model_config)` — [`L302`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L302) — Initializes the global expert mapping.
- `set_global_expert_location_metadata(metadata)` — [`L36`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L36)
- `set_global_server_args(args)` — [`L27`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L27)
- `topk_ids_logical_to_physical(topk_ids: jax.Array, info: ExpertLocationMetadata | None, layer_id: int = 0)` — [`L338`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L338) — Maps logical expert IDs to physical expert IDs.

## Module values
- `_GLOBAL_EXPERT_LOCATION_METADATA` — [`L20`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L20)
- `_GLOBAL_SERVER_ARGS` — [`L19`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L19)
- `logger` — [`L16`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/eplb/expert_location.py#L16)

