---
title: 'Module: src/maxtext/checkpoint_conversion/utils/load_dynamic.py'
type: catalog
provenance: extracted
module: src/maxtext/checkpoint_conversion/utils/load_dynamic.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.checkpoint_conversion.utils.load_dynamic`/
symbols:
  load_safetensors_dynamic_state: load_safetensors_dynamic_state().
  transform_hf_state_to_mt_state: transform_hf_state_to_mt_state().
  get_hf_config_and_mappings: get_hf_config_and_mappings().
  load_sharded_hf_state: load_sharded_hf_state().
  HF_MODEL_CONFIGS: HF_MODEL_CONFIGS.
  get_hf_loading_function: get_hf_loading_function.
  build_gcs_cache_worker: build_gcs_cache_worker().
  load_sharded_hf_state.combine_sharding: load_sharded_hf_state().combine_sharding().
  transform_hf_state_to_mt_state.tensor_getter: transform_hf_state_to_mt_state().tensor_getter().
---
# Module: [`src/maxtext/checkpoint_conversion/utils/load_dynamic.py`](../../../../../../../../raw/code/maxtext/src/maxtext/checkpoint_conversion/utils/load_dynamic.py)

## Functions
- `build_gcs_cache_worker(fpath, gcs_cache_dir, hf_access_token)` — [`L93`](../../../../../../../../raw/code/maxtext/src/maxtext/checkpoint_conversion/utils/load_dynamic.py#L93) — Caches a file from Hugging Face to a GCS bucket cache directory.
- `combine_sharding(sds, single_sharding)` — [`L181`](../../../../../../../../raw/code/maxtext/src/maxtext/checkpoint_conversion/utils/load_dynamic.py#L181)
- `get_hf_config_and_mappings(maxtext_config)` — [`L139`](../../../../../../../../raw/code/maxtext/src/maxtext/checkpoint_conversion/utils/load_dynamic.py#L139) — Gets HF config and parameter mapping based on the MaxText config.
- `load_safetensors_dynamic_state(path, abstract_unboxed_pre_state, maxtext_config)` — [`L254`](../../../../../../../../raw/code/maxtext/src/maxtext/checkpoint_conversion/utils/load_dynamic.py#L254) — Main entry point to dynamically build and load safetensors into MaxText format.
- `load_sharded_hf_state(path)` — [`L156`](../../../../../../../../raw/code/maxtext/src/maxtext/checkpoint_conversion/utils/load_dynamic.py#L156) — Loads HF state with maximal sharding across TPU mesh to avoid host OOM.
- `tensor_getter(key)` — [`L196`](../../../../../../../../raw/code/maxtext/src/maxtext/checkpoint_conversion/utils/load_dynamic.py#L196)
- `transform_hf_state_to_mt_state(hf_state, target_tree, param_map_mt_to_hf, hook_fn_map_mt, maxtext_config)` — [`L192`](../../../../../../../../raw/code/maxtext/src/maxtext/checkpoint_conversion/utils/load_dynamic.py#L192) — Transforms HF state into MaxText state by applying param mappings and mathematical hooks.

## Module values
- `HF_MODEL_CONFIGS` — [`L89`](../../../../../../../../raw/code/maxtext/src/maxtext/checkpoint_conversion/utils/load_dynamic.py#L89)
- `get_hf_loading_function` — [`L90`](../../../../../../../../raw/code/maxtext/src/maxtext/checkpoint_conversion/utils/load_dynamic.py#L90)

