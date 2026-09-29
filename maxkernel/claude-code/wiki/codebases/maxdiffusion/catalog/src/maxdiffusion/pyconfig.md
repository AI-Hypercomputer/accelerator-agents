---
title: 'Module: src/maxdiffusion/pyconfig.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/pyconfig.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.pyconfig`/
symbols:
  _HyperParameters.user_init: _HyperParameters#user_init().
  initialize: initialize().
  _HyperParameters.__init__: _HyperParameters#__init__().
  _config: _config.
  _yaml_types_to_parser: _yaml_types_to_parser.
  _lists_to_tuples: _lists_to_tuples().
  _HyperParameters.wan_init: _HyperParameters#wan_init().
  _HyperParameters.keys: _HyperParameters#keys.
  HyperParameters.__getattr__: HyperParameters#__getattr__().
  HyperParameters.get_keys: HyperParameters#get_keys().
  config: config.
  _HyperParameters: _HyperParameters#
  _validate_model_name: _validate_model_name().
  _validate_training_model_name: _validate_training_model_name().
  r: r.
  _ALLOWED_MODEL_NAMES: _ALLOWED_MODEL_NAMES.
  _ALLOWED_TRAINING_MODEL_NAMES: _ALLOWED_TRAINING_MODEL_NAMES.
  string_to_bool: string_to_bool().
  string_to_list: string_to_list().
  _HyperParameters._load_kwargs: _HyperParameters#_load_kwargs().
  _HyperParameters.calculate_global_batch_sizes: _HyperParameters#calculate_global_batch_sizes().
  get_num_slices: get_num_slices().
  get_quantization_local_shard_count: get_quantization_local_shard_count().
  HyperParameters: HyperParameters#
  print_system_information: print_system_information().
  get_num_target_devices: get_num_target_devices().
  HyperParameters.__init__: HyperParameters#__init__().
  HyperParameters.__setattr__: HyperParameters#__setattr__().
---
# Module: [`src/maxdiffusion/pyconfig.py`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py)

## Classes
### `HyperParameters`
- def: [`src/maxdiffusion/pyconfig.py:343`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L343)
- signature: `class HyperParameters:`
- members:
  - `get_keys(self)` — [`L356`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L356)
- protocol/private: `__getattr__`[`L348`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L348), `__init__`[`L345`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L345), `__setattr__`[`L353`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L353)
- uses (calls/refs, reference-scoped): [`_config`](pyconfig.md#_config), [`keys`](pyconfig.md#_HyperParameters.keys)
- used by: [`initialize`](pyconfig.md#initialize)

### `_HyperParameters`
- def: [`src/maxdiffusion/pyconfig.py:94`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L94)
- signature: `class _HyperParameters:`
- members:
  - `calculate_global_batch_sizes(per_device_batch_size)` — [`L186`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L186)
  - `user_init(raw_keys)` — [`L198`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L198) — Transformations between the config data and configs used at runtime
  - `wan_init(raw_keys)` — [`L155`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L155)
  - `keys` — [`L146`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L146)
- protocol/private: `__init__`[`L96`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L96), `_load_kwargs`[`L150`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L150)
- uses (calls/refs, reference-scoped): [`_yaml_types_to_parser`](pyconfig.md#_yaml_types_to_parser), [`_lists_to_tuples`](pyconfig.md#_lists_to_tuples), [`download_blobs`](max_utils.md#download_blobs), [`maybe_initialize_jax_distributed_system`](max_utils.md#maybe_initialize_jax_distributed_system), [`write_config_raw_keys_for_gcs`](max_utils.md#write_config_raw_keys_for_gcs), [`CAUSVID_TRANSFORMER_MODEL_NAME_OR_PATH`](models/wan/wan_utils.md#CAUSVID_TRANSFORMER_MODEL_NAME_OR_PATH), [`WAN_21_FUSION_X_MODEL_NAME_OR_PATH`](models/wan/wan_utils.md#WAN_21_FUSION_X_MODEL_NAME_OR_PATH), [`get_global_batch_size`](max_utils.md#get_global_batch_size), [`get_num_slices`](pyconfig.md#get_num_slices), [`get_quantization_local_shard_count`](pyconfig.md#get_quantization_local_shard_count)
- used by: [`initialize`](pyconfig.md#initialize), [`__getattr__`](pyconfig.md#HyperParameters.__getattr__), [`get_keys`](pyconfig.md#HyperParameters.get_keys)

## Functions
- `_lists_to_tuples(l: list[Any])` — [`L90`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L90)
- `_validate_model_name(model_name: str | None)` — [`L48`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L48) — Raise if model_name is not in the allowed list.
- `_validate_training_model_name(model_name: str | None)` — [`L56`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L56) — Raise if model_name is not in the allowed training list.
- `get_num_slices(raw_keys)` — [`L321`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L321)
- `get_num_target_devices(raw_keys)` — [`L339`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L339)
- `get_quantization_local_shard_count(raw_keys)` — [`L332`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L332)
- `initialize(argv, **kwargs)` — [`L360`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L360)
- `print_system_information()` — [`L84`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L84)
- `string_to_bool(s: str)` — [`L66`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L66)
- `string_to_list(string_list: str)` — [`L74`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L74)

## Module values
- `_ALLOWED_MODEL_NAMES` — [`L44`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L44)
- `_ALLOWED_TRAINING_MODEL_NAMES` — [`L45`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L45)
- `_config` — [`L80`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L80)
- `_yaml_types_to_parser` — [`L78`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L78)
- `config` — [`L81`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L81)
- `r` — [`L372`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pyconfig.py#L372)

