---
title: 'Module: src/maxtext/configs/pyconfig.py'
type: catalog
provenance: extracted
module: src/maxtext/configs/pyconfig.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.configs.pyconfig`/
symbols:
  _initialize_pydantic: _initialize_pydantic().
  _resolve_or_infer_config: _resolve_or_infer_config().
  logger: logger.
  _prepare_for_pydantic: _prepare_for_pydantic().
  initialize: initialize().
  _load_config: _load_config().
  _lists_to_tuples: _lists_to_tuples().
  HyperParameters: HyperParameters#
  initialize_pydantic: initialize_pydantic().
  _CallablePyconfigModule.__call__: _CallablePyconfigModule#__call__().
  _apply_rules: _apply_rules().
  _tuples_to_lists: _tuples_to_lists().
  yaml_key_to_env_key: yaml_key_to_env_key().
  HyperParameters.__init__: HyperParameters#__init__().
  HyperParameters.__deepcopy__: HyperParameters#__deepcopy__().
  _BASE_CONFIG_ATTR: _BASE_CONFIG_ATTR.
  resolve_config_path: resolve_config_path().
  _handle_config_exception: _handle_config_exception().
  _CONFIG_FILE_MAPPING._CONFIG_FILE_MAPPING: _CONFIG_FILE_MAPPING._CONFIG_FILE_MAPPING.
  _MAX_PREFIX: _MAX_PREFIX.
  _yaml_types_to_parser: _yaml_types_to_parser.
  KEYS_NO_LOGGING: KEYS_NO_LOGGING.
  _module_from_path: _module_from_path().
  _resolve_or_infer_addl_config: _resolve_or_infer_addl_config().
  validate_no_keys_overridden_twice: validate_no_keys_overridden_twice().
  _merge_logical_axis_rules: _merge_logical_axis_rules().
  _coerce_to_list: _coerce_to_list().
  HyperParameters.get_keys: HyperParameters#get_keys().
  _CallablePyconfigModule: _CallablePyconfigModule#
  HyperParameters.tree_flatten: HyperParameters#tree_flatten().
  HyperParameters.__getattr__: HyperParameters#__getattr__().
  HyperParameters.__setattr__: HyperParameters#__setattr__().
  validate_and_update_keys: validate_and_update_keys.
  __all__: __all__.
---
# Module: [`src/maxtext/configs/pyconfig.py`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py)

## Classes
### `HyperParameters`
- def: [`src/maxtext/configs/pyconfig.py:305`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L305)
- doc: Wrapper class to expose the configuration in a read-only manner,
- signature: `class HyperParameters:`
- members:
  - `__getattr__(self, attr: str)` — [`L337`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L337) — Provides attribute-style access to the final configuration dictionary.
  - `__setattr__(self, attr: str, value: Any)` — [`L349`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L349) — Makes the configuration object read-only.
  - `get_keys(self)` — [`L353`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L353) — Returns the configuration as a flat dictionary for backward compatibility.
  - `tree_flatten(self)` — [`L334`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L334)
- protocol/private: `__deepcopy__`[`L330`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L330), `__init__`[`L311`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L311)
- uses (calls/refs, reference-scoped): [`_lists_to_tuples`](pyconfig.md#_lists_to_tuples)
- used by: [`_initialize_pydantic`](pyconfig.md#_initialize_pydantic), [`initialize`](pyconfig.md#initialize), [`__call__`](pyconfig.md#_CallablePyconfigModule.__call__)

### `_CallablePyconfigModule`  ·  implements/extends ModuleType
- def: [`src/maxtext/configs/pyconfig.py:569`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L569)
- doc: Allows calling the module directly as mt.pyconfig().
- signature: `class _CallablePyconfigModule(sys.modules[__name__].__class__):`
- protocol/private: `__call__`[`L572`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L572)
- uses (calls/refs, reference-scoped): [`initialize`](pyconfig.md#initialize), [`HyperParameters`](pyconfig.md#HyperParameters)

## Functions
- `_apply_rules(base_rules, new_rules, config)` — [`L184`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L184)
- `_coerce_to_list(value: Any)` — [`L218`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L218) — Coerce string/tuple inputs for list[str] configuration fields into Python lists.
- `_handle_config_exception(e: Exception)` — [`L358`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L358) — Handles configuration exceptions, prints to stderr, writes log, and exits or raises.
- `_initialize_pydantic(argv: list[str] | None = None, **kwargs)` — [`L409`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L409) — Initializes the configuration by loading YAML files, and applying CLI, env, and kwarg overrides.
- `_lists_to_tuples(l: list | Any)` — [`L213`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L213) — Recursively converts nested lists to tuples for JAX compatibility.
- `_load_config(config_name: str)` — [`L190`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L190) — Loads a YAML file and its base_configs recursively using OmegaConf.
- `_merge_logical_axis_rules(base_rules, new_rules)` — [`L168`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L168) — Merges two lists of logical_axis_rules. Rules in new_rules override all rules
- `_module_from_path(path: str)` — [`L72`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L72) — Convert a file path to module path for config inference.
- `_prepare_for_pydantic(raw_keys: dict[str, Any])` — [`L241`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L241) — Prepares the raw dictionary for Pydantic model instantiation.
- `_resolve_or_infer_addl_config(**kwargs)` — [`L108`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L108) — Resolves or infers more configs from module.
- `_resolve_or_infer_config(argv: list[str] | None = None, **kwargs)` — [`L82`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L82) — Resolves or infers config file path from module.
- `_tuples_to_lists(l: list | tuple | Any)` — [`L208`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L208) — Recursively converts nested tuples to lists for Pydantic compatibility.
- `initialize(argv: list[str] | None = None, **kwargs)` — [`L382`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L382) — Initializes the configuration by loading YAML files, and applying CLI, env, and kwarg overrides.
- `initialize_pydantic(argv: list[str] | None = None, **kwargs)` — [`L395`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L395) — Initializes the configuration by loading YAML files, and applying CLI, env, and overrides.
- `resolve_config_path(param: str)` — [`L155`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L155) — Resolve config path to auto rewrite to use new src folder.
- `validate_no_keys_overridden_twice(model_loaded_cfg: omegaconf.DictConfig, overrides_cfg: omegaconf.DictConfig)` — [`L130`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L130) — Validates that no keys are overridden by both model config and overrides with different values.
- `yaml_key_to_env_key(s: str)` — [`L126`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L126)

## Module values
- `KEYS_NO_LOGGING` — [`L49`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L49)
- `_BASE_CONFIG_ATTR` — [`L44`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L44)
- `_CONFIG_FILE_MAPPING` — [`L52`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L52)
- `_MAX_PREFIX` — [`L45`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L45)
- `__all__` — [`L566`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L566)
- `_yaml_types_to_parser` — [`L46`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L46)
- `logger` — [`L41`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L41)
- `validate_and_update_keys` — [`L565`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig.py#L565)

