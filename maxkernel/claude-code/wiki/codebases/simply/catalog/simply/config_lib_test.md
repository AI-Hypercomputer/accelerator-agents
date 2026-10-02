---
title: 'Module: simply/config_lib_test.py'
type: catalog
provenance: extracted
module: simply/config_lib_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.config_lib_test`/
symbols:
  ConfigLibTest.test_dump_load_baseline_config: ConfigLibTest#test_dump_load_baseline_config().
  ConfigRegistryInstantiationTest.test_instantiate_config: ConfigRegistryInstantiationTest#test_instantiate_config().
  _get_all_config_names: _get_all_config_names().
  ConfigLibTest: ConfigLibTest#
  ConfigRegistryInstantiationTest: ConfigRegistryInstantiationTest#
---
# Module: [`simply/config_lib_test.py`](../../../../../raw/code/simply/simply/config_lib_test.py)

## Classes
### `ConfigLibTest`  ·  implements/extends TestCase
- def: [`simply/config_lib_test.py:24`](../../../../../raw/code/simply/simply/config_lib_test.py#L24)
- signature: `class ConfigLibTest(absltest.TestCase):`
- members:
  - `test_dump_load_baseline_config(self)` — [`L26`](../../../../../raw/code/simply/simply/config_lib_test.py#L26)
- uses (calls/refs, reference-scoped): [`ExperimentConfigRegistry`](config_lib.md#ExperimentConfigRegistry), [`get`](utils/registry.md#RootRegistry.get), [`TransformerLM`](model_lib.md#TransformerLM), [`ModuleRegistry`](utils/module.md#ModuleRegistry), [`dump_dataclasses`](utils/pytree.md#dump_dataclasses), [`ShardingConfigRegistry`](config_lib.md#ShardingConfigRegistry), [`load_dataclasses`](utils/pytree.md#load_dataclasses), [`get_config`](config_lib.md#ExperimentConfigRegistry.get_config), [`get_config`](config_lib.md#ShardingConfigRegistry.get_config)

### `ConfigRegistryInstantiationTest`  ·  implements/extends TestCase
- def: [`simply/config_lib_test.py:49`](../../../../../raw/code/simply/simply/config_lib_test.py#L49)
- signature: `class ConfigRegistryInstantiationTest(parameterized.TestCase):`
- members:
  - `test_instantiate_config(self, config_name: str)` — [`L52`](../../../../../raw/code/simply/simply/config_lib_test.py#L52)
- uses (calls/refs, reference-scoped): [`ExperimentConfigRegistry`](config_lib.md#ExperimentConfigRegistry), [`get`](utils/registry.md#RootRegistry.get), [`_get_all_config_names`](config_lib_test.md#_get_all_config_names)

## Functions
- `_get_all_config_names()` — [`L42`](../../../../../raw/code/simply/simply/config_lib_test.py#L42)

