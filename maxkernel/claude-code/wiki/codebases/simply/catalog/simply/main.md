---
title: 'Module: simply/main.py'
type: catalog
provenance: extracted
module: simply/main.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.main`/
symbols:
  override_mesh_and_sharding: override_mesh_and_sharding().
  load_experiment_config: load_experiment_config().
  main: main().
  execute_code_patch: execute_code_patch().
  _set_xla_dump_to_flag: _set_xla_dump_to_flag().
  override_mesh_and_sharding.parse_mesh_shape_flags: override_mesh_and_sharding().parse_mesh_shape_flags().
  _EXPERIMENT_CONFIG: _EXPERIMENT_CONFIG.
  _SHARDING_CONFIG: _SHARDING_CONFIG.
  _EXPERIMENT_CONFIG_PATH: _EXPERIMENT_CONFIG_PATH.
  _SHARDING_CONFIG_PATH: _SHARDING_CONFIG_PATH.
  _EXPERIMENT_DIR: _EXPERIMENT_DIR.
  _MESH_SHAPE: _MESH_SHAPE.
  _DCN_MESH_SHAPE: _DCN_MESH_SHAPE.
  _DECODING_MESH_SHAPE: _DECODING_MESH_SHAPE.
---
# Module: [`simply/main.py`](../../../../../raw/code/simply/simply/main.py)

## Functions
- `_set_xla_dump_to_flag()` — [`L181`](../../../../../raw/code/simply/simply/main.py#L181) — Sets --xla_dump_to flag in environment.
- `execute_code_patch(config)` — [`L124`](../../../../../raw/code/simply/simply/main.py#L124)
- `load_experiment_config()` — [`L135`](../../../../../raw/code/simply/simply/main.py#L135) — Loads the experiment configuration.
- `main(argv: Sequence[str])` — [`L165`](../../../../../raw/code/simply/simply/main.py#L165)
- `override_mesh_and_sharding(config)` — [`L79`](../../../../../raw/code/simply/simply/main.py#L79) — Updates sharding and mesh fields in the config.
- `parse_mesh_shape_flags(mesh_shape_flags)` — [`L92`](../../../../../raw/code/simply/simply/main.py#L92) — Parses mesh shape flags into a dict.

## Module values
- `_DCN_MESH_SHAPE` — [`L66`](../../../../../raw/code/simply/simply/main.py#L66)
- `_DECODING_MESH_SHAPE` — [`L72`](../../../../../raw/code/simply/simply/main.py#L72)
- `_EXPERIMENT_CONFIG` — [`L34`](../../../../../raw/code/simply/simply/main.py#L34)
- `_EXPERIMENT_CONFIG_PATH` — [`L42`](../../../../../raw/code/simply/simply/main.py#L42)
- `_EXPERIMENT_DIR` — [`L56`](../../../../../raw/code/simply/simply/main.py#L56)
- `_MESH_SHAPE` — [`L60`](../../../../../raw/code/simply/simply/main.py#L60)
- `_SHARDING_CONFIG` — [`L38`](../../../../../raw/code/simply/simply/main.py#L38)
- `_SHARDING_CONFIG_PATH` — [`L49`](../../../../../raw/code/simply/simply/main.py#L49)

