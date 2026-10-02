---
title: 'Module: tokamax/_src/autotuning/api.py'
type: catalog
provenance: extracted
module: tokamax/_src/autotuning/api.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.autotuning.api`/
symbols:
  _API_IMPLEMENTATIONS._API_IMPLEMENTATIONS: _API_IMPLEMENTATIONS._API_IMPLEMENTATIONS.
  autotune: autotune().
  get_bound_args: get_bound_args().
  _serialize_bound_args_autotuning_data: _serialize_bound_args_autotuning_data().
  _validate_bound_args_autotuning_data: _validate_bound_args_autotuning_data().
  AutotuningResult.data: AutotuningResult#data.
  AutotuningResult.__enter__: AutotuningResult#__enter__().
  dump_bound_args_to_json._strip_vjp_and_config: dump_bound_args_to_json()._strip_vjp_and_config().
  AutotuningResult.__or__: AutotuningResult#__or__().
  dump_bound_args_to_json: dump_bound_args_to_json().
  AutotuningResult.dump_cache_str: AutotuningResult#dump_cache_str().
  bound_args_to_json: bound_args_to_json().
  AutotuningResult: AutotuningResult#
  AutotuningResult.dumps: AutotuningResult#dumps().
  bound_args_from_json: bound_args_from_json().
  get_op_implementations: get_op_implementations().
  BoundArgsAutotuningData.BoundArgsAutotuningData: BoundArgsAutotuningData.BoundArgsAutotuningData.
  bound_args_from_json_file: bound_args_from_json_file().
  _AUTOTUNING_RESULT_ADAPTER: _AUTOTUNING_RESULT_ADAPTER.
  _BOUND_ARGS_ADAPTER: _BOUND_ARGS_ADAPTER.
  AutotuningResult.device_kind: AutotuningResult#device_kind.
  _P: _P.
  AutotuningResult.loads: AutotuningResult#loads().
  AutotuningResult.tokamax_version: AutotuningResult#tokamax_version.
  AutotuningResult.dump: AutotuningResult#dump().
  AutotuningResult.load: AutotuningResult#load().
  AutotuningResult.__exit__: AutotuningResult#__exit__().
---
# Module: [`tokamax/_src/autotuning/api.py`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py)

## Classes
### `AutotuningResult`
- def: [`tokamax/_src/autotuning/api.py:77`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L77)
- doc: Autotuning results.
- signature: `class AutotuningResult:`
- members:
  - `__or__(self, other: AutotuningResult)` — [`L159`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L159) — Returns a new AutotuningResult that is the merge of `self` and `other`.
  - `dump(self, fp, *, prune_errors: bool = False)` — [`L102`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L102)
  - `dump_cache_str(self)` — [`L116`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L116)
  - `dumps(self, *, prune_errors: bool = False)` — [`L105`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L105)
  - `load(cls, fp)` — [`L135`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L135)
  - `loads(cls, json_data: str)` — [`L139`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L139)
  - `data` — [`L89`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L89)
  - `device_kind` — [`L88`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L88)
  - `tokamax_version` — [`L100`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L100)
- protocol/private: `__enter__`[`L142`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L142), `__exit__`[`L154`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L154)
- uses (calls/refs, reference-scoped): [`op`](../ops/op.md#BoundArguments.op), [`arguments`](../ops/op.md#BoundArguments.arguments), [`_serialize_bound_args_autotuning_data`](api.md#_serialize_bound_args_autotuning_data), [`_validate_bound_args_autotuning_data`](api.md#_validate_bound_args_autotuning_data), [`autotuning_cache_key`](../ops/op.md#BoundArguments.autotuning_cache_key), [`BoundArgsAutotuningData`](api.md#BoundArgsAutotuningData.BoundArgsAutotuningData), [`get_autotuning_cache_overlay_state`](../ops/op.md#get_autotuning_cache_overlay_state), [`_AUTOTUNING_RESULT_ADAPTER`](api.md#_AUTOTUNING_RESULT_ADAPTER), [`TOKAMAX_VERSION`](../version.md#TOKAMAX_VERSION.TOKAMAX_VERSION)
- used by: [`autotune`](api.md#autotune), [`_AUTOTUNING_RESULT_ADAPTER`](api.md#_AUTOTUNING_RESULT_ADAPTER)

## Functions
- `_serialize_bound_args_autotuning_data(value: BoundArgsAutotuningData, info)` — [`L51`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L51)
- `_strip_vjp_and_config(bound_arg: op_lib.BoundArguments)` — [`L236`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L236) — Strips the VJP and config from the BoundArguments.
- `_validate_bound_args_autotuning_data(value: Any)` — [`L65`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L65)
- `autotune(f: (Callable[..., Any] | Sequence[op_lib.BoundArguments] | jax.stages.Lowered), *args, ignore_cache: bool = False, all_implementations: bool = False, progress_bar: bool = True, event_filter_regex: str | None = None)` — [`L308`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L308) — Autotunes all captured ops in x.
- `bound_args_from_json(json_string: str)` — [`L263`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L263) — Loads a sequence of BoundArguments from a JSON file.
- `bound_args_from_json_file(filename: str)` — [`L269`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L269) — Loads a sequence of BoundArguments from a JSON file.
- `bound_args_to_json(f: (Callable[_P, Any] | jax.stages.Lowered), filename: str)` — [`L249`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L249) — Dumps a sequence of BoundArguments to a JSON file.
- `dump_bound_args_to_json(bound_args: Sequence[op_lib.BoundArguments])` — [`L233`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L233) — Dumps a sequence of BoundArguments to a JSON string.
- `get_bound_args(f: (Callable[_P, Any] | hlo_utils.HloComputation), *args: _P.args, **kwargs: _P.kwargs)` — [`L193`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L193) — Returns a tuple of unique BoundArguments for all Tokamax ops in `f`.
- `get_op_implementations(op: op_lib.Op, device: jax.Device | None = None)` — [`L286`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L286) — Returns all implementations of the given op.

## Module values
- `BoundArgsAutotuningData` — [`L46`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L46)
- `_API_IMPLEMENTATIONS` — [`L275`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L275)
- `_AUTOTUNING_RESULT_ADAPTER` — [`L188`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L188)
- `_BOUND_ARGS_ADAPTER` — [`L189`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L189)
- `_P` — [`L190`](../../../../../../../raw/code/tokamax/tokamax/_src/autotuning/api.py#L190)

