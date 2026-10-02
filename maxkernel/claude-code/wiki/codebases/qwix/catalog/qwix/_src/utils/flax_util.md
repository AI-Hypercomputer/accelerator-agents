---
title: 'Module: qwix/_src/utils/flax_util.py'
type: catalog
provenance: extracted
module: qwix/_src/utils/flax_util.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.utils.flax_util`/
symbols:
  find_param: find_param().
  unbox: unbox().
  get_or_create_param: get_or_create_param().
  get_current_module: get_current_module().
  QuantStat: QuantStat#
  update_boxed: update_boxed().
  get_or_create_variable: get_or_create_variable().
  update_sharding: update_sharding().
  get_value_from_path: get_value_from_path().
  make_rng: make_rng().
  get_current_module_path: get_current_module_path().
  get_and_delete_variable: get_and_delete_variable().
  should_update_quant_stats: should_update_quant_stats().
  _check_shape: _check_shape().
  unbox.fn: unbox().fn().
---
# Module: [`qwix/_src/utils/flax_util.py`](../../../../../../../raw/code/qwix/qwix/_src/utils/flax_util.py)

## Classes
### `QuantStat`  ·  implements/extends Generic, Representable
- def: [`qwix/_src/utils/flax_util.py:29`](../../../../../../../raw/code/qwix/qwix/_src/utils/flax_util.py#L29)
- doc: NNX variable that stores quantization statistics.
- signature: `class QuantStat(nnx.Variable):`
- used by: [`test_srq`](../../../integration_tests/vae_qat_test.md#VaeQatTest.test_srq), [`test_nnx`](../../../integration_tests/odml_coverage_test.md#OdmlTest.test_nnx)  (9 test-only)

## Functions
- `_check_shape(value: Any, init_fn: Callable[[], Any])` — [`L429`](../../../../../../../raw/code/qwix/qwix/_src/utils/flax_util.py#L429) — Checks the shape of the value to match the shape of init_fn(). — documented in [qwix-_src-utils-flax_util](../../../../concepts/qwix-_src-utils-flax_util.md)
- `find_param(x: Any, ptq_array_type=None)` — [`L197`](../../../../../../../raw/code/qwix/qwix/_src/utils/flax_util.py#L197) — Finds the param name of a given array in the current module. — documented in [qwix-_src-providers-lora](../../../../concepts/qwix-_src-providers-lora.md)
- `fn(x)` — [`L297`](../../../../../../../raw/code/qwix/qwix/_src/utils/flax_util.py#L297)
- `get_and_delete_variable(collection: str, name: str)` — [`L123`](../../../../../../../raw/code/qwix/qwix/_src/utils/flax_util.py#L123) — Gets and deletes a quant_stat in the current module.
- `get_current_module()` — [`L45`](../../../../../../../raw/code/qwix/qwix/_src/utils/flax_util.py#L45) — Returns the current module, or raises an error if current module is not known. — documented in [qwix-_src-utils-flax_util](../../../../concepts/qwix-_src-utils-flax_util.md)
- `get_current_module_path()` — [`L71`](../../../../../../../raw/code/qwix/qwix/_src/utils/flax_util.py#L71) — Returns the path of the current module, or raises an error if current module is not known. — documented in [qwix-_src-utils-flax_util](../../../../concepts/qwix-_src-utils-flax_util.md)
- `get_or_create_param(name: str, init_fn: Callable[..., Any], *, nnx_param_type: type[nnx.Param] = nnx.Param, need_rng: bool = False)` — [`L148`](../../../../../../../raw/code/qwix/qwix/_src/utils/flax_util.py#L148) — Gets or creates a param in the current module. — documented in [qwix-_src-utils-flax_util](../../../../concepts/qwix-_src-utils-flax_util.md)
- `get_or_create_variable(collection: str, name: str, init_fn: Callable[[], Any])` — [`L89`](../../../../../../../raw/code/qwix/qwix/_src/utils/flax_util.py#L89) — Gets or creates a quant_stat in the current module. — documented in [qwix-_src-utils-flax_util](../../../../concepts/qwix-_src-utils-flax_util.md)
- `get_value_from_path(obj: Any, path: tuple[str | int, ...])` — [`L466`](../../../../../../../raw/code/qwix/qwix/_src/utils/flax_util.py#L466) — Helper that returns the value from the path in the object.
- `make_rng(rng_stream: str)` — [`L450`](../../../../../../../raw/code/qwix/qwix/_src/utils/flax_util.py#L450) — Generate a random key from the given rng_stream in the current module.
- `should_update_quant_stats()` — [`L33`](../../../../../../../raw/code/qwix/qwix/_src/utils/flax_util.py#L33) — Returns whether the current module should update the quantization statistics. — documented in [qwix-_src-utils-flax_util](../../../../concepts/qwix-_src-utils-flax_util.md)
- `unbox(maybe_boxed: Any)` — [`L284`](../../../../../../../raw/code/qwix/qwix/_src/utils/flax_util.py#L284) — Returns the raw array without applying hooks or sharding constraints. — documented in [qwix-_src-utils-flax_util](../../../../concepts/qwix-_src-utils-flax_util.md)
- `update_boxed(boxed: nn.meta.AxisMetadata | nnx.Variable | jax.Array, *, value: jax.Array | None = None, split: Collection[int] | None = None, merge: Collection[int] | None = None, transpose: Sequence[int | None] | None = None)` — [`L365`](../../../../../../../raw/code/qwix/qwix/_src/utils/flax_util.py#L365) — Update a boxed value and its metadata, i.e., nn.meta.AxisMetadata or nnx.Variable.
- `update_sharding(spec: Sequence[Any] | jax.sharding.PartitionSpec, *, shape: Sequence[int] | None = None, split: Collection[int] | None = None, merge: Collection[int] | None = None, transpose: Sequence[int | None] | None = None)` — [`L320`](../../../../../../../raw/code/qwix/qwix/_src/utils/flax_util.py#L320) — Derives the partition spec from an existing spec.

