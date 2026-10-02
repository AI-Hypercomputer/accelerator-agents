---
title: 'Module: tokamax/_src/ops/op_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/op_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.op_test`/
symbols:
  BoundArgumentsTest.test_roundtrip: BoundArgumentsTest#test_roundtrip().
  _FakeOp: _FakeOp#
  BoundArgumentsTest.test_get_config: BoundArgumentsTest#test_get_config().
  BoundArgumentsTest.test_default_config: BoundArgumentsTest#test_default_config().
  BoundArgumentsTest.test_autotuning_configs: BoundArgumentsTest#test_autotuning_configs().
  BoundArgumentsTest.test_autotune: BoundArgumentsTest#test_autotune().
  BoundArgumentsTest.test_autotune_vmap: BoundArgumentsTest#test_autotune_vmap().
  BoundArgumentsTest.test_heuristics_config: BoundArgumentsTest#test_heuristics_config().
  _bsd: _bsd().
  _HEURISTICS_CONFIG: _HEURISTICS_CONFIG.
  _FakeOpConfig: _FakeOpConfig#
  OpTest.test_bind: OpTest#test_bind().
  BoundArgumentsTest.test_equals: BoundArgumentsTest#test_equals().
  BoundArgumentsTest.test_hash: BoundArgumentsTest#test_hash().
  BoundArgumentsTest.test_ignore_cache_overlay: BoundArgumentsTest#test_ignore_cache_overlay().
  _ATTN_ARG_SPECS: _ATTN_ARG_SPECS.
  _RAGGED_DOT_ARG_SPECS: _RAGGED_DOT_ARG_SPECS.
  _FakeOp._get_heuristics_config: _FakeOp#_get_heuristics_config().
  _FakeOp._get_autotuning_configs: _FakeOp#_get_autotuning_configs().
  _AUTOTUNE_CONFIG: _AUTOTUNE_CONFIG.
  _GLU_ARG_SPECS: _GLU_ARG_SPECS.
  _NORM_ARG_SPECS: _NORM_ARG_SPECS.
  _eval_shape: _eval_shape().
  _FakeOp.config_cls: _FakeOp#config_cls.
  _eval_shape.f: _eval_shape().f().
  _FakeOpConfig.blah: _FakeOpConfig#blah.
  _FakeOp._fwd: _FakeOp#_fwd().
  OpTest: OpTest#
  BoundArgumentsTest: BoundArgumentsTest#
---
# Module: [`tokamax/_src/ops/op_test.py`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py)

## Classes
### `BoundArgumentsTest`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/op_test.py:106`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L106)
- signature: `class BoundArgumentsTest(parameterized.TestCase):`
- members:
  - `test_autotune(self)` — [`L154`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L154)
  - `test_autotune_vmap(self, x_vmap_axes, y_vmap_axes)` — [`L170`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L170)
  - `test_autotuning_configs(self)` — [`L149`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L149)
  - `test_default_config(self)` — [`L132`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L132)
  - `test_equals(self)` — [`L177`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L177)
  - `test_get_config(self)` — [`L108`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L108)
  - `test_hash(self)` — [`L184`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L184)
  - `test_heuristics_config(self)` — [`L145`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L145)
  - `test_ignore_cache_overlay(self)` — [`L227`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L227)
  - `test_roundtrip(self, op, arg_specs)` — [`L213`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L213)
- uses (calls/refs, reference-scoped): [`BoundArguments`](op.md#BoundArguments), [`DotProductAttention`](attention/base.md#DotProductAttention), [`bind`](op.md#Op.bind), [`supported_on`](op.md#Op.supported_on), [`op`](op.md#BoundArguments.op), [`RaggedDot`](ragged_dot/base.md#RaggedDot), [`autotune`](op.md#BoundArguments.autotune), [`GatedLinearUnit`](gated_linear_unit/base.md#GatedLinearUnit), [`get_config`](op.md#BoundArguments.get_config), [`cached_autotuning_data`](op.md#BoundArguments.cached_autotuning_data), [`Normalization`](normalization/base.md#Normalization), [`_FakeOp`](op_test.md#_FakeOp), [`get_autotuning_cache`](op.md#Op.get_autotuning_cache), [`autotuning_configs`](op.md#BoundArguments.autotuning_configs), [`replace`](op.md#Op.replace), [`PallasTritonFlashAttention`](attention/pallas_triton.md#PallasTritonFlashAttention), [`heuristics_config`](op.md#BoundArguments.heuristics_config), [`PallasTritonRaggedDot`](ragged_dot/pallas_triton.md#PallasTritonRaggedDot), [`default_config`](op.md#BoundArguments.default_config), [`_bsd`](op_test.md#_bsd), [`_HEURISTICS_CONFIG`](op_test.md#_HEURISTICS_CONFIG), [`_FakeOpConfig`](op_test.md#_FakeOpConfig), [`BOUND_ARGS_ADAPTER`](op.md#BOUND_ARGS_ADAPTER), [`_abstractify`](op.md#_abstractify), [`AUTO`](op.md#AUTO), [`bind`](normalization/base.md#Normalization.bind), [`_ATTN_ARG_SPECS`](op_test.md#_ATTN_ARG_SPECS), [`_RAGGED_DOT_ARG_SPECS`](op_test.md#_RAGGED_DOT_ARG_SPECS), [`replace`](op.md#BoundArguments.replace), [`_AUTOTUNE_CONFIG`](op_test.md#_AUTOTUNE_CONFIG), [`use_stable_softmax`](attention/pallas_triton.md#PallasTritonFlashAttention.use_stable_softmax), [`_GLU_ARG_SPECS`](op_test.md#_GLU_ARG_SPECS), [`_NORM_ARG_SPECS`](op_test.md#_NORM_ARG_SPECS), [`_eval_shape`](op_test.md#_eval_shape), [`split_k_intermediate_dtype`](ragged_dot/pallas_triton.md#PallasTritonRaggedDot.split_k_intermediate_dtype)

### `OpTest`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/op_test.py:98`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L98)
- signature: `class OpTest(parameterized.TestCase):`
- members:
  - `test_bind(self)` — [`L100`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L100)
- uses (calls/refs, reference-scoped): [`bind`](op.md#Op.bind), [`args`](op.md#BoundArguments.args), [`_FakeOp`](op_test.md#_FakeOp)

### `_FakeOp`  ·  implements/extends Op
- def: [`tokamax/_src/ops/op_test.py:81`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L81)
- signature: `class _FakeOp(op_lib.Op[Any, jax.Array, types.NoneType, _FakeOpConfig, Any]):`
- members:
  - `config_cls` — [`L82`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L82)
- protocol/private: `_fwd`[`L84`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L84), `_get_autotuning_configs`[`L93`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L93), `_get_heuristics_config`[`L89`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L89)
- uses (calls/refs, reference-scoped): [`BoundArguments`](op.md#BoundArguments), [`Op`](op.md#Op), [`_HEURISTICS_CONFIG`](op_test.md#_HEURISTICS_CONFIG), [`_FakeOpConfig`](op_test.md#_FakeOpConfig), [`_AUTOTUNE_CONFIG`](op_test.md#_AUTOTUNE_CONFIG)
- used by: [`Op`](op.md#Op), [`_get_heuristics_config`](op.md#Op._get_heuristics_config), [`_get_autotuning_configs`](op.md#Op._get_autotuning_configs), [`_fwd`](op.md#Op._fwd), [`test_get_config`](op_test.md#BoundArgumentsTest.test_get_config), [`test_default_config`](op_test.md#BoundArgumentsTest.test_default_config), [`test_autotune`](op_test.md#BoundArgumentsTest.test_autotune), [`test_autotune_vmap`](op_test.md#BoundArgumentsTest.test_autotune_vmap), [`test_autotuning_configs`](op_test.md#BoundArgumentsTest.test_autotuning_configs), [`test_heuristics_config`](op_test.md#BoundArgumentsTest.test_heuristics_config), [`test_bind`](op_test.md#OpTest.test_bind), [`test_equals`](op_test.md#BoundArgumentsTest.test_equals), [`test_hash`](op_test.md#BoundArgumentsTest.test_hash)

### `_FakeOpConfig`
- def: [`tokamax/_src/ops/op_test.py:73`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L73)
- signature: `class _FakeOpConfig:`
- members:
  - `blah` — [`L74`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L74)
- used by: [`_FakeOp`](op_test.md#_FakeOp), [`test_get_config`](op_test.md#BoundArgumentsTest.test_get_config), [`test_autotune`](op_test.md#BoundArgumentsTest.test_autotune), [`test_autotune_vmap`](op_test.md#BoundArgumentsTest.test_autotune_vmap), [`_HEURISTICS_CONFIG`](op_test.md#_HEURISTICS_CONFIG), [`_AUTOTUNE_CONFIG`](op_test.md#_AUTOTUNE_CONFIG), [`config_cls`](op_test.md#_FakeOp.config_cls)

## Functions
- `_bsd(shape, dtype, vmap_axes)` — [`L46`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L46)
- `_eval_shape(spec)` — [`L52`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L52)
- `f()` — [`L60`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L60)

## Module values
- `_ATTN_ARG_SPECS` — [`L40`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L40)
- `_AUTOTUNE_CONFIG` — [`L78`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L78)
- `_GLU_ARG_SPECS` — [`L41`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L41)
- `_HEURISTICS_CONFIG` — [`L77`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L77)
- `_NORM_ARG_SPECS` — [`L42`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L42)
- `_RAGGED_DOT_ARG_SPECS` — [`L43`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op_test.py#L43)

