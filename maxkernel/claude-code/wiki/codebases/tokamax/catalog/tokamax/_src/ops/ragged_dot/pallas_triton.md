---
title: 'Module: tokamax/_src/ops/ragged_dot/pallas_triton.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/pallas_triton.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.pallas_triton`/
symbols:
  _ragged_dot: _ragged_dot().
  PallasTritonRaggedDot._fwd: PallasTritonRaggedDot#_fwd().
  _ragged_contracting_dim_dot: _ragged_contracting_dim_dot().
  PallasTritonRaggedDot._get_autotuning_configs: PallasTritonRaggedDot#_get_autotuning_configs().
  PallasTritonRaggedDot._get_heuristics_config: PallasTritonRaggedDot#_get_heuristics_config().
  _ragged_contracting_dim_dot.f: _ragged_contracting_dim_dot().f().
  PallasTritonRaggedDot: PallasTritonRaggedDot#
  Config: Config#
  QArray: QArray.
  PallasTritonRaggedDot.__post_init__: PallasTritonRaggedDot#__post_init__().
  Config.block_m: Config#block_m.
  Config.block_n: Config#block_n.
  Config.block_k: Config#block_k.
  Config.split_k: Config#split_k.
  Config.num_warps: Config#num_warps.
  Config.num_stages: Config#num_stages.
  _ragged_dot.f: _ragged_dot().f().
  AsQArray: AsQArray.
  PallasTritonRaggedDot.handle_residuals: PallasTritonRaggedDot#handle_residuals().
  GroupSizes: GroupSizes.
  _ragged_contracting_dim_dot_kernel: _ragged_contracting_dim_dot_kernel().
  PallasTritonRaggedDot.split_k_intermediate_dtype: PallasTritonRaggedDot#split_k_intermediate_dtype.
  Residuals: Residuals.
  _ragged_dot_kernel.f: _ragged_dot_kernel().f().
  PallasTritonRaggedDot.config_cls: PallasTritonRaggedDot#config_cls.
  _ragged_contracting_dim_dot_kernel.body: _ragged_contracting_dim_dot_kernel().body().
  _ragged_dot_kernel: _ragged_dot_kernel().
  _ragged_dot_kernel.f.body: _ragged_dot_kernel().f().body().
  PallasTritonRaggedDot.supports_symbolic_shapes: PallasTritonRaggedDot#supports_symbolic_shapes.
  PallasTritonRaggedDot.supported_on: PallasTritonRaggedDot#supported_on().
---
# Module: [`tokamax/_src/ops/ragged_dot/pallas_triton.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py)

## Classes
### `Config`
- def: [`tokamax/_src/ops/ragged_dot/pallas_triton.py:48`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L48) — documented in [tokamax-_src-ops-ragged_dot-pallas_triton](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_triton.md)
- signature: `class Config:`
- members:
  - `block_k` — [`L51`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L51) — documented in [tokamax-_src-ops-ragged_dot-pallas_triton](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_triton.md)
  - `block_m` — [`L49`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L49) — documented in [tokamax-_src-ops-ragged_dot-pallas_triton](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_triton.md)
  - `block_n` — [`L50`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L50) — documented in [tokamax-_src-ops-ragged_dot-pallas_triton](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_triton.md)
  - `num_stages` — [`L54`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L54) — documented in [tokamax-_src-ops-ragged_dot-pallas_triton](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_triton.md)
  - `num_warps` — [`L53`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L53) — documented in [tokamax-_src-ops-ragged_dot-pallas_triton](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_triton.md)
  - `split_k` — [`L52`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L52) — documented in [tokamax-_src-ops-ragged_dot-pallas_triton](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_triton.md)
- used by: [`_ragged_dot`](pallas_triton.md#_ragged_dot), [`_fwd`](pallas_triton.md#PallasTritonRaggedDot._fwd), [`_ragged_contracting_dim_dot`](pallas_triton.md#_ragged_contracting_dim_dot), [`_get_autotuning_configs`](pallas_triton.md#PallasTritonRaggedDot._get_autotuning_configs), [`test_split_k`](pallas_triton_test.md#PallasTritonRaggedDotTest.test_split_k), [`test_split_k_quantized`](pallas_triton_test.md#PallasTritonRaggedDotTest.test_split_k_quantized), [`_get_heuristics_config`](pallas_triton.md#PallasTritonRaggedDot._get_heuristics_config), [`PallasTritonRaggedDot`](pallas_triton.md#PallasTritonRaggedDot), [`f`](pallas_triton.md#_ragged_contracting_dim_dot.f), [`config_cls`](pallas_triton.md#PallasTritonRaggedDot.config_cls)

### `PallasTritonRaggedDot`  ·  implements/extends RaggedDot
- def: [`tokamax/_src/ops/ragged_dot/pallas_triton.py:343`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L343)
- doc: Pallas-Triton ragged dot implementation.
- signature: `class PallasTritonRaggedDot(base.RaggedDot[Config, None]):`
- members:
  - `handle_residuals(dot_out: jax.Array)` — [`L408`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L408) — Handles the residuals for the ragged dot operation.
  - `supported_on(self, device: jax.Device)` — [`L503`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L503)
  - `config_cls` — [`L346`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L346)
  - `split_k_intermediate_dtype` — [`L348`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L348)
  - `supports_symbolic_shapes` — [`L347`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L347)
- protocol/private: `__post_init__`[`L350`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L350), `_fwd`[`L358`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L358), `_get_autotuning_configs`[`L467`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L467), `_get_heuristics_config`[`L456`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L456)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`_ragged_dot`](pallas_triton.md#_ragged_dot), [`RaggedDot`](base.md#RaggedDot), [`args`](../op.md#BoundArguments.args), [`_ragged_contracting_dim_dot`](pallas_triton.md#_ragged_contracting_dim_dot), [`vjp`](../op.md#Op.vjp), [`vjp`](base.md#vjp), [`Config`](pallas_triton.md#Config), [`DEFAULT_RAGGED_DOT_DIM_NUMS`](base.md#DEFAULT_RAGGED_DOT_DIM_NUMS), [`QArray`](pallas_triton.md#QArray), [`Residuals`](base.md#Residuals), [`CanonicalPrecision`](base.md#CanonicalPrecision), [`vmap_axis_sizes`](../op.md#BoundArguments.vmap_axis_sizes), [`block_k`](pallas_triton.md#Config.block_k), [`block_m`](pallas_triton.md#Config.block_m), [`block_n`](pallas_triton.md#Config.block_n), [`num_stages`](pallas_triton.md#Config.num_stages), [`num_warps`](pallas_triton.md#Config.num_warps), [`split_k`](pallas_triton.md#Config.split_k), [`AsQArray`](pallas_triton.md#AsQArray), [`RAGGED_CONTRACTING_DOT_DIM_NUMS`](base.md#RAGGED_CONTRACTING_DOT_DIM_NUMS), [`GroupSizes`](pallas_triton.md#GroupSizes), [`TRANS_RHS_RAGGED_DOT_DIM_NUMS`](base.md#TRANS_RHS_RAGGED_DOT_DIM_NUMS)
- used by: [`_get_heuristics_config`](../op.md#Op._get_heuristics_config), [`_get_autotuning_configs`](../op.md#Op._get_autotuning_configs), [`supported_on`](../op.md#Op.supported_on), [`test_roundtrip`](../op_test.md#BoundArgumentsTest.test_roundtrip), [`RaggedDot`](base.md#RaggedDot), [`_fwd`](base.md#RaggedDot._fwd), [`test_split_k`](pallas_triton_test.md#PallasTritonRaggedDotTest.test_split_k), [`test_split_k_quantized`](pallas_triton_test.md#PallasTritonRaggedDotTest.test_split_k_quantized), [`_DEFAULT_IMPLEMENTATIONS`](api.md#_DEFAULT_IMPLEMENTATIONS), [`_IMPLS`](bench.md#_IMPLS), [`__init__`](pallas_triton_test.md#PallasTritonRaggedDotTest.__init__)

## Functions
- `_ragged_contracting_dim_dot(lhs: jax.Array | QArray, rhs: jax.Array | QArray, *, group_sizes: jax.Array, ragged_dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers, precision: base.CanonicalPrecision, out_dtype: jnp.dtype, config: Config, activation: base.ActivationFunction | None = None)` — [`L284`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L284) — Pallas-Triton ragged dot for ragged contracting dimension. — documented in [tokamax-_src-ops-ragged_dot-pallas_triton](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_triton.md)
- `_ragged_contracting_dim_dot_kernel(a_ref, b_ref, lo_ref, hi_ref, out_ref, *, block_m, block_k, precision, activation)` — [`L244`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L244) — Pallas-Triton ragged dot kernel for ragged contracting dimension.
- `_ragged_dot(lhs: jax.Array | QArray, rhs: jax.Array | QArray, *, group_sizes: jax.Array, ragged_dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers, precision: base.CanonicalPrecision, out_dtype: jnp.dtype, split_k_intermediate_dtype: jax.typing.DTypeLike | None, config: Config, activation: base.ActivationFunction | None = None)` — [`L150`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L150) — Pallas-Triton ragged dot. — documented in [tokamax-_src-ops-ragged_dot-pallas_triton](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_triton.md)
- `_ragged_dot_kernel(a_ref, a_scales_ref, b_ref, b_scales_ref, lo_ref, hi_ref, out_ref, *, n, block_m, block_k, precision, activation)` — [`L57`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L57) — Pallas-Triton ragged dot kernel.
- `body(i, acc)` — [`L84`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L84)
- `body(i, acc, mask_k=False)` — [`L260`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L260)
- `f()` — [`L80`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L80)
- `f(lhs, rhs)` — [`L173`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L173)
- `f(lhs, rhs, lo, hi)` — [`L313`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L313)

## Module values
- `AsQArray` — [`L43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L43)
- `GroupSizes` — [`L44`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L44)
- `QArray` — [`L42`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L42)
- `Residuals` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_triton.py#L41)

