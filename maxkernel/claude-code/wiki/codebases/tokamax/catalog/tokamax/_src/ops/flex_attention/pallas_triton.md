---
title: 'Module: tokamax/_src/ops/flex_attention/pallas_triton.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/flex_attention/pallas_triton.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.flex_attention.pallas_triton`/
symbols:
  _fwd: _fwd().
  PallasTritonFlexAttention._fwd: PallasTritonFlexAttention#_fwd().
  PallasTritonFlexAttention._get_heuristics_config: PallasTritonFlexAttention#_get_heuristics_config().
  PallasTritonFlexAttention._get_autotuning_configs: PallasTritonFlexAttention#_get_autotuning_configs().
  QArray: QArray.
  _fwd.input_spec: _fwd().input_spec().
  Config: Config#
  _fwd_kernel: _fwd_kernel().
  PallasTritonFlexAttention: PallasTritonFlexAttention#
  _fwd.bcast_spec: _fwd().bcast_spec().
  _fwd.spec: _fwd().spec().
  _fwd_kernel.body: _fwd_kernel().body().
  _tile_mask_mod: _tile_mask_mod().
  _fwd_kernel.get_values_and_scales: _fwd_kernel().get_values_and_scales().
  ScoreMod: ScoreMod.
  MaskMod: MaskMod.
  Residuals: Residuals.
  _can_have_block_d: _can_have_block_d().
  PallasTritonFlexAttention.use_stable_softmax: PallasTritonFlexAttention#use_stable_softmax.
  _rescale: _rescale().
  Config.block_q: Config#block_q.
  Config.block_k: Config#block_k.
  Config.num_stages: Config#num_stages.
  Config.num_warps: Config#num_warps.
  Config.block_d: Config#block_d.
  Config.block_d_out: Config#block_d_out.
  PallasTritonFlexAttention.config_cls: PallasTritonFlexAttention#config_cls.
  _fwd_kernel.body.load_mod_fn_value: _fwd_kernel().body().load_mod_fn_value().
  _tile_score_mod: _tile_score_mod().
  _tile_mask_mod.score_mod: _tile_mask_mod().score_mod().
  _fwd.spec.batched_index_map: _fwd().spec().batched_index_map().
  _fwd.bcast_spec.bcast_index_map: _fwd().bcast_spec().bcast_index_map().
  PallasTritonFlexAttention.use_base2: PallasTritonFlexAttention#use_base2.
  PallasTritonFlexAttention.supports_symbolic_shapes: PallasTritonFlexAttention#supports_symbolic_shapes.
  PallasTritonFlexAttention.supports_batched_args_capture: PallasTritonFlexAttention#supports_batched_args_capture.
  PallasTritonFlexAttention.supported_on: PallasTritonFlexAttention#supported_on().
---
# Module: [`tokamax/_src/ops/flex_attention/pallas_triton.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py)

## Classes
### `Config`
- def: [`tokamax/_src/ops/flex_attention/pallas_triton.py:430`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L430) — documented in [tokamax-_src-ops-flex_attention-pallas_triton](../../../../../concepts/tokamax-_src-ops-flex_attention-pallas_triton.md)
- members:
  - `block_d` — [`L435`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L435) — documented in [tokamax-_src-ops-flex_attention-pallas_triton](../../../../../concepts/tokamax-_src-ops-flex_attention-pallas_triton.md)
  - `block_d_out` — [`L436`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L436) — documented in [tokamax-_src-ops-flex_attention-pallas_triton](../../../../../concepts/tokamax-_src-ops-flex_attention-pallas_triton.md)
  - `block_k` — [`L432`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L432) — documented in [tokamax-_src-ops-flex_attention-pallas_triton](../../../../../concepts/tokamax-_src-ops-flex_attention-pallas_triton.md)
  - `block_q` — [`L431`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L431) — documented in [tokamax-_src-ops-flex_attention-pallas_triton](../../../../../concepts/tokamax-_src-ops-flex_attention-pallas_triton.md)
  - `num_stages` — [`L433`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L433) — documented in [tokamax-_src-ops-flex_attention-pallas_triton](../../../../../concepts/tokamax-_src-ops-flex_attention-pallas_triton.md)
  - `num_warps` — [`L434`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L434) — documented in [tokamax-_src-ops-flex_attention-pallas_triton](../../../../../concepts/tokamax-_src-ops-flex_attention-pallas_triton.md)
- used by: [`_fwd`](pallas_triton.md#_fwd), [`_fwd`](pallas_triton.md#PallasTritonFlexAttention._fwd), [`_get_heuristics_config`](pallas_triton.md#PallasTritonFlexAttention._get_heuristics_config), [`_get_autotuning_configs`](pallas_triton.md#PallasTritonFlexAttention._get_autotuning_configs), [`PallasTritonFlexAttention`](pallas_triton.md#PallasTritonFlexAttention), [`config_cls`](pallas_triton.md#PallasTritonFlexAttention.config_cls)

### `PallasTritonFlexAttention`  ·  implements/extends FlexAttention
- def: [`tokamax/_src/ops/flex_attention/pallas_triton.py:440`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L440)
- doc: Pallas-Triton FlexAttention implementation.
- signature: `class PallasTritonFlexAttention(base.FlexAttention[Config, None]):`
- members:
  - `supported_on(self, device: jax.Device)` — [`L543`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L543)
  - `config_cls` — [`L443`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L443)
  - `supports_batched_args_capture` — [`L445`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L445)
  - `supports_symbolic_shapes` — [`L444`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L444)
  - `use_base2` — [`L446`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L446) — documented in [tokamax-_src-ops-flex_attention-pallas_triton](../../../../../concepts/tokamax-_src-ops-flex_attention-pallas_triton.md)
  - `use_stable_softmax` — [`L447`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L447)
- protocol/private: `_fwd`[`L451`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L451), `_get_autotuning_configs`[`L524`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L524), `_get_heuristics_config`[`L495`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L495)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`_fwd`](pallas_triton.md#_fwd), [`jaxtyped`](../../jaxtyping.md#jaxtyped), [`FlexAttention`](base.md#FlexAttention), [`args`](../op.md#BoundArguments.args), [`QArray`](pallas_triton.md#QArray), [`Config`](pallas_triton.md#Config), [`CanonicalPrecision`](base.md#CanonicalPrecision), [`MaskMod`](pallas_triton.md#MaskMod), [`Residuals`](pallas_triton.md#Residuals), [`ScoreMod`](pallas_triton.md#ScoreMod), [`_can_have_block_d`](pallas_triton.md#_can_have_block_d), [`block_d`](pallas_triton.md#Config.block_d), [`block_d_out`](pallas_triton.md#Config.block_d_out), [`block_k`](pallas_triton.md#Config.block_k), [`block_q`](pallas_triton.md#Config.block_q), [`num_stages`](pallas_triton.md#Config.num_stages), [`num_warps`](pallas_triton.md#Config.num_warps)
- used by: [`_get_heuristics_config`](../op.md#Op._get_heuristics_config), [`_get_autotuning_configs`](../op.md#Op._get_autotuning_configs), [`supported_on`](../op.md#Op.supported_on), [`FlexAttention`](base.md#FlexAttention), [`_fwd`](base.md#FlexAttention._fwd), [`_IMPLS`](../attention/bench.md#_IMPLS), [`__init__`](pallas_triton_test.md#PallasTritonFlexAttentionTest.__init__), [`__init__`](pallas_triton_test.md#WrappedPallasTritonFlexAttentionTest.__init__)

## Functions
- `_can_have_block_d(*args)` — [`L418`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L418)
- `_fwd(q: Float[Array | QArray, "*B T H D"], k: Float[Array | QArray, "*B t h D"], v: Float[Array | QArray, "*B t h d"], *, score_mod: ScoreMod | None, mask_mod: MaskMod | None, dropout_mask: Bool[Array, "*#B #H #T #t"] | None, dropout_rate: float, config: Config, use_base2: bool, use_stable_softmax: bool, normalize_output: bool, return_residuals: bool, q_k_dot_precision: jax.lax.DotAlgorithmPreset, weights_v_dot_precision: jax.lax.DotAlgorithmPreset)` — [`L271`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L271) — Forward pass of Pallas-Trtion FlexAttention.
- `_fwd_kernel(q_ref, k_ref, v_ref, score_mod_value_refs, mask_mod_value_refs, dropout_mask_ref, out_ref, l_ref, m_ref, block_k: int, block_d: int | None, block_d_out: int | None, score_mod_fn, score_mod_value_specs, mask_mod_fn, mask_mod_value_specs, dropout_rate: float, use_base2: bool, use_stable_softmax: bool, q_k_dot_precision: jax.lax.DotAlgorithmPreset, weights_v_dot_precision: jax.lax.DotAlgorithmPreset, normalize_output: bool)` — [`L66`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L66) — Pallas-Triton FlexAttention forward kernel implementation.
- `_rescale(values, scales_ref, slice_k, slice_d, keep_quantized=False)` — [`L45`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L45) — Load the scales correctly and apply them to the values if requested.
- `_tile_mask_mod(mask_mod, scores, block_q, block_k)` — [`L255`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L255) — Returns tiled version of `mask_mod` with input values and specs.
- `_tile_score_mod(score_mod, scores, block_q, block_k)` — [`L235`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L235) — Returns tiled version of `score_mod` with input values and specs.
- `batched_index_map(*pids)` — [`L307`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L307)
- `bcast_index_map(*pids)` — [`L317`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L317)
- `bcast_spec(x, index_map, block_shape)` — [`L314`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L314)
- `body(i, carry)` — [`L134`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L134)
- `get_values_and_scales(x)` — [`L92`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L92)
- `input_spec(x, index_map, block_shape)` — [`L327`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L327)
- `load_mod_fn_value(ref, spec)` — [`L157`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L157)
- `score_mod(scores)` — [`L262`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L262)
- `spec(index_map, block_shape)` — [`L306`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L306)

## Module values
- `MaskMod` — [`L40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L40) — documented in [tokamax-_src-ops-flex_attention-pallas_triton](../../../../../concepts/tokamax-_src-ops-flex_attention-pallas_triton.md)
- `QArray` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L41)
- `Residuals` — [`L42`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L42)
- `ScoreMod` — [`L39`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/pallas_triton.py#L39) — documented in [tokamax-_src-ops-flex_attention-pallas_triton](../../../../../concepts/tokamax-_src-ops-flex_attention-pallas_triton.md)

