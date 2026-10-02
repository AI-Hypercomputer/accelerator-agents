---
title: 'Module: tokamax/_src/ops/attention/pallas_triton_vjp.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/pallas_triton_vjp.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.pallas_triton_vjp`/
symbols:
  PallasTritonFlashAttentionVjp._fwd: PallasTritonFlashAttentionVjp#_fwd().
  _bwd: _bwd().
  PallasTritonFlashAttentionVjp._get_heuristics_config: PallasTritonFlashAttentionVjp#_get_heuristics_config().
  PallasTritonFlashAttentionVjp._get_autotuning_configs: PallasTritonFlashAttentionVjp#_get_autotuning_configs().
  Config: Config#
  PallasTritonFlashAttentionVjp: PallasTritonFlashAttentionVjp#
  _zero_ds: _zero_ds().
  Config.num_warps: Config#num_warps.
  _bwd_kernel.calc_dkdv: _bwd_kernel().calc_dkdv().
  Config.block_n1: Config#block_n1.
  Residuals: Residuals.
  Config.block_m2: Config#block_m2.
  Mask: Mask.
  PagingInfo: PagingInfo.
  _bwd_dkdv: _bwd_dkdv().
  _bwd_dq: _bwd_dq().
  _zero_ds.body: _zero_ds().body().
  Config.block_m1: Config#block_m1.
  Config.block_n2: Config#block_n2.
  Config.num_stages: Config#num_stages.
  PallasTritonFlashAttentionVjp.dbias_intermediate_dtype: PallasTritonFlashAttentionVjp#dbias_intermediate_dtype.
  PallasTritonFlashAttentionVjp.broadcast_to_rank: PallasTritonFlashAttentionVjp#broadcast_to_rank().
  _bwd_kernel.calc_dq: _bwd_kernel().calc_dq().
  PallasTritonFlashAttentionVjp.config_cls: PallasTritonFlashAttentionVjp#config_cls.
  _bwd_dkdv.body: _bwd_dkdv().body().
  _bwd_dq.body: _bwd_dq().body().
  _bwd_kernel: _bwd_kernel().
  PallasTritonFlashAttentionVjp.supports_symbolic_shapes: PallasTritonFlashAttentionVjp#supports_symbolic_shapes.
  PallasTritonFlashAttentionVjp.supported_on: PallasTritonFlashAttentionVjp#supported_on().
---
# Module: [`tokamax/_src/ops/attention/pallas_triton_vjp.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py)

## Classes
### `Config`
- def: [`tokamax/_src/ops/attention/pallas_triton_vjp.py:345`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L345)
- signature: `class Config:`
- members:
  - `block_m1` — [`L346`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L346)
  - `block_m2` — [`L348`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L348)
  - `block_n1` — [`L347`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L347)
  - `block_n2` — [`L349`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L349)
  - `num_stages` — [`L350`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L350)
  - `num_warps` — [`L351`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L351)
- uses (calls/refs, reference-scoped): [`jaxtyped`](../../jaxtyping.md#jaxtyped)
- used by: [`_fwd`](pallas_triton_vjp.md#PallasTritonFlashAttentionVjp._fwd), [`_bwd`](pallas_triton_vjp.md#_bwd), [`_get_autotuning_configs`](pallas_triton_vjp.md#PallasTritonFlashAttentionVjp._get_autotuning_configs), [`_get_heuristics_config`](pallas_triton_vjp.md#PallasTritonFlashAttentionVjp._get_heuristics_config), [`PallasTritonFlashAttentionVjp`](pallas_triton_vjp.md#PallasTritonFlashAttentionVjp), [`config_cls`](pallas_triton_vjp.md#PallasTritonFlashAttentionVjp.config_cls)

### `PallasTritonFlashAttentionVjp`  ·  implements/extends DotProductAttentionVjp
- def: [`tokamax/_src/ops/attention/pallas_triton_vjp.py:489`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L489)
- doc: Pallas-Triton FlashAttention VJP implementation.
- signature: `class PallasTritonFlashAttentionVjp(base.DotProductAttentionVjp[Config, None]):`
- members:
  - `broadcast_to_rank(x, rank)` — [`L539`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L539)
  - `supported_on(self, device: jax.Device)` — [`L621`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L621)
  - `config_cls` — [`L492`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L492)
  - `dbias_intermediate_dtype` — [`L494`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L494)
  - `supports_symbolic_shapes` — [`L493`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L493)
- protocol/private: `_fwd`[`L497`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L497), `_get_autotuning_configs`[`L602`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L602), `_get_heuristics_config`[`L587`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L587)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`as_array`](base.md#Mask.as_array), [`_bwd`](pallas_triton_vjp.md#_bwd), [`DotProductAttentionVjp`](base.md#DotProductAttentionVjp), [`CanonicalPrecision`](base.md#CanonicalPrecision), [`vmap_batch_dims`](base.md#vmap_batch_dims), [`Config`](pallas_triton_vjp.md#Config), [`DotProductAttentionGrads`](base.md#DotProductAttentionGrads), [`take`](base.md#Mask.take), [`num_warps`](pallas_triton_vjp.md#Config.num_warps), [`block_n1`](pallas_triton_vjp.md#Config.block_n1), [`Residuals`](pallas_triton_vjp.md#Residuals), [`block_m2`](pallas_triton_vjp.md#Config.block_m2), [`k`](base.md#DotProductAttentionGrads.k), [`q`](base.md#DotProductAttentionGrads.q), [`v`](base.md#DotProductAttentionGrads.v), [`Mask`](pallas_triton_vjp.md#Mask), [`PagingInfo`](pallas_triton_vjp.md#PagingInfo), [`bias`](base.md#DotProductAttentionGrads.bias), [`block_m1`](pallas_triton_vjp.md#Config.block_m1), [`block_n2`](pallas_triton_vjp.md#Config.block_n2), [`num_stages`](pallas_triton_vjp.md#Config.num_stages)
- used by: [`_get_heuristics_config`](../op.md#Op._get_heuristics_config), [`_get_autotuning_configs`](../op.md#Op._get_autotuning_configs), [`supported_on`](../op.md#Op.supported_on), [`_fwd`](base.md#DotProductAttentionVjp._fwd), [`DotProductAttentionVjp`](base.md#DotProductAttentionVjp), [`__init__`](pallas_triton_test.md#PallasTritonFlashAttentionWithPallasTritonVjpTest.__init__), [`__post_init__`](pallas_triton.md#PallasTritonFlashAttention.__post_init__)

## Functions
- `_bwd(q: Float[Array, "T H D"], k: Float[Array, "t h D"], v: Float[Array, "t h d"], bias: Float[Array, "#H #T #t"] | None, mask: Bool[Array, "#H #T #t"] | None, dropout_mask: Bool[Array, "#H #T #t"] | None, residuals: Residuals, out: Float[Array, "T H d"], dout: Float[Array, "T H d"], *, q_k_dot_precision: jax.lax.DotAlgorithmPreset, logits_dtype: jnp.dtype, logits_scale: float, logits_soft_cap: float | None, is_causal: bool, dropout_rate: float, weights_v_dot_precision: jax.lax.DotAlgorithmPreset, ds_dtype: jax.typing.DTypeLike | None, config: Config)` — [`L355`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L355) — Compute FlashAttention VJP.
- `_bwd_dkdv(dk, dv, q_ref, k, v, bias_ref, mask_ref, do_ref, m_ref, l_ref, delta_ref, ds_ref, lo, hi, *, block_m1: int, is_causal: bool = False, logits_dtype: jnp.dtype, q_k_dot_precision: jax.lax.DotAlgorithmPreset, weights_v_dot_precision: jax.lax.DotAlgorithmPreset)` — [`L42`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L42) — Computes dk and dv.
- `_bwd_dq(dq, q, k_ref, v_ref, bias_ref, mask_ref, do, m, l, delta, lo, hi, *, block_n2: int, is_causal: bool = False, logits_dtype: jnp.dtype, q_k_dot_precision: jax.lax.DotAlgorithmPreset, weights_v_dot_precision: jax.lax.DotAlgorithmPreset)` — [`L138`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L138) — Computes dq.
- `_bwd_kernel(q_ref, k_ref, v_ref, bias_ref, mask_ref, m_ref, l_ref, delta_ref, dout_ref, dq_ref, dk_ref, dv_ref, ds_ref, *, block_m1: int, block_n2: int, mask_block_slice_factor: int = 2, sm_scale: float, is_causal: bool, logits_dtype: jnp.dtype, q_k_dot_precision: jax.lax.DotAlgorithmPreset, weights_v_dot_precision: jax.lax.DotAlgorithmPreset)` — [`L221`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L221) — Pallas MHA backward kernel implementation.
- `_zero_ds(ds_ref, lo, hi, *, block_m: int, block_n: int)` — [`L211`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L211)
- `body(i, carry)` — [`L84`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L84)
- `body(i, dq)` — [`L178`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L178)
- `body(i, _)` — [`L214`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L214)
- `calc_dkdv()` — [`L255`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L255)
- `calc_dq()` — [`L301`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L301)

## Module values
- `Mask` — [`L37`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L37)
- `PagingInfo` — [`L39`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L39)
- `Residuals` — [`L38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton_vjp.py#L38)

