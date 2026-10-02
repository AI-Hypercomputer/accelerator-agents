---
title: 'Module: tokamax/_src/ops/attention/pallas_mosaic_gpu.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/pallas_mosaic_gpu.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.pallas_mosaic_gpu`/
symbols:
  PallasMosaicGpuFlashAttention._fwd: PallasMosaicGpuFlashAttention#_fwd().
  Config: Config.
  PallasMosaicGpuFlashAttention: PallasMosaicGpuFlashAttention#
  PallasMosaicGpuFlashAttention._get_heuristics_config: PallasMosaicGpuFlashAttention#_get_heuristics_config().
  PallasMosaicGpuFlashAttention._get_autotuning_configs: PallasMosaicGpuFlashAttention#_get_autotuning_configs().
  QArray: QArray.
  PallasMosaicGpuFlashAttention.__post_init__: PallasMosaicGpuFlashAttention#__post_init__().
  ConfigSM90: ConfigSM90.
  ConfigSM100: ConfigSM100.
  _broadcast_to_rank: _broadcast_to_rank().
  PagingInfo: PagingInfo.
  Residuals: Residuals.
  PallasMosaicGpuFlashAttention.use_stable_softmax: PallasMosaicGpuFlashAttention#use_stable_softmax.
  PallasMosaicGpuFlashAttention.pad_seq_k: PallasMosaicGpuFlashAttention#pad_seq_k().
  _get_kernel_module: _get_kernel_module().
  Mask: Mask.
  PallasMosaicGpuFlashAttention.config_cls: PallasMosaicGpuFlashAttention#config_cls.
  PallasMosaicGpuFlashAttention.rescale_threshold: PallasMosaicGpuFlashAttention#rescale_threshold.
  Key.Key: Key.Key.
  PallasMosaicGpuFlashAttention.supports_symbolic_shapes: PallasMosaicGpuFlashAttention#supports_symbolic_shapes.
  PallasMosaicGpuFlashAttention.supported_on: PallasMosaicGpuFlashAttention#supported_on().
---
# Module: [`tokamax/_src/ops/attention/pallas_mosaic_gpu.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu.py)

## Classes
### `PallasMosaicGpuFlashAttention`  ·  implements/extends DotProductAttention
- def: [`tokamax/_src/ops/attention/pallas_mosaic_gpu.py:63`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu.py#L63)
- doc: Flash attention with Mosaic GPU.
- signature: `class PallasMosaicGpuFlashAttention(base.DotProductAttention[Config, Key]):`
- members:
  - `_fwd(self, q: Float[Array | QArray, "*B T H D"], k: Float[Array | QArray, "*B t h D"], v: Float[Array | QArray, "*B t h d"], *, precision: tuple[base.CanonicalPrecision, base.CanonicalPrecision], logits_dtype: jnp.dtype, logits_scale: float, bias: Float[Array, "*#B #H #T #t"] | None, logits_soft_cap: float | None, mask: base.Mask, dropout_mask: Bool[Array, "*#B #H #T #t"] | None, dropout_rate: float, paging_info: PagingInfo | None, q_indices: Int[Array, "*#B #H T"] | None, k_indices: Int[Array, "*#B #h t"] | None, normalize_output: bool, return_residuals: bool, config: Config)` — [`L94`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu.py#L94) — Performs attention, optionally returning softmax residuals. — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_kernel_sm100](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_kernel_sm100.md)
  - `pad_seq_k(x, axis)` — [`L191`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu.py#L191) — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_common](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_common.md)
  - `supported_on(self, device: jax.Device)` — [`L228`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu.py#L228)
  - `config_cls` — [`L82`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu.py#L82)
  - `rescale_threshold` — [`L85`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu.py#L85)
  - `supports_symbolic_shapes` — [`L83`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu.py#L83)
  - `use_stable_softmax` — [`L84`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu.py#L84)
- protocol/private: `__post_init__`[`L87`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu.py#L87), `_get_autotuning_configs`[`L224`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu.py#L224), `_get_heuristics_config`[`L220`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu.py#L220)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`DotProductAttention`](base.md#DotProductAttention), [`Mask`](base.md#Mask), [`jaxtyped`](../../jaxtyping.md#jaxtyped), [`flash_attention_kernel`](pallas_mosaic_gpu_kernel_sm100.md#flash_attention_kernel), [`get_autotuning_configs`](pallas_mosaic_gpu_kernel_sm100.md#get_autotuning_configs), [`get_heuristics_config`](pallas_mosaic_gpu_kernel_sm100.md#get_heuristics_config), [`vjp`](../op.md#Op.vjp), [`AUTO`](base.md#AUTO), [`fold_q_sequence_heads`](base.md#fold_q_sequence_heads), [`CanonicalPrecision`](base.md#CanonicalPrecision), [`block_kv`](pallas_mosaic_gpu_common.md#ConfigBase.block_kv), [`vmap_batch_dims`](base.md#vmap_batch_dims), [`Config`](pallas_mosaic_gpu.md#Config), [`vmap_split`](../../batching.md#vmap_split), [`ConfigBase`](pallas_mosaic_gpu_common.md#ConfigBase), [`PallasMosaicGpuFlashAttentionVjp`](pallas_mosaic_gpu_vjp.md#PallasMosaicGpuFlashAttentionVjp), [`decompose_mask`](pallas_mosaic_gpu_common.md#decompose_mask), [`cast_qkv`](pallas_mosaic_gpu_common.md#cast_qkv), [`QArray`](pallas_mosaic_gpu.md#QArray), [`ConfigSM100`](pallas_mosaic_gpu.md#ConfigSM100), [`ConfigSM90`](pallas_mosaic_gpu.md#ConfigSM90), [`combine_partial_results`](base.md#combine_partial_results), [`_broadcast_to_rank`](pallas_mosaic_gpu.md#_broadcast_to_rank), [`split_k`](pallas_mosaic_gpu_common.md#ConfigBase.split_k), [`PagingInfo`](pallas_mosaic_gpu.md#PagingInfo), [`Residuals`](pallas_mosaic_gpu.md#Residuals), [`unfold_q_sequence_heads`](base.md#unfold_q_sequence_heads), [`_get_kernel_module`](pallas_mosaic_gpu.md#_get_kernel_module), [`fold_q_sequence_heads`](pallas_mosaic_gpu_common.md#ConfigBase.fold_q_sequence_heads), [`needs_stable_softmax`](base.md#needs_stable_softmax), [`Key`](pallas_mosaic_gpu.md#Key.Key)
- used by: [`DotProductAttention`](base.md#DotProductAttention), [`_get_heuristics_config`](../op.md#Op._get_heuristics_config), [`_get_autotuning_configs`](../op.md#Op._get_autotuning_configs), [`supported_on`](../op.md#Op.supported_on), [`_fwd`](base.md#DotProductAttention._fwd), [`_IMPLS`](bench.md#_IMPLS), [`_DEFAULT_IMPLEMENTATIONS`](api.md#_DEFAULT_IMPLEMENTATIONS), [`__init__`](pallas_mosaic_gpu_test.md#PallasMosaicGpuFlashAttentionTest.__init__)

## Functions
- `_broadcast_to_rank(x, rank)` — [`L50`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu.py#L50)
- `_get_kernel_module()` — [`L54`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu.py#L54)

## Module values
- `Config` — [`L42`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu.py#L42)
- `ConfigSM100` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu.py#L41)
- `ConfigSM90` — [`L40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu.py#L40)
- `Key` — [`L43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu.py#L43)
- `Mask` — [`L44`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu.py#L44)
- `PagingInfo` — [`L45`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu.py#L45)
- `QArray` — [`L46`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu.py#L46)
- `Residuals` — [`L47`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu.py#L47)

