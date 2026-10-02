---
title: 'Module: tpu_inference/layers/common/process_weights/moe_weights.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/common/process_weights/moe_weights.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.common.process_weights.moe_weights`/
symbols:
  FusedMoEWeights: FusedMoEWeights#
  FusedMoEWeights.w13_weight_scale: FusedMoEWeights#w13_weight_scale.
  FusedMoEWeights.w2_weight_scale: FusedMoEWeights#w2_weight_scale.
  process_moe_weights: process_moe_weights().
  FusedMoEWeights.w13_weight: FusedMoEWeights#w13_weight.
  shard_moe_weights: shard_moe_weights().
  process_unquantized_moe_weights: process_unquantized_moe_weights().
  FusedMoEWeights.w13_bias: FusedMoEWeights#w13_bias.
  FusedMoEWeights.w2_weight: FusedMoEWeights#w2_weight.
  _process_quantized_moe_weights_impl: _process_quantized_moe_weights_impl().
  FusedMoEWeights.w2_bias: FusedMoEWeights#w2_bias.
  _get_moe_weight_shardings: _get_moe_weight_shardings().
  _process_moe_weights_no_requant: _process_moe_weights_no_requant().
  process_quantized_moe_weights: process_quantized_moe_weights().
  _requant_and_process_local_fn: _requant_and_process_local_fn().
  quantize_moe_weights: quantize_moe_weights().
  process_w13_for_gmm: process_w13_for_gmm().
  get_w13_padding_config: get_w13_padding_config().
  shard_moe_weights_to_tpu: shard_moe_weights_to_tpu().
  UnfusedMoEWeights: UnfusedMoEWeights#
  process_w13_for_gmm._pad_tensor: process_w13_for_gmm()._pad_tensor().
  P: P.
  logger: logger.
  _requant_expert_batch_fn: _requant_expert_batch_fn().
  W13PaddingConfig.padded_intermediate_size: W13PaddingConfig#padded_intermediate_size.
  UnfusedMoEWeights.w1_weight: UnfusedMoEWeights#w1_weight.
  UnfusedMoEWeights.w2_weight: UnfusedMoEWeights#w2_weight.
  UnfusedMoEWeights.w3_weight: UnfusedMoEWeights#w3_weight.
  _get_expert_shard_axis: _get_expert_shard_axis().
  W13PaddingConfig: W13PaddingConfig#
  W13PaddingConfig.intermediate_size: W13PaddingConfig#intermediate_size.
  W13PaddingConfig.w13_reorder_size: W13PaddingConfig#w13_reorder_size.
  W13PaddingConfig.local_intermediate_size: W13PaddingConfig#local_intermediate_size.
  W13PaddingConfig.pad_amount: W13PaddingConfig#pad_amount.
  UnfusedMoEWeights.w1_weight_scale: UnfusedMoEWeights#w1_weight_scale.
  UnfusedMoEWeights.w1_bias: UnfusedMoEWeights#w1_bias.
  UnfusedMoEWeights.w2_weight_scale: UnfusedMoEWeights#w2_weight_scale.
  UnfusedMoEWeights.w2_bias: UnfusedMoEWeights#w2_bias.
  UnfusedMoEWeights.w3_weight_scale: UnfusedMoEWeights#w3_weight_scale.
  UnfusedMoEWeights.w3_bias: UnfusedMoEWeights#w3_bias.
---
# Module: [`tpu_inference/layers/common/process_weights/moe_weights.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py)

## Classes
### `FusedMoEWeights`
- def: [`tpu_inference/layers/common/process_weights/moe_weights.py:41`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L41)
- doc: Fused moe weights. weights can be either jax or torchax array.
- signature: `class FusedMoEWeights:`
- members:
  - `w13_bias` — [`L46`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L46)
  - `w13_weight` — [`L44`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L44)
  - `w13_weight_scale` — [`L45`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L45)
  - `w2_bias` — [`L49`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L49)
  - `w2_weight` — [`L47`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L47)
  - `w2_weight_scale` — [`L48`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L48)
- used by: [`moe_apply`](../moe.md#moe_apply), [`apply_jax`](../../jax/quantization/unquantized.md#UnquantizedFusedMoEMethod.apply_jax), [`apply_jax`](../../jax/quantization/fp8.md#Fp8FusedMoEMethod.apply_jax), [`process_weights_after_loading`](../../jax/quantization/fp8.md#Fp8FusedMoEMethod.process_weights_after_loading), [`process_moe_weights`](moe_weights.md#process_moe_weights), [`shard_moe_weights`](moe_weights.md#shard_moe_weights), [`process_unquantized_moe_weights`](moe_weights.md#process_unquantized_moe_weights), [`_process_quantized_moe_weights_impl`](moe_weights.md#_process_quantized_moe_weights_impl), [`extra_backend_kwargs`](../../jax/quantization/unquantized.md#UnquantizedFusedMoEMethod.extra_backend_kwargs), [`process_mxfp4_moe_weights`](../../vllm/quantization/mxfp4.md#VllmMxfp4MoEMethod.process_mxfp4_moe_weights), [`_get_moe_weight_shardings`](moe_weights.md#_get_moe_weight_shardings), [`process_weights_after_loading`](../../vllm/quantization/fp8.md#VllmFp8MoEMethod.process_weights_after_loading), [`process_awq_moe_weights`](../../vllm/quantization/awq.md#VllmAWQMoEMethod.process_awq_moe_weights), [`vllm_moe_apply`](../../vllm/interface/moe.md#vllm_moe_apply), [`_process_moe_weights_no_requant`](moe_weights.md#_process_moe_weights_no_requant), [`unpack_and_process`](../../vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.md#VllmCompressedTensorsW4A8MoEMethod.unpack_and_process), [`process_weights_after_loading`](../../vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.md#VllmCompressedTensorsW4A8MoEMethod.process_weights_after_loading), [`process_weights_after_loading`](../../vllm/quantization/nvfp4.md#VllmNvfp4MoEMethod.process_weights_after_loading), [`unpack_and_process`](../../vllm/quantization/nvfp4.md#VllmNvfp4MoEMethod.unpack_and_process), [`process_quantized_moe_weights`](moe_weights.md#process_quantized_moe_weights), [`process_weights_after_loading`](../../vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.md#VllmCompressedTensorsW8A8Fp8MoEMethod.process_weights_after_loading), [`_requant_and_process_local_fn`](moe_weights.md#_requant_and_process_local_fn), [`quantize_moe_weights`](moe_weights.md#quantize_moe_weights), [`apply_monolithic`](../../vllm/quantization/fp8.md#VllmFp8MoEMethod.apply_monolithic), [`apply_monolithic`](../../vllm/quantization/unquantized.md#VllmUnquantizedFusedMoEMethod.apply_monolithic), [`apply_monolithic`](../../vllm/quantization/awq.md#VllmAWQMoEMethod.apply_monolithic), [`apply_monolithic`](../../vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.md#VllmCompressedTensorsW4A8MoEMethod.apply_monolithic), [`apply_monolithic`](../../vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.md#VllmCompressedTensorsW8A8Fp8MoEMethod.apply_monolithic), [`apply_monolithic`](../../vllm/quantization/mxfp4.md#VllmMxfp4MoEMethod.apply_monolithic), [`apply_monolithic`](../../vllm/quantization/nvfp4.md#VllmNvfp4MoEMethod.apply_monolithic), [`shard_moe_weights_to_tpu`](moe_weights.md#shard_moe_weights_to_tpu)

### `UnfusedMoEWeights`
- def: [`tpu_inference/layers/common/process_weights/moe_weights.py:54`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L54)
- doc: Unfused moe weights. weights can be either jax or torchax array.
- signature: `class UnfusedMoEWeights:`
- members:
  - `w1_bias` — [`L59`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L59)
  - `w1_weight` — [`L57`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L57)
  - `w1_weight_scale` — [`L58`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L58)
  - `w2_bias` — [`L62`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L62)
  - `w2_weight` — [`L60`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L60)
  - `w2_weight_scale` — [`L61`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L61)
  - `w3_bias` — [`L65`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L65)
  - `w3_weight` — [`L63`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L63)
  - `w3_weight_scale` — [`L64`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L64)
- used by: [`moe_apply`](../moe.md#moe_apply), [`apply_jax`](../../jax/quantization/unquantized.md#UnquantizedFusedMoEMethod.apply_jax), [`sparse_moe_func`](../../jax/moe/sparse_moe.md#sparse_moe_func), [`dense_moe_fwd`](../../jax/moe/dense_moe.md#dense_moe_fwd), [`dense_moe_fwd_preapply_router_weights`](../../jax/moe/dense_moe.md#dense_moe_fwd_preapply_router_weights), [`dense_moe_func`](../../jax/moe/dense_moe.md#dense_moe_func)

### `W13PaddingConfig`
- def: [`tpu_inference/layers/common/process_weights/moe_weights.py:159`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L159)
- signature: `class W13PaddingConfig:`
- members:
  - `intermediate_size` — [`L160`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L160)
  - `local_intermediate_size` — [`L162`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L162)
  - `pad_amount` — [`L163`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L163)
  - `padded_intermediate_size` — [`L164`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L164)
  - `w13_reorder_size` — [`L161`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L161)
- used by: [`process_moe_weights`](moe_weights.md#process_moe_weights), [`process_w13_for_gmm`](moe_weights.md#process_w13_for_gmm), [`get_w13_padding_config`](moe_weights.md#get_w13_padding_config), [`_pad_tensor`](moe_weights.md#process_w13_for_gmm._pad_tensor)

## Functions
- `_get_expert_shard_axis(mesh: Mesh)` — [`L648`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L648)
- `_get_moe_weight_shardings(weights: FusedMoEWeights, moe_backend: MoEBackend, mesh: Mesh)` — [`L552`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L552) — Build sharding specs for MoE weights based on the backend type.
- `_pad_tensor(t)` — [`L235`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L235)
- `_process_moe_weights_no_requant(weights: FusedMoEWeights, moe_backend: MoEBackend, mesh: Mesh, w13_reorder_size: int, w13_interleave: bool, disable_weight_requantization: bool, weight_block_size: tuple[int, ...] | None)` — [`L753`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L753) — Process MoE weights without requantization.
- `_process_quantized_moe_weights_impl(weights: FusedMoEWeights, moe_backend: MoEBackend, mesh: Mesh, activation: str | MoEActivation, weight_block_size: tuple[int, ...] | None = None, desired_quant_dtype: jnp.dtype | None = None, requant_block_size: int | None = None, disable_weight_requantization: bool = False)` — [`L1059`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L1059)
- `_requant_and_process_local_fn(w13_w, w13_s, w13_b, w2_w, w2_s, w2_b, *, scan_batch_size: int, has_w13_scale: bool, has_w2_scale: bool, has_w13_bias: bool, has_w2_bias: bool, weight_block_size: tuple[int, ...] | None, orig_intermediate_size: int, w13_interleave: bool, inter_pad: int, hidden_pad: int, desired_quant_dtype: jnp.dtype, w13_block_size: int, w2_block_size: int, moe_backend: MoEBackend, w13_reorder_size: int)` — [`L931`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L931) — Per-device requantization and processing of MoE weights.
- `_requant_expert_batch_fn(carry, batch_inputs, *, has_w13_scale: bool, has_w2_scale: bool, weight_block_size: tuple[int, ...] | None, orig_intermediate_size: int, w13_interleave: bool, inter_pad: int, hidden_pad: int, desired_quant_dtype: jnp.dtype, w13_block_size: int, w2_block_size: int)` — [`L829`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L829) — Requantize a batch of experts.
- `get_w13_padding_config(intermediate_size: int, reorder_size: int, align: int = 128, outer_block_size: int = 1)` — [`L167`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L167) — Calculates padded dimensions and pad amounts for w13 tensors.
- `process_moe_weights(weights: FusedMoEWeights, moe_backend: MoEBackend, w13_reorder_size: int | None = None, w13_interleave: bool = False, disable_weight_requantization: bool = False)` — [`L269`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L269) — Process fused moe weights to a layout that moe backend expects.
- `process_quantized_moe_weights(weights: FusedMoEWeights, moe_backend: MoEBackend, mesh: Mesh, activation: str | MoEActivation, weight_block_size: tuple[int, ...] | None = None, desired_quant_dtype: jnp.dtype | None = None, requant_block_size: int | None = None, source_mesh: Mesh | None = None)` — [`L697`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L697) — Process quantized MoE weights for inference.
- `process_unquantized_moe_weights(*, mesh: Mesh, moe_backend: MoEBackend, activation: MoEActivation, w13_weight: jax.Array, w13_bias: jax.Array | None, w2_weight: jax.Array, w2_bias: jax.Array | None)` — [`L1204`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L1204) — Jit'ed version to process unquantized moe weights. See `process_moe_weights` for details.
- `process_w13_for_gmm(tensor, concat_dim: int, config: W13PaddingConfig, padded_output_sizes: list[int] | None = None, name: str = "w13")` — [`L202`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L202) — Splits, pads, concatenates, and optionally reorders W13 tensors for GMM backends.
- `quantize_moe_weights(weights: FusedMoEWeights, dtype: jnp.dtype, block_size: int | None, w13_interleave: bool = False)` — [`L68`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L68) — Quantize fused moe weights into a given dtype and block size.
- `shard_moe_weights(weights: FusedMoEWeights, moe_backend: MoEBackend, mesh: Mesh)` — [`L610`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L610)
- `shard_moe_weights_to_tpu(weights: FusedMoEWeights, mesh: Mesh, source_mesh: Mesh | None = None)` — [`L660`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L660) — Shard MoE weights onto TPU before requantization.

## Module values
- `P` — [`L34`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L34)
- `logger` — [`L36`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/moe_weights.py#L36)

