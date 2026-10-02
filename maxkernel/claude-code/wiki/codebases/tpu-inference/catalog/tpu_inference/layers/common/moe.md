---
title: 'Module: tpu_inference/layers/common/moe.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/common/moe.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.common.moe`/
symbols:
  moe_apply: moe_apply().
  MoEBackend: MoEBackend#
  MoEBackend.FUSED_MOE: MoEBackend#FUSED_MOE.
  MoEBackend.fused_moe_backends: MoEBackend#fused_moe_backends().
  MoEBackend.GMM_TP: MoEBackend#GMM_TP.
  MoEBackend.GMM_EP: MoEBackend#GMM_EP.
  MoEBackend.DENSE_MAT: MoEBackend#DENSE_MAT.
  MoEBackend.MEGABLX_GMM: MoEBackend#MEGABLX_GMM.
  logger: logger.
---
# Module: [`tpu_inference/layers/common/moe.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/moe.py)

## Classes
### `MoEBackend`  ·  implements/extends Enum
- def: [`tpu_inference/layers/common/moe.py:42`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/moe.py#L42) — documented in [tpu_inference-layers-jax-moe](../../../../concepts/tpu_inference-layers-jax-moe.md)
- signature: `class MoEBackend(Enum):`
- members:
  - `fused_moe_backends(cls)` — [`L69`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/moe.py#L69) — Returns those backends that use fused weights — documented in [tpu_inference-layers-jax-moe](../../../../concepts/tpu_inference-layers-jax-moe.md)
  - `DENSE_MAT` — [`L64`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/moe.py#L64)
  - `FUSED_MOE` — [`L48`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/moe.py#L48) — documented in [tpu_inference-layers-jax-moe](../../../../concepts/tpu_inference-layers-jax-moe.md)
  - `GMM_EP` — [`L55`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/moe.py#L55)
  - `GMM_TP` — [`L58`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/moe.py#L58) — documented in [tpu_inference-layers-jax-moe](../../../../concepts/tpu_inference-layers-jax-moe.md)
  - `MEGABLX_GMM` — [`L66`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/moe.py#L66)
- used by: [`moe_apply`](moe.md#moe_apply), [`apply_jax`](../jax/quantization/unquantized.md#UnquantizedFusedMoEMethod.apply_jax), [`shared_experts`](../../models/jax/deepseek_v3.md#DeepseekV2Moe.shared_experts), [`apply_jax`](../jax/quantization/fp8.md#Fp8FusedMoEMethod.apply_jax), [`process_moe_weights`](process_weights/moe_weights.md#process_moe_weights), [`shard_moe_weights`](process_weights/moe_weights.md#shard_moe_weights), [`process_unquantized_moe_weights`](process_weights/moe_weights.md#process_unquantized_moe_weights), [`_process_quantized_moe_weights_impl`](process_weights/moe_weights.md#_process_quantized_moe_weights_impl), [`extra_backend_kwargs`](../jax/quantization/unquantized.md#UnquantizedFusedMoEMethod.extra_backend_kwargs), [`_get_moe_weight_shardings`](process_weights/moe_weights.md#_get_moe_weight_shardings), [`_process_moe_weights_no_requant`](process_weights/moe_weights.md#_process_moe_weights_no_requant), [`select_moe_backend`](../jax/moe/utils.md#select_moe_backend), [`kernel_down_proj_EFD`](../jax/moe/moe.md#JaxMoE.kernel_down_proj_EFD), [`select_moe_backend_from_fused_moe_config`](../vllm/interface/moe.md#select_moe_backend_from_fused_moe_config), [`__call__`](../jax/moe/moe.md#Router.__call__), [`process_quantized_moe_weights`](process_weights/moe_weights.md#process_quantized_moe_weights), [`_requant_and_process_local_fn`](process_weights/moe_weights.md#_requant_and_process_local_fn), [`__call__`](../../models/jax/deepseek_v3.md#DeepSeekV3Router.__call__), [`__call__`](../jax/moe/moe.md#JaxMoE.__call__), [`moe_backend`](../jax/moe/moe.md#JaxMoE.moe_backend), [`process_weights_after_loading`](../jax/quantization/unquantized.md#UnquantizedFusedMoEMethod.process_weights_after_loading), [`moe_backend`](../../models/jax/deepseek_v3.md#DeepSeekV3.moe_backend), [`gmm_fn`](../jax/moe/utils.md#gmm_fn), [`FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS`](../jax/quantization/fp8.md#FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS), [`extra_backend_kwargs`](../vllm/quantization/awq.md#VllmAWQMoEMethod.extra_backend_kwargs), [`extra_backend_kwargs`](../vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.md#VllmCompressedTensorsW4A8MoEMethod.extra_backend_kwargs), [`extra_backend_kwargs`](../vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.md#VllmCompressedTensorsW8A8Fp8MoEMethod.extra_backend_kwargs), [`extra_backend_kwargs`](../vllm/quantization/fp8.md#VllmFp8MoEMethod.extra_backend_kwargs), [`extra_backend_kwargs`](../vllm/quantization/mxfp4.md#VllmMxfp4MoEMethod.extra_backend_kwargs), [`extra_backend_kwargs`](../vllm/quantization/nvfp4.md#VllmNvfp4MoEMethod.extra_backend_kwargs), [`extra_backend_kwargs`](../vllm/quantization/unquantized.md#VllmUnquantizedFusedMoEMethod.extra_backend_kwargs), [`__init__`](../../models/jax/deepseek_v3.md#DeepSeekV3Router.__init__), [`moe_backend`](../jax/moe/moe.md#Router.moe_backend)

## Functions
- `moe_apply(layer: Union[RoutedExperts, JaxRoutedExperts, JaxMoE], x: jax.Array, gating_output: Union[jax.Array, Tuple[jax.Array, jax.Array]], weights: Union[FusedMoEWeights, UnfusedMoEWeights], moe_backend: MoEBackend, mesh: Mesh, extra_backend_kwargs: dict)` — [`L74`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/moe.py#L74) — documented in [tpu_inference-layers-jax-moe](../../../../concepts/tpu_inference-layers-jax-moe.md)

## Module values
- `logger` — [`L39`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/moe.py#L39)

