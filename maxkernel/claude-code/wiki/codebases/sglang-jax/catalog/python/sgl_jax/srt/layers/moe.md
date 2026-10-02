---
title: 'Module: python/sgl_jax/srt/layers/moe.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/layers/moe.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.layers.moe`/
symbols:
  EPMoE.__call__: EPMoE#__call__().
  EPMoE.quantize_weights: EPMoE#quantize_weights().
  EPMoE._forward: EPMoE#_forward().
  EPMoE.weight_block_size: EPMoE#weight_block_size.
  create_moe_weights_mapping: create_moe_weights_mapping().
  EPMoE._gmm_compute: EPMoE#_gmm_compute().
  EPMoE.num_experts: EPMoE#num_experts.
  EPMoE.moe_mesh: EPMoE#moe_mesh.
  EPMoE.updated_mesh: EPMoE#updated_mesh.
  EPMoE.wi_0: EPMoE#wi_0.
  EPMoE.experts_per_device: EPMoE#experts_per_device.
  EPMoE.wo: EPMoE#wo.
  EPMoE.ep_size: EPMoE#ep_size.
  EPMoE.wi_1: EPMoE#wi_1.
  EPMoE.wi_0_scale: EPMoE#wi_0_scale.
  EPMoE.wi_1_scale: EPMoE#wi_1_scale.
  EPMoE.wo_scale: EPMoE#wo_scale.
  EPMoE._dispatch: EPMoE#_dispatch().
  EPMoE._permute: EPMoE#_permute().
  EPMoE._unpermute: EPMoE#_unpermute().
  EPMoE.tp_size: EPMoE#tp_size.
  EPMoE._normalize_scale_for_gmm: EPMoE#_normalize_scale_for_gmm().
  EPMoE.dtype: EPMoE#dtype.
  EPMoE.num_experts_per_tok: EPMoE#num_experts_per_tok.
  EPMoE.mesh: EPMoE#mesh.
  EPMoE.quantized_dtype: EPMoE#quantized_dtype.
  EPMoE.activation: EPMoE#activation.
  EPMoE.pre_gather_quant_dtype: EPMoE#pre_gather_quant_dtype.
  EPMoE: EPMoE#
  EPMoE._get_block_size_k: EPMoE#_get_block_size_k().
  EPMoE.activation_quantized_dtype: EPMoE#activation_quantized_dtype.
  EPMoE._combine: EPMoE#_combine().
  EPMoE.__init__: EPMoE#__init__().
  EPMoE.physical_to_logical_map: EPMoE#physical_to_logical_map.
  EPMoE.intermediate_dim: EPMoE#intermediate_dim.
  EPMoE.weight_dtype: EPMoE#weight_dtype.
  EPMoE.layer_id: EPMoE#layer_id.
  EPMoE.original_mesh: EPMoE#original_mesh.
  EPMoE.hidden_size: EPMoE#hidden_size.
  EPMoE._detect_device_capabilities: EPMoE#_detect_device_capabilities().
  EPMoE._get_all_to_all_params: EPMoE#_get_all_to_all_params().
---
# Module: [`python/sgl_jax/srt/layers/moe.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py)

## Classes
### `EPMoE`  ·  implements/extends Module
- def: [`python/sgl_jax/srt/layers/moe.py:27`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L27)
- signature: `class EPMoE(nnx.Module):`
- members:
  - `_get_block_size_k(*, hidden_size: int, intermediate_dim: int, weight_block_size: list[int] | tuple[int, int] | None)` — [`L258`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L258) — Extract the contracting-dimension block size for MoE weights.
  - `_normalize_scale_for_gmm(self, scale: jax.Array | None, weight: jax.Array, *, scale_name: str)` — [`L143`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L143) — Normalize offline/runtime scale tensors to GMM's 4D layout.
  - `quantize_weights(self, is_static: bool = False)` — [`L253`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L253) — Quantize MoE weights in-place or initialize params for static loading.
  - `activation` — [`L61`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L61)
  - `activation_quantized_dtype` — [`L68`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L68)
  - `dtype` — [`L56`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L56)
  - `ep_size` — [`L58`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L58)
  - `experts_per_device` — [`L81`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L81)
  - `hidden_size` — [`L62`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L62)
  - `intermediate_dim` — [`L54`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L54)
  - `layer_id` — [`L57`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L57)
  - `mesh` — [`L60`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L60)
  - `moe_mesh` — [`L84`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L84)
  - `num_experts` — [`L50`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L50)
  - `num_experts_per_tok` — [`L44`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L44)
  - `original_mesh` — [`L59`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L59)
  - `physical_to_logical_map` — [`L45`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L45)
  - `pre_gather_quant_dtype` — [`L46`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L46)
  - `quantized_dtype` — [`L65`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L65)
  - `tp_size` — [`L80`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L80)
  - `updated_mesh` — [`L91`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L91)
  - `weight_block_size` — [`L71`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L71)
  - `weight_dtype` — [`L55`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L55)
  - `wi_0` — [`L97`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L97)
  - `wi_0_scale` — [`L126`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L126)
  - `wi_1` — [`L106`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L106)
  - `wi_1_scale` — [`L127`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L127)
  - `wo` — [`L115`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L115)
  - `wo_scale` — [`L128`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L128)
- protocol/private: `__call__`[`L413`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L413), `__init__`[`L28`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L28), `_combine`[`L685`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L685), `_detect_device_capabilities`[`L130`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L130), `_dispatch`[`L661`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L661), `_forward`[`L500`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L500), `_get_all_to_all_params`[`L667`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L667), `_gmm_compute`[`L570`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L570), `_permute`[`L688`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L688), `_unpermute`[`L717`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L717)
- uses (calls/refs, reference-scoped): [`named_scope`](../utils/profiling_utils.md#named_scope), [`quantize_tensor`](../utils/quantization/quantization_utils.md#quantize_tensor), [`gmm`](../kernels/gmm/megablox_gmm_backend.md#gmm), [`get_global_expert_location_metadata`](../eplb/expert_location.md#get_global_expert_location_metadata), [`quantize_tensor_simple`](../utils/quantization/quantization_utils.md#quantize_tensor_simple)
- used by: [`apply_moe_quantization`](../utils/quantization/quantization_utils.md#apply_moe_quantization), [`_quantize_moe_recursive`](../utils/quantization/quantization_utils.md#apply_moe_quantization._quantize_moe_recursive)

## Functions
- `create_moe_weights_mapping(prefix: str, target_prefix: str, num_experts: int, expert_type_names: tuple[str, str, str] = ("gate_proj", "up_proj", "down_proj"), expert_concat_axis_map: dict[str, int] = None, moe_backend: str = "epmoe", moe_path: str = "mlp", source_expert_pattern: str = "experts.{i}", physical_to_logical_map=None)` — [`L758`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/moe.py#L758) — Generate a unified mapping dictionary for MoE layer expert weights.

