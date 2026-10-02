---
title: 'Module: python/sgl_jax/srt/layers/gate.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/layers/gate.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.layers.gate`/
symbols:
  TopK.__call__: TopK#__call__().
  TopK.num_expert_group: TopK#num_expert_group.
  GateLogit.bias: GateLogit#bias.
  TopK._grouped_topk: TopK#_grouped_topk().
  TopK._biased_grouped_topk: TopK#_biased_grouped_topk().
  GateLogit.__call__: GateLogit#__call__().
  GateLogit.score_func: GateLogit#score_func.
  TopK.topk: TopK#topk.
  TopK.layer_id: TopK#layer_id.
  TopK._topk: TopK#_topk().
  TopK._biased_topk: TopK#_biased_topk().
  TopK.topk_group: TopK#topk_group.
  TopK.routed_scaling_factor: TopK#routed_scaling_factor.
  GateLogit: GateLogit#
  GateLogit.weight_dtype: GateLogit#weight_dtype.
  GateLogit.kernel: GateLogit#kernel.
  TopK: TopK#
  TopK.renormalize: TopK#renormalize.
  GateLogit.__init__: GateLogit#__init__().
  GateLogit.enable_expert_bias: GateLogit#enable_expert_bias.
  TopK.__init__: TopK#__init__().
---
# Module: [`python/sgl_jax/srt/layers/gate.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/gate.py)

## Classes
### `GateLogit`  ·  implements/extends Module
- def: [`python/sgl_jax/srt/layers/gate.py:15`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/gate.py#L15)
- signature: `class GateLogit(nnx.Module):`
- members:
  - `bias` — [`L37`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/gate.py#L37)
  - `enable_expert_bias` — [`L25`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/gate.py#L25)
  - `kernel` — [`L28`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/gate.py#L28)
  - `score_func` — [`L26`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/gate.py#L26)
  - `weight_dtype` — [`L24`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/gate.py#L24)
- protocol/private: `__call__`[`L49`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/gate.py#L49), `__init__`[`L16`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/gate.py#L16)
- uses (calls/refs, reference-scoped): [`named_scope`](../utils/profiling_utils.md#named_scope)

### `TopK`  ·  implements/extends Module
- def: [`python/sgl_jax/srt/layers/gate.py:65`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/gate.py#L65)
- signature: `class TopK(nnx.Module):`
- members:
  - `layer_id` — [`L80`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/gate.py#L80)
  - `num_expert_group` — [`L77`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/gate.py#L77)
  - `renormalize` — [`L76`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/gate.py#L76)
  - `routed_scaling_factor` — [`L79`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/gate.py#L79)
  - `topk` — [`L75`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/gate.py#L75)
  - `topk_group` — [`L78`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/gate.py#L78)
- protocol/private: `__call__`[`L83`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/gate.py#L83), `__init__`[`L66`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/gate.py#L66), `_biased_grouped_topk`[`L159`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/gate.py#L159), `_biased_topk`[`L117`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/gate.py#L117), `_grouped_topk`[`L126`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/gate.py#L126), `_topk`[`L114`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/gate.py#L114)
- uses (calls/refs, reference-scoped): [`named_scope`](../utils/profiling_utils.md#named_scope), [`ExpertLocationMetadata`](../eplb/expert_location.md#ExpertLocationMetadata), [`topk_ids_logical_to_physical`](../eplb/expert_location.md#topk_ids_logical_to_physical)

