---
title: 'Module: graphcast/mlp.py'
type: catalog
provenance: extracted
module: graphcast/mlp.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.mlp`/LinearNormConditioning#
symbols:
  LinearNormConditioning: ''
  LinearNormConditioning.__init__: __init__().
  LinearNormConditioning.__call__: __call__().
---
# Module: [`graphcast/mlp.py`](../../../../../raw/code/graphcast/graphcast/mlp.py)

## Classes
### `LinearNormConditioning`
- def: [`graphcast/mlp.py:24`](../../../../../raw/code/graphcast/graphcast/mlp.py#L24)
- doc: Module for norm conditioning.
- signature: `class LinearNormConditioning(hk.Module):`
- protocol/private: `__call__`[`L35`](../../../../../raw/code/graphcast/graphcast/mlp.py#L35), `__init__`[`L32`](../../../../../raw/code/graphcast/graphcast/mlp.py#L32)
- used by: [`build_mlp_with_maybe_layer_norm`](deep_typed_graph_net.md#DeepTypedGraphNet.build_mlp_with_maybe_layer_norm), [`norm_conditioning_layer`](sparse_transformer.md#Block.norm_conditioning_layer), [`norm_conditioning_layer`](sparse_transformer.md#Transformer.norm_conditioning_layer)

