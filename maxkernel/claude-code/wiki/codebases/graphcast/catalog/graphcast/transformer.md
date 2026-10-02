---
title: 'Module: graphcast/transformer.py'
type: catalog
provenance: extracted
module: graphcast/transformer.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.transformer`/
symbols:
  _get_adj_matrix_for_edge_set: _get_adj_matrix_for_edge_set().
  MeshTransformer._maybe_init_batch_first_transformer: MeshTransformer#_maybe_init_batch_first_transformer().
  MeshTransformer.__call__: MeshTransformer#__call__().
  MeshTransformer._batch_first_transformer: MeshTransformer#_batch_first_transformer.
  MeshTransformer.__init__: MeshTransformer#__init__().
  Kwargs: Kwargs.
  MeshTransformer: MeshTransformer#
  MeshTransformer._transformer_ctor: MeshTransformer#_transformer_ctor.
  MeshTransformer._transformer_kwargs: MeshTransformer#_transformer_kwargs.
---
# Module: [`graphcast/transformer.py`](../../../../../raw/code/graphcast/graphcast/transformer.py)

## Classes
### `MeshTransformer`
- def: [`graphcast/transformer.py:60`](../../../../../raw/code/graphcast/graphcast/transformer.py#L60)
- doc: A Transformer for inputs with ordering \[nodes, batch, ...\].
- signature: `class MeshTransformer(hk.Module):`
- members:
  - `__call__(self, x: typed_graph.TypedGraph, global_norm_conditioning: jax.Array)` — [`L94`](../../../../../raw/code/graphcast/graphcast/transformer.py#L94) — Applies the model to the input graph and returns graph of same shape.
  - `__init__(self, transformer_ctor, transformer_kwargs: Kwargs, name: Optional[str] = None)` — [`L63`](../../../../../raw/code/graphcast/graphcast/transformer.py#L63) — Initialises the Transformer model.
- protocol/private: `_batch_first_transformer`[`L77`](../../../../../raw/code/graphcast/graphcast/transformer.py#L77), `_maybe_init_batch_first_transformer`[`L82`](../../../../../raw/code/graphcast/graphcast/transformer.py#L82), `_transformer_ctor`[`L78`](../../../../../raw/code/graphcast/graphcast/transformer.py#L78), `_transformer_kwargs`[`L79`](../../../../../raw/code/graphcast/graphcast/transformer.py#L79)
- uses (calls/refs, reference-scoped): [`nodes`](typed_graph.md#TypedGraph.nodes), [`TypedGraph`](typed_graph.md#TypedGraph), [`features`](typed_graph.md#NodeSet.features), [`_get_adj_matrix_for_edge_set`](transformer.md#_get_adj_matrix_for_edge_set), [`Kwargs`](transformer.md#Kwargs)
- used by: [`_mesh_gnn`](denoiser.md#_DenoiserArchitecture._mesh_gnn)

## Functions
- `_get_adj_matrix_for_edge_set(graph: typed_graph.TypedGraph, edge_set_name: str, add_self_edges: bool)` — [`L34`](../../../../../raw/code/graphcast/graphcast/transformer.py#L34) — Returns the adjacency matrix for the given graph and edge set.

## Module values
- `Kwargs` — [`L31`](../../../../../raw/code/graphcast/graphcast/transformer.py#L31)

