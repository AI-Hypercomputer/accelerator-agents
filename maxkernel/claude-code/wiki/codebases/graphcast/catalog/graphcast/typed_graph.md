---
title: 'Module: graphcast/typed_graph.py'
type: catalog
provenance: extracted
module: graphcast/typed_graph.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.typed_graph`/
symbols:
  TypedGraph.nodes: TypedGraph#nodes.
  TypedGraph: TypedGraph#
  NodeSet.features: NodeSet#features.
  TypedGraph.edges: TypedGraph#edges.
  TypedGraph.edge_key_by_name: TypedGraph#edge_key_by_name().
  EdgeSet.features: EdgeSet#features.
  NodeSet.n_node: NodeSet#n_node.
  TypedGraph.context: TypedGraph#context.
  NodeSet: NodeSet#
  EdgesIndices.senders: EdgesIndices#senders.
  Context.features: Context#features.
  EdgesIndices: EdgesIndices#
  EdgesIndices.receivers: EdgesIndices#receivers.
  EdgeSet.indices: EdgeSet#indices.
  EdgeSet.n_edge: EdgeSet#n_edge.
  Context.n_graph: Context#n_graph.
  EdgeSet: EdgeSet#
  EdgeSetKey: EdgeSetKey#
  TypedGraph.edge_by_name: TypedGraph#edge_by_name().
  Context: Context#
  ArrayLike: ArrayLike.
  ArrayLikeTree: ArrayLikeTree.
  EdgeSetKey.name: EdgeSetKey#name.
  EdgeSetKey.node_sets: EdgeSetKey#node_sets.
  _T: _T.
---
# Module: [`graphcast/typed_graph.py`](../../../../../raw/code/graphcast/graphcast/typed_graph.py)

## Classes
### `Context`  ·  implements/extends NamedTuple
- def: [`graphcast/typed_graph.py:64`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L64)
- signature: `class Context(NamedTuple):`
- members:
  - `features` — [`L68`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L68)
  - `n_graph` — [`L67`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L67)
- uses (calls/refs, reference-scoped): [`ArrayLike`](typed_graph.md#ArrayLike), [`ArrayLikeTree`](typed_graph.md#ArrayLikeTree)
- used by: [`_init_grid2mesh_graph`](denoiser.md#_DenoiserArchitecture._init_grid2mesh_graph), [`_init_grid2mesh_graph`](graphcast.md#GraphCast._init_grid2mesh_graph), [`_init_mesh2grid_graph`](graphcast.md#GraphCast._init_mesh2grid_graph), [`_init_mesh2grid_graph`](denoiser.md#_DenoiserArchitecture._init_mesh2grid_graph), [`_init_mesh_graph`](graphcast.md#GraphCast._init_mesh_graph), [`_init_mesh_graph`](denoiser.md#_DenoiserArchitecture._init_mesh_graph), [`_embed`](typed_graph_net.md#GraphMapFeatures._embed), [`_embed`](deep_typed_graph_net.md#DeepTypedGraphNet._embed), [`context`](typed_graph.md#TypedGraph.context)

### `EdgeSet`  ·  implements/extends NamedTuple
- def: [`graphcast/typed_graph.py:57`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L57) — documented in [graphcast-typed_graph](../../concepts/graphcast-typed_graph.md)
- doc: Represents a set of edges.
- signature: `class EdgeSet(NamedTuple):`
- members:
  - `features` — [`L61`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L61) — documented in [graphcast](../../concepts/graphcast.md)
  - `indices` — [`L60`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L60)
  - `n_edge` — [`L59`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L59)
- uses (calls/refs, reference-scoped): [`EdgesIndices`](typed_graph.md#EdgesIndices), [`ArrayLike`](typed_graph.md#ArrayLike), [`ArrayLikeTree`](typed_graph.md#ArrayLikeTree)
- used by: [`_init_grid2mesh_graph`](denoiser.md#_DenoiserArchitecture._init_grid2mesh_graph), [`_init_grid2mesh_graph`](graphcast.md#GraphCast._init_grid2mesh_graph), [`_init_mesh2grid_graph`](graphcast.md#GraphCast._init_mesh2grid_graph), [`_init_mesh2grid_graph`](denoiser.md#_DenoiserArchitecture._init_mesh2grid_graph), [`_init_mesh_graph`](graphcast.md#GraphCast._init_mesh_graph), [`_init_mesh_graph`](denoiser.md#_DenoiserArchitecture._init_mesh_graph), [`edges`](typed_graph.md#TypedGraph.edges), [`_run_grid2mesh_gnn`](denoiser.md#_DenoiserArchitecture._run_grid2mesh_gnn), [`_run_grid2mesh_gnn`](graphcast.md#GraphCast._run_grid2mesh_gnn), [`_embed`](typed_graph_net.md#GraphMapFeatures._embed), [`_run_mesh2grid_gnn`](denoiser.md#_DenoiserArchitecture._run_mesh2grid_gnn), [`_run_mesh2grid_gnn`](graphcast.md#GraphCast._run_mesh2grid_gnn), [`_run_mesh_gnn`](denoiser.md#_DenoiserArchitecture._run_mesh_gnn), [`_run_mesh_gnn`](graphcast.md#GraphCast._run_mesh_gnn), [`_get_adj_matrix_for_edge_set`](transformer.md#_get_adj_matrix_for_edge_set), [`_process_step`](deep_typed_graph_net.md#DeepTypedGraphNet._process_step), [`edge_by_name`](typed_graph.md#TypedGraph.edge_by_name)

### `EdgeSetKey`  ·  implements/extends NamedTuple
- def: [`graphcast/typed_graph.py:71`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L71)
- signature: `class EdgeSetKey(NamedTuple):`
- members:
  - `name` — [`L72`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L72)
  - `node_sets` — [`L75`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L75)
- used by: [`_init_grid2mesh_graph`](denoiser.md#_DenoiserArchitecture._init_grid2mesh_graph), [`_init_grid2mesh_graph`](graphcast.md#GraphCast._init_grid2mesh_graph), [`_init_mesh2grid_graph`](graphcast.md#GraphCast._init_mesh2grid_graph), [`_init_mesh2grid_graph`](denoiser.md#_DenoiserArchitecture._init_mesh2grid_graph), [`_init_mesh_graph`](graphcast.md#GraphCast._init_mesh_graph), [`_init_mesh_graph`](denoiser.md#_DenoiserArchitecture._init_mesh_graph), [`edges`](typed_graph.md#TypedGraph.edges), [`_get_adj_matrix_for_edge_set`](transformer.md#_get_adj_matrix_for_edge_set), [`edge_key_by_name`](typed_graph.md#TypedGraph.edge_key_by_name)

### `EdgesIndices`  ·  implements/extends NamedTuple
- def: [`graphcast/typed_graph.py:51`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L51)
- doc: Represents indices to nodes adjacent to the edges.
- signature: `class EdgesIndices(NamedTuple):`
- members:
  - `receivers` — [`L54`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L54)
  - `senders` — [`L53`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L53)
- uses (calls/refs, reference-scoped): [`ArrayLike`](typed_graph.md#ArrayLike)
- used by: [`_init_grid2mesh_graph`](denoiser.md#_DenoiserArchitecture._init_grid2mesh_graph), [`_init_grid2mesh_graph`](graphcast.md#GraphCast._init_grid2mesh_graph), [`_init_mesh2grid_graph`](graphcast.md#GraphCast._init_mesh2grid_graph), [`_init_mesh2grid_graph`](denoiser.md#_DenoiserArchitecture._init_mesh2grid_graph), [`_init_mesh_graph`](graphcast.md#GraphCast._init_mesh_graph), [`_init_mesh_graph`](denoiser.md#_DenoiserArchitecture._init_mesh_graph), [`indices`](typed_graph.md#EdgeSet.indices), [`_node_update`](typed_graph_net.md#_node_update), [`_global_update`](typed_graph_net.md#_global_update)

### `NodeSet`  ·  implements/extends NamedTuple
- def: [`graphcast/typed_graph.py:45`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L45) — documented in [graphcast-typed_graph](../../concepts/graphcast-typed_graph.md)
- doc: Represents a set of nodes.
- signature: `class NodeSet(NamedTuple):`
- members:
  - `features` — [`L48`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L48) — documented in [graphcast-typed_graph](../../concepts/graphcast-typed_graph.md)
  - `n_node` — [`L47`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L47) — documented in [graphcast](../../concepts/graphcast.md)
- uses (calls/refs, reference-scoped): [`ArrayLike`](typed_graph.md#ArrayLike), [`ArrayLikeTree`](typed_graph.md#ArrayLikeTree)
- used by: [`_init_grid2mesh_graph`](denoiser.md#_DenoiserArchitecture._init_grid2mesh_graph), [`_init_grid2mesh_graph`](graphcast.md#GraphCast._init_grid2mesh_graph), [`_init_mesh2grid_graph`](graphcast.md#GraphCast._init_mesh2grid_graph), [`_init_mesh2grid_graph`](denoiser.md#_DenoiserArchitecture._init_mesh2grid_graph), [`_init_mesh_graph`](graphcast.md#GraphCast._init_mesh_graph), [`_init_mesh_graph`](denoiser.md#_DenoiserArchitecture._init_mesh_graph), [`nodes`](typed_graph.md#TypedGraph.nodes), [`_run_grid2mesh_gnn`](denoiser.md#_DenoiserArchitecture._run_grid2mesh_gnn), [`_run_grid2mesh_gnn`](graphcast.md#GraphCast._run_grid2mesh_gnn), [`_embed`](typed_graph_net.md#GraphMapFeatures._embed), [`_run_mesh2grid_gnn`](denoiser.md#_DenoiserArchitecture._run_mesh2grid_gnn), [`_run_mesh2grid_gnn`](graphcast.md#GraphCast._run_mesh2grid_gnn), [`_run_mesh_gnn`](denoiser.md#_DenoiserArchitecture._run_mesh_gnn), [`_run_mesh_gnn`](graphcast.md#GraphCast._run_mesh_gnn), [`_embed`](deep_typed_graph_net.md#DeepTypedGraphNet._embed), [`_get_adj_matrix_for_edge_set`](transformer.md#_get_adj_matrix_for_edge_set), [`_process_step`](deep_typed_graph_net.md#DeepTypedGraphNet._process_step), [`__call__`](transformer.md#MeshTransformer.__call__)

### `TypedGraph`  ·  implements/extends NamedTuple
- def: [`graphcast/typed_graph.py:78`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L78) — documented in [graphcast-typed_graph](../../concepts/graphcast-typed_graph.md)
- doc: A graph with typed nodes and edges.
- signature: `class TypedGraph(NamedTuple):`
- members:
  - `edge_by_name(self, name: str)` — [`L96`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L96)
  - `edge_key_by_name(self, name: str)` — [`L89`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L89) — documented in [graphcast-typed_graph](../../concepts/graphcast-typed_graph.md)
  - `context` — [`L85`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L85)
  - `edges` — [`L87`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L87) — documented in [graphcast-typed_graph](../../concepts/graphcast-typed_graph.md)
  - `nodes` — [`L86`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L86) — documented in [graphcast-typed_graph](../../concepts/graphcast-typed_graph.md)
- uses (calls/refs, reference-scoped): [`NodeSet`](typed_graph.md#NodeSet), [`EdgeSet`](typed_graph.md#EdgeSet), [`EdgeSetKey`](typed_graph.md#EdgeSetKey), [`Context`](typed_graph.md#Context), [`name`](typed_graph.md#EdgeSetKey.name)
- used by: [`_init_grid2mesh_graph`](denoiser.md#_DenoiserArchitecture._init_grid2mesh_graph), [`_init_grid2mesh_graph`](graphcast.md#GraphCast._init_grid2mesh_graph), [`_init_mesh2grid_graph`](graphcast.md#GraphCast._init_mesh2grid_graph), [`_init_mesh2grid_graph`](denoiser.md#_DenoiserArchitecture._init_mesh2grid_graph), [`_init_mesh_graph`](graphcast.md#GraphCast._init_mesh_graph), [`_init_mesh_graph`](denoiser.md#_DenoiserArchitecture._init_mesh_graph), [`_networks_builder`](deep_typed_graph_net.md#DeepTypedGraphNet._networks_builder), [`_run_grid2mesh_gnn`](denoiser.md#_DenoiserArchitecture._run_grid2mesh_gnn), [`_run_grid2mesh_gnn`](graphcast.md#GraphCast._run_grid2mesh_gnn), [`_embed`](typed_graph_net.md#GraphMapFeatures._embed), [`_run_mesh2grid_gnn`](denoiser.md#_DenoiserArchitecture._run_mesh2grid_gnn), [`_run_mesh2grid_gnn`](graphcast.md#GraphCast._run_mesh2grid_gnn), [`_run_mesh_gnn`](denoiser.md#_DenoiserArchitecture._run_mesh_gnn), [`_run_mesh_gnn`](graphcast.md#GraphCast._run_mesh_gnn), [`_apply_graph_net`](typed_graph_net.md#GraphNetwork._apply_graph_net), [`_embed`](deep_typed_graph_net.md#DeepTypedGraphNet._embed), [`_get_adj_matrix_for_edge_set`](transformer.md#_get_adj_matrix_for_edge_set), [`_maybe_init_batch_first_transformer`](transformer.md#MeshTransformer._maybe_init_batch_first_transformer), [`_process_step`](deep_typed_graph_net.md#DeepTypedGraphNet._process_step), [`__call__`](deep_typed_graph_net.md#DeepTypedGraphNet.__call__), [`__call__`](transformer.md#MeshTransformer.__call__), [`_process`](deep_typed_graph_net.md#DeepTypedGraphNet._process), [`GraphToGraphNetwork`](deep_typed_graph_net.md#GraphToGraphNetwork), [`_output`](deep_typed_graph_net.md#DeepTypedGraphNet._output)

## Module values
- `ArrayLike` — [`L18`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L18)
- `ArrayLikeTree` — [`L19`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L19) — documented in [graphcast-typed_graph](../../concepts/graphcast-typed_graph.md)
- `_T` — [`L21`](../../../../../raw/code/graphcast/graphcast/typed_graph.py#L21)

