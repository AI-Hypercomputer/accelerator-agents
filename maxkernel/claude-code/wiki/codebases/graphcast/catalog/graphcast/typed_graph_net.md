---
title: 'Module: graphcast/typed_graph_net.py'
type: catalog
provenance: extracted
module: graphcast/typed_graph_net.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.typed_graph_net`/
symbols:
  GraphMapFeatures._embed: GraphMapFeatures()._embed().
  GraphNetwork._apply_graph_net: GraphNetwork()._apply_graph_net().
  GNUpdateNodeFn: GNUpdateNodeFn.
  GNUpdateGlobalFn: GNUpdateGlobalFn.
  GraphNetwork: GraphNetwork().
  _node_update: _node_update().
  InteractionNetwork: InteractionNetwork().
  _global_update: _global_update().
  InteractionUpdateNodeFn: InteractionUpdateNodeFn.
  GraphMapFeatures: GraphMapFeatures().
  InteractionUpdateNodeFnNoSentEdges: InteractionUpdateNodeFnNoSentEdges.
  Globals: Globals.
  ReceiverFeatures: ReceiverFeatures.
  NodeFeatures: NodeFeatures.
  SenderFeatures: SenderFeatures.
  EdgeFeatures: EdgeFeatures.
  _edge_update: _edge_update().
---
# Module: [`graphcast/typed_graph_net.py`](../../../../../raw/code/graphcast/graphcast/typed_graph_net.py)

## Functions
- `GraphMapFeatures(embed_edge_fn: Optional[Mapping[str, jraph.EmbedEdgeFn]] = None, embed_node_fn: Optional[Mapping[str, jraph.EmbedNodeFn]] = None, embed_global_fn: Optional[jraph.EmbedGlobalFn] = None)` — [`L278`](../../../../../raw/code/graphcast/graphcast/typed_graph_net.py#L278) — Returns function which embeds the components of a graph independently.
- `GraphNetwork(update_edge_fn: Mapping[str, jraph.GNUpdateEdgeFn], update_node_fn: Mapping[str, GNUpdateNodeFn], update_global_fn: Optional[GNUpdateGlobalFn] = None, aggregate_edges_for_nodes_fn: jraph.AggregateEdgesToNodesFn = jraph.segment_sum, aggregate_nodes_for_globals_fn: jraph.AggregateNodesToGlobalsFn = jraph.segment_sum, aggregate_edges_for_globals_fn: jraph.AggregateEdgesToGlobalsFn = jraph.segment_sum)` — [`L41`](../../../../../raw/code/graphcast/graphcast/typed_graph_net.py#L41) — Returns a method that applies a configured GraphNetwork.
- `InteractionNetwork(update_edge_fn: Mapping[str, jraph.InteractionUpdateEdgeFn], update_node_fn: Mapping[str, Union[InteractionUpdateNodeFn, InteractionUpdateNodeFnNoSentEdges]], aggregate_edges_for_nodes_fn: jraph.AggregateEdgesToNodesFn = jraph.segment_sum, include_sent_messages_in_node_update: bool = False)` — [`L228`](../../../../../raw/code/graphcast/graphcast/typed_graph_net.py#L228) — Returns a method that applies a configured InteractionNetwork.
- `_apply_graph_net(graph: typed_graph.TypedGraph)` — [`L85`](../../../../../raw/code/graphcast/graphcast/typed_graph_net.py#L85) — Applies a configured GraphNetwork to a graph.
- `_edge_update(graph, edge_fn, edge_set_key)` — [`L130`](../../../../../raw/code/graphcast/graphcast/typed_graph_net.py#L130) — Updates an edge set of a given key.
- `_embed(graph: typed_graph.TypedGraph)` — [`L292`](../../../../../raw/code/graphcast/graphcast/typed_graph_net.py#L292)
- `_global_update(graph, global_fn, edge_aggregation_fn, node_aggregation_fn)` — [`L187`](../../../../../raw/code/graphcast/graphcast/typed_graph_net.py#L187) — Updates an edge set of a given key.
- `_node_update(graph, node_fn, node_set_key, aggregation_fn)` — [`L155`](../../../../../raw/code/graphcast/graphcast/typed_graph_net.py#L155) — Updates an edge set of a given key.

## Module values
- `EdgeFeatures` — [`L25`](../../../../../raw/code/graphcast/graphcast/typed_graph_net.py#L25)
- `GNUpdateGlobalFn` — [`L36`](../../../../../raw/code/graphcast/graphcast/typed_graph_net.py#L36)
- `GNUpdateNodeFn` — [`L31`](../../../../../raw/code/graphcast/graphcast/typed_graph_net.py#L31)
- `Globals` — [`L25`](../../../../../raw/code/graphcast/graphcast/typed_graph_net.py#L25)
- `InteractionUpdateNodeFn` — [`L215`](../../../../../raw/code/graphcast/graphcast/typed_graph_net.py#L215)
- `InteractionUpdateNodeFnNoSentEdges` — [`L222`](../../../../../raw/code/graphcast/graphcast/typed_graph_net.py#L222)
- `NodeFeatures` — [`L25`](../../../../../raw/code/graphcast/graphcast/typed_graph_net.py#L25)
- `ReceiverFeatures` — [`L25`](../../../../../raw/code/graphcast/graphcast/typed_graph_net.py#L25)
- `SenderFeatures` — [`L25`](../../../../../raw/code/graphcast/graphcast/typed_graph_net.py#L25)

