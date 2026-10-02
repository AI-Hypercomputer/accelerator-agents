---
title: 'Module: graphcast/deep_typed_graph_net.py'
type: catalog
provenance: extracted
module: graphcast/deep_typed_graph_net.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.deep_typed_graph_net`/
symbols:
  DeepTypedGraphNet._networks_builder: DeepTypedGraphNet#_networks_builder().
  DeepTypedGraphNet._embed: DeepTypedGraphNet#_embed().
  DeepTypedGraphNet.build_mlp_with_maybe_layer_norm: DeepTypedGraphNet#build_mlp_with_maybe_layer_norm().
  DeepTypedGraphNet._process_step: DeepTypedGraphNet#_process_step().
  DeepTypedGraphNet.__call__: DeepTypedGraphNet#__call__().
  DeepTypedGraphNet._process: DeepTypedGraphNet#_process().
  DeepTypedGraphNet.build_mlp: DeepTypedGraphNet#build_mlp().
  GraphToGraphNetwork: GraphToGraphNetwork.
  DeepTypedGraphNet.aggregate_fn: DeepTypedGraphNet#aggregate_fn().
  DeepTypedGraphNet._output: DeepTypedGraphNet#_output().
  DeepTypedGraphNet: DeepTypedGraphNet#
  DeepTypedGraphNet._aggregate_edges_for_nodes_fn: DeepTypedGraphNet#_aggregate_edges_for_nodes_fn.
  DeepTypedGraphNet._aggregate_normalization: DeepTypedGraphNet#_aggregate_normalization.
  DeepTypedGraphNet._activation: DeepTypedGraphNet#_activation.
  _build_update_fns_for_node_types: _build_update_fns_for_node_types().
  _build_update_fns_for_edge_types: _build_update_fns_for_edge_types().
  DeepTypedGraphNet._node_latent_size: DeepTypedGraphNet#_node_latent_size.
  DeepTypedGraphNet._edge_latent_size: DeepTypedGraphNet#_edge_latent_size.
  DeepTypedGraphNet._node_output_size: DeepTypedGraphNet#_node_output_size.
  DeepTypedGraphNet._edge_output_size: DeepTypedGraphNet#_edge_output_size.
  DeepTypedGraphNet._use_norm_conditioning: DeepTypedGraphNet#_use_norm_conditioning.
  DeepTypedGraphNet._mlp_hidden_size: DeepTypedGraphNet#_mlp_hidden_size.
  DeepTypedGraphNet._mlp_num_hidden_layers: DeepTypedGraphNet#_mlp_num_hidden_layers.
  DeepTypedGraphNet._num_message_passing_steps: DeepTypedGraphNet#_num_message_passing_steps.
  DeepTypedGraphNet._num_processor_repetitions: DeepTypedGraphNet#_num_processor_repetitions.
  DeepTypedGraphNet._embed_nodes: DeepTypedGraphNet#_embed_nodes.
  DeepTypedGraphNet._embed_edges: DeepTypedGraphNet#_embed_edges.
  DeepTypedGraphNet._include_sent_messages_in_node_update: DeepTypedGraphNet#_include_sent_messages_in_node_update.
  DeepTypedGraphNet._use_layer_norm: DeepTypedGraphNet#_use_layer_norm.
  DeepTypedGraphNet._f32_aggregation: DeepTypedGraphNet#_f32_aggregation.
  _get_activation_fn: _get_activation_fn().
  _get_aggregate_edges_for_nodes_fn: _get_aggregate_edges_for_nodes_fn().
  DeepTypedGraphNet.__init__: DeepTypedGraphNet#__init__().
---
# Module: [`graphcast/deep_typed_graph_net.py`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py)

## Classes
### `DeepTypedGraphNet`
- def: [`graphcast/deep_typed_graph_net.py:53`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L53)
- doc: Deep Graph Neural Network.
- signature: `class DeepTypedGraphNet(hk.Module):`
- members:
  - `__call__(self, input_graph: typed_graph.TypedGraph, global_norm_conditioning: Optional[chex.Array] = None)` — [`L180`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L180) — Forward pass of the learnable dynamics model. — documented in [graphcast-deep_typed_graph_net](../../concepts/graphcast-deep_typed_graph_net.md)
  - `__init__(self, *, node_latent_size: Mapping[str, int], edge_latent_size: Mapping[str, int], mlp_hidden_size: int, mlp_num_hidden_layers: int, num_message_passing_steps: int, num_processor_repetitions: int = 1, embed_nodes: bool = True, embed_edges: bool = True, node_output_size: Optional[Mapping[str, int]] = None, edge_output_size: Optional[Mapping[str, int]] = None, include_sent_messages_in_node_update: bool = False, use_layer_norm: bool = True, use_norm_conditioning: bool = False, activation: str = "relu", f32_aggregation: bool = False, aggregate_edges_for_nodes_fn: str = "segment_sum", aggregate_normalization: Optional[float] = None, name: str = "DeepTypedGraphNet")` — [`L82`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L82) — Inits the model.
  - `_embed(self, input_graph: typed_graph.TypedGraph, embedder_network: GraphToGraphNetwork)` — [`L326`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L326) — Embeds the input graph features into a latent graph. — documented in [graphcast-deep_typed_graph_net](../../concepts/graphcast-deep_typed_graph_net.md)
  - `_output(self, latent_graph: typed_graph.TypedGraph, output_network: GraphToGraphNetwork)` — [`L396`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L396) — Produces the output from the latent graph.
  - `_process(self, latent_graph_0: typed_graph.TypedGraph, processor_networks: List[GraphToGraphNetwork])` — [`L356`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L356) — Processes the latent graph with several steps of message passing.
  - `_process_step(self, processor_network_k, latent_graph_prev_k: typed_graph.TypedGraph)` — [`L373`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L373) — Single step of message passing with node/edge residual connections. — documented in [graphcast-deep_typed_graph_net](../../concepts/graphcast-deep_typed_graph_net.md)
  - `aggregate_fn(data, *args, **kwargs)` — [`L275`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L275)
  - `build_mlp(name, output_size)` — [`L206`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L206)
  - `build_mlp_with_maybe_layer_norm(name, output_size)` — [`L212`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L212) — documented in [graphcast-deep_typed_graph_net](../../concepts/graphcast-deep_typed_graph_net.md)
- protocol/private: `_activation`[`L170`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L170), `_aggregate_edges_for_nodes_fn`[`L172`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L172), `_aggregate_normalization`[`L174`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L174), `_edge_latent_size`[`L152`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L152), `_edge_output_size`[`L160`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L160), `_embed_edges`[`L158`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L158), `_embed_nodes`[`L157`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L157), `_f32_aggregation`[`L171`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L171), `_include_sent_messages_in_node_update`[`L161`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L161), `_mlp_hidden_size`[`L153`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L153), `_mlp_num_hidden_layers`[`L154`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L154), `_networks_builder`[`L198`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L198), `_node_latent_size`[`L151`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L151), `_node_output_size`[`L159`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L159), `_num_message_passing_steps`[`L155`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L155), `_num_processor_repetitions`[`L156`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L156), `_use_layer_norm`[`L168`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L168), `_use_norm_conditioning`[`L169`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L169)
- uses (calls/refs, reference-scoped): [`nodes`](typed_graph.md#TypedGraph.nodes), [`TypedGraph`](typed_graph.md#TypedGraph), [`features`](typed_graph.md#NodeSet.features), [`edges`](typed_graph.md#TypedGraph.edges), [`features`](typed_graph.md#EdgeSet.features), [`n_node`](typed_graph.md#NodeSet.n_node), [`context`](typed_graph.md#TypedGraph.context), [`features`](typed_graph.md#Context.features), [`GraphToGraphNetwork`](deep_typed_graph_net.md#GraphToGraphNetwork), [`InteractionNetwork`](typed_graph_net.md#InteractionNetwork), [`GraphMapFeatures`](typed_graph_net.md#GraphMapFeatures), [`LinearNormConditioning`](mlp.md#LinearNormConditioning), [`_build_update_fns_for_edge_types`](deep_typed_graph_net.md#_build_update_fns_for_edge_types), [`_build_update_fns_for_node_types`](deep_typed_graph_net.md#_build_update_fns_for_node_types), [`_get_activation_fn`](deep_typed_graph_net.md#_get_activation_fn), [`_get_aggregate_edges_for_nodes_fn`](deep_typed_graph_net.md#_get_aggregate_edges_for_nodes_fn)
- used by: [`_mesh_gnn`](graphcast.md#GraphCast._mesh_gnn), [`_grid2mesh_gnn`](denoiser.md#_DenoiserArchitecture._grid2mesh_gnn), [`_mesh2grid_gnn`](denoiser.md#_DenoiserArchitecture._mesh2grid_gnn), [`_grid2mesh_gnn`](graphcast.md#GraphCast._grid2mesh_gnn), [`_mesh2grid_gnn`](graphcast.md#GraphCast._mesh2grid_gnn)

## Functions
- `_build_update_fns_for_edge_types(builder_fn, graph_template, prefix, output_sizes=None)` — [`L425`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L425) — Builds an edge function for all node types or a subset of them.
- `_build_update_fns_for_node_types(builder_fn, graph_template, prefix, output_sizes=None)` — [`L405`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L405) — Builds an update function for all node types or a subset of them.
- `_get_activation_fn(name)` — [`L445`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L445) — Return activation function corresponding to function_name.
- `_get_aggregate_edges_for_nodes_fn(name)` — [`L456`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L456) — Return aggregate_edges_for_nodes_fn corresponding to function_name.

## Module values
- `GraphToGraphNetwork` — [`L50`](../../../../../raw/code/graphcast/graphcast/deep_typed_graph_net.py#L50)

