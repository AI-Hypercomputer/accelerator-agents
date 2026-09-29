---
title: graphcast.deep_typed_graph_net — DeepTypedGraphNet, the Encode-Process message-passing GNN
type: concept
provenance: mixed
concept: graphcast-deep_typed_graph_net
updated: 2026-07-03
status: fresh
---
# graphcast.deep_typed_graph_net — DeepTypedGraphNet, the Encode-Process message-passing GNN

## Overview

[`DeepTypedGraphNet.__call__`](../catalog/graphcast/deep_typed_graph_net.md#DeepTypedGraphNet.__call__)
("Forward pass of the learnable dynamics model") is the generic message-passing GNN every
`TypedGraph`-based graph in the codebase runs through: embed
([`_embed`](../catalog/graphcast/deep_typed_graph_net.md#DeepTypedGraphNet._embed), "Embeds the input
graph features into a latent graph"), then `_num_message_passing_steps` rounds of
[`_process_step`](../catalog/graphcast/deep_typed_graph_net.md#DeepTypedGraphNet._process_step)
("Single step of message passing with node/edge residual connections"). Which per-node-type and
per-edge-type sub-networks exist is determined generically from a `graph_template: TypedGraph` at
construction time via
[`_networks_builder`](../catalog/graphcast/deep_typed_graph_net.md#DeepTypedGraphNet._networks_builder) —
the same GNN implementation runs unchanged whether the graph is Grid2Mesh, Mesh, or Mesh2Grid (see
[graphcast](graphcast.md)).

## Diagram

```mermaid
flowchart TD
  TEMPLATE["graph_template: TypedGraph"] -->|_networks_builder| NETS["(embedder, [processor_k]*N, decoder)\nGraphToGraphNetwork instances,\none set of per-type MLPs each"]
  IN["input_graph: TypedGraph"] -->|_embed(input_graph, embedder)| LATENT0["latent graph (round 0)"]
  LATENT0 -->|_process_step(processor_1, latent_0)| LATENT1["latent graph (round 1)\n+ node/edge residual"]
  LATENT1 -->|... N steps ...| LATENTN["latent graph (round N)"]
  LATENTN --> DECODE["decoder GraphToGraphNetwork"]
  DECODE --> OUT["output TypedGraph"]
```

## Design rationale (why it's built this way)

**Sub-networks (embedder, per-round processors, decoder) are built once from a `graph_template`, not
once per forward call — construction is separated from execution so the same compiled network
structure is reused across every call with the same graph topology.**
[`_networks_builder`](../catalog/graphcast/deep_typed_graph_net.md#DeepTypedGraphNet._networks_builder)
returns `Tuple[GraphToGraphNetwork, List[GraphToGraphNetwork], GraphToGraphNetwork]` (embedder,
processor list, decoder) built from `graph_template` alone — it reads
`_num_message_passing_steps`/`_node_latent_size`/`_edge_latent_size`/`_include_sent_messages_in_node_update`
config fields, and calls
[`build_mlp_with_maybe_layer_norm`](../catalog/graphcast/deep_typed_graph_net.md#DeepTypedGraphNet.build_mlp_with_maybe_layer_norm)
per node/edge type to build the actual per-type MLPs (`_build_update_fns_for_edge_types`/
`_build_update_fns_for_node_types`).

**Message passing carries explicit node *and* edge residual connections per step, not just node
residuals — because edge latents (representing e.g. relative position/distance) also benefit from
depth without vanishing.**
[`_process_step`](../catalog/graphcast/deep_typed_graph_net.md#DeepTypedGraphNet._process_step)'s own
doc — "Single step of message passing with node/edge residual connections" — makes this explicit;
each call reads `edges`/`features`/`nodes` off the previous step's `TypedGraph` and returns an
updated one.

**`build_mlp_with_maybe_layer_norm` makes layer normalization and conditioning-based normalization
independently optional per-network, not globally fixed.** It reads `_use_layer_norm`/
`_use_norm_conditioning` flags, optionally wrapping the base `build_mlp`-constructed MLP with
`LinearNormConditioning` — this lets different sub-networks (embedder vs. processor vs. decoder) use
different normalization strategies within the same overall model.

**Aggregation across incoming edges for a node update is itself parameterized
(`_aggregate_edges_for_nodes_fn`, `_aggregate_normalization`, `_f32_aggregation`), not hardcoded to
plain sum.** The presence of an explicit `_f32_aggregation` flag alongside the aggregation function
choice suggests aggregation precision is tunable independently of the rest of the model's compute
precision — plausibly to avoid accumulation error when summing over many incoming messages in bf16.

## Entry points

- [`DeepTypedGraphNet.__call__`](../catalog/graphcast/deep_typed_graph_net.md#DeepTypedGraphNet.__call__) —
  the forward pass; called once per `TypedGraph` (Grid2Mesh, Mesh, or Mesh2Grid) per model forward
  pass.
- [`DeepTypedGraphNet._networks_builder`](../catalog/graphcast/deep_typed_graph_net.md#DeepTypedGraphNet._networks_builder) —
  called once at model construction time per graph template.

## Mechanism (step-by-step)

1. **At construction, `_networks_builder` inspects `graph_template` and builds one embedder, one
   processor per message-passing round, and one decoder** — all `GraphToGraphNetwork` instances, each
   internally holding per-node-type/per-edge-type MLPs built via
   [`build_mlp_with_maybe_layer_norm`](../catalog/graphcast/deep_typed_graph_net.md#DeepTypedGraphNet.build_mlp_with_maybe_layer_norm).
2. **`__call__` embeds the input graph into latent space** via
   [`_embed`](../catalog/graphcast/deep_typed_graph_net.md#DeepTypedGraphNet._embed), which maps each
   node/edge type's raw `features` through its own embedder sub-network.
3. **`_process` repeats [`_process_step`](../catalog/graphcast/deep_typed_graph_net.md#DeepTypedGraphNet._process_step)
   `_num_message_passing_steps` times**, each step updating every edge type's latent (from its
   endpoints' node latents) then every node type's latent (aggregating incoming edge latents), with
   residual connections carrying the previous step's values forward.
4. **The final latent graph is returned by
   [`DeepTypedGraphNet.__call__`](../catalog/graphcast/deep_typed_graph_net.md#DeepTypedGraphNet.__call__)**
   (decoding to output feature space, via the decoder network,
   happens as part of the same overall `__call__`/`_process` pipeline).

## Key data structures

- **`GraphToGraphNetwork`** — one embedder/processor/decoder instance; internally a set of per-type
  MLPs operating over one `TypedGraph`'s node/edge types.
- **`TypedGraph`** (see [graphcast-typed_graph](graphcast-typed_graph.md)) — both the input and the
  per-step latent representation; `_embed`/`_process_step` both consume and produce one.

## Dynamics (design intent)

Because `_process_step` explicitly carries "node/edge residual connections," the message-passing
depth (`_num_message_passing_steps`) can be scaled without the usual vanishing-gradient degradation
deep GNNs without residuals suffer — consistent with GraphCast/GenCast using a comparatively deep
message-passing stack.

## Edge cases
None directly visible in this packet's subgraph.

## Open questions
- The exact aggregation function `_aggregate_edges_for_nodes_fn` defaults to (sum, mean, max, ...)
  isn't resolved by the symbols in this packet's subgraph.

## See also
- [graphcast-typed_graph](graphcast-typed_graph.md) — `TypedGraph`, the data structure this GNN
  operates over.
- [graphcast](graphcast.md) — `GraphCast`, which constructs the three `TypedGraph`s this network
  processes.
