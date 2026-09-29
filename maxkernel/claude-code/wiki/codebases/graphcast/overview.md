---
title: graphcast — overview
type: overview
updated: 2026-07-03
---
# graphcast — what it is and how it fits together

## In one paragraph

GraphCast (and its diffusion-based sibling GenCast) is a graph-neural-network weather forecaster: a
lat/lon grid is encoded onto an icosahedral mesh, processed by deep message-passing (or, in GenCast's
transformer variant, block-sparse/Splash attention) over the mesh, then decoded back onto the grid.
This catalog focuses on the TPU/JAX-perf-relevant surfaces: the three interchangeable attention
backends GenCast's transformer supports, the mixed-precision (bfloat16) casting wrapper, and the
generic `TypedGraph`/`DeepTypedGraphNet` message-passing machinery shared by every graph in the model.
The remaining ~30 files (data preprocessing, solar-radiation feature engineering, samplers, xarray/JAX
glue) are domain-science and I/O code, out of scope for this perf-focused catalog.

## Core architecture

```mermaid
flowchart TD
  GRID["Lat/Lon grid"] -->|_init_grid2mesh_graph (radius query)| G2M["Grid2Mesh TypedGraph"]
  G2M -->|DeepTypedGraphNet| MESH["Mesh TypedGraph\n(icosahedral mesh, _init_mesh_graph)"]
  MESH -->|DeepTypedGraphNet\n(GraphCast) OR sparse_transformer\n(GenCast)| MESHOUT["updated mesh latents"]
  MESHOUT -->|_init_mesh2grid_graph (triangle containment)| M2G["Mesh2Grid TypedGraph"]
  M2G --> GRIDOUT["decoded grid predictions"]
  CAST["Bfloat16Cast"] -.->|wraps| GRAPHCAST["GraphCast / GenCast Predictor"]
```

## Main concepts

**Three graphs, one shared `TypedGraph` representation.** Grid2Mesh (encode, radius-query
connectivity), Mesh (message-passing, icosahedral edges), and Mesh2Grid (decode, triangle-containment
connectivity) are all built once (lazily, on first input) and reused; all three share one
`TypedGraph`/`NodeSet`/`EdgeSet` type. See [graphcast](concepts/graphcast.md) and
[graphcast-typed_graph](concepts/graphcast-typed_graph.md).

**`DeepTypedGraphNet` is one generic Encode-Process GNN reused across every graph.** Sub-networks are
built once from a graph template; message passing carries both node and edge residual connections.
See [graphcast-deep_typed_graph_net](concepts/graphcast-deep_typed_graph_net.md).

**GenCast's transformer has three interchangeable attention backends behind one calling convention.**
Dense (`mha`), TPU Pallas Splash (`splash_mha`), and a custom block-diagonal sparse pattern
(`triblockdiag_mha`) share one `_ModelConfig` and `multihead_linear` projection — swapping backends is
a config change. See [graphcast-sparse_transformer](concepts/graphcast-sparse_transformer.md).

**Mixed precision is a composable `Predictor` decorator, not a training-loop-level cast.**
`Bfloat16Cast` casts inputs/outputs around any wrapped predictor while keeping Haiku parameters in
float32 via `bfloat16_variable_view` — the same `loss`/`loss_and_predictions`/`__call__` interface
every predictor implements makes this composition transparent. See
[graphcast-casting](concepts/graphcast-casting.md).

## How a request flows

`GraphCast.__call__` (or GenCast's denoiser equivalent) lazily builds all three graphs on first call
(`_maybe_init`), encodes grid features onto the mesh (`_run_grid2mesh_gnn`, an embed + message-passing
pass via `DeepTypedGraphNet`), processes the mesh graph (further message-passing rounds, or GenCast's
sparse-transformer blocks), then decodes back onto the grid via the Mesh2Grid graph. If wrapped in
`Bfloat16Cast`, every one of these stages runs with bf16 activations and float32 parameters.

## Map of the wiki

- "How is the three-graph architecture built and connected?" → [graphcast](concepts/graphcast.md).
- "What is `TypedGraph` and how is it shared across graphs?" →
  [graphcast-typed_graph](concepts/graphcast-typed_graph.md).
- "How does the message-passing GNN work?" →
  [graphcast-deep_typed_graph_net](concepts/graphcast-deep_typed_graph_net.md).
- "How do the three attention backends (dense/Splash/block-sparse) work?" →
  [graphcast-sparse_transformer](concepts/graphcast-sparse_transformer.md).
- "How does mixed-precision (bf16) casting work?" → [graphcast-casting](concepts/graphcast-casting.md).
- For the exhaustive per-symbol index, see `catalog/`; for the ranked concept list, see `index.md`.
