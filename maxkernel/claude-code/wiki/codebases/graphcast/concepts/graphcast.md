---
title: graphcast.graphcast — the three-graph Grid2Mesh/Mesh/Mesh2Grid GNN architecture
type: concept
provenance: mixed
concept: graphcast
updated: 2026-07-03
status: fresh
---
# graphcast.graphcast — the three-graph Grid2Mesh/Mesh/Mesh2Grid GNN architecture

## Overview

`GraphCast`'s forward pass is an Encode-Process-Decode GNN over three distinct
[`TypedGraph`](../catalog/graphcast/typed_graph.md#TypedGraph)s: a bipartite Grid2Mesh graph
(encoding lat/lon grid points onto the icosahedral mesh,
[`GraphCast._init_grid2mesh_graph`](../catalog/graphcast/graphcast.md#GraphCast._init_grid2mesh_graph)),
a mesh-internal graph for message passing
([`GraphCast._init_mesh_graph`](../catalog/graphcast/graphcast.md#GraphCast._init_mesh_graph)), and a
bipartite Mesh2Grid graph decoding back to the original grid
([`GraphCast._init_mesh2grid_graph`](../catalog/graphcast/graphcast.md#GraphCast._init_mesh2grid_graph)).
All three graphs are lazily built once the input coordinates are known
([`GraphCast._maybe_init`](../catalog/graphcast/graphcast.md#GraphCast._maybe_init)) and reused across
calls — the graph *topology* depends only on the grid/mesh geometry, not on the weather data itself.

## Diagram

```mermaid
flowchart LR
  GRID["Lat/Lon grid nodes"] -->|_init_grid2mesh_graph\n(bipartite, radius query)| G2M["Grid2Mesh TypedGraph"]
  G2M -->|_run_grid2mesh_gnn| MESHLATENT["latent mesh + grid nodes"]
  MESHLATENT --> MESH["Mesh TypedGraph\n(_init_mesh_graph: icosahedral mesh edges)"]
  MESH -->|message passing (deep_typed_graph_net)| MESHOUT["updated mesh latents"]
  MESHOUT -->|_init_mesh2grid_graph\n(bipartite, triangle-containment)| M2G["Mesh2Grid TypedGraph"]
  M2G --> GRIDOUT["decoded grid-node predictions"]
```

## Design rationale (why it's built this way)

**All three graphs share one `TypedGraph` representation, and graph *construction* is fully separated
from graph *execution* — building a graph populates static geometric features, while the GNN forward
pass consumes those features as ordinary array data.**
[`GraphCast._init_grid2mesh_graph`](../catalog/graphcast/graphcast.md#GraphCast._init_grid2mesh_graph)
("Build Grid2Mesh graph") and its siblings all construct a
[`TypedGraph`](../catalog/graphcast/typed_graph.md#TypedGraph) from `NodeSet`/`EdgeSet` objects whose
`features` are computed once via
[`get_bipartite_graph_spatial_features`](../catalog/graphcast/model_utils.md#get_bipartite_graph_spatial_features)/
[`get_graph_spatial_features`](../catalog/graphcast/model_utils.md#get_graph_spatial_features) — the
actual weather-state values are threaded through separately at call time, not baked into the graph
structure.

**Grid→Mesh connectivity is a radius query (nearest mesh nodes within a fixed radius of each grid
point), while Mesh→Grid connectivity is triangle containment (which mesh triangle a grid point falls
inside) — two different geometric criteria for the two directions, not a symmetric/shared edge set.**
[`_init_grid2mesh_graph`](../catalog/graphcast/graphcast.md#GraphCast._init_grid2mesh_graph) uses
`_query_radius`, while
[`_init_mesh2grid_graph`](../catalog/graphcast/graphcast.md#GraphCast._init_mesh2grid_graph) uses
[`in_mesh_triangle_indices`](../catalog/graphcast/grid_mesh_connectivity.md#in_mesh_triangle_indices)
("Returns mesh-grid edge indices for grid points contained in mesh triangles") — encoding favors local
density-aware coverage (radius query), decoding favors exact geometric containment (which triangle
each output point belongs to, for later interpolation-like decoding).

**`_maybe_init` defers all coordinate-dependent setup (the three graphs) until the first real input is
seen, rather than requiring an explicit init call with known coordinates up front.** Its own doc —
"Inits everything that has a dependency on the input coordinates" — combined with `GraphCast.__call__`
calling it unconditionally each forward pass (a no-op after the first call, presumably guarded
internally) makes the model usable without the caller needing to know grid/mesh details ahead of time.

## Entry points

- [`GraphCast.__call__`](../catalog/graphcast/graphcast.md#GraphCast.__call__) — the top-level forward
  pass: `inputs`, `targets_template`, `forcings` → predicted `xarray.Dataset`.
- [`GraphCast._maybe_init`](../catalog/graphcast/graphcast.md#GraphCast._maybe_init) — lazily
  constructs all three graphs the first time input coordinates are seen.
- [`GraphCast._run_grid2mesh_gnn`](../catalog/graphcast/graphcast.md#GraphCast._run_grid2mesh_gnn) —
  "Runs the grid2mesh_gnn, extracting latent mesh and grid nodes"; the encoder stage.

## Mechanism (step-by-step)

1. **On first call, `_maybe_init` builds all three graphs** — Grid2Mesh, Mesh, and Mesh2Grid — each a
   [`TypedGraph`](../catalog/graphcast/typed_graph.md#TypedGraph) with precomputed geometric
   `features` on its `nodes`/`edges`.
2. **[`_run_grid2mesh_gnn`](../catalog/graphcast/graphcast.md#GraphCast._run_grid2mesh_gnn) encodes
   grid-node features onto the mesh** via the Grid2Mesh graph, batching
   inputs (`_add_batch_second_axis`) and running the encoder GNN (`_grid2mesh_gnn`), returning both
   updated latent mesh and grid nodes.
3. **The mesh-internal graph (built by
   [`_init_mesh_graph`](../catalog/graphcast/graphcast.md#GraphCast._init_mesh_graph)) propagates
   information across mesh nodes** via message passing (see
   [graphcast-deep_typed_graph_net](graphcast-deep_typed_graph_net.md) for the per-node/edge MLP
   mechanics).
4. **The Mesh2Grid graph (built by
   [`_init_mesh2grid_graph`](../catalog/graphcast/graphcast.md#GraphCast._init_mesh2grid_graph))
   decodes updated mesh latents back onto the original grid points**, using the
   triangle-containment edges computed at graph-construction time.

## Key data structures

- **[`TypedGraph`](../catalog/graphcast/typed_graph.md#TypedGraph)** — "A graph with typed nodes and
  edges"; [`nodes`](../catalog/graphcast/typed_graph.md#TypedGraph.nodes) (a `Mapping[str, NodeSet]`),
  [`edges`](../catalog/graphcast/typed_graph.md#TypedGraph.edges) (a `Mapping[EdgeSetKey, EdgeSet]`).
- **`NodeSet`/`EdgeSet`** — each carries
  [`features`](../catalog/graphcast/typed_graph.md#NodeSet.features)/
  [`features`](../catalog/graphcast/typed_graph.md#EdgeSet.features) (an `ArrayLikeTree`) plus
  [`n_node`](../catalog/graphcast/typed_graph.md#NodeSet.n_node) or sender/receiver indices.
- **`icosahedral_mesh.TriangularMesh`** — [`vertices`](../catalog/graphcast/icosahedral_mesh.md#TriangularMesh.vertices)/
  [`faces`](../catalog/graphcast/icosahedral_mesh.md#TriangularMesh.faces); the geometric mesh
  `in_mesh_triangle_indices` queries against.

## Dynamics (design intent)
Not addressable beyond the lazy-init-then-reuse pipeline described above from this packet's subgraph.

## Edge cases
None directly visible in this packet's subgraph beyond the coordinate-dependent lazy initialization.

## Open questions
- Whether `_maybe_init` re-triggers graph construction if input coordinates change between calls (vs.
  asserting they must stay fixed for a given model instance) isn't resolved by the symbols in this
  packet's subgraph.

## See also
- [graphcast-icosahedral_mesh](graphcast-icosahedral_mesh.md) — `TriangularMesh`, the geometric mesh
  underlying `_init_mesh_graph`.
- [graphcast-sparse_transformer](graphcast-sparse_transformer.md) — GenCast's transformer-based
  alternative to this GNN message-passing architecture.
