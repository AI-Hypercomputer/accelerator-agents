---
title: 'Module: graphcast/graphcast.py'
type: catalog
provenance: extracted
module: graphcast/graphcast.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.graphcast`/
symbols:
  GraphCast._init_grid2mesh_graph: GraphCast#_init_grid2mesh_graph().
  GraphCast._init_mesh2grid_graph: GraphCast#_init_mesh2grid_graph().
  GraphCast._init_mesh_graph: GraphCast#_init_mesh_graph().
  GraphCast._maybe_init: GraphCast#_maybe_init().
  GraphCast._run_grid2mesh_gnn: GraphCast#_run_grid2mesh_gnn().
  GraphCast._run_mesh_gnn: GraphCast#_run_mesh_gnn().
  GraphCast._run_mesh2grid_gnn: GraphCast#_run_mesh2grid_gnn().
  GraphCast._mesh_gnn: GraphCast#_mesh_gnn.
  GraphCast._init_mesh_properties: GraphCast#_init_mesh_properties().
  TASK_13_PRECIP_OUT: TASK_13_PRECIP_OUT.
  TASK: TASK.
  TASK_13: TASK_13.
  GraphCast.__call__: GraphCast#__call__().
  GraphCast._init_grid_properties: GraphCast#_init_grid_properties().
  GraphCast._grid_node_outputs_to_prediction: GraphCast#_grid_node_outputs_to_prediction().
  FORCING_VARS: FORCING_VARS.
  ModelConfig.latent_size: ModelConfig#latent_size.
  GraphCast._finest_mesh: GraphCast#_finest_mesh().
  GraphCast._grid2mesh_gnn: GraphCast#_grid2mesh_gnn.
  GraphCast._mesh2grid_gnn: GraphCast#_mesh2grid_gnn.
  GraphCast._query_radius: GraphCast#_query_radius.
  GraphCast._inputs_to_grid_node_features: GraphCast#_inputs_to_grid_node_features().
  TARGET_ATMOSPHERIC_VARS: TARGET_ATMOSPHERIC_VARS.
  TaskConfig: TaskConfig#
  PRESSURE_LEVELS: PRESSURE_LEVELS.
  GraphCast._meshes: GraphCast#_meshes.
  GraphCast._mesh_nodes_lat: GraphCast#_mesh_nodes_lat.
  GraphCast._mesh_nodes_lon: GraphCast#_mesh_nodes_lon.
  GraphCast._grid_lat: GraphCast#_grid_lat.
  GraphCast._grid_lon: GraphCast#_grid_lon.
  GraphCast.loss_and_predictions: GraphCast#loss_and_predictions().
  TARGET_SURFACE_VARS: TARGET_SURFACE_VARS.
  GraphCast.__init__: GraphCast#__init__().
  GraphCast._num_mesh_nodes: GraphCast#_num_mesh_nodes.
  GraphCast.loss: GraphCast#loss().
  _add_batch_second_axis: _add_batch_second_axis().
  PRESSURE_LEVELS_WEATHERBENCH_13: PRESSURE_LEVELS_WEATHERBENCH_13.
  ALL_ATMOSPHERIC_VARS: ALL_ATMOSPHERIC_VARS.
  STATIC_VARS: STATIC_VARS.
  TaskConfig.target_variables: TaskConfig#target_variables.
  GraphCast._mesh2grid_edge_normalization_factor: GraphCast#_mesh2grid_edge_normalization_factor.
  _get_max_edge_distance: _get_max_edge_distance().
  GENERATED_FORCING_VARS: GENERATED_FORCING_VARS.
  ModelConfig.hidden_layers: ModelConfig#hidden_layers.
  GraphCast._spatial_features_kwargs: GraphCast#_spatial_features_kwargs.
  GraphCast._num_grid_nodes: GraphCast#_num_grid_nodes.
  GraphCast._grid_nodes_lat: GraphCast#_grid_nodes_lat.
  GraphCast._grid_nodes_lon: GraphCast#_grid_nodes_lon.
  GraphCast._grid2mesh_graph_structure: GraphCast#_grid2mesh_graph_structure.
  CheckPoint.model_config: CheckPoint#model_config.
  CheckPoint.task_config: CheckPoint#task_config.
  GraphCast: GraphCast#
  PRESSURE_LEVELS_ERA5_37: PRESSURE_LEVELS_ERA5_37.
  TaskConfig.pressure_levels: TaskConfig#pressure_levels.
  ModelConfig: ModelConfig#
  GraphCast._initialized: GraphCast#_initialized.
  GraphCast._mesh_graph_structure: GraphCast#_mesh_graph_structure.
  GraphCast._mesh2grid_graph_structure: GraphCast#_mesh2grid_graph_structure.
  PRESSURE_LEVELS_HRES_25: PRESSURE_LEVELS_HRES_25.
  TARGET_SURFACE_NO_PRECIP_VARS: TARGET_SURFACE_NO_PRECIP_VARS.
  EXTERNAL_FORCING_VARS: EXTERNAL_FORCING_VARS.
  ModelConfig.mesh_size: ModelConfig#mesh_size.
  ModelConfig.gnn_msg_steps: ModelConfig#gnn_msg_steps.
  ModelConfig.radius_query_fraction_edge_length: ModelConfig#radius_query_fraction_edge_length.
  ModelConfig.mesh2grid_edge_normalization_factor: ModelConfig#mesh2grid_edge_normalization_factor.
  Kwargs: Kwargs.
  GNN: GNN.
  TARGET_ATMOSPHERIC_NO_W_VARS: TARGET_ATMOSPHERIC_NO_W_VARS.
  TaskConfig.input_variables: TaskConfig#input_variables.
  TaskConfig.forcing_variables: TaskConfig#forcing_variables.
  TaskConfig.input_duration: TaskConfig#input_duration.
  ModelConfig.resolution: ModelConfig#resolution.
  CheckPoint: CheckPoint#
  CheckPoint.params: CheckPoint#params.
  CheckPoint.description: CheckPoint#description.
  CheckPoint.license: CheckPoint#license.
---
# Module: [`graphcast/graphcast.py`](../../../../../raw/code/graphcast/graphcast/graphcast.py)

## Classes
### `CheckPoint`
- def: [`graphcast/graphcast.py:205`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L205)
- signature: `class CheckPoint:`
- members:
  - `description` — [`L209`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L209)
  - `license` — [`L210`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L210)
  - `model_config` — [`L207`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L207)
  - `params` — [`L206`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L206)
  - `task_config` — [`L208`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L208)
- uses (calls/refs, reference-scoped): [`TaskConfig`](graphcast.md#TaskConfig), [`ModelConfig`](graphcast.md#ModelConfig)

### `GraphCast`  ·  implements/extends Predictor
- def: [`graphcast/graphcast.py:213`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L213)
- doc: GraphCast Predictor.
- signature: `class GraphCast(predictor_base.Predictor):`
- members:
  - `__init__(self, model_config: ModelConfig, task_config: TaskConfig)` — [`L243`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L243) — Initializes the predictor.
  - `_grid_node_outputs_to_prediction(self, grid_node_outputs: chex.Array, targets_template: xarray.Dataset)` — [`L760`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L760) — [num_grid_nodes, batch, num_outputs] -> xarray.
  - `_init_grid2mesh_graph(self)` — [`L467`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L467) — Build Grid2Mesh graph. — documented in [graphcast-typed_graph](../../concepts/graphcast-typed_graph.md)
  - `_init_grid_properties(self, grid_lat: np.ndarray, grid_lon: np.ndarray)` — [`L455`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L455) — Inits static properties that have to do with grid nodes.
  - `_init_mesh2grid_graph(self)` — [`L558`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L558) — Build Mesh2Grid graph. — documented in [graphcast](../../concepts/graphcast.md)
  - `_init_mesh_graph(self)` — [`L519`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L519) — Build Mesh graph. — documented in [graphcast](../../concepts/graphcast.md)
  - `_init_mesh_properties(self)` — [`L439`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L439) — Inits static properties that have to do with mesh nodes.
  - `_inputs_to_grid_node_features(self, inputs: xarray.Dataset, forcings: xarray.Dataset)` — [`L739`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L739) — xarrays -> [num_grid_nodes, batch, num_channels].
  - `_maybe_init(self, sample_inputs: xarray.Dataset)` — [`L427`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L427) — Inits everything that has a dependency on the input coordinates. — documented in [graphcast](../../concepts/graphcast.md)
  - `_run_grid2mesh_gnn(self, grid_node_features: chex.Array)` — [`L609`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L609) — Runs the grid2mesh_gnn, extracting latent mesh and grid nodes. — documented in [graphcast-typed_graph](../../concepts/graphcast-typed_graph.md)
  - `_run_mesh2grid_gnn(self, updated_latent_mesh_nodes: chex.Array, latent_grid_nodes: chex.Array)` — [`L700`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L700) — Runs the mesh2grid_gnn, extracting the output grid nodes.
  - `_run_mesh_gnn(self, latent_mesh_nodes: chex.Array)` — [`L665`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L665) — Runs the mesh_gnn, extracting updated latent mesh nodes.
  - `loss(self, inputs: xarray.Dataset, targets: xarray.Dataset, forcings: xarray.Dataset)` — [`L418`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L418)
  - `loss_and_predictions(self, inputs: xarray.Dataset, targets: xarray.Dataset, forcings: xarray.Dataset)` — [`L390`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L390)
- protocol/private: `__call__`[`L357`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L357), `_finest_mesh`[`L354`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L354), `_grid2mesh_gnn`[`L261`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L261), `_grid2mesh_graph_structure`[`L349`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L349), `_grid_lat`[`L342`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L342), `_grid_lon`[`L343`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L343), `_grid_nodes_lat`[`L345`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L345), `_grid_nodes_lon`[`L346`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L346), `_initialized`[`L333`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L333), `_mesh2grid_edge_normalization_factor`[`L327`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L327), `_mesh2grid_gnn`[`L304`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L304), `_mesh2grid_graph_structure`[`L351`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L351), `_mesh_gnn`[`L280`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L280), `_mesh_graph_structure`[`L350`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L350), `_mesh_nodes_lat`[`L338`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L338), `_mesh_nodes_lon`[`L339`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L339), `_meshes`[`L255`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L255), `_num_grid_nodes`[`L344`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L344), `_num_mesh_nodes`[`L337`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L337), `_query_radius`[`L325`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L325), `_spatial_features_kwargs`[`L245`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L245)
- uses (calls/refs, reference-scoped): [`nodes`](typed_graph.md#TypedGraph.nodes), [`TypedGraph`](typed_graph.md#TypedGraph), [`DataArray`](xarray_jax.md#DataArray), [`features`](typed_graph.md#NodeSet.features), [`edges`](typed_graph.md#TypedGraph.edges), [`Predictor`](predictor_base.md#Predictor), [`vertices`](icosahedral_mesh.md#TriangularMesh.vertices), [`edge_key_by_name`](typed_graph.md#TypedGraph.edge_key_by_name), [`features`](typed_graph.md#EdgeSet.features), [`LossAndDiagnostics`](predictor_base.md#LossAndDiagnostics), [`n_node`](typed_graph.md#NodeSet.n_node), [`get_hierarchy_of_triangular_meshes_for_sphere`](icosahedral_mesh.md#get_hierarchy_of_triangular_meshes_for_sphere), [`in_mesh_triangle_indices`](grid_mesh_connectivity.md#in_mesh_triangle_indices), [`faces`](icosahedral_mesh.md#TriangularMesh.faces), [`get_graph_spatial_features`](model_utils.md#get_graph_spatial_features), [`weighted_mse_per_level`](losses.md#weighted_mse_per_level), [`context`](typed_graph.md#TypedGraph.context), [`get_bipartite_graph_spatial_features`](model_utils.md#get_bipartite_graph_spatial_features), [`NodeSet`](typed_graph.md#NodeSet), [`latent_size`](graphcast.md#ModelConfig.latent_size), [`unwrap`](xarray_jax.md#unwrap), [`features`](typed_graph.md#Context.features), [`radius_query_indices`](grid_mesh_connectivity.md#radius_query_indices), [`senders`](typed_graph.md#EdgesIndices.senders), [`EdgesIndices`](typed_graph.md#EdgesIndices), [`indices`](typed_graph.md#EdgeSet.indices), [`merge_meshes`](icosahedral_mesh.md#merge_meshes), [`receivers`](typed_graph.md#EdgesIndices.receivers), [`n_edge`](typed_graph.md#EdgeSet.n_edge), [`n_graph`](typed_graph.md#Context.n_graph), [`EdgeSet`](typed_graph.md#EdgeSet), [`EdgeSetKey`](typed_graph.md#EdgeSetKey), [`TaskConfig`](graphcast.md#TaskConfig), [`dataset_to_stacked`](model_utils.md#dataset_to_stacked), [`Context`](typed_graph.md#Context), [`faces_to_edges`](icosahedral_mesh.md#faces_to_edges), [`cartesian_to_spherical`](model_utils.md#cartesian_to_spherical), [`spherical_to_lat_lon`](model_utils.md#spherical_to_lat_lon), [`DeepTypedGraphNet`](deep_typed_graph_net.md#DeepTypedGraphNet), [`_add_batch_second_axis`](graphcast.md#_add_batch_second_axis)  (+13 more)
- used by: [`Predictor`](predictor_base.md#Predictor), [`loss`](predictor_base.md#Predictor.loss), [`loss_and_predictions`](predictor_base.md#Predictor.loss_and_predictions)

### `ModelConfig`
- def: [`graphcast/graphcast.py:175`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L175)
- doc: Defines the architecture of the GraphCast neural network architecture.
- signature: `class ModelConfig:`
- members:
  - `gnn_msg_steps` — [`L198`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L198)
  - `hidden_layers` — [`L199`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L199)
  - `latent_size` — [`L197`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L197)
  - `mesh2grid_edge_normalization_factor` — [`L201`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L201)
  - `mesh_size` — [`L196`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L196)
  - `radius_query_fraction_edge_length` — [`L200`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L200)
  - `resolution` — [`L195`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L195)
- used by: [`_mesh_gnn`](graphcast.md#GraphCast._mesh_gnn), [`_grid2mesh_gnn`](graphcast.md#GraphCast._grid2mesh_gnn), [`_mesh2grid_gnn`](graphcast.md#GraphCast._mesh2grid_gnn), [`_query_radius`](graphcast.md#GraphCast._query_radius), [`_meshes`](graphcast.md#GraphCast._meshes), [`__init__`](graphcast.md#GraphCast.__init__), [`_mesh2grid_edge_normalization_factor`](graphcast.md#GraphCast._mesh2grid_edge_normalization_factor), [`model_config`](graphcast.md#CheckPoint.model_config)

### `TaskConfig`
- def: [`graphcast/graphcast.py:136`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L136)
- doc: Defines inputs and targets on which a model is trained and/or evaluated.
- signature: `class TaskConfig:`
- members:
  - `forcing_variables` — [`L141`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L141)
  - `input_duration` — [`L143`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L143)
  - `input_variables` — [`L138`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L138)
  - `pressure_levels` — [`L142`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L142)
  - `target_variables` — [`L140`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L140)
- used by: [`__init__`](gencast.md#GenCast.__init__), [`_mesh_gnn`](graphcast.md#GraphCast._mesh_gnn), [`TASK`](gencast.md#TASK), [`TASK_13_PRECIP_OUT`](graphcast.md#TASK_13_PRECIP_OUT), [`TASK`](graphcast.md#TASK), [`TASK_13`](graphcast.md#TASK_13), [`__init__`](graphcast.md#GraphCast.__init__), [`task_config`](gencast.md#CheckPoint.task_config), [`task_config`](graphcast.md#CheckPoint.task_config)

## Functions
- `_add_batch_second_axis(data, batch_size)` — [`L785`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L785)
- `_get_max_edge_distance(mesh)` — [`L792`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L792)

## Module values
- `ALL_ATMOSPHERIC_VARS` — [`L72`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L72)
- `EXTERNAL_FORCING_VARS` — [`L119`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L119)
- `FORCING_VARS` — [`L128`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L128)
- `GENERATED_FORCING_VARS` — [`L122`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L122)
- `GNN` — [`L46`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L46)
- `Kwargs` — [`L44`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L44)
- `PRESSURE_LEVELS` — [`L64`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L64)
- `PRESSURE_LEVELS_ERA5_37` — [`L50`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L50)
- `PRESSURE_LEVELS_HRES_25` — [`L56`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L56)
- `PRESSURE_LEVELS_WEATHERBENCH_13` — [`L61`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L61)
- `STATIC_VARS` — [`L129`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L129)
- `TARGET_ATMOSPHERIC_NO_W_VARS` — [`L112`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L112)
- `TARGET_ATMOSPHERIC_VARS` — [`L104`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L104)
- `TARGET_SURFACE_NO_PRECIP_VARS` — [`L98`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L98)
- `TARGET_SURFACE_VARS` — [`L91`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L91)
- `TASK` — [`L145`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L145)
- `TASK_13` — [`L154`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L154)
- `TASK_13_PRECIP_OUT` — [`L163`](../../../../../raw/code/graphcast/graphcast/graphcast.py#L163)

