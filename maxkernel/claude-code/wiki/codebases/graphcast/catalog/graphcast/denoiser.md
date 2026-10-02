---
title: 'Module: graphcast/denoiser.py'
type: catalog
provenance: extracted
module: graphcast/denoiser.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.denoiser`/
symbols:
  _DenoiserArchitecture._init_grid2mesh_graph: _DenoiserArchitecture#_init_grid2mesh_graph().
  _DenoiserArchitecture._init_mesh2grid_graph: _DenoiserArchitecture#_init_mesh2grid_graph().
  _DenoiserArchitecture._init_mesh_graph: _DenoiserArchitecture#_init_mesh_graph().
  _DenoiserArchitecture._maybe_init: _DenoiserArchitecture#_maybe_init().
  _DenoiserArchitecture._run_grid2mesh_gnn: _DenoiserArchitecture#_run_grid2mesh_gnn().
  _DenoiserArchitecture._run_mesh_gnn: _DenoiserArchitecture#_run_mesh_gnn().
  _DenoiserArchitecture._run_mesh2grid_gnn: _DenoiserArchitecture#_run_mesh2grid_gnn().
  _DenoiserArchitecture._init_mesh_properties: _DenoiserArchitecture#_init_mesh_properties().
  _DenoiserArchitecture.__call__: _DenoiserArchitecture#__call__().
  _DenoiserArchitecture._init_grid_properties: _DenoiserArchitecture#_init_grid_properties().
  _DenoiserArchitecture._inputs_to_grid_node_features_and_norm_conditioning: _DenoiserArchitecture#_inputs_to_grid_node_features_and_norm_conditioning().
  _DenoiserArchitecture._grid_node_outputs_to_prediction: _DenoiserArchitecture#_grid_node_outputs_to_prediction().
  FourierFeaturesMLP.__call__: FourierFeaturesMLP#__call__().
  _DenoiserArchitecture._grid2mesh_gnn: _DenoiserArchitecture#_grid2mesh_gnn.
  _DenoiserArchitecture._mesh2grid_gnn: _DenoiserArchitecture#_mesh2grid_gnn.
  _permute_mesh_to_banded: _permute_mesh_to_banded().
  _DenoiserArchitecture._mesh: _DenoiserArchitecture#_mesh.
  Denoiser.__call__: Denoiser#__call__().
  _DenoiserArchitecture._mesh_gnn: _DenoiserArchitecture#_mesh_gnn.
  _DenoiserArchitecture._query_radius: _DenoiserArchitecture#_query_radius.
  DenoiserArchitectureConfig.latent_size: DenoiserArchitectureConfig#latent_size.
  _DenoiserArchitecture._spatial_features_kwargs: _DenoiserArchitecture#_spatial_features_kwargs.
  Denoiser._predictor: Denoiser#_predictor.
  _DenoiserArchitecture._mesh_nodes_lat: _DenoiserArchitecture#_mesh_nodes_lat.
  _DenoiserArchitecture._mesh_nodes_lon: _DenoiserArchitecture#_mesh_nodes_lon.
  _DenoiserArchitecture._grid_lat: _DenoiserArchitecture#_grid_lat.
  _DenoiserArchitecture._grid_lon: _DenoiserArchitecture#_grid_lon.
  _DenoiserArchitecture._norm_conditioning_features: _DenoiserArchitecture#_norm_conditioning_features.
  Denoiser.__init__: Denoiser#__init__().
  _DenoiserArchitecture._num_mesh_nodes: _DenoiserArchitecture#_num_mesh_nodes.
  _add_batch_second_axis: _add_batch_second_axis().
  NoiseEncoderConfig: NoiseEncoderConfig#
  DenoiserArchitectureConfig: DenoiserArchitectureConfig#
  DenoiserArchitectureConfig.sparse_transformer_config: DenoiserArchitectureConfig#sparse_transformer_config.
  Denoiser: Denoiser#
  Denoiser._noise_level_encoder: Denoiser#_noise_level_encoder.
  _get_max_edge_distance: _get_max_edge_distance().
  _DenoiserArchitecture._num_grid_nodes: _DenoiserArchitecture#_num_grid_nodes.
  _DenoiserArchitecture._grid_nodes_lat: _DenoiserArchitecture#_grid_nodes_lat.
  _DenoiserArchitecture._grid_nodes_lon: _DenoiserArchitecture#_grid_nodes_lon.
  _DenoiserArchitecture._grid2mesh_graph_structure: _DenoiserArchitecture#_grid2mesh_graph_structure.
  _DenoiserArchitecture.__init__: _DenoiserArchitecture#__init__().
  DenoiserArchitectureConfig.hidden_layers: DenoiserArchitectureConfig#hidden_layers.
  DenoiserArchitectureConfig.node_output_size: DenoiserArchitectureConfig#node_output_size.
  _DenoiserArchitecture._initialized: _DenoiserArchitecture#_initialized.
  _DenoiserArchitecture._mesh_graph_structure: _DenoiserArchitecture#_mesh_graph_structure.
  _DenoiserArchitecture._mesh2grid_graph_structure: _DenoiserArchitecture#_mesh2grid_graph_structure.
  FourierFeaturesMLP: FourierFeaturesMLP#
  FourierFeaturesMLP._base_period: FourierFeaturesMLP#_base_period.
  FourierFeaturesMLP._num_frequencies: FourierFeaturesMLP#_num_frequencies.
  FourierFeaturesMLP._apply_log_first: FourierFeaturesMLP#_apply_log_first.
  FourierFeaturesMLP._mlp: FourierFeaturesMLP#_mlp.
  SparseTransformerConfig: SparseTransformerConfig#
  DenoiserArchitectureConfig.mesh_size: DenoiserArchitectureConfig#mesh_size.
  DenoiserArchitectureConfig.radius_query_fraction_edge_length: DenoiserArchitectureConfig#radius_query_fraction_edge_length.
  DenoiserArchitectureConfig.norm_conditioning_features: DenoiserArchitectureConfig#norm_conditioning_features.
  DenoiserArchitectureConfig.grid2mesh_aggregate_normalization: DenoiserArchitectureConfig#grid2mesh_aggregate_normalization.
  _DenoiserArchitecture: _DenoiserArchitecture#
  Kwargs: Kwargs.
  NoiseLevelEncoder: NoiseLevelEncoder.
  FourierFeaturesMLP.__init__: FourierFeaturesMLP#__init__().
  NoiseEncoderConfig.apply_log_first: NoiseEncoderConfig#apply_log_first.
  NoiseEncoderConfig.base_period: NoiseEncoderConfig#base_period.
  NoiseEncoderConfig.num_frequencies: NoiseEncoderConfig#num_frequencies.
  NoiseEncoderConfig.output_sizes: NoiseEncoderConfig#output_sizes.
  SparseTransformerConfig.attention_k_hop: SparseTransformerConfig#attention_k_hop.
  SparseTransformerConfig.d_model: SparseTransformerConfig#d_model.
  SparseTransformerConfig.num_layers: SparseTransformerConfig#num_layers.
  SparseTransformerConfig.num_heads: SparseTransformerConfig#num_heads.
  SparseTransformerConfig.attention_type: SparseTransformerConfig#attention_type.
  SparseTransformerConfig.mask_type: SparseTransformerConfig#mask_type.
  SparseTransformerConfig.block_q: SparseTransformerConfig#block_q.
  SparseTransformerConfig.block_kv: SparseTransformerConfig#block_kv.
  SparseTransformerConfig.block_kv_compute: SparseTransformerConfig#block_kv_compute.
  SparseTransformerConfig.block_q_dkv: SparseTransformerConfig#block_q_dkv.
  SparseTransformerConfig.block_kv_dkv: SparseTransformerConfig#block_kv_dkv.
  SparseTransformerConfig.block_kv_dkv_compute: SparseTransformerConfig#block_kv_dkv_compute.
  SparseTransformerConfig.ffw_winit_final_mult: SparseTransformerConfig#ffw_winit_final_mult.
  SparseTransformerConfig.attn_winit_final_mult: SparseTransformerConfig#attn_winit_final_mult.
  SparseTransformerConfig.ffw_hidden: SparseTransformerConfig#ffw_hidden.
  SparseTransformerConfig.name: SparseTransformerConfig#name.
---
# Module: [`graphcast/denoiser.py`](../../../../../raw/code/graphcast/graphcast/denoiser.py)

## Classes
### `Denoiser`  ·  implements/extends Denoiser
- def: [`graphcast/denoiser.py:198`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L198)
- doc: Wraps a general deterministic Predictor to act as a Denoiser.
- signature: `class Denoiser(base.Denoiser):`
- protocol/private: `__call__`[`L222`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L222), `__init__`[`L209`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L209), `_noise_level_encoder`[`L220`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L220), `_predictor`[`L214`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L214)
- uses (calls/refs, reference-scoped): [`Variable`](xarray_jax.md#Variable), [`unwrap_data`](xarray_jax.md#unwrap_data), [`Denoiser`](denoisers_base.md#Denoiser), [`DenoiserArchitectureConfig`](denoiser.md#DenoiserArchitectureConfig), [`NoiseEncoderConfig`](denoiser.md#NoiseEncoderConfig), [`FourierFeaturesMLP`](denoiser.md#FourierFeaturesMLP), [`_DenoiserArchitecture`](denoiser.md#_DenoiserArchitecture)
- used by: [`Denoiser`](denoisers_base.md#Denoiser), [`_denoiser`](gencast.md#GenCast._denoiser)

### `DenoiserArchitectureConfig`
- def: [`graphcast/denoiser.py:157`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L157)
- doc: Defines the GenCast architecture.
- signature: `class DenoiserArchitectureConfig:`
- members:
  - `grid2mesh_aggregate_normalization` — [`L194`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L194)
  - `hidden_layers` — [`L191`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L191)
  - `latent_size` — [`L190`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L190)
  - `mesh_size` — [`L189`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L189)
  - `node_output_size` — [`L195`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L195)
  - `norm_conditioning_features` — [`L193`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L193)
  - `radius_query_fraction_edge_length` — [`L192`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L192)
  - `sparse_transformer_config` — [`L188`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L188)
- uses (calls/refs, reference-scoped): [`SparseTransformerConfig`](denoiser.md#SparseTransformerConfig)
- used by: [`__init__`](gencast.md#GenCast.__init__), [`_grid2mesh_gnn`](denoiser.md#_DenoiserArchitecture._grid2mesh_gnn), [`_mesh2grid_gnn`](denoiser.md#_DenoiserArchitecture._mesh2grid_gnn), [`_mesh_gnn`](denoiser.md#_DenoiserArchitecture._mesh_gnn), [`_query_radius`](denoiser.md#_DenoiserArchitecture._query_radius), [`_spatial_features_kwargs`](denoiser.md#_DenoiserArchitecture._spatial_features_kwargs), [`_norm_conditioning_features`](denoiser.md#_DenoiserArchitecture._norm_conditioning_features), [`__init__`](denoiser.md#Denoiser.__init__), [`__init__`](denoiser.md#_DenoiserArchitecture.__init__), [`denoiser_architecture_config`](gencast.md#CheckPoint.denoiser_architecture_config)

### `FourierFeaturesMLP`
- def: [`graphcast/denoiser.py:41`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L41)
- doc: A simple MLP applied to Fourier features of values or their logarithms.
- signature: `class FourierFeaturesMLP(hk.Module):`
- members:
  - `__init__(self, base_period: float, num_frequencies: int, output_sizes: Sequence[int], apply_log_first: bool = False, w_init=None, activation=jax.nn.gelu, **mlp_kwargs)` — [`L44`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L44) — Initializes the module.
- protocol/private: `__call__`[`L91`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L91), `_apply_log_first`[`L76`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L76), `_base_period`[`L74`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L74), `_mlp`[`L85`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L85), `_num_frequencies`[`L75`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L75)
- uses (calls/refs, reference-scoped): [`fourier_features`](model_utils.md#fourier_features)
- used by: [`_noise_level_encoder`](denoiser.md#Denoiser._noise_level_encoder)

### `NoiseEncoderConfig`
- def: [`graphcast/denoiser.py:102`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L102)
- doc: Configures the noise level encoding.
- signature: `class NoiseEncoderConfig:`
- members:
  - `apply_log_first` — [`L118`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L118)
  - `base_period` — [`L119`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L119)
  - `num_frequencies` — [`L120`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L120)
  - `output_sizes` — [`L122`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L122)
- used by: [`__init__`](gencast.md#GenCast.__init__), [`_predictor`](denoiser.md#Denoiser._predictor), [`__init__`](denoiser.md#Denoiser.__init__), [`noise_encoder_config`](gencast.md#CheckPoint.noise_encoder_config)

### `SparseTransformerConfig`
- def: [`graphcast/denoiser.py:126`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L126)
- doc: Sparse Transformer config.
- signature: `class SparseTransformerConfig:`
- members:
  - `attention_k_hop` — [`L129`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L129)
  - `attention_type` — [`L137`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L137)
  - `attn_winit_final_mult` — [`L149`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L149)
  - `block_kv` — [`L141`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L141)
  - `block_kv_compute` — [`L142`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L142)
  - `block_kv_dkv` — [`L144`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L144)
  - `block_kv_dkv_compute` — [`L145`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L145)
  - `block_q` — [`L140`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L140)
  - `block_q_dkv` — [`L143`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L143)
  - `d_model` — [`L131`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L131)
  - `ffw_hidden` — [`L151`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L151)
  - `ffw_winit_final_mult` — [`L147`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L147)
  - `mask_type` — [`L139`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L139)
  - `name` — [`L153`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L153)
  - `num_heads` — [`L135`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L135)
  - `num_layers` — [`L133`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L133)
- used by: [`sparse_transformer_config`](denoiser.md#DenoiserArchitectureConfig.sparse_transformer_config)

### `_DenoiserArchitecture`
- def: [`graphcast/denoiser.py:249`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L249)
- members:
  - `__init__(self, denoiser_architecture_config: DenoiserArchitectureConfig)` — [`L278`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L278) — Initializes the predictor.
  - `_grid_node_outputs_to_prediction(self, grid_node_outputs: chex.Array, targets_template: xarray.Dataset)` — [`L794`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L794) — [num_grid_nodes, batch, num_outputs] -> xarray.
  - `_init_grid2mesh_graph(self)` — [`L477`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L477) — Build Grid2Mesh graph. — documented in [graphcast-typed_graph](../../concepts/graphcast-typed_graph.md)
  - `_init_grid_properties(self, grid_lat: np.ndarray, grid_lon: np.ndarray)` — [`L465`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L465) — Inits static properties that have to do with grid nodes.
  - `_init_mesh2grid_graph(self)` — [`L568`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L568) — Build Mesh2Grid graph.
  - `_init_mesh_graph(self)` — [`L529`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L529) — Build Mesh graph.
  - `_init_mesh_properties(self)` — [`L449`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L449) — Inits static properties that have to do with mesh nodes.
  - `_inputs_to_grid_node_features_and_norm_conditioning(self, inputs: xarray.Dataset, forcings: xarray.Dataset)` — [`L755`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L755) — xarray ->[n_grid_nodes, batch, n_channels], [batch, n_cond channels].
  - `_maybe_init(self, sample_inputs: xarray.Dataset)` — [`L437`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L437) — Inits everything that has a dependency on the input coordinates.
  - `_run_grid2mesh_gnn(self, grid_node_features: chex.Array, global_norm_conditioning: Optional[chex.Array] = None)` — [`L619`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L619) — Runs the grid2mesh_gnn, extracting latent mesh and grid nodes.
  - `_run_mesh2grid_gnn(self, updated_latent_mesh_nodes: chex.Array, latent_grid_nodes: chex.Array, global_norm_conditioning: Optional[chex.Array] = None)` — [`L715`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L715) — Runs the mesh2grid_gnn, extracting the output grid nodes.
  - `_run_mesh_gnn(self, latent_mesh_nodes: chex.Array, global_norm_conditioning: Optional[chex.Array] = None)` — [`L676`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L676) — Runs the mesh_gnn, extracting updated latent mesh nodes.
- protocol/private: `__call__`[`L397`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L397), `_grid2mesh_gnn`[`L302`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L302), `_grid2mesh_graph_structure`[`L393`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L393), `_grid_lat`[`L386`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L386), `_grid_lon`[`L387`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L387), `_grid_nodes_lat`[`L389`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L389), `_grid_nodes_lon`[`L390`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L390), `_initialized`[`L377`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L377), `_mesh`[`L298`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L298), `_mesh2grid_gnn`[`L340`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L340), `_mesh2grid_graph_structure`[`L395`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L395), `_mesh_gnn`[`L330`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L330), `_mesh_graph_structure`[`L394`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L394), `_mesh_nodes_lat`[`L382`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L382), `_mesh_nodes_lon`[`L383`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L383), `_norm_conditioning_features`[`L365`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L365), `_num_grid_nodes`[`L388`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L388), `_num_mesh_nodes`[`L381`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L381), `_query_radius`[`L370`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L370), `_spatial_features_kwargs`[`L283`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L283)
- uses (calls/refs, reference-scoped): [`nodes`](typed_graph.md#TypedGraph.nodes), [`TypedGraph`](typed_graph.md#TypedGraph), [`DataArray`](xarray_jax.md#DataArray), [`features`](typed_graph.md#NodeSet.features), [`edges`](typed_graph.md#TypedGraph.edges), [`vertices`](icosahedral_mesh.md#TriangularMesh.vertices), [`edge_key_by_name`](typed_graph.md#TypedGraph.edge_key_by_name), [`features`](typed_graph.md#EdgeSet.features), [`n_node`](typed_graph.md#NodeSet.n_node), [`in_mesh_triangle_indices`](grid_mesh_connectivity.md#in_mesh_triangle_indices), [`faces`](icosahedral_mesh.md#TriangularMesh.faces), [`get_graph_spatial_features`](model_utils.md#get_graph_spatial_features), [`context`](typed_graph.md#TypedGraph.context), [`get_bipartite_graph_spatial_features`](model_utils.md#get_bipartite_graph_spatial_features), [`NodeSet`](typed_graph.md#NodeSet), [`_permute_mesh_to_banded`](denoiser.md#_permute_mesh_to_banded), [`unwrap`](xarray_jax.md#unwrap), [`features`](typed_graph.md#Context.features), [`radius_query_indices`](grid_mesh_connectivity.md#radius_query_indices), [`senders`](typed_graph.md#EdgesIndices.senders), [`EdgesIndices`](typed_graph.md#EdgesIndices), [`indices`](typed_graph.md#EdgeSet.indices), [`receivers`](typed_graph.md#EdgesIndices.receivers), [`unwrap_data`](xarray_jax.md#unwrap_data), [`n_edge`](typed_graph.md#EdgeSet.n_edge), [`n_graph`](typed_graph.md#Context.n_graph), [`EdgeSet`](typed_graph.md#EdgeSet), [`EdgeSetKey`](typed_graph.md#EdgeSetKey), [`latent_size`](denoiser.md#DenoiserArchitectureConfig.latent_size), [`dataset_to_stacked`](model_utils.md#dataset_to_stacked), [`Context`](typed_graph.md#Context), [`faces_to_edges`](icosahedral_mesh.md#faces_to_edges), [`get_last_triangular_mesh_for_sphere`](icosahedral_mesh.md#get_last_triangular_mesh_for_sphere), [`cartesian_to_spherical`](model_utils.md#cartesian_to_spherical), [`spherical_to_lat_lon`](model_utils.md#spherical_to_lat_lon), [`DeepTypedGraphNet`](deep_typed_graph_net.md#DeepTypedGraphNet), [`_add_batch_second_axis`](denoiser.md#_add_batch_second_axis), [`DenoiserArchitectureConfig`](denoiser.md#DenoiserArchitectureConfig), [`_get_max_edge_distance`](denoiser.md#_get_max_edge_distance), [`sparse_transformer_config`](denoiser.md#DenoiserArchitectureConfig.sparse_transformer_config)  (+11 more)
- used by: [`_predictor`](denoiser.md#Denoiser._predictor)

## Functions
- `_add_batch_second_axis(data, batch_size)` — [`L818`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L818)
- `_get_max_edge_distance(mesh)` — [`L825`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L825)
- `_permute_mesh_to_banded(mesh)` — [`L834`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L834) — Permutes the mesh nodes such that adjacency matrix has banded structure.

## Module values
- `Kwargs` — [`L37`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L37)
- `NoiseLevelEncoder` — [`L38`](../../../../../raw/code/graphcast/graphcast/denoiser.py#L38)

