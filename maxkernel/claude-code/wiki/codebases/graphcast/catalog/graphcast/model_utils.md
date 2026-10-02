---
title: 'Module: graphcast/model_utils.py'
type: catalog
provenance: extracted
module: graphcast/model_utils.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.model_utils`/
symbols:
  get_relative_position_in_receiver_local_coordinates: get_relative_position_in_receiver_local_coordinates().
  get_bipartite_relative_position_in_receiver_local_coordinates: get_bipartite_relative_position_in_receiver_local_coordinates().
  get_graph_spatial_features: get_graph_spatial_features().
  get_bipartite_graph_spatial_features: get_bipartite_graph_spatial_features().
  NumpyInterface: NumpyInterface.
  spherical_to_cartesian: spherical_to_cartesian().
  lat_lon_to_cartesian: lat_lon_to_cartesian().
  cartesian_to_lat_lon: cartesian_to_lat_lon().
  dataset_to_stacked: dataset_to_stacked().
  get_rotation_matrices_to_local_coordinates: get_rotation_matrices_to_local_coordinates().
  lat_lon_deg_to_spherical: lat_lon_deg_to_spherical().
  rotate_with_matrices: rotate_with_matrices().
  spherical_to_lat_lon: spherical_to_lat_lon().
  cartesian_to_spherical: cartesian_to_spherical().
  TransformInterface: TransformInterface.
  get_graph_spatial_features.sine_cosine_transform: get_graph_spatial_features().sine_cosine_transform().
  lat_lon_to_leading_axes: lat_lon_to_leading_axes().
  restore_leading_axes: restore_leading_axes().
  stacked_to_dataset: stacked_to_dataset().
  variable_to_stacked: variable_to_stacked().
  fourier_features: fourier_features().
---
# Module: [`graphcast/model_utils.py`](../../../../../raw/code/graphcast/graphcast/model_utils.py)

## Functions
- `cartesian_to_lat_lon(x: np.ndarray, y: np.ndarray, z: np.ndarray, np_: NumpyInterface = np)` — [`L225`](../../../../../raw/code/graphcast/graphcast/model_utils.py#L225)
- `cartesian_to_spherical(x: np.ndarray, y: np.ndarray, z: np.ndarray, np_: NumpyInterface = np)` — [`L196`](../../../../../raw/code/graphcast/graphcast/model_utils.py#L196)
- `dataset_to_stacked(dataset: xarray.Dataset, sizes: Optional[Mapping[str, int]] = None, preserved_dims: Tuple[str, ...] = ("batch", "lat", "lon"))` — [`L675`](../../../../../raw/code/graphcast/graphcast/model_utils.py#L675) — Converts an xarray.Dataset to a single stacked array.
- `fourier_features(values: jnp.ndarray, base_period: float, num_frequencies: int)` — [`L777`](../../../../../raw/code/graphcast/graphcast/model_utils.py#L777) — Maps values to sin/cos features for a range of frequencies.
- `get_bipartite_graph_spatial_features(*, senders_node_lat: np.ndarray, senders_node_lon: np.ndarray, senders: np.ndarray, receivers_node_lat: np.ndarray, receivers_node_lon: np.ndarray, receivers: np.ndarray, add_node_positions: bool, add_node_latitude: bool, add_node_longitude: bool, add_relative_positions: bool, edge_normalization_factor: Optional[float] = None, relative_longitude_local_coordinates: bool, relative_latitude_local_coordinates: bool)` — [`L404`](../../../../../raw/code/graphcast/graphcast/model_utils.py#L404) — Computes spatial features for the nodes. — documented in [graphcast](../../concepts/graphcast.md)
- `get_bipartite_relative_position_in_receiver_local_coordinates(senders_node_phi: np.ndarray, senders_node_theta: np.ndarray, senders: np.ndarray, receivers_node_phi: np.ndarray, receivers_node_theta: np.ndarray, receivers: np.ndarray, latitude_local_coordinates: bool, longitude_local_coordinates: bool, np_: NumpyInterface = np, transform_: TransformInterface = transform)` — [`L545`](../../../../../raw/code/graphcast/graphcast/model_utils.py#L545) — Returns relative position features for the edges.
- `get_graph_spatial_features(*, node_lat: np.ndarray, node_lon: np.ndarray, senders: np.ndarray, receivers: np.ndarray, add_node_positions: bool, add_node_latitude: bool, add_node_longitude: bool, add_relative_positions: bool, edge_normalization_factor: Optional[float] = None, relative_longitude_local_coordinates: bool, relative_latitude_local_coordinates: bool, sine_cosine_encoding: bool = False, encoding_num_freqs: int = 10, encoding_multiplicative_factor: float = 1.2)` — [`L27`](../../../../../raw/code/graphcast/graphcast/model_utils.py#L27) — Computes spatial features for the nodes. — documented in [graphcast](../../concepts/graphcast.md)
- `get_relative_position_in_receiver_local_coordinates(node_phi: np.ndarray, node_theta: np.ndarray, senders: np.ndarray, receivers: np.ndarray, latitude_local_coordinates: bool, longitude_local_coordinates: bool, np_: NumpyInterface = np, transform_: TransformInterface = transform)` — [`L235`](../../../../../raw/code/graphcast/graphcast/model_utils.py#L235) — Returns relative position features for the edges.
- `get_rotation_matrices_to_local_coordinates(reference_phi: np.ndarray, reference_theta: np.ndarray, rotate_latitude: bool, rotate_longitude: bool, np_: NumpyInterface = np, transform_: TransformInterface = transform)` — [`L320`](../../../../../raw/code/graphcast/graphcast/model_utils.py#L320) — Returns a rotation matrix to rotate to a point based on a reference vector.
- `lat_lon_deg_to_spherical(node_lat: np.ndarray, node_lon: np.ndarray, np_: NumpyInterface = np)` — [`L178`](../../../../../raw/code/graphcast/graphcast/model_utils.py#L178)
- `lat_lon_to_cartesian(lat: np.ndarray, lon: np.ndarray, np_: NumpyInterface = np)` — [`L217`](../../../../../raw/code/graphcast/graphcast/model_utils.py#L217)
- `lat_lon_to_leading_axes(grid_xarray: xarray.DataArray)` — [`L153`](../../../../../raw/code/graphcast/graphcast/model_utils.py#L153) — Reorders xarray so lat/lon axes come first.
- `restore_leading_axes(grid_xarray: xarray.DataArray)` — [`L162`](../../../../../raw/code/graphcast/graphcast/model_utils.py#L162) — Reorders xarray so batch/time/level axes come first (if present).
- `rotate_with_matrices(rotation_matrices: np.ndarray, positions: np.ndarray, np_: NumpyInterface = np)` — [`L399`](../../../../../raw/code/graphcast/graphcast/model_utils.py#L399)
- `sine_cosine_transform(x: np.ndarray)` — [`L139`](../../../../../raw/code/graphcast/graphcast/model_utils.py#L139)
- `spherical_to_cartesian(phi: np.ndarray, theta: np.ndarray, np_: NumpyInterface = np)` — [`L207`](../../../../../raw/code/graphcast/graphcast/model_utils.py#L207)
- `spherical_to_lat_lon(phi: np.ndarray, theta: np.ndarray, np_: NumpyInterface = np)` — [`L187`](../../../../../raw/code/graphcast/graphcast/model_utils.py#L187)
- `stacked_to_dataset(stacked_array: xarray.Variable, template_dataset: xarray.Dataset, preserved_dims: Tuple[str, ...] = ("batch", "lat", "lon"))` — [`L711`](../../../../../raw/code/graphcast/graphcast/model_utils.py#L711) — The inverse of dataset_to_stacked.
- `variable_to_stacked(variable: xarray.Variable, sizes: Mapping[str, int], preserved_dims: Tuple[str, ...] = ("batch", "lat", "lon"))` — [`L643`](../../../../../raw/code/graphcast/graphcast/model_utils.py#L643) — Converts an xarray.Variable to preserved_dims + ("channels",).

## Module values
- `NumpyInterface` — [`L23`](../../../../../raw/code/graphcast/graphcast/model_utils.py#L23)
- `TransformInterface` — [`L24`](../../../../../raw/code/graphcast/graphcast/model_utils.py#L24)

