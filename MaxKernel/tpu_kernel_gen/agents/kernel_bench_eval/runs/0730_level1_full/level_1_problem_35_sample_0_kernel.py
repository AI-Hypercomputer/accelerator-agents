# Imports
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
features = 64
num_groups = 8
dim1 = 256
dim2 = 256

key = random.PRNGKey(0)
key_x, key_init = random.split(key)

x = random.normal(key_x, (batch_size, dim1, dim2, features))
gn = nn.GroupNorm(num_groups=num_groups)
params = gn.init(key_init, x)["params"]


def kernel(x_ref, scale_ref, bias_ref, out_ref):
  """Pallas kernel for Group Normalization.

  This kernel applies group normalization to a single element of a batch.
  The grid for the pallas_call is expected to be the batch size, so each
  invocation of this kernel handles one item from the batch.

  Args:
    x_ref: A reference to the input tensor for a single batch item.
      Expected shape: (1, dim1, dim2, features).
    scale_ref: A reference to the scale parameter (gamma).
      Expected shape: (features,).
    bias_ref: A reference to the bias parameter (beta).
      Expected shape: (features,).
    out_ref: A reference to the output tensor. The result of the
      computation will be written here.
  """
  # These constants are derived from the nn.GroupNorm layer initialization
  # in the source code.
  num_groups = 8
  epsilon = 1e-5

  # Load the data from device memory (HBM) into SRAM.
  # x_ref has shape (1, dim1, dim2, features)
  x = x_ref[...]
  scale = scale_ref[...]
  bias = bias_ref[...]

  # Squeeze the batch dimension to simplify calculations.
  # Shape becomes (dim1, dim2, features).
  x_squeezed = jnp.squeeze(x, axis=0)
  features = x_squeezed.shape[-1]
  group_size = features // num_groups

  # Pass 1: Normalize each group and write directly to the output reference.
  # This avoids creating an intermediate tensor and using the unsupported
  # .at[...].set(...) scatter operation.
  for g in range(num_groups):
    # Define the slice for the current group's features
    start_feature = g * group_size
    end_feature = (g + 1) * group_size

    # Extract the data for the current group
    # Shape: (dim1, dim2, group_size)
    group_data = x_squeezed[..., start_feature:end_feature]

    # Calculate mean and variance for the current group across all dimensions
    mean = jnp.mean(group_data)
    var = jnp.var(group_data)

    # Normalize the data for the current group
    normalized_group_data = (group_data - mean) / jnp.sqrt(var + epsilon)

    # Place the normalized group into the corresponding slice of the output tensor.
    # This is a direct store operation, which is supported.
    out_ref[0, :, :, start_feature:end_feature] = normalized_group_data

  # Pass 2: Apply scale and bias to the entire normalized tensor.
  # First, load the normalized data from out_ref.
  x_normalized = out_ref[...]
  # Then, apply the element-wise affine transformation.
  result = x_normalized * scale + bias
  # Finally, write the final result back to out_ref.
  out_ref[...] = result


# Computation
result = pl.pallas_call(
  kernel,
  out_shape=x,
  grid=(x.shape[0],),
  in_specs=[
    pl.BlockSpec(block_shape=(1, x.shape[1], x.shape[2], x.shape[3]), index_map=lambda i: (i, 0, 0, 0)),
    pl.BlockSpec(block_shape=(x.shape[3],), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(x.shape[3],), index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, x.shape[1], x.shape[2], x.shape[3]), index_map=lambda i: (i, 0, 0, 0)),
)(x, params["scale"], params["bias"]).block_until_ready()
