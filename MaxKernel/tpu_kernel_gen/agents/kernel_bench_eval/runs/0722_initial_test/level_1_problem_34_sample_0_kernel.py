# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
features = 64
dim1 = 256
dim2 = 256
key = random.PRNGKey(0)
key_x, key_init = random.split(key)
x = random.normal(key_x, (batch_size, features, dim1, dim2))
inorm = nn.InstanceNorm()
dummy_x_nhwc = jnp.empty((batch_size, dim1, dim2, features), dtype=x.dtype)
variables = inorm.init(key_init, dummy_x_nhwc)


# Computation
def kernel(x_ref, variables_ref, out_ref):
  """
  Pallas kernel for instance normalization.

  This kernel computes instance normalization for each feature map independently.
  The grid is set up to launch one program per batch element and feature.
  """
  # Get the program ID for the feature dimension.
  feature_idx = pl.program_id(axis=1)

  # Load the input data for the current program. This corresponds to a single
  # feature map (slice over H and W) for a specific batch and channel.
  # The shape of x_ref is (1, 1, dim1, dim2).
  feature_map = x_ref[...]

  # Calculate the mean and variance across the spatial dimensions (H, W).
  mean = jnp.mean(feature_map)
  var = jnp.var(feature_map)

  # Epsilon for numerical stability, matching the Flax default.
  epsilon = 1e-6

  # Normalize the feature map.
  normalized_map = (feature_map - mean) / jnp.sqrt(var + epsilon)

  # The entire scale and bias vectors are loaded. We dynamically select
  # the element corresponding to the current feature index.
  scale = variables_ref["params"]["scale"][feature_idx]
  bias = variables_ref["params"]["bias"][feature_idx]

  # Apply scale and bias.
  result = normalized_map * scale + bias

  # Write the final result to the output reference.
  out_ref[...] = result


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size, features),
  in_specs=[
    pl.BlockSpec(block_shape=(1, 1, dim1, dim2), index_map=lambda i, j: (i, j, 0, 0)),
    {
      "params": {
        "scale": pl.BlockSpec(block_shape=(features,), index_map=lambda i, j: (0,)),
        "bias": pl.BlockSpec(block_shape=(features,), index_map=lambda i, j: (0,)),
      }
    },
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, dim1, dim2), index_map=lambda i, j: (i, j, 0, 0)),
)(x, variables).block_until_ready()
