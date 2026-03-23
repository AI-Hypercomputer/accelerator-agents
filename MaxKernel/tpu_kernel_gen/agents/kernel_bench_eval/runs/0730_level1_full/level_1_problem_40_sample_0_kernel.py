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
# In Flax, LayerNorm takes feature_axes to specify normalization dimensions.
# For an input of shape (batch, features, dim1, dim2), PyTorch's
# normalized_shape=(features, dim1, dim2) normalizes over the last 3 axes.
ln = nn.LayerNorm(feature_axes=(1, 2, 3))
key = random.PRNGKey(0)
key_params, key_x = random.split(key)
x = random.normal(key_x, (batch_size, features, dim1, dim2))
params = ln.init(key_params, x)["params"]


# Computation
def kernel(x_ref, scale_ref, bias_ref, out_ref):
  """Pallas kernel for Layer Normalization.

  This kernel applies layer normalization to a slice of the input tensor.
  It computes the mean and variance over the feature dimensions, normalizes
  the input, and then applies a learned scale and bias.

  Args:
    x_ref: A reference to a slice of the input tensor. The slice corresponds
      to one element in the batch, with shape (1, features, dim1, dim2).
    scale_ref: A reference to the entire scale parameter tensor (gamma).
    bias_ref: A reference to the entire bias parameter tensor (beta).
    out_ref: A reference to the output buffer where the result is written
      in-place.
  """
  # Epsilon for numerical stability, matching Flax's default.
  epsilon = 1e-6

  # Load the input slice from HBM into SRAM.
  x = x_ref[0]

  # Calculate the mean and variance across all axes of the input slice.
  mean = jnp.mean(x)
  var = jnp.var(x)

  # Normalize the input slice.
  x_normalized = (x - mean) / jnp.sqrt(var + epsilon)

  # Apply the learned scale and bias parameters.
  # scale_ref[...] and bias_ref[...] load the parameters from HBM.
  y = x_normalized * scale_ref[...] + bias_ref[...]

  # Write the final result to the output buffer.
  out_ref[0] = y


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(1, features, dim1, dim2), index_map=lambda i: (i, 0, 0, 0)),
    pl.BlockSpec(block_shape=(features, dim1, dim2), index_map=lambda i: (0, 0, 0)),
    pl.BlockSpec(block_shape=(features, dim1, dim2), index_map=lambda i: (0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, features, dim1, dim2), index_map=lambda i: (i, 0, 0, 0)),
)(x, params["scale"], params["bias"]).block_until_ready()
