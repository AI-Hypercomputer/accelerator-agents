# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
features = 64
dim1 = 256
dim2 = 256
key = random.PRNGKey(0)
key_x, key_params = random.split(key)
x = random.normal(key_x, (batch_size, features, dim1, dim2))
bn = nn.BatchNorm(use_running_average=False)
variables = bn.init(key_params, x)


# Computation
def kernel(x_ref, scale_ref, bias_ref, out_ref):
  """
  Pallas kernel for Batch Normalization.

  This kernel processes a single feature channel of the input tensor `x`.
  It calculates the mean and variance for that channel, normalizes the data,
  and then applies the corresponding scale and bias parameters.

  Args:
    x_ref: A reference to a slice of the input tensor `x`, corresponding to
      a single feature channel. Shape: (batch_size, 1, dim1, dim2).
    scale_ref: A reference to the scale parameter for the current channel.
      Shape: (1,).
    bias_ref: A reference to the bias parameter for the current channel.
      Shape: (1,).
    out_ref: A reference to the output slice where the result for the
      processed channel should be stored.
  """
  # Pass 1: Compute mean and variance by iterating over tiles.
  # The full slice for a channel is too large to load into SRAM at once.
  sum_val = jnp.zeros((), dtype=jnp.float32)
  sum_sq_val = jnp.zeros((), dtype=jnp.float32)

  def pass1_body(i, carry):
    sum_val, sum_sq_val = carry
    # Load one example's data for this channel. Shape: (1, 1, dim1, dim2)
    x_slice = pl.load(x_ref, (i, 0, 0, 0), slice_shape=(1, 1, dim1, dim2))
    sum_val += jnp.sum(x_slice)
    sum_sq_val += jnp.sum(jnp.square(x_slice))
    return sum_val, sum_sq_val

  sum_val, sum_sq_val = jax.lax.fori_loop(0, batch_size, pass1_body, (sum_val, sum_sq_val))

  # Finalize mean and variance calculations.
  count = batch_size * dim1 * dim2
  mean = sum_val / count
  var = sum_sq_val / count - jnp.square(mean)
  epsilon = 1e-5

  # Load scale and bias parameters.
  scale = scale_ref[0]
  bias = bias_ref[0]
  inv_std = jax.lax.rsqrt(var + epsilon)

  # Pass 2: Apply normalization and store results.
  def pass2_body(i, _):
    # Load the same slice again.
    x_slice = pl.load(x_ref, (i, 0, 0, 0), slice_shape=(1, 1, dim1, dim2))
    # Apply the batch normalization formula.
    normalized_slice = (x_slice - mean) * inv_std * scale + bias
    # Store the result.
    pl.store(out_ref, (i, 0, 0, 0), normalized_slice)

  jax.lax.fori_loop(0, batch_size, pass2_body, None)


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(features,),
  in_specs=[
    # Map each feature channel in x to a kernel instance.
    pl.BlockSpec(block_shape=(batch_size, 1, dim1, dim2), index_map=lambda i: (0, i, 0, 0)),
    # Provide the single scale and bias value for the current feature.
    pl.BlockSpec(block_shape=(1,), index_map=lambda i: (i,)),
    pl.BlockSpec(block_shape=(1,), index_map=lambda i: (i,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(batch_size, 1, dim1, dim2), index_map=lambda i: (0, i, 0, 0)),
)(x, variables["params"]["scale"], variables["params"]["bias"]).block_until_ready()
