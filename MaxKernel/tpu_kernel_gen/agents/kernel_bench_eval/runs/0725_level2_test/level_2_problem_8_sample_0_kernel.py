# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas.ops.tpu import conv_general_dilated

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
sum_dim = -1
key = random.PRNGKey(0)
key, x_key, bias_key, params_key = random.split(key, 4)
x = random.normal(x_key, (batch_size, depth, height, width, in_channels))
conv = nn.Conv(features=out_channels, kernel_size=kernel_size)
params = conv.init(params_key, x)["params"]
bias = random.normal(bias_key, (1, 1, 1, 1, out_channels))


# Computation
def kernel(x_ref, kernel_ref, conv_bias_ref, divisor_ref, final_bias_ref, out_ref):
  # Define constants based on the source computation
  pool_size = (2, 2, 2)
  sum_dim = -1

  # Load data for the current program instance.
  # Pallas handles moving data from HBM to SRAM based on BlockSpecs.
  # Accessing the ref (e.g., x_ref[...]) loads it into registers for computation.
  x = x_ref[...]
  conv_kernel = kernel_ref[...]
  conv_bias = conv_bias_ref[...]
  divisor = divisor_ref[0]  # Extract scalar from the 1-element array
  final_bias = final_bias_ref[...]

  # 1. Perform 3D convolution.
  # This corresponds to `conv.apply({'params': params}, x)`.
  # We use the Pallas-specific primitive for TPUs.
  # The input format is 'NDHWC' (batch, depth, height, width, channels).
  # The kernel format is 'DHWIO' (depth, height, width, in_channels, out_channels).
  dn = ("NDHWC", "DHWIO", "NDHWC")
  x = conv_general_dilated(x, conv_kernel, window_strides=(1, 1, 1), padding="SAME", dimension_numbers=dn)
  x = x + conv_bias

  # 2. Divide by the divisor.
  x = x / divisor

  # 3. Apply max pooling.
  # This corresponds to `nn.max_pool`. We use `lax.reduce_window`.
  # The window shape and strides must account for the batch and channel dimensions,
  # which are not pooled over.
  pool_window_shape = (1, *pool_size, 1)
  pool_strides = (1, *pool_size, 1)
  x = lax.reduce_window(x, -jnp.inf, lax.max, pool_window_shape, pool_strides, "VALID")

  # 4. Compute the mean over the spatial dimensions.
  x = jnp.mean(x, axis=(1, 2, 3), keepdims=True)

  # 5. Add the final bias.
  x = x + final_bias

  # 6. Sum over the final (channel) dimension.
  result = jnp.sum(x, axis=sum_dim)

  # 7. Write the final result to the output buffer.
  # The shape of `result` is (1, 1, 1, 1), which matches the out_ref block shape.
  out_ref[...] = result


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, 1, 1, 1), x.dtype),
  grid=(batch_size,),
  in_specs=[
    # Input image batch
    pl.BlockSpec(
      block_shape=(1, depth, height, width, in_channels),
      index_map=lambda i: (i, 0, 0, 0, 0),
    ),
    # Conv kernel weights (shared across all instances)
    pl.BlockSpec(
      block_shape=params["kernel"].shape,
      index_map=lambda i: (0,) * params["kernel"].ndim,
    ),
    # Conv bias (shared across all instances)
    pl.BlockSpec(
      block_shape=params["bias"].shape,
      index_map=lambda i: (0,) * params["bias"].ndim,
    ),
    # Divisor (shared scalar, wrapped in an array)
    pl.BlockSpec(block_shape=(1,), index_map=lambda i: (0,)),
    # Final bias (shared across all instances)
    pl.BlockSpec(block_shape=bias.shape, index_map=lambda i: (0,) * bias.ndim),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, 1, 1), index_map=lambda i: (i, 0, 0, 0)),
)(
  x,
  params["kernel"],
  params["bias"],
  jnp.array([divisor]),
  bias,
).block_until_ready()
