# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
# The original PyTorch code uses a Conv layer (which includes a bias by default)
# and then adds a second, separate bias tensor. To replicate this behavior,
# we use the default `use_bias=True` in nn.Conv and create a second bias array.
conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size))
key = random.PRNGKey(0)
key_params, key_bias, key_x = random.split(key, 3)
# Flax's nn.Conv expects channels-last (NHWC) input by default, so we set the shape accordingly.
x_shape = (batch_size, height, width, in_channels)
params = conv.init(key_params, jnp.ones(x_shape))["params"]
# The external bias must be broadcastable to the conv output (N, H', W', C_out).
# The shape (1, 1, 1, C_out) is the NHWC equivalent of PyTorch's (1, C_out, 1, 1).
bias = random.normal(key_bias, (1, 1, 1, out_channels))
x = random.normal(key_x, x_shape)

# Computation
# High-level operations like convolution are not supported inside Pallas kernels.
# The correct approach is to perform the convolution with standard JAX/Lax
# and then use Pallas to fuse the subsequent element-wise operations.
conv_out = jax.lax.conv_general_dilated(
  lhs=x, rhs=params["kernel"], window_strides=(1, 1), padding="SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
)

# Reshape the convolution bias to be explicitly broadcastable. This avoids
# potential issues with implicit broadcasting of different-rank arrays inside
# the Pallas kernel, which can cause compilation errors.
conv_bias = jnp.reshape(params["bias"], (1, 1, 1, out_channels))


def add_biases_and_relu_kernel(conv_out_ref, conv_bias_ref, external_bias_ref, out_ref):
  # Add the first bias (from the convolution layer) to the result.
  with_conv_bias = conv_out_ref[...] + conv_bias_ref[...]

  # Apply the ReLU activation function element-wise.
  # Use jax.lax.relu, as it's the canonical JAX function for transformations.
  activated = jax.lax.relu(with_conv_bias)

  # Add the second, external bias and write the final result to the output ref.
  out_ref[...] = activated + external_bias_ref[...]


# The output shape of the Pallas call is the same as the convolution output.
out_shape = jax.ShapeDtypeStruct(conv_out.shape, conv_out.dtype)

x = pl.pallas_call(
  add_biases_and_relu_kernel,
  out_shape=out_shape,
  grid=(batch_size,),
  in_specs=[
    # A slice of the convolution output for each batch item.
    pl.BlockSpec(block_shape=(1, height, width, out_channels), index_map=lambda i: (i, 0, 0, 0)),
    # The BlockSpec for the reshaped conv_bias tensor.
    pl.BlockSpec(block_shape=(1, 1, 1, out_channels), index_map=lambda i: (0, 0, 0, 0)),
    # The full external bias tensor, shared across all instances.
    pl.BlockSpec(block_shape=(1, 1, 1, out_channels), index_map=lambda i: (0, 0, 0, 0)),
  ],
  # Each kernel instance writes to a corresponding slice in the output.
  out_specs=pl.BlockSpec(block_shape=(1, height, width, out_channels), index_map=lambda i: (i, 0, 0, 0)),
)(conv_out, conv_bias, bias).block_until_ready()
