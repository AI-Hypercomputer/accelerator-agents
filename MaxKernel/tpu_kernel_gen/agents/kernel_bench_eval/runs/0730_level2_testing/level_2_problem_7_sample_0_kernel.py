# Imports
import jax
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3

key = random.PRNGKey(0)
key, x_key, bias_key, params_key = random.split(key, 4)

# JAX uses channels-last convention: (N, D, H, W, C)
x = random.normal(x_key, (batch_size, depth, height, width, in_channels))
# Bias shape must be broadcastable to the output of the convolution
bias = random.normal(bias_key, (1, 1, 1, 1, out_channels))

conv = nn.Conv(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size, kernel_size),
  padding="SAME",
)
params = conv.init(params_key, x)["params"]
conv_output = conv.apply({"params": params}, x)


# Computation
def kernel(conv_output_ref, bias_ref, out_ref):
  """Pallas kernel for fused activations and bias add.

  This kernel performs the following sequence of operations:
  1. Applies a series of activation functions: ReLU, Leaky ReLU, GELU, Sigmoid.
  2. Adds a final, external bias.
  3. Writes the result to the corresponding output plane.

  Args:
    conv_output_ref: A reference to the input tensor slice, which is the
      output of a preceding convolution.
    bias_ref: A reference to an external bias tensor to be added after all
      activations.
    out_ref: A reference to the output tensor slice where the result is stored.
  """
  # Load data from HBM to registers. The convolution bias is already included
  # in conv_output_ref from the nn.Conv.apply call.
  y = conv_output_ref[...]

  # Apply the sequence of activation functions.
  y = jax.nn.relu(y)
  y = jax.nn.leaky_relu(y, negative_slope=0.01)
  y = jax.nn.gelu(y)
  y = jax.nn.sigmoid(y)

  # Add the final external bias.
  y = y + bias_ref[...]

  # Write the final result to the output buffer.
  out_ref[...] = y


# The Pallas kernel computes a fused operation equivalent to:
# relu -> leaky_relu -> gelu -> sigmoid -> add
# The grid is defined over the batch and depth dimensions. Each kernel instance
# processes a full (height, width) plane of data.
x = pl.pallas_call(
  kernel,
  # The output shape is the same as the convolution's output.
  out_shape=jax.ShapeDtypeStruct((batch_size, depth, height, width, out_channels), x.dtype),
  # Grid iterates over batch and depth.
  grid=(batch_size, depth),
  in_specs=[
    # Input 'conv_output': Each kernel instance (n, d) processes the
    # corresponding plane from the convolution output.
    pl.BlockSpec(
      block_shape=(1, 1, height, width, out_channels),
      index_map=lambda n, d: (n, d, 0, 0, 0),
    ),
    # External bias: The entire bias tensor is needed for the final add.
    pl.BlockSpec(block_shape=bias.shape, index_map=lambda n, d: tuple([0] * bias.ndim)),
  ],
  # Output: Each kernel instance at (n, d) writes to the corresponding
  # output plane at (n, d).
  out_specs=pl.BlockSpec(
    block_shape=(1, 1, height, width, out_channels),
    index_map=lambda n, d: (n, d, 0, 0, 0),
  ),
)(conv_output, bias).block_until_ready()
