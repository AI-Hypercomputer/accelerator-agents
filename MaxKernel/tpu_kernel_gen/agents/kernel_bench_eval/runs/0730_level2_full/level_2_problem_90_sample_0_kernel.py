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
depth, height, width = 16, 32, 32
kernel_size = 3
sum_tensor_shape = (1, 1, 1, 1, out_channels)
input_shape = (batch_size, depth, height, width, in_channels)

key = random.PRNGKey(0)
key_x, key_params, key_sum = random.split(key, 3)

x = random.normal(key_x, input_shape)
conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size, kernel_size))
params = conv.init(key_params, x)["params"]
sum_tensor = random.normal(key_sum, sum_tensor_shape)

block_h = height
block_w = width


# Computation
def kernel(x_ref, bias_ref, sum_tensor_ref, out_ref):
  """Pallas kernel for a sequence of element-wise operations.

  This kernel performs the following steps:
  1. Adds a bias term.
  2. Applies a Leaky ReLU activation.
  3. Adds a broadcasted tensor.
  4. Clips the result to the range [-1.0, 1.0].
  5. Applies a GELU activation.
  6. Writes the final result to the output buffer.

  Args:
    x_ref: Input tensor block (result of a previous convolution).
    bias_ref: Convolution bias.
    sum_tensor_ref: Tensor to be added element-wise.
    out_ref: Output tensor block to store the result.
  """
  # The input x_ref is the result of the convolution.
  # 1. Add bias.
  x = x_ref[...] + bias_ref[...]

  # 2. Apply Leaky ReLU activation.
  x = nn.leaky_relu(x, negative_slope=0.2)

  # 3. Add the sum_tensor. Broadcasting handles the shape difference.
  x = x + sum_tensor_ref[...]

  # 4. Clip the values.
  x = jnp.clip(x, a_min=-1.0, a_max=1.0)

  # 5. Apply GELU activation.
  x = nn.gelu(x)

  # 6. Write the final result to the output reference.
  out_ref[...] = x


# Perform the convolution outside of the Pallas kernel.
conv_out = conv.apply({"params": params}, x)

# Reshape bias for easier broadcasting in the kernel.
bias_reshaped = jnp.reshape(params["bias"], (1, 1, 1, 1, out_channels))

# Use pallas_call for the sequence of element-wise operations.
x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, depth, height, width, out_channels), x.dtype),
  grid=(batch_size, depth, height // block_h, width // block_w),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, 1, block_h, block_w, out_channels),
      index_map=lambda i, j, k, l: (i, j, k * block_h, l * block_w, 0),
    ),
    pl.BlockSpec(block_shape=bias_reshaped.shape, index_map=lambda *_: (0,) * 5),
    pl.BlockSpec(block_shape=sum_tensor.shape, index_map=lambda *_: (0,) * 5),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, 1, block_h, block_w, out_channels),
    index_map=lambda i, j, k, l: (i, j, k * block_h, l * block_w, 0),
  ),
)(conv_out, bias_reshaped, sum_tensor).block_until_ready()
