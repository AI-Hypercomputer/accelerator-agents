# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
# PyTorch padding=1, stride=2, kernel_size=3, output_padding=1 for ConvTranspose
# is equivalent to 'SAME' padding in JAX/Flax for calculating output shape.
padding = "SAME"
multiplier_shape = (1, 1, 1, out_channels)

key = random.PRNGKey(0)
key_x, key_multiplier, key_conv = random.split(key, 3)

# JAX/Flax uses channels-last convention: (N, D, H, W, C)
x = random.normal(key_x, (batch_size, depth, height, width, in_channels))
conv_transpose = nn.ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size, kernel_size),
  strides=(stride, stride, stride),
  padding=padding,
)
params = conv_transpose.init(key_conv, x)["params"]
multiplier = random.normal(key_multiplier, multiplier_shape)

# Pre-computation outside Pallas
# The conv_transpose and reduce_window primitives are not supported in Pallas on TPU.
# We perform these operations using standard JAX before and after the Pallas kernel.
y = conv_transpose.apply({"params": params}, x)


# Computation
def kernel(x_ref, multiplier_ref, out_ref):
  """
  Pallas kernel for a sequence of neural network operations.

  This kernel processes a single item from a batch and applies the following
  sequence of operations:
  1. Leaky ReLU activation.
  2. Element-wise multiplication.
  3. Leaky ReLU activation.

  Args:
    x_ref: Reference to the input tensor for a single batch item,
      which is the result of a preceding transposed convolution.
      Shape: (1, out_depth, out_height, out_width, out_channels)
    multiplier_ref: Reference to the multiplier tensor.
      Shape: (1, 1, 1, out_channels)
    out_ref: Reference to the output tensor for a single batch item.
      This is where the result is written.
      Shape: (1, out_depth, out_height, out_width, out_channels)
  """
  negative_slope = 0.2

  # Load the input data from memory.
  x = x_ref[...]

  # 1. Apply the first Leaky ReLU activation.
  x = jnp.maximum(x, x * negative_slope)

  # 2. Apply element-wise multiplication.
  x = x * multiplier_ref[...]

  # 3. Apply the second Leaky ReLU activation.
  x = jnp.maximum(x, x * negative_slope)

  # Write the final result to the output buffer.
  out_ref[...] = x


# The output shape of the conv_transpose operation.
# With stride=2 and padding='SAME', the spatial dimensions are doubled.
y_shape = (
  batch_size,
  depth * stride,
  height * stride,
  width * stride,
  out_channels,
)

# Define tiling strategy to fit into SRAM.
# We tile along the depth dimension to reduce the memory footprint of each block.
num_depth_chunks = 16
depth_chunk_size = y_shape[1] // num_depth_chunks

# The pallas_call will handle the element-wise operations.
x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(shape=y_shape, dtype=x.dtype),
  # The grid now iterates over both the batch and the depth chunks.
  grid=(batch_size, num_depth_chunks),
  in_specs=[
    # Input 'y', chunked along the batch and depth dimensions.
    pl.BlockSpec(
      block_shape=(
        1,
        depth_chunk_size,
        y_shape[2],
        y_shape[3],
        y_shape[4],
      ),
      index_map=lambda i, j: (i, j * depth_chunk_size, 0, 0, 0),
    ),
    # Multiplier, not chunked. It will be broadcasted.
    pl.BlockSpec(
      block_shape=multiplier.shape,
      index_map=lambda i, j: (0,) * multiplier.ndim,
    ),
  ],
  out_specs=pl.BlockSpec(
    # The output is chunked similarly to the input.
    block_shape=(
      1,
      depth_chunk_size,
      y_shape[2],
      y_shape[3],
      y_shape[4],
    ),
    index_map=lambda i, j: (i, j * depth_chunk_size, 0, 0, 0),
  ),
)(y, multiplier)

# Apply 3D max pooling after the Pallas kernel.
# The window shape and strides are (2, 2, 2) for the spatial dimensions,
# which halves the spatial resolution. The '1's correspond to the batch
# and channel dimensions, over which we do not pool.
window_dimensions = (1, 2, 2, 2, 1)
strides = (1, 2, 2, 2, 1)
x = jax.lax.reduce_window(x, -jnp.inf, jnp.maximum, window_dimensions, strides, "VALID")
x.block_until_ready()
