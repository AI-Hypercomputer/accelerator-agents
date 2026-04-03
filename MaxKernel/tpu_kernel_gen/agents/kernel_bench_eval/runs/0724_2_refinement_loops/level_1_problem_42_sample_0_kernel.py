# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
channels = 32
height = 128
width = 128
kernel_size = 2
stride = 2
padding = 1
dilation = 3
key = random.PRNGKey(0)
# JAX uses channels-last convention: (N, H, W, C)
x = random.normal(key, (batch_size, height, width, channels))

# Computation

# Pre-pad the input tensor. The kernel will operate on this padded tensor,
# which simplifies the kernel logic by removing the need for boundary checks.
# The padding value is -inf, the identity element for the max operation.
padded_x = jnp.pad(
  x,
  pad_width=((0, 0), (padding, padding), (padding, padding), (0, 0)),
  mode="constant",
  constant_values=-jnp.inf,
)

# Pre-calculate the output shape. The formula uses the original dimensions.
dilated_kernel_size_h = (kernel_size - 1) * dilation + 1
dilated_kernel_size_w = (kernel_size - 1) * dilation + 1
out_height = (height + 2 * padding - dilated_kernel_size_h) // stride + 1
out_width = (width + 2 * padding - dilated_kernel_size_w) // stride + 1
output_shape = (batch_size, out_height, out_width, channels)


# The Pallas kernel for max pooling on a pre-padded input.
def kernel(x_ref, out_ref):
  """
  Pallas kernel for max pooling on a pre-padded input.

  This kernel processes a horizontal stripe of the padded input tensor to
  produce a corresponding stripe of the output. Since the input is pre-padded,
  no boundary checks are needed inside the kernel.
  """
  # Get the shape of the output block.
  _, out_h, out_w, _ = out_ref.shape

  # Manually implement the max pooling operation with nested loops.
  for i in range(out_h):
    for j in range(out_w):
      # Calculate window start in the padded input block.
      h_start = i * stride
      w_start = j * stride

      # Find the maximum value in the window.
      max_val = jnp.full((channels,), -jnp.inf, dtype=x_ref.dtype)
      for kh in range(kernel_size):
        for kw in range(kernel_size):
          # Access is guaranteed to be in-bounds due to pre-padding.
          val = x_ref[0, h_start + kh * dilation, w_start + kw * dilation, :]
          max_val = lax.max(max_val, val)

      # Write the result to the output.
      out_ref[0, i, j, :] = max_val


# Define blocking strategy. We parallelize over the batch dimension and blocks
# of the output height dimension.
out_block_h = 16
grid_h = (out_height + out_block_h - 1) // out_block_h
grid = (batch_size, grid_h)

# For each output block, we need to load a corresponding input block from the
# *padded* input tensor.
in_block_h = (out_block_h - 1) * stride + dilated_kernel_size_h
# The width of the input block is the full width of the padded tensor.
in_block_w = padded_x.shape[2]

result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=grid,
  in_specs=[
    pl.BlockSpec(
      # Block shape for the padded input
      block_shape=(1, in_block_h, in_block_w, channels),
      # Index map calculates the start of the input block based on the
      # output block index. No negative offsets or boundary checks needed.
      index_map=lambda n, i: (n, i * out_block_h * stride, 0, 0),
    )
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, out_block_h, out_width, channels),
    index_map=lambda n, i: (n, i * out_block_h, 0, 0),
  ),
)(padded_x).block_until_ready()
