# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax.linen import ConvTranspose
from jax import lax
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
height_in = 16
width_in = 32

key = random.PRNGKey(0)
key_params, key_x = random.split(key)

# Note: JAX uses NHWC channel ordering by default
x_shape = (batch_size, height_in, width_in, in_channels)
x = random.normal(key_x, x_shape)

conv_transpose2d = ConvTranspose(
  features=out_channels,
  kernel_size=kernel_size,
  strides=(1, 1),
  padding="VALID",
  use_bias=False,
)
params = conv_transpose2d.init(key_params, x)["params"]

# Define a block size for parallelizing over the output channel dimension
block_c = 8

# The output height and width are calculated based on 'VALID' padding and stride (1,1):
# height_out = (height_in - 1) * stride + kernel_h = (16 - 1) * 1 + 3 = 18
# width_out = (width_in - 1) * stride + kernel_w = (32 - 1) * 1 + 5 = 36
height_out = 18
width_out = 36

# Transpose kernel for TPU layout compatibility.
# Original kernel shape from Flax: (KH, KW, C_in, C_out)
# We transpose it to (C_out, KH, KW, C_in) so that the last two dimensions
# (KW, C_in) are not sliced, satisfying TPU memory layout constraints.
kernel_transposed = params["kernel"].transpose(3, 0, 1, 2)


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 2D transposed convolution.

  This kernel implements the transposed convolution operation by iterating through
  each pixel of the input feature map. For each input pixel, it computes its
  contribution to the output by performing a dot product with the kernel weights
  and adds this contribution to the appropriate region in the output buffer.

  Args:
    x_ref: A reference to the input feature map for one batch item.
      Shape: (1, height_in, width_in, in_channels)
    kernel_ref: A reference to a slice of the transposed convolution kernel weights.
      Shape: (block_c, kernel_h, kernel_w, in_channels)
    out_ref: A reference to the output buffer slice to be written to.
      This kernel uses an NCHW layout for the output.
      Shape: (1, block_c, height_out, width_out)
  """
  # Get dimensions from the shapes of the input references.
  _, height_in, width_in, in_channels = x_ref.shape
  # The kernel_ref is a slice of the transposed kernel, with shape (block_c, KH, KW, C_in).
  _, kernel_h, kernel_w, _ = kernel_ref.shape

  # Load the kernel block from memory. This is necessary because JAX operations
  # like `tensordot` expect array-like values, not memory references.
  kernel_block = kernel_ref[...]

  # Initialize the output block for this program to zeros.
  out_ref[...] = jnp.zeros_like(out_ref)

  # Iterate over each spatial location (h, w) in the input feature map.
  for h in range(height_in):
    for w in range(width_in):
      # Get the feature vector for the input pixel at (h, w).
      x_pixel = x_ref[0, h, w, :]

      # Reshape x_pixel to be broadcastable for a batch dot product.
      x_pixel_bcast = x_pixel.reshape(1, 1, 1, in_channels)

      # Perform a batch dot product. The batch dimensions are (block_c, kernel_h, kernel_w).
      # For each element in the "batch", we contract the in_channels dimension.
      # The result, `contribution`, will have the shape (block_c, kernel_h, kernel_w).
      contribution = lax.dot_general(
        x_pixel_bcast,
        kernel_block,
        dimension_numbers=(([3], [3]), ([0, 1, 2], [0, 1, 2])),
      )

      # Add this contribution to the corresponding region in the NCHW output.
      out_ref[0, :, h : h + kernel_h, w : w + kernel_w] += contribution


output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_channels, height_out, width_out), x.dtype),
  grid=(batch_size, out_channels // block_c),
  in_specs=[
    # Input feature map 'x': Each kernel instance (i, j) gets the i-th batch item.
    pl.BlockSpec(
      block_shape=(1, height_in, width_in, in_channels),
      index_map=lambda i, j: (i, 0, 0, 0),
    ),
    # Transposed kernel: Each instance (i, j) gets the j-th block of output channels.
    # The kernel is transposed to (C_out, KH, KW, C_in) to meet TPU requirements.
    pl.BlockSpec(
      block_shape=(block_c, *kernel_size, in_channels),
      index_map=lambda i, j: (j, 0, 0, 0),
    ),
  ],
  out_specs=pl.BlockSpec(
    # Output: Each kernel instance (i, j) writes to the i-th batch item and j-th channel block.
    # The output layout is NCHW to satisfy TPU memory layout constraints.
    block_shape=(1, block_c, height_out, width_out),
    index_map=lambda i, j: (i, j, 0, 0),
  ),
)(x, kernel_transposed).block_until_ready()
