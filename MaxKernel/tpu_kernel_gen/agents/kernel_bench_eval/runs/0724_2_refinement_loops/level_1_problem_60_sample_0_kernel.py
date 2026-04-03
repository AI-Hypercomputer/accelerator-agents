# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax.linen import Conv
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)
width = 64
height = 64
depth = 64

key = random.PRNGKey(0)
key_x, key_init = random.split(key)

# Flax Conv expects input shape (N, spatial_dims..., C) by default.
# To match PyTorch's (N, C, ...spatial_dims) we need to transpose.
x_shape = (batch_size, in_channels, depth, height, width)
x = random.normal(key_x, x_shape)


# The Conv layer needs to be told to expect channels-first data.
# This is done by setting the `data_format` attribute.
class Conv3DChannelsFirst(Conv):
  def setup(self):
    self.data_format = "NCDHW"
    super().setup()


conv3d = Conv3DChannelsFirst(features=out_channels, kernel_size=kernel_size, padding="VALID", use_bias=False)
params = conv3d.init(key_init, x)["params"]


# Computation
def kernel(x_ref, w_ref, out_ref):
  """
  Pallas kernel for 3D convolution with 'VALID' padding.

  This kernel computes a 2D slice of the spatial output for a given batch
  item, output channel, and spatial depth position 'd'. The grid for the
  pallas_call is responsible for iterating over batch, output channels, and
  the depth dimension.

  Args:
    x_ref: A reference to the input tensor slice for a single batch item.
           Shape: (1, in_channels, depth, height, width)
    w_ref: A reference to the kernel weights for a single output channel.
           The weights have been pre-processed to have the shape
           (1, in_channels, kernel_depth, kernel_height, kernel_width).
    out_ref: A reference to the output tensor slice to be populated.
             Shape: (1, 1, 1, out_height, out_width)
  """
  # Get the spatial depth index for this kernel instance from the grid.
  d = pl.program_id(axis=2)

  # Extract kernel dimensions from the weight reference shape.
  _, in_channels, kernel_depth, kernel_height, kernel_width = w_ref.shape

  # Extract output spatial dimensions from the output reference shape.
  _, _, _, out_height, out_width = out_ref.shape

  # Load the full input and weight blocks into SRAM.
  x = x_ref[:]
  w = w_ref[:]

  # Iterate over the height and width dimensions to compute a 2D slice.
  for h in range(out_height):
    for w_idx in range(out_width):
      # Extract the input patch corresponding to the current output position.
      # The patch starts at (d, h, w_idx) in the input tensor `x`.
      input_patch = jax.lax.dynamic_slice(
        x,
        start_indices=(0, 0, d, h, w_idx),
        slice_sizes=(1, in_channels, kernel_depth, kernel_height, kernel_width),
      )

      # Compute the convolution for this output pixel (dot product).
      accumulator = jnp.sum(input_patch * w)

      # Write the result to the output slice.
      # The output reference is a slice of shape (1, 1, 1, out_height, out_width).
      out_ref[0, 0, 0, h, w_idx] = accumulator


# The kernel weights are stored in the 'params' dictionary.
w = params["kernel"]

# For 'VALID' padding, the output spatial dimensions are calculated as:
# output_dim = input_dim - kernel_dim + 1
out_depth = x.shape[2] - w.shape[0] + 1
out_height = x.shape[3] - w.shape[1] + 1
out_width = x.shape[4] - w.shape[2] + 1

# The kernel expects weights in (C_in, KD, KH, KW) layout for each output
# channel. We transpose the Flax kernel from (KD, KH, KW, C_in, C_out) to
# (C_out, C_in, KD, KH, KW) before passing it to the pallas_call.
w_permuted = jnp.transpose(w, (4, 3, 0, 1, 2))

# The grid is defined over batch, output channels, and the depth dimension.
# Each kernel instance computes one 2D slice (h, w) of the output feature map.
grid = (batch_size, out_channels, out_depth)

output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_channels, out_depth, out_height, out_width), x.dtype),
  grid=grid,
  in_specs=[
    # Input tensor 'x': Each kernel instance gets the full input volume
    # for its batch item 'n'.
    pl.BlockSpec(block_shape=(1, in_channels, depth, height, width), index_map=lambda n, c_out, d: (n, 0, 0, 0, 0)),
    # Permuted kernel weights 'w_permuted': Each kernel instance gets the
    # full filter for its output channel 'c_out'.
    pl.BlockSpec(
      block_shape=(1, in_channels, w.shape[0], w.shape[1], w.shape[2]),
      index_map=lambda n, c_out, d: (c_out, 0, 0, 0, 0),
    ),
  ],
  out_specs=pl.BlockSpec(
    # Each kernel computes one 2D slice of the output, corresponding to its
    # (n, c_out, d) indices.
    block_shape=(1, 1, 1, out_height, out_width),
    index_map=lambda n, c_out, d: (n, c_out, d, 0, 0),
  ),
)(x, w_permuted).block_until_ready()
