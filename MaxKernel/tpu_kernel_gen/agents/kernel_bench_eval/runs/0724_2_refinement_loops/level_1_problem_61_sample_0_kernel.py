# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
depth = 32
height = 32
width = 32
stride = 1
padding = "VALID"
bias = False

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

x = random.normal(key_x, (batch_size, depth, height, width, in_channels))

conv_transpose3d = nn.ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size, kernel_size),
  strides=(stride, stride, stride),
  padding=padding,
  use_bias=bias,
)

params = conv_transpose3d.init(key_params, x)["params"]

# Define block sizes for tiling the input's spatial dimensions
block_d, block_h, block_w = 8, 8, 8

# Calculate the shape of the output tensor based on transposed convolution rules
out_depth = (depth - 1) * stride + kernel_size
out_height = (height - 1) * stride + kernel_size
out_width = (width - 1) * stride + kernel_size
output_shape = (batch_size, out_depth, out_height, out_width, out_channels)


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 3D transposed convolution.

  This kernel implements transposed convolution as a "scatter" operation. Each
  program instance processes a block of the input tensor `x`. For each element
  in its input block, it calculates the corresponding contribution to the output
  by scaling the convolution kernel. These contributions are then atomically
  added to the correct locations in the output tensor.

  Args:
    x_ref: A reference to a block of the input tensor.
    kernel_ref: A reference to the full convolution kernel tensor.
    out_ref: A reference to the full output tensor.
  """
  # Each program instance handles a block of the input `x`. The program IDs
  # correspond to the grid dimensions: (batch, depth_block, height_block, width_block).
  b = pl.program_id(0)
  i = pl.program_id(1)
  j = pl.program_id(2)
  k = pl.program_id(3)

  # Get kernel and block dimensions from the shapes of the input references.
  kernel_size = kernel_ref.shape[0]
  _, block_d, block_h, block_w, _ = x_ref.shape
  # The stride is fixed based on the source computation.
  stride = 1

  # Load the kernel from its reference into a JAX array to make it usable
  # in JAX operations like `jnp.einsum`.
  kernel_val = kernel_ref[...]

  # Calculate the starting coordinates of the current input block in the full `x` tensor.
  d_offset = i * block_d
  h_offset = j * block_h
  w_offset = k * block_w

  # Iterate over each spatial location (d, h, w) within the input block.
  for bd in range(block_d):
    for bh in range(block_h):
      for bw in range(block_w):
        # Calculate the global coordinates of the current input element.
        d_in = d_offset + bd
        h_in = h_offset + bh
        w_in = w_offset + bw

        # Calculate the top-left corner of the output patch to which this
        # input element contributes.
        d_out_start = d_in * stride
        h_out_start = h_in * stride
        w_out_start = w_in * stride

        # Get the vector of input channels for the current spatial location.
        in_vals = x_ref[0, bd, bh, bw, :]

        # Compute the full contribution for this input location. This is done
        # by contracting the input channel dimension of the kernel with the
        # vector of input values using einsum.
        # kernel_val shape: (KD, KH, KW, C_in, C_out)
        # in_vals shape: (C_in,)
        # -> update shape: (KD, KH, KW, C_out)
        update = jnp.einsum("dhwio,i->dhwo", kernel_val, in_vals)

        # Define the slice in the output tensor to update.
        # pl.ds creates a dynamic-sized slice.
        d_slice = pl.ds(d_out_start, kernel_size)
        h_slice = pl.ds(h_out_start, kernel_size)
        w_slice = pl.ds(w_out_start, kernel_size)

        # Atomically add the computed contribution to the output slice.
        # This is crucial because different input blocks (handled by different
        # program instances) can contribute to overlapping regions in the output.
        pl.atomic_add(
          out_ref,
          (b, d_slice, h_slice, w_slice, slice(None)),
          update,
        )


# A transposed convolution can be implemented as a "scatter" operation where each
# input element's contribution is added to the output. We parallelize over
# blocks of the input tensor `x`.
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  # The grid is defined over the blocks of the input tensor `x`.
  grid=(batch_size, depth // block_d, height // block_h, width // block_w),
  in_specs=[
    # Input `x` is tiled. Each kernel instance receives one block.
    # The index_map `(b, i, j, k)` selects the (b,i,j,k)-th block of `x`.
    pl.BlockSpec(
      block_shape=(1, block_d, block_h, block_w, in_channels),
      index_map=lambda b, i, j, k: (b, i, j, k, 0),
    ),
    # The convolution kernel is a read-only input, broadcast to all instances.
    # The index_map `lambda *_: (0, ...)` gives the full kernel to each instance.
    pl.BlockSpec(
      block_shape=params["kernel"].shape,
      index_map=lambda *_: tuple([0] * params["kernel"].ndim),
    ),
  ],
  # The entire output buffer is passed to each kernel instance. The kernel
  # is responsible for atomically adding its contributions to the correct
  # locations in the output, as different input blocks contribute to
  # overlapping regions in the output.
  out_specs=pl.BlockSpec(
    block_shape=output_shape,
    index_map=lambda *_: tuple([0] * len(output_shape)),
  ),
)(x, params["kernel"]).block_until_ready()
