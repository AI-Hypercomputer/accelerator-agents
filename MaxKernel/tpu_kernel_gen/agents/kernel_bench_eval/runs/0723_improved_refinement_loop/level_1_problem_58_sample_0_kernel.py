# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax.linen import ConvTranspose
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 32
out_channels = 16
kernel_size = (3, 5, 7)
depth_in = 16
height_in = 32
width_in = 64
stride = (1, 1, 1)
padding = "VALID"
output_padding = (0, 0, 0)  # Not a direct parameter in flax.linen.ConvTranspose
groups = 1
bias = False

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX uses channels-last convention
x_shape = (batch_size, depth_in, height_in, width_in, in_channels)
x = random.normal(key_x, x_shape)

conv_transpose3d = ConvTranspose(
  features=out_channels, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=bias
)
params = conv_transpose3d.init(key_params, x)["params"]


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 3D transposed convolution.

  This kernel computes a single block of the output tensor. It implements a
  standard 3D convolution by iterating over the kernel's spatial dimensions.
  In each iteration, it slices a block from the input (`x_ref`) and performs
  a matrix multiplication (`jnp.dot`) with the corresponding kernel weights,
  accumulating the result. This manual implementation is necessary as high-level
  primitives like `lax.conv_general_dilated` are not supported inside Pallas
  kernels on TPU.

  Args:
    x_ref: A reference to the input block, including a halo region.
    kernel_ref: A reference to the complete convolution kernel.
    out_ref: A reference to the output block to be computed.
  """
  # Assume b_batch is 1 for simplicity inside the kernel.
  acc = jnp.zeros((b_depth, b_height, b_width, out_channels), dtype=out_ref.dtype)

  # Iterate over the kernel's spatial dimensions for the convolution.
  for kd in range(kernel_size[0]):
    for kh in range(kernel_size[1]):
      for kw in range(kernel_size[2]):
        # Slice the input block. The slice is the size of an output block,
        # offset by the current position in the kernel.
        x_slice = x_ref[
          0,
          kd : kd + b_depth,
          kh : kh + b_height,
          kw : kw + b_width,
          :,
        ]
        # Get the kernel weights for the current spatial position.
        kernel_weights = kernel_ref[kd, kh, kw, :, :]
        # Perform matrix multiplication and accumulate.
        acc += jnp.dot(x_slice, kernel_weights)

  # Write the accumulated result to the output block.
  out_ref[0] = acc


# Calculate the output shape based on 'VALID' padding and stride=(1,1,1)
# output_size = (input_size - 1) * stride + kernel_size
output_shape = (
  batch_size,
  (depth_in - 1) * stride[0] + kernel_size[0],
  (height_in - 1) * stride[1] + kernel_size[1],
  (width_in - 1) * stride[2] + kernel_size[2],
  out_channels,
)


# Define block sizes for tiling the output tensor
b_batch = 1
b_depth = 1
b_height = 4
b_width = output_shape[3]
# The kernel for each output block needs a halo based on the filter size
# kernel_size = (3, 5, 7) -> halo = (2, 4, 6)
# The input block size is the output block size + halo
in_b_depth = b_depth + kernel_size[0] - 1
in_b_height = b_height + kernel_size[1] - 1
in_b_width = b_width + kernel_size[2] - 1


# The grid is determined by how many blocks fit into the output tensor
grid = (
  batch_size // b_batch,
  (output_shape[1] + b_depth - 1) // b_depth,
  (output_shape[2] + b_height - 1) // b_height,
  (output_shape[3] + b_width - 1) // b_width,
)

# The transposed convolution requires reading from a padded version of the
# input. The padding is needed at the beginning of the spatial dimensions.
pad_d = kernel_size[0] - 1
pad_h = kernel_size[1] - 1
pad_w = kernel_size[2] - 1
x_padded = jnp.pad(
  x,
  (
    (0, 0),
    (pad_d, pad_d),
    (pad_h, pad_h),
    (pad_w, pad_w),
    (0, 0),
  ),
)
# The kernel needs to be flipped along its spatial dimensions for the equivalence
# conv_transpose(x, k) == conv(x_padded, flip(k)) to hold.
kernel_flipped = jnp.flip(params["kernel"], axis=(0, 1, 2))


output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=grid,
  in_specs=[
    # Input 'x' is sliced. Each kernel instance gets a block of the input
    # corresponding to its position in the grid.
    pl.BlockSpec(
      block_shape=(
        b_batch,
        in_b_depth,
        in_b_height,
        in_b_width,
        in_channels,
      ),
      index_map=lambda b, d, h, w: (
        b * b_batch,
        d * b_depth,
        h * b_height,
        w * b_width,
        0,
      ),
    ),
    # The convolution kernel is needed in its entirety by every instance.
    pl.BlockSpec(
      block_shape=params["kernel"].shape,
      index_map=lambda b, d, h, w: (0, 0, 0, 0, 0),
    ),
  ],
  # The output is tiled. Each kernel instance writes to a unique block.
  out_specs=pl.BlockSpec(
    block_shape=(b_batch, b_depth, b_height, b_width, out_channels),
    index_map=lambda b, d, h, w: (
      b * b_batch,
      d * b_depth,
      h * b_height,
      w * b_width,
      0,
    ),
  ),
)(x_padded, kernel_flipped).block_until_ready()
