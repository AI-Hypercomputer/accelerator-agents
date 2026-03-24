# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256
depth = 10
stride = 1
padding = "VALID"
dilation = 1
groups = 1
bias = False

key = random.PRNGKey(0)
key_x, key_init = random.split(key)

# JAX uses channels-last convention (N, H, W, D, C)
x = random.normal(key_x, (batch_size, height, width, depth, in_channels))

conv3d = nn.Conv(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size, 1),
  strides=stride,
  padding=padding,
  kernel_dilation=dilation,
  feature_group_count=groups,
  use_bias=bias,
)
params = conv3d.init(key_init, x)["params"]


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 3D convolution.

  Args:
    x_ref: A reference to a block of the input tensor. It includes a "halo"
      of extra data around the spatial dimensions (height and width) to
      accomodate the convolution kernel.
      Shape: (1, 34, 34, 10, 3)
    kernel_ref: A reference to the entire convolution kernel weights.
      Shape: (3, 3, 1, 3, 64)
    out_ref: A reference to a block of the output tensor where the result of
      the convolution is stored.
      Shape: (1, 32, 32, 10, 64)
  """
  # Initialize an accumulator in registers. This is more efficient than
  # repeatedly reading from and writing to `out_ref` in a loop.
  acc = jnp.zeros((1, 32, 32, 10, 64), dtype=x_ref.dtype)

  # The convolution operation is implemented by iterating through the spatial
  # dimensions of the kernel (3x3 in this case). For each position in the
  # kernel, we perform a dot product between a patch of the input and the
  # corresponding kernel weights, accumulating the results in the accumulator.

  # Iterate over the kernel's height dimension.
  for kh in range(3):
    # Iterate over the kernel's width dimension.
    for kw in range(3):
      # Extract the relevant patch from the input tensor. The slice size
      # (32x32) matches the output block's spatial dimensions. The offsets
      # (kh, kw) slide this window across the input block's halo region.
      # Shape of input_patch: (1, 32, 32, 10, 3)
      input_patch = x_ref[:, kh : kh + 32, kw : kw + 32, :, :]

      # Extract the corresponding weights from the kernel for the current
      # spatial position.
      # Shape of kernel_patch: (3, 64)
      kernel_patch = kernel_ref[kh, kw, 0, :, :]

      # Perform the core computation: a dot product between the input patch's
      # channels and the kernel patch.
      # (1, 32, 32, 10, 3) @ (3, 64) -> (1, 32, 32, 10, 64)
      acc += jnp.dot(input_patch, kernel_patch)

  # Write the final accumulated result to the output block in one go.
  out_ref[...] = acc


# The following pallas_call replaces the original 3D convolution computation.
# It parallelizes the operation by dividing the output feature map into tiles.
# The grid is defined over the batch size and the tiled spatial dimensions of the output.
# Each kernel instance is responsible for computing one (32x32) tile of the output
# across all output channels and depth for a given batch item.

# Grid dimensions:
# - 16: batch_size
# - 8: ceil(output_height / block_h) = ceil(254 / 32)
# - 8: ceil(output_width / block_w) = ceil(254 / 32)
grid = (16, 8, 8)

# Block shapes for data chunking:
# - out_specs: A (1, 32, 32, 10, 64) block of the output tensor.
# - in_specs[0] (for x): A (1, 34, 34, 10, 3) block of the input. The spatial
#   dimensions are larger (34x34 vs 32x32) to account for the 3x3 kernel halo.
# - in_specs[1] (for weights): The entire (3, 3, 1, 3, 64) weight tensor, as it's
#   needed by each kernel instance to compute all output channels.

output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((16, 254, 254, 10, 64), x.dtype),
  grid=grid,
  in_specs=[
    pl.BlockSpec(
      (1, 34, 34, 10, 3),
      lambda b, i, j: (b, i * 32, j * 32, 0, 0),
    ),
    pl.BlockSpec(
      (3, 3, 1, 3, 64),
      lambda b, i, j: (0, 0, 0, 0, 0),
    ),
  ],
  out_specs=pl.BlockSpec(
    (1, 32, 32, 10, 64),
    lambda b, i, j: (b, i * 32, j * 32, 0, 0),
  ),
)(x, params["kernel"]).block_until_ready()
