# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Initialization
batch_size = 16
in_channels = 32
out_channels = 16
kernel_size = 3
depth = 16
height = 32
width = 64

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX uses channels-last format (NDHWC)
x = random.normal(key_x, (batch_size, depth, height, width, in_channels))

# In Flax, ConvTranspose infers dimensions from the input.
# padding=0 in PyTorch is 'VALID' in Flax.
conv_transpose3d = nn.ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size, kernel_size),
  strides=(1, 1, 1),
  padding="VALID",
  use_bias=False,
)
params = conv_transpose3d.init(key_params, x)["params"]

# Pre-flip the kernel outside of the Pallas kernel function
# The kernel layout is assumed to be 'DHWIO'. We flip the spatial dimensions.
flipped_kernel = jnp.flip(params["kernel"], axis=(0, 1, 2))


# Computation
def kernel(x_ref, w_ref, out_ref):
  """Pallas kernel for 3D transposed convolution.

  This kernel implements the equivalent of a 3D transposed convolution by
  decomposing it into a series of 2D convolutions.
  """
  # Accumulator for the output slice. It has a 2D spatial shape.
  acc = jnp.zeros((1, out_ref.shape[2], out_ref.shape[3], out_ref.shape[4]), dtype=x_ref.dtype)

  # Loop over the depth dimension of the kernel
  for kd in range(w_ref.shape[0]):
    # Extract the kd-th 2D slice from the input block.
    x_slice = lax.dynamic_slice_in_dim(x_ref[...], kd, 1, axis=1)
    x_slice = jnp.squeeze(x_slice, axis=1)

    # Extract the kd-th 2D slice from the kernel block.
    w_slice = lax.dynamic_slice_in_dim(w_ref[...], kd, 1, axis=0)
    w_slice = jnp.squeeze(w_slice, axis=0)

    # Padding for 'FULL' convolution on spatial dimensions.
    h_padding = w_slice.shape[0] - 1
    w_padding = w_slice.shape[1] - 1

    # Perform 2D convolution on the slices.
    conv_out = pltpu.conv(
      x_slice,
      w_slice,
      window_strides=(1, 1),
      padding=((h_padding, h_padding), (w_padding, w_padding)),
      dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    acc += conv_out

  # Write the accumulated result, expanding the depth dimension back.
  out_ref[...] = jnp.expand_dims(acc, axis=1)


# Calculate output shapes based on variables from the initialization section
out_depth = (depth - 1) * 1 + kernel_size
out_height = (height - 1) * 1 + kernel_size
out_width = (width - 1) * 1 + kernel_size

output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_depth, out_height, out_width, out_channels), x.dtype),
  grid=(batch_size, out_depth),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, kernel_size, height, width, in_channels),
      index_map=lambda n, d: (n, d - (kernel_size - 1), 0, 0, 0),
    ),
    pl.BlockSpec(
      block_shape=(kernel_size, kernel_size, kernel_size, in_channels, out_channels),
      index_map=lambda n, d: (0, 0, 0, 0, 0),
    ),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, 1, out_height, out_width, out_channels), index_map=lambda n, d: (n, d, 0, 0, 0)
  ),
)(x, flipped_kernel).block_until_ready()
