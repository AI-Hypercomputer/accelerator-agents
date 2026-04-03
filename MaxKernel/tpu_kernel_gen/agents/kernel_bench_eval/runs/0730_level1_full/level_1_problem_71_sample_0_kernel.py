# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl
from jax.experimental.mosaic import tpu

# Initialization
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
height_in = 128
width_in = 256
stride = 1
padding = "VALID"
output_padding = 0  # Not a direct parameter in Flax ConvTranspose
groups = 1
bias = False

key = random.PRNGKey(0)
key_x, key_init = random.split(key)

x = random.normal(key_x, (batch_size, height_in, width_in, in_channels))

conv_transpose2d = nn.ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size),
  strides=(stride, stride),
  padding=padding,
  use_bias=bias,
)
variables = conv_transpose2d.init(key_init, x)

b_h_in = 16

height_out = (height_in - 1) * stride + kernel_size
width_out = (width_in - 1) * stride + kernel_size


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """
  Pallas kernel for 2D transposed convolution.

  This kernel computes a 2D transposed convolution on a block of the input
  tensor `x_ref` with the convolution kernel `kernel_ref` and writes the
  result to the output buffer `out_ref`.

  Args:
    x_ref: A reference to the input tensor block.
    kernel_ref: A reference to the convolution kernel weights.
    out_ref: A reference to the output tensor block to be written to.
  """
  # A transposed convolution is equivalent to a regular convolution
  # on a padded input with a flipped kernel. This avoids the unsupported
  # lax.conv_transpose primitive in Pallas for TPUs.
  # We use the TPU-specific convolution primitive from mosaic.
  padding_amount = kernel_ref.shape[0] - 1
  padding_config = ((padding_amount, padding_amount), (padding_amount, padding_amount))

  out_ref[...] = tpu.convolution(
    lhs=x_ref[...],
    rhs=kernel_ref[...],
    window_strides=(1, 1),
    padding=padding_config,
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
  )


# Flip the kernel weights spatially outside the Pallas kernel, as lax.rev is
# not supported.
kernel_flipped = jnp.flip(variables["params"]["kernel"], axis=[0, 1])

output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, height_out, width_out, out_channels), x.dtype),
  grid=(batch_size, height_in // b_h_in),
  in_specs=[
    pl.BlockSpec((1, b_h_in, width_in, in_channels), lambda i, j: (i, j * b_h_in, 0, 0)),
    pl.BlockSpec(
      (kernel_size, kernel_size, in_channels, out_channels),
      lambda i, j: (0, 0, 0, 0),
    ),
  ],
  out_specs=pl.BlockSpec(
    (1, b_h_in + kernel_size - 1, width_out, out_channels),
    lambda i, j: (i, j * b_h_in, 0, 0),
  ),
)(x, kernel_flipped).block_until_ready()
