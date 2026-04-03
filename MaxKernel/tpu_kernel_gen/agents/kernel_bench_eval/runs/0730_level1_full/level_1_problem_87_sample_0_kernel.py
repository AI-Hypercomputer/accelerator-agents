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
width = 256
height = 256
bias = False
key = random.PRNGKey(0)
key_x, key_params = random.split(key)
x = random.normal(key_x, (batch_size, height, width, in_channels))
conv2d = nn.Conv(
  features=out_channels,
  kernel_size=(1, 1),
  strides=(1, 1),
  padding="VALID",
  use_bias=bias,
)
params = conv2d.init(key_params, x)["params"]
block_h = 16
block_w = 128

# The kernel parameter is reshaped from (1, 1, 3, 64) to (3, 64).
kernel_param = params["kernel"].reshape(in_channels, out_channels)

# Pad the input channels to a multiple of 8 for TPU compatibility.
# This is a common practice to avoid performance cliffs or errors on TPUs,
# which are optimized for dimensions that are multiples of 8 or 128.
# The bounds check error suggests a mismatch between the logical tensor shape
# and the physical memory layout, which is often caused by unpadded dimensions.
in_channels_padded = 8
x_padded = jnp.pad(
  x,
  (
    (0, 0),
    (0, 0),
    (0, 0),
    (0, in_channels_padded - in_channels),
  ),
)
kernel_padded = jnp.pad(
  kernel_param,
  (
    (0, in_channels_padded - in_channels),
    (0, 0),
  ),
)


def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for a 1x1 convolution.

  This kernel computes a 1x1 convolution, which is mathematically equivalent
  to a batched matrix multiplication between the input channels and the kernel.

  Args:
    x_ref: A reference to a tile of the input tensor with shape
      (1, block_h, block_w, in_channels_padded).
    kernel_ref: A reference to the convolution kernel with shape
      (in_channels_padded, out_channels). Note that each program in the grid
      receives the full kernel.
    out_ref: A reference to the output tile with shape
      (1, block_h, block_w, out_channels), where the result is stored.
  """
  # A 1x1 convolution is equivalent to a dot product between the input channels
  # and the kernel, applied at each spatial location. jnp.einsum provides a
  # concise way to express this.
  # The einsum string '...i,io->...o' performs a standard batched
  # matrix multiplication.
  # - It contracts the shared dimension 'i' (in_channels_padded).
  # - It preserves the batch dimensions '...' from the first operand (x_ref),
  #   which are (1, block_h, block_w).
  # - It preserves the output dimension 'o' (out_channels) from the second
  #   operand (kernel_ref).
  out_ref[...] = jnp.einsum("...i,io->...o", x_ref[...], kernel_ref[...])


# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, height, width, out_channels), x.dtype),
  grid=(batch_size, height // block_h, width // block_w),
  in_specs=[
    pl.BlockSpec(
      # The block shape now uses the padded channel dimension.
      block_shape=(1, block_h, block_w, in_channels_padded),
      index_map=lambda b, i, j: (b, i * block_h, j * block_w, 0),
    ),
    pl.BlockSpec(
      # The block shape for the kernel also uses the padded dimension.
      block_shape=(in_channels_padded, out_channels),
      index_map=lambda b, i, j: (0, 0),
    ),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, block_h, block_w, out_channels),
    index_map=lambda b, i, j: (b, i * block_h, j * block_w, 0),
  ),
)(x_padded, kernel_padded).block_until_ready()
