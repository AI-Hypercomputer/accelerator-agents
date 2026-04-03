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

# JAX uses channels-last (NHWC) by default
x = random.normal(key_x, (batch_size, height, width, in_channels))

conv = nn.Conv(features=out_channels, kernel_size=(1, 1), strides=(1, 1), padding="VALID", use_bias=bias)
params = conv.init(key_params, x)["params"]

# Block sizes
bM = 128
bN = 64


# Computation
def kernel(x_ref, w_ref, out_ref):
  """Pallas kernel for 1x1 convolution.

  This kernel performs a matrix multiplication on blocks of the input tensors.
  The 1x1 convolution is implemented as a matrix multiplication between the
  reshaped input tensor (pixels, in_channels) and the reshaped kernel tensor
  (in_channels, out_channels).

  Args:
    x_ref: A reference to a block of the reshaped input tensor, with shape
      (bM, in_channels).
    w_ref: A reference to a block of the reshaped kernel tensor, with shape
      (in_channels, bN).
    out_ref: A reference to a block of the output tensor, with shape (bM, bN),
      which will be written to.
  """
  # Perform the matrix multiplication for the current block.
  # x_ref[...] has shape (bM, in_channels)
  # w_ref[...] has shape (in_channels, bN)
  # The result has shape (bM, bN), which matches out_ref.
  out_ref[...] = jnp.dot(x_ref[...], w_ref[...])


# Reshaped dimensions
num_pixels = batch_size * height * width
x_reshaped = x.reshape(num_pixels, in_channels)
w_reshaped = params["kernel"].reshape(in_channels, out_channels)

result_shape = jax.ShapeDtypeStruct((num_pixels, out_channels), x.dtype)

result = pl.pallas_call(
  kernel,
  out_shape=result_shape,
  grid=(num_pixels // bM, out_channels // bN),
  in_specs=[
    pl.BlockSpec(block_shape=(bM, in_channels), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(in_channels, bN), index_map=lambda i, j: (0, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i, j)),
)(x_reshaped, w_reshaped).block_until_ready()

# Reshape result back to the original 4D tensor format
result = result.reshape(batch_size, height, width, out_channels)
