# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 64
out_channels = 3
kernel_size = 3
length = 128
stride = 1
padding = "VALID"
output_padding = 0
groups = 1
bias = False

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

x = random.normal(key_x, (batch_size, length, in_channels))
# Flax's ConvTranspose does not have an output_padding parameter.
# For stride=1, output_padding=0 in PyTorch has no effect.
# PyTorch's padding=0 is equivalent to Flax's padding='VALID'.
# Flax's ConvTranspose does not take feature_group_count in the constructor for groups=1.
conv1d_transpose = nn.ConvTranspose(
  features=out_channels, kernel_size=(kernel_size,), strides=(stride,), padding=padding, use_bias=bias
)
params = conv1d_transpose.init(key_params, x)["params"]

output_length = (length - 1) * stride + kernel_size
output_shape = (batch_size, output_length, out_channels)
w = params["kernel"]


# Computation
def kernel(x_ref, w_ref, out_ref):
  """Pallas kernel for 1D transposed convolution with stride=1.

  Args:
    x_ref: Input tensor block of shape (1, length, in_channels).
    w_ref: Weight tensor block of shape (kernel_size, in_channels, out_channels).
    out_ref: Output tensor block of shape (1, output_length, out_channels).
  """
  # Initialize the output block to zeros.
  out_ref[...] = jnp.zeros_like(out_ref)

  # Reshape weights once for matmul.
  # w_ref shape: (kernel_size, in_channels, out_channels)
  # w_reshaped shape: (in_channels, kernel_size * out_channels)
  w_reshaped = w_ref[...].transpose(1, 0, 2).reshape(in_channels, -1)

  # The core logic of transposed convolution with stride=1 is to iterate
  # through each position in the input, and for each position, add the
  # scaled kernel to the corresponding output region.
  for l in range(length):
    # Get the input slice for the current position `l` and reshape for matmul.
    # x_slice has shape (1, in_channels).
    x_slice = x_ref[0, l, :].reshape(1, in_channels)

    # Perform the contraction as a matrix-matrix product.
    # x_slice: (1, in_channels)
    # w_reshaped: (in_channels, kernel_size * out_channels)
    # update_flat: (1, kernel_size * out_channels)
    update_flat = jnp.dot(x_slice, w_reshaped)

    # Reshape the result back to (kernel_size, out_channels).
    update = update_flat.reshape(kernel_size, out_channels)

    # Add the computed update to the output at the correct offset.
    # Since stride=1, the output window starts at index `l`.
    out_ref[0, l : l + kernel_size, :] += update


output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(1, length, in_channels), index_map=lambda i: (i, 0, 0)),
    pl.BlockSpec(block_shape=w.shape, index_map=lambda i: (0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, output_length, out_channels), index_map=lambda i: (i, 0, 0)),
)(x, w).block_until_ready()
