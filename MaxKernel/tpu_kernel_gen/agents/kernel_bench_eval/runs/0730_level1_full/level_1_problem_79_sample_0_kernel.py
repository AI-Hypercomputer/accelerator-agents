# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
length = 128
stride = 2
padding = 1
dilation = 2
key = random.PRNGKey(0)
key_x, key_init = random.split(key)

# JAX/Flax uses channels-last format
x = random.normal(key_x, (batch_size, length, in_channels))

# In Flax, model definition and initialization are separate steps.
conv1d_transpose = nn.ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size,),
  strides=(stride,),
  padding=padding,
  kernel_dilation=(dilation,),
  use_bias=False,
)
params = conv1d_transpose.init(key_init, x)["params"]

# Block size for the output channels dimension, chosen to be the full channel
# size to satisfy TPU memory layout constraints.
bOC = 64

# The output length of a transposed convolution is calculated based on the
# input length, stride, padding, and dilation. This formula matches the
# effective computation performed by JAX's `lax.conv_transpose`.
output_length = (length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """
  Pallas kernel for 1D transposed convolution.

  This kernel computes the transposed convolution for a single batch element and
  a block of output channels. The logic follows the "scatter-add" interpretation
  of a transposed convolution.

  Args:
    x_ref: A reference to the input tensor block for a single batch element.
      Shape: (1, length, in_channels)
    kernel_ref: A reference to the kernel (weights) block for a subset of
      output channels. Shape: (kernel_size, in_channels, bOC)
    out_ref: A reference to the output tensor block to be written to.
      Shape: (1, output_length, bOC)
  """
  # Initialize the output block with zeros before accumulation.
  out_ref[...] = jnp.zeros_like(out_ref)

  # Iterate over each position in the input's spatial dimension.
  for l in range(length):
    # Iterate over each position in the kernel's spatial dimension.
    for k in range(kernel_size):
      # Calculate the corresponding output spatial position.
      # This is the core formula for transposed convolution.
      out_l = l * stride + k * dilation - padding

      # Ensure the calculated output position is within the valid bounds
      # of the output buffer.
      if 0 <= out_l < output_length:
        # Extract the input vector and reshape to a 2D row-vector to ensure
        # it's treated as a matrix by `jnp.dot` for TPU compatibility.
        in_vec = x_ref[0, l, :][None, :]  # Shape: (1, in_channels)

        # Extract the kernel matrix for the current kernel position `k`.
        kernel_mat = kernel_ref[k, :, :]  # Shape: (in_channels, bOC)

        # Compute the dot product (matrix multiplication). The result will have
        # shape (1, bOC).
        update = jnp.dot(in_vec, kernel_mat)

        # Accumulate the result into the correct output position.
        # We squeeze the (1, bOC) update to (bOC,) to match the shape of the
        # output slice.
        out_ref[0, out_l, :] += jnp.squeeze(update, axis=0)


output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, output_length, out_channels), x.dtype),
  grid=(batch_size, out_channels // bOC),
  in_specs=[
    pl.BlockSpec(block_shape=(1, length, in_channels), index_map=lambda i, j: (i, 0, 0)),
    pl.BlockSpec(block_shape=(kernel_size, in_channels, bOC), index_map=lambda i, j: (0, 0, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, output_length, bOC), index_map=lambda i, j: (i, 0, j)),
)(x, params["kernel"]).block_until_ready()
