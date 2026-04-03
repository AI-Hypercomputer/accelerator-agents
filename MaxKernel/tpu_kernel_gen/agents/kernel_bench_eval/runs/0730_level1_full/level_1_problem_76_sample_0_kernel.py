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
length = 256
stride = 3
dilation = 4

key = random.PRNGKey(0)
x_key, init_key = random.split(key)

x = random.normal(x_key, (batch_size, in_channels, length))
conv1d = nn.Conv(
  features=out_channels,
  kernel_size=(kernel_size,),
  strides=(stride,),
  kernel_dilation=(dilation,),
  use_bias=False,
  padding="VALID",
)
variables = conv1d.init(init_key, jnp.transpose(x, (0, 2, 1)))

# Transpose kernel for TPU-compatible blocking and define block size
kernel_w_t = jnp.transpose(variables["params"]["kernel"], (2, 0, 1))
out_chan_block_size = 16

# Calculate output length for the convolution
effective_kernel_size = (kernel_size - 1) * dilation + 1
output_length = (x.shape[2] - effective_kernel_size) // stride + 1


# Computation
def kernel(x_ref, kernel_ref, y_ref):
  """
  Pallas kernel for 1D convolution.

  Args:
    x_ref: Input tensor reference of shape (1, in_channels, length).
    kernel_ref: Kernel weights reference of shape (out_chan_block_size, in_channels, kernel_size).
    y_ref: Output tensor reference of shape (1, out_chan_block_size, output_length).
  """
  # Extract dimensions from the input shapes for clarity.
  output_length = y_ref.shape[2]
  kernel_size = kernel_ref.shape[2]
  out_chan_block_size = y_ref.shape[1]
  in_channels = x_ref.shape[1]

  # Iterate over each position in the output's length dimension.
  for k in range(output_length):
    # Create an accumulator for the output at this position.
    acc = jnp.zeros((out_chan_block_size,), dtype=x_ref.dtype)

    # Iterate over the kernel's spatial dimension (the receptive field).
    for l in range(kernel_size):
      # Calculate the index in the original input tensor `x`, accounting for
      # stride and dilation.
      input_idx = k * stride + l * dilation

      # Load the relevant slice of the input `x` and the kernel.
      # kernel_ref[:, :, l] has shape (out_chan_block_size, in_channels).
      # x_ref[0, :, input_idx] has shape (in_channels,).
      # Reshape the input slice to a column vector for matrix multiplication.
      x_col = x_ref[0, :, input_idx].reshape(in_channels, 1)
      # Perform matrix-matrix multiplication. The result has shape (out_chan_block_size, 1).
      result = jnp.dot(kernel_ref[:, :, l], x_col)
      # Squeeze the result and add to the accumulator.
      acc += result.squeeze(axis=-1)

    # Store the accumulated result for position `k` into the output buffer.
    y_ref[0, :, k] = acc


y = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_channels, output_length), x.dtype),
  grid=(batch_size, out_channels // out_chan_block_size),
  in_specs=[
    # Input data `x`: take a full slice for each batch item.
    pl.BlockSpec(block_shape=(1, x.shape[1], x.shape[2]), index_map=lambda i, j: (i, 0, 0)),
    # Transposed kernel weights `kernel_w_t`: take a slice for each output channel chunk.
    pl.BlockSpec(
      block_shape=(out_chan_block_size, kernel_w_t.shape[1], kernel_w_t.shape[2]),
      index_map=lambda i, j: (j * out_chan_block_size, 0, 0),
    ),
  ],
  out_specs=pl.BlockSpec(
    # Output `y`: map each grid element to a unique output block.
    block_shape=(1, out_chan_block_size, output_length),
    index_map=lambda i, j: (i, j * out_chan_block_size, 0),
  ),
)(x, kernel_w_t).block_until_ready()
