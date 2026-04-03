# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax.linen import ConvTranspose
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (5,)
length = 256
stride = (1,)
padding = 0
dilation = (3,)
bias = False

key = random.PRNGKey(0)
key_input, key_params = random.split(key)

# Note: JAX/Flax uses channels-last (NLC) format by default
x = random.normal(key_input, (batch_size, length, in_channels))

conv1d_transpose = ConvTranspose(
  features=out_channels,
  kernel_size=kernel_size,
  strides=stride,
  padding=padding,
  kernel_dilation=dilation,
  use_bias=bias,
)

params = conv1d_transpose.init(key_params, x)["params"]


def kernel(x_ref, kernel_ref, out_ref):
  """
  Pallas kernel for 1D transposed convolution.

  This kernel implements the "gather" approach. It iterates through each
  output position and gathers contributions from the relevant input positions.

  Args:
    x_ref: Input tensor with shape (1, length, in_channels). The batch
      dimension is handled by the grid mapping.
    kernel_ref: Kernel tensor with shape (kernel_size, in_channels, out_channels).
    out_ref: Output tensor block to be written to, with shape
      (1, block_l, out_channels).
  """
  # Get problem dimensions from the shapes of the references.
  length = x_ref.shape[1]
  in_channels = x_ref.shape[2]
  out_channels = kernel_ref.shape[2]
  kernel_size = kernel_ref.shape[0]
  block_l = out_ref.shape[1]

  # Hardcoded convolution parameters from the source computation.
  stride = 1
  dilation = 3

  # Each program instance computes a block of the output along the length dim.
  # `j` is the index of the block this program is responsible for.
  j = pl.program_id(1)
  offset_l = j * block_l

  # Iterate over each position in the output block this program is responsible for.
  for out_l_local in range(block_l):
    # Calculate the global output position.
    out_l = offset_l + out_l_local

    # Accumulator for the current output position, held in registers.
    acc_vec = jnp.zeros((out_channels,), dtype=out_ref.dtype)

    # Iterate over the kernel positions to find and gather contributing inputs.
    for k in range(kernel_size):
      # Calculate the source input position based on the output position and kernel pos.
      in_l = out_l - k * dilation

      # Check if the calculated input position is valid (within bounds).
      is_valid_in_l = (in_l >= 0) & (in_l < length) & (in_l % stride == 0)
      in_l = in_l // stride

      # Define the function to perform the update.
      # It reads the input, computes the dot product with the kernel, and adds to the accumulator.
      def do_update(current_acc_vec):
        # Load the input vector and reshape to a 2D matrix for matmul.
        x_vec = x_ref[0, in_l, :]
        x_mat = jnp.reshape(x_vec, (1, in_channels))
        # Perform the matrix multiplication.
        update_vec = jnp.dot(x_mat, kernel_ref[k, :, :])
        # Add the result to the accumulator for this output position.
        return current_acc_vec + update_vec[0, :]

      # Define the function to do nothing if the input is out of bounds.
      def do_nothing(current_acc_vec):
        return current_acc_vec

      # Use lax.cond to conditionally apply the update. This is crucial as it
      # avoids out-of-bounds reads from x_ref when the condition is false.
      acc_vec = jax.lax.cond(is_valid_in_l, do_update, do_nothing, acc_vec)

    # Write the final computed vector for the current output position to the output buffer.
    out_ref[0, out_l_local, :] = acc_vec


# Computation
# Calculate output shape based on convolution parameters
dilated_kernel_size = (kernel_size[0] - 1) * dilation[0] + 1
output_length = (length - 1) * stride[0] + dilated_kernel_size
output_shape = (batch_size, output_length, out_channels)

# Define block size for the length dimension
block_l = 128

# Pallas call to replace the ConvTranspose operation
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=(batch_size, -(-output_length // block_l)),
  in_specs=[
    pl.BlockSpec(block_shape=(1, length, in_channels), index_map=lambda i, j: (i, 0, 0)),
    pl.BlockSpec(block_shape=params["kernel"].shape, index_map=lambda i, j: (0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, block_l, out_channels), index_map=lambda i, j: (i, j, 0)),
)(x, params["kernel"]).block_until_ready()
