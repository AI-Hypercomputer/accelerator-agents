# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
features = 64
sequence_length = 128
kernel_size = 4
stride = 2
padding = 2
dilation = 3
key = random.PRNGKey(0)
# JAX/Flax convention is channel-last: (batch_size, sequence_length, features)
x = random.normal(key, (batch_size, sequence_length, features))


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for 1D max pooling.

  This kernel implements max pooling with support for kernel size, stride,
  padding, and dilation. It iterates through each output position, calculates
  the corresponding window in the input, and finds the maximum value, handling
  padding by checking boundary conditions.

  Args:
    x_ref: Input tensor reference of shape (1, sequence_length, features).
    out_ref: Output tensor reference of shape (1, output_sequence_length, features)
      to be populated.
  """
  # Define pooling parameters based on the source computation.
  kernel_size = 4
  stride = 2
  padding = 2
  dilation = 3

  # Get shape information from the input/output references.
  sequence_length = x_ref.shape[1]
  output_sequence_length = out_ref.shape[1]
  features = x_ref.shape[2]

  # Iterate over each position in the output sequence.
  for j in range(output_sequence_length):
    # Initialize a temporary accumulator for the max value for the current window.
    # This will be held in registers.
    max_val = jnp.full((features,), -jnp.inf, dtype=x_ref.dtype)

    # Iterate over each element in the pooling window.
    for k in range(kernel_size):
      # Calculate the corresponding index in the original (unpadded) input sequence.
      input_idx = j * stride + k * dilation - padding

      # Check if the calculated index is within the valid bounds of the input.
      # This effectively handles the padding by ignoring out-of-bounds elements.
      if (input_idx >= 0) & (input_idx < sequence_length):
        val = pl.load(
          x_ref,
          (0, input_idx, slice(0, features)),
        )
        # Update the maximum value in the register.
        max_val = jnp.maximum(max_val, val)

    # Write the final maximum value for the window to the output reference in SRAM.
    pl.store(out_ref, (0, j, slice(0, features)), max_val)


# Calculate the output shape based on the pooling parameters
output_sequence_length = (sequence_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
output_shape_struct = jax.ShapeDtypeStruct((batch_size, output_sequence_length, features), x.dtype)

output = pl.pallas_call(
  kernel,
  out_shape=output_shape_struct,
  grid=(batch_size,),
  in_specs=[pl.BlockSpec(block_shape=(1, sequence_length, features), index_map=lambda i: (i, 0, 0))],
  out_specs=pl.BlockSpec(block_shape=(1, output_sequence_length, features), index_map=lambda i: (i, 0, 0)),
)(x).block_until_ready()
