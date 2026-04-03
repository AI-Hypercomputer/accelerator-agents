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
# JAX/Flax convention for 1D pooling is (batch, length, features)
x = random.normal(key, (batch_size, sequence_length, features))


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for 1D max pooling.

  This kernel implements the equivalent of jax.lax.reduce_window with max
  pooling parameters. Each program in the grid processes one item from the batch.
  """
  # Parameters from the original reduce_window call
  kernel_size = 4
  stride = 2
  padding = 2
  dilation = 3

  # Dimensions from the input/output shapes
  # x_ref shape: (1, sequence_length, features)
  # out_ref shape: (1, output_length, features)
  sequence_length = x_ref.shape[1]
  output_length = out_ref.shape[1]
  features = x_ref.shape[2]

  # Iterate over each position in the output sequence.
  for j in range(output_length):
    # Initialize a register to hold the maximum value for the current window.
    # The initial value is -infinity, which is the identity for the max operation.
    max_val = jnp.full((features,), -jnp.inf, dtype=x_ref.dtype)

    # Iterate over the elements of the pooling window.
    for k in range(kernel_size):
      # Calculate the index in the input sequence based on the output position (j),
      # stride, padding, and dilation.
      input_idx = j * stride - padding + k * dilation

      # The padding is handled by checking if the calculated index is within
      # the valid bounds of the input sequence.
      if 0 <= input_idx < sequence_length:
        # Load the value from the input reference (SRAM) into a register.
        current_val = x_ref[0, input_idx, :]
        # Update the running maximum.
        max_val = jnp.maximum(max_val, current_val)

    # After scanning the entire window, write the final maximum value
    # to the output reference (SRAM).
    out_ref[0, j, :] = max_val


output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, 62, features), x.dtype),
  grid=(batch_size,),
  in_specs=[pl.BlockSpec(block_shape=(1, sequence_length, features), index_map=lambda i: (i, 0, 0))],
  out_specs=pl.BlockSpec(block_shape=(1, 62, features), index_map=lambda i: (i, 0, 0)),
)(x).block_until_ready()
